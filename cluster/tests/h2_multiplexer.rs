//! H2 (§14.2): 1,000 concurrent virtual streams × 64 KiB over one in-memory
//! socket pair, with a 63 MB event interleaved. Every stream arrives intact;
//! no stream's frame waits behind more than one 1 MiB fragment; a control frame
//! crosses within its bound while the socket is saturated. Budget 3 s.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cluster::frame::{Frame, Kind, Receipt};
use cluster::pipe::memory::{pair, MemoryTx};
use cluster::pipe::{Drain, Message, PipeClosed, PipeTx};
use cluster::session::{start, Incoming, Role};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const STREAMS: usize = 1000;
const PER_STREAM: usize = 64 << 10;
const EVENT_BYTES: usize = 63_000_000;
/// The socket's buffer, as a WebSocket's send buffer would be.
const PIPE_CAP: usize = 2 << 20;
const BUDGET: Duration = Duration::from_secs(3);
/// A control frame's crossing time while the socket is saturated: behind at
/// most the socket's buffer and one fragment. Ten times the first measurement
/// on the RTX PRO 5000 workstation (4.4 ms worst of 20, 2026-10-11; the whole
/// run took 340 ms).
const CONTROL_BOUND: Duration = Duration::from_millis(50);

/// A sender that records each frame's kind as it goes on the wire.
struct Tap {
    inner: MemoryTx,
    kinds: Arc<Mutex<Vec<u8>>>,
}

impl PipeTx for Tap {
    async fn send(&mut self, message: Message) -> Result<(), PipeClosed> {
        if let Message::Frame(bytes) = &message {
            self.kinds.lock().unwrap().push(bytes[0]);
        }
        self.inner.send(message).await
    }

    fn drain(&self) -> &Drain {
        self.inner.drain()
    }
}

fn payload(stream: usize) -> Vec<u8> {
    (0..PER_STREAM).map(|i| (i ^ stream) as u8).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn h2_multiplexer() {
    let started = Instant::now();
    let (g, m) = pair(PIPE_CAP);
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let gateway = start(
        Role::Gateway,
        Tap {
            inner: g.tx,
            kinds: kinds.clone(),
        },
        g.rx,
    );
    let mut member = start(Role::Member, m.tx, m.rx);

    // The member reads every stream and every event and control frame.
    let control_sent: Arc<Mutex<HashMap<u64, Instant>>> = Arc::default();
    let latencies = Arc::new(Mutex::new(Vec::new()));
    let reader = {
        let control_sent = control_sent.clone();
        let latencies = latencies.clone();
        tokio::spawn(async move {
            let mut readers = Vec::new();
            let mut event = None;
            let mut receipts = 0;
            while readers.len() < STREAMS || event.is_none() || receipts < 20 {
                match member.incoming.recv().await.expect("session open") {
                    Incoming::Stream(mut conn) => readers.push(tokio::spawn(async move {
                        let mut got = Vec::with_capacity(PER_STREAM + 16);
                        conn.read_to_end(&mut got).await.unwrap();
                        (conn.target().parse::<usize>().unwrap(), got)
                    })),
                    Incoming::Event { whole, .. } => event = Some(whole),
                    Incoming::Frame(Frame::Receipt(Receipt { through })) => {
                        let sent = control_sent.lock().unwrap()[&through];
                        latencies.lock().unwrap().push(sent.elapsed());
                        receipts += 1;
                    }
                    Incoming::Frame(_) => {}
                }
            }
            let mut intact = 0;
            for r in readers {
                let (stream, got) = r.await.unwrap();
                assert_eq!(got, payload(stream), "stream {stream}");
                intact += 1;
            }
            (intact, event.unwrap())
        })
    };

    // The gateway queues the event first, then opens and writes every stream.
    let record: Vec<u8> = (0..EVENT_BYTES).map(|i| (i % 253) as u8).collect();
    let event_sent = gateway.handle.event(1, 1, &record, None, false).unwrap();
    let mut writers = Vec::new();
    for s in 0..STREAMS {
        let mut conn = gateway.handle.open(&s.to_string()).unwrap();
        writers.push(tokio::spawn(async move {
            conn.write_all(&payload(s)).await.unwrap();
            conn.shutdown().await.unwrap();
        }));
    }
    // Control frames, stamped, while the socket is busy.
    let probe = {
        let handle = gateway.handle.clone();
        let control_sent = control_sent.clone();
        tokio::spawn(async move {
            for n in 1..=20u64 {
                control_sent.lock().unwrap().insert(n, Instant::now());
                handle
                    .control(Frame::Receipt(Receipt { through: n }))
                    .unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
    };
    for w in writers {
        w.await.unwrap();
    }
    probe.await.unwrap();
    let (intact, event) = reader.await.unwrap();
    event_sent.await.unwrap();
    let elapsed = started.elapsed();

    assert_eq!(intact, STREAMS);
    assert_eq!(event.bytes, record);

    // On the wire, while stream data was flowing, no two event fragments
    // went back to back.
    let kinds = kinds.lock().unwrap();
    let data = Kind::Data.tag();
    let ev = Kind::Event.tag();
    let first = kinds.iter().position(|k| *k == data).unwrap();
    let last = kinds.iter().rposition(|k| *k == data).unwrap();
    let back_to_back = kinds[first..=last]
        .windows(2)
        .filter(|w| w[0] == ev && w[1] == ev)
        .count();
    assert_eq!(
        back_to_back, 0,
        "a stream frame waited behind two fragments"
    );
    let fragments = kinds.iter().filter(|k| **k == ev).count();
    assert_eq!(fragments, 61);

    let latencies = latencies.lock().unwrap();
    let worst = latencies.iter().max().copied().unwrap_or_default();
    println!(
        "H2: {STREAMS} streams × {} KiB and a {} MB event in {:?}; worst control frame {:?}",
        PER_STREAM >> 10,
        EVENT_BYTES / 1_000_000,
        elapsed,
        worst
    );
    assert_eq!(latencies.len(), 20);
    assert!(worst <= CONTROL_BOUND, "control frame took {worst:?}");
    assert!(elapsed <= BUDGET, "took {elapsed:?}");
}
