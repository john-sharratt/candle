//! H5, the handshake half (§14.2, chunk 2): 500 handshakes — valid, wrong key,
//! another protocol version, and a member that speaks out of turn — end in
//! exact codes, and the gateway sends nothing but `CHALLENGE` before `ADMIT`.
//! The build-floor, model-lock and duplicate-node half runs against the
//! gateway's admission. Budget 2 s.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cluster::close::Close;
use cluster::frame::{Admit, BuildId, Done, Frame, Hello, Kind};
use cluster::handshake::{admit, join, GroupKey, HandshakeError, PROTO};
use cluster::pipe::memory::{pair, MemoryTx};
use cluster::pipe::{Message, PipeClosed, PipeRx, PipeTx};

const HANDSHAKES: usize = 500;
const BUDGET: Duration = Duration::from_secs(2);

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

    fn unsent_bytes(&self) -> usize {
        self.inner.unsent_bytes()
    }
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Valid,
    WrongKey,
    OtherProtocol,
    OutOfTurn,
}

fn hello(node: u64, proto: u16) -> Hello {
    Hello {
        proto,
        app: "zend".into(),
        node,
        host: format!("m{node}"),
        build: BuildId {
            count: 100,
            hash: "abc1234".into(),
            local: String::new(),
        },
        model: None,
        power: 100,
        repos: vec!["candle".into()],
        now_ms: 0,
        nonce: [0; 32],
    }
}

fn admission(node: u64) -> Result<Admit, Close> {
    Ok(Admit {
        session: node,
        members: vec![],
        claims: vec![],
        floor_count: 100,
        floor_hash: "abc1234".into(),
        routes_wanted: true,
        own_applied_through: 0,
    })
}

/// One handshake: the code each side ends with (0 for an admission) and the
/// kinds the gateway sent.
async fn one(case: Case, node: u64) -> (u16, u16, Vec<u8>) {
    let gateway_key = GroupKey::from_bytes([1; 32]);
    let member_key = match case {
        Case::WrongKey => GroupKey::from_bytes([2; 32]),
        _ => gateway_key.clone(),
    };
    let (m, g) = pair(1 << 16);
    let kinds = Arc::new(Mutex::new(Vec::new()));
    let mut gtx = Tap {
        inner: g.tx,
        kinds: kinds.clone(),
    };
    let mut grx = g.rx;
    let (mut mtx, mut mrx) = (m.tx, m.rx);
    let member = async {
        match case {
            Case::OutOfTurn => {
                let done = Frame::Done(Done {
                    request: 1,
                    reply_to: node,
                });
                mtx.send(Message::Frame(done.encode())).await.unwrap();
                match mrx.recv().await {
                    Some(Message::Close(c)) => c.code,
                    other => panic!("{other:?}"),
                }
            }
            _ => {
                let proto = match case {
                    Case::OtherProtocol => PROTO + 1,
                    _ => PROTO,
                };
                match join(&mut mtx, &mut mrx, &member_key, hello(node, proto)).await {
                    Ok(_) => 0,
                    Err(HandshakeError::Closed(c)) | Err(HandshakeError::Refused(c)) => c.code,
                    Err(e) => panic!("{e}"),
                }
            }
        }
    };
    let gateway = async {
        match admit(&mut gtx, &mut grx, &gateway_key, |h| admission(h.node)).await {
            Ok(_) => 0,
            Err(HandshakeError::Closed(c)) | Err(HandshakeError::Refused(c)) => c.code,
            Err(e) => panic!("{e}"),
        }
    };
    let (mc, gc) = tokio::join!(member, gateway);
    let sent = kinds.lock().unwrap().clone();
    (mc, gc, sent)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn h5_handshake_storm() {
    let started = Instant::now();
    let cases = [
        Case::Valid,
        Case::WrongKey,
        Case::OtherProtocol,
        Case::OutOfTurn,
    ];
    let mut runs = Vec::new();
    for i in 0..HANDSHAKES {
        let case = cases[i % cases.len()];
        runs.push(tokio::spawn(async move {
            (case, one(case, i as u64 + 1).await)
        }));
    }
    let mut admitted = 0;
    for r in runs {
        let (case, (member, gateway, sent)) = r.await.unwrap();
        let (want_member, want_gateway) = match case {
            Case::Valid => (0, 0),
            Case::WrongKey => (4401, 4401),
            Case::OtherProtocol => (4426, 4426),
            Case::OutOfTurn => (4400, 4400),
        };
        assert_eq!((member, gateway), (want_member, want_gateway), "{case:?}");
        // Nothing but the handshake crosses before ADMIT.
        let allowed = [Kind::Challenge.tag(), Kind::Admit.tag()];
        assert!(
            sent.iter().all(|k| allowed.contains(k)),
            "{case:?}: {sent:?}"
        );
        if member == 0 {
            admitted += 1;
            assert_eq!(sent, allowed);
        }
    }
    let elapsed = started.elapsed();
    println!("H5 (handshake half): {HANDSHAKES} handshakes, {admitted} admitted, in {elapsed:?}");
    assert_eq!(admitted, HANDSHAKES / 4);
    assert!(elapsed <= BUDGET, "took {elapsed:?}");
}
