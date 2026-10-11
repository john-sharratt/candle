//! H4 — recovery under load (§14.2). Live traffic runs throughout while:
//!
//! 1. a member's socket drops mid-burst with thousands of events in flight,
//!    and reconnects;
//! 2. a member is away while 100 k records across three origins go by, some
//!    since compacted away, then comes back — and is killed mid-catch-up and
//!    restarted;
//! 3. a restarted member offers 10 k records nobody else has;
//! 4. one member reads its socket slowly.
//!
//! Every member ends holding the same applied set and the same records; the
//! returning member never fetches a compacted record, fetches each one it
//! lacked once, and after its restart resumes from what it applied, with at
//! most one fetch window coming again; live events stay prompt while catch-up
//! runs; no catch-up frame is ever dropped at the gateway's cap.

mod support;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cluster::consensus::votes::MemoryVotes;
use cluster::replicate::{MemoryStore, Store, FETCH_WINDOW};

use support::{converged, gateway, member, member_over, until, Recording, APP};

const WITHIN: Duration = Duration::from_secs(60);

/// Records written while the returning member is away, across three origins.
const AWAY: usize = 100_000;

/// The bound on a live event's p99 latency while catch-up runs: about twice
/// the first measurements on the RTX PRO 5000 Blackwell box (190–295 ms) —
/// the round-robin grace, a relay hop behind at most 1 MiB of catch-up, and a
/// batch's apply.
const LIVE_P99: Duration = Duration::from_millis(600);

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn h4_recovery_under_load() {
    let started = Instant::now();
    let gw = gateway("gw");
    let a = member(
        &[&gw],
        1,
        Arc::new(MemoryStore::new(1)),
        MemoryVotes::new(),
        true,
    );
    let b = member(
        &[&gw],
        2,
        Arc::new(MemoryStore::new(2)),
        MemoryVotes::new(),
        true,
    );
    let c_store = Recording::new(3);
    let c = member_over(
        &[&gw],
        3,
        c_store.clone(),
        c_store.inner.clone(),
        MemoryVotes::new(),
    );
    until("ready", WITHIN, || a.ready() && b.ready() && c.ready()).await;

    // Live traffic throughout: A writes an event every 2 ms, keeping when.
    let stop = Arc::new(AtomicBool::new(false));
    let written: Arc<Mutex<HashMap<u64, Instant>>> = Arc::default();
    let live = {
        let (store, stop, written) = (a.store.clone(), stop.clone(), written.clone());
        tokio::spawn(async move {
            let mut k = 0u64;
            while !stop.load(Ordering::SeqCst) {
                let oseq = store.write(format!("live/{k}").as_bytes(), b"l");
                written.lock().unwrap().insert(oseq, Instant::now());
                k += 1;
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
    };

    // 1. A burst from B, its socket dropped part-way.
    for k in 0..5_000u32 {
        b.store.write(format!("burst/{k}").as_bytes(), b"b");
    }
    b.app.close_all("dropped mid-burst");
    until("b back", WITHIN, || b.ready()).await;

    // 2. D away while 100 k records go by; some are compacted away on every
    // member, as compaction drops what every member drops.
    let per = AWAY / 3;
    for k in 0..per {
        for n in [&a, &b, &c] {
            n.store
                .write(format!("away/{}/{k}", n.node).as_bytes(), b"w");
        }
    }
    let tops: Vec<u64> = [&a, &b, &c].iter().map(|n| n.store.own_through()).collect();
    until("a, b and c hold the away records", WITHIN, || {
        [&a, &b, &c].iter().all(|n| {
            let applied = n.store.applied();
            (0..3).all(|j| applied.through(j as u64 + 1) >= tops[j])
        })
    })
    .await;
    let compacted: Vec<(u64, u64)> = (0..3u64)
        .flat_map(|o| (0..50u64).map(move |i| (o + 1, 6_000 + i * 37)))
        .collect();
    for n in [&a, &b, &c] {
        for (o, q) in &compacted {
            n.store.compact_away(*o, *q);
        }
    }

    let catching_up = Instant::now();
    let d_store = Recording::new(4);
    let d = member_over(
        &[&gw],
        4,
        d_store.clone(),
        d_store.inner.clone(),
        MemoryVotes::new(),
    );
    // Killed part-way through its catch-up and restarted over the same
    // substrate: it resumes from what it applied.
    until("d part-way", WITHIN, || d_store.inner.len() > 10_000).await;
    let (_, d_votes) = d.stop();
    // A batch already handed to the store when it stopped finishes applying.
    tokio::time::sleep(Duration::from_millis(200)).await;
    let first_run = d_store.arrivals.lock().unwrap().len();
    let d = member_over(&[&gw], 4, d_store.clone(), d_store.inner.clone(), d_votes);

    // 3. E restarted, holding 10 k records of an origin nobody else has.
    let ghost = MemoryStore::new(9);
    for k in 0..10_000u32 {
        ghost.write(format!("ghost/{k}").as_bytes(), b"g");
    }
    let e_store = Arc::new(MemoryStore::new(5));
    let ghosts: Vec<_> = (1..=10_000)
        .map(|q| (9, q, ghost.read(9, q).unwrap()))
        .collect();
    e_store.apply(&ghosts);
    let e = member(&[&gw], 5, e_store, MemoryVotes::new(), true);

    // 4. F reads slowly for a while.
    let f = member(
        &[&gw],
        6,
        Arc::new(MemoryStore::new(6)),
        MemoryVotes::new(),
        true,
    );
    until("f ready", WITHIN, || f.ready()).await;
    f.pause(0, true);
    tokio::time::sleep(Duration::from_millis(300)).await;
    f.pause(0, false);

    until("d caught up", WITHIN, || d.ready()).await;
    let caught_up = Instant::now();
    stop.store(true, Ordering::SeqCst);
    live.await.unwrap();

    let all = [&a, &b, &c, &d, &e, &f];
    let stores: Vec<&MemoryStore> = all.iter().map(|n| &*n.store).collect();
    until("everyone converged", WITHIN, || {
        converged(&stores) && stores.iter().all(|s| s.applied() == stores[0].applied())
    })
    .await;
    assert_eq!(d.store.applied().through(9), 10_000, "E's offer was taken");

    // D never fetched a compacted record. Its first run fetched each record it
    // lacked once. A's live events are left out: one of them can arrive live
    // and be listed by a catch-up too, and is skipped as a duplicate (§6.5).
    let arrivals = d_store.arrivals.lock().unwrap().clone();
    let live_oseqs: HashSet<u64> = written.lock().unwrap().keys().copied().collect();
    let away = |(o, q): &(u64, u64)| {
        *o <= 3 && *q <= tops[*o as usize - 1] && !(*o == 1 && live_oseqs.contains(q))
    };
    let (first, second) = arrivals.split_at(first_run);
    let mut once = HashSet::new();
    for k in first.iter().filter(|k| away(k)) {
        assert!(!compacted.contains(k), "fetched compacted {k:?}");
        assert!(once.insert(*k), "the first run fetched {k:?} twice");
    }
    // The restart resumes from what was applied. What comes twice is only
    // what was in flight when it was killed: the first run's outstanding
    // fetch, still answered, reaches the node's new session — at most one
    // fetch window of bytes.
    let mut again = HashMap::new();
    for k in second.iter().filter(|k| away(k)) {
        assert!(!compacted.contains(k), "fetched compacted {k:?}");
        *again.entry(*k).or_insert(0u32) += 1;
    }
    let refetched: u64 = again
        .iter()
        .filter(|(k, n)| once.contains(k) || **n > 1)
        .map(|((o, q), _)| d.store.read(*o, *q).map_or(0, |b| b.len() as u64))
        .sum();
    assert!(
        refetched <= FETCH_WINDOW,
        "{refetched} bytes came again after the restart, over one fetch window"
    );

    // No catch-up frame was dropped at the cap.
    for m in gw.status(0).members.iter().filter(|m| m.app == APP) {
        assert_eq!(m.queues.dropped, 0, "{m:?}");
    }

    // Live events reached C promptly while D caught up.
    let written = written.lock().unwrap();
    let mut latencies: Vec<Duration> = written
        .iter()
        .filter(|(_, at)| **at >= catching_up && **at <= caught_up)
        .filter_map(|(oseq, at)| c_store.arrived(1, *oseq).map(|got| got.duration_since(*at)))
        .collect();
    assert!(latencies.len() > 50, "{} live samples", latencies.len());
    latencies.sort();
    let p99 = latencies[latencies.len() * 99 / 100];
    println!(
        "H4: {} live events, live p99 {p99:?} during catch-up, converged in {:?}",
        written.len(),
        started.elapsed()
    );
    assert!(p99 <= LIVE_P99, "live p99 {p99:?} over {LIVE_P99:?}");
}
