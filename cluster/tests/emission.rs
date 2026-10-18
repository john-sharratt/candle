//! Which socket an event takes, and a member that cannot keep up (§6.3, §6.4,
//! §4.4): the emission and lagging tables of §14.1, and the catch-up that
//! finds a lost last event through `high_water` (§7.1).

mod support;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cluster::consensus::votes::MemoryVotes;
use cluster::dispatch::Dispatch;
use cluster::gateway::{Pick, LAG_CAP};
use cluster::replicate::{MemoryStore, GAP_WAIT};

use support::{converged, gateway, member, until, APP};

const WITHIN: Duration = Duration::from_secs(30);

fn store(node: u64) -> Arc<MemoryStore> {
    Arc::new(MemoryStore::new(node))
}

/// A request's events leave on the request's own socket: a member reached
/// only through the other gateway does not see them live, and finds the last
/// of them through `high_water` once its origin has announced it for 5 s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_requests_events_leave_on_its_socket() {
    let g0 = gateway("g0");
    let g1 = gateway("g1");
    let a = member(&[&g0, &g1], 1, store(1), MemoryVotes::new(), true);
    let on_g0 = member(&[&g0], 2, store(2), MemoryVotes::new(), true);
    let on_g1 = member(&[&g1], 3, store(3), MemoryVotes::new(), true);
    until("ready", WITHIN, || {
        a.ready() && on_g0.ready() && on_g1.ready()
    })
    .await;

    a.store.write(b"k", b"the request's write");
    a.app.replicator().unwrap().flush(1).await;
    until("the request's gateway relayed it", WITHIN, || {
        on_g1.store.value(b"k").is_some()
    })
    .await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        on_g0.store.value(b"k").is_none(),
        "it went on the request's socket alone"
    );
    // A's `ACK` announces it as `high_water`; after 5 s the member behind
    // asks A directly.
    until("the lost last event found", WITHIN + GAP_WAIT, || {
        on_g0.store.value(b"k").is_some()
    })
    .await;
}

/// Events no request took are dealt round robin, one whole event at a time:
/// each gateway carries some of them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn other_events_go_round_robin() {
    let g0 = gateway("g0");
    let g1 = gateway("g1");
    let a = member(&[&g0, &g1], 1, store(1), MemoryVotes::new(), true);
    let on_g0 = member(&[&g0], 2, store(2), MemoryVotes::new(), true);
    let on_g1 = member(&[&g1], 3, store(3), MemoryVotes::new(), true);
    until("ready", WITHIN, || {
        a.ready() && on_g0.ready() && on_g1.ready()
    })
    .await;
    for k in 0..100u32 {
        a.store.write(format!("rr/{k}").as_bytes(), b"v");
    }
    // Before any gap catch-up could run, each side holds what its gateway
    // relayed live.
    tokio::time::sleep(Duration::from_secs(1)).await;
    let held = |s: &MemoryStore| {
        (0..100u32)
            .filter(|k| s.value(format!("rr/{k}").as_bytes()).is_some())
            .count()
    };
    let (zero, one) = (held(&on_g0.store), held(&on_g1.store));
    assert_eq!(
        zero + one,
        100,
        "each event once, on one socket: {zero} + {one}"
    );
    assert!((30..=70).contains(&zero), "{zero} of 100 on g0");
}

/// A member that stops reading its socket becomes lagging once its lane
/// passes the cap: no further live event is queued for it and no request is
/// routed to it; its socket stays open. Once its lane drains it takes live
/// events again and syncs — **while the origin is still writing**, which a
/// lagging member, chasing writes it is not sent, could never catch up with —
/// and is routed again once in sync.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_slow_member_lags_and_comes_back() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let slow = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && slow.ready()).await;
    let session = gw.member(APP, 2).unwrap().session;

    slow.pause(0, true);
    let value = vec![7u8; (1 << 20) - 64];
    let n = (LAG_CAP >> 20) + 8;
    for k in 0..n {
        a.store.write(format!("big/{k}").as_bytes(), &value);
    }
    until("lagging", WITHIN, || {
        gw.member(APP, 2).is_some_and(|m| m.lagging)
    })
    .await;
    for _ in 0..4 {
        assert_ne!(
            gw.pick(APP, &Dispatch::Any, None, None).pick,
            Pick::Member(2),
            "no request routed to a lagging member"
        );
    }
    let status = gw.status(0);
    let row = status
        .members
        .iter()
        .find(|m| m.node == format!("{:016x}", 2))
        .unwrap();
    assert!(row.queues.skipped > 0);

    // The origin keeps writing throughout the recovery, as an ingesting
    // member does.
    let writing = Arc::new(AtomicBool::new(true));
    let writer = tokio::spawn({
        let store = a.store.clone();
        let writing = writing.clone();
        async move {
            let mut k = 0u64;
            while writing.load(Ordering::SeqCst) {
                store.write(format!("trickle/{k}").as_bytes(), b"t");
                k += 1;
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }
    });
    slow.pause(0, false);
    until(
        "routed again while the origin writes",
        WITHIN + GAP_WAIT,
        || gw.member(APP, 2).is_some_and(|m| m.routable()),
    )
    .await;
    writing.store(false, Ordering::SeqCst);
    writer.await.unwrap();
    until("converged", WITHIN + GAP_WAIT, || {
        converged(&[&a.store, &slow.store])
    })
    .await;
    assert_eq!(
        gw.member(APP, 2).unwrap().session,
        session,
        "its socket was never closed"
    );
}
