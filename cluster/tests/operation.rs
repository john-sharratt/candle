//! Operating the cluster (§7.7, §12): fault mode for a full disk, a member's
//! drain, what `COORD` puts on the gateway's page and in its log, the quorum
//! admin call, and a gateway's own drain — the fault-mode, quorum and
//! status-page tables of §14.1.

mod support;

use std::sync::Arc;
use std::time::Duration;

use cluster::consensus::votes::MemoryVotes;
use cluster::gateway::Hub;
use cluster::member::Check;
use cluster::replicate::{HandedBack, MemoryStore, Phase, Store};

use support::{converged, gateway, member, until};

const WITHIN: Duration = Duration::from_secs(20);
const GIB: u64 = 1 << 30;

fn store(node: u64) -> Arc<MemoryStore> {
    Arc::new(MemoryStore::new(node))
}

fn state_on(gw: &Hub, node: u64) -> Option<&'static str> {
    let hex = format!("{node:016x}");
    gw.status(0)
        .members
        .into_iter()
        .find(|m| m.node == hex)
        .map(|m| m.state)
}

fn logged(gw: &Hub, node: u64, text: &str) -> bool {
    let hex = format!("{node:016x}");
    gw.log
        .read(None, Some(&hex), 1000)
        .iter()
        .any(|e| e.text.contains(text))
}

/// A member whose volume falls under the floor stays connected in fault
/// mode: it hands back its claims, applies nothing, still sends its own
/// events, and shows on the page. Once space is back above twice the floor it
/// leaves by itself, catches up on what it missed, and is ready again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_disk_faults_and_recovers_by_itself() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready()).await;
    assert!(matches!(b.app.claim(b"conv"), Check::Took(_)));
    until("the claim is known", WITHIN, || {
        gw.status(0).claims.iter().any(|c| c.key == "conv")
    })
    .await;

    // Under the 2 GiB floor of a 100 GiB volume.
    b.store.set_disk(GIB, 100 * GIB, false);
    let rb = b.app.replicator().unwrap().clone();
    until("b in fault mode", WITHIN, || rb.faulted()).await;
    until("the gateway sees the fault", WITHIN, || {
        state_on(&gw, 2) == Some("fault")
            && gw
                .status(0)
                .faults
                .iter()
                .any(|f| f.node == format!("{:016x}", 2) && f.fault.is_some())
    })
    .await;
    until("the claim handed back", WITHIN, || {
        gw.status(0).claims.iter().all(|c| c.key != "conv")
    })
    .await;
    assert_eq!(rb.fault().unwrap().claims_released, 1);
    assert!(!b.ready() || state_on(&gw, 2) == Some("fault"));

    // It applies nothing that arrives, and still sends what it writes.
    for k in 0..20u32 {
        a.store.write(format!("a/{k}").as_bytes(), b"x");
    }
    b.store.write(b"b/own", b"y");
    until("b's own event reaches a", WITHIN, || {
        a.store.value(b"b/own").is_some()
    })
    .await;
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(
        (0..20u32).all(|k| b.store.value(format!("a/{k}").as_bytes()).is_none()),
        "nothing is applied in fault mode"
    );

    // Space comes back, but not past twice the floor: it stays.
    b.store.set_disk(3 * GIB, 100 * GIB, false);
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(
        rb.faulted(),
        "the hysteresis holds it below twice the floor"
    );

    b.store.set_disk(10 * GIB, 100 * GIB, false);
    until("b left fault mode and caught up", WITHIN, || {
        !rb.faulted() && b.ready() && converged(&[&a.store, &b.store])
    })
    .await;
    until("ready on the gateway", WITHIN, || {
        state_on(&gw, 2) == Some("ready")
    })
    .await;
    until("both logged", WITHIN, || {
        logged(&gw, 2, "entered fault mode") && logged(&gw, 2, "left fault mode")
    })
    .await;
}

/// A failed write enters fault mode however much space the volume reports,
/// and leaving needs a test write to succeed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_write_faults_until_a_test_write_succeeds() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready()).await;
    let ra = a.app.replicator().unwrap().clone();
    a.store.set_disk(50 * GIB, 100 * GIB, true);
    until("faulted", WITHIN, || ra.faulted()).await;
    assert_eq!(
        ra.fault().unwrap().reason,
        "a write failed for lack of space"
    );
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(ra.faulted(), "the test write still fails");
    a.store.set_disk(50 * GIB, 100 * GIB, false);
    until("left", WITHIN, || !ra.faulted() && a.ready()).await;
}

/// A member in fault mode still votes: a cut-off member takes its generation
/// with the vote of a faulted one as the quorum's second (§7.7, §5.9).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_faulted_member_still_votes() {
    let gw = gateway("gw");
    let c_store = store(3);
    c_store.set_max_members(3);
    c_store.detach();
    c_store.write(b"turn/t/c1", b"offline");
    let b_store = store(2);
    b_store.set_disk(GIB, 100 * GIB, false);
    let b = member(&[&gw], 2, b_store, MemoryVotes::new(), true);
    let rb = b.app.replicator().unwrap().clone();
    until("b faulted", WITHIN, || rb.faulted()).await;

    let c = member(&[&gw], 3, c_store, MemoryVotes::new(), true);
    until(
        "c rejoined on a faulted member's vote",
        Duration::from_secs(40),
        || c.app.replicator().unwrap().phase() == Phase::Joined,
    )
    .await;
    assert!(rb.faulted(), "b voted from fault mode");
    assert!(!c.store.detached());
    until("the rejoin is logged", WITHIN, || {
        logged(&gw, 3, "rejoined: took generation")
    })
    .await;
    // The rounds it carried are in the log under the generation's key: C
    // prepared, and B — faulted — promised and accepted.
    let steps: Vec<String> = gw
        .log
        .read(None, None, 1000)
        .into_iter()
        .filter(|e| e.subject.starts_with("gen/"))
        .map(|e| e.text)
        .collect();
    for want in [
        "prepared at",
        "0000000000000002 promised",
        "proposed at",
        "0000000000000002 accepted",
    ] {
        assert!(steps.iter().any(|t| t.contains(want)), "{want}: {steps:#?}");
    }
}

/// `drain` hands back the member's claims and routes nothing new to it while
/// it stays connected; `undrain` makes it ready again after a catch-up.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drained_member_stays_connected_and_comes_back() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready()).await;
    assert!(matches!(b.app.claim(b"conv"), Check::Took(_)));
    let rb = b.app.replicator().unwrap().clone();
    // The gateway has B's first report, so the drain shows as a change.
    until("b reporting", WITHIN, || {
        logged(&gw, 2, "reporting: joined")
    })
    .await;

    let handed = rb.drain().await;
    assert_eq!(
        handed,
        HandedBack {
            claims: 1,
            leases: 0
        }
    );
    until("draining on the gateway", WITHIN, || {
        state_on(&gw, 2) == Some("draining")
    })
    .await;
    until("the drain logged", WITHIN, || {
        logged(&gw, 2, "mode joined → draining")
    })
    .await;
    assert!(gw.status(0).claims.iter().all(|c| c.key != "conv"));
    assert!(rb.undrain().is_ok());
    until("ready again", WITHIN, || {
        state_on(&gw, 2) == Some("ready") && !rb.drained()
    })
    .await;
    until("logged", WITHIN, || {
        logged(&gw, 2, "mode draining → joined")
    })
    .await;
}

/// Every member's `COORD` reaches the page: its mode, generation and quorum
/// on its row, `max_members` and the quorum on the app, and its first report
/// in the log.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn coord_reaches_the_page() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready()).await;
    until("both report a quorum of two", WITHIN, || {
        let s = gw.status(0);
        s.members.len() == 2
            && s.members.iter().all(|m| {
                m.coord
                    .as_ref()
                    .is_some_and(|c| c.mode == "joined" && c.paxos.max_members == 2)
            })
    })
    .await;
    let s = gw.status(0);
    let zend = s.apps.iter().find(|x| x.app == "zend").unwrap();
    assert_eq!((zend.max_members, zend.quorum), (Some(2), Some(2)));
    let row = s
        .members
        .iter()
        .find(|m| m.node == format!("{:016x}", 1))
        .unwrap();
    let coord = row.coord.as_ref().unwrap();
    assert_eq!(coord.clusters.len(), 1);
    assert_eq!(coord.clusters[0].state, "admitted");
    // 1% of the test volume's 2 TiB, over the 2 GiB minimum.
    assert_eq!(coord.disk.floor, 21_990_232_555);
    assert!(row.clock_offset_ms.abs() < 1000);
    assert!(logged(&gw, 1, "reporting: joined"));
    assert!(s.faults.is_empty());
}

/// The quorum admin call writes `max_members` lower, every member takes it,
/// and the change is logged.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_quorum_call_lowers_max_members() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    let c = member(&[&gw], 3, store(3), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready() && c.ready()).await;
    until("three", WITHIN, || {
        a.store.max_members() == 3 && b.store.max_members() == 3
    })
    .await;
    c.stop();
    // Every member's `MEMBERS` has dropped C before the call, or a member
    // still counting three ready raises the value straight back.
    until("c gone", WITHIN, || gw.status(0).members.len() == 2).await;
    tokio::time::sleep(Duration::from_secs(1)).await;
    a.app.replicator().unwrap().set_quorum(2).await;
    until("both hold two", WITHIN, || {
        a.store.max_members() == 2 && b.store.max_members() == 2
    })
    .await;
    until("logged", WITHIN, || {
        logged(&gw, 2, "max_members 3 → 2, quorum 2")
    })
    .await;
}

/// A drained gateway closes its members' sockets and admits none; they carry
/// on through their other cluster, and come back once it undrains.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drained_gateway_sends_members_to_the_other_cluster() {
    let one = gateway("one");
    let two = gateway("two");
    let a = member(&[&one, &two], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&one, &two], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready()).await;
    until("on both", WITHIN, || {
        one.status(0).members.len() == 2 && two.status(0).members.len() == 2
    })
    .await;

    assert_eq!(one.drain_gateway().await, 0);
    assert!(one.status(0).draining);
    until("gone from the drained gateway", WITHIN, || {
        one.status(0).members.is_empty()
    })
    .await;
    a.store.write(b"after/drain", b"x");
    until("still replicating through the other", WITHIN, || {
        b.store.value(b"after/drain").is_some()
    })
    .await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert!(one.status(0).members.is_empty(), "it admits nobody");

    one.undrain_gateway();
    until("back on it", Duration::from_secs(60), || {
        one.status(0).members.len() == 2
    })
    .await;
}
