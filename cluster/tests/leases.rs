//! The app's leases and pulses (§9.1, §9.2, §9.6): a world lease held by one
//! member at a time, handed back to the next; its holder's claim routing the
//! world's `Owner` routes to it; and pulses reaching the other members.

mod support;

use std::sync::Arc;
use std::time::Duration;

use cluster::consensus::votes::MemoryVotes;
use cluster::coord::CoordState;
use cluster::dispatch::{Dispatch, KeySource};
use cluster::frame::Pulse;
use cluster::gateway::Pick;
use cluster::replicate::{LeaseEvent, MemoryStore, Store, LEASE_TTL};
use tokio::sync::broadcast::Receiver;
use tokio::time::timeout;

use support::{gateway, member, until, APP};

const WITHIN: Duration = Duration::from_secs(20);

fn store(node: u64) -> Arc<MemoryStore> {
    Arc::new(MemoryStore::new(node))
}

/// The next event about `key` that is about this member's own standing:
/// another member's release is read with [`released`].
async fn next(events: &mut Receiver<LeaseEvent>, key: &[u8]) -> LeaseEvent {
    timeout(WITHIN, async {
        loop {
            let e = events.recv().await.unwrap();
            if e.key() == key && !matches!(e, LeaseEvent::Released { .. }) {
                return e;
            }
        }
    })
    .await
    .expect("a lease event")
}

/// The next release of `key` by another member.
async fn released(events: &mut Receiver<LeaseEvent>, key: &[u8]) -> LeaseEvent {
    timeout(WITHIN, async {
        loop {
            let e = events.recv().await.unwrap();
            if e.key() == key && matches!(e, LeaseEvent::Released { .. }) {
                return e;
            }
        }
    })
    .await
    .expect("a release")
}

/// One member holds a world's lease; another asking is told who; once it is
/// handed back, the other takes it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_world_lease_has_one_holder_and_moves_when_handed_back() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready()).await;
    until("a quorum of two", WITHIN, || {
        a.store.max_members() == 2 && b.store.max_members() == 2
    })
    .await;
    let ra = a.app.replicator().unwrap().clone();
    let rb = b.app.replicator().unwrap().clone();
    let mut ea = ra.lease_events();
    let mut eb = rb.lease_events();
    let mut seen_by_b = rb.lease_events();
    let key = b"world/w1".to_vec();

    assert!(ra.acquire(&key, LEASE_TTL));
    assert_eq!(
        next(&mut ea, &key).await,
        LeaseEvent::Held { key: key.clone() }
    );
    assert!(ra.holds(&key));

    assert!(rb.acquire(&key, LEASE_TTL));
    assert_eq!(
        next(&mut eb, &key).await,
        LeaseEvent::Busy {
            key: key.clone(),
            holder: 1
        }
    );
    assert!(!rb.holds(&key));

    ra.release_lease(&key, 42);
    assert_eq!(
        next(&mut ea, &key).await,
        LeaseEvent::Lost { key: key.clone() }
    );
    // B hears how far A's writes reach, to wait for them before it runs the
    // world.
    assert_eq!(
        released(&mut seen_by_b, &key).await,
        LeaseEvent::Released {
            key: key.clone(),
            by: 1,
            through: 42
        }
    );
    // The release reaches every acceptor in its own time: until it has, a
    // promise still carries A's lease, and B asks again, as the proposer
    // rendezvous names does.
    loop {
        assert!(rb.acquire(&key, LEASE_TTL));
        match next(&mut eb, &key).await {
            LeaseEvent::Held { .. } => break,
            LeaseEvent::Busy { holder: 1, .. } => {
                tokio::time::sleep(Duration::from_millis(100)).await
            }
            other => panic!("{other:?}"),
        }
    }
    assert!(rb.holds(&key));
}

/// **A member arriving in a group its substrate still counts as one cannot
/// take a lease the first member holds**: the quorum counts every ready member
/// it sees, so before `max_members` has risen it already needs the holder's
/// promise, and the holder's lease stops it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_arriving_before_the_quorum_rises_cannot_take_a_held_lease() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    until("a ready", WITHIN, || a.ready()).await;
    let ra = a.app.replicator().unwrap().clone();
    let mut ea = ra.lease_events();
    let key = b"world/w3".to_vec();
    assert!(ra.acquire(&key, LEASE_TTL));
    assert_eq!(
        next(&mut ea, &key).await,
        LeaseEvent::Held { key: key.clone() }
    );

    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("b ready, and seen ready by both", WITHIN, || {
        b.ready() && a.app.group().len() == 2 && b.app.group().len() == 2
    })
    .await;
    let rb = b.app.replicator().unwrap().clone();
    let mut eb = rb.lease_events();
    // At once, before either has raised `max_members` from one.
    assert!(rb.acquire(&key, LEASE_TTL));
    assert_eq!(
        next(&mut eb, &key).await,
        LeaseEvent::Busy {
            key: key.clone(),
            holder: 1
        }
    );
    assert!(!rb.holds(&key));
    assert!(ra.holds(&key));
}

/// **A lease taken by a member alone is made known to the group as it grows.**
/// Once a majority of the larger group has accepted it, the holder cut off
/// still holds it there: the others are told it is held, until its term runs
/// out, rather than granting it again at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lease_taken_alone_is_known_to_the_group_it_grows_into() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    until("a ready", WITHIN, || a.ready()).await;
    let ra = a.app.replicator().unwrap().clone();
    let mut ea = ra.lease_events();
    let key = b"world/w4".to_vec();
    assert!(ra.acquire(&key, LEASE_TTL));
    assert_eq!(
        next(&mut ea, &key).await,
        LeaseEvent::Held { key: key.clone() }
    );

    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    let c = member(&[&gw], 3, store(3), MemoryVotes::new(), true);
    until("all ready", WITHIN, || a.ready() && b.ready() && c.ready()).await;
    // A asks again under the larger quorum: b and c's acceptors take its
    // lease.
    let accepted_a = |coord: &CoordState| {
        coord.paxos.keys.iter().any(|k| {
            k.key == "world/w4"
                && k.accepted
                    .as_deref()
                    .is_some_and(|b| b.ends_with("0000000000000001"))
        })
    };
    let rb = b.app.replicator().unwrap().clone();
    let rc = c.app.replicator().unwrap().clone();
    timeout(WITHIN, async {
        loop {
            let (cb, cc) = (rb.coord().await.unwrap(), rc.coord().await.unwrap());
            if accepted_a(&cb) && accepted_a(&cc) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the group learnt a's lease");

    a.reach(0, false);
    a.app.close_all("the network went");
    let mut ec = rc.lease_events();
    assert!(rc.acquire(&key, LEASE_TTL));
    assert_eq!(
        next(&mut ec, &key).await,
        LeaseEvent::Busy {
            key: key.clone(),
            holder: 1
        },
        "a's term still runs at b and c"
    );
}

/// A drained member asks for no lease and hands back the ones it held.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drained_member_holds_no_lease() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready()).await;
    let ra = a.app.replicator().unwrap().clone();
    let mut ea = ra.lease_events();
    let key = b"world/w2".to_vec();
    assert!(ra.acquire(&key, LEASE_TTL));
    assert_eq!(
        next(&mut ea, &key).await,
        LeaseEvent::Held { key: key.clone() }
    );
    let handed = ra.drain().await;
    assert_eq!(handed.leases, 1);
    assert_eq!(
        next(&mut ea, &key).await,
        LeaseEvent::Lost { key: key.clone() }
    );
    assert!(
        !ra.acquire(&key, LEASE_TTL),
        "a drained member proposes nothing"
    );
}

/// A lease's holder claims the world's id: the gateway routes the world's
/// `Owner` routes to it, and to the next holder once it lets go.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_holders_claim_routes_the_world() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready()).await;
    let owner = Dispatch::Owner(KeySource::Path("wid".into()));
    a.app.hold(b"w1");
    until("routed to a", WITHIN, || {
        gw.pick(APP, &owner, Some("w1"), None).pick == Pick::Member(1)
    })
    .await;
    a.app.let_go(b"w1", 7);
    b.app.hold(b"w1");
    until("routed to b", WITHIN, || {
        gw.pick(APP, &owner, Some("w1"), None).pick == Pick::Member(2)
    })
    .await;
}

/// A pulse reaches every other member, once, and never the sender.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pulse_reaches_the_other_members() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    let c = member(&[&gw], 3, store(3), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready() && c.ready()).await;
    let mut pa = a.app.replicator().unwrap().pulses();
    let mut pb = b.app.replicator().unwrap().pulses();
    let mut pc = c.app.replicator().unwrap().pulses();
    let pulse = Pulse {
        world: "w1".into(),
        kind: "step".into(),
        body: b"{\"npc\":\"n1\"}".to_vec(),
    };
    assert!(a.app.replicator().unwrap().pulse(pulse.clone()));
    for rx in [&mut pb, &mut pc] {
        let got = timeout(WITHIN, rx.recv()).await.unwrap().unwrap();
        assert_eq!(got, pulse);
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(pa.try_recv().is_err(), "the sender hears nothing back");
    assert!(pb.try_recv().is_err(), "once");
}
