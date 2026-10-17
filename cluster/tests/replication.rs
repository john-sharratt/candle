//! Replication end to end (§6.3–§7.5): members on [`MemoryStore`]s behind
//! in-process gateways — the emission, retransmit, catch-up, full-resync and
//! generation tables of §14.1.

mod support;

use std::sync::Arc;
use std::time::Duration;

use cluster::consensus::votes::MemoryVotes;
use cluster::member::{Check, Link};
use cluster::replicate::{MemoryRecord, MemoryStore, Phase, Store};

use support::{converged, gateway, member, until};

const WITHIN: Duration = Duration::from_secs(20);

fn store(node: u64) -> Arc<MemoryStore> {
    Arc::new(MemoryStore::new(node))
}

/// Every member's writes reach every other member, once, and all end
/// holding the same records.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn writes_reach_every_other_member() {
    let gw = gateway("gw");
    let nodes: Vec<_> = (1..=3)
        .map(|n| member(&[&gw], n, store(n), MemoryVotes::new(), true))
        .collect();
    until("all ready", WITHIN, || nodes.iter().all(|n| n.ready())).await;
    for (i, n) in nodes.iter().enumerate() {
        for k in 0..50u32 {
            n.store
                .write(format!("k/{i}/{k}").as_bytes(), &k.to_le_bytes());
        }
    }
    let stores: Vec<&MemoryStore> = nodes.iter().map(|n| &*n.store).collect();
    until("converged", WITHIN, || {
        converged(&stores) && stores[0].len() >= 150
    })
    .await;
    // Nobody received its own events back.
    for n in &nodes {
        let r = n.app.replicator().unwrap().view();
        assert_eq!(r.stats.refused, 0, "{r:?}");
    }
}

/// A member away while the others write fetches exactly what it lacked when
/// it comes back, by header compare (§7.2).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_member_away_catches_up_by_headers() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("a and b ready", WITHIN, || a.ready() && b.ready()).await;
    for k in 0..200u32 {
        a.store.write(format!("a/{k}").as_bytes(), b"x");
        b.store.write(format!("b/{k}").as_bytes(), b"y");
    }
    until("a and b converged", WITHIN, || {
        converged(&[&a.store, &b.store]) && a.store.len() >= 400
    })
    .await;

    let answered = || {
        a.app.replicator().unwrap().view().stats.answered
            + b.app.replicator().unwrap().view().stats.answered
    };
    let before = answered();
    let c = member(&[&gw], 3, store(3), MemoryVotes::new(), true);
    until("c caught up", WITHIN, || {
        c.ready() && converged(&[&a.store, &b.store, &c.store])
    })
    .await;
    // Each of C's catch-ups was answered by one member alone; no other member
    // saw it.
    let asked = c.app.replicator().unwrap().view().stats.catchups;
    assert!(asked >= 1);
    assert_eq!(answered() - before, asked);
}

/// A record the answerer compacted away is learned as gone, not fetched,
/// and the requester does not ask again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_compacted_record_is_learned_gone() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    until("a ready", WITHIN, || a.ready()).await;
    for k in 0..10u32 {
        a.store.write(format!("a/{k}").as_bytes(), b"x");
    }
    // `a/3`, its fourth record.
    a.store.compact_away(1, 4);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("b caught up", WITHIN, || {
        b.ready()
            && (0..10u32)
                .filter(|k| *k != 3)
                .all(|k| b.store.value(format!("a/{k}").as_bytes()).is_some())
    })
    .await;
    until("the gone record recorded", WITHIN, || {
        b.store.applied().contains(1, 4)
    })
    .await;
    assert!(b.store.read(1, 4).is_none(), "learned gone, never fetched");
    assert!(b.store.value(b"a/3").is_none());
}

/// **A refused record is not caught up on.** It arrives, the store refuses
/// it, and it is fetched again — so no catch-up may count it: counted, a
/// catch-up that refused everything read as one that kept receiving, past its
/// own total, and held a deploy's trial open (`switch::SYNCING`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_refused_record_is_not_counted_as_caught_up() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    until("a ready", WITHIN, || a.ready()).await;
    for k in 0..10u32 {
        a.store.write(format!("a/{k}").as_bytes(), b"x");
    }
    let refusing = store(2);
    refusing.refuse_origin(1);
    let b = member(&[&gw], 2, refusing, MemoryVotes::new(), true);
    let mut counted = 0;
    until("b refused a's records", WITHIN, || {
        let view = b.app.replicator().unwrap().view();
        counted = counted.max(
            view.catchups
                .iter()
                .map(|c| c.progress.fetched)
                .max()
                .unwrap_or(0),
        );
        view.stats.refused >= 10
    })
    .await;
    assert_eq!(counted, 0, "a refused record counted as fetched");
    assert!(b.store.value(b"a/0").is_none());
}

/// A restarted member offers what it holds; a member lacking some of it
/// catches up from it directly (§7.5). Here A holds records of an origin
/// that never joins.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restarted_members_offer_is_taken() {
    let gw = gateway("gw");
    let ghost = MemoryStore::new(9);
    for k in 0..20u32 {
        ghost.write(format!("ghost/{k}").as_bytes(), b"g");
    }
    let a_store = store(1);
    let records: Vec<_> = (1..=20)
        .map(|q| (9, q, ghost.read(9, q).unwrap()))
        .collect();
    a_store.apply(&records);

    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("b ready", WITHIN, || b.ready()).await;
    let a = member(&[&gw], 1, a_store, MemoryVotes::new(), true);
    until("b holds the ghost's records", WITHIN, || {
        b.store.applied().through(9) == 20
    })
    .await;
    until("converged", WITHIN, || converged(&[&a.store, &b.store])).await;
}

/// A member's own events not acknowledged by any member when it restarted
/// go out again; those acknowledged do not (§7.5).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restart_resends_only_what_no_member_acknowledged() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready()).await;
    for k in 0..30u32 {
        a.store.write(format!("a/{k}").as_bytes(), b"x");
    }
    until("b has a's", WITHIN, || {
        b.store.applied().through(1) == a.store.own_through()
    })
    .await;
    // B's ACK reaches the gateway within a second.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let (a_store, votes) = a.stop();
    // Written just before the stop and never sent: what a restart owes.
    for k in 30..40u32 {
        a_store.write(format!("a/{k}").as_bytes(), b"x");
    }
    let a = member(&[&gw], 1, a_store, votes, true);
    until("b has the rest", WITHIN, || {
        b.store.applied().through(1) == a.store.own_through()
    })
    .await;
    until("a ready", WITHIN, || a.ready()).await;
    let resent = a.app.replicator().unwrap().view();
    assert_eq!(resent.unsent, 0, "{resent:?}");
}

/// A member cut off writes detached; on rejoin it catches up, takes a
/// generation by Paxos, and its turns sort after the group's on every member
/// (§6.7).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cut_off_members_turns_sort_after_the_groups() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    let c = member(&[&gw], 3, store(3), MemoryVotes::new(), true);
    until("all ready", WITHIN, || a.ready() && b.ready() && c.ready()).await;
    a.store.write(b"turn/t/0", b"opening");
    until("everyone has the opening", WITHIN, || {
        converged(&[&a.store, &b.store, &c.store]) && c.store.len() >= a.store.len()
    })
    .await;

    c.reach(0, false);
    c.app.close_all("the network went");
    until("c cut off", WITHIN, || {
        c.app.replicator().unwrap().phase() == Phase::CutOff
    })
    .await;
    assert_eq!(c.app.link(), Link::CutOff);
    c.store.write(b"turn/t/c1", b"offline 1");
    c.store.write(b"turn/t/c2", b"offline 2");
    a.store.write(b"turn/t/a1", b"group 1");
    b.store.write(b"turn/t/b1", b"group 2");
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        a.store.value(b"turn/t/c1").is_none(),
        "nothing detached is sent"
    );

    c.reach(0, true);
    until("c rejoined", WITHIN, || {
        c.app.replicator().unwrap().phase() == Phase::Joined && c.ready()
    })
    .await;
    until("converged", WITHIN, || {
        converged(&[&a.store, &b.store, &c.store]) && a.store.value(b"turn/t/c2").is_some()
    })
    .await;
    let order = |s: &MemoryStore| -> Vec<Vec<u8>> {
        s.ordered(b"turn/t/")
            .into_iter()
            .map(|(_, v, _)| v)
            .collect()
    };
    let want: Vec<Vec<u8>> = vec![
        b"opening".to_vec(),
        b"group 1".to_vec(),
        b"group 2".to_vec(),
        b"offline 1".to_vec(),
        b"offline 2".to_vec(),
    ];
    for s in [&a.store, &b.store, &c.store] {
        let got = order(s);
        assert_eq!(got[0], want[0]);
        assert_eq!(got[3..], want[3..], "the offline turns come last");
        assert_eq!(got.len(), 5);
    }
    // The generation was decided and recorded everywhere.
    let g = c.store.generation();
    assert!(g >= 1);
    for s in [&a.store, &b.store] {
        assert_eq!(
            s.decision(format!("gen/{g}").as_bytes()),
            Some(3u64.to_le_bytes().to_vec())
        );
    }
    let fixed = MemoryRecord::decode(&a.store.read(3, c.store.own_through()).unwrap()).unwrap();
    assert_eq!(fixed.gen, g);
}

/// A turn that arrives ranked below a turn the member holds moves that turn's
/// index: the member stops taking work, hands back its claims, reloads, and
/// comes back (§6.6).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_turn_ranked_below_one_held_reloads_the_member() {
    let gw = gateway("gw");
    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    let b = member(&[&gw], 2, store(2), MemoryVotes::new(), true);
    until("ready", WITHIN, || a.ready() && b.ready()).await;
    assert!(matches!(b.app.claim(b"conv"), Check::Took(_)));

    // B holds a turn under a later epoch; A's, written after it under an
    // earlier one, arrives ranked below it.
    b.store.write_under(9, b"turn/conv/b", b"later");
    a.store.write_under(5, b"turn/conv/a", b"earlier");
    until("b reloaded", WITHIN, || {
        b.store.reloads() >= 1 && b.app.replicator().unwrap().phase() == Phase::Joined
    })
    .await;
    assert_eq!(
        b.app.replicator().unwrap().view().stats.reloads,
        b.store.reloads()
    );
    until("b's claim handed back", WITHIN, || {
        gw.status(0).claims.iter().all(|c| c.key != "conv")
    })
    .await;
    until("converged", WITHIN, || converged(&[&a.store, &b.store])).await;
}

/// Without a quorum a rejoining member stays cut off, serving locally and
/// sending nothing it wrote cut off; it takes its generation once a quorum is
/// reachable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rejoin_without_a_quorum_stays_cut_off() {
    let gw = gateway("gw");
    let c_store = store(3);
    // The group has been three; C is alone with the gateway.
    c_store.set_max_members(3);
    c_store.detach();
    c_store.write(b"turn/t/c1", b"offline");
    let c = member(&[&gw], 3, c_store, MemoryVotes::new(), true);
    tokio::time::sleep(Duration::from_secs(12)).await;
    assert_eq!(c.app.replicator().unwrap().phase(), Phase::CutOff);
    assert_eq!(c.app.link(), Link::CutOff);
    assert!(c.store.detached());

    let a = member(&[&gw], 1, store(1), MemoryVotes::new(), true);
    until(
        "c rejoined once a quorum is there",
        Duration::from_secs(30),
        || c.app.replicator().unwrap().phase() == Phase::Joined,
    )
    .await;
    until("a has c's offline write", WITHIN, || {
        a.store.value(b"turn/t/c1").is_some()
    })
    .await;
}
