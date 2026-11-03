//! Records a member carries to another itself, outside the stream
//! (`Replicator::apply_carried`): what a member handing work on sends of its
//! own records the member taking it lacks.

mod support;

use std::sync::Arc;
use std::time::Duration;

use cluster::consensus::votes::MemoryVotes;
use cluster::replicate::{MemoryStore, Store};

use support::{gateway, member, until};

const WITHIN: Duration = Duration::from_secs(20);

/// **Carried records apply as arrivals do**: the member taking them holds
/// them and counts them applied, and one its store refuses is counted
/// refused, not applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn carried_records_apply_as_arrivals_do() {
    // The member that wrote them is no member here: nothing of its reaches
    // the other by the stream.
    let writer = MemoryStore::new(1);
    let oseqs: Vec<u64> = (0..3u32)
        .map(|k| writer.write(format!("w/{k}").as_bytes(), &k.to_le_bytes()))
        .collect();
    let carried = || -> Vec<(u64, Vec<u8>)> {
        oseqs
            .iter()
            .map(|&oseq| (oseq, writer.read(1, oseq).unwrap()))
            .collect()
    };

    let gw = gateway("gw");
    let b = member(
        &[&gw],
        2,
        Arc::new(MemoryStore::new(2)),
        MemoryVotes::new(),
        true,
    );
    until("b ready", WITHIN, || b.ready()).await;
    let replicator = b.app.replicator().unwrap().clone();
    assert_eq!(replicator.apply_carried(1, carried()).await, 3);
    for (k, oseq) in oseqs.iter().enumerate() {
        assert!(b.store.applied().contains(1, *oseq));
        assert_eq!(
            b.store.value(format!("w/{k}").as_bytes()),
            Some((k as u32).to_le_bytes().to_vec())
        );
    }
    assert_eq!(replicator.view().stats.applied, 3);
    assert_eq!(replicator.apply_carried(1, Vec::new()).await, 0);

    // A cluster of its own, so no catch-up offers it what `b` now holds.
    let alone = gateway("alone");
    let c = member(
        &[&alone],
        3,
        Arc::new(MemoryStore::new(3)),
        MemoryVotes::new(),
        true,
    );
    until("c ready", WITHIN, || c.ready()).await;
    c.store.refuse_origin(1);
    let refusing = c.app.replicator().unwrap().clone();
    let before = refusing.view().stats;
    assert_eq!(refusing.apply_carried(1, carried()).await, 0);
    let after = refusing.view().stats;
    assert_eq!(after.refused - before.refused, 3);
    assert_eq!(after.applied, before.applied);
}
