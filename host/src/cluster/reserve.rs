//! This member stepping aside for the machine's reservation
//! (`docs/house_modes.md` §4, `cluster::reserve::step_aside`): once a second,
//! while the machine is reserved and another member is ready for every app,
//! every app drains; the moment the reservation ends or an app has nobody
//! else, the drain it made is undone.

use std::sync::{Arc, Weak};
use std::time::Duration;

use cluster::replicate::Replicator;
use cluster::reserve::machine as reservation;
use cluster::reserve::step_aside::{step, Step};
use tokio::time::interval;

use super::Cluster;

/// How often the member looks again.
const EVERY: Duration = Duration::from_secs(1);

/// Every app's replication, once its member has joined.
fn replicators(cluster: &Cluster) -> Vec<Arc<Replicator>> {
    cluster
        .links()
        .filter_map(|l| l.member()?.replicator().cloned())
        .collect()
}

/// Follow the reservation for `cluster`'s apps, for as long as the cluster
/// lives.
pub fn step_aside(cluster: &Arc<Cluster>) {
    let weak: Weak<Cluster> = Arc::downgrade(cluster);
    cluster.runtime.spawn(async move {
        let mut tick = interval(EVERY);
        // The apps this loop drained, which it alone undrains: a drain a
        // person asked for stays.
        let mut drained: Vec<Arc<Replicator>> = Vec::new();
        loop {
            tick.tick().await;
            let Some(cluster) = weak.upgrade() else {
                return;
            };
            let apps = replicators(&cluster);
            let peers: Vec<usize> = apps.iter().map(|r| r.peers_ready()).collect();
            match step(reservation().holds(), &peers, !drained.is_empty()) {
                Step::Drain => {
                    for r in apps.into_iter().filter(|r| !r.drained()) {
                        let handed = r.drain().await;
                        tracing::warn!(
                            claims = handed.claims,
                            "reserved: drained, every app has another member ready"
                        );
                        drained.push(r);
                    }
                }
                Step::Undrain => {
                    for r in drained.drain(..) {
                        match r.undrain() {
                            Ok(()) => tracing::warn!(
                                "the reservation's drain undone: catching up, then taking work \
                                 again"
                            ),
                            Err(e) => tracing::error!("the reservation's drain stays: {e}"),
                        }
                    }
                }
                Step::Stay => {}
            }
        }
    });
}
