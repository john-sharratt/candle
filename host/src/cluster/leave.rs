//! A member leaving its clusters before the process stops
//! (`docs/replicated_substrate.md` §12.4): every app drains — its claims and
//! leases handed back, the gateways routing it nothing new — and the process
//! waits, at most [`LEAVE_WAIT`], for what each app's tenant has written to be
//! on disk and sent. The HTTP server then finishes the requests in flight, the
//! sockets close with `1001`, and the tenants run their own shutdown.

use std::sync::Arc;
use std::time::Duration;

use cluster::replicate::Replicator;
use tokio::task::spawn_blocking;
use tokio::time::{sleep, Instant};

use super::Cluster;
use crate::stopping::Stopping;

/// The longest a member waits for its own events to leave before it closes.
pub const LEAVE_WAIT: Duration = Duration::from_secs(60);

/// How often the wait looks again.
const LOOK: Duration = Duration::from_millis(100);

/// Every app's replication.
fn replicators(cluster: &Cluster) -> Vec<Arc<Replicator>> {
    cluster
        .links()
        .filter_map(|l| l.member()?.replicator().cloned())
        .collect()
}

/// Own events neither receipted nor sent, over every app.
fn owed(replicators: &[Arc<Replicator>]) -> usize {
    replicators
        .iter()
        .map(|r| {
            let v = r.view();
            v.unsent + v.in_flight
        })
        .sum()
}

/// Drain every app, then wait for its writes to be committed and its own
/// events to leave. Returns how many were still owed when it stopped waiting.
pub async fn drain_and_send(cluster: &Cluster, stopping: &Stopping) -> usize {
    let replicators = replicators(cluster);
    stopping.set_step("Draining", "handing back claims and leases");
    let mut claims = 0;
    for r in &replicators {
        claims += r.drain().await.claims;
    }
    tracing::info!(claims, "drained: routing nothing new to this member");

    stopping.set_step("Flushing", "committing the log");
    for r in &replicators {
        let store = r.store().clone();
        let _ = spawn_blocking(move || store.wait_written(store.ticket())).await;
    }

    let deadline = Instant::now() + LEAVE_WAIT;
    loop {
        let left = owed(&replicators);
        if left == 0 {
            return 0;
        }
        if Instant::now() >= deadline {
            tracing::warn!(
                left,
                "own events still unsent after {} s: they go out when this member next starts",
                LEAVE_WAIT.as_secs()
            );
            return left;
        }
        stopping.set_step(
            "Sending",
            format!("{left} own events still to reach the group"),
        );
        sleep(LOOK).await;
    }
}
