//! The MCP sessions' bookkeeping, off the request path: every
//! [`TEND_EVERY`], the uses calls noted in memory are written into their
//! sessions' records and the stale are swept ([`RecordStore::tend`]), so a
//! call itself only reads its session.

use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::runtime::Handle;

use super::state::RecordStore;
use super::workers::now_ms;
use crate::session::ZendSession;

/// How often the sessions are tended. A use is written at most this much
/// later than an hour after the last, against a month before a session is
/// stale.
pub const TEND_EVERY: Duration = Duration::from_secs(10 * 60);

/// Tend the sessions every [`TEND_EVERY`] for as long as the session lives.
pub async fn tending(session: Weak<ZendSession>) {
    let mut every = tokio::time::interval(TEND_EVERY);
    loop {
        every.tick().await;
        let Some(session) = session.upgrade() else {
            return;
        };
        // Records are written through the persistence owner, whose flow
        // control can wait for room: off the runtime's threads.
        let tended = tokio::task::spawn_blocking(move || {
            let records = session.shared_substrate()?;
            let store = RecordStore::new(records, Arc::clone(session.mcp().used()));
            Some(store.tend(now_ms()))
        })
        .await;
        match tended {
            Ok(Some((0, 0))) | Ok(None) => {}
            Ok(Some((touched, ceased))) => {
                tracing::info!(touched, ceased, "mcp: tended the sessions")
            }
            Err(e) => tracing::warn!("mcp: tending the sessions failed: {e}"),
        }
    }
}

/// Start [`tending`] on `runtime` for `session`.
pub fn start_tending(session: &Arc<ZendSession>, runtime: &Handle) {
    runtime.spawn(tending(Arc::downgrade(session)));
}
