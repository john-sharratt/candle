//! The MCP sessions' bookkeeping, off the request path: every
//! [`TEND_EVERY`], the uses calls noted in memory are written into their
//! sessions' records and the stale are swept ([`RecordStore::tend`]), so a
//! call itself only reads its session; and every [`WRITE_EVERY`], the calls
//! noted for each conversation's activity are written into its journal
//! ([`write_noted`]).

use std::sync::{Arc, Weak};
use std::time::Duration;

use tokio::runtime::Handle;

use super::noted_calls::{write_noted, WRITE_EVERY};
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

/// Write the noted calls every [`WRITE_EVERY`] for as long as the session
/// lives.
pub async fn writing_calls(session: Weak<ZendSession>) {
    let mut every = tokio::time::interval(WRITE_EVERY);
    loop {
        every.tick().await;
        let Some(session) = session.upgrade() else {
            return;
        };
        if session.mcp().noted().waiting() == 0 {
            continue;
        }
        // Each conversation's calls are a commit to the journals' repository.
        let wrote = tokio::task::spawn_blocking(move || {
            write_noted(&session, session.mcp().noted())
        })
        .await;
        match wrote {
            Ok((written, dropped, again)) => tracing::info!(
                written,
                dropped,
                again,
                "journals: wrote the conversations' noted calls"
            ),
            Err(e) => tracing::warn!("journals: writing the noted calls failed: {e}"),
        }
    }
}

/// Start [`tending`] and [`writing_calls`] on `runtime` for `session`.
pub fn start_tending(session: &Arc<ZendSession>, runtime: &Handle) {
    runtime.spawn(tending(Arc::downgrade(session)));
    runtime.spawn(writing_calls(Arc::downgrade(session)));
}
