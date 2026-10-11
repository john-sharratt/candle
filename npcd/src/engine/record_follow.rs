//! The mind folder kept at the record while the daemon runs
//! (`docs/replicated_substrate.md` §10.3).
//!
//! A publish on this member brings the mind folder up at once
//! ([`crate::engine::archive`], [`crate::drafts::routes`]). One on another
//! member moves only `origin`, so every member fetches the record on a fixed
//! interval and checks it out; the watcher then ingests whatever changed, as it
//! does any edit to the mind, and the authored registries are read again.

use std::sync::Arc;
use std::time::Duration;

use tokio::task::spawn_blocking;

use crate::engine::runtime::Runtime;
use crate::opstore::merge::follow::Followed;

/// How often the record is fetched.
const EVERY: Duration = Duration::from_secs(60);

/// Keep the mind folder at the record until the daemon stops.
pub fn spawn(rt: Arc<Runtime>) {
    if rt.ops.mind_dir().is_none() {
        return;
    }
    tokio::spawn(async move {
        // What was said last, so an origin that stays down is said once.
        let mut said = String::new();
        loop {
            tokio::time::sleep(EVERY).await;
            if rt.stopping() {
                return;
            }
            let stores = Arc::clone(&rt.ops);
            let line = match spawn_blocking(move || stores.follow()).await {
                Ok(Some(Ok(Some(Followed::Moved { to })))) => {
                    tracing::info!(%to, "mind: brought up to the record");
                    String::new()
                }
                Ok(Some(Ok(Some(Followed::Dirty(paths))))) => format!(
                    "mind: changes of its own on disk ({}), so it was not brought up to the \
                     record",
                    paths.join(", ")
                ),
                Ok(Some(Ok(Some(Followed::Diverged)))) => {
                    "mind: its branch holds commits the record does not, so it was not brought up"
                        .to_string()
                }
                Ok(Some(Err(e))) => format!("mind: the record could not be fetched: {e}"),
                Ok(_) => String::new(),
                Err(e) => format!("mind: following the record failed: {e}"),
            };
            if !line.is_empty() && line != said {
                tracing::warn!("{line}");
            }
            said = line;
        }
    });
}
