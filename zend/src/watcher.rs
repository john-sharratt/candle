//! `notify`-backed workspace watcher that drives the repo_map and
//! code_reading layer refresh paths.
//!
//! Filesystem events are noisy and bursty (an editor's "save" can fire
//! create / modify / rename in quick succession; bulk operations like
//! `git checkout` raise thousands of events at once).  The watcher
//! debounces events into a single "something changed" pulse on a
//! short window, then asks the session to re-walk and decide whether
//! a refresh is actually warranted — content edits short-circuit on
//! file-content hash equality, and repo-map refresh short-circuits
//! on per-cluster file-name hash equality.
//!
//! Events that can't move either hash record (file access, pure
//! permission / xattr changes) are filtered out before they reach
//! the debounce window so the burst counter stays meaningful.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use notify::event::{EventKind, ModifyKind};
use notify::{Event, RecommendedWatcher, RecursiveMode, Result as NotifyResult, Watcher};

/// Time we wait after the first relevant event before firing the
/// refresh.  Subsequent events inside the window extend the deadline
/// — typical editors emit a burst within a few hundred milliseconds
/// of a save; bulk git operations emit thousands inside a second.
pub const DEBOUNCE_WINDOW: Duration = Duration::from_millis(500);

/// Outermost ceiling on debounce extension — a long-running bulk
/// operation must not block the refresh forever.
pub const MAX_DEBOUNCE_HOLD: Duration = Duration::from_secs(5);

/// Spawn the watcher in the background.  Filesystem events that
/// can move either the repo-map or code-reading hash record
/// (create / remove / rename / content-modify) trigger a refresh on
/// the supplied callback after `DEBOUNCE_WINDOW` of quiet, bounded
/// by `MAX_DEBOUNCE_HOLD`.  Pure access events and metadata-only
/// modifies are dropped at the filter.
///
/// Returns the [`RecommendedWatcher`] so the caller can keep it
/// alive for the daemon's lifetime; dropping it stops the watch.
/// The dispatch task runs detached on the global executor.
pub fn spawn(
    workspace: &Path,
    on_refresh: Arc<dyn Fn() + Send + Sync + 'static>,
) -> anyhow::Result<RecommendedWatcher> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let mut watcher = notify::recommended_watcher(move |res: NotifyResult<Event>| {
        let Ok(event) = res else {
            return;
        };
        if !is_refresh_relevant(&event.kind) {
            return;
        }
        // Drop events confined to ignored directories. Critically this includes
        // the daemon's OWN `.substrate/` redo-log writes — without this filter,
        // persistence I/O continuously self-triggers a workspace refresh (a
        // feedback loop) — plus build/VCS churn that never affects source.
        if !event.paths.is_empty() && event.paths.iter().all(|p| is_ignored_path(p)) {
            return;
        }
        let _ = tx.send(());
    })?;
    watcher.watch(workspace, RecursiveMode::Recursive)?;
    tracing::info!(workspace = %workspace.display(), "repo-map watcher armed");

    tokio::spawn(async move {
        loop {
            // Block until the first event.
            if rx.recv().await.is_none() {
                break;
            }
            let start = tokio::time::Instant::now();
            // Drain the burst — extend the deadline on each event,
            // capped by MAX_DEBOUNCE_HOLD from the first one.
            loop {
                let elapsed = start.elapsed();
                if elapsed >= MAX_DEBOUNCE_HOLD {
                    break;
                }
                let remaining_cap = MAX_DEBOUNCE_HOLD - elapsed;
                let wait = DEBOUNCE_WINDOW.min(remaining_cap);
                match tokio::time::timeout(wait, rx.recv()).await {
                    Ok(Some(())) => continue,
                    _ => break,
                }
            }
            let cb = Arc::clone(&on_refresh);
            tokio::task::spawn_blocking(move || cb()).await.ok();
        }
    });
    Ok(watcher)
}

/// Whether a path lives under a directory the watcher must ignore: the
/// daemon's own substrate store (self-trigger feedback), the build output, or
/// the VCS / dependency dirs. Matched on any path component so it catches the
/// dir itself and everything beneath it.
fn is_ignored_path(path: &Path) -> bool {
    path.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some(".substrate") | Some(".git") | Some("target") | Some("node_modules")
        )
    })
}

fn is_refresh_relevant(kind: &EventKind) -> bool {
    match kind {
        // Create / Remove always matter — either the file-name set
        // or the file-content set just moved.
        EventKind::Create(_) | EventKind::Remove(_) => true,
        // Name changes (renames) move the file-name set.
        EventKind::Modify(ModifyKind::Name(_)) => true,
        // Content edits move the code-reading hash.
        EventKind::Modify(ModifyKind::Data(_)) => true,
        // Some backends report `Any` for save-style events; pass.
        EventKind::Modify(ModifyKind::Any) => true,
        // Pure metadata changes (perm bits, owners, xattrs) can't
        // move either hash.
        EventKind::Modify(ModifyKind::Metadata(_)) => false,
        // Unknown / backend-specific modify variant — be conservative
        // and pass.
        EventKind::Modify(ModifyKind::Other) => true,
        // Access events never matter.
        EventKind::Access(_) => false,
        // Other / unknown — be conservative and pass.
        EventKind::Other => true,
        EventKind::Any => true,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use notify::event::{
        AccessKind, AccessMode, CreateKind, DataChange, MetadataKind, RemoveKind, RenameMode,
    };

    #[test]
    fn filter_lets_creates_and_removes_through() {
        assert!(is_refresh_relevant(&EventKind::Create(CreateKind::File)));
        assert!(is_refresh_relevant(&EventKind::Remove(RemoveKind::File)));
        assert!(is_refresh_relevant(&EventKind::Create(CreateKind::Folder)));
    }

    #[test]
    fn filter_lets_renames_through() {
        assert!(is_refresh_relevant(&EventKind::Modify(ModifyKind::Name(
            RenameMode::To
        ))));
        assert!(is_refresh_relevant(&EventKind::Modify(ModifyKind::Name(
            RenameMode::From
        ))));
        assert!(is_refresh_relevant(&EventKind::Modify(ModifyKind::Name(
            RenameMode::Both
        ))));
    }

    #[test]
    fn filter_lets_content_edits_through() {
        // Code-reading hash is over file content — content-modify
        // events must reach the refresh path.
        assert!(is_refresh_relevant(&EventKind::Modify(ModifyKind::Data(
            DataChange::Content
        ))));
        assert!(is_refresh_relevant(&EventKind::Modify(ModifyKind::Data(
            DataChange::Size
        ))));
    }

    #[test]
    fn filter_drops_metadata_only_modifies() {
        // Pure permission / xattr changes can't move either hash.
        assert!(!is_refresh_relevant(&EventKind::Modify(
            ModifyKind::Metadata(MetadataKind::Permissions)
        )));
        assert!(!is_refresh_relevant(&EventKind::Modify(
            ModifyKind::Metadata(MetadataKind::Ownership)
        )));
    }

    #[test]
    fn filter_drops_access_events() {
        assert!(!is_refresh_relevant(&EventKind::Access(AccessKind::Open(
            AccessMode::Read
        ))));
        assert!(!is_refresh_relevant(&EventKind::Access(AccessKind::Read)));
    }
}
