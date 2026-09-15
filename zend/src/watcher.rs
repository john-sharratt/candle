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
//! on per-directory content-hash equality.
//!
//! Events that can't move either hash record (file access, pure
//! permission / xattr changes) are filtered out before they reach
//! the debounce window so the burst counter stays meaningful.

use std::path::{Component, Path, PathBuf};
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

/// `--max-depth` for the watcher. An event counts toward the source refresh
/// only when its path lies within `max` path components of a bounded layer's
/// content root — exactly what the `repo_map` / `code_reading` walks read — or
/// anywhere under an unbounded layer's root (a raw layer's folder, which the
/// bound does not cover). Anything else is frozen or outside every walk, so its
/// events are dropped instead of buying a re-walk that cannot change anything.
#[derive(Debug, Clone)]
pub struct WatchDepth {
    max: usize,
    bounded: Vec<PathBuf>,
    unbounded: Vec<PathBuf>,
}

impl WatchDepth {
    pub fn new(
        max: usize,
        bounded: impl IntoIterator<Item = PathBuf>,
        unbounded: impl IntoIterator<Item = PathBuf>,
    ) -> Self {
        Self {
            max,
            bounded: bounded.into_iter().map(|r| without_cur_dir(&r)).collect(),
            unbounded: unbounded.into_iter().map(|r| without_cur_dir(&r)).collect(),
        }
    }

    /// Whether an event at `path` can move anything a walk reads.
    fn admits(&self, path: &Path) -> bool {
        self.unbounded.iter().any(|root| path.starts_with(root))
            || self.bounded.iter().any(|root| {
                path.strip_prefix(root).is_ok_and(|rel| {
                    rel.components()
                        .filter(|c| matches!(c, Component::Normal(_)))
                        .count()
                        <= self.max
                })
            })
    }
}

/// `root` with its `.` components dropped: a layer's content root defaults to
/// `workspace.join(".")`, which no event path spells, so a prefix match against
/// it as written would admit nothing.
fn without_cur_dir(root: &Path) -> PathBuf {
    root.components()
        .filter(|c| !matches!(c, Component::CurDir))
        .collect()
}

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
///
/// `depth` is the `--max-depth` filter ([`WatchDepth`]); `None` lets every
/// non-ignored source path through.
pub fn spawn(
    workspace: &Path,
    depth: Option<WatchDepth>,
    on_refresh: Arc<dyn Fn() + Send + Sync + 'static>,
    on_uploads_changed: Arc<dyn Fn() + Send + Sync + 'static>,
) -> anyhow::Result<RecommendedWatcher> {
    let (src_tx, src_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let (up_tx, up_rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let root = workspace.to_path_buf();
    let max_depth = depth.as_ref().map(|d| d.max);
    let mut watcher = notify::recommended_watcher(move |res: NotifyResult<Event>| {
        let Ok(event) = res else {
            return;
        };
        if !is_refresh_relevant(&event.kind) {
            return;
        }
        // Classify the burst's paths into two disjoint drivers:
        //
        //  * **source** changes (anything not ignored, not under the top-level
        //    `uploads/` dir) drive the repo-map / code-reading refresh.
        //  * **uploads** changes drive ONLY the cheap upload-deletion reconcile
        //    — never the refresh. A watcher-driven re-ingest would race the
        //    upload endpoint's measured read_file stage (making it cache-hit
        //    "instant, 0 tokens"); but an upload *deletion* still has to retire
        //    the file's substrate conversation, which the reconcile does by
        //    tombstoning-if-absent (a no-op for still-present files, so create /
        //    modify events during an in-flight upload are harmless).
        //
        // The daemon's own `.substrate/` redo-log writes and build/VCS churn are
        // ignored entirely (see `is_ignored_path`), so they drive neither.
        let Signal { source, uploads } = classify(&event.paths, &root, depth.as_ref());
        if source {
            let _ = src_tx.send(());
        }
        if uploads {
            let _ = up_tx.send(());
        }
    })?;
    watcher.watch(workspace, RecursiveMode::Recursive)?;
    tracing::info!(workspace = %workspace.display(), max_depth = ?max_depth, "repo-map watcher armed");

    spawn_debounced(src_rx, on_refresh);
    spawn_debounced(up_rx, on_uploads_changed);
    Ok(watcher)
}

/// Fire `cb` once per debounced burst drained from `rx`: block for the first
/// event, then extend the deadline on each follow-up (typical editor saves and
/// bulk git operations emit their whole burst inside `DEBOUNCE_WINDOW`), capped
/// by `MAX_DEBOUNCE_HOLD` from the first event so a long-running bulk operation
/// can't defer the callback forever. Runs detached on the global executor; `cb`
/// itself runs on the blocking pool since both the source refresh and the
/// uploads reconcile do synchronous engine work.
fn spawn_debounced(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<()>,
    cb: Arc<dyn Fn() + Send + Sync + 'static>,
) {
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
            let cb = Arc::clone(&cb);
            tokio::task::spawn_blocking(move || cb()).await.ok();
        }
    });
}

/// Whether a path lives under a directory the watcher must ignore: the
/// daemon's own substrate store (self-trigger feedback), the build output, or
/// the VCS / dependency dirs. Matched on any path component so it catches the
/// dir itself and everything beneath it.
///
/// The top-level `uploads/` dir is handled separately by [`is_top_level_uploads`]
/// (a root-relative, first-component match) rather than here — an any-component
/// match would also suppress a legitimate nested `src/uploads/` source dir,
/// diverging from [`crate::repo_scan::walk_workspace`], which excludes only the
/// top-level dir.
fn is_ignored_path(path: &Path) -> bool {
    path.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some(".substrate") | Some(".git") | Some("target") | Some("node_modules")
        )
    })
}

/// Whether `path` is under the daemon's TOP-LEVEL `uploads/` dir (first
/// component of the workspace-relative path, case-insensitively — the win32 FS
/// is case-insensitive). Uploaded files are ingested (and measured) exclusively
/// by the upload endpoint; a watcher-driven background refresh would race it and
/// make the endpoint's measured read_file stage cache-hit ("instant, 0 tokens").
/// Matched precisely so a nested `src/uploads/` source dir keeps its watch —
/// mirroring `walk_workspace`'s exclusion so the two never disagree.
fn is_top_level_uploads(path: &Path, root: &Path) -> bool {
    path.strip_prefix(root)
        .ok()
        .and_then(|rel| rel.components().next())
        .and_then(|c| c.as_os_str().to_str())
        .is_some_and(|s| s.eq_ignore_ascii_case("uploads"))
}

/// Which debounced driver(s) a filesystem-event burst feeds. Disjoint per path,
/// but a single burst can touch both (an editor that saves a source file while
/// an upload lands), so both flags can be set at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Signal {
    /// Repo-map / code-reading refresh — any non-ignored path outside the
    /// top-level `uploads/` dir, or a path-less (unlocalisable) event.
    source: bool,
    /// Upload-deletion reconcile — a path under the top-level `uploads/` dir.
    uploads: bool,
}

/// Route a burst's `paths` (relative to workspace `root`) to the refresh signal,
/// the uploads-reconcile signal, or neither. Ignored paths (`.substrate/`,
/// `.git/`, `target/`, `node_modules/`) contribute to neither, and neither does a
/// source path the `--max-depth` filter (`depth`) does not admit. A path-less
/// event is backend-specific noise we can't localise — treated conservatively
/// as a source change so a real edit is never missed.
fn classify(paths: &[PathBuf], root: &Path, depth: Option<&WatchDepth>) -> Signal {
    let mut source = false;
    let mut uploads = false;
    for p in paths {
        if is_ignored_path(p) {
            continue;
        }
        if is_top_level_uploads(p, root) {
            uploads = true;
        } else if depth.is_none_or(|d| d.admits(p)) {
            source = true;
        }
    }
    Signal {
        source: source || paths.is_empty(),
        uploads,
    }
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

    #[test]
    fn ignores_daemon_managed_and_noise_dirs() {
        for p in [
            "ws/.substrate/substrate.log",
            "ws/.git/index",
            "ws/target/debug/x",
            "ws/node_modules/pkg/i.js",
        ] {
            assert!(is_ignored_path(Path::new(p)), "should ignore {p}");
        }
        // Ordinary source files still trigger refreshes — and `uploads` is NOT
        // matched by is_ignored_path (it's a root-relative check, below).
        for p in ["ws/src/main.rs", "ws/docs/readme.md", "ws/lib/x.py"] {
            assert!(!is_ignored_path(Path::new(p)), "should watch {p}");
        }
        assert!(!is_ignored_path(Path::new("ws/src/uploads/real.rs")));
    }

    fn paths(list: &[&str]) -> Vec<std::path::PathBuf> {
        list.iter().map(std::path::PathBuf::from).collect()
    }

    #[test]
    fn classify_routes_source_edits_to_refresh_only() {
        let root = Path::new("ws");
        let s = classify(&paths(&["ws/src/main.rs"]), root, None);
        assert_eq!(
            s,
            Signal {
                source: true,
                uploads: false
            }
        );
    }

    #[test]
    fn classify_routes_upload_events_to_reconcile_only() {
        // The whole point of the fix: an uploaded-file event no longer vanishes
        // — it drives the tombstone reconcile (and never the source refresh, so
        // it can't race the endpoint's measured read_file).
        let root = Path::new("ws");
        let s = classify(&paths(&["ws/uploads/notes.py"]), root, None);
        assert_eq!(
            s,
            Signal {
                source: false,
                uploads: true
            }
        );
        let nested = classify(&paths(&["ws/uploads/nested/a.rs"]), root, None);
        assert_eq!(
            nested,
            Signal {
                source: false,
                uploads: true
            }
        );
    }

    #[test]
    fn classify_mixed_burst_drives_both() {
        // An editor save and an upload landing in the same debounce window.
        let root = Path::new("ws");
        let s = classify(&paths(&["ws/src/main.rs", "ws/uploads/a.py"]), root, None);
        assert_eq!(
            s,
            Signal {
                source: true,
                uploads: true
            }
        );
    }

    #[test]
    fn classify_ignored_only_burst_drives_neither() {
        // `.substrate/` self-writes and build/VCS churn must move nothing.
        let root = Path::new("ws");
        let s = classify(
            &paths(&["ws/.substrate/substrate.log", "ws/target/debug/x"]),
            root,
            None,
        );
        assert_eq!(
            s,
            Signal {
                source: false,
                uploads: false
            }
        );
    }

    #[test]
    fn classify_pathless_event_is_conservatively_source() {
        let root = Path::new("ws");
        let s = classify(&[], root, None);
        assert_eq!(
            s,
            Signal {
                source: true,
                uploads: false
            }
        );
    }

    #[test]
    fn classify_nested_src_uploads_is_source_not_reconcile() {
        // A real project's nested `src/uploads/` is ordinary source, never the
        // endpoint-managed dir.
        let root = Path::new("ws");
        let s = classify(&paths(&["ws/src/uploads/real.rs"]), root, None);
        assert_eq!(
            s,
            Signal {
                source: true,
                uploads: false
            }
        );
    }

    /// `--max-depth 2` over the default content root (`ws/.`): the root's own
    /// files and one folder down drive a refresh; deeper paths are frozen, so
    /// their events drive nothing. Unbounded, everything counts.
    #[test]
    fn a_depth_bound_drops_events_past_it() {
        let root = Path::new("ws");
        let depth = WatchDepth::new(2, [PathBuf::from("ws/.")], Vec::new());
        let source = |list: &[&str], d: Option<&WatchDepth>| classify(&paths(list), root, d).source;
        assert!(source(&["ws/a.rs"], Some(&depth)));
        assert!(source(&["ws/src/b.rs"], Some(&depth)));
        assert!(!source(&["ws/src/deep/c.rs"], Some(&depth)));
        assert!(
            source(&["ws/src/deep/c.rs", "ws/a.rs"], Some(&depth)),
            "a burst with one shallow path still refreshes"
        );
        assert!(source(&["ws/src/deep/c.rs"], None));
        // Uploads are routed as before, whatever the bound.
        assert!(classify(&paths(&["ws/uploads/a/b/c.py"]), root, Some(&depth)).uploads);
    }

    /// The bound is measured from each layer's own root (`--ingest-dir
    /// code_reading=zend/src`); a path under no walked root is out of reach, and
    /// an unbounded (raw) layer's folder admits any depth.
    #[test]
    fn a_depth_bound_is_measured_from_each_layer_root() {
        let root = Path::new("ws");
        let depth = WatchDepth::new(
            1,
            [PathBuf::from("ws/zend/src")],
            [PathBuf::from("ws/responses")],
        );
        let source = |list: &[&str]| classify(&paths(list), root, Some(&depth)).source;
        assert!(source(&["ws/zend/src/main.rs"]));
        assert!(!source(&["ws/zend/src/api/mod.rs"]));
        assert!(!source(&["ws/README.md"]));
        assert!(source(&["ws/responses/greet/hello.chatml"]));
    }

    #[test]
    fn top_level_uploads_is_excluded_precisely() {
        let root = Path::new("ws");
        // The daemon's top-level uploads dir (any case) — endpoint-managed.
        assert!(is_top_level_uploads(Path::new("ws/uploads/notes.py"), root));
        assert!(is_top_level_uploads(
            Path::new("ws/uploads/nested/a.rs"),
            root
        ));
        assert!(is_top_level_uploads(Path::new("ws/Uploads/notes.py"), root));
        // A nested `src/uploads/` in a real project keeps its watch.
        assert!(!is_top_level_uploads(
            Path::new("ws/src/uploads/real.rs"),
            root
        ));
        assert!(!is_top_level_uploads(Path::new("ws/src/main.rs"), root));
        // Path outside the root → not matched (no strip_prefix).
        assert!(!is_top_level_uploads(Path::new("other/uploads/a"), root));
    }
}
