//! The mind-directory watcher: an edited world file takes effect without a
//! restart.
//!
//! # What "reload" means here
//!
//! Not "re-read the file". The mind's files are *ingested into the substrate* as
//! turns — that is what makes a world's history and technology tree available to
//! the gather at all — so a changed file has to reconcile against turns that are
//! already there. Three cases, and each is a different write:
//!
//! - **Unchanged** — the content hash matches. Nothing is written. This is the
//!   overwhelmingly common case during an editing session, because a save
//!   touches one file in a directory of two hundred, and it has to cost nothing
//!   or the daemon spends its life re-ingesting.
//! - **Changed** — the old turn is **tombstoned** and the new content ingested
//!   as a fresh turn. Not overwritten: the substrate is an append-only redo log,
//!   and the previous version stays reachable. A tombstone retires a turn from
//!   selection; it does not delete it.
//! - **Deleted** — reported by the reload, and **not yet retired from the
//!   substrate**. The watcher sees the file go and the report counts it; the
//!   ingest side has no path that tombstones the turn it wrote, so a deleted
//!   document's content stays selectable by the gather. Stated here rather than
//!   implied by an unused affordance: the method that listed a ledger's
//!   no-longer-present files had no caller, which made the gap read as an
//!   oversight in the wiring instead of as work that has not been done.
//!
//! # Why hashing, and why per file
//!
//! Filesystem events are unreliable in both directions: an editor's single save
//! can fire create, modify and rename in quick succession, and a bulk operation
//! (`git checkout`, a directory copy) can raise thousands at once. Debouncing
//! collapses the burst into one pulse; hashing decides whether that pulse means
//! anything. Without the hash, every save of any file in the mind would
//! re-ingest the whole directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use notify::event::{EventKind, ModifyKind};
use notify::{Event, RecommendedWatcher, RecursiveMode, Result as NotifyResult, Watcher};
use sha2::{Digest, Sha256};

/// Quiet time after the first relevant event before a reload fires. Subsequent
/// events inside the window extend it — an editor's burst lands well inside.
pub const DEBOUNCE_WINDOW: Duration = Duration::from_millis(400);

/// Ceiling on that extension, so a long bulk operation cannot defer the reload
/// for ever.
pub const MAX_DEBOUNCE_HOLD: Duration = Duration::from_secs(5);

/// What a mind file's content hashed to, when it was last ingested.
///
/// # Why this is on disk
///
/// Because ingesting a document is a real prefill — measured at about four
/// seconds each on a 3090 — and a mind with four hundred of them is half an hour
/// of startup. In memory only, that half hour was paid **on every boot**, for
/// work already in the substrate.
///
/// It lives *inside* `.substrate/`, deliberately: the ledger's claim is "this
/// document is already a turn in there", so wiping the substrate has to wipe the
/// ledger with it. Beside the substrate rather than inside, a `rm -rf
/// .substrate` would leave a ledger asserting turns that no longer exist, and
/// the next boot would skip every document and stand up a cast that knows
/// nothing — with a green loading screen.
#[derive(Debug, Default)]
pub struct Ledger {
    hashes: Mutex<HashMap<PathBuf, String>>,
    /// What the filesystem watcher has already reported, which is a different
    /// fact from what has been ingested and therefore a different map.
    ///
    /// `hashes` means *this document is a turn in the substrate* and is written
    /// only by the ingest seal. `seen` means *the watcher has already announced
    /// this content* and is written only by [`reload`]. Sharing one map made
    /// each corrupt the other: a watcher event during the multi-minute startup
    /// load hashed the whole mind tree before `ingest::pending` reached it, so
    /// every document then read `Unchanged`, nothing was written, and `flush`
    /// persisted that across restarts — the mind never ingesting again on any
    /// boot until the files were edited.
    ///
    /// Not persisted: a run's first reload should describe the tree as it finds
    /// it, and a `seen` carried across restarts would suppress the report of a
    /// file edited while the daemon was down.
    seen: Mutex<HashMap<PathBuf, String>>,
    /// Where to persist. `None` keeps the ledger in memory — the shape the
    /// tests use, and the honest state for a daemon with no data directory.
    path: Option<PathBuf>,
}

/// What reconciling one file decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reconcile {
    /// Content hash matched. Nothing written.
    Unchanged,
    /// First time this file has been seen. Ingest it.
    Added,
    /// Content moved. Tombstone the old turn, ingest the new content.
    Changed,
    /// The file is gone. Tombstone with no replacement.
    Removed,
}

/// The ledger's file, inside the substrate directory it describes.
pub fn ledger_path(data: &Path) -> PathBuf {
    data.join(".substrate").join("ingest-ledger.json")
}

impl Ledger {
    /// An in-memory ledger that persists nothing.
    ///
    /// The tests' shape. Production always goes through [`Ledger::open`], which
    /// is what makes a restart cheap — an in-memory ledger re-ingests the whole
    /// mind on every boot, and at four seconds a document that is the difference
    /// between a twelve-second start and half an hour of one.
    #[cfg(test)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open the ledger for a data directory, reading what is already there.
    ///
    /// A missing or unreadable file is an empty ledger, not an error: the
    /// consequence is re-ingesting, which is slow and correct. The opposite
    /// failure — trusting a ledger that does not match the substrate — is fast
    /// and wrong, so every ambiguity resolves towards doing the work again.
    pub fn open(data: &Path) -> Self {
        let path = ledger_path(data);
        let hashes = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<HashMap<PathBuf, String>>(&t).ok())
            .unwrap_or_default();
        if !hashes.is_empty() {
            tracing::info!(
                "ingest ledger: {} document(s) already in the substrate",
                hashes.len()
            );
        }
        Self {
            hashes: Mutex::new(hashes),
            seen: Mutex::new(HashMap::new()),
            path: Some(path),
        }
    }

    /// Write the ledger out.
    ///
    /// Called after each layer rather than after each document: a crash mid-layer
    /// then re-ingests that layer, which is correct and costs one layer rather
    /// than the whole mind. Per-document would be four hundred writes to save
    /// nothing that matters.
    ///
    /// Written to a temporary file and renamed, so a crash during the write
    /// cannot leave a half-parsed ledger — which would read as an empty one and
    /// silently re-ingest everything.
    pub fn flush(&self) {
        let Some(path) = &self.path else { return };
        let snapshot = { self.hashes.lock().unwrap().clone() };
        let Ok(json) = serde_json::to_string(&snapshot) else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, json).is_ok() {
            if let Err(e) = std::fs::rename(&tmp, path) {
                tracing::warn!("ingest ledger not saved: {e} — the next boot re-ingests");
                let _ = std::fs::remove_file(&tmp);
            }
        }
    }

    /// The verdict [`Self::reconcile`] would give, **without recording it**.
    ///
    /// The ledger's entry is a claim that a document is a turn in the substrate, so writing one
    /// is only ever correct next to the write that puts it there. Anything that merely wants to
    /// *know* what a document still owes — planning a layer's turns, a console listing, a dry
    /// run — asks this instead, and the seal calls [`Self::reconcile`] afterwards.
    ///
    /// Splitting the two is not tidiness. `ingest::pending` used `reconcile`, so merely asking
    /// what a layer owed recorded every document as already ingested: a document whose turn
    /// then failed was skipped on every later boot, unrecoverable without editing the file.
    pub fn inspect(&self, path: &Path, content: Option<&str>) -> Reconcile {
        let h = self.hashes.lock().unwrap();
        match content {
            None => {
                if h.contains_key(path) {
                    Reconcile::Removed
                } else {
                    Reconcile::Unchanged
                }
            }
            Some(text) => {
                let digest = hash(text);
                match h.get(path) {
                    None => Reconcile::Added,
                    Some(prev) if *prev == digest => Reconcile::Unchanged,
                    Some(_) => Reconcile::Changed,
                }
            }
        }
    }

    /// What moved since the watcher last looked, recorded against [`Self::seen`].
    ///
    /// The reload report's counterpart to `reconcile`: same arithmetic, different map, and it
    /// never touches the ingest claim. That is what lets an editor event land during the
    /// startup load without convincing the ledger that the tree is already in the substrate.
    ///
    /// The two maps also answer over different sets. `reload` walks the whole mind directory —
    /// `world.yaml`, the registries, everything — while only layer documents ever become turns,
    /// so a shared map would leave every non-document file reading `Added` forever and report
    /// the entire tree as new on every save.
    fn observe(&self, path: &Path, content: Option<&str>) -> Reconcile {
        let mut s = self.seen.lock().unwrap();
        match content {
            None => {
                if s.remove(path).is_some() {
                    Reconcile::Removed
                } else {
                    Reconcile::Unchanged
                }
            }
            Some(text) => {
                let digest = hash(text);
                match s.insert(path.to_path_buf(), digest.clone()) {
                    None => Reconcile::Added,
                    Some(prev) if prev == digest => Reconcile::Unchanged,
                    Some(_) => Reconcile::Changed,
                }
            }
        }
    }

    /// Record that a document is now a turn in the substrate.
    ///
    /// Only correct next to the write that puts it there — see [`Self::inspect`] for the
    /// question that merely wants the verdict.
    pub fn reconcile(&self, path: &Path, content: Option<&str>) -> Reconcile {
        let mut h = self.hashes.lock().unwrap();
        match content {
            None => {
                if h.remove(path).is_some() {
                    Reconcile::Removed
                } else {
                    // Never ingested and now absent — a temp file an editor
                    // created and deleted inside one debounce window. Nothing
                    // to retire.
                    Reconcile::Unchanged
                }
            }
            Some(text) => {
                let digest = hash(text);
                match h.insert(path.to_path_buf(), digest.clone()) {
                    None => Reconcile::Added,
                    Some(prev) if prev == digest => Reconcile::Unchanged,
                    Some(_) => Reconcile::Changed,
                }
            }
        }
    }

    /// Files the *watcher* has reported that are no longer on disk.
    ///
    /// `observe`'s counterpart, over [`Self::seen`] and for the same reason: a deletion the
    /// reload report has already announced is not a deletion the substrate has retired, and
    /// asking the ingest map here would report every non-document file's removal on the first
    /// save after it went.
    fn unseen(&self, present: &[PathBuf]) -> Vec<PathBuf> {
        let s = self.seen.lock().unwrap();
        s.keys().filter(|p| !present.contains(p)).cloned().collect()
    }
}

/// The content hash. Sha-256, hex, truncated — this identifies a version, it is
/// not a security boundary, and sixteen hex characters is far past the point
/// where a collision within one mind directory is credible.
fn hash(content: &str) -> String {
    // Line endings are normalised first. A mind directory edited on Windows and
    // on Linux would otherwise re-ingest wholesale on every platform change,
    // and the *content* is identical — which is the thing the hash is for.
    let normalised = content.replace("\r\n", "\n");
    let mut hasher = Sha256::new();
    hasher.update(normalised.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Files in the mind that carry ingestible content.
///
/// Authored content is YAML and Markdown. Everything else in a mind directory —
/// a stray image, an editor's swap file, a `.git` — is not content and must not
/// become a turn.
pub fn is_ingestible(path: &Path) -> bool {
    if path
        .components()
        .any(|c| matches!(c.as_os_str().to_str(), Some(".git") | Some(".substrate")))
    {
        return false;
    }
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("yaml") | Some("yml") | Some("md")
    )
}

/// Walk the mind directory for ingestible files.
pub fn walk(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            // An unreadable subdirectory is not a reason to fail the whole
            // walk — a mind with one permission-denied folder should still
            // ingest the rest, and the log says which one was skipped.
            Err(e) => {
                tracing::warn!("mind: skipping {}: {e}", dir.display());
                continue;
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if is_ingestible_dir(&path) {
                    stack.push(path);
                }
            } else if is_ingestible(&path) {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

fn is_ingestible_dir(path: &Path) -> bool {
    !matches!(
        path.file_name().and_then(|n| n.to_str()),
        Some(".git") | Some(".substrate") | Some("node_modules") | Some("target")
    )
}

/* There is no `ingest_all` here any more, and its absence is the point.
 *
 * It walked the mind, moved the progress bar and wrote nothing — under a load
 * phase named "Ingesting mind layers". A world with sixty-six documents in
 * `layers/world/` booted in a second and no character could reach any of them,
 * with a full progress bar saying otherwise. The write lives in
 * `engine::ingest` now, which puts each document on its layer's conversation as
 * a turn; this module keeps the part it was always good at, which is deciding
 * what actually moved. */

/// What one reload pass did, for the log and the Pulse view.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReloadReport {
    pub added: usize,
    pub changed: usize,
    pub removed: usize,
    pub unchanged: usize,
}

impl ReloadReport {
    /// Whether anything actually moved. The common case during an editing
    /// session is that nothing did.
    pub fn is_noop(&self) -> bool {
        self.added == 0 && self.changed == 0 && self.removed == 0
    }
}

/// Diff the mind directory against the ledger.
pub fn reload(root: &Path, ledger: &Ledger) -> std::io::Result<ReloadReport> {
    let files = walk(root)?;
    let mut report = ReloadReport::default();

    for path in &files {
        let content = std::fs::read_to_string(path).ok();
        // `observe`, not `reconcile`: this records what the watcher has *reported*, which is a
        // different claim from what has been ingested and lives in a different map. See
        // `Ledger::seen` for what conflating the two cost.
        match ledger.observe(path, content.as_deref()) {
            Reconcile::Added => report.added += 1,
            Reconcile::Changed => report.changed += 1,
            Reconcile::Removed => report.removed += 1,
            Reconcile::Unchanged => report.unchanged += 1,
        }
    }

    // Files the ledger holds that are no longer present. Walked separately
    // because the loop above only visits what exists.
    for gone in ledger.unseen(&files) {
        if ledger.observe(&gone, None) == Reconcile::Removed {
            report.removed += 1;
        }
    }

    Ok(report)
}

/// Whether a filesystem event could possibly have moved a content hash.
///
/// Access events and metadata-only modifies cannot, so they are dropped before
/// the debounce window rather than after — otherwise a directory being read
/// keeps extending the hold and the reload never fires.
fn is_relevant(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_)
            | EventKind::Remove(_)
            | EventKind::Modify(ModifyKind::Data(_))
            | EventKind::Modify(ModifyKind::Name(_))
            // `Any` and `Other` are what several platforms report for an
            // ordinary save, so they have to be taken as possibly-relevant. The
            // content hash is what decides afterwards, which is why being
            // generous here costs nothing.
            | EventKind::Modify(ModifyKind::Any)
            | EventKind::Modify(ModifyKind::Other)
    )
}

/// Arm the watcher. The returned handle must be held for the daemon's lifetime;
/// dropping it stops the watch.
///
/// `ledger` is the **runtime's own**, not a fresh one. The startup ingest fills
/// it as it writes each document, so the watcher starts knowing exactly what is
/// already in the substrate — and the first edit after a boot is one changed
/// file rather than a whole-tree rewrite. A private ledger here would have to
/// re-hash the mind to catch up, and would disagree with the ingest about what
/// had been written the moment either one raced the other.
pub fn spawn(
    root: &Path,
    ledger: Arc<Ledger>,
    on_reload: Arc<dyn Fn(ReloadReport) + Send + Sync + 'static>,
) -> anyhow::Result<RecommendedWatcher> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<()>();
    let mut watcher = notify::recommended_watcher(move |res: NotifyResult<Event>| {
        let Ok(event) = res else { return };
        if !is_relevant(&event.kind) {
            return;
        }
        if event.paths.iter().any(|p| is_ingestible(p)) {
            let _ = tx.send(());
        }
    })?;
    watcher.watch(root, RecursiveMode::Recursive)?;
    tracing::info!("mind watcher armed at {}", root.display());

    let root = root.to_path_buf();

    tokio::spawn(async move {
        while rx.recv().await.is_some() {
            // Collapse the burst: extend on each follow-up, capped so a bulk
            // operation cannot defer the reload for ever.
            let deadline = tokio::time::Instant::now() + MAX_DEBOUNCE_HOLD;
            loop {
                let quiet = tokio::time::timeout_at(
                    deadline.min(tokio::time::Instant::now() + DEBOUNCE_WINDOW),
                    rx.recv(),
                );
                match quiet.await {
                    // Another event inside the window — keep waiting.
                    Ok(Some(())) if tokio::time::Instant::now() < deadline => continue,
                    _ => break,
                }
            }
            let root = root.clone();
            let ledger = Arc::clone(&ledger);
            let cb = Arc::clone(&on_reload);
            // The reconcile is synchronous file work; off the async pool.
            let _ = tokio::task::spawn_blocking(move || match reload(&root, &ledger) {
                Ok(report) => {
                    if report.is_noop() {
                        tracing::debug!("mind: nothing moved ({} unchanged)", report.unchanged);
                    } else {
                        tracing::info!(
                            "mind reload: +{} ~{} -{} ({} unchanged)",
                            report.added,
                            report.changed,
                            report.removed,
                            report.unchanged
                        );
                        cb(report);
                    }
                }
                Err(e) => tracing::warn!("mind reload failed: {e}"),
            })
            .await;
        }
    });

    Ok(watcher)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "npcd-watch-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn only_authored_content_is_ingestible() {
        assert!(is_ingestible(Path::new("world.yaml")));
        assert!(is_ingestible(Path::new("a/b/history.md")));
        assert!(is_ingestible(Path::new("x.yml")));
        for no in ["portrait.png", "notes.txt", "a.yaml.swp", "Cargo.toml"] {
            assert!(!is_ingestible(Path::new(no)), "{no} should not be ingested");
        }
    }

    /// The daemon's own redo log lives under the data directory, and a mind
    /// pointed at the same root would otherwise ingest the substrate into
    /// itself — which grows without bound and is very confusing to watch.
    #[test]
    fn the_substrate_and_git_are_never_ingested() {
        assert!(!is_ingestible(Path::new(".substrate/seg-1.log")));
        assert!(!is_ingestible(Path::new("mind/.git/config.yaml")));
        assert!(!is_ingestible(Path::new(".substrate/x.yaml")));
    }

    /// The common case during an editing session is that nothing moved, and it
    /// has to cost nothing.
    #[test]
    fn an_unchanged_file_reconciles_to_nothing() {
        let l = Ledger::new();
        let p = Path::new("world.yaml");
        assert_eq!(l.reconcile(p, Some("a: 1")), Reconcile::Added);
        assert_eq!(l.reconcile(p, Some("a: 1")), Reconcile::Unchanged);
        assert_eq!(l.reconcile(p, Some("a: 2")), Reconcile::Changed);
    }

    /// A mind edited on two platforms must not re-ingest wholesale just because
    /// the line endings changed — the content is what the hash is for.
    #[test]
    fn line_endings_do_not_count_as_a_change() {
        let l = Ledger::new();
        let p = Path::new("history.md");
        l.reconcile(p, Some("one\ntwo\n"));
        assert_eq!(l.reconcile(p, Some("one\r\ntwo\r\n")), Reconcile::Unchanged);
    }

    #[test]
    fn a_deleted_file_is_removed_once() {
        let l = Ledger::new();
        let p = Path::new("gone.yaml");
        l.reconcile(p, Some("x: 1"));
        assert_eq!(l.reconcile(p, None), Reconcile::Removed);
        // Already retired — a second pass must not tombstone it again.
        assert_eq!(l.reconcile(p, None), Reconcile::Unchanged);
    }

    /// An editor creating and deleting a temp file inside one debounce window
    /// must not produce a tombstone for a turn that never existed.
    #[test]
    fn a_file_never_seen_and_now_absent_is_not_a_removal() {
        let l = Ledger::new();
        assert_eq!(
            l.reconcile(Path::new("never.yaml"), None),
            Reconcile::Unchanged
        );
        // And it left nothing behind: a second pass must not then see it as a
        // file it once knew.
        assert_eq!(
            l.reconcile(Path::new("never.yaml"), None),
            Reconcile::Unchanged
        );
    }

    #[test]
    fn the_walk_finds_content_and_skips_the_rest() {
        let root = tmp();
        std::fs::write(root.join("world.yaml"), "a: 1").unwrap();
        std::fs::write(root.join("notes.txt"), "ignored").unwrap();
        std::fs::create_dir_all(root.join("layers")).unwrap();
        std::fs::write(root.join("layers/history.md"), "# History").unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/config.yaml"), "x: 1").unwrap();

        let found = walk(&root).unwrap();
        let names: Vec<String> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"world.yaml".to_string()));
        assert!(names.contains(&"history.md".to_string()));
        assert!(!names.contains(&"notes.txt".to_string()));
        assert!(
            !names.contains(&"config.yaml".to_string()),
            "walked into .git"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The end-to-end reconcile: a real directory, edited.
    #[test]
    fn a_reload_reports_exactly_what_moved() {
        let root = tmp();
        std::fs::write(root.join("a.yaml"), "one").unwrap();
        std::fs::write(root.join("b.yaml"), "two").unwrap();
        let l = Ledger::new();

        let first = reload(&root, &l).unwrap();
        assert_eq!(
            first,
            ReloadReport {
                added: 2,
                changed: 0,
                removed: 0,
                unchanged: 0
            }
        );

        // Nothing touched.
        let second = reload(&root, &l).unwrap();
        assert!(
            second.is_noop(),
            "an untouched mind reported work: {second:?}"
        );
        assert_eq!(second.unchanged, 2);

        // One edited, one deleted, one new.
        std::fs::write(root.join("a.yaml"), "one, revised").unwrap();
        std::fs::remove_file(root.join("b.yaml")).unwrap();
        std::fs::write(root.join("c.yaml"), "three").unwrap();
        let third = reload(&root, &l).unwrap();
        assert_eq!(
            third,
            ReloadReport {
                added: 1,
                changed: 1,
                removed: 1,
                unchanged: 0
            }
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// **A reload does not claim anything was ingested.**
    ///
    /// The two maps answer different questions, and this is the failure that proved it. An
    /// editor event during the multi-minute startup load ran a reload over the whole mind
    /// tree; when `reload` recorded into the ingest map, `ingest::pending` then arrived to
    /// find every document `Unchanged`, wrote nothing, and `flush` persisted that. The mind
    /// never ingested again, on any boot, until the files were edited — and the loading
    /// screen went green over a cast that knew nothing.
    #[test]
    fn a_reload_leaves_the_ingest_claim_alone() {
        let root = tmp();
        let doc = root.join("a.yaml");
        std::fs::write(&doc, "one").unwrap();
        let l = Ledger::new();

        // The watcher gets there first, as it does during a slow startup.
        assert_eq!(reload(&root, &l).unwrap().added, 1);

        // Ingest still owes the document.
        assert_eq!(
            l.inspect(&doc, Some("one")),
            Reconcile::Added,
            "the reload marked the document ingested — nothing had ingested it"
        );

        // And once it has genuinely been sealed, it stops owing it.
        l.reconcile(&doc, Some("one"));
        assert_eq!(l.inspect(&doc, Some("one")), Reconcile::Unchanged);

        // The reload's own record is likewise untouched by the seal: it already reported this
        // content, so the next walk is a no-op rather than a second announcement.
        assert!(reload(&root, &l).unwrap().is_noop());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Only layer documents ever become turns, but a reload walks the whole mind — `world.yaml`,
    /// the registries, everything. Reading those against the ingest map would leave every one of
    /// them `Added` forever, so each save would report the entire tree as new.
    #[test]
    fn files_that_never_become_turns_still_settle() {
        let root = tmp();
        std::fs::write(root.join("world.yaml"), "name: Ardh").unwrap();
        std::fs::write(root.join("registry.yaml"), "cast: []").unwrap();
        let l = Ledger::new();

        assert_eq!(reload(&root, &l).unwrap().added, 2);
        let second = reload(&root, &l).unwrap();
        assert!(
            second.is_noop(),
            "a file nothing ingests reported as new twice: {second:?}"
        );
        assert_eq!(second.unchanged, 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Access events cannot move a hash, and a directory being read would
    /// otherwise keep extending the debounce hold so the reload never fires.
    #[test]
    fn irrelevant_filesystem_events_are_dropped_before_the_debounce() {
        use notify::event::{AccessKind, DataChange, MetadataKind};
        assert!(!is_relevant(&EventKind::Access(AccessKind::Read)));
        assert!(!is_relevant(&EventKind::Modify(ModifyKind::Metadata(
            MetadataKind::Permissions
        ))));
        assert!(is_relevant(&EventKind::Modify(ModifyKind::Data(
            DataChange::Content
        ))));
        assert!(is_relevant(&EventKind::Create(
            notify::event::CreateKind::File
        )));
        assert!(is_relevant(&EventKind::Remove(
            notify::event::RemoveKind::File
        )));
    }

    /// The debounce must collapse an editor's burst but not defer a bulk
    /// operation for ever.
    #[test]
    fn the_debounce_window_is_shorter_than_its_ceiling() {
        assert!(DEBOUNCE_WINDOW < MAX_DEBOUNCE_HOLD);
    }

    /// **A restart must not re-ingest what is already in the substrate.**
    ///
    /// Each document is a real prefill — about four seconds — so a mind with
    /// four hundred of them is half an hour of startup, and in memory only that
    /// half hour was paid on every boot for work already done.
    #[test]
    fn a_saved_ledger_is_read_back_and_skips_what_it_holds() {
        let data = tmp();
        let mind = data.join("mind");
        std::fs::create_dir_all(&mind).unwrap();
        let doc = mind.join("world.md");
        std::fs::write(&doc, "# Alpha").unwrap();

        let first = Ledger::open(&data);
        assert_eq!(first.reconcile(&doc, Some("# Alpha")), Reconcile::Added);
        first.flush();

        // A second daemon, same data directory.
        let second = Ledger::open(&data);
        assert_eq!(
            second.reconcile(&doc, Some("# Alpha")),
            Reconcile::Unchanged,
            "a restart re-ingested a document already in the substrate"
        );
        // And an edit still registers across the restart.
        assert_eq!(
            second.reconcile(&doc, Some("# Alpha, revised")),
            Reconcile::Changed
        );
        let _ = std::fs::remove_dir_all(&data);
    }

    /// It lives *inside* `.substrate/`, so wiping the substrate wipes the
    /// ledger. Beside it, a `rm -rf .substrate` would leave a ledger asserting
    /// turns that no longer exist and the next boot would skip every document —
    /// standing up a cast that knows nothing, behind a green loading screen.
    #[test]
    fn the_ledger_is_wiped_by_wiping_the_substrate() {
        let data = Path::new("/data");
        assert_eq!(
            ledger_path(data),
            Path::new("/data/.substrate/ingest-ledger.json")
        );
        assert!(ledger_path(data).starts_with(data.join(".substrate")));
    }

    /// A ledger with no path is in-memory and must not try to write.
    #[test]
    fn an_in_memory_ledger_flushes_to_nothing() {
        let l = Ledger::new();
        l.reconcile(Path::new("x.md"), Some("x"));
        l.flush(); // must not panic, must not create anything
    }
}
