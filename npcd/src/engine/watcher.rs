//! The ingest ledger: what each mind document hashed to when it became turns in
//! the substrate, so a start writes only what moved since.
//!
//! The mind is read at a commit from the substrate (`docs/substrate_git.md`
//! §8), and its documents are keyed here by their mind path. Three cases, each a
//! different write:
//!
//! - **Unchanged** — the content hash matches. Nothing is written.
//! - **Changed** — the old turn is **tombstoned** and the new content ingested
//!   as a fresh turn. Not overwritten: the substrate is an append-only redo log,
//!   and the previous version stays reachable.
//! - **Removed** — the document left the record; its conversations are
//!   retired at the next start ([`Ledger::missing`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use candle_conversation::persistence::SUBSTRATE_DIR;
use sha2::{Digest, Sha256};

use crate::mind::files::MindFiles;

/// What a mind file's content hashed to, when it was last ingested.
///
/// # Why this is on disk
///
/// Because ingesting a document is a real prefill — measured at about four
/// seconds each on a 3090 — and a mind with four hundred of them is half an hour
/// of startup. In memory only, that half hour was paid **on every boot**, for
/// work already in the substrate.
///
/// It lives *inside* `substrate/`, deliberately: the ledger's claim is "this
/// document is already a turn in there", so wiping the substrate has to wipe the
/// ledger with it. Beside the substrate rather than inside, a `rm -rf
/// substrate` would leave a ledger asserting turns that no longer exist, and
/// the next boot would skip every document and stand up a cast that knows
/// nothing — with a green loading screen.
#[derive(Debug, Default)]
pub struct Ledger {
    hashes: Mutex<HashMap<PathBuf, String>>,
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
    data.join(SUBSTRATE_DIR).join("ingest-ledger.json")
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
        // Ledgers before mind paths keyed each document by the absolute path a
        // walk of `<mind>/layers` found it at. Rewritten once to the mind path
        // from `layers/` on, keeping the hash — or every document re-ingests.
        let (mut migrated, mut dropped) = (0usize, 0usize);
        let hashes: HashMap<PathBuf, String> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str::<HashMap<PathBuf, String>>(&t).ok())
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(key, h)| match mind_key(&key) {
                Some(k) if k == key => Some((k, h)),
                Some(k) => {
                    migrated += 1;
                    Some((k, h))
                }
                None => {
                    dropped += 1;
                    None
                }
            })
            .collect();
        let ledger = Self {
            hashes: Mutex::new(hashes),
            path: Some(path),
        };
        if migrated + dropped > 0 {
            ledger.flush();
            tracing::info!(
                "ingest ledger: {migrated} key(s) migrated to mind paths, {dropped} outside `layers/` dropped"
            );
        }
        let hashes = ledger.hashes.lock().unwrap().len();
        if hashes > 0 {
            tracing::info!("ingest ledger: {hashes} document(s) already in the substrate");
        }
        ledger
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

    /// Forget every document, and persist that.
    ///
    /// For a substrate whose conversations have all been retired
    /// (`--wipe-conversations`): the ledger's claim is "this document is a turn
    /// in there", and after the wipe none is, so every document is owed again
    /// and the whole mind re-ingests.
    pub fn forget_all(&self) {
        self.hashes.lock().unwrap().clear();
        self.flush();
    }

    /// Every document the ledger records as ingested whose file is gone —
    /// deleted, or moved out of the record (a rejected draft goes to
    /// `rejected/`).
    pub fn missing(&self, mind: &MindFiles) -> Vec<PathBuf> {
        let mut gone: Vec<PathBuf> = self
            .hashes
            .lock()
            .unwrap()
            .keys()
            .filter(|p| !mind.is_file(&p.to_string_lossy().replace('\\', "/")))
            .cloned()
            .collect();
        gone.sort();
        gone
    }

    /// Stop recording `path` — its conversations have been retired. Not
    /// persisted until the next [`Self::flush`].
    pub fn forget(&self, path: &Path) {
        self.hashes.lock().unwrap().remove(path);
    }

    /// Forget every document under the mind folder `dir`.
    pub fn forget_under(&self, dir: &Path) {
        self.hashes
            .lock()
            .unwrap()
            .retain(|p, _| !p.starts_with(dir));
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

/// A ledger key as a mind path: itself when it already is one, else its suffix
/// from the first `layers` component, joined with '/'. `None` for a key with no
/// `layers` in it — nothing a mind path could be recovered from.
fn mind_key(key: &Path) -> Option<PathBuf> {
    let text = key.to_string_lossy();
    let parts: Vec<&str> = text.split(['/', '\\']).collect();
    let valid = !key.has_root()
        && !key.is_absolute()
        && !text.contains('\\')
        && parts
            .iter()
            .all(|p| !p.is_empty() && *p != "." && *p != ".." && !p.contains(':'));
    if valid {
        return Some(key.to_path_buf());
    }
    let from = parts.iter().position(|p| *p == "layers")?;
    Some(PathBuf::from(parts[from..].join("/")))
}

/// Files in the mind that carry ingestible content.
///
/// Authored content is YAML and Markdown. Everything else in a mind directory —
/// a stray image, an editor's swap file, a `.git` — is not content and must not
/// become a turn.
pub fn is_ingestible(path: &Path) -> bool {
    if path
        .components()
        .any(|c| matches!(c.as_os_str().to_str(), Some(".git") | Some(SUBSTRATE_DIR)))
    {
        return false;
    }
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("yaml") | Some("yml") | Some("md")
    )
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
        assert!(!is_ingestible(Path::new("substrate/seg-1.log")));
        assert!(!is_ingestible(Path::new("mind/.git/config.yaml")));
        assert!(!is_ingestible(Path::new("substrate/x.yaml")));
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

    /// **A recorded document whose file is gone is listed as missing**, and
    /// forgetting it stops listing it.
    #[test]
    fn a_recorded_document_whose_file_is_gone_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let here = PathBuf::from("here.md");
        let gone = PathBuf::from("gone.md");
        std::fs::write(dir.path().join(&here), "x").unwrap();
        let mind = MindFiles::folder(dir.path());
        let l = Ledger::new();
        l.reconcile(&here, Some("x"));
        l.reconcile(&gone, Some("y"));
        assert_eq!(l.missing(&mind), vec![gone.clone()]);
        l.forget(&gone);
        assert!(l.missing(&mind).is_empty());
        assert_eq!(l.inspect(&gone, Some("y")), Reconcile::Added);
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

    /// **A restart must not re-ingest what is already in the substrate.**
    ///
    /// Each document is a real prefill — about four seconds — so a mind with
    /// four hundred of them is half an hour of startup, and in memory only that
    /// half hour was paid on every boot for work already done.
    #[test]
    fn a_saved_ledger_is_read_back_and_skips_what_it_holds() {
        let data = tmp();
        let doc = PathBuf::from("layers/world.md");

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

    /// **A ledger keyed by absolute paths opens keyed by mind paths**, hashes
    /// kept — or the first boot after mind paths re-ingests the whole mind.
    #[test]
    fn an_absolute_keyed_ledger_opens_as_mind_paths() {
        let data = tmp();
        let old = PathBuf::from("D:/prog/mind\\layers\\world\\x.md");
        let stray = PathBuf::from("D:/prog/mind\\missions.yaml");
        let h = hash("# X");
        let file = ledger_path(&data);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        let old_map: HashMap<PathBuf, String> =
            [(old, h.clone()), (stray, h.clone())].into_iter().collect();
        std::fs::write(&file, serde_json::to_string(&old_map).unwrap()).unwrap();

        let l = Ledger::open(&data);
        let new = Path::new("layers/world/x.md");
        assert_eq!(
            l.hashes.lock().unwrap().clone(),
            [(new.to_path_buf(), h)]
                .into_iter()
                .collect::<HashMap<_, _>>()
        );
        assert_eq!(l.reconcile(new, Some("# X")), Reconcile::Unchanged);
        // Flushed once, so the next open reads mind paths straight off disk.
        assert_eq!(
            Ledger::open(&data).inspect(new, Some("# X")),
            Reconcile::Unchanged
        );
        let _ = std::fs::remove_dir_all(&data);
    }

    /// **After a wipe, every document is owed again** — on disk as well as in
    /// memory. `--wipe-conversations` retires every turn the ledger vouches for,
    /// and a ledger that still vouched for them after a restart would skip the
    /// whole mind and stand up a cast that knows nothing.
    #[test]
    fn a_forgotten_ledger_owes_every_document_again() {
        let data = tmp();
        let doc = PathBuf::from("layers/world.md");

        let first = Ledger::open(&data);
        first.reconcile(&doc, Some("# Alpha"));
        first.flush();
        first.forget_all();
        assert_eq!(first.inspect(&doc, Some("# Alpha")), Reconcile::Added);

        let second = Ledger::open(&data);
        assert_eq!(
            second.inspect(&doc, Some("# Alpha")),
            Reconcile::Added,
            "a restart skipped a document the wipe had retired"
        );
        let _ = std::fs::remove_dir_all(&data);
    }

    /// It lives *inside* `substrate/`, so wiping the substrate wipes the
    /// ledger. Beside it, a `rm -rf substrate` would leave a ledger asserting
    /// turns that no longer exist and the next boot would skip every document —
    /// standing up a cast that knows nothing, behind a green loading screen.
    #[test]
    fn the_ledger_is_wiped_by_wiping_the_substrate() {
        let data = Path::new("/data");
        assert_eq!(
            ledger_path(data),
            Path::new("/data/substrate/ingest-ledger.json")
        );
        assert!(ledger_path(data).starts_with(data.join(SUBSTRATE_DIR)));
    }

    /// A ledger with no path is in-memory and must not try to write.
    #[test]
    fn an_in_memory_ledger_flushes_to_nothing() {
        let l = Ledger::new();
        l.reconcile(Path::new("x.md"), Some("x"));
        l.flush(); // must not panic, must not create anything
    }
}
