//! Records kept as one YAML file each in a directory.
//!
//! Two kinds of collection are kept this way. Worlds and personalities are
//! **written** by an author, in the mind: the setting an author invents and
//! the doctrine they give a kind of character, wanting what files give — a
//! diff, a review, a revert, survival across a substrate wipe. A registry of
//! them is what the mind folder holds as published; a save is an author's
//! change to the mind ([`authored`], [`crate::drafts`]) and the registry is
//! read again when the mind folder moves. Device tokens are written by this
//! daemon, in its data directory, through [`Registry::put`].
//!
//! # The whole directory is read at once
//!
//! After that the registry answers from memory. That is a security decision
//! before it is a performance one: **a request never names a path.** An id
//! arriving on a URL is a key into a `BTreeMap`, and a key that is not in the
//! map is a 404 rather than an open() of something. There is no traversal to
//! defend against on the read path because there is no path.
//!
//! Writing is the one place an id becomes a file name, and it goes through
//! [`id::check`] — an allowlist of 37 characters plus the Win32 device names
//! that allowlist cannot see. Belt and braces, the assembled path is then
//! checked to still be a direct child of the directory before anything is
//! written; that check should be unreachable, and is cheap enough to keep.

pub mod authored;
pub mod id;
/// Public because the mind's field editor patches documents the registry does
/// not own — a canon page, a response section — and both need the same
/// comment-preserving splice. One implementation, two callers, rather than a
/// second one that would be free to lose comments differently.
pub mod yaml_edit;

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

/// Largest document this will write.
///
/// Twenty times the biggest real personality in the mind, which is enough that
/// no author meets it, and small enough that the API is not a way to fill a
/// disk one `PUT` at a time. Applied to the rendered bytes rather than the
/// request body, because that is the thing that lands.
const MAX_BYTES: usize = 256 * 1024;

/// Why a save was refused.
///
/// The variants exist so the HTTP layer can tell the author's mistake from the
/// server's, which a single error string cannot: `id::check`'s message names
/// what is wrong with an id and is exactly what the author needs, while an
/// `io::Error` from the write carries the absolute path of a file on this
/// machine and is exactly what a stranger should not be handed. That leak was
/// real — a failed write answered with
/// `world: writing C:\Users\...\mind\worlds\x.yaml`.
#[derive(Debug)]
pub enum PutError {
    /// The id could not become a file name. Safe to report verbatim.
    BadId(id::IdError),
    /// The document is larger than [`MAX_BYTES`]. Safe to report: the limit is
    /// part of the API.
    TooLarge { bytes: usize, max: usize },
    /// The path exists and is not a regular file — a symlink, a directory, a
    /// junction. Refused rather than followed.
    NotAPlainFile,
    /// Something went wrong on this machine. **Logged, never returned**: the
    /// detail is for the operator reading the log, not for the caller.
    Io(anyhow::Error),
}

impl fmt::Display for PutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PutError::BadId(e) => write!(f, "{e}"),
            PutError::TooLarge { bytes, max } => {
                write!(f, "the document is {bytes} bytes; the limit is {max}")
            }
            PutError::NotAPlainFile => {
                write!(f, "that name is taken by something that is not a file")
            }
            PutError::Io(e) => write!(f, "{e:#}"),
        }
    }
}

impl std::error::Error for PutError {}

impl PutError {
    /// Whether this is something the caller did, and may therefore be told.
    ///
    /// The default is *no*. A variant added later is a server fault until
    /// somebody decides otherwise, which is the safe direction for a decision
    /// about what leaves the machine.
    pub fn is_callers_fault(&self) -> bool {
        matches!(
            self,
            PutError::BadId(_) | PutError::TooLarge { .. } | PutError::NotAPlainFile
        )
    }
}

/// One record, and where it came from.
#[derive(Debug, Clone)]
pub struct Record {
    pub id: String,
    /// The document as written. Kept as JSON rather than a typed struct
    /// because the schema is still moving and a field the daemon does not
    /// understand yet must survive a load/save round trip rather than being
    /// silently dropped — losing an author's work to a struct update is not a
    /// trade worth making.
    pub body: Value,
}

/// An in-memory collection of records backed by one directory.
#[derive(Debug)]
pub struct Registry {
    dir: PathBuf,
    /// Ordered so listings are stable — a GUI list that reshuffles between
    /// loads looks broken even when it is not.
    items: BTreeMap<String, Record>,
    /// What this collection is called in errors and logs (`world`, `personality`).
    kind: &'static str,
}

impl Registry {
    /// Read every `*.yaml` in `dir` into memory.
    ///
    /// A missing directory is an empty registry, not an error: a fresh clone
    /// has no worlds yet, and refusing to start would be the wrong answer to
    /// "you have not written one".
    ///
    /// A file that does not parse is **skipped with a loud log**, not fatal.
    /// One malformed world must not take the daemon down and with it every
    /// other world — the same call the blog index makes about a post with
    /// broken front matter.
    pub fn load(kind: &'static str, dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let items = Self::read(kind, &dir)?;
        tracing::info!("{kind}: {} loaded from {}", items.len(), dir.display());
        Ok(Self { dir, items, kind })
    }

    /// Read the directory again, as it stands now. The ids that were here and
    /// are gone come back, so whatever runs from one can be stopped.
    pub fn reload(&mut self) -> Result<Vec<String>> {
        let items = Self::read(self.kind, &self.dir)?;
        let gone = self
            .items
            .keys()
            .filter(|id| !items.contains_key(*id))
            .cloned()
            .collect();
        self.items = items;
        Ok(gone)
    }

    fn read(kind: &'static str, dir: &Path) -> Result<BTreeMap<String, Record>> {
        let mut items = BTreeMap::new();
        if !dir.is_dir() {
            return Ok(items);
        }
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {kind} directory {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
            .collect();
        entries.sort();

        for path in entries {
            let Some(file) = path.file_name().and_then(|s| s.to_str()) else {
                tracing::warn!(path = %path.display(), "{kind}: file name is not UTF-8, skipping");
                continue;
            };
            let text = match std::fs::read_to_string(&path) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "{kind}: unreadable, skipping");
                    continue;
                }
            };
            if let Some(record) = authored::record_of(kind, file, &text) {
                items.insert(record.id.clone(), record);
            }
        }
        Ok(items)
    }

    /// The directory the records are read from.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether nothing has been registered. A registry over an empty directory
    /// is a legitimate state — a world with no personalities authored yet.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Look up by id. This is what a URL segment reaches — a map lookup, never
    /// a path.
    pub fn get(&self, id: &str) -> Option<&Record> {
        self.items.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Record> {
        self.items.values()
    }

    /// Write a record this daemon keeps to disk and into memory, serialised
    /// whole: nobody has ever commented one, so there is nothing in the file
    /// to keep.
    ///
    /// Disk first: if the write fails, memory still matches what is on disk,
    /// and the caller gets an error instead of a registry that has quietly
    /// diverged from the directory it is supposed to be.
    pub fn put(&mut self, id: &str, body: Value) -> std::result::Result<(), PutError> {
        id::check(id).map_err(PutError::BadId)?;
        let path = self.path_for(id).map_err(PutError::Io)?;

        // Refuse anything that is not a plain file *before* writing, because
        // `fs::write` follows a symlink and would put this document wherever
        // the link points. `path_for` validates the path string; this validates
        // what is actually there, which is the half a string check cannot do.
        match std::fs::symlink_metadata(&path) {
            Ok(m) if !m.is_file() => return Err(PutError::NotAPlainFile),
            _ => {}
        }

        let yaml = serde_yaml::to_string(&body)
            .with_context(|| format!("{}: serialising `{id}`", self.kind))
            .map_err(PutError::Io)?;
        if yaml.len() > MAX_BYTES {
            return Err(PutError::TooLarge {
                bytes: yaml.len(),
                max: MAX_BYTES,
            });
        }

        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))
            .map_err(PutError::Io)?;
        std::fs::write(&path, yaml)
            .with_context(|| format!("{}: writing {}", self.kind, path.display()))
            .map_err(PutError::Io)?;

        self.items.insert(
            id.to_string(),
            Record {
                id: id.to_string(),
                body,
            },
        );
        Ok(())
    }

    /// Remove a record from disk and memory. Absent is not an error — the
    /// caller asked for it to be gone and it is gone.
    pub fn delete(&mut self, id: &str) -> std::result::Result<bool, PutError> {
        id::check(id).map_err(PutError::BadId)?;
        let path = self.path_for(id).map_err(PutError::Io)?;
        let existed = self.items.remove(id).is_some();
        // `symlink_metadata`, not `exists`: a dangling symlink is not `exists`
        // but is very much a thing to remove, and a live one must be unlinked
        // rather than followed.
        if std::fs::symlink_metadata(&path).is_ok() {
            std::fs::remove_file(&path)
                .with_context(|| format!("{}: removing {}", self.kind, path.display()))
                .map_err(PutError::Io)?;
        }
        Ok(existed)
    }

    /// The file an id maps to, having proved it stays in the directory.
    ///
    /// `id::check` already guarantees this — the allowlist contains no
    /// separator, no dot and no colon, so the join cannot go anywhere else. The
    /// check is here anyway because it is two comparisons against a mistake
    /// that would be catastrophic and silent, and because it keeps the
    /// guarantee true if the allowlist is ever widened by someone who has not
    /// read [`id`].
    fn path_for(&self, id: &str) -> Result<PathBuf> {
        let path = self.dir.join(format!("{id}.yaml"));
        anyhow::ensure!(
            path.parent() == Some(self.dir.as_path()),
            "{}: `{id}` does not resolve inside {}",
            self.kind,
            self.dir.display()
        );
        anyhow::ensure!(
            path.file_name().and_then(|f| f.to_str()) == Some(&format!("{id}.yaml")),
            "{}: `{id}` does not name the file it should",
            self.kind
        );
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// **What may leave the machine.** The HTTP layer decides by variant, so
    /// this is the rule itself rather than a rendering of it.
    ///
    /// A failed write used to answer with the anyhow context, which carried the
    /// absolute path of a file on this host — `world: writing
    /// C:\Users\...\worlds\x.yaml` — to an unauthenticated caller. The default
    /// for a new variant is *not* the caller's fault, so the safe answer is the
    /// one you get by forgetting to think about it.
    #[test]
    fn only_the_callers_own_mistakes_are_reportable() {
        assert!(PutError::BadId(id::IdError::Empty).is_callers_fault());
        assert!(PutError::TooLarge { bytes: 1, max: 0 }.is_callers_fault());
        assert!(PutError::NotAPlainFile.is_callers_fault());
        assert!(!PutError::Io(anyhow::anyhow!("writing C:\\secret\\path")).is_callers_fault());

        // And the ones that are reportable say nothing about this machine. Note
        // `BadChar('/')` quotes the character the *caller* typed, which is the
        // whole value of the message — so the check is for path shapes rather
        // than for a slash.
        for e in [
            PutError::BadId(id::IdError::BadChar('/')),
            PutError::BadId(id::IdError::Reserved),
            PutError::TooLarge {
                bytes: 999,
                max: 100,
            },
            PutError::NotAPlainFile,
        ] {
            let text = e.to_string();
            for shape in [":\\", "/Users/", "/home/", ".yaml", "\\\\?\\"] {
                assert!(!text.contains(shape), "`{shape}` in `{text}`");
            }
        }
    }

    /// A document larger than the limit is refused rather than written.
    #[test]
    fn an_oversized_document_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Registry::load("token", dir.path()).unwrap();
        let big = json!({ "name": "x", "setting": "a".repeat(MAX_BYTES + 1) });
        assert!(matches!(r.put("big", big), Err(PutError::TooLarge { .. })));
        assert!(
            !dir.path().join("big.yaml").exists(),
            "it was written anyway"
        );
    }

    /// `fs::write` follows a symlink, so a link at `<id>.yaml` would put the
    /// document wherever it points. Exercised with a directory because creating
    /// a symlink needs privileges a test cannot assume on Windows — the guard is
    /// the same one, and it is the `!is_file()` branch either way.
    #[test]
    fn a_name_taken_by_something_that_is_not_a_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("taken.yaml")).unwrap();
        let mut r = Registry::load("token", dir.path()).unwrap();
        assert!(matches!(
            r.put("taken", json!({ "name": "x" })),
            Err(PutError::NotAPlainFile)
        ));
        assert!(dir.path().join("taken.yaml").is_dir(), "it was clobbered");
    }

    #[test]
    fn a_missing_directory_is_an_empty_registry_not_a_failure() {
        let dir = tempfile::tempdir().unwrap();
        let r = Registry::load("world", dir.path().join("does-not-exist")).unwrap();
        assert_eq!(r.len(), 0);
    }

    /// **A save is serialised whole**: a daemon's record has no comments to
    /// keep, and the bytes are exactly the serialisation.
    #[test]
    fn a_save_is_the_whole_serialisation() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("t.yaml"), "# a note\nname: A\n").unwrap();
        let mut r = Registry::load("token", dir.path()).unwrap();
        r.put("t", json!({"name": "B", "npc": 7})).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.path().join("t.yaml")).unwrap(),
            "name: B\nnpc: 7\n"
        );
    }

    #[test]
    fn a_record_survives_a_save_and_a_reload() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Registry::load("token", dir.path()).unwrap();
        r.put(
            "ardh",
            json!({"name": "Ardh", "some_future_field": {"deep": [1, 2, 3]}}),
        )
        .unwrap();
        let again = Registry::load("token", dir.path()).unwrap();
        assert_eq!(again.len(), 1);
        assert_eq!(
            again.get("ardh").unwrap().body["some_future_field"]["deep"][2],
            3
        );
    }

    #[test]
    fn a_bad_id_is_refused_before_anything_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Registry::load("token", dir.path()).unwrap();
        for bad in ["../escape", "a/b", "con", "", "x.yaml", "-x"] {
            assert!(r.put(bad, json!({})).is_err(), "accepted `{bad}`");
        }
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    /// The point of the whole design: a lookup is a map lookup. An id nobody
    /// wrote is simply absent, whatever it says.
    #[test]
    fn an_unknown_id_is_absent_rather_than_a_path() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Registry::load("token", dir.path()).unwrap();
        r.put("ardh", json!({"name": "Ardh"})).unwrap();
        for miss in ["nope", "../../etc/passwd", "ardh.yaml", "ARDH"] {
            assert!(r.get(miss).is_none(), "`{miss}` resolved to something");
        }
    }

    #[test]
    fn a_file_that_does_not_parse_or_is_misnamed_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("good.yaml"), "name: Good\n").unwrap();
        std::fs::write(dir.path().join("broken.yaml"), "name: [unclosed\n").unwrap();
        std::fs::write(dir.path().join("Upper.yaml"), "name: Upper\n").unwrap();
        let r = Registry::load("world", dir.path()).unwrap();
        assert_eq!(
            r.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(),
            ["good"]
        );
    }

    /// **A reload reads the directory as it stands and names what went.**
    #[test]
    fn a_reload_reads_the_directory_again() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.yaml"), "name: A\n").unwrap();
        std::fs::write(dir.path().join("b.yaml"), "name: B\n").unwrap();
        let mut r = Registry::load("world", dir.path()).unwrap();
        std::fs::remove_file(dir.path().join("a.yaml")).unwrap();
        std::fs::write(dir.path().join("b.yaml"), "name: B2\n").unwrap();
        std::fs::write(dir.path().join("c.yaml"), "name: C\n").unwrap();
        assert_eq!(r.reload().unwrap(), ["a"]);
        assert_eq!(r.get("b").unwrap().body["name"], "B2");
        assert_eq!(
            r.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(),
            ["b", "c"]
        );
    }

    #[test]
    fn delete_removes_from_both_disk_and_memory() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Registry::load("token", dir.path()).unwrap();
        r.put("ardh", json!({"name": "Ardh"})).unwrap();
        assert!(r.delete("ardh").unwrap());
        assert!(r.get("ardh").is_none());
        assert!(!dir.path().join("ardh.yaml").exists());
        assert!(!r.delete("ardh").unwrap());
    }

    #[test]
    fn listing_order_is_stable() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Registry::load("token", dir.path()).unwrap();
        for id in ["zeta", "alpha", "mid"] {
            r.put(id, json!({"name": id})).unwrap();
        }
        let ids: Vec<_> = Registry::load("token", dir.path())
            .unwrap()
            .iter()
            .map(|x| x.id.clone())
            .collect();
        assert_eq!(ids, ["alpha", "mid", "zeta"]);
    }
}
