//! Authored content that lives in the repository rather than in the substrate.
//!
//! Worlds and personalities are **written**, not accumulated. They are the setting
//! an author invents and the doctrine they give a kind of character, and they
//! want the things files give you — a diff, a review, a revert, and survival
//! across a substrate wipe. NPCs are the opposite: they are lived experience,
//! they belong to the substrate, and nothing here touches them.
//!
//! # The whole directory is read once, at boot
//!
//! After that the registry answers from memory and the filesystem is touched
//! only when something is saved. That is a security decision before it is a
//! performance one: **a request never names a path.** An id arriving on a URL
//! is a key into a `BTreeMap`, and a key that is not in the map is a 404 rather
//! than an open() of something. There is no traversal to defend against on the
//! read path because there is no path.
//!
//! Writing is the one place an id becomes a file name, and it goes through
//! [`id::check`] — an allowlist of 37 characters plus the Win32 device names
//! that allowlist cannot see. Belt and braces, the assembled path is then
//! checked to still be a direct child of the directory before anything is
//! written; that check should be unreachable, and is cheap enough to keep.
//!
//! # A save edits the file; it does not rewrite it
//!
//! These documents carry their reasoning in their comments, and none of that is
//! data. Saving through `serde_yaml` would round-trip the parsed value and
//! delete every comment and block scalar in the file — so [`Registry::put`]
//! goes through [`yaml_edit`], which replaces only the values that changed and
//! copies every other byte through. See that module for why, and for what it
//! does when a document is one it cannot edit.

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

/// One authored record, and where it came from.
#[derive(Debug, Clone)]
pub struct Record {
    pub id: String,
    /// The document as authored. Kept as JSON rather than a typed struct
    /// because the schema is still moving and a field the daemon does not
    /// understand yet must survive a load/save round trip rather than being
    /// silently dropped — losing an author's work to a struct update is not a
    /// trade worth making.
    pub body: Value,
}

/// An in-memory collection of authored records backed by one directory.
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
        let mut items = BTreeMap::new();

        if dir.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(&dir)
                .with_context(|| format!("reading {kind} directory {}", dir.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
                .collect();
            entries.sort();

            for path in entries {
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    tracing::warn!(path = %path.display(), "{kind}: file name is not UTF-8, skipping");
                    continue;
                };
                // A file already on disk gets the same gate as one we would
                // write. Something dropped in by hand — or by a checkout of a
                // branch written on another OS — must not become an id the GUI
                // then cannot save back.
                if let Err(e) = id::check(stem) {
                    tracing::warn!(path = %path.display(), error = %e, "{kind}: unusable file name, skipping");
                    continue;
                }
                let text = match std::fs::read_to_string(&path) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "{kind}: unreadable, skipping");
                        continue;
                    }
                };
                match serde_yaml::from_str::<Value>(&text) {
                    Ok(body) => {
                        items.insert(
                            stem.to_string(),
                            Record {
                                id: stem.to_string(),
                                body,
                            },
                        );
                    }
                    Err(e) => {
                        tracing::warn!(path = %path.display(), error = %e, "{kind}: does not parse, skipping");
                    }
                }
            }
        }

        tracing::info!("{kind}: {} loaded from {}", items.len(), dir.display());
        Ok(Self { dir, items, kind })
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Look up by id. This is what a URL segment reaches — a map lookup, never
    /// a path.
    pub fn get(&self, id: &str) -> Option<&Record> {
        self.items.get(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Record> {
        self.items.values()
    }

    /// Write a record to disk and into memory.
    ///
    /// Disk first: if the write fails, memory still matches what is on disk,
    /// and the caller gets an error instead of a registry that has quietly
    /// diverged from the repository it is supposed to be.
    ///
    /// An existing file is **edited, not rewritten** — see [`yaml_edit`]. The
    /// document's comments are the most valuable thing in it and none of them
    /// are data, so a save that changes a name changes one line and leaves the
    /// rest of the author's file alone.
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

        let yaml = self.render(id, &path, &body).map_err(PutError::Io)?;
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
        tracing::info!("{}: saved `{id}` to {}", self.kind, path.display());
        Ok(())
    }

    /// The bytes to write for `body`: an edit of the file that is there, or a
    /// fresh serialisation when there is nothing to edit.
    ///
    /// The base is re-read from disk rather than taken from memory, so a
    /// comment, a blank line or a rewrap somebody added by hand while the
    /// daemon was up survives the next save. It is **not** protection against a
    /// lost update: a `PUT` is a whole-document replacement, so a data field the
    /// caller sends overwrites a hand edit to that same field regardless of
    /// which text is used as the base. Closing that gap needs a revision on the
    /// wire, or a watcher keeping memory current; neither exists yet, and this
    /// re-read is worth doing on its own for the formatting.
    fn render(&self, id: &str, path: &Path, body: &Value) -> Result<String> {
        let whole = || {
            serde_yaml::to_string(body)
                .with_context(|| format!("{}: serialising `{id}`", self.kind))
        };
        let Some(map) = body.as_object() else {
            return whole();
        };
        let Ok(original) = std::fs::read_to_string(path) else {
            return whole(); // a new record; nothing to preserve
        };
        match yaml_edit::splice(&original, map) {
            Some(edited) => Ok(edited),
            None => {
                // The file exists and could not be edited — an alias, a root
                // that is not a mapping, or a rendering that failed its own
                // read-back. Saying so matters: the save still succeeds, and
                // the cost is the author's comments.
                tracing::warn!(
                    path = %path.display(),
                    "{}: `{id}` could not be edited in place — rewriting whole, comments will be lost",
                    self.kind
                );
                whole()
            }
        }
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

    fn tmp() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "npcd-registry-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

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

    /// A document larger than the limit is refused rather than written. Without
    /// this, an authored file was capped only by axum's 2 MB body default and a
    /// caller could fill a disk one save at a time.
    #[test]
    fn an_oversized_document_is_refused() {
        let dir = tmp();
        let mut r = Registry::load("world", &dir).unwrap();
        let big = json!({ "name": "x", "setting": "a".repeat(MAX_BYTES + 1) });
        assert!(matches!(r.put("big", big), Err(PutError::TooLarge { .. })));
        assert!(!dir.join("big.yaml").exists(), "it was written anyway");

        // The limit is generous: the largest real personality is ~11 KiB.
        let ok = json!({ "name": "x", "setting": "a".repeat(64 * 1024) });
        assert!(r.put("fine", ok).is_ok());
    }

    /// `fs::write` follows a symlink, so a link at `<id>.yaml` would put the
    /// document wherever it points. `path_for` validates the path string; this
    /// validates what is actually there, which a string check cannot.
    ///
    /// Exercised with a directory because creating a symlink needs privileges a
    /// test cannot assume on Windows — the guard is the same one, and it is the
    /// `!is_file()` branch either way.
    #[test]
    fn a_name_taken_by_something_that_is_not_a_file_is_refused() {
        let dir = tmp();
        std::fs::create_dir_all(dir.join("taken.yaml")).unwrap();
        let mut r = Registry::load("world", &dir).unwrap();
        assert!(matches!(
            r.put("taken", json!({ "name": "x" })),
            Err(PutError::NotAPlainFile)
        ));
        assert!(dir.join("taken.yaml").is_dir(), "it was clobbered");
    }

    #[test]
    fn a_missing_directory_is_an_empty_registry_not_a_failure() {
        let r = Registry::load("world", tmp().join("does-not-exist")).unwrap();
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn a_record_survives_a_save_and_a_reload() {
        let dir = tmp();
        let mut r = Registry::load("world", &dir).unwrap();
        r.put("ardh", json!({"name": "Ardh", "setting": "A kingdom."}))
            .unwrap();

        let again = Registry::load("world", &dir).unwrap();
        assert_eq!(again.len(), 1);
        assert_eq!(again.get("ardh").unwrap().body["name"], "Ardh");
    }

    /// A field the daemon has no struct for must come back unchanged. Losing an
    /// author's work to a schema that has not caught up yet is the failure this
    /// prevents.
    #[test]
    fn unknown_fields_survive_the_round_trip() {
        let dir = tmp();
        let mut r = Registry::load("world", &dir).unwrap();
        r.put(
            "ardh",
            json!({"name": "Ardh", "some_future_field": {"deep": [1, 2, 3]}}),
        )
        .unwrap();

        let again = Registry::load("world", &dir).unwrap();
        assert_eq!(
            again.get("ardh").unwrap().body["some_future_field"]["deep"][2],
            3
        );
    }

    #[test]
    fn a_bad_id_is_refused_before_anything_is_written() {
        let dir = tmp();
        let mut r = Registry::load("world", &dir).unwrap();
        for bad in ["../escape", "a/b", "con", "", "x.yaml", "-x"] {
            assert!(r.put(bad, json!({})).is_err(), "accepted `{bad}`");
        }
        // Nothing reached the disk.
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    }

    /// The point of the whole design: a lookup is a map lookup. An id nobody
    /// authored is simply absent, whatever it says.
    #[test]
    fn an_unknown_id_is_absent_rather_than_a_path() {
        let dir = tmp();
        let mut r = Registry::load("world", &dir).unwrap();
        r.put("ardh", json!({"name": "Ardh"})).unwrap();
        for miss in ["nope", "../../etc/passwd", "ardh.yaml", "ARDH"] {
            assert!(r.get(miss).is_none(), "`{miss}` resolved to something");
        }
    }

    #[test]
    fn a_file_that_does_not_parse_is_skipped_and_the_rest_still_load() {
        let dir = tmp();
        std::fs::write(dir.join("good.yaml"), "name: Good\n").unwrap();
        std::fs::write(dir.join("broken.yaml"), "name: [unclosed\n").unwrap();
        let r = Registry::load("world", &dir).unwrap();
        assert_eq!(r.len(), 1, "the good one should still be there");
        assert!(r.get("good").is_some());
        assert!(r.get("broken").is_none());
    }

    /// A file whose name could not have been written by us — dropped in by
    /// hand, or arriving from a checkout made on another OS.
    #[test]
    fn a_file_with_an_unusable_name_is_skipped() {
        let dir = tmp();
        std::fs::write(dir.join("Ardh.yaml"), "name: Uppercase\n").unwrap();
        std::fs::write(dir.join("ok.yaml"), "name: Fine\n").unwrap();
        let r = Registry::load("world", &dir).unwrap();
        assert!(r.get("Ardh").is_none(), "an unsaveable id was loaded");
        assert!(r.get("ok").is_some());
    }

    /// The property the console depends on: saving one field does not cost the
    /// author the rest of the file. Tested here as well as in [`yaml_edit`]
    /// because this is the layer the API calls, and a `put` that forgot to use
    /// the editor would pass every test in that module.
    #[test]
    fn a_save_keeps_the_comments_and_the_shape_of_the_file() {
        let dir = tmp();
        let authored = "\
# Sandbox -- why this world is written the way it is.
#
# The setting says what IS there, not what is absent.

id: sandbox
name: Sandbox
public: false

setting: >-
  A room, a view from it, and objects that behave the same way
  every time you touch them.

# No shared canon: this world's whole content is one life.
selects: []
";
        std::fs::write(dir.join("sandbox.yaml"), authored).unwrap();

        let mut r = Registry::load("world", &dir).unwrap();
        let mut body = r.get("sandbox").unwrap().body.clone();
        body["name"] = json!("Sandbox Prime");
        r.put("sandbox", body).unwrap();

        let after = std::fs::read_to_string(dir.join("sandbox.yaml")).unwrap();
        assert_eq!(
            after,
            authored.replace("name: Sandbox\n", "name: Sandbox Prime\n"),
            "a save changed more than the one field it was given"
        );
    }

    /// A record with no file yet has nothing to preserve, and must still land.
    #[test]
    fn a_brand_new_record_is_written_from_scratch() {
        let dir = tmp();
        let mut r = Registry::load("world", &dir).unwrap();
        r.put("ardh", json!({ "name": "Ardh", "selects": ["north"] }))
            .unwrap();
        let text = std::fs::read_to_string(dir.join("ardh.yaml")).unwrap();
        assert!(text.contains("name: Ardh"), "{text}");
        assert_eq!(
            Registry::load("world", &dir)
                .unwrap()
                .get("ardh")
                .unwrap()
                .body["selects"][0],
            "north"
        );
    }

    /// The base for an edit is the file on disk, not the copy in memory — so
    /// prose somebody added by hand while the daemon was up is still there
    /// afterwards.
    ///
    /// What this deliberately does **not** claim is protection against a lost
    /// update. A `PUT` replaces the document, so a data field the caller sends
    /// wins over a hand edit to that same field; the comment survives and the
    /// value does not. Re-reading the file cannot change that and is not
    /// pretending to — it is worth doing for the formatting alone.
    #[test]
    fn a_save_edits_the_file_on_disk_not_the_copy_in_memory() {
        let dir = tmp();
        std::fs::write(dir.join("ardh.yaml"), "name: Ardh\nsetting: Old.\n").unwrap();
        let mut r = Registry::load("world", &dir).unwrap();

        // Somebody edits the file directly, adding a comment and a field the
        // registry has never seen. It knows about neither.
        std::fs::write(
            dir.join("ardh.yaml"),
            "# written by hand\nname: Ardh\nsetting: Old.\n\n# and a new field\nmood: bleak\n",
        )
        .unwrap();

        let mut body = r.get("ardh").unwrap().body.clone();
        body["name"] = json!("Low Fen");
        r.put("ardh", body).unwrap();

        let after = std::fs::read_to_string(dir.join("ardh.yaml")).unwrap();
        // The proof that the file was the base: this comment exists nowhere in
        // the registry's memory, and it is still here.
        assert!(after.contains("# written by hand"), "{after}");
        assert!(after.contains("name: Low Fen"), "{after}");
        // And the other half of the same fact. `mood` is absent from the body,
        // and a PUT *replaces* the document — so the key goes. That is the rule
        // the console relies on to remove a field, and it cannot be told apart
        // here from a field the caller had never heard of. A caller that means
        // "change only this" needs a PATCH, which these routes do not offer.
        assert!(
            !after.contains("mood: bleak"),
            "a field absent from the body survived:\n{after}"
        );
        // The comment that introduced it **stays**, which is deliberate rather
        // than an oversight. A person wrote that line; it may explain why the
        // field went away, or be about to introduce its replacement. Deleting
        // somebody's prose because a neighbouring key left is the destructive
        // reading of "replace the document", and this is the one place where
        // the conservative answer is also the cheaper one.
        assert!(
            after.contains("# and a new field"),
            "a comment was deleted with its key:\n{after}"
        );
    }

    #[test]
    fn delete_removes_from_both_disk_and_memory() {
        let dir = tmp();
        let mut r = Registry::load("world", &dir).unwrap();
        r.put("ardh", json!({"name": "Ardh"})).unwrap();
        assert!(r.delete("ardh").unwrap());
        assert!(r.get("ardh").is_none());
        assert!(!dir.join("ardh.yaml").exists());
        // Deleting again is not an error.
        assert!(!r.delete("ardh").unwrap());
    }

    #[test]
    fn listing_order_is_stable() {
        let dir = tmp();
        let mut r = Registry::load("world", &dir).unwrap();
        for id in ["zeta", "alpha", "mid"] {
            r.put(id, json!({"name": id})).unwrap();
        }
        let ids: Vec<_> = Registry::load("world", &dir)
            .unwrap()
            .iter()
            .map(|x| x.id.clone())
            .collect();
        assert_eq!(ids, ["alpha", "mid", "zeta"]);
    }
}
