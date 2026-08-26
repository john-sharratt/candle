//! Authored content that lives in the repository rather than in the substrate.
//!
//! Worlds and archetypes are **written**, not accumulated. They are the setting
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

pub mod id;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

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
    /// What this collection is called in errors and logs (`world`, `archetype`).
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

    /// Write a record to disk and into memory.
    ///
    /// Disk first: if the write fails, memory still matches what is on disk,
    /// and the caller gets an error instead of a registry that has quietly
    /// diverged from the repository it is supposed to be.
    pub fn put(&mut self, id: &str, body: Value) -> Result<()> {
        id::check(id).map_err(|e| anyhow::anyhow!("{}: {e}", self.kind))?;
        let path = self.path_for(id)?;

        let yaml = serde_yaml::to_string(&body)
            .with_context(|| format!("{}: serialising `{id}`", self.kind))?;

        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("creating {}", self.dir.display()))?;
        std::fs::write(&path, yaml)
            .with_context(|| format!("{}: writing {}", self.kind, path.display()))?;

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

    /// Remove a record from disk and memory. Absent is not an error — the
    /// caller asked for it to be gone and it is gone.
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        id::check(id).map_err(|e| anyhow::anyhow!("{}: {e}", self.kind))?;
        let path = self.path_for(id)?;
        let existed = self.items.remove(id).is_some();
        if path.exists() {
            std::fs::remove_file(&path)
                .with_context(|| format!("{}: removing {}", self.kind, path.display()))?;
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

    #[test]
    fn a_missing_directory_is_an_empty_registry_not_a_failure() {
        let r = Registry::load("world", tmp().join("does-not-exist")).unwrap();
        assert!(r.is_empty());
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
