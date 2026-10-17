//! The refs of a repository read from the substrate, held in memory
//! (`docs/substrate_git.md` §4.3).
//!
//! `HEAD` names `refs/heads/substrate-store`, which is never made. Every other
//! ref is direct or symbolic as git's are; those under [`KEPT_REFS`] are also
//! written to the store, so they outlive a restart. A ref libgit2 locks for a
//! transaction stays locked until the transaction ends.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use super::pack::RawId;
use super::store::{PackStore, KEPT_REFS};
use crate::error::GitError;
use crate::types::RefName;

/// What `HEAD` names: a branch never made, so nothing is checked out.
pub const STORE_HEAD: &str = "refs/heads/substrate-store";

/// What a ref holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RefValue {
    Direct(RawId),
    Symbolic(String),
}

/// A repository's refs.
pub(crate) struct Refs {
    map: BTreeMap<String, RefValue>,
    locked: HashSet<String>,
    store: Arc<dyn PackStore>,
}

impl Refs {
    /// `HEAD`, and the refs `store` keeps.
    pub(crate) fn open(store: Arc<dyn PackStore>) -> Result<Self, GitError> {
        let mut map = BTreeMap::new();
        map.insert(
            "HEAD".to_string(),
            RefValue::Symbolic(STORE_HEAD.to_string()),
        );
        for (name, oid) in store.kept_refs()? {
            map.insert(
                name.as_str().to_string(),
                RefValue::Direct(RawId::from_oid(&oid)?),
            );
        }
        Ok(Self {
            map,
            locked: HashSet::new(),
            store,
        })
    }

    /// What `name` holds.
    pub(crate) fn get(&self, name: &str) -> Option<&RefValue> {
        self.map.get(name)
    }

    /// Set `name` to `value`.
    pub(crate) fn set(&mut self, name: &str, value: RefValue) -> Result<(), GitError> {
        if name.starts_with(KEPT_REFS) {
            match &value {
                RefValue::Direct(id) => self
                    .store
                    .keep_ref(&RefName::parse(name)?, Some(&id.to_oid()))?,
                RefValue::Symbolic(_) => {
                    return Err(GitError::invalid(format!(
                        "{name} is kept, and cannot be symbolic"
                    )))
                }
            }
        }
        self.map.insert(name.to_string(), value);
        Ok(())
    }

    /// Remove `name`; whether it was there.
    pub(crate) fn delete(&mut self, name: &str) -> Result<bool, GitError> {
        if name.starts_with(KEPT_REFS) {
            self.store.keep_ref(&RefName::parse(name)?, None)?;
        }
        Ok(self.map.remove(name).is_some())
    }

    /// Lock `name` for a transaction; `false` when it already is.
    pub(crate) fn lock(&mut self, name: &str) -> bool {
        self.locked.insert(name.to_string())
    }

    /// Release `name`'s lock.
    pub(crate) fn unlock(&mut self, name: &str) {
        self.locked.remove(name);
    }

    /// Every ref whose name `glob` matches — every ref with none — by name.
    pub(crate) fn matching(&self, glob: Option<&str>) -> Vec<(String, RefValue)> {
        self.map
            .iter()
            .filter(|(name, _)| glob.is_none_or(|g| wildmatch(g.as_bytes(), name.as_bytes())))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }
}

/// Whether `name` matches `glob` as libgit2's ref globbing does: `*` any run
/// of characters, `/` included, `?` any one, anything else itself.
pub(crate) fn wildmatch(glob: &[u8], name: &[u8]) -> bool {
    match glob.split_first() {
        None => name.is_empty(),
        Some((b'*', rest)) => (0..=name.len()).any(|i| wildmatch(rest, &name[i..])),
        Some((b'?', rest)) => !name.is_empty() && wildmatch(rest, &name[1..]),
        Some((c, rest)) => name.first() == Some(c) && wildmatch(rest, &name[1..]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::odb::memory_store::MemoryStore;
    use crate::types::Oid;

    #[test]
    fn globs_match_as_libgit2s_do() {
        assert!(wildmatch(b"refs/heads/*", b"refs/heads/main"));
        assert!(wildmatch(b"refs/heads/*", b"refs/heads/zen/work"));
        assert!(!wildmatch(b"refs/heads/*", b"refs/tags/v1"));
        assert!(wildmatch(b"refs/*/ma?n", b"refs/heads/main"));
        assert!(wildmatch(b"HEAD", b"HEAD"));
        assert!(!wildmatch(b"HEAD", b"HEADS"));
    }

    /// **HEAD names a branch never made; kept refs outlive the set**: a
    /// slice ref is written to the store and read back by a fresh set, an
    /// ordinary ref is not.
    #[test]
    fn kept_refs_outlive_a_restart_and_others_do_not() {
        let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
        let mut refs = Refs::open(Arc::clone(&store)).unwrap();
        assert_eq!(
            refs.get("HEAD"),
            Some(&RefValue::Symbolic(STORE_HEAD.to_string()))
        );
        let id = RawId([3; 20]);
        refs.set("refs/substrate-slices/x", RefValue::Direct(id))
            .unwrap();
        refs.set("refs/heads/main", RefValue::Direct(id)).unwrap();
        let again = Refs::open(Arc::clone(&store)).unwrap();
        assert_eq!(
            again.get("refs/substrate-slices/x"),
            Some(&RefValue::Direct(id))
        );
        assert_eq!(again.get("refs/heads/main"), None);
        assert_eq!(
            store.kept_refs().unwrap(),
            vec![(
                RefName::parse("refs/substrate-slices/x").unwrap(),
                Oid::parse(&id.to_string()).unwrap()
            )]
        );
        refs.delete("refs/substrate-slices/x").unwrap();
        assert!(store.kept_refs().unwrap().is_empty());
    }

    #[test]
    fn a_ref_locks_once() {
        let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
        let mut refs = Refs::open(store).unwrap();
        assert!(refs.lock("refs/heads/a"));
        assert!(!refs.lock("refs/heads/a"));
        refs.unlock("refs/heads/a");
        assert!(refs.lock("refs/heads/a"));
    }
}
