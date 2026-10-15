//! Pending objects packed and kept: what a publish does before it names a ref
//! (`docs/substrate_git.md` §4.4).
//!
//! libgit2's packbuilder writes the pack, with deltas among the objects, and
//! the database takes it like any other. A pack over the limit is split in
//! two, each half packed again; an object that cannot fit a pack alone is
//! refused rather than written.

use git2::{Buf, Oid as LibOid, Repository};

use super::db::ObjectDb;
use super::pack::RawId;
use super::repository::repository;
use super::store::PackId;
use crate::error::GitError;

fn lib_error(what: &str, e: git2::Error) -> GitError {
    GitError::Packs(format!("libgit2 {what}: {}", e.message()))
}

/// A pack of exactly `ids`, as libgit2's packbuilder writes it.
fn build(repo: &Repository, ids: &[RawId]) -> Result<Vec<u8>, GitError> {
    let mut builder = repo
        .packbuilder()
        .map_err(|e| lib_error("packbuilder", e))?;
    for id in ids {
        let oid = LibOid::from_bytes(&id.0).map_err(|e| lib_error("oid", e))?;
        builder
            .insert_object(oid, None)
            .map_err(|e| lib_error("packbuilder_insert", e))?;
    }
    let mut buf = Buf::new();
    builder
        .write_buf(&mut buf)
        .map_err(|e| lib_error("packbuilder_write_buf", e))?;
    Ok(buf.to_vec())
}

/// Pack `ids` into packs of at most the database's limit (`MAX_PACK_BYTES`)
/// each, keeping each.
pub(super) fn pack_into(
    db: &ObjectDb,
    repo: &Repository,
    ids: &[RawId],
    kept: &mut Vec<PackId>,
) -> Result<(), GitError> {
    let pack = build(repo, ids)?;
    if pack.len() <= db.max_pack() {
        kept.extend(db.take_pack(pack)?);
        return Ok(());
    }
    if ids.len() == 1 {
        return Err(GitError::Packs(format!(
            "object {} packs to {} bytes, over the {}-byte limit a pack keeps to",
            ids[0],
            pack.len(),
            db.max_pack()
        )));
    }
    let (a, b) = ids.split_at(ids.len() / 2);
    pack_into(db, repo, a, kept)?;
    pack_into(db, repo, b, kept)
}

impl ObjectDb {
    /// Pack every pending object and keep the packs; their ids. Nothing
    /// pending, nothing written.
    pub fn flush(&self) -> Result<Vec<PackId>, GitError> {
        let mut ids = self.pending().ids();
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        ids.sort();
        let repo = repository(self)?;
        let mut kept = Vec::new();
        pack_into(self, &repo, &ids, &mut kept)?;
        Ok(kept)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use git2::{Signature, Time};

    use super::*;
    use crate::odb::db::DEFAULT_PACK_BUDGET;
    use crate::odb::memory_store::MemoryStore;
    use crate::odb::store::PackStore;

    /// **A flush packs what was written, and a fresh database over the same
    /// store reads it**: nothing pending afterwards, one pack written, its
    /// objects there for a member that never held them in memory.
    #[test]
    fn a_flush_packs_what_was_written_for_any_member_to_read() {
        let store: Arc<dyn PackStore> = Arc::new(MemoryStore::default());
        let db = ObjectDb::open("r", Arc::clone(&store), DEFAULT_PACK_BUDGET).unwrap();
        let repo = repository(&db).unwrap();
        let blob = repo.blob(b"flushed\n").unwrap();
        let mut tree = repo.treebuilder(None).unwrap();
        tree.insert("f.txt", blob, 0o100644).unwrap();
        let tree = repo.find_tree(tree.write().unwrap()).unwrap();
        let sig = Signature::new("Ada", "ada@example.com", &Time::new(1_700_000_000, 0)).unwrap();
        let commit = repo
            .commit(None, &sig, &sig, "flush\n", &tree, &[])
            .unwrap();
        assert_eq!(db.pending_objects(), 3);

        let kept = db.flush().unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(db.pending_objects(), 0);
        assert_eq!(db.take_unannounced(), kept);
        assert!(db.flush().unwrap().is_empty(), "nothing left to pack");

        let other = ObjectDb::open("r", store, DEFAULT_PACK_BUDGET).unwrap();
        let read = repository(&other).unwrap();
        assert_eq!(read.find_commit(commit).unwrap().message(), Some("flush\n"));
        assert_eq!(read.find_blob(blob).unwrap().content(), b"flushed\n");
    }
}
