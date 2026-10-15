//! A pack larger than the log keeps, kept as several that fit
//! (`docs/substrate_git.md` §4.2).
//!
//! An answer to a fetch is one pack, and the agent that cut it bounds it — or
//! does not: an agent that ignores the bound, or a single commit whose objects
//! outgrow it, hands over more than one record of the log may hold. The pack
//! is held in memory only, read from like any other, while its objects are
//! packed again in parts, each kept; then it is let go. Nothing larger than
//! the limit is ever written.

use super::db::ObjectDb;
use super::flush::pack_into;
use super::pack::idx::PackIndex;
use super::pack::RawId;
use super::repository::repository;
use super::store::PackId;
use crate::error::GitError;

impl ObjectDb {
    /// Keep the indexed pack `bytes` as packs within the database's limit
    /// (`MAX_PACK_BYTES`); their ids. `transient` names it while it is held.
    /// Each part aims at half the limit, so a part the packbuilder writes
    /// larger than its share still fits; one that does not is halved again.
    pub(super) fn split(
        &self,
        transient: PackId,
        bytes: Vec<u8>,
        index: PackIndex,
    ) -> Result<Vec<PackId>, GitError> {
        let ids: Vec<RawId> = index.ids().collect();
        let parts = bytes.len().div_ceil(self.max_pack() / 2).max(1);
        let per_part = ids.len().div_ceil(parts).max(1);
        self.packs().add_transient(transient.clone(), index, bytes);
        let kept = self.pack_parts(&ids, per_part);
        self.packs().drop_transient(&transient);
        kept
    }

    fn pack_parts(&self, ids: &[RawId], per_part: usize) -> Result<Vec<PackId>, GitError> {
        let repo = repository(self)?;
        let mut kept = Vec::new();
        for part in ids.chunks(per_part) {
            pack_into(self, &repo, part, &mut kept)?;
        }
        Ok(kept)
    }
}

#[cfg(test)]
mod tests;
