//! One repository's packs, as the log keeps them, for the object database a
//! member reads it through (`docs/substrate_git.md` §4.1).
//!
//! A pack is held once both its blob and its index's have arrived; one the
//! log names that has not is left out until it has. The refs a sliced fetch
//! keeps are `git/slice/<repo>/<oid>` objects, so another member taking the
//! lease resumes from them.

use std::sync::Arc;

use zend_vfs::{GitError, KeptPack, Oid, PackId, PackStore, RefName, KEPT_REFS};

use crate::records::{Log, Records};

/// `repo`'s packs in `records`.
pub struct LogPacks {
    records: Arc<dyn Records>,
    repo: String,
}

impl LogPacks {
    pub fn new(records: Arc<dyn Records>, repo: &str) -> Self {
        Self {
            records,
            repo: repo.to_string(),
        }
    }

    fn log(&self) -> Result<Log<'_>, GitError> {
        Log::new(self.records.as_ref(), &self.repo).map_err(packs)
    }
}

fn packs(e: anyhow::Error) -> GitError {
    GitError::Packs(format!("{e:#}"))
}

impl PackStore for LogPacks {
    fn packs(&self) -> Result<Vec<PackId>, GitError> {
        let log = self.log()?;
        let mut held = Vec::new();
        for (sha, entry) in log.packs().map_err(packs)? {
            if log.arrived(&sha, &entry).map_err(packs)? {
                held.push(sha);
            }
        }
        Ok(held)
    }

    fn pack(&self, id: &PackId) -> Result<Vec<u8>, GitError> {
        self.log()?
            .pack_bytes(id)
            .map_err(packs)?
            .ok_or_else(|| GitError::Packs(format!("pack {id} of {} has not arrived", self.repo)))
    }

    fn index(&self, id: &PackId) -> Result<Vec<u8>, GitError> {
        let log = self.log()?;
        let entry = log.packs().map_err(packs)?.remove(id).ok_or_else(|| {
            GitError::Packs(format!("the log names no pack {id} of {}", self.repo))
        })?;
        log.index_bytes(&entry)
            .map_err(packs)?
            .ok_or_else(|| GitError::Packs(format!("the index of pack {id} has not arrived")))
    }

    fn put(&self, pack: KeptPack) -> Result<PackId, GitError> {
        self.log()?
            .put_pack(pack.pack, pack.index, &pack.checksum, pack.objects)
            .map_err(packs)
    }

    fn kept_refs(&self) -> Result<Vec<(RefName, Oid)>, GitError> {
        self.log()?
            .slices()
            .into_iter()
            .map(|oid| {
                Ok((
                    RefName::parse(&format!("{KEPT_REFS}{oid}"))?,
                    Oid::parse(&oid)?,
                ))
            })
            .collect()
    }

    fn keep_ref(&self, name: &RefName, target: Option<&Oid>) -> Result<(), GitError> {
        let oid = name
            .as_str()
            .strip_prefix(KEPT_REFS)
            .ok_or_else(|| GitError::invalid(format!("{name} is not a kept ref")))?;
        self.log()?.set_slice(oid, target.is_some()).map_err(packs)
    }
}
