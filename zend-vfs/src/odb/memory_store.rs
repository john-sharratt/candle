//! A [`PackStore`] in memory, for tests: what the substrate keeps, without one.

use std::collections::BTreeMap;
use std::sync::Mutex;

use sha2::{Digest, Sha256};

use super::store::{KeptPack, PackId, PackStore};
use crate::error::GitError;
use crate::types::{Oid, RefName};

/// Packs and kept refs in maps. `hidden` packs are named and not yet arrived,
/// as replication delivering the entry before its blobs would leave them.
#[derive(Default)]
pub(crate) struct MemoryStore {
    pub(crate) packs: Mutex<BTreeMap<PackId, KeptPack>>,
    pub(crate) hidden: Mutex<Vec<PackId>>,
    pub(crate) refs: Mutex<BTreeMap<RefName, Oid>>,
}

impl PackStore for MemoryStore {
    fn packs(&self) -> Result<Vec<PackId>, GitError> {
        let hidden = self.hidden.lock().unwrap();
        Ok(self
            .packs
            .lock()
            .unwrap()
            .keys()
            .filter(|id| !hidden.contains(id))
            .cloned()
            .collect())
    }

    fn pack(&self, id: &PackId) -> Result<Vec<u8>, GitError> {
        self.packs
            .lock()
            .unwrap()
            .get(id)
            .map(|p| p.pack.clone())
            .ok_or_else(|| GitError::Packs(format!("no pack {id}")))
    }

    fn index(&self, id: &PackId) -> Result<Vec<u8>, GitError> {
        self.packs
            .lock()
            .unwrap()
            .get(id)
            .map(|p| p.index.clone())
            .ok_or_else(|| GitError::Packs(format!("no pack {id}")))
    }

    fn put(&self, pack: KeptPack) -> Result<PackId, GitError> {
        let id: PackId = Sha256::digest(&pack.pack)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        self.packs.lock().unwrap().insert(id.clone(), pack);
        Ok(id)
    }

    fn kept_refs(&self) -> Result<Vec<(RefName, Oid)>, GitError> {
        Ok(self
            .refs
            .lock()
            .unwrap()
            .iter()
            .map(|(n, o)| (n.clone(), o.clone()))
            .collect())
    }

    fn keep_ref(&self, name: &RefName, target: Option<&Oid>) -> Result<(), GitError> {
        let mut refs = self.refs.lock().unwrap();
        match target {
            Some(oid) => refs.insert(name.clone(), oid.clone()),
            None => refs.remove(name),
        };
        Ok(())
    }
}
