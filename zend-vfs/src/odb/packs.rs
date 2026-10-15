//! The packs a repository is read from: every held pack's index, and the
//! bytes of those read recently, within a budget (`docs/substrate_git.md`
//! §4.1).
//!
//! An object is found by asking each pack's index in turn; a pack's bytes are
//! read from the store when an object in it is first wanted, and the least
//! recently read pack is let go once the budget is passed.

use std::collections::HashSet;
use std::sync::Arc;

use super::pack::base_cache::BaseCache;
use super::pack::idx::{starts_with, PackIndex};
use super::pack::reader::{offset_in, Pack};
use super::pack::{Object, ObjectType, RawId};
use super::store::{PackId, PackStore};
use crate::error::GitError;

/// What one pack's decoded delta bases may hold.
const BASES_PER_PACK: usize = 16 * 1024 * 1024;

/// One held pack.
struct Held {
    id: PackId,
    index: PackIndex,
    /// Its bytes, when read and not since let go.
    bytes: Option<Arc<Vec<u8>>>,
    bases: BaseCache,
    /// When its bytes were last used, on the set's clock.
    used: u64,
    /// Held in memory only, never in the store, while it is split
    /// (`super::split`): its bytes are never let go, and it is no pack the
    /// database reads from.
    transient: bool,
}

/// What a refresh took in.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Refreshed {
    /// Packs taken in.
    pub added: usize,
    /// Packs the store holds whose index could not be read, with why.
    pub failed: Vec<(PackId, String)>,
}

/// A repository's packs.
pub(crate) struct Packs {
    store: Arc<dyn PackStore>,
    held: Vec<Held>,
    /// What the packs' bytes may hold in memory together.
    budget: usize,
    clock: u64,
}

fn no_external(_: &RawId) -> Result<Option<Arc<Object>>, GitError> {
    Ok(None)
}

fn no_external_header(_: &RawId) -> Result<Option<(ObjectType, u64)>, GitError> {
    Ok(None)
}

impl Packs {
    /// The packs `store` holds now, their bytes kept within `budget`.
    pub(crate) fn open(store: Arc<dyn PackStore>, budget: usize) -> Result<Self, GitError> {
        let mut packs = Self {
            store,
            held: Vec::new(),
            budget,
            clock: 0,
        };
        let refreshed = packs.refresh()?;
        for (id, why) in &refreshed.failed {
            tracing::error!("pack {id} cannot be read: {why}");
        }
        Ok(packs)
    }

    /// Take in every pack the store holds that this set does not. A pack
    /// whose index cannot be read is reported and passed over, never the
    /// refresh's failure: every other pack is still taken, and the next
    /// refresh tries it again.
    pub(crate) fn refresh(&mut self) -> Result<Refreshed, GitError> {
        let known: HashSet<PackId> = self.held.iter().map(|h| h.id.clone()).collect();
        let mut refreshed = Refreshed::default();
        for id in self.store.packs()? {
            if known.contains(&id) {
                continue;
            }
            let index = match self.store.index(&id).and_then(PackIndex::parse) {
                Ok(index) => index,
                Err(e) => {
                    refreshed.failed.push((id, e.to_string()));
                    continue;
                }
            };
            self.held.push(Held {
                id,
                index,
                bytes: None,
                bases: BaseCache::new(BASES_PER_PACK),
                used: 0,
                transient: false,
            });
            refreshed.added += 1;
        }
        Ok(refreshed)
    }

    /// A pack just written, with its bytes already in hand.
    pub(crate) fn add(&mut self, id: PackId, index: PackIndex, bytes: Vec<u8>) {
        self.hold(id, index, bytes, false);
    }

    /// A pack kept in memory only, read from like any other until
    /// [`Self::drop_transient`]: one too large for the store, while it is
    /// split into packs that fit.
    pub(crate) fn add_transient(&mut self, id: PackId, index: PackIndex, bytes: Vec<u8>) {
        self.hold(id, index, bytes, true);
    }

    /// Let go of the transient pack `id`.
    pub(crate) fn drop_transient(&mut self, id: &PackId) {
        self.held.retain(|h| !(h.transient && &h.id == id));
    }

    fn hold(&mut self, id: PackId, index: PackIndex, bytes: Vec<u8>, transient: bool) {
        if self.held.iter().any(|h| h.id == id) {
            return;
        }
        self.clock += 1;
        self.held.push(Held {
            id,
            index,
            bytes: Some(Arc::new(bytes)),
            bases: BaseCache::new(BASES_PER_PACK),
            used: self.clock,
            transient,
        });
        self.trim(self.held.len() - 1);
    }

    /// Which pack holds `id`, and where in it.
    fn locate(&self, id: &RawId) -> Option<(usize, u64)> {
        self.held
            .iter()
            .enumerate()
            .find_map(|(p, h)| offset_in(&h.index, id).map(|at| (p, at)))
    }

    /// Whether any pack holds `id`.
    pub(crate) fn contains(&self, id: &RawId) -> bool {
        self.locate(id).is_some()
    }

    /// The bytes of pack `p`, read from the store if let go.
    fn bytes(&mut self, p: usize) -> Result<Arc<Vec<u8>>, GitError> {
        self.clock += 1;
        self.held[p].used = self.clock;
        if let Some(bytes) = &self.held[p].bytes {
            return Ok(Arc::clone(bytes));
        }
        let bytes = Arc::new(self.store.pack(&self.held[p].id)?);
        self.held[p].bytes = Some(Arc::clone(&bytes));
        self.trim(p);
        Ok(bytes)
    }

    /// Let go of the least recently used packs' bytes, never `keep`'s, until
    /// the rest fit the budget.
    fn trim(&mut self, keep: usize) {
        loop {
            let total: usize = self
                .held
                .iter()
                .filter_map(|h| h.bytes.as_ref().map(|b| b.len()))
                .sum();
            if total <= self.budget {
                return;
            }
            let oldest = self
                .held
                .iter()
                .enumerate()
                .filter(|(p, h)| *p != keep && !h.transient && h.bytes.is_some())
                .min_by_key(|(_, h)| h.used)
                .map(|(p, _)| p);
            let Some(p) = oldest else {
                return;
            };
            self.held[p].bytes = None;
            self.held[p].bases = BaseCache::new(BASES_PER_PACK);
        }
    }

    /// The object `id`, when a pack holds it.
    pub(crate) fn read(&mut self, id: &RawId) -> Result<Option<Arc<Object>>, GitError> {
        let Some((p, at)) = self.locate(id) else {
            return Ok(None);
        };
        let bytes = self.bytes(p)?;
        let held = &mut self.held[p];
        let index = &held.index;
        let find = |id: &RawId| offset_in(index, id);
        let pack = Pack {
            bytes: &bytes,
            find: &find,
        };
        pack.read_at(at, &mut held.bases, &no_external).map(Some)
    }

    /// The kind and size of `id`, when a pack holds it.
    pub(crate) fn header(&mut self, id: &RawId) -> Result<Option<(ObjectType, u64)>, GitError> {
        let Some((p, at)) = self.locate(id) else {
            return Ok(None);
        };
        let bytes = self.bytes(p)?;
        let index = &self.held[p].index;
        let find = |id: &RawId| offset_in(index, id);
        let pack = Pack {
            bytes: &bytes,
            find: &find,
        };
        pack.header_at(at, &no_external_header).map(Some)
    }

    /// Every id whose first `hex` hex digits are `prefix`'s, each once.
    pub(crate) fn prefixed(&self, prefix: &RawId, hex: usize) -> Vec<RawId> {
        let mut out: Vec<RawId> = Vec::new();
        for h in &self.held {
            for i in h.index.prefixed(prefix, hex) {
                let id = h.index.id(i);
                if starts_with(&id, prefix, hex) && !out.contains(&id) {
                    out.push(id);
                }
            }
        }
        out
    }

    /// Every id every pack holds; an id two packs hold, twice.
    pub(crate) fn ids(&self) -> impl Iterator<Item = RawId> + '_ {
        self.held.iter().flat_map(|h| h.index.ids())
    }

    /// The ids of every pack the store holds and this set reads from.
    pub(crate) fn pack_ids(&self) -> Vec<PackId> {
        self.held
            .iter()
            .filter(|h| !h.transient)
            .map(|h| h.id.clone())
            .collect()
    }

    /// The store the packs are read from.
    pub(crate) fn store(&self) -> &Arc<dyn PackStore> {
        &self.store
    }

    /// How many bytes of pack data are held in memory.
    #[cfg(test)]
    pub(crate) fn resident(&self) -> usize {
        self.held
            .iter()
            .filter_map(|h| h.bytes.as_ref().map(|b| b.len()))
            .sum()
    }
}

#[cfg(test)]
mod tests;
