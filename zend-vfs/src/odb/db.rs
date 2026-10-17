//! The object database itself: one repository's packs, pending objects, refs
//! and settings, shared by every handle to it (`docs/substrate_git.md` §4).

use std::fmt;
use std::mem;
use std::sync::{Arc, Mutex, MutexGuard};

use super::pack::idx::{starts_with, PackIndex};
use super::pack::indexer::index_pack;
use super::pack::{Object, ObjectType, RawId};
use super::packs::{Packs, Refreshed};
use super::pending::Pending;
use super::refs::Refs;
use super::settings::Settings;
use super::store::{KeptPack, PackId, PackStore};
use crate::error::GitError;

/// What the packs' bytes may hold in memory together, unless the caller says
/// otherwise: candle's whole history fits.
pub const DEFAULT_PACK_BUDGET: usize = 1024 * 1024 * 1024;

/// The largest pack written: one relocation stripe of the log, and 64
/// replication fragments (`docs/substrate_git.md` §3).
pub const MAX_PACK_BYTES: usize = 64 * 1024 * 1024;

pub(crate) struct Inner {
    pub(crate) name: String,
    pub(crate) packs: Mutex<Packs>,
    pub(crate) pending: Mutex<Pending>,
    pub(crate) refs: Mutex<Refs>,
    pub(crate) settings: Mutex<Settings>,
    /// Packs written since a ref last named what it needs.
    unannounced: Mutex<Vec<PackId>>,
    /// The largest pack kept: [`MAX_PACK_BYTES`], or a test's smaller one.
    max_pack: usize,
}

/// One repository held in process, read from a [`PackStore`]. Cloning shares
/// it.
#[derive(Clone)]
pub struct ObjectDb {
    pub(crate) inner: Arc<Inner>,
}

impl fmt::Debug for ObjectDb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ObjectDb({})", self.inner.name)
    }
}

fn locked<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ObjectDb {
    /// Repository `name`, read from `store`, its packs' bytes kept within
    /// `budget`.
    pub fn open(name: &str, store: Arc<dyn PackStore>, budget: usize) -> Result<Self, GitError> {
        Self::open_limited(name, store, budget, MAX_PACK_BYTES)
    }

    /// [`Self::open`], keeping no pack larger than `max_pack`: a test's
    /// limit, small enough to cross.
    pub(crate) fn open_limited(
        name: &str,
        store: Arc<dyn PackStore>,
        budget: usize,
        max_pack: usize,
    ) -> Result<Self, GitError> {
        Ok(Self {
            inner: Arc::new(Inner {
                name: name.to_string(),
                packs: Mutex::new(Packs::open(Arc::clone(&store), budget)?),
                pending: Mutex::new(Pending::default()),
                refs: Mutex::new(Refs::open(store)?),
                settings: Mutex::new(Settings::default()),
                unannounced: Mutex::new(Vec::new()),
                max_pack,
            }),
        })
    }

    /// The largest pack this database keeps.
    pub(crate) fn max_pack(&self) -> usize {
        self.inner.max_pack
    }

    /// The repository's name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub(crate) fn packs(&self) -> MutexGuard<'_, Packs> {
        locked(&self.inner.packs)
    }

    pub(crate) fn pending(&self) -> MutexGuard<'_, Pending> {
        locked(&self.inner.pending)
    }

    pub(crate) fn refs(&self) -> MutexGuard<'_, Refs> {
        locked(&self.inner.refs)
    }

    pub(crate) fn settings(&self) -> MutexGuard<'_, Settings> {
        locked(&self.inner.settings)
    }

    /// The object `id`: pending, or in a pack.
    pub(crate) fn read(&self, id: &RawId) -> Result<Option<Arc<Object>>, GitError> {
        let pending = self.pending().get(id);
        match pending {
            Some(object) => Ok(Some(object)),
            None => self.packs().read(id),
        }
    }

    /// The kind and size of `id`.
    pub(crate) fn header(&self, id: &RawId) -> Result<Option<(ObjectType, u64)>, GitError> {
        let pending = self.pending().get(id);
        match pending {
            Some(object) => Ok(Some((object.kind, object.data.len() as u64))),
            None => self.packs().header(id),
        }
    }

    /// Whether the database holds `id`.
    ///
    /// The two locks are taken one after the other, never one inside the
    /// other: [`Self::drop_packed`] holds the packs' while it takes the
    /// pending objects', and the reverse order here would deadlock against it.
    pub(crate) fn contains(&self, id: &RawId) -> bool {
        let pending = self.pending().get(id).is_some();
        pending || self.packs().contains(id)
    }

    /// Every id whose first `hex` hex digits are `prefix`'s, each once.
    pub(crate) fn prefixed(&self, prefix: &RawId, hex: usize) -> Vec<RawId> {
        let mut found = self.packs().prefixed(prefix, hex);
        let pending = self.pending().ids();
        for id in pending {
            if starts_with(&id, prefix, hex) && !found.contains(&id) {
                found.push(id);
            }
        }
        found
    }

    /// Hold `object` until it is packed; its id.
    pub(crate) fn write(&self, object: Object) -> RawId {
        self.pending().insert(object)
    }

    /// Take in every pack the store holds that this database does not, and
    /// name each whose index cannot be read.
    pub fn refresh(&self) -> Result<Refreshed, GitError> {
        let refreshed = self.packs().refresh()?;
        if refreshed.added > 0 {
            self.drop_packed();
        }
        Ok(refreshed)
    }

    /// The ids of every pack the database reads from.
    pub fn held_packs(&self) -> Vec<PackId> {
        self.packs().pack_ids()
    }

    /// Let go of every pending object a pack now holds.
    fn drop_packed(&self) {
        let packs = self.packs();
        self.pending().drop_packed(|id| packs.contains(id));
    }

    /// Index `pack` (a thin pack's bases taken from this database and
    /// appended), keep it in the store, and read from it: the ids of the packs
    /// kept (`docs/substrate_git.md` §4.2). One larger than the log keeps is
    /// kept as several that fit ([`Self::split`]), whoever made it.
    pub fn take_pack(&self, pack: Vec<u8>) -> Result<Vec<PackId>, GitError> {
        let external = |id: &RawId| self.read(id);
        let indexed = index_pack(pack, &external)?;
        let index = PackIndex::parse(indexed.index.clone())?;
        if indexed.pack.len() > self.max_pack() {
            let transient = format!("transient-{}", indexed.checksum);
            return self.split(transient, indexed.pack, index);
        }
        let id = self.inner_store().put(KeptPack {
            pack: indexed.pack.clone(),
            index: indexed.index,
            checksum: indexed.checksum.to_string(),
            objects: indexed.objects,
        })?;
        self.packs().add(id.clone(), index, indexed.pack);
        self.drop_packed();
        locked(&self.inner.unannounced).push(id.clone());
        Ok(vec![id])
    }

    fn inner_store(&self) -> Arc<dyn PackStore> {
        Arc::clone(self.packs().store())
    }

    /// The packs written since this was last asked: what the next ref
    /// published needs (`docs/substrate_git.md` §4.4).
    pub fn take_unannounced(&self) -> Vec<PackId> {
        mem::take(&mut *locked(&self.inner.unannounced))
    }

    /// How many objects are written and not yet packed.
    pub fn pending_objects(&self) -> usize {
        self.pending().len()
    }
}
