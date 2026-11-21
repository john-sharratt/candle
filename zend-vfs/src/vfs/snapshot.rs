//! Saving a store's session state and restoring it exactly.
//!
//! A snapshot records every changed path's chain with each delta's moment, the
//! size the chain leaves the file at, whether it is in conflict, and the base it
//! is made on. Restoring replays nothing, so a restored store reads what the saved
//! one did, which is how a conversation's copy of a repository outlives the store
//! that held it. A snapshot naming a protected path, past the size cap, or with a
//! base that does not name objects is refused whole, and the store is left as it
//! was. A roll-back applies only if the store still holds the state the change
//! left, so a change that landed since is never undone by another.

use super::*;

/// A store's session state, saved: each changed path's chain — every delta
/// with the moment it was made, the size the chain leaves the file at (`None`
/// once deleted), and whether it is in conflict — and, over a branch, the
/// [`Base`] they are made on, so a restored conversation reads the commit it
/// read before however the branch has moved since.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) base: Option<SavedBase>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub(super) chains: BTreeMap<String, SavedChain>,
}

impl Snapshot {
    /// A snapshot of `base` and `chains` — what a saved form other than this
    /// one's own serialisation is read back into.
    pub fn new(base: Option<SavedBase>, chains: BTreeMap<String, SavedChain>) -> Self {
        Self { base, chains }
    }

    /// Whether there is nothing to save: no base taken, no path changed.
    pub fn is_empty(&self) -> bool {
        self.base.is_none() && self.chains.is_empty()
    }

    /// What the chains are made on; `None` before the store first read its
    /// branch.
    pub fn base(&self) -> Option<&SavedBase> {
        self.base.as_ref()
    }

    /// Every changed path's chain, by path.
    pub fn chains(&self) -> &BTreeMap<String, SavedChain> {
        &self.chains
    }
}

/// One path's saved chain — see [`Snapshot`]: every delta with the moment it
/// was made, the size it leaves the file at (`None` once deleted), and
/// whether it is in conflict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedChain {
    pub deltas: Vec<TimedDelta>,
    pub size: Option<usize>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub conflict: bool,
}

impl VfsStore {
    /// The session's state as it is saved: every changed path's chain, each
    /// delta with its moment, the size it leaves the file at and whether it
    /// is in conflict, with the base they are made on — what
    /// [`Self::restore`] puts back exactly, with nothing replayed.
    pub fn snapshot(&self) -> Snapshot {
        Self::saved(&self.upper.read().unwrap())
    }

    pub(super) fn saved(upper: &Upper) -> Snapshot {
        Snapshot {
            base: upper.base.as_ref().map(Base::saved),
            chains: upper
                .chains
                .iter()
                .map(|(path, chain)| {
                    let saved = SavedChain {
                        deltas: chain.deltas.clone(),
                        size: chain.size,
                        conflict: chain.conflict,
                    };
                    (path.clone(), saved)
                })
                .collect(),
        }
    }

    /// Put back what a [`Self::snapshot`] saved, replacing whatever the
    /// session layer holds — how a conversation's own copy of a repository
    /// outlives the store that held it, reading the base it read before. A
    /// protected path is refused, and so is a set past the size cap or a base
    /// that does not name objects; any of these leaves the store as it was.
    pub fn restore(&self, snapshot: Snapshot) -> Result<(), VfsError> {
        if snapshot.is_empty() {
            return Ok(());
        }
        let restored = Self::upper_of(snapshot)?;
        *self.upper.write().unwrap() = restored;
        Ok(())
    }

    /// Put back `before`, a [`Self::snapshot`] taken before a change, only if
    /// the store still holds exactly `after` — what that change left. Returns
    /// whether it did: a change of the conversation's that landed since is
    /// never undone by rolling back another.
    pub fn roll_back(&self, after: &Snapshot, before: Snapshot) -> Result<bool, VfsError> {
        let restored = Self::upper_of(before)?;
        let mut upper = self.upper.write().unwrap();
        if Self::saved(&upper) != *after {
            return Ok(false);
        }
        *upper = restored;
        Ok(true)
    }

    /// The session layer `snapshot` saved, checked: no protected path, within
    /// the size cap, a base naming objects.
    pub(super) fn upper_of(snapshot: Snapshot) -> Result<Upper, VfsError> {
        let base = snapshot.base.as_ref().map(SavedBase::parse).transpose()?;
        let mut chains = HashMap::with_capacity(snapshot.chains.len());
        for (path, saved) in snapshot.chains {
            let norm = Self::normalize(&path);
            Self::guard(&norm)?;
            chains.insert(
                norm,
                Chain {
                    deltas: saved.deltas,
                    size: saved.size,
                    conflict: saved.conflict,
                },
            );
        }
        let restored = Upper { chains, base };
        if restored.bytes() > MAX_BYTES {
            return Err(VfsError::Full);
        }
        Ok(restored)
    }

    /// Every change the session holds, path by path, made against its base's
    /// tree — what a checkout replays onto a repository's files to put this
    /// conversation on disk (see [`crate::checkout`]).
    pub fn changes(&self) -> Result<FileChanges, VfsError> {
        self.view()?;
        let upper = self.upper.read().unwrap();
        let mut changes = FileChanges::new();
        for (path, chain) in &upper.chains {
            for delta in &chain.deltas {
                changes.push_timed(path.clone(), delta.clone());
            }
        }
        Ok(changes)
    }
}
