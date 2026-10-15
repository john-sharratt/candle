//! Where a repository's packs are kept: the substrate's log, behind a trait so
//! this layer names no substrate (`docs/substrate_git.md` §4.1).
//!
//! `substrate-git` implements it over its records; a test, over a map.

use crate::error::GitError;
use crate::types::{Oid, RefName};

/// A pack's name in its store: the SHA-256 of its bytes, in hex.
pub type PackId = String;

/// A pack and its index, as kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeptPack {
    pub pack: Vec<u8>,
    /// Its `.idx`, version 2.
    pub index: Vec<u8>,
    /// Its trailing checksum, in hex: the name git gives it.
    pub checksum: String,
    /// How many objects it holds.
    pub objects: u32,
}

/// One repository's packs, and the few refs that outlive a restart.
pub trait PackStore: Send + Sync {
    /// Every pack whose bytes and index have both arrived, by id. One the
    /// store names that has not arrived is left out until it has.
    fn packs(&self) -> Result<Vec<PackId>, GitError>;

    /// The bytes of the pack `id`.
    fn pack(&self, id: &PackId) -> Result<Vec<u8>, GitError>;

    /// The index of the pack `id`.
    fn index(&self, id: &PackId) -> Result<Vec<u8>, GitError>;

    /// Keep `pack`; its id.
    fn put(&self, pack: KeptPack) -> Result<PackId, GitError>;

    /// The refs kept in the store: those under [`KEPT_REFS`].
    fn kept_refs(&self) -> Result<Vec<(RefName, Oid)>, GitError>;

    /// Keep ref `name` at `target`, or forget it with `None`.
    fn keep_ref(&self, name: &RefName, target: Option<&Oid>) -> Result<(), GitError>;
}

/// The refs a repository read from the substrate keeps in its store rather
/// than in memory: the commits an unfinished sliced fetch has taken, so the
/// next lease holder — this member after a restart, or another — resumes from
/// them (`docs/substrate_git.md` §5.1).
pub const KEPT_REFS: &str = "refs/substrate-slices/";
