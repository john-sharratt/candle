//! The object database a repository read from the substrate is held in
//! (`docs/substrate_git.md` §4): objects read from the log's packs, refs and
//! settings in memory, and libgit2 reading and writing through both. No
//! folder, and no `git` program.
//!
//! | Module | Concern |
//! |---|---|
//! | [`pack`] | Git's pack format: entries, deltas, indexes, the indexer |
//! | [`store`] | Where the packs are kept: the trait the substrate implements |
//! | [`packs`] | The held packs, their bytes within a budget |
//! | [`pending`] | Objects made here and not yet packed |
//! | [`refs`] | The refs, in memory; a few kept in the store |
//! | [`settings`] | Remotes and upstreams, in memory |
//! | [`db`] | [`ObjectDb`]: all of the above for one repository |
//! | [`backend`] | The `git_odb_backend` libgit2 reads objects through |
//! | [`refdb`] | The `git_refdb_backend` libgit2 reads refs through |
//! | [`repository`] | A libgit2 repository over an [`ObjectDb`] |
//! | [`flush`] | Pending objects packed and kept |
//! | [`split`] | A pack too large to keep, kept as several that fit |
//! | [`bundle`] | Bundles cut for a push, and taken from a fetch |

mod backend;
pub(crate) mod bundle;
mod db;
mod flush;
#[cfg(test)]
pub(crate) mod memory_store;
pub(crate) mod pack;
mod packs;
mod pending;
mod refdb;
pub(crate) mod refs;
pub(crate) mod repository;
pub(crate) mod settings;
mod split;
mod store;

pub use db::{ObjectDb, DEFAULT_PACK_BUDGET, MAX_PACK_BYTES};
pub use packs::Refreshed;
pub use refs::STORE_HEAD;
pub use store::{KeptPack, PackId, PackStore, KEPT_REFS};
