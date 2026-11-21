//! Turns a compaction pass's chunk moves into new identities for every holder, and
//! installs them.
//!
//! A chunk's gid carries a backing pointer to the refcount table of the arena it
//! came from, so renumbering the id alone would corrupt refcounts. The map therefore
//! carries a whole replacement gid. The rewrite is keyed on the allocation, since
//! every sharing path shares one allocation rather than cloning gids. A rewritten
//! chunk gets a new record built from its new gids, and the old record stays as it
//! is for whoever still holds it, because a shared record rewritten in place
//! corrupted K/V.
//!
//! - `compaction_map`: where each relocated chunk and record went.
//! - `rewrite`: rewriting one sealed payload through the map.
//! - `sweep`: the state one sweep carries across all holders.
//! - Tests: `tests_compact`.

use ahash::{AHashMap, AHashSet};

use super::arena_set::ArenaSet;
#[cfg(feature = "cuda")]
use super::backing::ChunkedKvBacking;
#[cfg(feature = "cuda")]
use super::compact_mint::{RecordInputs, RecordMint};
use super::gid_pool::ChunkGid;
#[cfg(feature = "cuda")]
use super::gpu_chunks::ChunkPin;
use super::head_gids::HeadGids;
#[cfg(feature = "cuda")]
use super::meta_pool::MetaGid;
use super::types::{SealedChunk, SealedSequence};
use crate::kv_cache::ArenaLocation;

mod compaction_map;
mod rewrite;
mod sweep;
#[cfg(test)]
mod tests_compact;

pub use compaction_map::*;
pub use rewrite::*;
pub use sweep::*;
