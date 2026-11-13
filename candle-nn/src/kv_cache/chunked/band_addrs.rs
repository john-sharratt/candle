//! Band addresses for a walk over many bands under one storage read.
//!
//! A band's address is its arena's slot pointer for its chunk index. A walk
//! over a table — every band of every head of every block — meets the same few
//! arenas over and over, so looking each up in the arena map is the walk's
//! cost, not the arithmetic. This keeps the arenas it has met in a short list
//! that the next band checks first.

use candle::{Error, Result};

use super::arena::{Arena, ArenaStorageState};
use super::gid_pool::ChunkGid;

/// Arenas a walk remembers; a head's bands span at most a handful.
const REMEMBERED: usize = 8;

/// Resolves band addresses against one storage read, remembering the arenas
/// it has met.
pub(crate) struct BandAddrs<'s> {
    storage: &'s ArenaStorageState,
    met: Vec<(usize, &'s Arena)>,
}

impl<'s> BandAddrs<'s> {
    pub(crate) fn new(storage: &'s ArenaStorageState) -> Self {
        Self {
            storage,
            met: Vec::with_capacity(REMEMBERED),
        }
    }

    /// The device address of `gid`'s slot. `what` names the walk in an error.
    pub(crate) fn addr(&mut self, gid: &ChunkGid, what: &str) -> Result<u64> {
        let ai = gid.arena_idx();
        let arena = match self.met.iter().position(|&(idx, _)| idx == ai) {
            Some(at) => self.met[at].1,
            None => {
                let arena = self
                    .storage
                    .arenas()
                    .get(&ai)
                    .ok_or_else(|| Error::Msg(format!("{what} arena {ai} not found")))?;
                if self.met.len() == REMEMBERED {
                    self.met.remove(0);
                }
                self.met.push((ai, arena));
                arena
            }
        };
        arena
            .slot_ptr(gid.chunk_idx())
            .ok_or_else(|| Error::Msg(format!("{what} arena {ai} is not GPU-resident")))
    }
}
