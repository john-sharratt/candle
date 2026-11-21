//! The handles one serialised chunk's slice header depends on, held alive for
//! as long as any launch reads them.
//!
//! A header names two addresses that live elsewhere: the chunk's own record,
//! and through it each band's allocation. Both handles are held, because
//! holding only the record left the bands unpinned under a running launch.
//! `describes` is the cheap check an update uses to keep the pin it already
//! has when the header still names the same band allocation and record slot.

use super::*;

/// Everything one serialised chunk's headers dereference, held alive for the life of
/// any launch that reads them.
///
/// Both halves are addresses in a slice header: the record's own (`kvheads_ptr`) and,
/// through it, each band's. So both handles have to be held — see [`GpuChunks::pins`],
/// where holding only the first cost a night.
#[derive(Clone, Debug)]
pub struct ChunkPin {
    /// The chunk's bands.
    pub(super) gids: HeadGids,
    /// The chunk's resident record, when it has one. `None` for a live writer window,
    /// whose heads are serialised inline and name no record slot.
    pub(super) meta: Option<MetaGid>,
}

impl ChunkPin {
    pub(super) fn of(chunk: &ChunkWindow) -> Self {
        Self {
            gids: chunk.gids.clone(),
            meta: chunk.meta.clone(),
        }
    }

    /// `HeadGids::alloc_id` of the bands this pin holds. Stable for as long as the pin
    /// lives, because the pin is what keeps the allocation alive.
    pub(crate) fn bands_alloc_id(&self) -> usize {
        self.gids.alloc_id()
    }

    /// Raw id of the record slot this pin holds, if the chunk has a record. Unique for
    /// as long as the pin lives, for the same reason.
    pub(crate) fn record_raw(&self) -> Option<i64> {
        self.meta.as_ref().map(MetaGid::raw)
    }

    /// Whether this pin already holds exactly what `chunk`'s header will name — its band
    /// allocation and its record slot — the cheap check `update_chunk` uses before
    /// replacing a pin.
    pub(super) fn describes(&self, chunk: &ChunkWindow) -> bool {
        self.gids.is_same_alloc(&chunk.gids)
            && self.record_raw() == chunk.meta.as_ref().map(MetaGid::raw)
    }
}
