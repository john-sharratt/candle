//! Moves recurrent state halves into lower slots, so the state arenas can hand
//! regions back to the span.
//!
//! Compaction runs between steps and never on a store with a wave open, because
//! an open wave has already resolved the old addresses into its pointer tables. The
//! planned moves are carried out and their destinations claimed. A source that
//! nothing reaches keeps its slot, and its claimed destination goes back unused.
//! A moved half is read at its new address by the next forward.

use super::*;

/// What one recurrent-state compaction did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentCompaction {
    /// Moves the pass planned, each with its destination claimed.
    pub planned: usize,
    /// Halves that moved onto their destination. Lower than `planned` only when a
    /// source was held by nothing the pass could reach — a handle that outlived
    /// its store — whose claimed destination simply goes back.
    pub moved: usize,
    /// Regions every state arena on the device held before the pass and after it.
    pub regions_before: usize,
    pub regions_after: usize,
}

impl RecurrentCompaction {
    /// Regions the pass handed back to the span.
    pub fn regions_released(&self) -> usize {
        self.regions_before.saturating_sub(self.regions_after)
    }
}

/// Compact the state arenas `stores` live in: plan the two-cursor pass for their
/// geometry and move every half whose slot is a source onto its destination.
///
/// **Between steps, and never a store with a wave open.** A relocation rebuilds a
/// half's tensors on a new slot, and an open wave has already resolved the old
/// addresses into its pointer tables, so its stores keep their slots; between
/// steps nothing else holds one — the tables are rebuilt every forward — so a moved
/// half is simply read at its new address by the next forward.
///
/// Every store that could hold a source must be in `stores`: a source left behind
/// keeps its slot, and its claimed destination goes back unused. That is safe, but it
/// is reclaim lost — `moved` falling short of `planned` is how it shows.
pub fn compact_stores<'a>(
    stores: impl IntoIterator<Item = &'a mut RecurrentStateStore>,
    dims: &DeltaNetDims,
    device: &Device,
    max_moves: usize,
) -> Result<RecurrentCompaction> {
    #[cfg(feature = "cuda")]
    {
        if !matches!(device, Device::Cuda(_)) {
            return Ok(RecurrentCompaction::default());
        }
        // A store with a wave open keeps its slots: the wave has resolved them.
        // Every other store moves, its pending timeline rows with it.
        let stores: Vec<&mut RecurrentStateStore> =
            stores.into_iter().filter(|s| !s.open).collect();
        let regions_before = arena_regions(device, SlotTenant::RecurrentState);
        let moves = plan_slot_moves(
            device,
            SlotTenant::RecurrentState,
            state_block(dims).bytes,
            max_moves,
        )?;
        let planned = moves.len();
        let mut by_source: HashMap<u64, ArenaSlot> =
            moves.into_iter().map(|m| (m.src, m.dst)).collect();
        let mut moved = 0usize;
        for store in stores {
            moved += store.relocate(&mut by_source)?;
        }
        // Destinations whose source nothing here held go back to their arenas.
        drop(by_source);
        Ok(RecurrentCompaction {
            planned,
            moved,
            regions_before,
            regions_after: arena_regions(device, SlotTenant::RecurrentState),
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (stores, dims, device, max_moves);
        Ok(RecurrentCompaction::default())
    }
}
