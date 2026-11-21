//! Holds each recurrent layer's state slot and the pending timeline beside it,
//! and the rules for when a layer may be advanced.
//!
//! A half is one slot's state, conv tail and timeline, and every tensor built on
//! it keeps the slot alive. A layer that is already pending refuses a second span,
//! and the prefill scratch is a second slot kept only while the sequence prefills.
//! The block lays the state, the tail and the timeline on fixed boundaries, so the
//! kernels can address each part directly.

use super::*;

/// One slot's worth of a layer: the state and the timeline after it, and — on
/// CUDA — the state-arena slot they are views into.
pub(super) struct Half {
    pub(super) state: DeltaNetState,
    pub(super) timeline: Tensor,
    /// **Not the only holder, by design.** Every tensor built on the slot holds a
    /// share of it too (the views are anchored to it), as does every clone, view
    /// and re-lease of those tensors. A slot therefore goes back to its arena
    /// only when the last thing that could read it has gone — dropping the store
    /// is what normally ends that, but a handle taken along the way cannot end up
    /// reading a slot another sequence now holds. Held here so the store can move
    /// a half to another slot ([`RecurrentStateStore::relocate`]) and say what it
    /// costs without walking its tensors.
    ///
    /// `None` on a CPU device, where the buffers are ordinary allocations.
    #[cfg(feature = "cuda")]
    pub(super) held: Option<Arc<ArenaSlot>>,
}

impl Half {
    /// Buffers off any reservation — a CPU device, or a CUDA build running on
    /// one. A state that is read at zero (a sequence start) is zeroed; a scratch,
    /// which a long span overwrites whole before anything reads it, is not
    /// (hot-path invariant 6). Nor is the timeline: a fold reads only the rows a
    /// span wrote.
    pub(super) fn host(dims: &DeltaNetDims, device: &Device, zeroed: bool) -> Result<Self> {
        Ok(Self {
            state: if zeroed {
                DeltaNetState::zeros(dims, device)?
            } else {
                DeltaNetState::uninit(dims, device)?
            },
            timeline: TimelineLayout::of(dims).alloc(device)?,
            #[cfg(feature = "cuda")]
            held: None,
        })
    }

    /// The half living in `slot`, its tensors anchored to the slot so no view of
    /// them can outlive it.
    #[cfg(feature = "cuda")]
    pub(super) fn in_slot(
        dims: &DeltaNetDims,
        device: &Device,
        slot: Arc<ArenaSlot>,
    ) -> Result<Self> {
        let block = state_block(dims);
        debug_assert!(
            slot.stride() >= block.bytes,
            "a slot narrower than its half"
        );
        let lay = TimelineLayout::of(dims);
        // SAFETY: the slot spans at least `block.bytes` from its 256-aligned base —
        // `s` at the start, the conv tail at `conv_off`, the timeline at
        // `timeline_off` — and the anchors keep it held for as long as any view of
        // these tensors exists.
        let (state, timeline) = unsafe {
            (
                DeltaNetState::at(
                    dims,
                    device,
                    slot.ptr(),
                    slot.ptr() + block.conv_off as u64,
                    LeaseAnchor::new(Arc::clone(&slot)),
                )?,
                Tensor::from_anchored_cuda_ptr(
                    slot.ptr() + block.timeline_off as u64,
                    DType::F32,
                    lay.elems(),
                    device,
                    LeaseAnchor::new(Arc::clone(&slot)),
                )?,
            )
        };
        Ok(Self {
            state,
            timeline,
            held: Some(slot),
        })
    }
}

/// Rows of a short span's timeline nothing has applied yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Pending {
    /// The span's rows — every row its timeline holds.
    pub(super) len: usize,
    /// Written by the wave now open, so its rollback drops them. A wave that
    /// committed leaves its rows pending with this cleared: a later segment's
    /// rollback does not reach them.
    pub(super) this_wave: bool,
}

/// Per-layer slot: the live half, the prefill scratch, and the layer's
/// bookkeeping.
pub(super) struct LayerSlot {
    /// Trunk layer index (recurrent layers only — attention layers have no
    /// slot here).
    pub(super) layer_index: usize,
    /// The state as it stands, and its timeline.
    pub(super) live: Half,
    /// Where a long span writes the advanced state — held only while the
    /// sequence prefills ([`RecurrentStateStore::ensure_scratch`]).
    pub(super) scratch: Option<Half>,
    /// The timeline's rows nothing has applied yet.
    pub(super) pending: Option<Pending>,
    /// Whether this wave handed the layer to a long span, i.e. whether the
    /// scratch holds an advanced state that commit should install.
    ///
    /// Per slot, not per store, because a sweep may cover only part of the
    /// stack: exchanging a layer the wave never ran would install whatever its
    /// scratch held two waves ago.
    pub(super) advanced: bool,
}

/// Where one layer's half sits in its arena slot: `s` at the start, the conv
/// tail after it on the next [`SLOT_ALIGN`] boundary, the timeline after that on
/// the next.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StateBlock {
    /// The conv tail's offset.
    pub(super) conv_off: usize,
    /// Where the state and its tail end — the bytes a copy of the state moves.
    pub(super) state_end: usize,
    /// The timeline's offset.
    pub(super) timeline_off: usize,
    /// What a slot for this geometry must hold.
    pub(super) bytes: usize,
}

#[cfg(feature = "cuda")]
pub(super) fn state_block(dims: &DeltaNetDims) -> StateBlock {
    let (s_bytes, conv_bytes) = DeltaNetState::byte_sizes(dims);
    let conv_off = s_bytes.next_multiple_of(SLOT_ALIGN);
    let state_end = conv_off + conv_bytes;
    let timeline_off = state_end.next_multiple_of(SLOT_ALIGN);
    StateBlock {
        conv_off,
        state_end,
        timeline_off,
        bytes: timeline_off + TimelineLayout::of(dims).bytes(),
    }
}

/// `n` state-arena slots, one per recurrent layer, or `None` on a device with no
/// reservation to carve from (a CPU device in a CUDA build, which is every unit
/// test here).
#[cfg(feature = "cuda")]
pub(super) fn claim_layer_states(
    dims: &DeltaNetDims,
    device: &Device,
    n: usize,
) -> Result<Option<Vec<Arc<ArenaSlot>>>> {
    if !matches!(device, Device::Cuda(_)) {
        return Ok(None);
    }
    // No recurrent layer, no state, and no stride to key an arena by.
    if n == 0 {
        return Ok(Some(Vec::new()));
    }
    // **The store is the authority that refuses a geometry with an empty half.** A
    // buffer of no bytes has no address that is not some other buffer's: an empty
    // conv tail (`conv_kernel == 1`) would be leased at the end of `s`, which is the
    // next region of the slot, and the kernels would be handed that as a tail pointer.
    let (s_bytes, conv_bytes) = DeltaNetState::byte_sizes(dims);
    if s_bytes == 0 || conv_bytes == 0 {
        candle::bail!(
            "recurrent state: the geometry asks for a state with an empty half \
             ({s_bytes} B `s`, {conv_bytes} B conv tail — conv_kernel = 1?), which has \
             no address of its own"
        );
    }
    Ok(Some(
        claim_arena_slots(
            device,
            SlotTenant::RecurrentState,
            state_block(dims).bytes,
            n,
        )?
        .into_iter()
        .map(Arc::new)
        .collect(),
    ))
}
