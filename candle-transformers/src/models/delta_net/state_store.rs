//! The recurrent state store: one sequence's DeltaNet memory across all
//! recurrent layers, with wave-atomic advance/rollback and the export/import
//! bridge the turn-seal snapshot record is built from.
//!
//! # Wave atomicity
//!
//! The engine's relief design fails waves on purpose, and a failed wave must
//! leave no trace (`rollback_wave_kv` truncates KV; this store is the
//! recurrent analogue). The contract:
//!
//! ```text
//!   begin_wave()      nothing on the device — mark the slots un-advanced
//!   … the wave READS each layer's live `s` and WRITES the other buffer …
//!   commit_wave()     swap the two buffers of every layer that advanced
//!   rollback_wave()   nothing — the entering state was never written
//! ```
//!
//! A second `begin_wave` without a commit/rollback is refused — an overlapping
//! wave on one session is the bug wave atomicity exists to catch.
//!
//! # Why there is no snapshot
//!
//! The KV side gets its rollback free by being append-only: the pre-wave bytes
//! are still there, below the offset, so undoing a wave is `truncate_to_offset`
//! and costs nothing. The recurrent state has no such structure — `s` is a
//! fixed-size accumulator every token rewrites — so the first implementation
//! took the instruction "the same rollback discipline as KV" to mean copying the
//! entering state aside: ~2 MB and two `slice_set` launches per layer per wave,
//! on every wave, to insure against a rollback that almost never fires.
//!
//! Copying is not what makes KV's rollback free, though; *not destroying the old
//! value* is. So each slot holds two `s` buffers and the wave writes the one it
//! is not reading — the ping-pong `TableRing` and the expert staging ring
//! already use in this tree. Commit is a host `mem::swap`, rollback is nothing
//! at all, and a wave that fails at layer 7 leaves layers 0–6 correct because
//! their entering buffers were never written.
//!
//! Two consequences worth stating:
//!
//! - **`advanced` is per slot**, not per store. A sweep may cover part of the
//!   stack, and swapping a layer the wave never ran would install whatever its
//!   write buffer held two waves ago.
//! - **Both buffers ping-pong.** `s` and the conv tail are one state and swap
//!   together, because the conv kernels take the entering and advanced tails as
//!   two pointers: the decode kernel shifts one into the other and the prefill
//!   kernel writes the advance where the copy-back used to land. That copy-back
//!   was the last `slice_set` on this path — one launch per prefill span per
//!   layer, and the largest single source of `copy2d_f32` in the engine.
//!
//! A slot's buffers are still allocated once for its whole life; what a commit
//! changes is which of the two is live, so a device address resolved from the
//! store is good for the wave that resolved it. That is already how the engine
//! works — `build_wave_table` resolves the pointer table once per forward.
//!
//! # Export / import
//!
//! [`RecurrentStateStore::export`] reads every layer back as LE F32 bytes in
//! [`ExportedLayerState`] rows — field-for-field what the persistence layer's
//! `SnapshotLayer` carries (candle-conversation depends on this crate, not
//! the reverse, so the byte-layout contract lives here and the record
//! assembly there). [`RecurrentStateStore::import`] is the resume path and
//! validates dims + [`schedule_hash`] before touching any tensor.

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle::{
    cuda_backend::cudarc::driver::result::{memcpy_dtod_async, memcpy_dtod_sync},
    CudaDevice, Error, LeaseAnchor, Storage,
};
use candle::{Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    arena_regions, claim_arena_slots, plan_slot_moves, slot_stride, ArenaSlot, SlotTenant,
    SpanRegion, SLOT_ALIGN,
};

use super::mix::{DeltaNetOut, DeltaNetState};
use super::types::{DeltaNetDims, LayerKind};

/// One recurrent layer's state, exported as LE F32 bytes. Field-for-field the
/// persistence `SnapshotLayer` payload row.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportedLayerState {
    pub layer_index: u32,
    pub n_v_heads: u32,
    pub d_v: u32,
    pub d_k: u32,
    pub state: Vec<u8>,
    pub conv_channels: u32,
    pub conv_tail_cols: u32,
    pub conv_tail: Vec<u8>,
}

/// Fingerprint of a model's recurrent layout: the layer schedule plus the
/// DeltaNet dims. A snapshot taken under one hash must never be scattered
/// into a store built under another — resume recomputes instead.
pub fn schedule_hash(layer_kinds: &[LayerKind], dims: &DeltaNetDims) -> u64 {
    // FNV-1a: stable, dependency-free, and this is an identity check, not
    // crypto.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |b: u64| {
        for byte in b.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(layer_kinds.len() as u64);
    for (i, k) in layer_kinds.iter().enumerate() {
        mix(i as u64);
        mix(match k {
            LayerKind::DeltaNet => 1,
            LayerKind::Attention => 2,
        });
    }
    mix(dims.head_dim as u64);
    mix(dims.n_k_heads as u64);
    mix(dims.n_v_heads as u64);
    mix(dims.conv_kernel as u64);
    h
}

/// Per-layer slot: the two halves of the state's ping-pong.
struct LayerSlot {
    /// Trunk layer index (recurrent layers only — attention layers have no
    /// slot here).
    layer_index: usize,
    /// The state as it stands. A wave READS this and never writes it.
    live: DeltaNetState,
    /// Where a wave WRITES the advanced state. Fully overwritten by the
    /// kernels, so it carries nothing forward from whatever it last held.
    backup: DeltaNetState,
    /// Whether this wave handed the layer its write buffer, i.e. whether
    /// `backup` holds an advanced state that commit should install.
    ///
    /// Per slot, not per store, because a sweep may cover only part of the
    /// stack: swapping a layer the wave never ran would install whatever its
    /// write buffer held two waves ago.
    advanced: bool,
    /// The state-arena slots `live` and `backup` are views into, in that order —
    /// exchanged with them at `commit_wave`, so the pair always says which slot
    /// backs which half.
    ///
    /// **Not the only holder, by design.** Every tensor built on a slot holds a
    /// share of it too (`DeltaNetState::at` anchors the lease), as does every clone,
    /// view and re-lease of those tensors. A slot therefore goes back to its arena
    /// only when the last thing that could read it has gone — dropping the store is
    /// what normally ends that, but a handle taken along the way cannot end up
    /// reading a slot another sequence now holds. Held here so the store can move a
    /// half to another slot ([`RecurrentStateStore::relocate`]) and say what it costs
    /// without walking its tensors.
    ///
    /// `None` on a CPU device, where the buffers are ordinary allocations.
    #[cfg(feature = "cuda")]
    held: Option<[Arc<ArenaSlot>; 2]>,
}

/// One sequence's recurrent memory across every DeltaNet layer.
pub struct RecurrentStateStore {
    dims: DeltaNetDims,
    hash: u64,
    slots: Vec<LayerSlot>,
    /// Whether a wave is open, i.e. whether the backups hold an entry copy.
    ///
    /// One flag for the store rather than one per slot: the three wave
    /// operations act on every slot together, so a per-slot answer could only
    /// ever disagree with its neighbours by being wrong.
    open: bool,
    /// Whether this store's state was put here deliberately — by a fork or a
    /// restore — and must therefore survive one `offset == 0` reset.
    ///
    /// Store-level, unlike `advanced`, because seeding is a property of where
    /// the whole state came from rather than of which layers a sweep reached.
    seeded: bool,
    device: Device,
}

/// Where one layer state sits in its arena slot: `s` at the start, the conv tail
/// after it on the next [`SLOT_ALIGN`] boundary. Answers `(conv tail offset, slot
/// bytes)` — the second is what a state-arena slot for this geometry must hold.
#[cfg(feature = "cuda")]
fn state_block(dims: &DeltaNetDims) -> (usize, usize) {
    let (s_bytes, conv_bytes) = DeltaNetState::byte_sizes(dims);
    let conv_off = s_bytes.next_multiple_of(SLOT_ALIGN);
    (conv_off, conv_off + conv_bytes)
}

/// Two state-arena slots per recurrent layer — the live state and the half a wave
/// writes — or `None` on a device with no reservation to carve from (a CPU device in
/// a CUDA build, which is every unit test here).
#[cfg(feature = "cuda")]
fn claim_layer_states(
    dims: &DeltaNetDims,
    device: &Device,
    layers: usize,
) -> Result<Option<Vec<Arc<ArenaSlot>>>> {
    if !matches!(device, Device::Cuda(_)) {
        return Ok(None);
    }
    // No recurrent layer, no state, and no stride to key an arena by.
    if layers == 0 {
        return Ok(Some(Vec::new()));
    }
    // **The store is the authority that refuses a geometry with an empty half.** A
    // buffer of no bytes has no address that is not some other buffer's: an empty
    // conv tail (`conv_kernel == 1`) would be leased at the end of `s`, which is the
    // next slot's base, and the kernels would be handed that as a tail pointer.
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
            state_block(dims).1,
            2 * layers,
        )?
        .into_iter()
        .map(Arc::new)
        .collect(),
    ))
}

/// The next layer's `(live, backup)` slots from a store's claim — two per recurrent
/// layer, in layer order.
#[cfg(feature = "cuda")]
fn next_pair(it: &mut impl Iterator<Item = Arc<ArenaSlot>>) -> (Arc<ArenaSlot>, Arc<ArenaSlot>) {
    let live = it
        .next()
        .expect("two slots were claimed per recurrent layer");
    let backup = it
        .next()
        .expect("two slots were claimed per recurrent layer");
    (live, backup)
}

/// What one recurrent-state compaction did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentCompaction {
    /// Moves the pass planned, each with its destination claimed.
    pub planned: usize,
    /// Layer-state halves that moved onto their destination. Lower than `planned`
    /// only when a source was held by nothing the pass could reach — a handle that
    /// outlived its store — whose claimed destination simply goes back.
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
/// **Between forwards, and a no-op while any store has a wave open.** A relocation
/// rebuilds a half's tensors on a new slot, and an open wave has already resolved the
/// old addresses into its pointer tables; between forwards nothing holds one — the
/// tables are rebuilt every forward — so a moved half is simply read at its new
/// address by the next forward.
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
        let stores: Vec<&mut RecurrentStateStore> = stores.into_iter().collect();
        if stores.iter().any(|s| s.open) {
            return Ok(RecurrentCompaction::default());
        }
        let regions_before = arena_regions(device, SlotTenant::RecurrentState);
        let moves = plan_slot_moves(
            device,
            SlotTenant::RecurrentState,
            state_block(dims).1,
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

/// The layer state living in `slot`, its tensors anchored to the slot so no view of
/// them can outlive it.
#[cfg(feature = "cuda")]
fn state_in(dims: &DeltaNetDims, device: &Device, slot: &Arc<ArenaSlot>) -> Result<DeltaNetState> {
    let (conv_off, bytes) = state_block(dims);
    debug_assert!(slot.stride() >= bytes, "a slot narrower than its state");
    // SAFETY: the slot spans at least `bytes` from its 256-aligned base — `s` at the
    // start, the tail at `conv_off` — and the anchor keeps it held for as long as any
    // view of these tensors exists.
    unsafe {
        DeltaNetState::at(
            dims,
            device,
            slot.ptr(),
            slot.ptr() + conv_off as u64,
            LeaseAnchor::new(Arc::clone(slot)),
        )
    }
}

/// Zero one layer state's slot — for a `live` half, which is genuinely READ at zero,
/// being the sequence-start state (hot-path invariant 6's read-before-write case).
///
/// Needed because a slot is recycled: it last held some other sequence's state.
#[cfg(feature = "cuda")]
fn zero_state(dims: &DeltaNetDims, device: &Device, slot: &ArenaSlot) -> Result<()> {
    slot.zero(state_block(dims).1, device)
}

/// Copy one state's two buffers into another's, device to device.
///
/// The fork path's replacement for `DeltaNetState::snapshot`, which allocates.
/// Here the destination already exists — it is a view into the child's own
/// state-arena slot — so the copy writes into it rather than producing a new buffer
/// somewhere the reservation does not cover.
#[cfg(feature = "cuda")]
fn copy_state_into(device: &Device, src: &DeltaNetState, dst: &DeltaNetState) -> Result<()> {
    let Device::Cuda(cuda) = device else {
        candle::bail!("copy_state_into: expected a CUDA device");
    };
    for (s, d) in [(&src.s, &dst.s), (&src.conv_tail, &dst.conv_tail)] {
        let bytes = s.elem_count() * s.dtype().size_in_bytes();
        let src_ptr = tensor_device_ptr(cuda, s)?;
        let dst_ptr = tensor_device_ptr(cuda, d)?;
        // SAFETY: both ranges are `bytes` long, live, and disjoint — the
        // destination belongs to a store being built, which nothing else has
        // yet seen.
        unsafe {
            memcpy_dtod_sync(dst_ptr, src_ptr, bytes)
                .map_err(|e| Error::Msg(format!("recurrent fork copy: {e}")))?;
        }
    }
    Ok(())
}

/// Base device address of a contiguous CUDA tensor.
#[cfg(feature = "cuda")]
fn tensor_device_ptr(cuda: &CudaDevice, t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        candle::bail!("recurrent state buffers are contiguous by construction");
    }
    let Storage::Cuda(c) = &*storage else {
        candle::bail!("recurrent state: expected CUDA storage");
    };
    let stream = cuda.cuda_stream();
    let base = c.slice.device_ptr(&stream);
    Ok(base + (layout.start_offset() * t.dtype().size_in_bytes()) as u64)
}

impl RecurrentStateStore {
    /// Fresh zeros for every recurrent layer in `layer_kinds`.
    pub fn new(layer_kinds: &[LayerKind], dims: &DeltaNetDims, device: &Device) -> Result<Self> {
        let mut slots = Vec::new();
        // On CUDA every layer state is a state-arena slot inside the device
        // reservation. `live` is zeroed — it is genuinely READ at zero, being the
        // sequence-start state, and a slot is recycled from whatever sequence held
        // it last — while `backup` is not, being fully stamped by the first wave
        // before anything reads it (invariant 6).
        // The cfg gates COMPILATION; the device gates behaviour. A CUDA build
        // still runs on a CPU device — every unit test here does — and there is
        // no reservation there to carve from.
        #[cfg(feature = "cuda")]
        let mut claimed = claim_layer_states(
            dims,
            device,
            layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::DeltaNet)
                .count(),
        )?
        .map(Vec::into_iter);
        for (i, k) in layer_kinds.iter().enumerate() {
            if *k == LayerKind::DeltaNet {
                #[cfg(feature = "cuda")]
                let (live, backup, held) = match claimed.as_mut() {
                    Some(it) => {
                        let (live, backup) = next_pair(it);
                        zero_state(dims, device, &live)?;
                        (
                            state_in(dims, device, &live)?,
                            state_in(dims, device, &backup)?,
                            Some([live, backup]),
                        )
                    }
                    None => (
                        DeltaNetState::zeros(dims, device)?,
                        DeltaNetState::uninit(dims, device)?,
                        None,
                    ),
                };
                #[cfg(not(feature = "cuda"))]
                let (live, backup) = (
                    DeltaNetState::zeros(dims, device)?,
                    DeltaNetState::uninit(dims, device)?,
                );
                slots.push(LayerSlot {
                    layer_index: i,
                    live,
                    backup,
                    advanced: false,
                    #[cfg(feature = "cuda")]
                    held,
                });
            }
        }
        Ok(Self {
            dims: *dims,
            hash: schedule_hash(layer_kinds, dims),
            slots,
            open: false,
            // A fresh store already holds the sequence-start value, so there is
            // nothing for a reset to destroy.
            seeded: false,
            device: device.clone(),
        })
    }

    pub fn schedule_hash(&self) -> u64 {
        self.hash
    }

    /// Trunk layer indices of the recurrent layers, in slot order — what the
    /// decode pointer table iterates to collect every layer's state address.
    pub fn recurrent_layer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().map(|s| s.layer_index)
    }

    pub fn n_recurrent_layers(&self) -> usize {
        self.slots.len()
    }

    /// The live state of trunk layer `layer_index`, for the layer forward /
    /// decode kernel. Errors on an attention layer's index.
    pub fn layer_state(&self, layer_index: usize) -> Result<&DeltaNetState> {
        self.slots
            .iter()
            .find(|s| s.layer_index == layer_index)
            .map(|s| &s.live)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })
    }

    /// Trunk layer `layer_index`'s live state, to be **written into** —
    /// **outside a wave only**.
    ///
    /// This is the in-place form, and a wave must not use it: a wave advances a
    /// layer by writing the buffer it is *not* reading
    /// ([`Self::layer_state_pair_mut`]), and writing `live` instead destroys the
    /// entering state that a rollback returns to, while `commit_wave` then swaps
    /// the untouched other buffer in and discards the work. Both failures are
    /// silent. What legitimately uses this is code holding a store no wave is
    /// open on — the verification path builds a fresh single-sequence store per
    /// block and advances it directly.
    ///
    /// There is deliberately no setter: a store that could be handed a
    /// *different* tensor is one where prefill and decode end up advancing the
    /// state two different ways.
    pub fn layer_state_mut(&mut self, layer_index: usize) -> Result<&mut DeltaNetState> {
        self.slots
            .iter_mut()
            .find(|s| s.layer_index == layer_index)
            .map(|s| &mut s.live)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })
    }

    /// The layer's `(entering, advanced)` buffers **without** recording that it
    /// advanced.
    ///
    /// For resolving addresses ahead of the work: the decode pointer table is
    /// built once per forward over every recurrent layer, including ones a
    /// partial sweep will never reach, so building it must not be what decides
    /// a layer gets swapped at commit. The layer records itself when it runs,
    /// through [`Self::layer_state_pair_mut`].
    pub fn layer_state_pair(&self, layer_index: usize) -> Result<(&DeltaNetState, DeltaNetOut)> {
        let slot = self
            .slots
            .iter()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })?;
        Ok((&slot.live, slot.backup.write_half()))
    }

    /// The layer's `(entering, advanced)` buffers — what a wave reads and what
    /// it writes — and the record that this layer advanced.
    ///
    /// Taking this pair is what marks the slot for the swap at
    /// [`Self::commit_wave`], so a caller asks for it exactly when it is about
    /// to run the layer, never to peek.
    pub fn layer_state_pair_mut(
        &mut self,
        layer_index: usize,
    ) -> Result<(&mut DeltaNetState, DeltaNetOut)> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })?;
        slot.advanced = true;
        let out = slot.backup.write_half();
        Ok((&mut slot.live, out))
    }

    /// The layer's halves **the other way round**: the state the last committed
    /// wave *entered* with, and the live buffer to write a corrected advance
    /// into.
    ///
    /// This is the rewind primitive. `commit_wave` exchanges a slot's two
    /// buffers, so immediately afterwards the half that is no longer live still
    /// holds the pre-wave state — untouched, because a wave writes only the
    /// buffer it is not reading. Re-running a *prefix* of the wave's tokens
    /// from there lands the correct shorter advance in the live buffer, which
    /// is how a speculative block keeps the accepted tokens and drops the rest;
    /// `S` is a running sum with no suffix to subtract, so replaying forward is
    /// the only exact answer.
    ///
    /// **Valid only between the commit and the next `begin_wave`.** After
    /// another wave has run, the non-live half holds *that* wave's entry state
    /// and this returns a rewind to the wrong point. Refused while a wave is
    /// open, which is the half of that the store can see.
    pub fn layer_state_rewind(
        &mut self,
        layer_index: usize,
    ) -> Result<(&mut DeltaNetState, DeltaNetOut)> {
        if self.open {
            candle::bail!(
                "recurrent store: layer_state_rewind mid-wave — the entering state \
                 to rewind to is the buffer the open wave is writing"
            );
        }
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| {
                candle::Error::Msg(format!(
                    "recurrent store: layer {layer_index} holds no recurrent state"
                ))
            })?;
        let out = slot.live.write_half();
        Ok((&mut slot.backup, out))
    }

    /// Open a wave. Refuses while one is already open.
    ///
    /// **Costs nothing on the device.** The entering state is preserved by not
    /// being written: a wave reads `live` and writes `backup`, so opening a wave
    /// is bookkeeping and rolling one back is doing nothing at all. This is the
    /// same trick the KV side gets for free by being append-only — its rollback
    /// is `truncate_to_offset`, because the pre-wave bytes were never touched.
    ///
    /// It replaces a copy of every layer's state into its backup: ~2 MB per
    /// layer per wave, two `slice_set` launches each, paid on every wave to
    /// insure against a rollback that almost never happens.
    pub fn begin_wave(&mut self) -> Result<()> {
        if self.open {
            candle::bail!(
                "recurrent store: begin_wave with a wave already open — overlapping \
                 waves on one session are exactly what atomicity forbids"
            );
        }
        for slot in &mut self.slots {
            slot.advanced = false;
        }
        self.open = true;
        self.assert_state("gdn.entry.s.L", "gdn.entry.tail.L");
        Ok(())
    }

    /// Fold every layer's live state into the assert slots under `prefix`.
    ///
    /// **Entry and exit, because the pair is what says who broke it.** A wave reads
    /// `live` and writes `backup`, so the state at entry is whatever the last
    /// committed wave left plus anything that has happened to the memory since. If
    /// entry is bad while the previous exit was good, nothing computed it — the
    /// bytes changed while the store sat idle, which is a foreign write into the
    /// span. If exit is the first bad site, the wave computed it, and the cause is
    /// upstream in the same forward.
    ///
    /// Asynchronous: one reduction kernel per buffer, no readback and no fence. The
    /// per-wave drain in `wave_driver` reports the slots and names the first bad
    /// site by the kernel's own ticket, so the ordering this depends on is the
    /// device's, not the host's. That matters more than it looks — these faults stop
    /// reproducing in a fenced build, so an instrument that synchronised here would
    /// suppress the thing it is watching for.
    #[cfg(feature = "tensor-assert")]
    fn assert_state(&self, s_prefix: &'static str, tail_prefix: &'static str) {
        use candle::tensor_assert::names::site;
        for slot in &self.slots {
            let l = slot.layer_index;
            slot.live.s.assert(site(s_prefix, l));
            slot.live.conv_tail.assert(site(tail_prefix, l));
        }
    }

    #[cfg(not(feature = "tensor-assert"))]
    #[inline]
    fn assert_state(&self, _s_prefix: &'static str, _tail_prefix: &'static str) {}

    /// The wave's writes stand: every layer the wave advanced exchanges its two
    /// buffers, so what the wave wrote becomes the state and what the state was
    /// becomes the next wave's write buffer.
    ///
    /// A host pointer swap per advanced layer, and no device work at all. Layers
    /// the sweep did not reach keep their buffers as they are — their write
    /// buffer holds an older wave's output, which is exactly why the flag is per
    /// slot.
    pub fn commit_wave(&mut self) {
        for slot in &mut self.slots {
            if slot.advanced {
                // The whole state: `s` and the conv tail are both written into
                // the backup half by the wave's kernels — the conv kernels take
                // the entering and advanced tails as two pointers — so they are
                // installed together.
                std::mem::swap(&mut slot.live, &mut slot.backup);
                // The slots backing the two halves change roles with them.
                #[cfg(feature = "cuda")]
                if let Some(held) = slot.held.as_mut() {
                    held.swap(0, 1);
                }
                slot.advanced = false;
            }
        }
        self.open = false;
        // After the swap, so this is the state the NEXT wave will read — the same
        // buffers `gdn.entry` will fold on the way in. A pair that disagrees across
        // the gap between two waves is a write nothing in the forward performed.
        self.assert_state("gdn.exit.s.L", "gdn.exit.tail.L");
    }

    /// The wave never happened.
    ///
    /// Nothing to undo: a wave writes only into the buffers `commit_wave` would
    /// have swapped in, so declining to swap *is* the rollback. Refuses when no
    /// wave is open (a rollback with nothing to roll back to is a sequencing
    /// bug, not a no-op).
    pub fn rollback_wave(&mut self) -> Result<()> {
        if !self.open {
            candle::bail!("recurrent store: rollback_wave with no wave open");
        }
        for slot in &mut self.slots {
            slot.advanced = false;
        }
        self.open = false;
        Ok(())
    }

    /// An independent store carrying this one's state — the fork primitive.
    ///
    /// Device-to-device: each slot's live `s` and conv tail go through
    /// [`DeltaNetState::snapshot`], which is `Tensor::copy` and never touches
    /// the host. The write half of the ping-pong is **not** copied — a wave
    /// fully overwrites it before reading it, so its contents are not state,
    /// they are scratch.
    ///
    /// Refused mid-wave, and the reason is sharper than `export`'s. Mid-wave
    /// the *advanced* state is in `backup` while `live` is one wave stale, so a
    /// mid-wave fork would not merely copy a moving value — it would copy the
    /// wrong buffer, confidently, and the child would come up a wave behind its
    /// parent with every shape correct.
    ///
    /// Reads the slot fields directly rather than going through
    /// [`Self::layer_state_pair_mut`], which marks a slot `advanced` and would
    /// make the parent's next commit swap in a buffer no wave ever wrote.
    pub fn fork_from(&self) -> Result<Self> {
        if self.open {
            candle::bail!(
                "recurrent store: fork_from mid-wave — the advanced state is in the \
                 write buffer and `live` is a wave behind, so the child would come up \
                 stale. Fork at a wave boundary."
            );
        }
        let mut slots = Vec::with_capacity(self.slots.len());
        // The child's memory comes from the state arenas for the same reason the
        // parent's does — a fork is another sequence, and at ~3 forks per turn
        // this was ~126 MiB of pool traffic each.
        #[cfg(feature = "cuda")]
        let mut claimed =
            claim_layer_states(&self.dims, &self.device, self.slots.len())?.map(Vec::into_iter);
        for slot in &self.slots {
            // Scratch, not state, in either arm: the kernels fully overwrite
            // the write buffer before anything reads it, so copying it would be
            // ~2 MB per layer of device traffic for bytes nobody reads — and
            // for the same reason it is left UNINITIALISED (invariant 6).
            #[cfg(feature = "cuda")]
            let (live, backup, held) = match claimed.as_mut() {
                Some(it) => {
                    let (live_slot, backup_slot) = next_pair(it);
                    let (live, backup) = (
                        state_in(&self.dims, &self.device, &live_slot)?,
                        state_in(&self.dims, &self.device, &backup_slot)?,
                    );
                    // The fork's whole point: the child starts from the
                    // parent's state. A device-to-device copy into the child's
                    // own slot, rather than `snapshot()`, which would allocate
                    // a fresh pool buffer and hand back a tensor pointing
                    // outside the span.
                    copy_state_into(&self.device, &slot.live, &live)?;
                    (live, backup, Some([live_slot, backup_slot]))
                }
                None => (
                    slot.live.snapshot()?,
                    DeltaNetState::uninit(&self.dims, &self.device)?,
                    None,
                ),
            };
            #[cfg(not(feature = "cuda"))]
            let (live, backup) = (
                slot.live.snapshot()?,
                DeltaNetState::uninit(&self.dims, &self.device)?,
            );
            slots.push(LayerSlot {
                layer_index: slot.layer_index,
                live,
                backup,
                advanced: false,
                #[cfg(feature = "cuda")]
                held,
            });
        }
        Ok(Self {
            dims: self.dims,
            hash: self.hash,
            slots,
            open: false,
            seeded: true,
            device: self.device.clone(),
        })
    }

    /// Move every half of this store whose slot is the source of a planned move onto
    /// that move's destination, taking the destination out of `moves`; answers how
    /// many halves moved.
    ///
    /// Per half: one device copy of the block on the primary stream, the half's
    /// tensors rebuilt on the destination slot, the source slot dropped. The drop is
    /// on the host while the copy may still be queued, which is sound for the reason
    /// `ArenaSlot` gives — the source's next tenant is ordered behind the copy on the
    /// same stream, or, if it empties its arena, behind the region pool's fence.
    /// Anything else still holding the old half keeps the old slot alive until it
    /// lets go.
    ///
    /// Both halves move, not only `live`: the write half is scratch between waves,
    /// but immediately after a commit it holds the entering state
    /// [`Self::layer_state_rewind`] rewinds to, so it is copied like the other.
    #[cfg(feature = "cuda")]
    pub fn relocate(&mut self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        if self.open {
            candle::bail!(
                "recurrent store: relocate mid-wave — the open wave has already \
                 resolved this store's addresses"
            );
        }
        let Device::Cuda(cuda) = &self.device else {
            return Ok(0);
        };
        let stream = cuda.cuda_stream();
        let bytes = state_block(&self.dims).1;
        let mut moved = 0usize;
        for slot in &mut self.slots {
            let LayerSlot {
                live, backup, held, ..
            } = slot;
            let Some(held) = held.as_mut() else {
                continue;
            };
            for (at, state) in held.iter_mut().zip([live, backup]) {
                let Some(dst) = moves.remove(&at.ptr()) else {
                    continue;
                };
                // SAFETY: both ranges are `bytes` of state-arena slots — the source
                // held by this store, the destination claimed for this move — and they
                // are distinct slots, so they do not overlap.
                unsafe { memcpy_dtod_async(dst.ptr(), at.ptr(), bytes, stream.cu_stream()) }
                    .map_err(|e| Error::Msg(format!("relocating recurrent state: {e}")))?;
                let dst = Arc::new(dst);
                *state = state_in(&self.dims, &self.device, &dst)?;
                *at = dst;
                moved += 1;
            }
        }
        Ok(moved)
    }

    /// Reservation bytes this sequence's recurrent memory holds — its state-arena
    /// slots, at their stride. Zero off CUDA, where the buffers are ordinary
    /// allocations.
    ///
    /// A per-sequence figure, so it leaves out what no one sequence holds: an arena's
    /// free slots and unused tail, and slots kept by a handle that outlived its
    /// store. [`Self::arena_reserved_bytes`] is the whole-card figure that includes
    /// them.
    pub fn reserved_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.slots
                .iter()
                .flat_map(|s| s.held.iter().flatten())
                .map(|s| s.stride())
                .sum()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Reservation bytes every recurrent-state arena on `device` holds — and every
    /// rewind-stash arena, the buffers a speculative verify keeps to rewind this
    /// state — as whole regions, whatever is in them.
    ///
    /// **The accounting figure.** It is what recurrent state denies the rest of the
    /// span: the slots every store holds, the free slots and unused tails of their
    /// arenas, and any slot a handle kept after its store went. Memory nothing can
    /// total is memory that goes missing (`AccountingSection`), and summing stores
    /// would miss all but the first. Zero off CUDA.
    pub fn arena_reserved_bytes(device: &Device) -> usize {
        #[cfg(feature = "cuda")]
        {
            (arena_regions(device, SlotTenant::RecurrentState)
                + arena_regions(device, SlotTenant::RewindStash))
                * SpanRegion::bytes()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            0
        }
    }

    /// What one sequence's store **will** reserve, from the geometry alone —
    /// [`Self::reserved_bytes`] for a store that does not exist yet.
    ///
    /// **This is the figure admission needs, and it is the one a live store
    /// cannot give.** A store is priced before it is built, and at that moment
    /// there may be none in the process to measure: the first sequence of a
    /// session, or the first after a seal evicted every store.
    ///
    /// Deriving it from residency instead — summing the live stores and
    /// dividing by what the scheduler has in flight — is not a per-sequence
    /// figure at all, because the two counts range over different populations.
    /// The sum covers every store the process holds, parked conversations
    /// included; the divisor covers only what is in flight, and admission runs
    /// *between* forwards, where that is 0 or 1. The quotient therefore rises
    /// with the number of idle conversations and peaks when the engine is
    /// quiet, which is precisely when admission should be cheapest. Measured on
    /// a 72 GB card: a 41-row turn priced at 4,450 MiB and refused as
    /// throughput-worse on fifteen consecutive passes, seven turns queued
    /// behind it and 20 GiB standing free above the floor.
    ///
    /// The price is the store's slots at their stride: two per DeltaNet layer,
    /// each [`slot_stride`] of the layer's block (`s`, then its conv tail on the
    /// next 256-byte boundary) — exactly what [`Self::reserved_bytes`] reports once
    /// the store stands. The arenas those slots live in are shared by every
    /// sequence of the geometry, so a region's unused tail is not charged to any one
    /// of them. Nothing here reads the device, so it answers on any backend and at
    /// any moment.
    pub fn reserved_bytes_for(layer_kinds: &[LayerKind], dims: &DeltaNetDims) -> usize {
        #[cfg(feature = "cuda")]
        {
            let layers = layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::DeltaNet)
                .count();
            2 * layers * slot_stride(state_block(dims).1)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (layer_kinds, dims);
            0
        }
    }

    /// Whether this store's state arrived by fork or restore and must survive
    /// its first `offset == 0` reset. Consumed by that reset — see
    /// [`Self::take_seeded`].
    pub fn is_seeded(&self) -> bool {
        self.seeded
    }

    /// Mark the state as externally seeded (a restore).
    pub fn mark_seeded(&mut self) {
        self.seeded = true;
    }

    /// Read and clear the seeded flag: `true` exactly once after a fork or a
    /// restore, and the caller must then not reset the store.
    ///
    /// The flag exists because "was this slot's state put here deliberately?"
    /// has no other answer. `ensure_recurrent` resets on `offset == 0` because
    /// a sequence with no history must hold the sequence-start value, and a
    /// freshly restored slot standing at offset 0 before its first wave looks
    /// exactly like one. Relying on the projection to have moved the offset
    /// first is correct today by ordering nothing asserts; this makes it
    /// explicit.
    pub fn take_seeded(&mut self) -> bool {
        std::mem::take(&mut self.seeded)
    }

    /// Read every layer back as LE F32 bytes — the turn-seal snapshot body.
    /// Refused mid-wave: a snapshot must capture a sealed boundary, never a
    /// wave in flight.
    pub fn export(&self) -> Result<Vec<ExportedLayerState>> {
        if self.open {
            candle::bail!("recurrent store: export mid-wave — seal, then snapshot");
        }
        let d = &self.dims;
        let mut out = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let state_v: Vec<f32> = slot.live.s.flatten_all()?.to_vec1()?;
            let tail_v: Vec<f32> = slot.live.conv_tail.flatten_all()?.to_vec1()?;
            out.push(ExportedLayerState {
                layer_index: slot.layer_index as u32,
                n_v_heads: d.n_v_heads as u32,
                d_v: d.head_dim as u32,
                d_k: d.head_dim as u32,
                state: state_v.iter().flat_map(|f| f.to_le_bytes()).collect(),
                conv_channels: d.conv_dim() as u32,
                conv_tail_cols: (d.conv_kernel - 1) as u32,
                conv_tail: tail_v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            });
        }
        Ok(out)
    }

    /// Scatter a snapshot back into the store — the resume path. Validates
    /// the schedule hash and every layer's dims before touching any tensor;
    /// on any mismatch the store is left untouched and the caller recomputes.
    pub fn import(&mut self, snapshot_hash: u64, layers: &[ExportedLayerState]) -> Result<()> {
        if snapshot_hash != self.hash {
            candle::bail!(
                "recurrent store: snapshot schedule hash {snapshot_hash:#x} does not match \
                 this model's {:#x} — recompute the state instead of scattering a foreign \
                 layout",
                self.hash
            );
        }
        if self.open {
            candle::bail!("recurrent store: import mid-wave");
        }
        let d = &self.dims;
        if layers.len() != self.slots.len() {
            candle::bail!(
                "recurrent store: snapshot has {} layers, store has {}",
                layers.len(),
                self.slots.len()
            );
        }
        // Validate everything first — import is all-or-nothing.
        for (slot, l) in self.slots.iter().zip(layers) {
            if l.layer_index as usize != slot.layer_index
                || l.n_v_heads as usize != d.n_v_heads
                || l.d_v as usize != d.head_dim
                || l.d_k as usize != d.head_dim
                || l.conv_channels as usize != d.conv_dim()
                || l.conv_tail_cols as usize != d.conv_kernel - 1
                || l.state.len() != d.state_elems() * 4
                || l.conv_tail.len() != d.conv_state_elems() * 4
            {
                candle::bail!(
                    "recurrent store: snapshot layer {} does not match the store's \
                     geometry",
                    l.layer_index
                );
            }
        }
        for (slot, l) in self.slots.iter_mut().zip(layers) {
            let state_f: Vec<f32> = l
                .state
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let tail_f: Vec<f32> = l
                .conv_tail
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            // Written into the slot's buffers rather than replacing them: the
            // slot's tensors keep their identity for the store's whole life, and
            // the fused decode kernels rely on that.
            slot.live.copy_from(&DeltaNetState {
                s: Tensor::from_vec(state_f, (d.n_v_heads, d.head_dim, d.head_dim), &self.device)?,
                conv_tail: Tensor::from_vec(
                    tail_f,
                    (d.conv_dim(), d.conv_kernel - 1),
                    &self.device,
                )?,
            })?;
        }
        // Restored state is state someone put here on purpose. Without this the
        // first wave on a resumed slot standing at offset 0 would reset it, and
        // the conversation would come back fluent and amnesiac — the exact
        // failure resume exists to remove.
        self.seeded = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims() -> DeltaNetDims {
        DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        }
    }

    fn kinds() -> Vec<LayerKind> {
        vec![
            LayerKind::DeltaNet,
            LayerKind::DeltaNet,
            LayerKind::Attention,
            LayerKind::DeltaNet,
        ]
    }

    /// One wave's worth of writes into a layer's destination half: `live + 1`
    /// in both buffers, which is what the kernels do to their two pointers.
    fn bump_into(live: &DeltaNetState, out: &DeltaNetOut) {
        let one = |src: &Tensor, dst: &Tensor| {
            let ones = Tensor::ones(src.shape(), src.dtype(), &Device::Cpu).unwrap();
            dst.slice_set(&src.add(&ones).unwrap(), 0, 0).unwrap();
        };
        one(&live.s, &out.s);
        one(&live.conv_tail, &out.conv_tail);
    }

    fn filled_store() -> RecurrentStateStore {
        let dev = Device::Cpu;
        let d = dims();
        let mut store = RecurrentStateStore::new(&kinds(), &d, &dev).unwrap();
        for (i, li) in [0usize, 1, 3].iter().enumerate() {
            let n = d.state_elems();
            let s: Vec<f32> = (0..n).map(|j| (i * 1000 + j) as f32 * 0.01).collect();
            let tn = d.conv_state_elems();
            let t: Vec<f32> = (0..tn).map(|j| (i * 100 + j) as f32 * 0.1).collect();
            let live = store.layer_state_mut(*li).unwrap();
            live.copy_from(&DeltaNetState {
                s: Tensor::from_vec(s, (d.n_v_heads, d.head_dim, d.head_dim), &dev).unwrap(),
                conv_tail: Tensor::from_vec(t, (d.conv_dim(), d.conv_kernel - 1), &dev).unwrap(),
            })
            .unwrap();
        }
        store
    }

    #[test]
    fn export_import_roundtrips_exactly() {
        let store = filled_store();
        let hash = store.schedule_hash();
        let exported = store.export().unwrap();
        assert_eq!(exported.len(), 3);
        assert_eq!(exported[2].layer_index, 3);

        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        fresh.import(hash, &exported).unwrap();
        let re = fresh.export().unwrap();
        assert_eq!(exported, re, "export→import→export must be byte-identical");
    }

    #[test]
    fn import_refuses_wrong_hash_and_wrong_geometry() {
        let store = filled_store();
        let exported = store.export().unwrap();

        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        let before = fresh.export().unwrap();
        let err = fresh
            .import(store.schedule_hash() ^ 1, &exported)
            .unwrap_err();
        assert!(err.to_string().contains("schedule hash"));
        assert_eq!(
            fresh.export().unwrap(),
            before,
            "refusal must not touch state"
        );

        let mut bad = exported.clone();
        bad[0].d_k = 5;
        let err = fresh.import(store.schedule_hash(), &bad).unwrap_err();
        assert!(err.to_string().contains("geometry"));
        assert_eq!(fresh.export().unwrap(), before);
    }

    #[test]
    fn wave_rollback_restores_entry_state_and_commit_keeps_writes() {
        let mut store = filled_store();
        let entry = store.export().unwrap();

        // A wave writes into the slot's OTHER buffer — the half `commit_wave`
        // swaps in — so the entering state survives by never being written.
        let bump = |store: &mut RecurrentStateStore| {
            let (live, out) = store.layer_state_pair_mut(0).unwrap();
            // Stands in for the kernels' writes into the destination buffers —
            // both of them, because commit installs the whole state.
            bump_into(live, &out);
        };
        store.begin_wave().unwrap();
        bump(&mut store);
        store.rollback_wave().unwrap();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "rollback must restore the wave-entry state exactly"
        );

        // A successful wave: mutate, commit — the write stands, in BOTH
        // buffers. Asserting only on `s` would pass while the conv tail was
        // left behind in the half the swap filed away, which is precisely the
        // failure a partial swap produces: a state one wave ahead of its tail.
        store.begin_wave().unwrap();
        bump(&mut store);
        store.commit_wave();
        let committed = store.export().unwrap();
        assert_ne!(
            committed[0].state, entry[0].state,
            "commit must install `s`"
        );
        assert_ne!(
            committed[0].conv_tail, entry[0].conv_tail,
            "commit must install the advanced conv tail, not just `s`"
        );
        // Layer 1 never ran, so its slot keeps both buffers as they were.
        assert_eq!(committed[1], entry[1], "an unrun layer must not be swapped");
    }

    /// **A wave never writes the buffer it read, so an entering alias is never
    /// disturbed by a wave that fails.**
    ///
    /// This replaces the inverse contract — that rollback must copy the entry
    /// values back into the same allocation, because an alias resolved before
    /// the wave would otherwise still see the failed wave's writes. Under the
    /// ping-pong there are no writes to undo: the wave's output went to the
    /// other buffer, so the alias holds the entry values throughout and a
    /// rollback is doing nothing.
    ///
    /// The price is that `commit_wave` DOES change which tensor is live, so a
    /// resolved address is valid for one wave only. That is what the engine
    /// already does — `build_wave_table` resolves the pointers once per forward
    /// (`qwen35/forward.rs`), inside the wave that uses them.
    #[test]
    fn a_wave_leaves_the_entering_buffer_untouched() {
        let mut store = filled_store();
        // Shares storage with the slot's entering state — the same view the
        // decode kernel's pointer table holds for this wave.
        let alias = store.layer_state(0).unwrap().s.clone();
        let entry: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();

        store.begin_wave().unwrap();
        {
            let (live, out) = store.layer_state_pair_mut(0).unwrap();
            bump_into(live, &out);
        }
        let during: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            during, entry,
            "the wave wrote into the buffer it was reading — the entering state \
             is gone and a rollback has nothing to return to"
        );

        store.rollback_wave().unwrap();
        let after: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(after, entry, "rollback must leave the entering state alone");

        // And on the committing path the swap installs the wave's output.
        store.begin_wave().unwrap();
        {
            let (live, out) = store.layer_state_pair_mut(0).unwrap();
            bump_into(live, &out);
        }
        store.commit_wave();
        let committed: Vec<f32> = store
            .layer_state(0)
            .unwrap()
            .s
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_ne!(committed, entry, "commit must install the wave's output");
    }

    #[test]
    fn wave_sequencing_is_enforced() {
        let mut store = filled_store();
        assert!(store.rollback_wave().is_err(), "rollback with no wave open");
        store.begin_wave().unwrap();
        assert!(store.begin_wave().is_err(), "overlapping wave");
        assert!(store.export().is_err(), "export mid-wave");
        store.commit_wave();
        assert!(store.export().is_ok());
    }

    /// A fork carries the parent's state exactly. Byte equality through
    /// `export`, not a tolerance: this is a memory copy, and a tolerance would
    /// hide a layout bug behind "close enough".
    #[test]
    fn fork_carries_the_parents_state_exactly() {
        let parent = filled_store();
        let child = parent.fork_from().unwrap();
        assert_eq!(
            child.export().unwrap(),
            parent.export().unwrap(),
            "the fork must carry the parent's state byte for byte"
        );
        assert_eq!(child.schedule_hash(), parent.schedule_hash());
        assert_eq!(child.n_recurrent_layers(), parent.n_recurrent_layers());
    }

    /// **The `Clone`-shares-storage hazard, on both halves of the ping-pong.**
    ///
    /// `Tensor::clone` is a shallow handle clone, so a fork built from clones
    /// would look right and then track every mutation. The live half is the
    /// obvious one. The write half matters just as much and is easier to miss:
    /// `layer_state_pair` hands out a [`DeltaNetOut`] whose tensors are clones
    /// of `backup`'s, so a fork that shared it would read correct until the
    /// child's first commit swapped that buffer into the parent's live position.
    #[test]
    fn fork_buffers_are_distinct_allocations_on_both_halves() {
        let parent = filled_store();
        let mut child = parent.fork_from().unwrap();
        let parent_before = parent.export().unwrap();

        // Live half: mutate the child, the parent must not move.
        {
            let live = child.layer_state_mut(0).unwrap();
            let ones = Tensor::ones(live.s.shape(), live.s.dtype(), &Device::Cpu).unwrap();
            live.s.add_mut(&ones).unwrap();
            live.conv_tail
                .add_mut(
                    &Tensor::ones(live.conv_tail.shape(), live.conv_tail.dtype(), &Device::Cpu)
                        .unwrap(),
                )
                .unwrap();
        }
        assert_eq!(
            parent.export().unwrap(),
            parent_before,
            "the child shares the parent's LIVE buffer"
        );

        // Write half: writing the child's `backup` must not reach the parent's.
        //
        // Both write buffers are STAMPED to a known value first. `backup` is
        // allocated uninitialised (invariant 6 — the kernels overwrite it whole
        // before any read), so "is the parent's write buffer still zero?" is
        // not a question with an answer, and uninitialised f32 can hold NaN,
        // which compares unequal even to itself. The property under test is
        // aliasing — did the child's write move the parent's bytes? — and
        // stamping makes that the only thing the assertion can fail on.
        let (_, child_out) = child.layer_state_pair(0).unwrap();
        let (_, parent_out) = parent.layer_state_pair(0).unwrap();
        let stamp = |t: &Tensor, v: f32| {
            let full = Tensor::full(v, t.shape(), &Device::Cpu)
                .unwrap()
                .to_dtype(t.dtype())
                .unwrap();
            t.slice_set(&full, 0, 0).unwrap();
        };
        for (c, p) in [
            (&child_out.s, &parent_out.s),
            (&child_out.conv_tail, &parent_out.conv_tail),
        ] {
            stamp(c, 0.0);
            stamp(p, 0.0);
            stamp(c, 1.0);
            let parent_v: Vec<f32> = p.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                parent_v.iter().all(|&x| x == 0.0),
                "the child shares the parent's WRITE buffer — this reads correct \
                 until the child's first commit swaps it into the parent's live slot"
            );
        }
    }

    /// Mid-wave the advanced state is in `backup` and `live` is a wave behind,
    /// so a fork taken there is not merely racy — it copies the wrong buffer.
    #[test]
    fn fork_mid_wave_is_refused() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        let err = match store.fork_from() {
            Ok(_) => panic!("a mid-wave fork must be refused, not silently stale"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("mid-wave"), "{err}");
        store.commit_wave();
        assert!(store.fork_from().is_ok(), "a wave boundary is fine");
    }

    /// Forking must not mark the parent's slots `advanced`. Reading through
    /// `layer_state_pair_mut` would, and the parent's next commit would then
    /// swap in a write buffer no wave ever wrote — installing, on the layers
    /// the fork touched, whatever was there two waves ago.
    #[test]
    fn forking_does_not_disturb_the_parents_wave_bookkeeping() {
        let mut parent = filled_store();
        let entry = parent.export().unwrap();

        let _child = parent.fork_from().unwrap();

        // A wave that touches nothing: if the fork marked the slots advanced,
        // this commit swaps their untouched write buffers into live.
        parent.begin_wave().unwrap();
        parent.commit_wave();
        assert_eq!(
            parent.export().unwrap(),
            entry,
            "forking marked the parent's slots advanced, so a commit installed \
             a buffer no wave wrote"
        );
    }

    /// The fork reads `live` — the committed state — never the write buffer.
    ///
    /// Under the ping-pong "the current state" is whichever tensor `live`
    /// points at *after* the last commit's swap, so a fork taken between waves
    /// must see the wave's result, not the buffer it is about to reuse.
    #[test]
    fn fork_reads_the_committed_buffer_not_the_write_buffer() {
        let mut store = filled_store();
        let before = store.export().unwrap();

        // Run a wave properly: read `live`, write the pair's out-buffer.
        store.begin_wave().unwrap();
        {
            let (live, out) = store.layer_state_pair_mut(0).unwrap();
            bump_into(live, &out);
        }
        store.commit_wave();
        let after = store.export().unwrap();
        assert_ne!(after, before, "the wave advanced layer 0");

        let child = store.fork_from().unwrap();
        assert_eq!(
            child.export().unwrap(),
            after,
            "the fork read the pre-commit buffer — a child a wave behind its \
             parent, with every shape correct"
        );
    }

    /// A fork is seeded: its state was put there deliberately, so the first
    /// `offset == 0` wave must not reset it — once.
    #[test]
    fn a_fork_is_seeded_exactly_once() {
        let parent = filled_store();
        let mut child = parent.fork_from().unwrap();
        assert!(child.is_seeded(), "a fresh fork carries seeded state");
        assert!(child.take_seeded(), "the first read reports it");
        assert!(
            !child.take_seeded(),
            "and consumes it — a second offset-0 wave resets normally"
        );
        assert!(
            !RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu)
                .unwrap()
                .is_seeded(),
            "a fresh store holds the sequence-start value already"
        );
    }

    /// Import is a restore, so it seeds for the same reason a fork does.
    #[test]
    fn import_seeds_the_store() {
        let store = filled_store();
        let exported = store.export().unwrap();
        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        assert!(!fresh.is_seeded());
        fresh.import(store.schedule_hash(), &exported).unwrap();
        assert!(
            fresh.is_seeded(),
            "a restored slot standing at offset 0 before its first wave looks \
             exactly like a fresh one — without the flag the reset wipes it"
        );
    }

    /// A refused import must not seed either: the store still holds zeros, and
    /// claiming otherwise would suppress the one reset that keeps it honest.
    #[test]
    fn a_refused_import_does_not_seed() {
        let store = filled_store();
        let exported = store.export().unwrap();
        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        assert!(fresh.import(store.schedule_hash() ^ 1, &exported).is_err());
        assert!(!fresh.is_seeded(), "a rejected restore seeded the store");
    }

    /// **A failed wave that reached only part of the stack leaves NO trace.**
    ///
    /// The composition behind `heal_tail_divergence`: when a wave fails, the
    /// recurrent rollback puts every layer back to its entry value and the KV
    /// heal trims the layers back to the offset the session actually delivered,
    /// so the two agree afterwards. This pins the recurrent half at its hardest
    /// point — a sweep that advanced layers 0 and 1 and never reached layer 3.
    ///
    /// Rolling back is doing nothing, so the risk is not that it fails to
    /// restore but that a later `commit_wave` swaps in a write buffer no wave
    /// wrote. The `advanced` flag is per slot precisely for this, and a partial
    /// sweep is the only shape that can catch it being per store.
    #[test]
    fn a_partial_sweep_that_rolls_back_leaves_every_layer_at_its_entry_value() {
        let mut store = filled_store();
        let entry = store.export().unwrap();

        // A wave that reaches layers 0 and 1 but dies before layer 3.
        store.begin_wave().unwrap();
        for li in [0usize, 1] {
            let (live, out) = store.layer_state_pair_mut(li).unwrap();
            bump_into(live, &out);
        }
        store.rollback_wave().unwrap();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "a rolled-back partial sweep moved the state"
        );

        // And the next wave must not inherit the dead one's bookkeeping: a
        // commit here would swap layers 0 and 1's write buffers — still holding
        // the failed wave's output — into live if `advanced` had survived.
        store.begin_wave().unwrap();
        store.commit_wave();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "a later commit installed the FAILED wave's output — `advanced` \
             outlived the rollback"
        );
    }

    /// A partial sweep that COMMITS advances exactly the layers it reached, and
    /// leaves the rest alone. The mirror of the test above: together they pin
    /// that `advanced` tracks the sweep rather than the store.
    #[test]
    fn a_partial_sweep_that_commits_advances_only_the_layers_it_reached() {
        let mut store = filled_store();
        let entry = store.export().unwrap();

        store.begin_wave().unwrap();
        {
            let (live, out) = store.layer_state_pair_mut(0).unwrap();
            bump_into(live, &out);
        }
        store.commit_wave();

        let after = store.export().unwrap();
        assert_ne!(after[0].state, entry[0].state, "layer 0 advanced");
        assert_eq!(
            after[1].state, entry[1].state,
            "layer 1 was never reached and must not have moved"
        );
        assert_eq!(after[2].state, entry[2].state, "nor layer 3");
    }

    /// **The resume oracle.** Seal → drop the store entirely → resume from the
    /// exported rows → the state is bit-identical.
    ///
    /// Byte equality, not a tolerance. This is a memory copy end to end, and a
    /// tolerance would hide exactly the layout bug the test exists to catch:
    /// a state scattered into the wrong slots reads as "close" and is wrong.
    #[test]
    fn seal_drop_resume_restores_a_bit_identical_state() {
        let sealed = {
            let store = filled_store();
            (store.schedule_hash(), store.export().unwrap())
        }; // the store is dropped here — nothing of it survives but the bytes

        let mut resumed = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        resumed.import(sealed.0, &sealed.1).unwrap();
        assert_eq!(
            resumed.export().unwrap(),
            sealed.1,
            "the resumed state must be byte-identical to the sealed one"
        );
    }

    /// A resume under a different model or a changed layer schedule refuses and
    /// leaves the store untouched, so the caller recomputes from a known state
    /// rather than from a half-scattered foreign one.
    #[test]
    fn resume_under_a_foreign_schedule_refuses_and_changes_nothing() {
        let store = filled_store();
        let exported = store.export().unwrap();

        let mut fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        let zeros = fresh.export().unwrap();
        assert!(fresh.import(store.schedule_hash() ^ 1, &exported).is_err());
        assert_eq!(fresh.export().unwrap(), zeros, "the refusal is total");
        assert!(!fresh.is_seeded(), "and it does not claim to be seeded");
    }

    /// Resume, then fork: a restored conversation forks exactly like a live
    /// one. This is the daemon-restart path — resume the timeline, then carve a
    /// view for the first turn — and it must not depend on the state having
    /// arrived by wave rather than by import.
    #[test]
    fn a_resumed_store_forks_like_a_live_one() {
        let sealed = {
            let store = filled_store();
            (store.schedule_hash(), store.export().unwrap())
        };
        let mut resumed = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        resumed.import(sealed.0, &sealed.1).unwrap();

        let child = resumed.fork_from().unwrap();
        assert_eq!(child.export().unwrap(), sealed.1);
        assert!(child.is_seeded());
    }

    #[test]
    fn schedule_hash_pins_layout() {
        let h = schedule_hash(&kinds(), &dims());
        assert_eq!(h, schedule_hash(&kinds(), &dims()), "deterministic");
        let mut other = kinds();
        other[2] = LayerKind::DeltaNet;
        assert_ne!(h, schedule_hash(&other, &dims()), "schedule change");
        let mut d2 = dims();
        d2.conv_kernel = 4;
        assert_ne!(h, schedule_hash(&kinds(), &d2), "dims change");
    }

    /// The whole point of the geometry price: it answers the same for a process
    /// holding no stores as for one holding a hundred, because it never looks at
    /// them. Three DeltaNet layers, two 512 B slots each (a 256 B `s` and a 256 B
    /// conv tail), is 3,072 B.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_store_is_priced_from_geometry_not_from_what_is_resident() {
        let (s_bytes, conv_bytes) = DeltaNetState::byte_sizes(&dims());
        assert_eq!((s_bytes, conv_bytes), (256, 256), "the fixture's halves");
        assert_eq!(state_block(&dims()), (256, 512), "one slot per layer state");

        let priced = RecurrentStateStore::reserved_bytes_for(&kinds(), &dims());
        assert_eq!(priced, 3 * 2 * 512, "six slots");

        // Nothing in the call depends on a store existing, so building some
        // cannot move it. This is the property the scheduler relies on.
        let _live: Vec<_> = (0..4)
            .map(|_| RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap())
            .collect();
        assert_eq!(
            RecurrentStateStore::reserved_bytes_for(&kinds(), &dims()),
            priced,
            "four live stores must not change what one store costs"
        );
    }

    /// An empty conv tail costs nothing beyond its `s`, and the price is slots, not
    /// regions. With `conv_kernel = 1` each layer state is a bare 6 MiB `s`, so a
    /// layer is two 6 MiB slots however the arenas they land in are shared.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_empty_conv_tail_costs_nothing_and_the_price_is_slots() {
        let d = DeltaNetDims {
            head_dim: 64,
            n_k_heads: 1,
            n_v_heads: 384,
            conv_kernel: 1,
        };
        let (s_bytes, conv_bytes) = DeltaNetState::byte_sizes(&d);
        assert_eq!((s_bytes, conv_bytes), (6 * 1024 * 1024, 0), "6 MiB, empty");
        assert_eq!(state_block(&d), (6 * 1024 * 1024, 6 * 1024 * 1024));

        let one = vec![LayerKind::DeltaNet];
        assert_eq!(
            RecurrentStateStore::reserved_bytes_for(&one, &d),
            12 * 1024 * 1024,
            "two 6 MiB slots"
        );

        let two = vec![LayerKind::DeltaNet, LayerKind::DeltaNet];
        assert_eq!(
            RecurrentStateStore::reserved_bytes_for(&two, &d),
            24 * 1024 * 1024,
            "four 6 MiB slots"
        );

        // An attention-only stack carries no recurrent state and costs nothing.
        assert_eq!(
            RecurrentStateStore::reserved_bytes_for(&[LayerKind::Attention], &d),
            0,
            "no DeltaNet layers, no reservation"
        );
    }
}
