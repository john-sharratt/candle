//! The recurrent state store: one sequence's DeltaNet memory across all
//! recurrent layers — each layer's state, the timeline beside it, and the
//! prefill scratch it holds while it prefills — with wave-atomic advance and the
//! export/import bridge the turn-seal snapshot record is built from
//! (`docs/deltanet_timeline.md`).
//!
//! # Wave atomicity
//!
//! The engine's relief design fails waves on purpose, and a failed wave must
//! leave no trace (`rollback_wave_kv` truncates KV; this store is the recurrent
//! analogue). Nothing a wave runs writes a layer's live state:
//!
//! - A **short span** — at most [`TIMELINE_ROWS`] rows: a decode row, a
//!   speculative verify block, a short prompt — reads the state and writes the
//!   layer's **timeline**, the generators its tokens add. The layer is then
//!   **pending**: its timeline holds rows nothing has applied yet. A **fold**
//!   applies the rows the step keeps, in place; a **discard** drops them.
//! - A **long span** — a prefill chunk — reads the state and writes the layer's
//!   **scratch**, a second slot the store holds only while the sequence
//!   prefills, and `commit_wave` exchanges the two.
//!
//! ```text
//!   begin_wave()      bookkeeping only
//!   … short spans write timelines, long spans write the scratch …
//!   commit_wave()     exchange every layer a long span advanced
//!   rollback_wave()   drop what this wave wrote — S was never touched
//!   fold / discard    end pending, between waves
//! ```
//!
//! **Pending is per layer, and outlives the wave.** A step opens and commits a
//! store's wave once per *segment*, each segment writing a disjoint range of
//! layers, and a speculative block's layers stay pending across those commits
//! until the step knows how many of its rows to keep. A layer handed a second
//! timeline while one is pending refuses — the overlapping-wave refusal, at the
//! layer. Everything that reads the store as a whole between steps — export,
//! fork, relocation, compaction — refuses a pending store, so between steps no
//! layer holds rows nothing has applied.
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
use std::sync::Arc;

#[cfg(feature = "cuda")]
use candle::{
    cuda_backend::cudarc::driver::result::memcpy_dtod_async, quantized::pinned_staging::PinnedBuf,
    CudaDevice, DType, LeaseAnchor, Storage,
};
#[cfg(feature = "cuda")]
use std::sync::Mutex;

/// The pinned host buffer every store's seal readback lands in, kept between
/// seals and grown only when a store needs more than it holds.
///
/// One per process rather than one per store: the scheduler seals one turn at a
/// time on its loop thread, so a single buffer serves them all, where a buffer per
/// conversation would pin ~65 MB of host memory for every sequence alive. It never
/// shrinks while an engine runs, and has nothing to shrink from: a store's size is
/// its model's geometry, identical for every sequence, so the buffer settles at one
/// store's size on the first seal. An engine's shutdown frees it
/// ([`release_seal_readback`]).
#[cfg(feature = "cuda")]
static READBACK: Mutex<Option<PinnedBuf>> = Mutex::new(None);

/// Free the seal readback buffer ([`READBACK`]), unpinning its pages.
///
/// The buffer is process-wide, so it outlives the engine whose seals sized it:
/// without this, a shut-down engine leaves one store's whole recurrent state —
/// 19.3 MiB on the 0.8B, ~65 MB on the 35B-A3B — pinned for the life of the
/// process. Called once the engine's scheduler, the only thread that seals, has
/// joined; the next engine's first seal allocates it again.
pub fn release_seal_readback() {
    #[cfg(feature = "cuda")]
    release_readback_in(&READBACK);
}

/// Empty `slot`, freeing its buffer outside the lock (the drop is a
/// `cuMemFreeHost`). A poisoned lock still releases: the buffer is only ever
/// replaced whole, so a panicking holder cannot leave it half-written.
#[cfg(feature = "cuda")]
fn release_readback_in(slot: &Mutex<Option<PinnedBuf>>) {
    let held = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
    drop(held);
}
use candle::{Device, Error, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{
    arena_regions, claim_arena_slots, plan_slot_moves, slot_stride, ArenaSlot, SlotTenant,
    SpanRegion, SLOT_ALIGN,
};

#[cfg(feature = "cuda")]
use super::cuda::{delta_net_fold, fold_runs_fused, FoldEntry};
use super::mix::{DeltaNetState, SpanWrite};
use super::timeline::{fold_by_ops, TimelineLayout, TIMELINE_ROWS};
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

/// A checked snapshot held for a sequence that has no store yet — installed
/// into one when the sequence's state is first wanted. The layers are shared:
/// every slot opened on one prompt branch holds the same rows.
#[derive(Debug, Clone)]
pub struct StagedSnapshot {
    pub schedule_hash: u64,
    pub layers: Arc<[ExportedLayerState]>,
}

impl StagedSnapshot {
    /// A store of `layer_kinds`/`dims` holding this snapshot, marked seeded so
    /// its first wave keeps it.
    pub fn build(
        &self,
        layer_kinds: &[LayerKind],
        dims: &DeltaNetDims,
        device: &Device,
    ) -> Result<RecurrentStateStore> {
        let mut store = RecurrentStateStore::new(layer_kinds, dims, device)?;
        store.import(self.schedule_hash, &self.layers)?;
        Ok(store)
    }
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

/// One slot's worth of a layer: the state and the timeline after it, and — on
/// CUDA — the state-arena slot they are views into.
struct Half {
    state: DeltaNetState,
    timeline: Tensor,
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
    held: Option<Arc<ArenaSlot>>,
}

impl Half {
    /// Buffers off any reservation — a CPU device, or a CUDA build running on
    /// one. A state that is read at zero (a sequence start) is zeroed; a scratch,
    /// which a long span overwrites whole before anything reads it, is not
    /// (hot-path invariant 6). Nor is the timeline: a fold reads only the rows a
    /// span wrote.
    fn host(dims: &DeltaNetDims, device: &Device, zeroed: bool) -> Result<Self> {
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
    fn in_slot(dims: &DeltaNetDims, device: &Device, slot: Arc<ArenaSlot>) -> Result<Self> {
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
struct Pending {
    /// The span's rows — every row its timeline holds.
    len: usize,
    /// Written by the wave now open, so its rollback drops them. A wave that
    /// committed leaves its rows pending with this cleared: a later segment's
    /// rollback does not reach them.
    this_wave: bool,
}

/// Per-layer slot: the live half, the prefill scratch, and the layer's
/// bookkeeping.
struct LayerSlot {
    /// Trunk layer index (recurrent layers only — attention layers have no
    /// slot here).
    layer_index: usize,
    /// The state as it stands, and its timeline.
    live: Half,
    /// Where a long span writes the advanced state — held only while the
    /// sequence prefills ([`RecurrentStateStore::ensure_scratch`]).
    scratch: Option<Half>,
    /// The timeline's rows nothing has applied yet.
    pending: Option<Pending>,
    /// Whether this wave handed the layer to a long span, i.e. whether the
    /// scratch holds an advanced state that commit should install.
    ///
    /// Per slot, not per store, because a sweep may cover only part of the
    /// stack: exchanging a layer the wave never ran would install whatever its
    /// scratch held two waves ago.
    advanced: bool,
}

/// One sequence's recurrent memory across every DeltaNet layer.
pub struct RecurrentStateStore {
    dims: DeltaNetDims,
    hash: u64,
    slots: Vec<LayerSlot>,
    /// Whether a wave is open.
    ///
    /// One flag for the store rather than one per slot: the wave operations act
    /// on every slot together, so a per-slot answer could only ever disagree
    /// with its neighbours by being wrong.
    open: bool,
    /// Whether the wave open — or, after its commit, the last one — handed any
    /// layer to a span, and whether any of them was long. A commit that ran
    /// layers and no long span releases the scratch: the sequence is no longer
    /// prefilling. One that ran no layer at all — a segment whose window holds
    /// only attention layers — says nothing either way and keeps it.
    ran_this_wave: bool,
    long_this_wave: bool,
    /// Whether this store's state was put here deliberately — by a fork or a
    /// restore — and must therefore survive one `offset == 0` reset.
    ///
    /// Store-level, unlike `advanced`, because seeding is a property of where
    /// the whole state came from rather than of which layers a sweep reached.
    seeded: bool,
    /// The step open on this store, if any — see [`Self::begin_step`].
    step: Option<StepRecord>,
    device: Device,
}

/// What a step's waves left pending, so the step can be kept or taken back
/// whole — see [`RecurrentStateStore::begin_step`].
#[derive(Debug)]
struct StepRecord {
    /// Per slot: whether it held rows when the step began. They are not the
    /// step's — an undo leaves them — and no span of the step ran that layer,
    /// because a pending layer refuses one.
    entered_pending: Vec<bool>,
    /// A commit inside the step owed this store a fold — it is outside the
    /// armed verify cohort — which the step's end runs instead.
    fold_owed: bool,
    /// Something an undo cannot take back happened inside the step: a long
    /// span's exchange, a fold, or the live state written outside a wave.
    broken: bool,
}

/// Where one layer's half sits in its arena slot: `s` at the start, the conv
/// tail after it on the next [`SLOT_ALIGN`] boundary, the timeline after that on
/// the next.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StateBlock {
    /// The conv tail's offset.
    conv_off: usize,
    /// Where the state and its tail end — the bytes a copy of the state moves.
    state_end: usize,
    /// The timeline's offset.
    timeline_off: usize,
    /// What a slot for this geometry must hold.
    bytes: usize,
}

#[cfg(feature = "cuda")]
fn state_block(dims: &DeltaNetDims) -> StateBlock {
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
fn claim_layer_states(
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
/// **Between steps, and a no-op while any store has a wave open or a layer
/// pending.** A relocation rebuilds a half's tensors on a new slot, and an open
/// wave has already resolved the old addresses into its pointer tables; between
/// steps nothing holds one — the tables are rebuilt every forward — so a moved
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
        let stores: Vec<&mut RecurrentStateStore> = stores.into_iter().collect();
        if stores.iter().any(|s| s.open || s.has_pending()) {
            return Ok(RecurrentCompaction::default());
        }
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

/// Fold every pending layer of every store in `jobs`, each store keeping the
/// first `kept` rows of its timelines — or every row, for `None` — and end
/// pending on all of them.
///
/// **One launch for the whole set** on CUDA (`delta_net_fold`): a step folds
/// every sequence it ran, every recurrent layer of each, together. A store
/// keeping `Some(0)` rows folds nothing and simply drops its timelines — a span
/// that committed nothing.
///
/// Between waves: a store with a wave open refuses, because its timelines may
/// still be being written. A forward's own fold goes through [`commit_stores`].
pub fn fold_stores(jobs: &mut [(&mut RecurrentStateStore, Option<usize>)]) -> Result<()> {
    if jobs.iter().any(|(s, _)| s.open) {
        candle::bail!(
            "recurrent store: a fold with a wave open — the wave may still be writing the \
             timelines it would apply"
        );
    }
    fold_pending(jobs)
}

/// Close a successful wave over `stores`: fold every pending row of each store
/// marked `true` — every short span of a sequence outside the armed verify
/// cohort, in ONE launch over all of them — then commit every store's wave.
///
/// **The fold runs first, with the waves still open**, because it is the one
/// step of the close that can fail: a fold that is refused writes nothing, so
/// the caller rolls the waves back and the stores stand exactly as they
/// entered. Once it has run, the commits are infallible host bookkeeping, so a
/// wave is never reported failed over state it already moved.
///
/// **A store inside a step folds nothing here** ([`RecurrentStateStore::begin_step`]):
/// its rows stay pending, and the fold it is owed runs at [`end_steps`], so an
/// undo can still drop them.
pub fn commit_stores(stores: &mut [(&mut RecurrentStateStore, bool)]) -> Result<()> {
    let mut folds: Vec<(&mut RecurrentStateStore, Option<usize>)> = Vec::new();
    for (store, fold) in stores.iter_mut() {
        if !*fold || !store.has_pending() {
            continue;
        }
        match store.step.as_mut() {
            Some(step) => step.fold_owed = true,
            None => folds.push((&mut **store, None)),
        }
    }
    fold_pending(&mut folds)?;
    for (store, _) in stores.iter_mut() {
        store.commit_wave();
    }
    Ok(())
}

/// Close the step on every store in `stores` and keep its waves: each store a
/// commit inside the step owed a fold folds every pending row, in ONE launch
/// over all of them. A store whose rows belong to the armed verify cohort was
/// owed none, and its rows stay pending for the reconcile.
///
/// Between waves, like [`fold_stores`]. A refused fold writes nothing and the
/// rows stay pending; the steps are closed either way.
pub fn end_steps(stores: &mut [&mut RecurrentStateStore]) -> Result<()> {
    if stores.iter().any(|s| s.open) {
        candle::bail!("recurrent store: a step closed with a wave open");
    }
    let mut folds: Vec<(&mut RecurrentStateStore, Option<usize>)> = Vec::new();
    for store in stores.iter_mut() {
        let owed = store.step.take().is_some_and(|s| s.fold_owed);
        if owed && store.has_pending() {
            folds.push((&mut **store, None));
        }
    }
    fold_pending(&mut folds)
}

/// [`fold_stores`]' body, for a caller that knows no wave is still writing.
/// A fold moves the live state, so a store inside a step can no longer undo it.
fn fold_pending(jobs: &mut [(&mut RecurrentStateStore, Option<usize>)]) -> Result<()> {
    for (store, _) in jobs.iter_mut() {
        store.break_step();
    }
    let Some((first, _)) = jobs.first() else {
        return Ok(());
    };
    let dims = first.dims;
    #[cfg(feature = "cuda")]
    let fused = matches!(first.device, Device::Cuda(_))
        && fold_runs_fused(&dims)
        && jobs
            .iter()
            .all(|(s, _)| s.slots.iter().all(|l| l.live.held.is_some()));
    #[cfg(feature = "cuda")]
    if fused {
        let mut entries: Vec<FoldEntry> = Vec::new();
        for (store, kept) in jobs.iter() {
            if store.dims != dims {
                candle::bail!("recurrent store: one fold across stores of different geometry");
            }
            entries.extend(store.fold_entries(*kept));
        }
        let device = first.device.clone();
        delta_net_fold(&entries, &dims, &device)?;
        for (store, _) in jobs.iter_mut() {
            store.discard_pending();
            store.assert_state("gdn.fold.s.L", "gdn.fold.tail.L");
        }
        return Ok(());
    }
    for (store, kept) in jobs.iter_mut() {
        let dims = store.dims;
        for slot in store.slots.iter_mut() {
            let Some(p) = slot.pending.take() else {
                continue;
            };
            let m = kept.map_or(p.len, |k| k.min(p.len));
            if m > 0 {
                fold_by_ops(&mut slot.live.state, &slot.live.timeline, m, &dims)?;
            }
        }
        // The state a fold lands is a state no wave computed directly — the
        // exit of the span that wrote the timeline — so it is checked here.
        store.assert_state("gdn.fold.s.L", "gdn.fold.tail.L");
    }
    Ok(())
}

/// Zero one layer's state and conv tail in its slot — the live half, which is
/// genuinely READ at zero, being the sequence-start state (hot-path invariant
/// 6's read-before-write case). The timeline is not touched: nothing reads it
/// before a span writes it.
///
/// Needed because a slot is recycled: it last held some other sequence's state.
#[cfg(feature = "cuda")]
fn zero_state(dims: &DeltaNetDims, device: &Device, slot: &ArenaSlot) -> Result<()> {
    slot.zero(state_block(dims).state_end, device)
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
        // On the device's stream, behind the writes that produced the source
        // and ahead of every launch that reads the copy.
        let stream = cuda.cuda_stream();
        // SAFETY: both ranges are `bytes` long, live, and disjoint — the
        // destination belongs to a store being built, which nothing else has
        // yet seen.
        unsafe {
            memcpy_dtod_async(dst_ptr, src_ptr, bytes, stream.cu_stream())
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

/// The slot of the store's layer `layer_index`, or the error naming it.
fn no_slot(layer_index: usize) -> Error {
    Error::Msg(format!(
        "recurrent store: layer {layer_index} holds no recurrent state"
    ))
}

impl RecurrentStateStore {
    /// Fresh zeros for every recurrent layer in `layer_kinds`.
    pub fn new(layer_kinds: &[LayerKind], dims: &DeltaNetDims, device: &Device) -> Result<Self> {
        let mut slots = Vec::new();
        // On CUDA every layer is a state-arena slot inside the device reservation,
        // its state zeroed — it is genuinely READ at zero, being the
        // sequence-start state, and a slot is recycled from whatever sequence held
        // it last. The cfg gates COMPILATION; the device gates behaviour. A CUDA
        // build still runs on a CPU device — every unit test here does — and there
        // is no reservation there to carve from.
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
            if *k != LayerKind::DeltaNet {
                continue;
            }
            #[cfg(feature = "cuda")]
            let live = match claimed.as_mut() {
                Some(it) => {
                    let slot = it.next().expect("one slot was claimed per recurrent layer");
                    zero_state(dims, device, &slot)?;
                    Half::in_slot(dims, device, slot)?
                }
                None => Half::host(dims, device, true)?,
            };
            #[cfg(not(feature = "cuda"))]
            let live = Half::host(dims, device, true)?;
            slots.push(LayerSlot {
                layer_index: i,
                live,
                scratch: None,
                pending: None,
                advanced: false,
            });
        }
        Ok(Self {
            dims: *dims,
            hash: schedule_hash(layer_kinds, dims),
            slots,
            open: false,
            ran_this_wave: false,
            long_this_wave: false,
            // A fresh store already holds the sequence-start value, so there is
            // nothing for a reset to destroy.
            seeded: false,
            step: None,
            device: device.clone(),
        })
    }

    /// Back to the sequence-start state, on the slots this store already holds.
    ///
    /// Each layer's live state is zeroed where it stands. This is how a sequence
    /// starting over gets its zeros — not by dropping the store and building
    /// another. A store's slots go back to their arena only once every tensor
    /// viewing them has gone, so a rebuild can ask for a second store's worth of
    /// slots while the first is still held; on a span with no room to spare that
    /// claim fails part-way, and the turn that was only restarting fails with it.
    ///
    /// Called before a wave opens, so a store still marked open here is one
    /// whose last wave was never closed — a failure that skipped its rollback.
    /// That wave is rolled back first, every pending timeline is dropped, and the
    /// prefill scratch goes: the restart goes ahead on a store that was otherwise
    /// left unusable.
    pub fn reset(&mut self) -> Result<()> {
        if self.open {
            self.rollback_wave()?;
        }
        self.break_step();
        self.discard_pending();
        self.release_scratch();
        for slot in &mut self.slots {
            slot.advanced = false;
            #[cfg(feature = "cuda")]
            if let Some(held) = slot.live.held.as_ref() {
                zero_state(&self.dims, &self.device, held)?;
                continue;
            }
            slot.live.state = DeltaNetState::zeros(&self.dims, &self.device)?;
        }
        self.seeded = false;
        Ok(())
    }

    pub fn schedule_hash(&self) -> u64 {
        self.hash
    }

    pub fn dims(&self) -> &DeltaNetDims {
        &self.dims
    }

    /// Trunk layer indices of the recurrent layers, in slot order — what the
    /// decode pointer table iterates to collect every layer's state address.
    pub fn recurrent_layer_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots.iter().map(|s| s.layer_index)
    }

    pub fn n_recurrent_layers(&self) -> usize {
        self.slots.len()
    }

    fn slot(&self, layer_index: usize) -> Result<&LayerSlot> {
        self.slots
            .iter()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| no_slot(layer_index))
    }

    fn slot_mut(&mut self, layer_index: usize) -> Result<&mut LayerSlot> {
        self.slots
            .iter_mut()
            .find(|s| s.layer_index == layer_index)
            .ok_or_else(|| no_slot(layer_index))
    }

    /// The live state of trunk layer `layer_index`. Errors on an attention
    /// layer's index.
    pub fn layer_state(&self, layer_index: usize) -> Result<&DeltaNetState> {
        Ok(&self.slot(layer_index)?.live.state)
    }

    /// Trunk layer `layer_index`'s live state, to be **written into** —
    /// **outside a wave only**.
    ///
    /// This is the in-place form, and a wave must not use it: a wave advances a
    /// layer through [`Self::layer_span_mut`], which leaves the live state as it
    /// entered until a fold or a commit installs the advance; writing it directly
    /// destroys the entering state a failed wave returns to. What legitimately
    /// uses this is code holding a store no wave is open on — the verification
    /// path builds a fresh single-sequence store per block and advances it
    /// directly.
    ///
    /// There is deliberately no setter: a store that could be handed a
    /// *different* tensor is one where prefill and decode end up advancing the
    /// state two different ways.
    pub fn layer_state_mut(&mut self, layer_index: usize) -> Result<&mut DeltaNetState> {
        self.break_step();
        Ok(&mut self.slot_mut(layer_index)?.live.state)
    }

    /// Every recurrent layer's decode-table addresses, in slot order:
    /// `[tail, state, timeline]` of the live half — the three rows
    /// [`build_wave_table`] lays out per layer — without marking any layer
    /// (a sweep may run only part of the stack, and a layer handed a timeline
    /// is one that ran).
    ///
    /// **Read off the slots, not the tensors.** Each half is a view built by
    /// `Half::in_slot` at its slot's base — `s` at the base, the conv tail and
    /// the timeline at their [`SLOT_ALIGN`] boundaries — and every place that
    /// rebuilds a half (`new`, `fork_from`, `relocate`) or exchanges two
    /// (`commit_wave`) moves `held` with it, so the slot always names the buffers
    /// the half stands on. Resolving a tensor's address takes its storage lock
    /// and a stream-guarded device pointer; the table needs three per layer per
    /// decode sequence per forward, which at 256 sequences on an 18-layer
    /// recurrent stack was milliseconds of host time with the device idle.
    ///
    /// Debug builds check every address against the tensor it describes, which
    /// is the one way the slots and the halves could part: a caller rebinding a
    /// half through `layer_state_mut` rather than writing into it.
    ///
    /// [`build_wave_table`]: super::cuda::build_wave_table
    #[cfg(feature = "cuda")]
    pub fn decode_table_addresses(&self) -> Result<impl Iterator<Item = [u64; 3]> + '_> {
        let block = state_block(&self.dims);
        let at = move |slot: &ArenaSlot| {
            [
                slot.ptr() + block.conv_off as u64,
                slot.ptr(),
                slot.ptr() + block.timeline_off as u64,
            ]
        };
        for slot in &self.slots {
            if slot.live.held.is_none() {
                candle::bail!(
                    "recurrent store: layer {} has no state-arena slot, so its decode \
                     addresses are not the slot's — a store built off a CUDA device",
                    slot.layer_index
                );
            }
            #[cfg(debug_assertions)]
            {
                let held = slot.live.held.as_ref().expect("checked above");
                let Device::Cuda(cuda) = &self.device else {
                    candle::bail!("recurrent store: arena slots on a non-CUDA device");
                };
                let want = [
                    tensor_device_ptr(cuda, &slot.live.state.conv_tail)?,
                    tensor_device_ptr(cuda, &slot.live.state.s)?,
                    tensor_device_ptr(cuda, &slot.live.timeline)?,
                ];
                let got = at(held);
                if got != want {
                    candle::bail!(
                        "recurrent store: layer {}'s slot names {got:x?} but its half sits \
                         at {want:x?} — a half was rebound instead of written into",
                        slot.layer_index
                    );
                }
            }
        }
        Ok(self
            .slots
            .iter()
            .map(move |slot| at(slot.live.held.as_ref().expect("checked above"))))
    }

    /// The layer's live state, to be read by a span of `len` rows, and where the
    /// span's work lands — and the record that this layer ran.
    ///
    /// - A span of at most [`TIMELINE_ROWS`] rows writes the layer's timeline,
    ///   which leaves the layer pending until a fold or a discard. A layer
    ///   already pending refuses: its rows have not been applied, and a second
    ///   span would read a state they have not reached.
    /// - A longer span writes the prefill scratch, which must already be held
    ///   ([`Self::ensure_scratch`], between forwards); `commit_wave` installs it.
    ///
    /// Taking this is what marks the layer, so a caller asks for it exactly when
    /// it is about to run the layer, never to peek.
    pub fn layer_span_mut(
        &mut self,
        layer_index: usize,
        len: usize,
    ) -> Result<(&mut DeltaNetState, SpanWrite)> {
        if !self.open {
            candle::bail!(
                "recurrent store: a span for layer {layer_index} with no wave open — a span's \
                 writes belong to a wave, which is what can roll them back"
            );
        }
        if len == 0 {
            candle::bail!("recurrent store: a span of no rows for layer {layer_index}");
        }
        let long = len > TIMELINE_ROWS;
        let at = self
            .slots
            .iter()
            .position(|s| s.layer_index == layer_index)
            .ok_or_else(|| no_slot(layer_index))?;
        let slot = &mut self.slots[at];
        if let Some(p) = slot.pending {
            candle::bail!(
                "recurrent store: layer {layer_index} still holds {} pending timeline rows — \
                 a {len}-row span would read a state they have not reached. Fold or discard \
                 them first.",
                p.len
            );
        }
        let write = if long {
            let Some(scratch) = slot.scratch.as_ref() else {
                candle::bail!(
                    "recurrent store: a {len}-row span for layer {layer_index} needs the \
                     prefill scratch, which the store does not hold — claim it before the \
                     forward (`ensure_scratch`)"
                );
            };
            slot.advanced = true;
            SpanWrite::State(scratch.state.write_half())
        } else {
            slot.pending = Some(Pending {
                len,
                this_wave: true,
            });
            SpanWrite::Timeline(slot.live.timeline.clone())
        };
        self.ran_this_wave = true;
        self.long_this_wave |= long;
        Ok((&mut self.slots[at].live.state, write))
    }

    /// Open a wave. Refuses while one is already open.
    ///
    /// **Costs nothing on the device.** The entering state is preserved by not
    /// being written: a short span writes its timeline and a long span the
    /// scratch, so opening a wave is bookkeeping and rolling one back is
    /// dropping that bookkeeping.
    pub fn begin_wave(&mut self) -> Result<()> {
        if self.open {
            candle::bail!(
                "recurrent store: begin_wave with a wave already open — overlapping \
                 waves on one session are exactly what atomicity forbids"
            );
        }
        for slot in &mut self.slots {
            slot.advanced = false;
            if let Some(p) = slot.pending.as_mut() {
                p.this_wave = false;
            }
        }
        self.ran_this_wave = false;
        self.long_this_wave = false;
        self.open = true;
        self.assert_state("gdn.entry.s.L", "gdn.entry.tail.L");
        Ok(())
    }

    /// Fold every layer's live state into the assert slots under `prefix`.
    ///
    /// **Entry and exit, because the pair is what says who broke it.** A wave
    /// writes nothing live, so the state at entry is whatever the last fold or
    /// commit left plus anything that has happened to the memory since. If entry
    /// is bad while the previous exit was good, nothing computed it — the bytes
    /// changed while the store sat idle, which is a foreign write into the span.
    /// If exit is the first bad site, the wave computed it, and the cause is
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
            slot.live.state.s.assert(site(s_prefix, l));
            slot.live.state.conv_tail.assert(site(tail_prefix, l));
        }
    }

    #[cfg(not(feature = "tensor-assert"))]
    #[inline]
    fn assert_state(&self, _s_prefix: &'static str, _tail_prefix: &'static str) {}

    /// The wave's writes stand: every layer a long span advanced exchanges its
    /// live half with the scratch, so what the span wrote becomes the state and
    /// what the state was becomes the next long span's scratch. Layers a short
    /// span ran stay pending — their timelines wait for a fold.
    ///
    /// A host exchange per advanced layer, and no device work at all. A wave that
    /// ran layers and handed none of them to a long span ends the sequence's
    /// prefill, and the scratch goes back; one that ran no layer — a segment of
    /// attention layers only — leaves it.
    pub fn commit_wave(&mut self) {
        // An exchange inside a step is not one an undo can take back.
        if self.slots.iter().any(|s| s.advanced) {
            self.break_step();
        }
        for slot in &mut self.slots {
            if let Some(p) = slot.pending.as_mut() {
                p.this_wave = false;
            }
            if slot.advanced {
                let scratch = slot
                    .scratch
                    .as_mut()
                    .expect("a layer advanced by a long span holds the scratch it wrote");
                // The whole half: the state, its tail and the timeline region of
                // the slot move together, so the live slot always carries a
                // timeline beside its state.
                std::mem::swap(&mut slot.live, scratch);
                slot.advanced = false;
            }
        }
        self.open = false;
        if self.ran_this_wave && !self.long_this_wave {
            self.release_scratch();
        }
        // After the exchange, so this is the state the NEXT wave will read — the
        // same buffers `gdn.entry` will fold on the way in. A pair that disagrees
        // across the gap between two waves is a write nothing in the forward
        // performed.
        self.assert_state("gdn.exit.s.L", "gdn.exit.tail.L");
    }

    /// The wave never happened.
    ///
    /// Nothing to undo on the device: a wave writes only timelines and the
    /// scratch, never a live state. The timelines this wave wrote are dropped;
    /// rows an earlier, committed segment left pending stay. Refuses when no
    /// wave is open (a rollback with nothing to roll back to is a sequencing
    /// bug, not a no-op).
    pub fn rollback_wave(&mut self) -> Result<()> {
        if !self.open {
            candle::bail!("recurrent store: rollback_wave with no wave open");
        }
        for slot in &mut self.slots {
            slot.advanced = false;
            if slot.pending.is_some_and(|p| p.this_wave) {
                slot.pending = None;
            }
        }
        self.open = false;
        Ok(())
    }

    /// Whether any layer holds timeline rows nothing has applied yet.
    pub fn has_pending(&self) -> bool {
        self.slots.iter().any(|s| s.pending.is_some())
    }

    /// Every pending layer and its span's rows, in slot order.
    pub fn pending_layers(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.slots
            .iter()
            .filter_map(|s| s.pending.map(|p| (s.layer_index, p.len)))
    }

    /// The fold entries for every pending layer, keeping the first `kept` rows
    /// of each timeline (every row for `None`). A layer keeping none has no
    /// entry.
    #[cfg(feature = "cuda")]
    fn fold_entries(&self, kept: Option<usize>) -> Vec<FoldEntry> {
        let block = state_block(&self.dims);
        self.slots
            .iter()
            .filter_map(|slot| {
                let p = slot.pending?;
                let m = kept.map_or(p.len, |k| k.min(p.len));
                let held = slot.live.held.as_ref()?;
                (m > 0).then(|| FoldEntry {
                    tail: held.ptr() + block.conv_off as u64,
                    state: held.ptr(),
                    timeline: held.ptr() + block.timeline_off as u64,
                    m,
                })
            })
            .collect()
    }

    /// Fold this store's pending layers — [`fold_stores`] for one store.
    pub fn fold(&mut self, kept: Option<usize>) -> Result<()> {
        fold_stores(&mut [(self, kept)])
    }

    /// Drop every pending timeline: the spans that wrote them are not kept, and
    /// the state stands as they found it.
    pub fn discard_pending(&mut self) {
        for slot in &mut self.slots {
            slot.pending = None;
        }
    }

    /// Hold the prefill scratch: a second slot per layer, where a long span
    /// writes its advanced state. **Between forwards** — a claim from inside a
    /// forward is refused by the arena — so a step's prepare pass takes it for
    /// every sequence it hands a long span. Held until a wave runs the sequence
    /// with no long span, or [`Self::release_scratch`].
    ///
    /// Not zeroed: a long span overwrites the scratch whole before anything
    /// reads it (hot-path invariant 6).
    pub fn ensure_scratch(&mut self) -> Result<()> {
        if self.open {
            candle::bail!("recurrent store: claiming the prefill scratch mid-wave");
        }
        let missing = self.slots.iter().filter(|s| s.scratch.is_none()).count();
        if missing == 0 {
            return Ok(());
        }
        // The scratch goes back when the sequence next runs with no long span,
        // so it may buy past the defended weight residency.
        #[cfg(feature = "cuda")]
        let mut claimed = {
            let _transient = candle_nn::kv_cache::TransientPurchase::begin();
            claim_layer_states(&self.dims, &self.device, missing)?.map(Vec::into_iter)
        };
        for slot in self.slots.iter_mut().filter(|s| s.scratch.is_none()) {
            #[cfg(feature = "cuda")]
            let half = match claimed.as_mut() {
                Some(it) => Half::in_slot(
                    &self.dims,
                    &self.device,
                    it.next().expect("one slot was claimed per missing scratch"),
                )?,
                None => Half::host(&self.dims, &self.device, false)?,
            };
            #[cfg(not(feature = "cuda"))]
            let half = Half::host(&self.dims, &self.device, false)?;
            slot.scratch = Some(half);
        }
        Ok(())
    }

    /// Give the prefill scratch back. Its slots return to their arena once no
    /// view of them remains.
    pub fn release_scratch(&mut self) {
        for slot in &mut self.slots {
            slot.scratch = None;
        }
    }

    /// Whether the store holds the prefill scratch.
    pub fn holds_scratch(&self) -> bool {
        self.slots.iter().any(|s| s.scratch.is_some())
    }

    /// Open a step: several waves that stand or fall together.
    ///
    /// **A decode step is swept in segments**, each its own wave, and each
    /// commits as it ends — so a refused second segment would roll back only its
    /// own layers and leave the first segment's advanced. The scheduler retries
    /// the step from layer 0, which would run those layers' tokens into their
    /// state a second time: a recurrent state that has absorbed the same token
    /// twice is wrong in a way nothing downstream can see, until it is far
    /// enough wrong to be NaN.
    ///
    /// **Free: the timeline is the undo.** Inside a step a commit folds nothing
    /// ([`commit_stores`]): every short span's rows stay pending, so each layer's
    /// live state is still the one the step entered with. [`end_steps`] folds
    /// them; [`Self::undo_step`] drops them. The segments of a step cover
    /// disjoint layers, and a pending layer refuses a second span, so no layer
    /// is advanced twice within one. Refused mid-wave.
    pub fn begin_step(&mut self) -> Result<()> {
        if self.open {
            candle::bail!("recurrent store: begin_step mid-wave");
        }
        self.step = Some(StepRecord {
            entered_pending: self.slots.iter().map(|s| s.pending.is_some()).collect(),
            fold_owed: false,
            broken: false,
        });
        Ok(())
    }

    /// Close the step and keep its waves — [`end_steps`] for one store.
    pub fn end_step(&mut self) -> Result<()> {
        end_steps(&mut [self])
    }

    /// The step never happened: every row a wave of the step left pending is
    /// dropped, so each layer stands at the state the step entered with.
    /// Refuses mid-wave, with no step open, and when the record cannot vouch
    /// for the undo — the state moved inside the step other than by a short
    /// span — closing the step either way.
    pub fn undo_step(&mut self) -> Result<()> {
        if self.open {
            candle::bail!("recurrent store: undo_step mid-wave");
        }
        let Some(step) = self.step.take() else {
            candle::bail!("recurrent store: undo_step with no step open");
        };
        if step.broken {
            candle::bail!(
                "recurrent store: the step cannot be undone — its state moved by an exchange, \
                 a fold or a write outside a wave"
            );
        }
        for (slot, entered) in self.slots.iter_mut().zip(step.entered_pending) {
            if !entered {
                slot.pending = None;
            }
        }
        Ok(())
    }

    /// Whatever just happened to the live state, the open step (if any) can no
    /// longer be undone by dropping its rows.
    fn break_step(&mut self) {
        if let Some(step) = self.step.as_mut() {
            step.broken = true;
        }
    }

    /// An independent store carrying this one's state — the fork primitive.
    ///
    /// Device-to-device: each layer's live `s` and conv tail are copied into the
    /// child's own slot. The child holds no scratch and nothing pending.
    ///
    /// Refused mid-wave and with a layer pending: either way the state this
    /// store will stand at is not yet the one in its live buffers, and the child
    /// would come up a span behind its parent with every shape correct.
    pub fn fork_from(&self) -> Result<Self> {
        if self.open {
            candle::bail!(
                "recurrent store: fork_from mid-wave — the wave's advance is not yet in \
                 the live state, so the child would come up stale. Fork at a wave boundary."
            );
        }
        if self.has_pending() {
            candle::bail!(
                "recurrent store: fork_from with timeline rows pending — fold or discard \
                 them first, or the child comes up a span behind its parent"
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
            #[cfg(feature = "cuda")]
            let live = match claimed.as_mut() {
                Some(it) => {
                    let half = Half::in_slot(
                        &self.dims,
                        &self.device,
                        it.next().expect("one slot was claimed per recurrent layer"),
                    )?;
                    // The fork's whole point: the child starts from the parent's
                    // state. A device-to-device copy into the child's own slot,
                    // rather than `snapshot()`, which would allocate a fresh pool
                    // buffer and hand back a tensor pointing outside the span.
                    copy_state_into(&self.device, &slot.live.state, &half.state)?;
                    half
                }
                None => Half {
                    state: slot.live.state.snapshot()?,
                    timeline: TimelineLayout::of(&self.dims).alloc(&self.device)?,
                    held: None,
                },
            };
            #[cfg(not(feature = "cuda"))]
            let live = Half {
                state: slot.live.state.snapshot()?,
                timeline: TimelineLayout::of(&self.dims).alloc(&self.device)?,
            };
            slots.push(LayerSlot {
                layer_index: slot.layer_index,
                live,
                scratch: None,
                pending: None,
                advanced: false,
            });
        }
        Ok(Self {
            dims: self.dims,
            hash: self.hash,
            slots,
            open: false,
            ran_this_wave: false,
            long_this_wave: false,
            seeded: true,
            step: None,
            device: self.device.clone(),
        })
    }

    /// Move every half of this store whose slot is the source of a planned move onto
    /// that move's destination, taking the destination out of `moves`; answers how
    /// many halves moved.
    ///
    /// Per half: the half's tensors rebuilt on the destination slot, the source slot
    /// dropped. A live half's state and conv tail are copied first — one device copy
    /// on the primary stream; its timeline is not, because nothing is pending
    /// between steps, and neither is a scratch, which a long span overwrites whole
    /// before reading. The drop is on the host while the copy may still be queued,
    /// which is sound for the reason `ArenaSlot` gives — the source's next tenant is
    /// ordered behind the copy on the same stream, or, if it empties its arena,
    /// behind the region pool's fence. Anything else still holding the old half
    /// keeps the old slot alive until it lets go.
    #[cfg(feature = "cuda")]
    pub fn relocate(&mut self, moves: &mut HashMap<u64, ArenaSlot>) -> Result<usize> {
        if self.open || self.has_pending() {
            candle::bail!(
                "recurrent store: relocate mid-wave or with timeline rows pending — the \
                 store's addresses are resolved or its timelines still unapplied"
            );
        }
        let Device::Cuda(cuda) = &self.device else {
            return Ok(0);
        };
        let stream = cuda.cuda_stream();
        let copy = state_block(&self.dims).state_end;
        let mut moved = 0usize;
        for slot in &mut self.slots {
            let halves = std::iter::once((&mut slot.live, true))
                .chain(slot.scratch.as_mut().map(|s| (s, false)));
            for (half, carries_state) in halves {
                let Some(at) = half.held.as_ref() else {
                    continue;
                };
                let Some(dst) = moves.remove(&at.ptr()) else {
                    continue;
                };
                if carries_state {
                    // SAFETY: both ranges are state-arena slots of at least `copy`
                    // bytes — the source held by this store, the destination claimed
                    // for this move — and they are distinct slots, so they do not
                    // overlap.
                    unsafe { memcpy_dtod_async(dst.ptr(), at.ptr(), copy, stream.cu_stream()) }
                        .map_err(|e| Error::Msg(format!("relocating recurrent state: {e}")))?;
                }
                *half = Half::in_slot(&self.dims, &self.device, Arc::new(dst))?;
                moved += 1;
            }
        }
        Ok(moved)
    }

    /// Reservation bytes this sequence's recurrent memory holds — its state-arena
    /// slots, live and scratch, at their stride. Zero off CUDA, where the buffers
    /// are ordinary allocations.
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
                .flat_map(|s| std::iter::once(&s.live).chain(s.scratch.as_ref()))
                .filter_map(|h| h.held.as_ref())
                .map(|s| s.stride())
                .sum()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Reservation bytes every recurrent-state arena on `device` holds, as whole
    /// regions, whatever is in them.
    ///
    /// **The accounting figure.** It is what recurrent state denies the rest of the
    /// span: the slots every store holds, the free slots and unused tails of their
    /// arenas, and any slot a handle kept after its store went. Memory nothing can
    /// total is memory that goes missing (`AccountingSection`), and summing stores
    /// would miss all but the first. Zero off CUDA.
    pub fn arena_reserved_bytes(device: &Device) -> usize {
        #[cfg(feature = "cuda")]
        {
            arena_regions(device, SlotTenant::RecurrentState) * SpanRegion::bytes()
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = device;
            0
        }
    }

    /// What one decoding sequence's store **will** reserve, from the geometry
    /// alone — [`Self::reserved_bytes`] for a store that does not exist yet and
    /// holds no scratch.
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
    /// The price is the store's slots at their stride: one per DeltaNet layer,
    /// each [`slot_stride`] of the layer's block (`s`, its conv tail and its
    /// timeline, each on a 256-byte boundary) — exactly what
    /// [`Self::reserved_bytes`] reports once the store stands. A prefilling
    /// sequence holds [`Self::scratch_bytes_for`] more. The arenas those slots
    /// live in are shared by every sequence of the geometry, so a region's unused
    /// tail is not charged to any one of them. Nothing here reads the device, so
    /// it answers on any backend and at any moment.
    pub fn reserved_bytes_for(layer_kinds: &[LayerKind], dims: &DeltaNetDims) -> usize {
        #[cfg(feature = "cuda")]
        {
            let layers = layer_kinds
                .iter()
                .filter(|k| **k == LayerKind::DeltaNet)
                .count();
            layers * slot_stride(state_block(dims).bytes)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (layer_kinds, dims);
            0
        }
    }

    /// What the prefill scratch adds to a sequence's store while it prefills —
    /// one more slot per DeltaNet layer.
    pub fn scratch_bytes_for(layer_kinds: &[LayerKind], dims: &DeltaNetDims) -> usize {
        Self::reserved_bytes_for(layer_kinds, dims)
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
    /// Refused mid-wave and with timeline rows pending: a snapshot must capture a
    /// sealed boundary, never a wave in flight or a span not yet applied.
    pub fn export(&self) -> Result<Vec<ExportedLayerState>> {
        if self.open {
            candle::bail!("recurrent store: export mid-wave — seal, then snapshot");
        }
        if self.has_pending() {
            candle::bail!(
                "recurrent store: export with timeline rows pending — fold or discard them, \
                 then snapshot"
            );
        }
        let d = &self.dims;
        let bytes = self.read_back()?;
        Ok(self
            .slots
            .iter()
            .zip(bytes)
            .map(|(slot, (state, conv_tail))| ExportedLayerState {
                layer_index: slot.layer_index as u32,
                n_v_heads: d.n_v_heads as u32,
                d_v: d.head_dim as u32,
                d_k: d.head_dim as u32,
                state,
                conv_channels: d.conv_dim() as u32,
                conv_tail_cols: (d.conv_kernel - 1) as u32,
                conv_tail,
            })
            .collect())
    }

    /// Every layer's live state and conv tail as LE F32 bytes, in slot order.
    ///
    /// **On CUDA, one synchronisation for the whole store.** Every tensor's bytes
    /// are enqueued into one pinned host buffer and the stream is waited on once:
    /// a readback per tensor was two synchronising copies per recurrent layer into
    /// freshly allocated pageable memory — 60 of them on the 35B-A3B, ~27 ms a seal
    /// at ~2.2 GB/s, on the scheduler's loop thread. The buffer is kept between
    /// seals ([`READBACK`]), so neither its pinning nor its first-touch faults are
    /// paid per turn.
    fn read_back(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        #[cfg(feature = "cuda")]
        if let Some(first) = self.slots.first() {
            if let Device::Cuda(dev) = first.live.state.s.device() {
                return self.read_back_cuda(dev);
            }
        }
        self.slots
            .iter()
            .map(|slot| {
                let state: Vec<f32> = slot.live.state.s.flatten_all()?.to_vec1()?;
                let tail: Vec<f32> = slot.live.state.conv_tail.flatten_all()?.to_vec1()?;
                Ok((f32_le_bytes(&state), f32_le_bytes(&tail)))
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn read_back_cuda(&self, dev: &CudaDevice) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let state_bytes = self.dims.state_elems() * 4;
        let tail_bytes = self.dims.conv_state_elems() * 4;
        let total = self.slots.len() * (state_bytes + tail_bytes);
        let mut held = READBACK.lock().unwrap_or_else(|e| e.into_inner());
        if held.as_ref().is_none_or(|b| b.len() < total) {
            // Plain pinned memory, not write-combined: the CPU reads it straight
            // back. Portable, because the buffer is shared by every store in the
            // process, whichever device holds it. Every copy into the buffer it
            // replaces has completed — each call below synchronises before it
            // returns, on success and on error alike.
            *held = Some(PinnedBuf::alloc_owned_portable(total)?);
        }
        let buf = held
            .as_mut()
            .expect("the readback buffer was just sized")
            .as_mut_slice();
        let stream = dev.cuda_stream();
        // Every copy enqueued, then one wait. The wait runs even when an enqueue
        // fails part-way: copies already queued are still writing into `buf`, and
        // returning before they land would hand the next seal a buffer under DMA.
        let enqueued = (|| -> Result<()> {
            let mut at = 0usize;
            for slot in &self.slots {
                for (tensor, bytes) in [
                    (&slot.live.state.s, state_bytes),
                    (&slot.live.state.conv_tail, tail_bytes),
                ] {
                    let (storage, layout) = tensor.storage_and_layout();
                    let Storage::Cuda(cuda) = &*storage else {
                        candle::bail!(
                            "recurrent store: a CUDA store holds a non-CUDA state tensor"
                        );
                    };
                    let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                        Error::Msg("recurrent store: a state tensor is not contiguous".into())
                    })?;
                    if (end - start) * 4 != bytes {
                        candle::bail!(
                            "recurrent store: layer {} holds {} bytes of state where its \
                             geometry says {bytes}",
                            slot.layer_index,
                            (end - start) * 4,
                        );
                    }
                    cuda.copy_f32_bytes_to_host_on_stream(
                        &mut buf[at..at + bytes],
                        &stream,
                        start,
                        end - start,
                    )?;
                    at += bytes;
                }
            }
            Ok(())
        })();
        let synced = stream
            .synchronize()
            .map_err(|e| Error::Msg(format!("recurrent store: readback sync: {e:?}")));
        enqueued?;
        synced?;
        let mut at = 0usize;
        Ok(self
            .slots
            .iter()
            .map(|_| {
                let state = buf[at..at + state_bytes].to_vec();
                at += state_bytes;
                let tail = buf[at..at + tail_bytes].to_vec();
                at += tail_bytes;
                (state, tail)
            })
            .collect())
    }

    /// Whether a snapshot fits a store of this geometry — its schedule hash, and
    /// every layer's index and dims — without building one. What [`Self::import`]
    /// checks before it touches a tensor, available to a caller that holds the
    /// snapshot back until a store is wanted, so a foreign layout is refused when
    /// it arrives rather than when it is first used.
    pub fn check_snapshot(
        layer_kinds: &[LayerKind],
        dims: &DeltaNetDims,
        snapshot_hash: u64,
        layers: &[ExportedLayerState],
    ) -> Result<()> {
        let hash = schedule_hash(layer_kinds, dims);
        if snapshot_hash != hash {
            candle::bail!(
                "recurrent store: snapshot schedule hash {snapshot_hash:#x} does not match \
                 this model's {hash:#x} — recompute the state instead of scattering a foreign \
                 layout"
            );
        }
        let recurrent: Vec<usize> = layer_kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::DeltaNet)
            .map(|(i, _)| i)
            .collect();
        if layers.len() != recurrent.len() {
            candle::bail!(
                "recurrent store: snapshot has {} layers, store has {}",
                layers.len(),
                recurrent.len()
            );
        }
        for (&layer_index, l) in recurrent.iter().zip(layers) {
            if l.layer_index as usize != layer_index
                || l.n_v_heads as usize != dims.n_v_heads
                || l.d_v as usize != dims.head_dim
                || l.d_k as usize != dims.head_dim
                || l.conv_channels as usize != dims.conv_dim()
                || l.conv_tail_cols as usize != dims.conv_kernel - 1
                || l.state.len() != dims.state_elems() * 4
                || l.conv_tail.len() != dims.conv_state_elems() * 4
            {
                candle::bail!(
                    "recurrent store: snapshot layer {} does not match the store's \
                     geometry",
                    l.layer_index
                );
            }
        }
        Ok(())
    }

    /// Scatter a snapshot back into the store — the resume path. Validates
    /// the schedule hash and every layer's dims before touching any tensor
    /// ([`Self::check_snapshot`]); on any mismatch the store is left untouched
    /// and the caller recomputes. Refused mid-wave and with timeline rows
    /// pending, as `export` is: a step still deciding which rows it keeps is not
    /// a point a restore may land on.
    pub fn import(&mut self, snapshot_hash: u64, layers: &[ExportedLayerState]) -> Result<()> {
        if self.open {
            candle::bail!("recurrent store: import mid-wave");
        }
        self.break_step();
        if self.has_pending() {
            candle::bail!(
                "recurrent store: import with timeline rows pending — fold or discard them, \
                 then restore"
            );
        }
        if snapshot_hash != self.hash {
            candle::bail!(
                "recurrent store: snapshot schedule hash {snapshot_hash:#x} does not match \
                 this model's {:#x} — recompute the state instead of scattering a foreign \
                 layout",
                self.hash
            );
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
        self.break_step();
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
            slot.live.state.write_host(&state_f, &tail_f)?;
        }
        // Restored state is state someone put here on purpose. Without this the
        // first wave on a resumed slot standing at offset 0 would reset it, and
        // the conversation would come back fluent and amnesiac — the exact
        // failure resume exists to remove.
        self.seeded = true;
        Ok(())
    }
}

/// `v` as little-endian bytes — the snapshot's wire form of a state tensor.
///
/// Written into a pre-sized buffer a fixed four bytes at a time, which the compiler
/// turns into a straight copy. The `flat_map(to_le_bytes).collect()` this replaced
/// could neither pre-size nor vectorise, and it ran over every float of every layer
/// on every turn seal — ~16 M floats on the 35B-A3B, on the scheduler's loop thread.
fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; v.len() * 4];
    for (dst, f) in out.chunks_exact_mut(4).zip(v) {
        dst.copy_from_slice(&f.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::delta_net::mix::DeltaNetOut;
    #[cfg(feature = "cuda")]
    use crate::models::gpu_test_lock::gpu_serial;

    /// The wire bytes are each float's IEEE-754 little-endian image, in order.
    #[test]
    fn f32_le_bytes_is_each_floats_little_endian_image_in_order() {
        assert_eq!(
            f32_le_bytes(&[1.0, -2.5, 0.0]),
            vec![0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x20, 0xC0, 0x00, 0x00, 0x00, 0x00]
        );
        assert!(f32_le_bytes(&[]).is_empty());
    }

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

    /// One long span's writes into a layer's scratch: `live + 1` in both
    /// buffers, which is what the scan kernels do to their two write pointers.
    fn bump_into(live: &DeltaNetState, out: &DeltaNetOut) {
        let one = |src: &Tensor, dst: &Tensor| {
            let ones = Tensor::ones(src.shape(), src.dtype(), src.device()).unwrap();
            dst.slice_set(&src.add(&ones).unwrap(), 0, 0).unwrap();
        };
        one(&live.s, &out.s);
        one(&live.conv_tail, &out.conv_tail);
    }

    /// Run a long span over `layer` of `store`: take its write and bump.
    fn long_span(store: &mut RecurrentStateStore, layer: usize) {
        let (live, write) = store.layer_span_mut(layer, TIMELINE_ROWS + 1).unwrap();
        let SpanWrite::State(out) = write else {
            panic!("a long span writes a state");
        };
        bump_into(live, &out);
    }

    /// Run a one-row span over `layer` of `store`, writing its timeline with a
    /// known generator: G = 0, k = 1, v_new = `v`, x = `x`. Folded, every element
    /// of S moves by `v` and the tail shifts `x` in.
    fn short_span(store: &mut RecurrentStateStore, layer: usize, v: f32, x: f32) {
        let d = dims();
        let lay = TimelineLayout::of(&d);
        let (_, write) = store.layer_span_mut(layer, 1).unwrap();
        let SpanWrite::Timeline(tl) = write else {
            panic!("a one-row span writes a timeline");
        };
        let dev = tl.device().clone();
        let full = |shape: &[usize], v: f32| Tensor::full(v, shape, &dev).unwrap();
        lay.write_rows(
            &tl,
            &full(&[1, d.n_v_heads], 0.0),
            &full(&[1, d.n_k_heads, d.head_dim], 1.0),
            &full(&[1, d.n_v_heads, d.head_dim], v),
            &full(&[1, d.conv_dim()], x),
        )
        .unwrap();
    }

    fn filled_store() -> RecurrentStateStore {
        filled_store_on(&Device::Cpu)
    }

    fn filled_store_on(dev: &Device) -> RecurrentStateStore {
        let dev = dev.clone();
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

    /// A snapshot is checked against the geometry with no store built: the
    /// store's own export passes, and a wrong hash, a wrong layer count and a
    /// wrong layer geometry are each refused — the same answers `import` gives.
    #[test]
    fn check_snapshot_agrees_with_import_without_a_store() {
        let store = filled_store();
        let hash = store.schedule_hash();
        let exported = store.export().unwrap();
        let check = |h: u64, l: &[ExportedLayerState]| {
            RecurrentStateStore::check_snapshot(&kinds(), &dims(), h, l)
        };
        check(hash, &exported).unwrap();
        assert!(check(hash ^ 1, &exported)
            .unwrap_err()
            .to_string()
            .contains("schedule hash"));
        assert!(check(hash, &exported[..2])
            .unwrap_err()
            .to_string()
            .contains("snapshot has 2 layers"));
        let mut bad = exported.clone();
        bad[1].layer_index = 2;
        assert!(check(hash, &bad)
            .unwrap_err()
            .to_string()
            .contains("geometry"));
    }

    /// A held snapshot builds a store holding exactly it — byte for byte through
    /// `export` — marked seeded so its first wave keeps it; and a snapshot that
    /// does not fit the geometry builds nothing.
    #[test]
    fn a_staged_snapshot_builds_a_seeded_store_holding_it() {
        let source = filled_store();
        let staged = StagedSnapshot {
            schedule_hash: source.schedule_hash(),
            layers: source.export().unwrap().into(),
        };
        let built = staged.build(&kinds(), &dims(), &Device::Cpu).unwrap();
        assert_eq!(built.export().unwrap(), source.export().unwrap());
        assert!(built.is_seeded());

        let foreign = StagedSnapshot {
            schedule_hash: staged.schedule_hash ^ 1,
            layers: staged.layers.clone(),
        };
        assert!(foreign.build(&kinds(), &dims(), &Device::Cpu).is_err());
    }

    /// **A short span leaves the state as it entered; a fold lands it.** The span
    /// writes its timeline and the layer is pending; the live state does not move
    /// until the fold, which applies exactly the timeline's rows.
    #[test]
    fn a_short_span_writes_its_timeline_and_a_fold_lands_it() {
        let mut store = filled_store();
        let entry = store.export().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 2.0, 7.0);
        store.commit_wave();
        assert!(store.has_pending());
        assert_eq!(store.pending_layers().collect::<Vec<_>>(), vec![(0, 1)]);
        assert_eq!(
            store
                .layer_state(0)
                .unwrap()
                .s
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            {
                let e: Vec<f32> = entry[0]
                    .state
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                e
            },
            "a short span wrote the live state"
        );

        store.fold(None).unwrap();
        assert!(!store.has_pending());
        let after = store.export().unwrap();
        // S·e^0 + 2·1 on every element of layer 0; layers 1 and 3 untouched.
        let s0: Vec<f32> = entry[0]
            .state
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b) + 2.0)
            .collect();
        assert_eq!(after[0].state, f32_le_bytes(&s0));
        // The tail shifted the row's raw input in: [t0 t1] → [t1 7].
        let t: Vec<f32> = entry[0]
            .conv_tail
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let shifted: Vec<f32> = t.chunks(2).flat_map(|c| [c[1], 7.0]).collect();
        assert_eq!(after[0].conv_tail, f32_le_bytes(&shifted));
        assert_eq!(after[1], entry[1]);
        assert_eq!(after[2], entry[2]);
    }

    /// **The close folds the stores it is told to and leaves the armed one
    /// pending**, then commits both: a plain row's rows land at its own commit,
    /// a verifying sequence's wait for the reconcile.
    #[test]
    fn commit_stores_folds_the_unarmed_and_leaves_the_armed_pending() {
        let mut plain = filled_store();
        let mut armed = filled_store();
        let entry = plain.export().unwrap();
        for store in [&mut plain, &mut armed] {
            store.begin_wave().unwrap();
            short_span(store, 0, 2.0, 7.0);
        }
        commit_stores(&mut [(&mut plain, true), (&mut armed, false)]).unwrap();
        assert!(!plain.has_pending(), "the plain row folded at its commit");
        assert!(
            armed.has_pending(),
            "the armed block waits for its reconcile"
        );
        assert_ne!(plain.export().unwrap()[0], entry[0]);
        // Both waves are closed: a new one opens on either.
        plain.begin_wave().unwrap();
        plain.rollback_wave().unwrap();
        armed.begin_wave().unwrap();
        armed.rollback_wave().unwrap();
        assert!(
            armed.has_pending(),
            "a later wave's rollback keeps the armed rows"
        );
    }

    /// A span with no wave open is refused: its writes would belong to nothing
    /// that could roll them back.
    #[test]
    fn a_span_outside_a_wave_is_refused() {
        let mut store = filled_store();
        let err = store.layer_span_mut(0, 1).unwrap_err();
        assert!(err.to_string().contains("no wave open"), "{err}");
    }

    /// A discard drops the rows: the state is as the span found it.
    #[test]
    fn a_discard_leaves_the_state_as_it_entered() {
        let mut store = filled_store();
        let entry = store.export().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 5.0, 1.0);
        store.commit_wave();
        store.discard_pending();
        assert!(!store.has_pending());
        assert_eq!(store.export().unwrap(), entry);
    }

    /// A fold keeping fewer rows than the span wrote applies only those; one
    /// keeping none applies nothing.
    #[test]
    fn a_fold_keeps_the_rows_it_is_told_to() {
        let mut store = filled_store();
        let entry = store.export().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 3.0, 1.0);
        store.commit_wave();
        store.fold(Some(0)).unwrap();
        assert!(!store.has_pending(), "a fold of nothing still ends pending");
        assert_eq!(
            store.export().unwrap(),
            entry,
            "a fold of none applied rows"
        );
    }

    /// **A pending layer refuses a second span**, and every whole-store reader
    /// refuses a pending store: its live state is not where the sequence stands.
    #[test]
    fn a_pending_layer_refuses_a_second_span_and_the_store_refuses_its_readers() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 1.0, 1.0);
        store.commit_wave();

        store.begin_wave().unwrap();
        let err = store.layer_span_mut(0, 1).unwrap_err();
        assert!(err.to_string().contains("pending"), "{err}");
        assert!(
            store.layer_span_mut(1, 1).is_ok(),
            "another layer runs freely"
        );
        store.rollback_wave().unwrap();

        assert!(store.export().is_err(), "export with rows pending");
        assert!(store.fork_from().is_err(), "fork with rows pending");
        store.fold(None).unwrap();
        assert!(store.export().is_ok());
        assert!(store.fork_from().is_ok());
    }

    /// **A rollback drops what its own wave wrote and nothing else.** A step's
    /// first segment commits with layer 0 pending; its second segment runs
    /// layer 1 and fails: layer 1's rows go, layer 0's stay for the reconcile.
    #[test]
    fn a_rollback_drops_its_own_waves_rows_and_keeps_an_earlier_segments() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 1.0, 1.0);
        store.commit_wave();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 1.0, 1.0);
        store.rollback_wave().unwrap();
        assert_eq!(store.pending_layers().collect::<Vec<_>>(), vec![(0, 1)]);
    }

    /// A long span needs the scratch, claimed between forwards.
    #[test]
    fn a_long_span_needs_the_prefill_scratch() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        let err = store.layer_span_mut(0, TIMELINE_ROWS + 1).unwrap_err();
        assert!(err.to_string().contains("scratch"), "{err}");
        assert!(store.ensure_scratch().is_err(), "claimed mid-wave");
        store.rollback_wave().unwrap();
        store.ensure_scratch().unwrap();
        assert!(store.holds_scratch());
    }

    #[test]
    fn wave_rollback_restores_entry_state_and_commit_keeps_writes() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        let entry = store.export().unwrap();

        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.rollback_wave().unwrap();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "rollback must restore the wave-entry state exactly"
        );

        // A successful wave: mutate, commit — the write stands, in BOTH
        // buffers. Asserting only on `s` would pass while the conv tail was
        // left behind in the half the exchange filed away, which is precisely
        // the failure a partial exchange produces: a state one wave ahead of
        // its tail.
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
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
        // Layer 1 never ran, so its slot keeps its state as it was.
        assert_eq!(
            committed[1], entry[1],
            "an unrun layer must not be exchanged"
        );
    }

    /// **A wave never writes the state it read, so an entering alias is never
    /// disturbed by a wave that fails.** Short and long spans alike: the short
    /// one writes its timeline, the long one the scratch.
    #[test]
    fn a_wave_leaves_the_entering_buffer_untouched() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        // Shares storage with the slot's entering state — the same view the
        // decode kernel's pointer table holds for this wave.
        let alias = store.layer_state(0).unwrap().s.clone();
        let entry: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();

        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        short_span(&mut store, 1, 1.0, 1.0);
        let during: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            during, entry,
            "the wave wrote into the buffer it was reading — the entering state \
             is gone and a rollback has nothing to return to"
        );
        store.rollback_wave().unwrap();
        let after: Vec<f32> = alias.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(after, entry, "rollback must leave the entering state alone");
    }

    /// **The scratch is held while the sequence prefills and no longer.** A
    /// commit that ran a long span keeps it; the first commit that ran none —
    /// the sequence decoding — gives it back.
    #[test]
    fn the_scratch_goes_back_after_the_first_wave_with_no_long_span() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();
        assert!(
            store.holds_scratch(),
            "a prefilling sequence keeps its scratch"
        );

        store.begin_wave().unwrap();
        short_span(&mut store, 0, 1.0, 1.0);
        store.commit_wave();
        assert!(
            !store.holds_scratch(),
            "a decoding sequence holds no scratch"
        );
    }

    /// **A segment that reaches no recurrent layer keeps the scratch.** A
    /// prefill chunk split into layer windows can run a window of attention
    /// layers only; its commit says nothing about whether the sequence still
    /// prefills, and dropping the scratch there would refuse the chunk's next
    /// window.
    #[test]
    fn a_wave_that_runs_no_recurrent_layer_keeps_the_scratch() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();

        store.begin_wave().unwrap();
        store.commit_wave();
        assert!(
            store.holds_scratch(),
            "an attention-only window dropped the scratch"
        );

        store.begin_wave().unwrap();
        long_span(&mut store, 3);
        store.commit_wave();
        assert!(store.holds_scratch());
    }

    /// A reset store reads exactly as a fresh one — every layer's `s` and conv
    /// tail back to zero, nothing pending, no scratch, and no longer seeded —
    /// even when a wave was left open on it.
    #[test]
    fn reset_returns_the_store_to_the_sequence_start_state() {
        let mut store = filled_store();
        store.mark_seeded();
        store.ensure_scratch().unwrap();
        let fresh = RecurrentStateStore::new(&kinds(), &dims(), &Device::Cpu).unwrap();
        assert_ne!(store.export().unwrap(), fresh.export().unwrap());

        // Rows left pending by a committed segment, then a wave left open.
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 1.0, 1.0);
        store.commit_wave();
        store.begin_wave().unwrap();

        store.reset().unwrap();
        assert!(!store.has_pending());
        assert!(!store.holds_scratch());
        assert_eq!(
            store.export().unwrap(),
            fresh.export().unwrap(),
            "a reset store must hold the sequence-start state byte for byte"
        );
        assert!(!store.is_seeded());

        // A span on the reset store starts from zero: one row of v = 1 reads as ones.
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 1.0, 1.0);
        store.commit_wave();
        store.fold(None).unwrap();
        let s: Vec<f32> = store
            .layer_state(0)
            .unwrap()
            .s
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(s.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn wave_sequencing_is_enforced() {
        let mut store = filled_store();
        assert!(store.rollback_wave().is_err(), "rollback with no wave open");
        store.begin_wave().unwrap();
        assert!(store.begin_wave().is_err(), "overlapping wave");
        assert!(store.export().is_err(), "export mid-wave");
        assert!(store.fold(None).is_err(), "fold mid-wave");
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
        assert!(!child.holds_scratch());
    }

    /// **The `Clone`-shares-storage hazard.** `Tensor::clone` is a shallow handle
    /// clone, so a fork built from clones would look right and then track every
    /// mutation of its parent — or move its parent with every mutation of its own.
    #[test]
    fn fork_buffers_are_distinct_allocations() {
        let parent = filled_store();
        let mut child = parent.fork_from().unwrap();
        let parent_before = parent.export().unwrap();
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
            "the child shares the parent's live buffer"
        );
    }

    /// A fork mid-wave is refused: the wave's advance is not yet in the live
    /// state.
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

    /// The fork reads the committed state: a long span exchanged in, or a short
    /// span folded, is in the child.
    #[test]
    fn fork_reads_the_committed_state() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        let before = store.export().unwrap();
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 1.0, 1.0);
        store.commit_wave();
        store.fold(None).unwrap();
        let after = store.export().unwrap();
        assert_ne!(after[0], before[0]);
        assert_ne!(after[1], before[1]);
        let child = store.fork_from().unwrap();
        assert_eq!(child.export().unwrap(), after);
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

    /// **A failed long-span wave that reached only part of the stack leaves NO
    /// trace**, and the next wave does not inherit its bookkeeping: a commit that
    /// exchanged layers the dead wave touched would install its output.
    #[test]
    fn a_partial_sweep_that_rolls_back_leaves_every_layer_at_its_entry_value() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        let entry = store.export().unwrap();

        store.begin_wave().unwrap();
        for li in [0usize, 1] {
            long_span(&mut store, li);
        }
        store.rollback_wave().unwrap();
        assert_eq!(
            store.export().unwrap(),
            entry,
            "a rolled-back partial sweep moved the state"
        );

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
    /// leaves the rest alone.
    #[test]
    fn a_partial_sweep_that_commits_advances_only_the_layers_it_reached() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        let entry = store.export().unwrap();

        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        store.commit_wave();

        let after = store.export().unwrap();
        assert_ne!(after[0].state, entry[0].state, "layer 0 advanced");
        assert_eq!(
            after[1].state, entry[1].state,
            "layer 1 was never reached and must not have moved"
        );
        assert_eq!(after[2].state, entry[2].state, "nor layer 3");
    }

    /// **A step whose later segment fails is undone whole.** Segment one
    /// commits layers 0 and 1 — their rows stay pending, the step folding
    /// nothing — and segment two is refused on layer 3: undoing the step drops
    /// the committed rows, the state never having moved, so the step's retry
    /// runs their tokens into the state once rather than twice.
    #[test]
    fn an_undone_step_returns_every_committed_segment_to_its_entry_state() {
        let mut store = filled_store();
        let entry = store.export().unwrap();

        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        for layer in [0, 1] {
            short_span(&mut store, layer, 2.0, 7.0);
        }
        commit_stores(&mut [(&mut store, true)]).unwrap();
        assert!(
            store.has_pending(),
            "inside a step the commit folds nothing"
        );
        store.begin_wave().unwrap();
        short_span(&mut store, 3, 2.0, 7.0);
        store.rollback_wave().unwrap();

        store.undo_step().unwrap();
        assert!(!store.has_pending());
        assert_eq!(store.export().unwrap(), entry, "every layer back at entry");
        // The retry runs the step's layers again.
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 2.0, 7.0);
        store.rollback_wave().unwrap();
    }

    /// **A kept step folds at its end what its commits owed**, in one fold
    /// over every store; an armed store was owed none and stays pending for
    /// its reconcile.
    #[test]
    fn a_kept_step_folds_what_its_commits_owed() {
        let mut plain = filled_store();
        let mut armed = filled_store();
        let entry = plain.export().unwrap();
        for store in [&mut plain, &mut armed] {
            store.begin_step().unwrap();
        }
        for layer in [0, 1] {
            for store in [&mut plain, &mut armed] {
                store.begin_wave().unwrap();
                short_span(store, layer, 2.0, 7.0);
            }
            commit_stores(&mut [(&mut plain, true), (&mut armed, false)]).unwrap();
        }
        assert_eq!(
            plain.pending_layers().collect::<Vec<_>>(),
            vec![(0, 1), (1, 1)],
            "nothing folded mid-step"
        );
        end_steps(&mut [&mut plain, &mut armed]).unwrap();
        assert!(!plain.has_pending());
        let after = plain.export().unwrap();
        assert_ne!(after[0], entry[0]);
        assert_ne!(after[1], entry[1]);
        assert_eq!(after[2], entry[2], "layer 3 never ran");
        assert_eq!(
            armed.pending_layers().collect::<Vec<_>>(),
            vec![(0, 1), (1, 1)],
            "the armed rows wait for the reconcile"
        );
        // Both steps closed: a commit folds again.
        plain.begin_wave().unwrap();
        short_span(&mut plain, 3, 2.0, 7.0);
        commit_stores(&mut [(&mut plain, true)]).unwrap();
        assert!(!plain.has_pending());
    }

    /// Rows pending when the step began are not the step's: an undo keeps them
    /// and drops only what the step's waves wrote.
    #[test]
    fn a_step_undo_keeps_the_rows_it_entered_with() {
        let mut store = filled_store();
        store.begin_wave().unwrap();
        short_span(&mut store, 0, 2.0, 7.0);
        commit_stores(&mut [(&mut store, false)]).unwrap();

        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 2.0, 7.0);
        commit_stores(&mut [(&mut store, false)]).unwrap();
        store.undo_step().unwrap();
        assert_eq!(store.pending_layers().collect::<Vec<_>>(), vec![(0, 1)]);
    }

    /// A step whose state moved other than by a short span refuses its undo
    /// instead of dropping rows from a state they no longer describe — an
    /// exchange, or a fold — and the refusal closes the step. A layer the step
    /// already ran refuses a second span outright.
    #[test]
    fn a_step_undo_refuses_what_dropping_rows_cannot_restore() {
        let mut store = filled_store();
        store.ensure_scratch().unwrap();
        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        long_span(&mut store, 0);
        commit_stores(&mut [(&mut store, true)]).unwrap();
        assert!(store.undo_step().is_err(), "a long span's exchange");
        assert!(
            store.undo_step().is_err(),
            "the refused undo closed the step"
        );

        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 2.0, 7.0);
        commit_stores(&mut [(&mut store, false)]).unwrap();
        store.fold(None).unwrap();
        assert!(store.undo_step().is_err(), "a fold inside the step");

        store.begin_step().unwrap();
        store.begin_wave().unwrap();
        short_span(&mut store, 1, 2.0, 7.0);
        commit_stores(&mut [(&mut store, true)]).unwrap();
        store.begin_wave().unwrap();
        assert!(
            store.layer_span_mut(1, 1).is_err(),
            "a layer the step ran holds its rows"
        );
        store.rollback_wave().unwrap();
        store.undo_step().unwrap();
        assert!(!store.has_pending());
    }

    /// **The resume oracle.** Seal → drop the store entirely → resume from the
    /// exported rows → the state is bit-identical.
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
    /// one.
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

    /// The fixture's slot: a 256 B `s`, a 256 B conv tail on the next 256-byte
    /// boundary, and a 1,920 B timeline (8 rows of 4 + 8 + 16 + 32 floats) on
    /// the next — a 2,432 B block in a 2,560 B slot.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_slot_is_state_then_tail_then_timeline() {
        assert_eq!(DeltaNetState::byte_sizes(&dims()), (256, 256));
        assert_eq!(
            state_block(&dims()),
            StateBlock {
                conv_off: 256,
                state_end: 512,
                timeline_off: 512,
                bytes: 2432,
            }
        );
        assert_eq!(slot_stride(state_block(&dims()).bytes), 2560);
    }

    /// **A reset zeroes the state on the slots it already holds.** Same decode
    /// addresses before and after — no claim, no release — and the export reads
    /// as a fresh store's.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_reset_zeroes_in_place_without_a_new_claim() -> Result<()> {
        let _gpu = gpu_serial();
        let dev = Device::new_cuda(0)?;
        let mut store = filled_store_on(&dev);
        let before: Vec<[u64; 3]> = store.decode_table_addresses()?.collect();
        store.reset()?;
        let after: Vec<[u64; 3]> = store.decode_table_addresses()?.collect();
        assert_eq!(after, before, "a reset must keep the store's slots");
        let fresh = RecurrentStateStore::new(&kinds(), &dims(), &dev)?;
        assert_eq!(store.export()?, fresh.export()?);
        Ok(())
    }

    /// **The pinned readback is the same bytes the per-tensor one was.** A CUDA
    /// store filled with the fixture's values exports byte-for-byte what the CPU
    /// store does — every layer, state and conv tail, in slot order.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_export_is_byte_identical_to_the_cpu_one() -> Result<()> {
        let _gpu = gpu_serial();
        let cuda = filled_store_on(&Device::new_cuda(0)?).export()?;
        let cpu = filled_store().export()?;
        assert_eq!(cuda, cpu);
        // Twice, so the second export reads through the buffer the first sized.
        let again = filled_store_on(&Device::new_cuda(0)?).export()?;
        assert_eq!(again, cpu);
        Ok(())
    }

    /// **A CUDA fold lands what the CPU fold lands**, bit for bit on these
    /// exactly representable rows: the store's one-launch fold over its slots.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_store_fold_is_the_cpu_ones() -> Result<()> {
        let _gpu = gpu_serial();
        // The fold kernel runs at d = 128.
        let d = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 1,
            n_v_heads: 2,
            conv_kernel: 4,
        };
        let one = vec![LayerKind::DeltaNet, LayerKind::DeltaNet];
        let run = |dev: &Device| -> Result<Vec<ExportedLayerState>> {
            let mut store = RecurrentStateStore::new(&one, &d, dev)?;
            store.begin_wave()?;
            for li in 0..2 {
                let (_, write) = store.layer_span_mut(li, 1)?;
                let SpanWrite::Timeline(tl) = write else {
                    candle::bail!("a one-row span writes a timeline");
                };
                let full = |shape: &[usize], v: f32| Tensor::full(v, shape, dev);
                TimelineLayout::of(&d).write_rows(
                    &tl,
                    &full(&[1, 2], 0.0)?,
                    &full(&[1, 1, 128], 0.5)?,
                    &full(&[1, 2, 128], 4.0 + li as f32)?,
                    &full(&[1, d.conv_dim()], 3.0)?,
                )?;
            }
            store.commit_wave();
            store.fold(None)?;
            store.export()
        };
        assert_eq!(run(&Device::new_cuda(0)?)?, run(&Device::Cpu)?);
        Ok(())
    }

    /// **A released readback buffer unpins exactly what the seal pinned.** The
    /// fixture's three recurrent layers each read back 256 bytes of state and
    /// 256 of conv tail, so one export pins a 1,536-byte owned staging buffer;
    /// releasing it empties the slot, and an owned buffer unpins on drop — what
    /// an engine's shutdown relies on to leave no pinned memory behind.
    ///
    /// Read from the slot, not from the process's pinned-staging total: every
    /// test that stages moves that total, and they run in parallel.
    #[cfg(feature = "cuda")]
    #[test]
    fn releasing_the_seal_readback_unpins_it() -> Result<()> {
        let _gpu = gpu_serial();
        let held = || {
            READBACK
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .map(|b| (matches!(b, PinnedBuf::Owned { .. }), b.len()))
        };
        let store = filled_store_on(&Device::new_cuda(0)?);
        release_seal_readback();
        assert_eq!(held(), None);
        store.export()?;
        assert_eq!(held(), Some((true, 1_536)), "one owned, pinned buffer");
        release_seal_readback();
        assert_eq!(held(), None, "the release dropped it");
        // The next seal allocates it afresh and reads the same bytes.
        assert_eq!(store.export()?, filled_store().export()?);
        release_seal_readback();
        Ok(())
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

    /// The decode table's addresses are the live half's own — read off the
    /// slot, they must name exactly the buffers the tensors view — and a
    /// commit that exchanged a layer with its scratch moves that layer's, and
    /// no other's.
    ///
    /// The fixture's block puts the conv tail 256 bytes past the state and the
    /// timeline 512 past it.
    #[cfg(feature = "cuda")]
    #[test]
    fn decode_addresses_are_the_live_halfs_and_follow_an_exchange() -> Result<()> {
        let _gpu = gpu_serial();
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!("new_cuda answered a CUDA device")
        };
        let views = |store: &RecurrentStateStore| -> Result<Vec<[u64; 3]>> {
            store
                .slots
                .iter()
                .map(|s| {
                    Ok([
                        tensor_device_ptr(cuda, &s.live.state.conv_tail)?,
                        tensor_device_ptr(cuda, &s.live.state.s)?,
                        tensor_device_ptr(cuda, &s.live.timeline)?,
                    ])
                })
                .collect()
        };
        let mut store = RecurrentStateStore::new(&kinds(), &dims(), &dev)?;
        let before: Vec<[u64; 3]> = store.decode_table_addresses()?.collect();
        assert_eq!(before.len(), 3, "one row per DeltaNet layer: 0, 1 and 3");
        assert_eq!(before, views(&store)?);
        for [tail, s, tl] in &before {
            assert_eq!((*tail, *tl), (s + 256, s + 512));
        }

        // A long span over layer 3 only — the third slot — committed.
        store.ensure_scratch()?;
        store.begin_wave()?;
        store.layer_span_mut(3, TIMELINE_ROWS + 1)?;
        store.commit_wave();
        let after: Vec<[u64; 3]> = store.decode_table_addresses()?.collect();
        assert_eq!(after, views(&store)?);
        assert_eq!(
            &after[..2],
            &before[..2],
            "layers that did not run keep their slots"
        );
        assert_ne!(after[2], before[2], "layer 3 exchanged with its scratch");
        Ok(())
    }

    /// The whole point of the geometry price: it answers the same for a process
    /// holding no stores as for one holding a hundred, because it never looks at
    /// them. Three DeltaNet layers, one 2,560 B slot each, is 7,680 B — and the
    /// scratch a prefilling sequence adds is as much again.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_store_is_priced_from_geometry_not_from_what_is_resident() {
        let priced = RecurrentStateStore::reserved_bytes_for(&kinds(), &dims());
        assert_eq!(priced, 3 * 2560, "three slots");
        assert_eq!(
            RecurrentStateStore::scratch_bytes_for(&kinds(), &dims()),
            3 * 2560
        );

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
        // An attention-only stack carries no recurrent state and costs nothing.
        assert_eq!(
            RecurrentStateStore::reserved_bytes_for(&[LayerKind::Attention], &dims()),
            0,
            "no DeltaNet layers, no reservation"
        );
    }

    /// An empty conv tail has no address of its own, so the store refuses the
    /// geometry rather than leasing a tail at the next region's base.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_empty_conv_tail_is_refused() -> Result<()> {
        let _gpu = gpu_serial();
        let d = DeltaNetDims {
            head_dim: 64,
            n_k_heads: 1,
            n_v_heads: 4,
            conv_kernel: 1,
        };
        let err = match RecurrentStateStore::new(&[LayerKind::DeltaNet], &d, &Device::new_cuda(0)?)
        {
            Ok(_) => panic!("an empty conv tail must be refused"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("empty half"), "{err}");
        Ok(())
    }

    /// **A release empties the readback slot**, so the buffer a shut-down engine's
    /// seals pinned is freed rather than held for the life of the process — and an
    /// empty slot releases as a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_release_empties_the_readback_slot() {
        let slot = Mutex::new(Some(PinnedBuf::Host {
            data: vec![0u8; 64],
        }));
        release_readback_in(&slot);
        assert!(slot.lock().unwrap().is_none(), "the buffer was released");
        release_readback_in(&slot);
        assert!(slot.lock().unwrap().is_none(), "an empty slot stays empty");
    }

    /// **A poisoned slot still releases.** A seal that panicked while holding the
    /// lock must not keep the buffer pinned past its engine's shutdown.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_poisoned_readback_slot_still_releases() {
        let slot = Arc::new(Mutex::new(Some(PinnedBuf::Host {
            data: vec![0u8; 64],
        })));
        let poisoner = Arc::clone(&slot);
        let _ = std::thread::spawn(move || {
            let _held = poisoner.lock().unwrap();
            panic!("a seal panicked holding the readback lock");
        })
        .join();
        assert!(slot.is_poisoned());
        release_readback_in(&slot);
        assert!(
            slot.lock().unwrap_or_else(|e| e.into_inner()).is_none(),
            "the buffer was released through the poison"
        );
    }
}
