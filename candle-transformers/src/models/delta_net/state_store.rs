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
use candle::DType;
#[cfg(feature = "cuda")]
use candle::{cuda_backend::cudarc::driver::result::memcpy_dtod_sync, CudaDevice, Error, Storage};
use candle::{Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{span_region_refusal, SpanClaims, SpanRegion};

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
    /// The reservation regions every buffer above is a view into.
    ///
    /// **This is the store's lifetime, and its cleanup.** Held for as long as
    /// the sequence has recurrent memory; dropping the store returns them to
    /// the region free list, exactly as a KV arena's handles do. There is no
    /// eviction path and nothing to remember to call — the state is reclaimed
    /// by the store ceasing to exist, which is the only moment at which it is
    /// certainly dead.
    ///
    /// Empty on a CPU device, where the buffers are ordinary allocations.
    #[cfg(feature = "cuda")]
    regions: Vec<SpanRegion>,
}

/// Copy one state's two buffers into another's, device to device.
///
/// The fork path's replacement for `DeltaNetState::snapshot`, which allocates.
/// Here the destination already exists — it is a view into the child's own
/// regions — so the copy writes into it rather than producing a new buffer
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

/// A bump allocator over freshly claimed reservation regions.
///
/// One store's buffers are laid down left to right; a buffer that would cross a
/// region boundary starts the next region instead. The waste that costs is
/// bounded by one buffer per region and is the price of every buffer being a
/// single contiguous range — which the kernels require, since they take a base
/// pointer and a stride, not a scatter list.
#[cfg(feature = "cuda")]
pub(crate) struct RegionBump {
    pub(crate) device: Device,
    /// The arena window, open for the whole store.
    ///
    /// One window rather than one per region: entering it hands back a standing
    /// tier, so claiming a region at a time would release and re-place the tier
    /// once per region, each carrying a device-wide quiesce. A store takes eight
    /// or so, and that much churn reaches the WDDM watchdog.
    claims: SpanClaims,
    pub(crate) regions: Vec<SpanRegion>,
    /// Bytes used in the last region.
    cursor: usize,
}

#[cfg(feature = "cuda")]
impl RegionBump {
    /// The claimed regions, dropping the bump — **and with it the arena
    /// window**.
    ///
    /// A holder that keeps the whole bump keeps [`SpanClaims`] alive, and that
    /// is an *open arena window*: every later wave blocks in `wave_gate`
    /// waiting for it to close, and the engine simply stops. Measured as a
    /// 58-minute hang with the process alive and no output.
    ///
    /// So a caller that needs the regions to outlive the allocation takes them
    /// this way rather than storing the bump. `RecurrentStateStore` has always
    /// done exactly this (`regions: bump.map_or_else(Vec::new, |b| b.regions)`);
    /// this names the operation so the next caller does not have to notice.
    pub(crate) fn into_regions(self) -> Vec<SpanRegion> {
        self.regions
    }
}

#[cfg(feature = "cuda")]
impl RegionBump {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            claims: SpanClaims::open(device)?,
            regions: Vec::new(),
            // Forces the first `take` to claim, so there is no empty-vec case.
            cursor: SpanRegion::bytes(),
        })
    }

    /// A bump for `device`, or `None` when there is no reservation to carve
    /// from — a CPU device in a CUDA build, which is every unit test here.
    pub(crate) fn for_device(device: &Device) -> Result<Option<Self>> {
        match device {
            Device::Cuda(_) => Self::new(device).map(Some),
            _ => Ok(None),
        }
    }

    /// One layer's `(live, backup)` pair, laid down in one fixed order so the
    /// layout is identical for every layer and every sequence.
    fn take_state_pair(
        &mut self,
        dims: &DeltaNetDims,
        device: &Device,
    ) -> Result<(DeltaNetState, DeltaNetState)> {
        let (s_bytes, conv_bytes) = DeltaNetState::byte_sizes(dims);
        let live_s = self.take(s_bytes)?;
        let live_conv = self.take(conv_bytes)?;
        let backup_s = self.take(s_bytes)?;
        let backup_conv = self.take(conv_bytes)?;
        // SAFETY: every address names a distinct, non-overlapping range of a
        // region this store holds for its whole life, sized by `byte_sizes` for
        // exactly this state.
        unsafe {
            Ok((
                DeltaNetState::at(dims, device, live_s, live_conv)?,
                DeltaNetState::at(dims, device, backup_s, backup_conv)?,
            ))
        }
    }

    /// A **zeroed** tensor of `shape` on region memory.
    ///
    /// The general-purpose form of [`Self::take_state_pair`], for a buffer that
    /// outlives the wave and so cannot come from the wave arena, but must still
    /// be inside the reservation so the partition can see it. The rewind stash
    /// is the other user (`qwen35::spec::VerifyStash`).
    ///
    /// **Zeroed, unlike a wave buffer.** Its consumers read rows they did not
    /// write — the replay hands the mixer the whole `cap`-row buffer while only
    /// the captured spans were filled — so this is one of the cases hot-path
    /// invariant 6 explicitly exempts: a zero that is read before being written.
    /// It replaces a `Tensor::zeros`, which zeroed for the same reason.
    pub(crate) fn take_zeroed(
        &mut self,
        shape: impl Into<candle::Shape>,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        use candle::cuda_backend::cudarc::driver::result::memset_d8_async;
        use candle::cuda_backend::wave_provenance::LeaseOrigin;
        let shape = shape.into();
        let bytes = shape.elem_count() * dtype.size_in_bytes();
        // **An empty buffer needs no region.** A cohort with no blocks to verify
        // builds a zero-row stash, which is legitimate — there is simply nothing
        // to stash yet. `take` refuses a zero-byte request, and rightly: for its
        // own caller a state with an empty half is a geometry fault. Here it is
        // not, so the empty case is answered before the allocator sees it rather
        // than by weakening a guard that is load-bearing elsewhere.
        if bytes == 0 {
            return Tensor::zeros(shape, dtype, device);
        }
        let at = self.take(bytes)?;
        let Device::Cuda(cuda) = device else {
            candle::bail!("region bump: a region buffer needs a CUDA device");
        };
        // SAFETY: `at` names `bytes` of a region this bump holds and nothing
        // else addresses, and the stream orders the fill ahead of every reader.
        unsafe { memset_d8_async(at, 0, bytes, cuda.cuda_stream().cu_stream()) }
            .map_err(|e| candle::Error::Msg(format!("zeroing a region buffer: {e}")))?;
        // `Foreign`: a lease the wave allocator did not issue and must not
        // reclaim. Its ticket is absent deliberately — this buffer outlives
        // every wave that reads it.
        unsafe { Tensor::from_leased_cuda_ptr(at, dtype, shape, device, LeaseOrigin::Foreign) }
    }

    /// Address of `bytes` of region memory, claiming another region if this one
    /// cannot hold the request contiguously.
    fn take(&mut self, bytes: usize) -> Result<u64> {
        let cap = SpanRegion::bytes();
        if bytes > cap {
            candle::bail!(
                "recurrent state: a {bytes} B buffer exceeds the {cap} B region size — \
                 the state no longer fits the allocator's unit"
            );
        }
        // A zero-byte request would take the `else` branch on the very first
        // call — the cursor starts AT `cap` precisely so the first `take`
        // claims — and then read `regions.last()` of an empty vec. Reachable
        // from `byte_sizes` with `conv_kernel == 1`, where the conv tail has no
        // history to keep. There is no address to hand back for no bytes, and
        // inventing one inside a region nobody claimed is worse than saying so.
        if bytes == 0 {
            candle::bail!(
                "recurrent state: a zero-byte buffer has no address — the geometry \
                 asks for a state with an empty half (conv_kernel = 1?)"
            );
        }
        // 256-byte aligned: what the CUDA driver guarantees a fresh allocation
        // and what the kernels' vectorised loads assume of a base pointer.
        let aligned = self.cursor.next_multiple_of(256);
        if aligned + bytes > cap {
            let Some(region) = self.claims.claim()? else {
                candle::bail!(
                    "recurrent state: no region for this sequence after {} claimed — {}",
                    self.regions.len(),
                    span_region_refusal(&self.device),
                );
            };
            self.regions.push(region);
            self.cursor = 0;
        } else {
            self.cursor = aligned;
        }
        let base = self
            .regions
            .last()
            .expect("a region was just claimed or already stood")
            .base();
        let at = base + self.cursor as u64;
        self.cursor += bytes;
        Ok(at)
    }
}

impl RecurrentStateStore {
    /// Fresh zeros for every recurrent layer in `layer_kinds`.
    pub fn new(layer_kinds: &[LayerKind], dims: &DeltaNetDims, device: &Device) -> Result<Self> {
        let mut slots = Vec::new();
        // On CUDA the whole store is carved from the device reservation. A
        // region arrives zeroed, which is what `live` needs — it is genuinely
        // READ at zero, being the sequence-start state — so no fill is issued
        // for it and `backup` needs none either, being fully stamped by the
        // first wave before anything reads it (invariant 6).
        // The cfg gates COMPILATION; the device gates behaviour. A CUDA build
        // still runs on a CPU device — every unit test here does — and there is
        // no reservation there to carve from.
        #[cfg(feature = "cuda")]
        let mut bump = RegionBump::for_device(device)?;
        for (i, k) in layer_kinds.iter().enumerate() {
            if *k == LayerKind::DeltaNet {
                #[cfg(feature = "cuda")]
                let (live, backup) = match bump.as_mut() {
                    Some(bump) => bump.take_state_pair(dims, device)?,
                    None => (
                        DeltaNetState::zeros(dims, device)?,
                        DeltaNetState::uninit(dims, device)?,
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
            #[cfg(feature = "cuda")]
            regions: bump.map_or_else(Vec::new, |b| b.regions),
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
        Ok(())
    }

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
                slot.advanced = false;
            }
        }
        self.open = false;
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
        // The child's memory comes from the reservation for the same reason the
        // parent's does — a fork is another sequence, and at ~3 forks per turn
        // this was ~126 MiB of pool traffic each.
        #[cfg(feature = "cuda")]
        let mut bump = RegionBump::for_device(&self.device)?;
        for slot in &self.slots {
            // Scratch, not state, in either arm: the kernels fully overwrite
            // the write buffer before anything reads it, so copying it would be
            // ~2 MB per layer of device traffic for bytes nobody reads — and
            // for the same reason it is left UNINITIALISED (invariant 6).
            #[cfg(feature = "cuda")]
            let (live, backup) = match bump.as_mut() {
                Some(bump) => {
                    let (live, backup) = bump.take_state_pair(&self.dims, &self.device)?;
                    // The fork's whole point: the child starts from the
                    // parent's state. A device-to-device copy into the child's
                    // own range, rather than `snapshot()`, which would allocate
                    // a fresh pool buffer and hand back a tensor pointing
                    // outside the span.
                    copy_state_into(&self.device, &slot.live, &live)?;
                    (live, backup)
                }
                None => (
                    slot.live.snapshot()?,
                    DeltaNetState::uninit(&self.dims, &self.device)?,
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
            });
        }
        Ok(Self {
            dims: self.dims,
            hash: self.hash,
            slots,
            open: false,
            seeded: true,
            device: self.device.clone(),
            #[cfg(feature = "cuda")]
            regions: bump.map_or_else(Vec::new, |b| b.regions),
        })
    }

    /// Reservation bytes this sequence's recurrent memory holds.
    ///
    /// Zero off CUDA, where the buffers are ordinary allocations. Reported
    /// rather than merely held so a whole-card accounting can name it: this is
    /// several GiB across a wide wave, and memory nothing can total is memory
    /// that goes missing (`AccountingSection`).
    pub fn reserved_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.regions.len() * SpanRegion::bytes()
        }
        #[cfg(not(feature = "cuda"))]
        {
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
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let tail_f: Vec<f32> = l
                .conv_tail
                .chunks_exact(4)
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
}
