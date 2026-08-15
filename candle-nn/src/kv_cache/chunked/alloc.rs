//! Internal allocation methods for ChunkedKvBacking.
//!
//! This module contains methods for:
//! - Ensuring max block capacity
//! - Creating arenas
//! - Allocating chunks from free lists or new arenas
//! - Ensuring chunks are allocated for token writes

use std::cmp;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use candle::cuda_backend::wave_provenance::LeaseOrigin;
use candle::{DType, Device, Result, Tensor};

use super::arena::ArenaKey;
use super::backing::ChunkedKvBacking;
#[cfg(feature = "cuda")]
use super::backing::KV_DEVICE_OOM_MARKER;
#[cfg(feature = "cuda")]
use super::bump_arena::{enter_arena_window, ArenaWindow, KV_ARENA_MID_WAVE};
use super::gid_pool::ChunkGid;
use super::head_gids::HeadGids;
#[cfg(feature = "cuda")]
use super::region_pool;
use super::size_class::{elems_per_chunk, SizeClass};
use super::types::{ChunkWindow, DecodeLayout, CHUNK_SIZE};
use super::{Arena, ArenaLocation};
use crate::kv_cache::arena_table::{ArenaFormatTag, N_PALETTE};
use crate::kv_cache::chunked::backing::BackingInner;
use crate::kv_cache::chunked::ArenaStorageState;
use crate::kv_cache::{KvFormat, QuantFormat};

/// How many bytes of KV the reservation can still hold: free regions × the
/// region size.
///
/// An **exact count**, not an estimate. It replaced three overlapping
/// approximations — driver headroom, the pool's reserved-but-free gap, and
/// `init_free − pool_used` — that disagreed with each other precisely when it
/// mattered: the pool's reuse gap read as available while a contiguous arena
/// could not fit in any single free block, so admission kept widening into a
/// wall. Under the reservation the question has one answer, and asking it costs
/// a mutex and an integer.
///
/// `None` when this device has no reservation yet (non-CUDA, or before the
/// first KV cache exists) — callers treat that as "unknown", never as zero.
///
/// Counts `free + blocked`, not `free` alone. `blocked` is unowned ground the
/// current wave's transient tier stands on, and this budget is spent by the
/// *next* forward's admission — which claims in phase 1, after phase 0 has
/// released that tier, so the blocked ground is claimable by the time any
/// claim priced against this number runs. Counting only `free` made every
/// standing tier read as KV pressure between forwards and admission starved
/// itself against ground it was guaranteed to get back.
#[cfg(feature = "cuda")]
pub fn vram_budget_available(device: &Device) -> Option<usize> {
    let candle::DeviceLocation::Cuda { gpu_id } = device.location() else {
        return None;
    };
    region_pool::region_stats(gpu_id).map(|s| (s.free + s.blocked) * region_pool::REGION_BYTES)
}

#[cfg(not(feature = "cuda"))]
pub fn vram_budget_available(_device: &Device) -> Option<usize> {
    None
}

static ARENA_STATS_ENABLED: OnceLock<bool> = OnceLock::new();
static ARENA_CREATE_COUNT: AtomicU64 = AtomicU64::new(0);
static ARENA_CREATE_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

pub(super) fn arena_stats_enabled() -> bool {
    *ARENA_STATS_ENABLED.get_or_init(|| std::env::var("KV_ARENA_STATS").is_ok())
}

fn record_arena_create(kind: &str, location: ArenaLocation, index: usize, elapsed_ns: u64) {
    if !arena_stats_enabled() {
        return;
    }
    let total_count = ARENA_CREATE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let total_ns = ARENA_CREATE_TOTAL_NS.fetch_add(elapsed_ns, Ordering::Relaxed) + elapsed_ns;
    let took_ms = elapsed_ns as f64 / 1_000_000.0;
    let total_ms = total_ns as f64 / 1_000_000.0;
    let avg_ms = total_ms / (total_count as f64);
    eprintln!(
        "[arena-create] kind={kind} location={location:?} index={index} took_ms={took_ms:.3} total_count={total_count} total_ms={total_ms:.3} avg_ms={avg_ms:.3}"
    );
}

static CLASS_PROMOTIONS: AtomicU64 = AtomicU64::new(0);

/// Count a scarcity promotion and, under `KV_ARENA_STATS`, name it.
///
/// Worth its own counter rather than folding into the arena-create line: a
/// promotion is the allocator reporting that a class could not get a region,
/// which is the signal the ladder's shape is tuned against. A steady trickle is
/// the mechanism working; a flood means the ladder's shape is wrong for the
/// workload, or the reservation is undersized.
fn record_class_promotion(from: SizeClass, to: SizeClass) {
    let total = CLASS_PROMOTIONS.fetch_add(1, Ordering::Relaxed) + 1;
    if arena_stats_enabled() {
        eprintln!(
            "[class-promote] {} B -> {} B (no region for the smaller class) total={total}",
            from.bytes(),
            to.bytes(),
        );
    }
}

/// Total scarcity promotions since process start.
pub fn class_promotion_count() -> u64 {
    CLASS_PROMOTIONS.load(Ordering::Relaxed)
}

fn push_unique_key(keys: &mut Vec<super::arena::ArenaKey>, key: super::arena::ArenaKey) {
    if !keys.iter().any(|k| k == &key) {
        keys.push(key);
    }
}

impl ChunkedKvBacking {
    /// Pre-create one arena for baseline formats and quant candidates, then
    /// mark those arena indices as protected so compaction never tombstones them.
    pub(super) fn warm_protected_arenas(
        &self,
        compression: Option<&super::CompressionPolicy>,
    ) -> Result<()> {
        let location = self.inner.storage.default_location();
        let mut keys = Vec::new();
        let push = |keys: &mut Vec<ArenaKey>, fmt: KvFormat| -> Result<()> {
            push_unique_key(keys, self.inner.arena_key_for(fmt, location)?);
            Ok(())
        };

        // Baseline warm set requested for runtime stability: F16 and R16.
        push(&mut keys, KvFormat::Float(DType::F16))?;
        push(&mut keys, KvFormat::Quantized(QuantFormat::R16))?;

        // Include the backing's default target formats.
        push(&mut keys, self.inner.storage.k_format())?;
        push(&mut keys, self.inner.storage.v_format())?;

        // Include quantized candidates used by the shared adaptive profile.
        //
        // Under size classes several of these collapse onto one key — the whole
        // sub-320 B tail is a single class — so the warm set is a handful of
        // slabs rather than one per candidate format. That collapse is the
        // point: a slot freed by any of them is allocatable by all of them.
        if let Some(compression) = compression {
            let (k_candidates, v_candidates) =
                super::compression_policy::production_adaptive_candidates(
                    compression.compression_level,
                );
            for fmt in k_candidates.iter().chain(v_candidates.iter()) {
                if matches!(fmt, KvFormat::Quantized(_)) {
                    push(&mut keys, *fmt)?;
                }
            }
        }

        for key in keys {
            let arena_idx = self.inner.pool.register_arena(key);
            self.inner.pool.protect_arena(arena_idx);
            self.ensure_arena_exists(arena_idx, key)?;
        }
        Ok(())
    }

    /// Ensure the backing can hold at least `required_max_blocks` blocks per sequence.
    pub(super) fn ensure_max_blocks(&self, required_max_blocks: usize) -> Result<()> {
        if required_max_blocks <= 1 {
            return Ok(());
        }
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if required_max_blocks <= state.max_blocks {
            return Ok(());
        }

        let mut new_max_blocks = state.max_blocks;
        while new_max_blocks < required_max_blocks {
            new_max_blocks = cmp::max(required_max_blocks, new_max_blocks.saturating_mul(2));
            if new_max_blocks == 0 {
                new_max_blocks = required_max_blocks;
                break;
            }
        }

        state.max_blocks = new_max_blocks;

        Ok(())
    }

    pub(super) fn create_arena(&self, key: ArenaKey, index: usize) -> Result<Arena> {
        self.inner.create_arena(key, index)
    }
}

impl BackingInner {
    /// Where this backing's *hot* tier is: the device it was built on.
    ///
    /// A CPU-only backing has one tier, so lifting out of warm lands back in
    /// host memory. Asking for `Gpu` there would name a region of a reservation
    /// that does not exist.
    pub(super) fn hot_location(&self) -> ArenaLocation {
        if self.device.is_cpu() {
            ArenaLocation::Cpu
        } else {
            ArenaLocation::Gpu
        }
    }

    /// Elements one chunk slot holds at this backing's geometry — one
    /// `(head, palette-band, side)` of `CHUNK_SIZE` tokens.
    ///
    /// The single authority for the number every size-class question is asked
    /// with, so a model whose `head_dim / N_PALETTE` is not 32 cannot end up
    /// with half its arenas sized from the assumption that it is (audit A9).
    pub(super) fn elems_per_chunk(&self) -> usize {
        elems_per_chunk((self.head_dim / N_PALETTE).max(1))
    }

    /// The arena key a chunk of `format` allocates from at this geometry.
    pub(super) fn arena_key_for(
        &self,
        format: KvFormat,
        location: ArenaLocation,
    ) -> Result<ArenaKey> {
        ArenaKey::for_format(format, self.elems_per_chunk(), location)
    }

    /// Claim the gap between forwards for one arena, or refuse.
    ///
    /// `None` — no gate at all — in the two cases where creating an arena cannot
    /// disturb a wave:
    ///
    /// - **a CPU arena**, which is a host `Tensor::zeros` ([`Self::claim_slab`]).
    ///   It claims no region, sits nowhere in the reservation, and cannot move
    ///   the arena frontier the transient tier was placed against. Gating it was
    ///   a bug, not caution: the hot→warm migration allocates its warm copies
    ///   exactly this way, under the sequence-state write lock, and blocking
    ///   there deadlocked the daemon against a forward that wanted the same lock
    ///   mid-layer.
    /// - **a CPU device**, which has no reservation and no tier.
    ///
    /// Every caller of [`Self::create_arena`] must hold one of these. The one
    /// that cannot take it here — [`ChunkedKvBacking::alloc_chunk_with_arenas`]
    /// creates arenas with the storage write lock already held — takes it at its
    /// own call site instead, above that lock.
    #[cfg(feature = "cuda")]
    pub(super) fn arena_window(&self, key: ArenaKey) -> Result<Option<ArenaWindow>> {
        let Device::Cuda(cd) = &self.device else {
            return Ok(None);
        };
        if key.location == ArenaLocation::Cpu {
            return Ok(None);
        }
        enter_arena_window(&cd.cuda_stream()).map(Some)
    }

    /// The same gate for an operation that **cannot be refused part-way**.
    ///
    /// A fork is per *layer* — `KvCache::fork` runs once per layer and the caller
    /// loops over all forty-eight — and there is no transaction across them. So a
    /// refusal on layer N leaves N forked layers and 48−N unforked ones, which is
    /// not a deferred fork: it is a sequence whose token windows differ by layer,
    /// which no single decode position map can describe. It surfaced as
    /// `chunked decode layout diverged across layers … could not be reconciled`
    /// on the first message of a new conversation.
    ///
    /// Refusing costs more than proceeding here, so this one proceeds — but
    /// only past the *wave-in-flight* refusal. That refusal it absorbs and
    /// **records the class**, so the next inter-forward gap creates an arena
    /// for it and the case stops arising; the region claim is still counted by
    /// `fresh_claims_during_wave`, so an engine that takes this path often says
    /// so in the `in-wave-arenas` tripwire rather than hiding it. Every other
    /// error — a poisoned gate, a driver fault — is not a deferral and
    /// propagates: proceeding past a broken gate would fork against arenas
    /// whose state can no longer be trusted.
    #[cfg(feature = "cuda")]
    pub(super) fn arena_window_uninterruptible(
        &self,
        key: ArenaKey,
    ) -> Result<Option<ArenaWindow>> {
        match self.arena_window(key) {
            Ok(w) => Ok(w),
            Err(e) if Self::is_wave_deferral(&e) => {
                self.note_deferred_arena(key);
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Whether an allocation failure is the wave-in-flight deferral.
    ///
    /// A deferral means "the ground exists, come back between forwards"; every
    /// other failure here means "there is no ground". They call for opposite
    /// responses, so no path may fold one into the other.
    #[cfg(feature = "cuda")]
    fn is_wave_deferral(e: &candle::Error) -> bool {
        e.to_string().contains(KV_ARENA_MID_WAVE)
    }

    /// Remember that a wave-in-flight refusal wanted an arena of this class.
    ///
    /// Idempotent per class: one arena of a class serves every chunk that fits
    /// it, so a pass refused twenty times for `1088 B` needs one arena, not
    /// twenty. Recording the count would over-create on the very workload the
    /// refusal is most common on.
    #[cfg(feature = "cuda")]
    fn note_deferred_arena(&self, key: ArenaKey) {
        if let Ok(mut pending) = self.deferred_arenas.lock() {
            if !pending.contains(&key) {
                pending.push(key);
            }
        }
    }

    /// Create the arenas that mid-wave refusals asked for, and answer with how
    /// many were made.
    ///
    /// **Call this between forwards, with no KV lock held** — the top of a
    /// sealing pass is the intended site. It is the other half of the refusal:
    /// the pass that was turned away recorded what it wanted, and this creates
    /// it while the partition is idle, so the pass that follows finds the arena
    /// already there and only ever *fills* it. Filling is legal at any time; it
    /// is creation that moves the arena frontier, and this is what keeps the two
    /// apart without either side waiting on the other.
    ///
    /// Refusals here are ordinary and silent: if a forward has started, the
    /// demand stays recorded and the next gap gets it. Only a genuine allocation
    /// failure — the reservation truly full — propagates.
    #[cfg(feature = "cuda")]
    pub(super) fn create_deferred_arenas(&self) -> Result<usize> {
        let pending: Vec<ArenaKey> = match self.deferred_arenas.lock() {
            Ok(mut p) => std::mem::take(&mut *p),
            Err(_) => return Ok(0),
        };
        if pending.is_empty() {
            return Ok(0);
        }
        let mut made = 0usize;
        let mut remaining = pending.into_iter();
        while let Some(key) = remaining.next() {
            // Every exit that does not finish this key puts back the key **and
            // everything after it** — the list was taken whole, so leaving with
            // only the current one restored would silently drop the rest and
            // the classes behind it would never be created. That holds for the
            // benign "a wave started" refusal and for genuine errors alike: an
            // error aborts this drain, not the demand, and the next gap must
            // still see it.
            let requeue = |first: ArenaKey, rest: &mut dyn Iterator<Item = ArenaKey>| {
                self.note_deferred_arena(first);
                for key in rest {
                    self.note_deferred_arena(key);
                }
            };
            // One window per arena rather than one for the batch: a forward that
            // arrives mid-drain then takes the partition at the next arena
            // instead of waiting for the whole list.
            let window = match self.arena_window(key) {
                Ok(w) => w,
                Err(e) if Self::is_wave_deferral(&e) => {
                    requeue(key, &mut remaining);
                    break;
                }
                Err(e) => {
                    requeue(key, &mut remaining);
                    return Err(e);
                }
            };
            let idx = self.pool.register_arena(key);
            match self.create_arena(key, idx).and_then(|arena| {
                self.storage.try_write(|s| {
                    if !s.has_arena(idx) {
                        s.push_arena(arena, idx);
                    }
                    Ok(())
                })
            }) {
                Ok(()) => made += 1,
                Err(e) => {
                    // The slab (if it was even created) never reached storage:
                    // release the registration so the index is not leaked, and
                    // put the demand back so the next gap retries it.
                    self.pool.force_release_arena(idx);
                    drop(window);
                    requeue(key, &mut remaining);
                    return Err(e);
                }
            }
        }
        Ok(made)
    }

    /// Create one arena: a `chunks x class_bytes` slab of raw bytes.
    ///
    /// **Only ever between forwards.** The caller holds an
    /// [`Self::arena_window`], because this claims a region and so moves the
    /// arena frontier a running wave's transient tier was placed against.
    ///
    /// There is no float/quantized fork any more. A slot is a fixed number of
    /// bytes and its tenant is whatever the owning chunk's tag says it is, so
    /// the allocator has exactly one thing to decide — how wide the slots are —
    /// and the class carries that (`docs/archived/arena_unification.md` principle 8).
    ///
    /// A GPU arena **carves a region** out of the device reservation: no
    /// `cuMemAlloc`, no pool growth, no chance of the driver spilling the slab
    /// to host memory, and a base pointer that stays valid until the process
    /// ends. Running out is a `KV_DEVICE_OOM_MARKER` error the way exceeding
    /// the old budget was — the difference is that it now means "every region
    /// is occupied", an exact count rather than an estimate of driver headroom.
    pub(super) fn create_arena(&self, key: ArenaKey, index: usize) -> Result<Arena> {
        let t0 = Instant::now();
        let chunks = key.chunks();
        let stride = key.slot_stride();
        let arena_bytes = chunks.saturating_mul(stride);
        let out = self.claim_slab(key, index, arena_bytes);
        record_arena_create("slab", key.location, index, t0.elapsed().as_nanos() as u64);
        out
    }

    #[cfg(feature = "cuda")]
    fn claim_slab(&self, key: ArenaKey, index: usize, arena_bytes: usize) -> Result<Arena> {
        if key.location == ArenaLocation::Cpu {
            let data = Tensor::zeros(arena_bytes, DType::U8, &Device::Cpu)?;
            return Ok(Arena::new(data, key.class, key.location, index));
        }
        let Device::Cuda(cuda) = &self.device else {
            candle::bail!("a GPU arena needs a CUDA device, not {:?}", self.device)
        };
        let stream = cuda.cuda_stream();
        let Some(region) = region_pool::claim_region(&stream)? else {
            // **Name which of the two refusals this is.** A claim that runs out
            // of ground buys more from the weight side, so reaching here means
            // one of exactly two things: a wave's tier stands over the free
            // regions (no concession can reach them — the wave must narrow), or
            // the purchase itself was refused because the weight zone is at its
            // floor. They want opposite responses and the message has to say
            // which. It used to report only `live` and assert the reservation
            // was occupied, which sent the first investigation of this looking
            // for a KV leak while the real answer was 31 free regions standing
            // under a 496 MiB tier.
            let s = region_pool::region_stats(stream.context().ordinal());
            let (live, total, blocked, ceiling, tier) = s
                .map(|s| {
                    (
                        s.live,
                        s.total,
                        s.blocked,
                        s.transient_ceiling,
                        s.transient_bytes,
                    )
                })
                .unwrap_or((0, 0, 0, 0, 0));
            if blocked > 0 {
                candle::bail!(
                    "{KV_DEVICE_OOM_MARKER}: no region is claimable for class {} B — the wave's \
                     {} MiB transient tier caps the pool at {ceiling} of {total} regions, all \
                     {live} of which are live. {blocked} regions above the tier are unowned but \
                     out of reach until the wave ends. This is a wave too wide for the ground \
                     below the tier, not an exhausted reservation.",
                    key.class.bytes(),
                    tier / (1 << 20),
                )
            }
            candle::bail!(
                "{KV_DEVICE_OOM_MARKER}: no region is claimable for class {} B — every one of \
                 the KV reservation's {total} regions is occupied ({live} live), and the weight \
                 side would not sell any. It is at its floor: the fewest expert slots the cache \
                 can serve a token with. The partition has nothing left to trade.",
                key.class.bytes(),
            )
        };
        // A lease over the region: writes through it land in the reservation,
        // and dropping the tensor frees nothing. What returns the bytes is the
        // region handle the arena carries (§3.7).
        //
        // SAFETY: the region is `REGION_BYTES` of mapped, read/write device
        // memory that only this arena holds, and `arena_bytes` never exceeds a
        // region (asserted by the size-class ladder).
        let data = unsafe {
            Tensor::from_leased_cuda_ptr(
                region.base(),
                DType::U8,
                arena_bytes,
                &self.device,
                LeaseOrigin::Foreign,
            )?
        };
        Ok(Arena::new(data, key.class, key.location, index).in_region(region))
    }

    #[cfg(not(feature = "cuda"))]
    fn claim_slab(&self, key: ArenaKey, index: usize, arena_bytes: usize) -> Result<Arena> {
        let device = match key.location {
            ArenaLocation::Gpu => &self.device,
            ArenaLocation::Cpu => &Device::Cpu,
        };
        let data = Tensor::zeros(arena_bytes, DType::U8, device)?;
        Ok(Arena::new(data, key.class, key.location, index))
    }
}

impl ChunkedKvBacking {
    /// Allocate a chunk with pre-acquired arena lock.
    /// This version takes a pre-acquired arenas lock to avoid deadlock when called
    /// from contexts that already hold the arena lock.
    ///
    /// **The caller must already hold an [`BackingInner::arena_window`].** Both
    /// branches below can create an arena, and they do it with the storage write
    /// lock held — so they cannot wait for the inter-forward gap themselves
    /// without sleeping on a lock a forward's admit needs.
    pub(super) fn alloc_chunk_with_arenas(
        &self,
        arena_state: &mut ArenaStorageState,
        key: ArenaKey,
    ) -> Result<ChunkGid> {
        if let Some(gid) = self.inner.pool.allocate_for(key) {
            let arena_idx = gid.arena_idx();
            let arena_was_fresh = !arena_state.has_arena(arena_idx);
            if arena_was_fresh {
                let arena = self.create_arena(key, arena_idx)?;
                arena_state.push_arena(arena, arena_idx);
            } else {
                // Free-list reuse on an existing arena: the chunk's bytes are
                // whatever the prior tenant left. Zero them so the new tenant
                // (and any persist quantize pass that reads past token_count)
                // sees clean storage. Fresh arenas are already zero from
                // `Tensor::zeros` at creation, so the arena_was_fresh branch
                // above skips the work.
                self.zero_recycled_chunk(arena_state, arena_idx, gid.chunk_idx())?;
            }
            return Ok(gid);
        }

        // The class is starved. Stamp a region for it; failing that, take a
        // free slot from a wider class that already has one. Same gate and
        // same order as `BackingInner::stamp_region_promoting`, open-coded
        // because this path already holds the storage lock and so cannot call
        // `ensure_arena_exists`.
        let mut key = key;
        let arena_idx = self.inner.pool.register_arena(key);
        if !arena_state.has_arena(arena_idx) {
            match self.create_arena(key, arena_idx) {
                Ok(arena) => arena_state.push_arena(arena, arena_idx),
                Err(e) => {
                    self.inner.pool.force_release_arena(arena_idx);
                    let mut class = key.class;
                    let mut placed = None;
                    while let Some(next) = class.promote() {
                        class = next;
                        let wider = ArenaKey::new(class, key.location);
                        if let Some(gid) = self.inner.pool.allocate_for(wider) {
                            let ai = gid.arena_idx();
                            drop(gid);
                            if arena_state.has_arena(ai) {
                                record_class_promotion(key.class, class);
                                placed = Some(wider);
                                break;
                            }
                        }
                    }
                    match placed {
                        Some(w) => key = w,
                        None => return Err(e),
                    }
                }
            }
        }

        let gid = self
            .inner
            .pool
            .allocate_for(key)
            .ok_or_else(|| candle::Error::Msg("no slot in the region just claimed".into()))?;
        Ok(gid)
    }

    /// Zero one recycled chunk's bytes. Asynchronous on CUDA — the write is
    /// enqueued on the slab's own stream and the call returns once queued.
    /// Same-stream FIFO ordering guarantees the next reader of this chunk sees
    /// the zeros without an explicit fence.
    fn zero_recycled_chunk(
        &self,
        arena_state: &mut ArenaStorageState,
        arena_idx: usize,
        chunk_idx: usize,
    ) -> Result<()> {
        let Some(arena) = arena_state.arenas_mut().get_mut(&arena_idx) else {
            return Ok(());
        };
        arena.zero_chunk_at(chunk_idx)
    }

    /// ArenaKey for active (unfilled) K chunks.
    ///
    /// On CUDA we keep the fast R16 active-K path for the decode/prefill kernels.
    /// On CPU we keep active K chunks in float so partial-token writes and tests
    /// do not require block-aligned quantization on every append.
    pub(super) fn active_k_arena_key(&self) -> Result<ArenaKey> {
        let location = self.inner.storage.default_location();
        let (k, _) = crate::kv_cache::active_kv_formats(
            self.inner.storage.k_format(),
            matches!(location, ArenaLocation::Gpu),
        );
        self.inner.arena_key_for(k, location)
    }

    /// ArenaKey for active (unfilled) V chunks — always float.
    pub(super) fn active_v_arena_key(&self) -> Result<ArenaKey> {
        let location = self.inner.storage.default_location();
        let (_, v) = crate::kv_cache::active_kv_formats(
            self.inner.storage.k_format(),
            matches!(location, ArenaLocation::Gpu),
        );
        self.inner.arena_key_for(v, location)
    }

    /// Allocate a full block's worth of flat chunks for the palette4 arenas.
    ///
    /// Returns a `ChunkWindow` with `GIDS_PER_HEAD * n_kv_head` GIDs (N_PALETTE per
    /// head × {K, V}).  Each chunk stores exactly `CHUNK_SIZE * (head_dim / N_PALETTE)`
    /// elements — one head, one palette sub-band, one side.
    ///
    /// HeadGids layout: `head * GIDS_PER_HEAD + palette * 2 + is_value`.
    pub(super) fn alloc_block_chunks(&self, usage: u32, offset: u16) -> Result<ChunkWindow> {
        // Band count per head: LATENT_N_BANDS (single-latent) or N_PALETTE (GQA).
        let np = self.inner.n_palette();
        let n = np * 2 * self.inner.n_kv_head;
        let k_key = self.active_k_arena_key()?;
        let v_key = self.active_v_arena_key()?;
        // Rope-region key for the single latent: the 64 RoPE dims (bands
        // [LATENT_NOPE_BANDS, np)) are pinned BF16 regardless of the writer
        // format, matching the reference (`nope FP8 ‖ rope BF16`). When the
        // writer format is already BF16 (the wave window) this equals `k_key`,
        // so the store is uniform BF16 — the pre-existing behaviour, only the
        // arena width narrows to the single-latent band width.
        let rope_key = self.inner.arena_key_for(
            KvFormat::Float(DType::BF16),
            self.inner.storage.default_location(),
        )?;
        let mut gids = Vec::with_capacity(n);
        // Per head: one CONTIGUOUS run of N_PALETTE K slots and one of V slots
        // (see `alloc_chunk_run_for_key`). Correctness does NOT depend on the
        // run layout — every kernel addresses each band through its own gid
        // (`resolve_band_source`, and the per-palette KvHead record pointers) —
        // but contiguous bands give the select/QREL walk better spatial
        // locality, so we mint them as runs where a run is available.
        // HeadGids layout stays `head * GIDS_PER_HEAD + palette * 2 + is_value`.
        let single_latent = self
            .inner
            .single_latent
            .load(std::sync::atomic::Ordering::Relaxed);
        for _h in 0..self.inner.n_kv_head {
            if single_latent {
                // Two-region window: bands [0, LATENT_NOPE_BANDS) back the 448-d
                // nope span in the writer format (FP8 E4M3 for the reference
                // config); bands [LATENT_NOPE_BANDS, np) back the 64-d rope tail
                // in BF16. Each region is one contiguous chunk run; the KvHead
                // record still fills all 16 band slots (bands resolve their
                // per-band {ptr, fmt, scale} from their own gid's arena, so the
                // format tag follows the region automatically).
                //
                // K≡V: the V band aliases the K band. `ChunkGid` is a refcounted
                // handle, so the double reference keeps the chunk alive until
                // both drop — V storage costs nothing and every table consumer
                // sees v_ptr == k_ptr.
                let nope_bands = crate::kv_cache::arena_table::LATENT_NOPE_BANDS.min(np);
                let rope_bands = np - nope_bands;
                let nope_run = self.inner.alloc_chunk_run_for_key(k_key, nope_bands)?;
                let rope_run = if rope_bands > 0 {
                    self.inner.alloc_chunk_run_for_key(rope_key, rope_bands)?
                } else {
                    Vec::new()
                };
                for k_gid in nope_run.into_iter().chain(rope_run) {
                    let v_gid = k_gid.clone();
                    gids.push(k_gid);
                    gids.push(v_gid);
                }
            } else {
                let k_run = self.inner.alloc_chunk_run_for_key(k_key, np)?;
                let v_run = self.inner.alloc_chunk_run_for_key(v_key, np)?;
                for (k_gid, v_gid) in k_run.into_iter().zip(v_run) {
                    gids.push(k_gid);
                    gids.push(v_gid);
                }
            }
        }

        Ok(ChunkWindow {
            gids: HeadGids::from_vec(gids),
            usage,
            offset,
            k_pal: self.inner.identity_pal.clone(),
            v_pal: self.inner.identity_pal.clone(),
            k_scale: self.inner.identity_scale.clone(),
            v_scale: self.inner.identity_scale.clone(),
            // Active formats (R16 K / F16 V on GPU) — this chunk is a writer
            // and will not reach its configured sealed format until its turn
            // seals and quantizes. Shared `Arc`, no per-chunk allocation.
            k_fmt: self.inner.active_k_fmt.clone(),
            v_fmt: self.inner.active_v_fmt.clone(),
            // Fresh float writer chunk: transient, no resident record. The host
            // serializer builds per-forward scratch heads for it.
            meta: None,
        })
    }

    pub(super) fn alloc_chunk_for_key(
        &self,
        key: super::arena::ArenaKey,
    ) -> Result<super::gid_pool::ChunkGid> {
        self.inner.alloc_chunk_for_key(key)
    }

    /// Create the arenas that mid-wave refusals asked this backing for.
    ///
    /// See [`BackingInner::create_deferred_arenas`]. Between forwards only, with
    /// no KV lock held — the top of a sealing pass.
    #[cfg(feature = "cuda")]
    pub fn create_deferred_arenas(&self) -> Result<usize> {
        self.inner.create_deferred_arenas()
    }

    /// See [`BackingInner::alloc_chunk_run_for_key`].
    pub(super) fn alloc_chunk_run_for_key(
        &self,
        key: super::arena::ArenaKey,
        len: usize,
    ) -> Result<Vec<super::gid_pool::ChunkGid>> {
        self.inner.alloc_chunk_run_for_key(key, len)
    }

    /// Bulk variant of [`Self::alloc_chunk_for_key`] — allocates `n`
    /// GIDs of the same `key` while paying the per-format pool mutex
    /// and the per-arena storage write lock only **once each**
    /// (instead of `n` times). Used by the cold-load
    /// `alloc_sealed_blocks_bulk` path where a single layer can need
    /// ~600 GIDs of the same format.
    pub(super) fn alloc_chunks_for_key_bulk(
        &self,
        key: super::arena::ArenaKey,
        n: usize,
    ) -> Result<Vec<super::gid_pool::ChunkGid>> {
        self.inner.alloc_chunks_for_key_bulk(key, n)
    }

    pub(super) fn ensure_arena_exists(
        &self,
        arena_idx: usize,
        key: super::arena::ArenaKey,
    ) -> Result<()> {
        self.inner.ensure_arena_exists(arena_idx, key)
    }
}

/// # Claiming a chunk slot
///
/// The pool's lock-free refcount table is the single source of truth for "this
/// slot is allocated" — `pool.allocate_for` performs the CAS-claim that flips
/// it. Storage only has to ensure the physical arena tensor exists; no separate
/// per-slot bookkeeping is needed.
///
/// 1. Try `pool.allocate_for(key)` — reuses a freed slot from any arena of this class.
/// 2. If no capacity: stamp a region for the class.
/// 3. If no region can be had: take a free slot from a wider class that
///    already has one (scarcity-only promotion, §3.4).
impl BackingInner {
    /// Register a fresh region for `key`, or place the chunk in an
    /// **already-stamped** wider class when no region can be had. Returns the
    /// key the slot actually came from and the arena holding it.
    ///
    /// # Why promotion reuses rather than re-stamps
    ///
    /// Every class's region is the same [`TARGET_ARENA_BYTES`], so if a region
    /// cannot be had for one class it cannot be had for a wider one either —
    /// walking up the ladder *stamping* would fail identically at every rung
    /// while paying `ensure_vram_budget`'s global compaction each time. (It
    /// did, in the first version of this: one refusal became seven, and the
    /// gate lost 30 % of single-stream decode.)
    ///
    /// What promotion is actually for is stated in §3.4: stopping a trickle of
    /// a rare format from stamping a whole 16 MiB region for a class that will
    /// never fill it. That means taking a **free slot in a class that already
    /// has a region**, which costs nothing and is the only outcome that can
    /// succeed where the stamp failed.
    ///
    /// Strictly scarcity-gated, and in this order: a class gets its own region
    /// whenever one is available, so promotion cannot become a background
    /// mixing vector.
    fn stamp_region_promoting(&self, key: ArenaKey) -> Result<(ArenaKey, usize)> {
        let stamp_err = match self.claim_fresh_region(key) {
            Ok(arena_idx) => return Ok((key, arena_idx)),
            // **A wave-in-flight deferral is not scarcity, and must not be
            // treated as it.** Promotion exists to stop a rare format stamping a
            // whole region for itself when no region can be had; here a region
            // can be had, just not this instant. Widening in response puts the
            // chunk in the wrong class for a reason that will have passed by the
            // next pass — and, worse, the caller's retry loop then reports the
            // failure as `VRAM exhaustion on arena creation` while a fifth of the
            // reservation stands free. Hand the deferral straight back so it
            // reaches the sealing pass as itself.
            #[cfg(feature = "cuda")]
            Err(e) if Self::is_wave_deferral(&e) => return Err(e),
            Err(e) => e,
        };
        // No region. Look for a wider class that already has one with room.
        let mut class = key.class;
        while let Some(next) = class.promote() {
            class = next;
            let wider = ArenaKey::new(class, key.location);
            if let Some(gid) = self.pool.allocate_for(wider) {
                let arena_idx = gid.arena_idx();
                // The gid is dropped here; the caller re-claims from `wider`.
                // Dropping returns the slot, so this is a probe, not a leak.
                drop(gid);
                if self
                    .storage
                    .read(|s| s.has_arena(arena_idx))
                    .unwrap_or(false)
                {
                    record_class_promotion(key.class, class);
                    return Ok((wider, arena_idx));
                }
            }
        }
        Err(stamp_err)
    }

    /// Register and materialise one region for `key`, rolling the registration
    /// back if it cannot be materialised.
    ///
    /// Without the rollback the pool would advertise free slots that storage
    /// cannot produce: every later claim into that arena fails the same way,
    /// and `total_arenas` inflates the occupancy diagnostic.
    fn claim_fresh_region(&self, key: ArenaKey) -> Result<usize> {
        let arena_idx = self.pool.register_arena(key);
        if let Err(e) = self.ensure_arena_exists(arena_idx, key) {
            self.pool.force_release_arena(arena_idx);
            return Err(e);
        }
        Ok(arena_idx)
    }

    /// Claim one chunk slot for `key`, **promoting up the size-class ladder**
    /// when the class is starved and no region can be had for it.
    ///
    /// Correctness is untouched by the wider slot: every read takes its extent
    /// from the band's *format* bytes, never from the stride, so a chunk in a
    /// larger class is simply a chunk with more unread pad
    /// (`docs/archived/arena_unification.md` §3.4, invariant 8). Only the waste changes.
    pub(super) fn claim_slot_promoting(&self, key: ArenaKey) -> Result<super::gid_pool::ChunkGid> {
        if let Some(gid) = self.pool.allocate_for(key) {
            self.ensure_arena_exists(gid.arena_idx(), key)?;
            return Ok(gid);
        }
        let (placed, _) = self.stamp_region_promoting(key)?;
        self.pool.allocate_for(placed).ok_or_else(|| {
            candle::Error::Msg(
                "claim_slot_promoting: the region that just reported room has none".into(),
            )
        })
    }

    pub(super) fn alloc_chunk_for_key(
        &self,
        key: super::arena::ArenaKey,
    ) -> Result<super::gid_pool::ChunkGid> {
        self.claim_slot_promoting(key)
    }

    /// Allocate `len` CONSECUTIVE slots in one arena of `key`. Contiguity is a
    /// LOCALITY optimization for the paged select/QREL walk, not a correctness
    /// requirement — each band is addressed through its own gid
    /// (`resolve_band_source`), so scattered bands read correctly, just with
    /// worse spatial locality. Falls back to singleton allocation when no run
    /// is available. Mirrors [`Self::alloc_chunk_for_key`]'s
    /// register-on-exhaustion retry.
    pub(super) fn alloc_chunk_run_for_key(
        &self,
        key: super::arena::ArenaKey,
        len: usize,
    ) -> Result<Vec<super::gid_pool::ChunkGid>> {
        // A run larger than one arena can NEVER be satisfied — arenas are
        // fixed-capacity slabs and a run must be contiguous within one. Fail
        // with the sizes so this permanent condition is never mistaken for the
        // transient race below (they used to share one message, and a night went
        // into telling them apart).
        let arena_chunks = key.chunks();
        if len > arena_chunks {
            candle::bail!(
                "palette run of {len} chunks exceeds arena capacity {arena_chunks} \
                 for class {} B — cannot be satisfied by any arena",
                key.slot_stride(),
            );
        }
        if let Some(gids) = self.pool.allocate_run_for(key, len) {
            self.ensure_arena_exists(gids[0].arena_idx(), key)?;
            self.replenish_if_nearly_dry(key, len);
            return Ok(gids);
        }
        // No existing arena has tail room: register a fresh one and claim from
        // it BY INDEX. The old shape — register, then re-walk the whole pool —
        // raced: between registration and the re-walk, concurrent claimers (the
        // scheduler's prefills and the persistence thread's elevations allocate
        // the same formats in parallel) could consume the fresh arena's tail,
        // and the single retry then failed spuriously as "fresh arena cannot
        // fit palette run", killing the whole forward. Targeting the registered
        // index removes the which-arena race; losing even that (racers landing
        // in OUR arena via their own global walks) just means another
        // registration, bounded.
        const ATTEMPTS: usize = 4;
        let mut key = key;
        for _ in 0..ATTEMPTS {
            // The whole run widens together when the class cannot get a
            // region, so its bands stay in one arena and keep the locality the
            // run exists for.
            let (placed, arena_idx) = self.stamp_region_promoting(key)?;
            key = placed;
            let _ = arena_chunks;
            if let Some(gids) = self.pool.allocate_run_for_in(key, arena_idx, len) {
                self.replenish_if_nearly_dry(key, len);
                return Ok(gids);
            }
            // Raced into our fresh arena — the racer may equally have vacated
            // tail room elsewhere; check the whole pool before registering again.
            if let Some(gids) = self.pool.allocate_run_for(key, len) {
                self.ensure_arena_exists(gids[0].arena_idx(), key)?;
                self.replenish_if_nearly_dry(key, len);
                return Ok(gids);
            }
        }
        candle::bail!(
            "palette run of {len} chunks unsatisfied after {ATTEMPTS} fresh arenas \
             (capacity {arena_chunks} each, class {} B) — allocator contention or \
             VRAM exhaustion on arena creation",
            key.slot_stride(),
        )
    }

    /// Bulk allocator — mirrors [`Self::alloc_chunk_for_key`]'s
    /// register-on-exhaustion loop but in batch.
    ///
    /// Per pass:
    /// - **One** `pool.allocate_n_for(key, remaining)` returns up to
    ///   `remaining` GIDs; CAS-claim makes the refcount table
    ///   immediately authoritative, no follow-up bookkeeping required.
    /// - **One** `ensure_arena_exists` per unique arena index we
    ///   touched (cheap — the inner check is a `storage.read`).
    ///
    /// If the pool returned fewer GIDs than requested, the format's
    /// pool was exhausted — we register a fresh arena (one
    /// `register_arena + ensure_arena_exists` round) and re-enter the
    /// loop to fill the remainder. Same termination guarantee as the
    /// singular path.
    pub(super) fn alloc_chunks_for_key_bulk(
        &self,
        key: super::arena::ArenaKey,
        n: usize,
    ) -> Result<Vec<super::gid_pool::ChunkGid>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut out: Vec<super::gid_pool::ChunkGid> = Vec::with_capacity(n);
        let mut key = key;
        while out.len() < n {
            let remaining = n - out.len();
            let batch = self.pool.allocate_n_for(key, remaining);
            if batch.is_empty() {
                // Pool exhausted — stamp a fresh region, widening if this
                // class cannot get one. Later passes then fill from the class
                // that could, so a partially-promoted batch is normal.
                let (placed, _) = self.stamp_region_promoting(key)?;
                key = placed;
                continue;
            }
            // Ensure every unique arena index we just got is materialised
            // in storage. Most calls hit the cheap `storage.read`-only
            // path because the arena already exists.
            let mut seen: ahash::HashSet<usize> =
                ahash::HashSet::with_capacity_and_hasher(4, ahash::RandomState::new());
            for gid in &batch {
                let ai = gid.arena_idx();
                if seen.insert(ai) {
                    self.ensure_arena_exists(ai, key)?;
                }
            }
            out.extend(batch);
        }
        // One single-slot probe for the whole batch, at the final key. The bulk
        // allocator takes arbitrary slots rather than contiguous runs, so "could
        // this batch run again" has no cheap probe — but "is the class bone dry"
        // does, and a dry class is the case that costs the sealer a deferred
        // pass.
        self.replenish_if_nearly_dry(key, 1);
        Ok(out)
    }

    /// Keep the sealer one arena ahead of its demand: if `key`'s class could
    /// not serve `len` more slots right now, ask the next inter-forward gap
    /// for an arena — **before** anything actually runs out.
    ///
    /// This is the sealing buffer. The sealer may fill existing arenas at any
    /// point in a wave but may only *create* one between forwards, so a class
    /// that runs dry mid-wave costs a deferred pass: the selection work is
    /// redone and the hot→warm drain slips a wave. Probing after each
    /// successful allocation converts that into a class that is replenished in
    /// the gap *before* the next pass needs it — steady state never sees the
    /// refusal, and the deferral remains only for the cold start of a class no
    /// one has used yet.
    ///
    /// The probe is an allocate-and-drop, the same pattern
    /// [`Self::stamp_region_promoting`] uses: dropping the gids returns the
    /// slots, so it observes without consuming. CPU classes are skipped — their
    /// creation is never gated, so there is nothing to get ahead of.
    #[cfg(feature = "cuda")]
    fn replenish_if_nearly_dry(&self, key: ArenaKey, len: usize) {
        if key.location == ArenaLocation::Cpu {
            return;
        }
        // Read-only: an allocate-and-drop probe here permanently burns run
        // capacity, because a run claim advances the arena's never-used
        // high-water mark and dropped run gids recycle through the singleton
        // stack that run claims never read. Measured consequence of the probe:
        // every successful run allocation consumed double its length, arenas
        // ran dry mid-sweep, and the in-wave refusal this function exists to
        // prevent came back.
        if !self.pool.run_would_fit(key, len.max(1)) {
            self.note_deferred_arena(key);
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn replenish_if_nearly_dry(&self, _key: ArenaKey, _len: usize) {}

    /// Ensure that an arena exists at the given index in storage.
    ///
    /// Creates the arena if it does not exist yet — **waiting for the gap
    /// between forwards to do it**. A wave is pre-allocated end to end, and a
    /// new arena is the one KV allocation that cannot be: it moves the arena
    /// frontier the running wave's transient tier was placed against. Filling an
    /// existing arena is unrestricted and is what the sealing thread does for
    /// nearly every chunk; only the creation waits.
    ///
    /// A refusal **records what it wanted** before it propagates, so the next
    /// pass can create it in the gap rather than rediscovering the need at the
    /// same depth and failing the same way — see [`Self::create_deferred_arenas`].
    pub(super) fn ensure_arena_exists(&self, arena_idx: usize, key: ArenaKey) -> Result<()> {
        let exists = self.storage.read(|s| s.has_arena(arena_idx))?;
        if exists {
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        let _window = match self.arena_window(key) {
            Ok(w) => w,
            Err(e) => {
                self.note_deferred_arena(key);
                return Err(e);
            }
        };
        let arena = self.create_arena(key, arena_idx)?;
        self.storage.try_write(|s| {
            if !s.has_arena(arena_idx) {
                s.push_arena(arena, arena_idx);
            }
            Ok(())
        })?;
        Ok(())
    }
}

impl ChunkedKvBacking {
    /// Ensure that chunks needed to write `add` tokens at `offsets` are allocated.
    ///
    /// `offsets` must have exactly `batch_capacity()` elements, one per sequence slot.
    pub fn ensure_for_offsets(&self, offsets: &[usize], adds: &[usize]) -> Result<()> {
        let batch = self.batch_capacity();
        if offsets.len() != batch {
            candle::bail!(
                "offset count mismatch: got {} offsets for chunked backing batch {}",
                offsets.len(),
                batch
            )
        }
        if adds.len() != offsets.len() {
            candle::bail!(
                "ensure_for_offsets: {} adds for {} offsets",
                adds.len(),
                offsets.len()
            )
        }
        if adds.iter().all(|&a| a == 0) {
            return Ok(());
        }

        let mut required_max_blocks = 1usize;
        for (i, &off) in offsets.iter().enumerate() {
            let end_pos = off.saturating_add(adds[i]).saturating_sub(1);
            let need_blocks = (end_pos / CHUNK_SIZE) + 1;
            required_max_blocks = cmp::max(required_max_blocks, need_blocks);
        }
        self.ensure_max_blocks(required_max_blocks)?;

        let chunk_size = CHUNK_SIZE;

        // Count first, allocate WITHOUT the guard, then install. `alloc_block_chunks`
        // can reach `request_global_compact`, which needs a write guard on every
        // layer's block table; allocating under one made that a self-deadlock and,
        // once the compactor was made non-blocking, a permanent no-op — so arena
        // compaction could never run from the prefill path that needs it most.
        // See `ensure_for_batch_entries` for the full rationale.
        // Plan blocks AND predict tail needs in one read pass, allocate both with
        // no guard held, then mutate under ONE write guard — extending a sequence
        // and making its tail writable must be atomic (a reader seeing blocks
        // pushed but the tail unreplaced would write into a full or closed-quant
        // chunk). Predicting before the installs over-estimates safely: a freshly
        // pushed block is writable, so installs only shrink the need.
        let mut plan: Vec<(usize, usize)> = Vec::new();
        let mut tail_maybe: Vec<usize> = Vec::new();
        {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (b, &off) in offsets.iter().enumerate() {
                if state.sequences[b].is_none() || adds[b] == 0 {
                    continue;
                }
                let end_pos = off.saturating_add(adds[b]).saturating_sub(1);
                let need_blocks = (end_pos / chunk_size) + 1;
                let slot = state.sequences[b].as_ref().unwrap();
                let missing = (0..need_blocks)
                    .filter(|&blk| slot.chunk_at(blk).is_none())
                    .count();
                if missing > 0 {
                    plan.push((b, missing));
                }
            }
            for b in 0..batch {
                if state.sequences[b].is_some() && self.tail_needs_new_block(&state, b) {
                    tail_maybe.push(b);
                }
            }
        }
        let mut prealloc: Vec<(usize, Vec<_>)> = Vec::with_capacity(plan.len());
        for (b, missing) in plan {
            let mut cws = Vec::with_capacity(missing);
            for _ in 0..missing {
                cws.push(self.alloc_block_chunks(0, 0)?);
            }
            prealloc.push((b, cws));
        }
        let mut tail_spares: Vec<(usize, _)> = Vec::with_capacity(tail_maybe.len());
        for b in tail_maybe {
            tail_spares.push((b, self.alloc_block_chunks(0, 0)?));
        }

        // Fast path — see `ensure_for_batch_entries`.
        if prealloc.is_empty() && tail_spares.is_empty() {
            return Ok(());
        }

        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (b, cws) in prealloc {
                let Some(slot) = state.sequences[b].as_mut() else {
                    continue;
                };
                for cw in cws {
                    // Seal the current write target to full capacity. We are
                    // allocating a new block because the write range extends past
                    // the previous last block; the kernel will fill every remaining
                    // position in that block. The seal must stay immediately before
                    // its push.
                    if let Some(last) = slot.last_chunk_mut() {
                        let cur_offset = last.offset;
                        let capacity = chunk_size - cur_offset as usize;
                        last.usage = capacity as u32;
                    }
                    slot.push_chunk(cw);
                }
            }
            // Writable-tail pass, same guard.
            for (b, cw) in tail_spares {
                if state.sequences[b].is_some() && self.tail_needs_new_block(&state, b) {
                    let slot = state.sequences[b].as_mut().unwrap();
                    slot.push_chunk(cw);
                }
            }
        }

        Ok(())
    }

    /// Force-push a fresh empty writer chunk onto a slot's chunk list.
    ///
    /// Unlike [`ensure_for_offset`] / [`ensure_for_batch_entries`], this
    /// always pushes a new chunk regardless of whether the slot's tail
    /// is technically writable.  Used by cumulative section ingest:
    /// after `inject_sealed_at_tail` Arc-clones the prefix sections'
    /// substrate chunks onto a fresh scratch slot, the slot's tail is
    /// the last prefix section's partial chunk (shared with substrate).
    /// Writing into it would mutate bytes other holders see as
    /// immutable section content.  Pushing a fresh empty chunk here
    /// makes the slot's *write target* a writer-owned chunk; the
    /// shared partial sits read-only just before it, and the prefill
    /// kernel starts writing at chunk-internal position 0 of the new
    /// chunk (= logical position `prefix_token_count`).
    pub fn push_empty_writer_chunk(&self, batch_idx: usize) -> Result<()> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        let current_block_count = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            state
                .sequences
                .get(batch_idx)
                .and_then(|s| s.as_ref())
                .map(|s| s.block_count())
                .unwrap_or(0)
        };
        self.ensure_max_blocks(current_block_count + 1)?;
        let cw = self.alloc_block_chunks(0, 0)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if let Some(Some(slot)) = state.sequences.get_mut(batch_idx) {
            slot.push_chunk(cw);
            // The freshly-appended empty chunk is now the writer. Advance the
            // writer boundary to it so any partial sealed chunk before it sits
            // below the boundary — its empty tail (gap) is then excluded from
            // the writer region and never written to or attended.
            let new_idx = slot.block_count().saturating_sub(1);
            slot.set_writer_start_idx(new_idx);
            slot.invalidate_gpu_chunks();
        } else {
            candle::bail!("push_empty_writer_chunk: slot {} not allocated", batch_idx)
        }
        Ok(())
    }

    /// Reserve an in-place **glue gap**: a fresh chunk of `n_tokens` valid slots
    /// appended at the slot tail, returning its block index. The gap's K/V is
    /// left uninitialised — the glue forward fills it via explicit `(slice,
    /// in_blk)` write targets, scattering before it streams, so nothing reads it
    /// unfilled.
    ///
    /// **The gap is full by construction.** It is allocated `offset =
    /// CHUNK_SIZE - n_tokens`, `usage = n_tokens`, so its valid window is the
    /// tail `[offset, CHUNK_SIZE)` and `offset + usage == CHUNK_SIZE`. This is
    /// the load-bearing invariant: a *partial* writer-owned chunk is, by the
    /// cache's own rules, an extendable writable tail — `extend_for_write_region`
    /// walks into it, `set_len` advances its usage, the writable-tail pass
    /// CoW-extends it. A *full* chunk is immutable to all of them: `write_slice`
    /// and `decode_write_chunk_idx` skip it, `set_len`'s cap is 0, `ensure`'s
    /// available-space sum counts it as 0, and the writable-tail pass pushes a
    /// fresh writer chunk instead of extending into it. The gap can therefore
    /// never be mistaken for the live writer region — which is exactly what makes
    /// the next prefill incapable of overflowing into it.
    ///
    /// `usage` is still exactly `n_tokens`, so the cumulative-usage `rope_base`
    /// of every later chunk equals its logical position by construction — the
    /// single positional convention the decode and glue kernels both read via
    /// `slice_rope` (a column's position is `slice_rope(c) + (in_blk - offset)`,
    /// so the tail window maps to `[rope_base, rope_base + n_tokens)`). The GIDs
    /// are unique (rc=1), so the glue's explicit write is safe and the next
    /// reproject's truncate frees them by refcount. `writer_start` is advanced
    /// PAST the gap so a subsequent sealed inject lands after it.
    ///
    /// Returns `(gap_block_index, in_blk_base)`, where `in_blk_base == offset` is
    /// the first valid slot of the tail window — the caller scatters the glue's
    /// K/V into `[in_blk_base, in_blk_base + n_tokens)` so the write lands exactly
    /// where this chunk's `slice_offset` expects it (no second, independent
    /// computation of the window can drift from the reservation).
    pub fn reserve_glue_gap_chunk(&self, batch_idx: usize, n_tokens: u32) -> Result<(usize, u32)> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        if n_tokens == 0 || n_tokens as usize > CHUNK_SIZE {
            candle::bail!(
                "reserve_glue_gap_chunk: n_tokens {} must be in 1..={CHUNK_SIZE}",
                n_tokens
            )
        }
        let current_block_count = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            state
                .sequences
                .get(batch_idx)
                .and_then(|s| s.as_ref())
                .map(|s| s.block_count())
                .unwrap_or(0)
        };
        // +2: the immutable gap chunk PLUS an empty writable chunk placed after it
        // (see below).
        self.ensure_max_blocks(current_block_count + 2)?;
        // Full-by-construction: valid window is the chunk tail `[offset, 32)` with
        // `offset + usage == CHUNK_SIZE`, so the gap is immutable to every
        // writer-region scan (see the doc above).
        let offset = (CHUNK_SIZE as u32 - n_tokens) as u16;
        let cw = self.alloc_block_chunks(n_tokens, offset)?;
        // A fresh empty writer chunk to sit AFTER the gap. Without it the gap is
        // `last_chunk()`, and a co-batched decode/prefill on this same slot in the
        // unified wave validates its `last_chunk()` as the writable tail — the gap
        // is full-by-construction (`offset+usage == CHUNK_SIZE`), so the write-slice
        // check fails ("writable tail is already full/stale"), the wave forward
        // aborts, and the paged kernel is left to read a stale slot → illegal
        // address. Leaving an empty writable chunk past the gap keeps the decode's
        // `last_chunk()` a real writer (matching the write path, which already
        // targets `writer_start_idx`, not the gap). The crash root at 42553ca3.
        let writer_cw = self.alloc_block_chunks(0, 0)?;
        let mut state = self
            .state
            .write()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        if let Some(Some(slot)) = state.sequences.get_mut(batch_idx) {
            slot.push_chunk(cw);
            let gap_idx = slot.block_count().saturating_sub(1);
            // Advance the writer boundary PAST the gap so it is never the active
            // writer; the glue forward fills it by explicit target, and the next
            // sealed inject appends after it. The empty writer chunk we push next
            // is exactly at `gap_idx + 1`, so the boundary lands on a real tail.
            slot.set_writer_start_idx(gap_idx + 1);
            slot.push_chunk(writer_cw);
            slot.invalidate_gpu_chunks();
            Ok((gap_idx, offset as u32))
        } else {
            candle::bail!("reserve_glue_gap_chunk: slot {} not allocated", batch_idx)
        }
    }

    /// What [`Self::ensure_for_batch_entries_all`] needs to know about this
    /// layer, read under ONE guard: whether
    /// [`Self::ensure_for_batch_entries`] would allocate anything, and each
    /// entry's [`DecodeLayout`].
    ///
    /// Both answers come from the same block-table walk, and the steady-state
    /// decode step asks 48 layers this question per token, so they are read
    /// together rather than under a guard each. A layout is `None` for a slot
    /// that is not allocated — such an entry always reports work, and an
    /// unallocated slot has no layout to compare against the other layers.
    fn probe_decode_entries(
        &self,
        entries: &[(usize, usize)],
        add: usize,
    ) -> Result<(bool, Vec<Option<DecodeLayout>>)> {
        let state = self
            .state
            .read()
            .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
        let mut needs_work = false;
        let mut layouts = Vec::with_capacity(entries.len());
        for &(batch_idx, _off) in entries.iter() {
            let slot = state.sequences.get(batch_idx).and_then(|s| s.as_ref());
            let Some(slot) = slot else {
                // Out of range or unallocated: let the real call allocate the
                // slot, or report the index.
                needs_work = true;
                layouts.push(None);
                continue;
            };
            layouts.push(Some(slot.decode_layout()));
            if add == 0 || needs_work {
                continue;
            }
            let available = slot
                .last_chunk()
                .map(|cw| CHUNK_SIZE - (cw.offset as usize + cw.usage as usize).min(CHUNK_SIZE))
                .unwrap_or(0);
            if available < add || self.tail_needs_new_block(&state, batch_idx) {
                needs_work = true;
            }
        }
        Ok((needs_work && add != 0, layouts))
    }

    /// Ensure EVERY layer's backing has capacity for the upcoming write.
    ///
    /// Hoisted out of the per-layer decode loop deliberately, but only the
    /// EXPENSIVE half was hoisted. `ensure_for_batch_entries` takes the state
    /// write guard, walks `ensure_max_blocks`, and allocates GIDs it usually
    /// drops again; running that 48 times per decoded token, in a steady state
    /// where the answer is "nothing to allocate", was almost entirely wasted
    /// work. What each layer still answers for itself is the read-only probe —
    /// one read guard over its own block table — and only the layers that
    /// report work enter the allocation path.
    ///
    /// **Each layer is asked, because layer 0 cannot answer for the rest.**
    /// Block structure is *meant* to be layer-invariant, and in steady decode it
    /// is, but it is not unconditionally so: a windowed creep prefill leaves the
    /// resumed layers holding an empty writer chunk for the next window while
    /// the layers still pending resume hold a full tail — the same skew
    /// `BatchedInferenceSession::reserve_glue_gap` has to pad out before it can
    /// reserve a gap index every layer agrees on. While layer 0 answered for all
    /// of them, "layer 0 has a writable tail" suppressed the allocation those
    /// other layers needed, and the first one reached refused the step:
    /// `computed write len 32 is invalid for chunk_size 32` out of
    /// `validate_decode_state`, mid-conversation, on a tail that was exactly
    /// full. A `debug_assert!` asserting the invariance is not a check in the
    /// release build the daemon runs.
    ///
    /// This is also the last point before the decode metadata builder collapses
    /// all 48 layers onto ONE position map, so the same probe carries the
    /// layout each layer would produce and [`Self::unify_decode_layout`]
    /// establishes the invariance that map depends on.
    pub fn ensure_for_batch_entries_all(
        backings: &[ChunkedKvBacking],
        entries: &[(usize, usize)],
        add: usize,
    ) -> Result<()> {
        let mut layouts: Vec<Vec<Option<DecodeLayout>>> = Vec::with_capacity(backings.len());
        for b in backings {
            let (needs_work, layout) = b.probe_decode_entries(entries, add)?;
            if needs_work {
                b.ensure_for_batch_entries(entries, add)?;
                // The allocation changed this layer's structure, so the layout
                // read alongside the predicate no longer describes it.
                layouts.push(b.probe_decode_entries(entries, 0)?.1);
            } else {
                layouts.push(layout);
            }
        }
        Self::unify_decode_layout(backings, entries, &layouts)
    }

    /// Bring every layer's block structure for `entries` back into agreement,
    /// so the one position map built from layer 0 describes all of them.
    ///
    /// The map encodes a `(slice_idx, in_blk)` pair per logical token and a
    /// final entry naming the write slot, and every layer's slot header points
    /// at it. The per-layer `write_slice` the kernel SCATTERS through, however,
    /// comes from that layer's own block table. Let the two disagree and the
    /// token is written into one chunk while attention is told to read it from
    /// another — no fault, no error, just a wrong answer.
    ///
    /// Divergence is repaired rather than reported because it is a state the
    /// engine legitimately produces: a windowed creep prefill leaves the
    /// resumed layers holding an empty writer chunk for the next window while
    /// the layers still pending resume do not have one, the same skew
    /// `BatchedInferenceSession::reserve_glue_gap` pads out before it can pick
    /// one gap index. The repair gives every layer a fresh empty writer chunk
    /// at a common index: pad the short layers up to the longest, then push one
    /// more onto ALL of them, so afterwards every layer has the same block
    /// count and its writer is that last, empty chunk. An empty chunk carries
    /// no tokens, so it shifts no position — the cumulative-usage rope base of
    /// every earlier chunk is untouched.
    ///
    /// What cannot be repaired is a difference in the token windows themselves:
    /// the shared prefix of the map is then wrong for some layer and no
    /// appending fixes it. That is a corrupted block table, and it errors.
    /// The first chunk index whose `(offset, usage)` differs between any two
    /// layers, with both readings and the layer numbers — the sentence that
    /// turns a digest mismatch into a place to look.
    ///
    /// Compares every layer against layer 0 and reports the earliest index that
    /// disagrees. Earliest rather than all of them because a partially-applied
    /// per-layer operation diverges from the point it stopped, so the first
    /// index *is* the boundary, and the layer number says how far it got.
    fn first_window_divergence(
        backings: &[ChunkedKvBacking],
        batch_idx: usize,
    ) -> Option<String> {
        let windows = |b: &ChunkedKvBacking| -> Option<Vec<(u16, u32)>> {
            let state = b.state.read().ok()?;
            let slot = state.sequences.get(batch_idx)?.as_ref()?;
            Some(
                slot.chunks_slice()
                    .iter()
                    .map(|c| (c.offset, c.usage))
                    .collect(),
            )
        };
        let all: Vec<Option<Vec<(u16, u32)>>> = backings.iter().map(windows).collect();
        let base = all.first()?.as_ref()?;
        // The earliest chunk any layer disagrees with layer 0 about.
        let mut first_bad: Option<usize> = None;
        for other in all.iter().flatten().skip(1) {
            for i in 0..base.len().min(other.len()) {
                if base[i] != other[i] {
                    first_bad = Some(first_bad.map_or(i, |f: usize| f.min(i)));
                    break;
                }
            }
        }
        let Some(i) = first_bad else {
            let lens: Vec<usize> = all.iter().flatten().map(|w| w.len()).collect();
            let (min, max) = (lens.iter().min()?, lens.iter().max()?);
            return Some(format!(
                "Windows agree chunk-for-chunk; the layers hold between {min} and {max} \
                 chunks, so the difference is in trailing structure alone."
            ));
        };
        // **Group the layers by what they hold there.** Which layers differ is
        // the whole diagnosis: one layer out of forty-eight means something
        // special-cases that index, a contiguous prefix means a per-layer loop
        // stopped early, and a scatter means an operation applied per layer with
        // its own predicate. "Layer 0 differs from layer 1" distinguishes none
        // of those, which is why the first report of this could not be placed.
        let mut groups: Vec<((u16, u32), Vec<usize>)> = Vec::new();
        for (li, w) in all.iter().enumerate() {
            let Some(w) = w else { continue };
            let Some(&v) = w.get(i) else { continue };
            match groups.iter_mut().find(|(k, _)| *k == v) {
                Some((_, ls)) => ls.push(li),
                None => groups.push((v, vec![li])),
            }
        }
        let render = |ls: &[usize]| -> String {
            if ls.len() > 6 {
                format!("{} layers ({}…{})", ls.len(), ls[0], ls[ls.len() - 1])
            } else {
                format!(
                    "layer{} {}",
                    if ls.len() == 1 { "" } else { "s" },
                    ls.iter()
                        .map(|l| l.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        };
        let split = groups
            .iter()
            .map(|(v, ls)| format!("(offset {}, usage {}) on {}", v.0, v.1, render(ls)))
            .collect::<Vec<_>>()
            .join("; ");
        Some(format!(
            "First difference at chunk {i} of {}: {split}.",
            base.len(),
        ))
    }

    /// Repair a divergence confined to the tail: truncate every layer to the
    /// shortest layer's token count. Answers whether the layers agree afterwards.
    ///
    /// **A failed wave's signature, healed from its own properties.** The layer
    /// sweep advances usage per layer, so a wave that dies mid-sweep leaves the
    /// early layers up to one attention window ahead of the rest — tokens that
    /// were never delivered, since the wave that wrote them never retired. The
    /// rollback in `forward_wave` undoes that at the failure; this covers the
    /// state that predates the rollback or arrived through the substrate, where
    /// a sealed turn can persist the skew and reload it into a fresh sequence.
    ///
    /// Cutting to the shortest is sound precisely because the surplus is
    /// undelivered: no sampled token, no turn, no summary refers to it. Two
    /// guards keep this from ever being a destructive repair:
    ///
    /// - **Spread cap.** The layers may differ by at most `CHUNK_SIZE` tokens —
    ///   the most one failed operation leaves behind. A wider spread is not a
    ///   failed wave and gets no repair.
    /// - **The sealed prefix is untouchable.** `truncate_sequence_to_tokens`
    ///   clamps at the sealed boundary rather than cut Arc-shared ground, so a
    ///   "tail" divergence that is actually deep history leaves those layers
    ///   unchanged — and the verification below then reads the layers as still
    ///   disagreeing and reports the heal as failed, which it is.
    /// Whether any two layers hold `batch_idx`'s **tokens** differently —
    /// per-chunk `(offset, usage)` with trailing empty chunks ignored, so
    /// trailing structure (which the writer-pad repair owns) does not count.
    fn data_windows_disagree(backings: &[ChunkedKvBacking], batch_idx: usize) -> bool {
        let data_windows = |b: &ChunkedKvBacking| -> Option<Vec<(u16, u32)>> {
            let state = b.state.read().ok()?;
            let slot = state.sequences.get(batch_idx)?.as_ref()?;
            let mut w: Vec<(u16, u32)> = slot
                .chunks_slice()
                .iter()
                .map(|c| (c.offset, c.usage))
                .collect();
            while w.last().is_some_and(|&(_, u)| u == 0) {
                w.pop();
            }
            Some(w)
        };
        let mut layers = backings.iter().filter_map(data_windows);
        match layers.next() {
            Some(base) => !layers.all(|w| w == base),
            None => false,
        }
    }

    fn heal_tail_divergence(
        backings: &[ChunkedKvBacking],
        batch_idx: usize,
        target_tokens: usize,
    ) -> Result<bool> {
        let totals: Vec<usize> = backings
            .iter()
            .filter_map(|b| {
                let state = b.state.read().ok()?;
                let slot = state.sequences.get(batch_idx)?.as_ref()?;
                Some(slot.chunks_slice().iter().map(|c| c.usage as usize).sum())
            })
            .collect();
        let (Some(&min), Some(&max)) = (totals.iter().min(), totals.iter().max()) else {
            return Ok(false);
        };
        // The session's own offset — how many tokens have actually been
        // delivered — is the anchor, not the shortest layer. Truncating to the
        // min looked equivalent until a failed wave left *every* layer with the
        // surplus token but one of them one further along: min was then
        // offset+1, and "heal to the shortest" preserved a token the caller
        // never received. Three honesty checks before touching anything:
        if min < target_tokens {
            // A layer holds fewer tokens than the session has delivered.
            // Truncation cannot restore missing history.
            return Ok(false);
        }
        if max == target_tokens {
            // Every layer already sits at the delivered count, yet the windows
            // disagree — that is mid-history corruption, not a tail surplus.
            return Ok(false);
        }
        if max - target_tokens > CHUNK_SIZE {
            // A surplus past one chunk is more than any single failed wave
            // leaves behind.
            return Ok(false);
        }
        for b in backings {
            b.truncate_sequence_to_tokens(batch_idx, target_tokens)?;
        }
        // Verified, not assumed — and on the *data* windows, with trailing
        // empty chunks ignored. The truncation can leave layers with different
        // counts of empty tail chunks, and that is trailing structure the
        // caller's writer-pad repair equalises; what the heal has to establish
        // is that every token the layers still hold is held identically.
        let healed = !Self::data_windows_disagree(backings, batch_idx);
        if healed {
            tracing::warn!(
                batch_idx,
                dropped = max - target_tokens,
                total = target_tokens,
                "chunked decode: healed per-layer tail divergence by truncating every \
                 layer to the session's delivered offset — the dropped tokens were \
                 written by a wave that failed before delivering them"
            );
        }
        Ok(healed)
    }

    fn unify_decode_layout(
        backings: &[ChunkedKvBacking],
        entries: &[(usize, usize)],
        layouts: &[Vec<Option<DecodeLayout>>],
    ) -> Result<()> {
        for (ei, &(batch_idx, offset)) in entries.iter().enumerate() {
            let mut seen = layouts.iter().filter_map(|l| l[ei]);
            let Some(first) = seen.next() else {
                continue;
            };
            if seen.all(|l| l == first) {
                continue;
            }

            // Two repairs compose here, and **the heal runs first**. The
            // structure repair's `push_empty_writer_chunk` advances
            // `writer_start_idx` past every existing chunk — sealing them — and
            // a failed wave's surplus token lives in exactly the chunk that
            // would seal. Repairing structure first therefore puts the surplus
            // beyond the heal's reach (truncation clamps at the sealed
            // boundary), which is how the first ordering of this loop turned a
            // healable one-token skew into a refusal. So: while the layers hold
            // *tokens* differently, heal; once only trailing structure differs,
            // pad and push a common writer.
            let mut tail_healed = false;
            loop {
                if Self::data_windows_disagree(backings, batch_idx) {
                    if !tail_healed && Self::heal_tail_divergence(backings, batch_idx, offset)? {
                        tail_healed = true;
                        continue;
                    }
                } else {
                    // Tokens agree; equalise trailing structure. Re-read rather
                    // than trust: this round starts from the layers as they
                    // are, not from the caller's pre-repair probe.
                    let fresh: Vec<Option<DecodeLayout>> = backings
                        .iter()
                        .map(|b| b.probe_decode_entries(entries, 0).map(|(_, l)| l[ei]))
                        .collect::<Result<_>>()?;
                    let max_blocks = fresh.iter().flatten().map(|l| l.blocks).max().unwrap_or(0);
                    for (li, backing) in backings.iter().enumerate() {
                        let Some(layout) = fresh[li] else {
                            continue;
                        };
                        for _ in layout.blocks..=max_blocks {
                            backing.push_empty_writer_chunk(batch_idx)?;
                        }
                    }

                    let after: Vec<Option<DecodeLayout>> = backings
                        .iter()
                        .map(|b| b.probe_decode_entries(entries, 0).map(|(_, l)| l[ei]))
                        .collect::<Result<_>>()?;
                    let mut it = after.iter().flatten();
                    let Some(&expected) = it.next() else { break };
                    if it.all(|l| *l == expected) {
                        break;
                    }
                }

                // **Name the chunk, not just the digest.** The digest proves
                // a difference and says nothing about where, and "somewhere
                // in fifty-eight chunks" is not a lead — the first time this
                // fired the cause (a per-layer fork refused part-way) had to
                // be inferred from what the user happened to be doing. The
                // windows are already in hand; reporting the first index
                // that differs, with both sides' values, costs a walk of a
                // list this code has just read.
                let detail = Self::first_window_divergence(backings, batch_idx)
                    .unwrap_or_else(|| "no per-chunk difference found on re-read".to_string());
                candle::bail!(
                    "chunked decode layout diverged across layers for batch_idx {batch_idx} \
                     and could not be reconciled or healed: {detail} Appending a common \
                     writer chunk equalises trailing structure, and a tail difference of up \
                     to one chunk is healed by truncating every layer to the shortest — a \
                     difference that survives both is either deep in the sealed history or \
                     wider than any single failed operation leaves, which no repair from \
                     here can be trusted with."
                )
            }

            // `warn`, not `debug`: layers disagreeing on block structure is an
            // anomaly the engine repaired, not routine bookkeeping, and it is
            // rare by construction — the repair leaves every layer identical,
            // so a second report means something re-created the skew. If this
            // ever becomes frequent enough to be noise, the frequency is the
            // finding.
            tracing::warn!(
                batch_idx,
                tail_healed,
                "chunked decode: layer block structure diverged; reconciled onto a common \
                 writer chunk"
            );
        }
        Ok(())
    }

    pub fn ensure_for_batch_entries(&self, entries: &[(usize, usize)], add: usize) -> Result<()> {
        if entries.is_empty() || add == 0 {
            return Ok(());
        }

        let batch = self.batch_capacity();
        for &(batch_idx, _off) in entries.iter() {
            if batch_idx >= batch {
                candle::bail!(
                    "batch_idx {} out of range for chunked backing (capacity {})",
                    batch_idx,
                    batch
                )
            }
        }

        let chunk_size = CHUNK_SIZE;

        // Under cum_token addressing the slot's `state.offset` is the
        // sum of slice.usage — NOT chunk_count × CHUNK_SIZE.  We can't
        // use positional math (`(offset + add) / CHUNK_SIZE`) to count
        // needed chunks because partial-tail slices (from injected
        // prefix sections) make positional and cum_token indices
        // diverge.  Instead, compute how many additional chunks each
        // slot needs based on the actual remaining capacity in
        // existing chunks starting at the first empty (or last
        // partial) slice — the same `write_slice` rule used in
        // `slot_state.rs`.
        let mut alloc_plan: Vec<(usize, usize)> = Vec::with_capacity(entries.len());
        let mut tail_maybe: Vec<usize> = Vec::new();
        let mut required_max_blocks = 1usize;
        {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for &(batch_idx, _off) in entries.iter() {
                let (current_chunks, available) =
                    match state.sequences.get(batch_idx).and_then(|s| s.as_ref()) {
                        Some(slot) => {
                            let chunks = slot.chunks_slice();
                            // Writer-owned capacity ONLY: chunks before
                            // `writer_start_idx` are Arc-shared/sealed, and a
                            // partial sealed tail is a GAP — its free slots
                            // are dead, never a write target. When the
                            // boundary sits past the last chunk (a freshly
                            // injected prefix), available is ZERO; clamping
                            // into the sealed tail counts its gap as writer
                            // capacity and under-allocates the write region
                            // by up to one chunk.
                            let start = slot.writer_start_idx();
                            let avail: usize = if start >= chunks.len() {
                                0
                            } else {
                                chunks[start..]
                                    .iter()
                                    .map(|c| chunk_size - (c.offset as usize + c.usage as usize))
                                    .sum()
                            };
                            (chunks.len(), avail)
                        }
                        None => (0usize, 0usize),
                    };
                let needed_extra = add.saturating_sub(available);
                let additional_chunks = (needed_extra + chunk_size - 1) / chunk_size;
                let new_total_chunks = current_chunks + additional_chunks;
                required_max_blocks = cmp::max(required_max_blocks, new_total_chunks.max(1));
                alloc_plan.push((batch_idx, additional_chunks));
                // Predict the tail need in the SAME read pass — this runs once per
                // layer per decode step, so a second guard acquisition here is
                // pure overhead on the hot path.
                if self.tail_needs_new_block(&state, batch_idx) {
                    tail_maybe.push(batch_idx);
                }
            }
        }
        self.ensure_max_blocks(required_max_blocks)?;

        // Allocate BEFORE taking the state guard.
        //
        // `alloc_block_chunks` can reach `request_global_compact`, and compaction
        // needs a write guard on EVERY layer's block table to remap relocated
        // GIDs. Allocating while holding one of those guards made that a
        // self-deadlock; once the compactor was made non-blocking it became a
        // permanent no-op instead, so arena compaction could never run from the
        // path that needs it most. The chunk counts are already known from
        // `alloc_plan`, so the allocation needs no state access at all.
        let mut prealloc: Vec<(usize, Vec<_>)> = Vec::with_capacity(alloc_plan.len());
        for (batch_idx, additional_chunks) in alloc_plan {
            let mut cws = Vec::with_capacity(additional_chunks);
            for _ in 0..additional_chunks {
                cws.push(self.alloc_block_chunks(0, 0)?);
            }
            prealloc.push((batch_idx, cws));
        }

        // Predict which tails will need a fresh block, so their chunks can be
        // allocated alongside the rest and ALL mutation can happen under ONE
        // guard. The prediction is a safe over-estimate: installing a block makes
        // that sequence's tail fresh and therefore writable, so the block installs
        // below can only ever *reduce* this set, never grow it. Spares that turn
        // out to be unnecessary simply drop, returning their GIDs to the pool.
        let mut tail_spares: Vec<(usize, _)> = Vec::with_capacity(tail_maybe.len());
        for batch_idx in tail_maybe {
            tail_spares.push((batch_idx, self.alloc_block_chunks(0, 0)?));
        }

        // FAST PATH. This runs 48x per decode step (once per layer), and on a
        // normal step the block is not full and the tail is still writable, so
        // there is nothing to install. Returning before the write guard keeps the
        // steady-state decode cost at one read guard, as it was before the
        // allocation was hoisted out of the mutation guard.
        if prealloc.is_empty() && tail_spares.is_empty() {
            return Ok(());
        }

        // Single guard for every mutation. Extending a sequence and making its
        // tail writable must be ATOMIC: a reader that observes blocks pushed but
        // the tail not yet replaced would write into a full or closed-quant chunk.
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            for (batch_idx, cws) in prealloc {
                // Auto-allocate slot if needed (mirrors ensure_for_offset behavior).
                if state.sequences[batch_idx].is_none() {
                    state.sequences[batch_idx] = Some(self.make_sequence_state()?);
                }
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                for cw in cws {
                    slot.push_chunk(cw);
                    slot.invalidate_gpu_chunks();
                }
            }

            // Writable-tail pass, still under the same guard.
            for (batch_idx, cw) in tail_spares {
                if self.tail_needs_new_block(&state, batch_idx) {
                    let slot = state.sequences[batch_idx].as_mut().unwrap();
                    slot.push_chunk(cw);
                    slot.invalidate_gpu_chunks();
                }
            }
        }

        Ok(())
    }

    /// Whether `batch_idx`'s tail block can still be written into, or a fresh
    /// block must be pushed. Pure read over state + arena storage; extracted so
    /// the decision can be made under a guard while the allocation it implies
    /// happens outside one.
    fn tail_needs_new_block(
        &self,
        state: &super::types::BlockTableState,
        batch_idx: usize,
    ) -> bool {
        let needs: Option<bool> = state.sequences[batch_idx].as_ref().and_then(|s| {
            let cw = s.last_chunk()?;
            debug_assert!(
                cw.gids.iter().all(|g| g.strong_count() <= cw.gids.len()),
                "tail block must not be shared — fork should have copied it"
            );
            let is_full = (cw.offset as usize + cw.usage as usize) >= CHUNK_SIZE;
            if is_full {
                Some(true)
            } else {
                // A block-quantized band cannot take a partial-token append —
                // writes have to land on whole blocks — so a tail that has
                // already been compressed needs a fresh block behind it.
                //
                // The question is asked of the chunk's own band tags, not of
                // the arenas its gids point into: a size-class arena holds
                // whatever fits its stride, so "is this arena quantized" has no
                // answer (`docs/archived/arena_unification.md` principle 8). R16 is
                // excluded because it is the raw active-K capture format, not
                // compression — the writer chunk's own K side is R16 and must
                // stay writable.
                let quantized_band = |tags: &[u8]| {
                    tags.iter()
                        .copied()
                        .map(ArenaFormatTag::from_u8)
                        .any(|t| t.is_quantized() && t != ArenaFormatTag::R16)
                };
                if quantized_band(&cw.k_fmt) || quantized_band(&cw.v_fmt) {
                    Some(true)
                } else {
                    None
                }
            }
        });
        matches!(needs, Some(true))
    }

    pub fn ensure_for_offset(&self, batch_idx: usize, offset: usize, add: usize) -> Result<()> {
        let batch = self.batch_capacity();
        if batch_idx >= batch {
            candle::bail!(
                "batch_idx {} out of range for chunked backing (capacity {})",
                batch_idx,
                batch
            )
        }
        if add == 0 {
            return Ok(());
        }

        let end_pos = offset.saturating_add(add).saturating_sub(1);
        let need_blocks = (end_pos / CHUNK_SIZE) + 1;
        self.ensure_max_blocks(need_blocks)?;

        // Count first, allocate WITHOUT the guard, then install. `alloc_block_chunks`
        // can reach `request_global_compact`, which needs a write guard on every
        // layer's block table; allocating under one made that a self-deadlock.
        // See `ensure_for_batch_entries` for the full rationale.
        // Count blocks AND predict the tail need in one read pass, allocate both
        // without a guard, then mutate under ONE write guard. Extending a sequence
        // and making its tail writable must be atomic: a reader that saw blocks
        // pushed but the tail not yet replaced would write into a full or
        // closed-quant chunk. Predicting the tail before the installs is a safe
        // over-estimate — a freshly pushed block is itself writable, so installs
        // can only shrink the need. An unused spare drops, returning its GIDs.
        let (missing, maybe_tail) = {
            let state = self
                .state
                .read()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            let have = state.sequences[batch_idx]
                .as_ref()
                .map(|s| {
                    (0..need_blocks)
                        .filter(|&b| s.chunk_at(b).is_some())
                        .count()
                })
                .unwrap_or(0);
            (
                need_blocks.saturating_sub(have),
                self.tail_needs_new_block(&state, batch_idx),
            )
        };
        let mut fresh = Vec::with_capacity(missing);
        for _ in 0..missing {
            fresh.push(self.alloc_block_chunks(0, 0)?);
        }
        let tail_spare = if maybe_tail {
            Some(self.alloc_block_chunks(0, 0)?)
        } else {
            None
        };

        // Fast path — see `ensure_for_batch_entries`.
        if fresh.is_empty() && tail_spare.is_none() {
            return Ok(());
        }

        {
            let mut state = self
                .state
                .write()
                .map_err(|_| candle::Error::Msg("chunked state lock poisoned".into()))?;
            // Auto-allocate slot if needed
            if state.sequences[batch_idx].is_none() {
                state.sequences[batch_idx] = Some(self.make_sequence_state()?);
            }
            {
                let slot = state.sequences[batch_idx].as_mut().unwrap();
                // Under cum_token addressing we never bump the previous tail's
                // usage when allocating a new chunk. See `ensure_for_batch_entries`.
                for cw in fresh {
                    if (0..need_blocks).any(|b| slot.chunk_at(b).is_none()) {
                        slot.push_chunk(cw);
                    }
                }
            }
            // Writable-tail pass, same guard.
            if let Some(cw) = tail_spare {
                if self.tail_needs_new_block(&state, batch_idx) {
                    let slot = state.sequences[batch_idx].as_mut().unwrap();
                    slot.push_chunk(cw);
                }
            }
        }

        Ok(())
    }
}
