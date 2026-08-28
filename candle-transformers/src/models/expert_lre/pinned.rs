//! The warm tier: a static, stratified, immutable mirror of part of the pack.
//!
//! ```text
//! cold    the repacked pack file        authoritative, holds every expert
//! warm    pinned host RAM (this file)   static, immutable, a stratified subset
//! hot     VRAM                          dynamic cache, eviction = drop
//! ```
//!
//! The pool is filled **once**, at startup, and never changes: no eviction
//! policy, no ranking, no background filler, no free list. That is not a
//! simplification of a richer design — it is what licenses the rule that makes
//! the rest of the cache simple. Because no other expert can ever claim a slot,
//! a promoted expert's slot is not worth reclaiming, so its host copy is never
//! surrendered and an eviction from VRAM lands it back on RAM by doing nothing.
//! A warm tier with an eviction policy could not make that promise.
//!
//! # Why a random subset, and why stratified
//!
//! A uniform random subset of size *X*% yields *X*% hit rate on the accesses
//! that reach it, whatever the popularity distribution. The usual objection —
//! pick the hottest, not the random — does not apply, because **VRAM has
//! already done the popularity filtering**: what reaches RAM is the miss stream,
//! which is the long tail, and the tail is far flatter than the head.
//!
//! The sample is taken *per layer* rather than globally. A global draw gives
//! per-layer residency `~ Binomial(experts_per_layer, X)`; at 128 experts and
//! *X* = 40% that is mean 51.2, sd 5.5, so across 48 layers the unluckiest lands
//! near 37 — 29% residency against a 40% average. That matters more here than it
//! would for an ordinary cache, for two reasons: the fill is immutable, so an
//! unlucky layer is slow on *every* forward for the process lifetime rather than
//! for one; and the wave is sequential, so lucky layers cannot compensate —
//! layer *N+1* cannot start until layer *N*'s experts land, and the sweep runs
//! at its worst stage rather than its mean. Stratifying drives that variance to
//! zero for the same fill, the same immutability and one different line in the
//! sampler.

#[cfg(feature = "cuda")]
use candle::Result;

/// Per-layer geometry: shapes, dtypes, and repacked byte sizes.
///
/// Within a MoE layer, all experts share the same geometry (but different
/// layers may use different dtypes, e.g. Q4_K_M uses Q6_K for first/last).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub(crate) struct LayerGeometry {
    pub gate_shape: Vec<usize>,
    pub gate_dtype: candle::quantized::GgmlDType,
    pub gate_repacked_size: usize,

    pub up_shape: Vec<usize>,
    pub up_dtype: candle::quantized::GgmlDType,
    pub up_repacked_size: usize,

    pub down_shape: Vec<usize>,
    pub down_dtype: candle::quantized::GgmlDType,
    pub down_repacked_size: usize,

    /// Total repacked bytes for one expert in this layer.
    pub total_repacked_size: usize,
}

/// Per-layer geometry for every MoE layer, from the GGUF references.
///
/// The pack and the warm pool cache the *target* format: for `Off` that is the
/// gemx K/128 repack of the source dtype; for int8 it is the KO twin (repacked
/// once per expert, then reloaded by copy on a miss — no per-miss re-quant).
///
/// A free function rather than a step inside `ExpertCache::new` because the
/// model loader needs the answer **before** the cache exists: the weight zone's
/// slot size comes from these repacked byte counts, and the zone has to be sized
/// and its floor installed before a single expert is uploaded into it.
#[cfg(feature = "cuda")]
pub(crate) fn layer_geometries(
    host_refs: &[Vec<super::types::MmapExpertRef>],
    int8mode: candle::quantized::Int8Mode,
) -> Result<Vec<LayerGeometry>> {
    let mut geoms: Vec<LayerGeometry> = Vec::with_capacity(host_refs.len());
    for layer in host_refs {
        let Some(r) = layer.first() else { continue };
        let tko = |d: candle::quantized::GgmlDType| {
            if int8mode.is_int8() {
                d.to_ko(int8mode)
            } else {
                Ok(d)
            }
        };
        let gate_dtype = tko(r.gate_dtype)?;
        let up_dtype = tko(r.up_dtype)?;
        let down_dtype = tko(r.down_dtype)?;
        let gate_repacked_size =
            candle::quantized::repacked_size_bytes(r.gate_shape[0], r.gate_shape[1], gate_dtype)?;
        let up_repacked_size =
            candle::quantized::repacked_size_bytes(r.up_shape[0], r.up_shape[1], up_dtype)?;
        let down_repacked_size =
            candle::quantized::repacked_size_bytes(r.down_shape[0], r.down_shape[1], down_dtype)?;
        geoms.push(LayerGeometry {
            gate_shape: r.gate_shape.clone(),
            gate_dtype,
            gate_repacked_size,
            up_shape: r.up_shape.clone(),
            up_dtype,
            up_repacked_size,
            down_shape: r.down_shape.clone(),
            down_dtype,
            down_repacked_size,
            total_repacked_size: gate_repacked_size + up_repacked_size + down_repacked_size,
        });
    }
    Ok(geoms)
}

/// Where an expert's bytes are **in addition to** the pack file, which holds
/// every expert always.
///
/// The two fields are independent facts over an always-present base, not a
/// choice between three places. An expert can be in VRAM and RAM at once, and
/// usually is: promoting a warm expert *copies* its bytes to the device rather
/// than moving them, because the warm tier is immutable and would not reuse the
/// vacated slot anyway.
///
/// # Why a product and not a sum
///
/// The sum — `Vram { slot, ram_backed } | Ram { slot } | Disk` — carries exactly
/// the same four states and exactly the same semantics. The product is preferred
/// because `ram` is an **immutable per-expert fact** decided at startup, and the
/// sum makes every VRAM transition rewrite it. Five sites construct a VRAM
/// residency (startup fill, demand miss, prefetch, hint, relocation); under the
/// sum each must restate `ram_backed` correctly, and the compiler would force
/// them to supply *a* value but not the *right* one. A relocation between VRAM
/// slots is precisely where a hand-written `ram_backed: None` would silently
/// orphan a live pinned slot for the process lifetime. Here those sites assign
/// `vram` and `ram` is not in the expression, so it cannot be dropped.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ExpertResidency {
    /// Slot in the weight zone, while a device copy exists.
    pub vram: Option<usize>,
    /// Slot in the warm pool. Written at fill and never cleared.
    pub ram: Option<usize>,
}

/// Which `(layer, expert)` each warm slot holds, chosen once at startup.
///
/// `slots[i]` is the expert in warm slot `i`.
///
/// # The experts VRAM will not hold come first
///
/// The startup fill takes the first `vram_prefix` experts in flat
/// `(layer, expert)` order into VRAM, so those are hits from the first forward
/// and a warm slot spent on one buys nothing until it is evicted. The experts
/// *past* that prefix are the miss stream, and covering them is the whole
/// purpose of the tier.
///
/// So the draw runs in two passes over disjoint pools: everything outside the
/// VRAM prefix first, then — only if slots remain — the rest, as insurance
/// against eviction. A tier that can hold the complement of VRAM covers **every
/// miss** at startup, which is 3,767 of 6,144 experts on Qwen3-30B-A3B rather
/// than all 6,144. That is the property the two-tier cache had for free, by
/// construction, and the first build of this lost by drawing uniformly: it put
/// 36 % of its slots on experts VRAM was already holding.
///
/// # Both passes are stratified per layer
///
/// A global draw gives per-layer residency `~ Binomial`, so across 48 layers the
/// unluckiest lands far below the mean. That matters more here than for an
/// ordinary cache: the fill is immutable, so an unlucky layer is slow on *every*
/// forward for the process lifetime; and the wave is sequential, so lucky layers
/// cannot compensate — layer *N+1* cannot start until layer *N*'s experts land.
/// Each pass therefore hands slots out one per layer at a time, so no layer is
/// ever more than one ahead of another that still has candidates.
///
/// The choice within a layer is a deterministic draw from `seed`: reproducible
/// across runs of the same build on the same model, which is what makes a
/// regression in warm-tier hit rate attributable to a change rather than to
/// luck.
pub(crate) fn stratified_membership(
    num_layers: usize,
    experts_per_layer: usize,
    warm_slots: usize,
    vram_prefix: usize,
    pinned_layers: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    if num_layers == 0 || experts_per_layer == 0 || warm_slots == 0 {
        return Vec::new();
    }
    // Layers `0..pinned_layers` are permanently VRAM-resident and never evicted,
    // so a warm copy of one could only ever be read by a load that cannot
    // happen. They are excluded from the draw entirely rather than sorted to the
    // back of it: on the 3.6-35B that is 512 experts, 943 MiB of pinned host
    // memory that would otherwise sit unread for the life of the process while
    // the evictable set — the only set that generates misses — went short.
    let drawable = num_layers.saturating_sub(pinned_layers);
    if drawable == 0 {
        return Vec::new();
    }
    let total = drawable * experts_per_layer;
    let mut remaining = warm_slots.min(total);

    // Per layer, how many of its experts fall inside the VRAM prefix. The
    // prefix is a flat run, so a layer is fully inside it, fully outside it, or
    // the single layer it ends in.
    let in_vram = |layer: usize| -> usize {
        let start = layer * experts_per_layer;
        vram_prefix.saturating_sub(start).min(experts_per_layer)
    };

    let mut out = Vec::with_capacity(remaining);
    let mut rng = SplitMix64::new(seed);
    // Per drawable layer, the experts not yet drawn, partitioned so that the
    // ones outside the VRAM prefix are taken first. Index `i` here is MoE layer
    // `pinned_layers + i`.
    let mut decks: Vec<(Vec<usize>, Vec<usize>)> = (pinned_layers..num_layers)
        .map(|layer| {
            let covered = in_vram(layer);
            (
                (covered..experts_per_layer).collect(),
                (0..covered).collect(),
            )
        })
        .collect();

    // Pass 1 over the complement of VRAM, pass 2 over the rest. Within a pass,
    // one slot per layer per round, so the shares stay level.
    for pass in 0..2 {
        loop {
            let mut handed_out = 0usize;
            for (deck_idx, pair) in decks.iter_mut().enumerate() {
                let layer = pinned_layers + deck_idx;
                if remaining == 0 {
                    break;
                }
                let deck = if pass == 0 { &mut pair.0 } else { &mut pair.1 };
                if deck.is_empty() {
                    continue;
                }
                // Draw uniformly from what is left by swapping the pick to the
                // end and popping it.
                let j = rng.below(deck.len());
                let last = deck.len() - 1;
                deck.swap(j, last);
                let expert = deck.pop().expect("non-empty");
                out.push((layer, expert));
                remaining -= 1;
                handed_out += 1;
            }
            if handed_out == 0 || remaining == 0 {
                break;
            }
        }
        if remaining == 0 {
            break;
        }
    }
    out
}

/// SplitMix64 — the fixed-increment generator, used here for a reproducible
/// draw and nothing else.
///
/// Written out rather than pulled from `rand` because the draw must be stable
/// across builds: a dependency bump that changed the algorithm would silently
/// change which experts are warm, and the fill it produces is compared between
/// runs.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `[0, n)`, via the multiply-shift reduction — unbiased enough
    /// for a fill and free of the modulo's rejection loop.
    fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        ((self.next() as u128 * n as u128) >> 64) as usize
    }
}

/// Pinned host memory — physically locked RAM the GPU DMA engine can read at
/// full PCIe bandwidth without an OS page fault or a bounce copy.
///
/// Allocated once via `cuMemAllocHost` and divided into fixed-size slots, each
/// holding one expert's record **at the pack's stride**, so a cold read can land
/// directly in a slot with nothing in between.
///
/// # Thread safety
///
/// Owned exclusively by the pipeline thread (`&mut self` access). No interior
/// mutability, no atomics.
#[cfg(feature = "cuda")]
pub(crate) struct WarmPool {
    /// Base pointer from `cuMemAllocHost`.
    base: *mut u8,
    /// Total allocation size in bytes.
    total_size: usize,
    /// Per-slot byte size — the pack's record stride, so it is a multiple of a
    /// direct-I/O sector and every slot base is sector-aligned.
    slot_size: usize,
    /// Number of slots.
    num_slots: usize,
}

#[cfg(feature = "cuda")]
impl WarmPool {
    /// Allocate as much of `want_slots` as the machine will give.
    ///
    /// The warm tier is a **performance** choice, not a correctness one — the
    /// cold tier holds every expert at any warm size, including zero — so it is
    /// sized by what the machine can spare and `cuMemAllocHost` refusing is the
    /// answer to "was that too much". A refusal halves the request and tries
    /// again rather than failing the load: pinned pages are non-pageable, so a
    /// refusal is a real answer about the machine and not a hint, but it is an
    /// answer about *this size*, and the next size down is still worth having.
    pub(crate) fn new(want_slots: usize, slot_size: usize) -> Self {
        let mut slots = want_slots;
        while slots > 0 && slot_size > 0 {
            match Self::try_alloc(slots, slot_size) {
                Some(pool) => return pool,
                None => {
                    tracing::warn!(
                        target: "candle_transformers::expert_lre",
                        slots,
                        gib = slots.saturating_mul(slot_size) as f64 / 1e9,
                        "warm tier: cuMemAllocHost refused; halving"
                    );
                    slots /= 2;
                }
            }
        }
        Self::empty()
    }

    fn try_alloc(num_slots: usize, slot_size: usize) -> Option<Self> {
        // A wrapped product would ask `cuMemAllocHost` for a small buffer and
        // then hand out `num_slots` slots over it, which is the same
        // out-of-bounds write the accessors above refuse — reached through
        // arithmetic instead of through an index. `None` is the honest answer
        // and puts it on the halving path in [`Self::new`], which converges on
        // a size that does not wrap.
        let total_size = num_slots.checked_mul(slot_size)?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, total_size) };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return None;
        }
        // Feed the process-wide gauge: pinned memory is non-pageable, so the
        // host-RAM availability measurement must treat it as structural.
        candle::vram::note_host_pinned_alloc(
            candle::vram::PinnedUse::ExpertWarmTier,
            total_size as u64,
        );
        // The deepest point of the run for free host RAM: the tier is the
        // largest pinned claim the engine makes and everything it needs after
        // this has to fit in what is left. Sampling here is what makes the
        // headroom tunable against a measurement instead of a guess.
        candle::vram::sample_available_low_water();
        tracing::info!(
            target: "candle_transformers::expert_lre",
            slots = num_slots,
            gib = total_size as f64 / 1e9,
            slot_kib = slot_size as f64 / 1024.0,
            "warm tier: pinned host pool allocated"
        );
        Some(Self {
            base: ptr as *mut u8,
            total_size,
            slot_size,
            num_slots,
        })
    }

    /// A mutable byte slice for a slot — the destination of the one fill it
    /// ever receives.
    ///
    /// # The bounds hold in release
    ///
    /// This hands out a raw slice over `cuMemAllocHost` memory, so an index or
    /// length past the pool writes into whatever the process has after it —
    /// silently, with no fault to catch it, because pinned host pages are as
    /// writable as any other part of the address space. Every other bound in
    /// this cache is enforced rather than assumed (`claim_dense` against the
    /// weight floor, `copy_to_host_on_stream` against `dst.len()`), and a
    /// `debug_assert` here would be exactly the one that is absent from the
    /// builds that run: the gates and the daemon are `--release`.
    ///
    /// A violation is a bookkeeping bug — `slot_idx` comes from the residency
    /// map and `len` is the pack's record stride, which is also what
    /// `slot_size` was cut to — so it panics rather than returning a `Result`.
    /// There is no recovery to offer a caller whose own indices are wrong, and
    /// a loud abort at the offending call is worth far more than a corrupted
    /// heap discovered later somewhere else.
    #[inline]
    pub(crate) fn slot_mut(&mut self, slot_idx: usize, len: usize) -> &mut [u8] {
        assert!(
            slot_idx < self.num_slots,
            "warm tier: slot {slot_idx} is past the pool's {} slots",
            self.num_slots,
        );
        assert!(
            len <= self.slot_size,
            "warm tier: a {len} B write does not fit slot {slot_idx}'s {} B",
            self.slot_size,
        );
        unsafe {
            let ptr = self.base.add(slot_idx * self.slot_size);
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// A mutable slice spanning `n_slots` consecutive slots from `first`.
    ///
    /// The bulk fill's destination: the pool is one contiguous allocation, so a
    /// run of slots is a run of bytes, and the whole warm tier can be read in
    /// one batch of positioned reads rather than one call per expert.
    /// [`Self::slot_mut`] would be a lie here — it promises a single slot's
    /// worth and asserts it.
    ///
    /// Bounds-checked in release for the reason given on [`Self::slot_mut`],
    /// and the sum is checked too: `first + n_slots` is the one arithmetic in
    /// this file that can wrap on its way to being compared.
    #[inline]
    pub(crate) fn span_mut(&mut self, first: usize, n_slots: usize) -> &mut [u8] {
        let end = first
            .checked_add(n_slots)
            .expect("warm tier: span end overflows usize");
        assert!(
            end <= self.num_slots,
            "warm tier: span [{first}, {end}) is past the pool's {} slots",
            self.num_slots,
        );
        unsafe {
            let ptr = self.base.add(first * self.slot_size);
            std::slice::from_raw_parts_mut(ptr, n_slots * self.slot_size)
        }
    }

    /// A shared byte slice for a slot — the source of every promotion.
    ///
    /// Bounds-checked in release for the reason given on [`Self::slot_mut`].
    /// A read past the pool is the milder half of the same bug — it feeds an
    /// H2D copy, so it turns unrelated process memory into expert weights
    /// rather than the other way round.
    #[inline]
    pub(crate) fn slot_ref(&self, slot_idx: usize, len: usize) -> &[u8] {
        assert!(
            slot_idx < self.num_slots,
            "warm tier: slot {slot_idx} is past the pool's {} slots",
            self.num_slots,
        );
        assert!(
            len <= self.slot_size,
            "warm tier: a {len} B read exceeds slot {slot_idx}'s {} B",
            self.slot_size,
        );
        unsafe {
            let ptr = self.base.add(slot_idx * self.slot_size);
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Total number of slots.
    #[inline]
    pub(crate) fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Bytes held, for the memory report.
    #[inline]
    pub(crate) fn total_bytes(&self) -> usize {
        self.total_size
    }

    /// An empty pool — no warm tier at all, which the invariant permits.
    pub(crate) fn empty() -> Self {
        Self {
            base: std::ptr::null_mut(),
            total_size: 0,
            slot_size: 0,
            num_slots: 0,
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for WarmPool {
    fn drop(&mut self) {
        if !self.base.is_null() {
            let result =
                unsafe { cudarc::driver::sys::cuMemFreeHost(self.base as *mut std::ffi::c_void) };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                tracing::warn!("WarmPool: cuMemFreeHost failed: {:?}", result);
            }
            candle::vram::note_host_pinned_free(
                candle::vram::PinnedUse::ExpertWarmTier,
                self.total_size as u64,
            );
        }
    }
}

// SAFETY: The pinned memory is allocated via cuMemAllocHost which returns
// a host-accessible pointer valid for any thread.  The pool is exclusively
// owned by the pipeline thread (no shared access).
#[cfg(feature = "cuda")]
unsafe impl Send for WarmPool {}

// SAFETY: after the startup fill the pool is immutable by design (this file's
// module doc: "static, immutable, a stratified subset") — the mutating
// accessors (`slot_mut`, `span_mut`) are only reachable through `&mut self`,
// and the pool is shared read-only via `Arc` (which makes `&mut` unreachable)
// between the pipeline thread and the expert streamer. Concurrent `slot_ref`
// reads of stable pinned memory are race-free.
#[cfg(feature = "cuda")]
unsafe impl Sync for WarmPool {}

#[cfg(test)]
mod tests {
    use super::stratified_membership;
    use super::WarmPool;
    use cudarc::driver::DevicePtr;
    use std::collections::HashSet;

    fn per_layer_counts(m: &[(usize, usize)], num_layers: usize) -> Vec<usize> {
        let mut counts = vec![0usize; num_layers];
        for &(layer, _) in m {
            counts[layer] += 1;
        }
        counts
    }

    /// The defect stratification exists to remove: every layer gets the same
    /// count, so no layer is left permanently short by a coin flip at startup.
    #[test]
    fn every_layer_gets_the_same_share() {
        let m = stratified_membership(48, 128, 48 * 40, 0, 0, 0xC0FFEE);
        assert_eq!(m.len(), 48 * 40);
        assert!(per_layer_counts(&m, 48).iter().all(|&c| c == 40));
    }

    /// A total that does not divide evenly spreads the remainder one per layer,
    /// so the spread between the fullest and emptiest layer is never above one.
    #[test]
    fn a_remainder_is_spread_one_per_layer() {
        let m = stratified_membership(48, 128, 48 * 40 + 7, 0, 0, 1);
        let counts = per_layer_counts(&m, 48);
        assert_eq!(m.len(), 48 * 40 + 7);
        assert_eq!(counts.iter().filter(|&&c| c == 41).count(), 7);
        assert_eq!(counts.iter().filter(|&&c| c == 40).count(), 41);
    }

    /// No expert may appear twice: two warm slots holding one expert would
    /// waste a slot and leave `ram` pointing at whichever was written last.
    #[test]
    fn no_expert_is_drawn_twice() {
        let m = stratified_membership(8, 16, 8 * 9, 0, 0, 42);
        let unique: HashSet<(usize, usize)> = m.iter().copied().collect();
        assert_eq!(unique.len(), m.len());
    }

    /// The draw is reproducible from its seed, and a different seed moves it —
    /// so a warm-tier measurement is attributable to a change, not to luck.
    #[test]
    fn the_draw_is_deterministic_in_the_seed() {
        let a = stratified_membership(8, 64, 8 * 10, 0, 0, 7);
        let b = stratified_membership(8, 64, 8 * 10, 0, 0, 7);
        let c = stratified_membership(8, 64, 8 * 10, 0, 0, 8);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// Asking for more than the model has yields the whole model once, not a
    /// list with repeats or an over-long one.
    #[test]
    fn asking_for_everything_yields_each_expert_once() {
        let m = stratified_membership(4, 8, 999, 0, 0, 3);
        assert_eq!(m.len(), 32);
        let unique: HashSet<(usize, usize)> = m.iter().copied().collect();
        assert_eq!(unique.len(), 32);
    }

    /// A warm tier of zero is legal — the cold tier holds every expert — and
    /// must produce an empty membership rather than a panic.
    #[test]
    fn a_zero_sized_warm_tier_is_empty_not_an_error() {
        assert!(stratified_membership(48, 128, 0, 0, 0, 1).is_empty());
        assert!(stratified_membership(0, 128, 100, 0, 0, 1).is_empty());
        assert!(stratified_membership(48, 0, 100, 0, 0, 1).is_empty());
        // Every layer pinned: nothing is evictable, so nothing is drawable.
        assert!(stratified_membership(2, 128, 100, 0, 2, 1).is_empty());
    }

    /// The draw covers the whole expert range rather than clustering at the
    /// front — a sampler that always took `0..take` would pass every test above.
    #[test]
    fn the_draw_reaches_the_far_end_of_a_layer() {
        let m = stratified_membership(1, 128, 16, 0, 0, 99);
        assert!(
            m.iter().any(|&(_, e)| e >= 64),
            "every drawn expert came from the first half: {m:?}"
        );
    }

    /// **The pinned prefix is never drawn, in either pass.**
    ///
    /// Those experts are permanently VRAM-resident, so a warm copy could only be
    /// read by a load that cannot happen — and unlike the VRAM prefix, which
    /// pass 2 legitimately insures against eviction, there is no eviction to
    /// insure against. A slot spent here is a slot the evictable set does not
    /// get. Asking for more than the drawable set holds must therefore return
    /// only the drawable set, not fall back to the pinned layers.
    #[test]
    fn the_pinned_prefix_is_never_drawn() {
        // 4 layers × 8 experts, layers 0–1 pinned: 16 drawable, ask for all 32.
        let m = stratified_membership(4, 8, 32, 0, 2, 5);
        assert_eq!(m.len(), 16, "the draw reached past the drawable set");
        for &(layer, expert) in &m {
            assert!(
                layer >= 2,
                "warm slot spent on L{layer}E{expert}, which is permanently resident"
            );
        }
        let unique: HashSet<(usize, usize)> = m.iter().copied().collect();
        assert_eq!(unique.len(), 16);
        // Both drawable layers get an equal share, as they do without pinning.
        let counts = per_layer_counts(&m, 4);
        assert_eq!(counts, vec![0, 0, 8, 8]);
    }

    /// The pinned skip composes with the VRAM prefix rather than replacing it:
    /// the prefix still orders *which* evictable experts are taken first.
    #[test]
    fn pinning_and_the_vram_prefix_compose() {
        // 4 layers × 8; layers 0–1 pinned; VRAM holds the first 20 (so layer 2
        // is entirely inside the prefix and layer 3 has 4 inside, 4 outside).
        let m = stratified_membership(4, 8, 4, 20, 2, 9);
        assert_eq!(m.len(), 4);
        for &(layer, expert) in &m {
            assert!(layer >= 2, "drew pinned L{layer}E{expert}");
            assert!(
                layer * 8 + expert >= 20,
                "drew L{layer}E{expert}, which VRAM already holds, while the \
                 complement still had candidates"
            );
        }
    }

    /// **The point of the VRAM prefix.** With room for exactly the complement of
    /// the startup VRAM fill, every warm slot goes to an expert VRAM will *not*
    /// hold — so the whole miss stream is covered and none of the tier is spent
    /// on guaranteed hits.
    #[test]
    fn the_complement_of_vram_is_taken_first() {
        // 4 layers × 8 experts; VRAM takes the first 12 (layers 0–1 entirely).
        let m = stratified_membership(4, 8, 20, 12, 0, 5);
        assert_eq!(m.len(), 20);
        for &(layer, expert) in &m {
            assert!(
                layer * 8 + expert >= 12,
                "warm slot spent on L{layer}E{expert}, which VRAM already holds"
            );
        }
    }

    /// Once the complement is covered, the remaining slots insure the
    /// VRAM-resident experts against eviction rather than going unused.
    #[test]
    fn slots_past_the_complement_fall_back_to_the_vram_resident() {
        let m = stratified_membership(4, 8, 28, 12, 0, 5);
        assert_eq!(m.len(), 28);
        let unique: HashSet<(usize, usize)> = m.iter().copied().collect();
        assert_eq!(unique.len(), 28, "an expert was drawn twice across passes");
        // All 20 outside the prefix, plus 8 of the 12 inside it.
        let inside = m.iter().filter(|&&(l, e)| l * 8 + e < 12).count();
        assert_eq!(inside, 8);
    }

    /// The complement pass stays stratified even though the prefix leaves the
    /// layers with unequal pools: a layer that has candidates never falls two
    /// behind one that does.
    #[test]
    fn the_complement_pass_stays_level_across_layers() {
        // VRAM takes layers 0–1 entirely, so only layers 2–3 have candidates.
        let m = stratified_membership(4, 8, 9, 16, 0, 11);
        let counts = per_layer_counts(&m, 4);
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 0);
        assert!(
            counts[2].abs_diff(counts[3]) <= 1,
            "layers 2 and 3 got {} and {}",
            counts[2],
            counts[3]
        );
    }

    /// A prefix that ends mid-layer splits that layer's pool rather than
    /// rounding it to a whole layer either way.
    #[test]
    fn a_prefix_ending_mid_layer_splits_that_layer() {
        // 4 layers × 8; prefix 10 ⇒ layer 1 has 2 covered, 6 not.
        let m = stratified_membership(4, 8, 22, 10, 0, 3);
        assert_eq!(m.len(), 22);
        for &(layer, expert) in &m {
            assert!(layer * 8 + expert >= 10, "L{layer}E{expert} is inside");
        }
        assert_eq!(per_layer_counts(&m, 4)[1], 6);
    }

    /// **The bandwidth denominator for the expert streamer.**
    ///
    /// Whether prefetching can help at all depends on which side of saturation
    /// the warm→VRAM path runs on, and that cannot be read off the load counts
    /// alone — it needs the achievable ceiling on THIS machine. Measures the
    /// real path: `cuMemAllocHost_v2` pinned source, one whole expert slot per
    /// copy, issued on a stream exactly as `build_slot_from_record_on_stream`
    /// does.
    ///
    /// Reports three numbers, because they answer different questions:
    ///  * **pinned, back-to-back** — the ceiling a perfectly-pipelined streamer
    ///    could reach.
    ///  * **pinned, one-at-a-time (sync per copy)** — what a demand load that
    ///    something is waiting on actually gets, latency included.
    ///  * **pageable** — the penalty for missing the pinned tier.
    ///
    /// Prints rather than asserts a threshold: this is a property of the host's
    /// PCIe link, not of our code, and a hard number here would just encode one
    /// machine's bus into the test suite.
    #[test]
    #[ignore]
    fn measure_h2d_bandwidth_at_expert_slot_size() {
        use candle::{DType, Device, Tensor};

        // The real geometry: 14.2 MB per expert slot on DeepSeek-V4-Flash.
        const SLOT_BYTES: usize = 14_200_000;
        const COPIES: usize = 64;

        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let Device::Cuda(cuda) = &device else {
            unreachable!("new_cuda yields a cuda device")
        };

        let Some(mut pool) = WarmPool::try_alloc(1, SLOT_BYTES) else {
            eprintln!("[skip] could not pin {SLOT_BYTES} bytes");
            return;
        };
        // Touch the source so the pages are resident and the first copy is not
        // paying for a fault that belongs to setup rather than to the link.
        // Through the accessor, so the benchmark measures the same bounded path
        // the pipeline takes rather than a raw slice beside it.
        let src = pool.slot_mut(0, SLOT_BYTES);
        src.fill(0xA5);

        let elems = SLOT_BYTES / 4;
        let dst = Tensor::zeros(elems, DType::F32, &device).expect("device buffer");
        let stream = cuda.cuda_stream();
        let (storage, _) = dst.storage_and_layout();
        // `_guard` keeps the device pointer valid for every copy below.
        let (dst_ptr, _guard) = match &*storage {
            candle::Storage::Cuda(s) => s
                .as_cuda_slice::<f32>()
                .expect("f32 device buffer")
                .device_ptr(&stream),
            _ => unreachable!("cuda tensor"),
        };

        let copy = |n: usize, sync_each: bool| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                unsafe {
                    cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                        dst_ptr,
                        pool.base as *const std::ffi::c_void,
                        SLOT_BYTES,
                        stream.cu_stream(),
                    );
                }
                if sync_each {
                    stream.synchronize().expect("sync");
                }
            }
            if !sync_each {
                stream.synchronize().expect("sync");
            }
            let secs = t0.elapsed().as_secs_f64();
            (n * SLOT_BYTES) as f64 / secs / 1e9
        };

        // Warm up: the first transfer on a fresh context pays one-off setup.
        let _ = copy(4, false);
        let streamed = copy(COPIES, false);
        let per_copy = copy(COPIES, true);

        let mut pageable_src = vec![0xA5u8; SLOT_BYTES];
        pageable_src[0] = 1;
        let t0 = std::time::Instant::now();
        for _ in 0..COPIES {
            unsafe {
                cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                    dst_ptr,
                    pageable_src.as_ptr() as *const std::ffi::c_void,
                    SLOT_BYTES,
                    stream.cu_stream(),
                );
            }
        }
        stream.synchronize().expect("sync");
        let pageable = (COPIES * SLOT_BYTES) as f64 / t0.elapsed().as_secs_f64() / 1e9;

        // ── The other half of the question: what does a synchronous D2H COST,
        // independent of how much data it moves or what the GPU is doing? ──
        //
        // The MoE routing readback is 3.3 ms per layer, 43 times per token, and
        // whether that is recoverable depends entirely on which it is: GPU
        // catch-up (real work, only fixable by making the work cheaper) or WDDM
        // round-trip latency (pure overhead, fixable only by removing the sync).
        // Draining the queue first and then timing a 512-byte readback isolates
        // the latency with certainty — there is nothing left to catch up on.
        let tiny = Tensor::zeros(128, DType::F32, &device).expect("tiny buffer");
        stream.synchronize().expect("drain");
        let t0 = std::time::Instant::now();
        const READBACKS: usize = 200;
        for _ in 0..READBACKS {
            let _ = tiny.to_vec1::<f32>().expect("readback");
        }
        let per_readback_us = t0.elapsed().as_secs_f64() * 1e6 / READBACKS as f64;

        eprintln!(
            "[h2d] slot={:.1} MB, {COPIES} copies",
            SLOT_BYTES as f64 / 1e6
        );
        eprintln!(
            "[d2h] empty-queue sync readback (512 B) = {per_readback_us:.0} us \
             — the floor under the per-layer routing readback"
        );
        eprintln!("[h2d] pinned, back-to-back   = {streamed:.1} GB/s");
        eprintln!(
            "[h2d] pinned, sync per copy  = {per_copy:.1} GB/s  ({:.2} ms per expert)",
            SLOT_BYTES as f64 / per_copy / 1e6
        );
        eprintln!("[h2d] pageable, back-to-back = {pageable:.1} GB/s");
    }

    // ── Pinned-memory bounds ──────────────────────────────────────────────
    //
    // The accessors hand out raw slices over `cuMemAllocHost` memory, where an
    // overrun is a silent write into unrelated process memory rather than a
    // fault. These run in the same `--release` configuration the gates do,
    // which is the whole point of them: a `debug_assert` is absent from every
    // build that matters.
    //
    // The first four need no device — an empty pool has zero slots, so every
    // index is out of bounds, and the overflow is refused before
    // `cuMemAllocHost` is reached.

    #[test]
    #[should_panic(expected = "past the pool's 0 slots")]
    fn an_empty_pool_refuses_a_slot_read() {
        let pool = WarmPool::empty();
        let _ = pool.slot_ref(0, 0);
    }

    #[test]
    #[should_panic(expected = "past the pool's 0 slots")]
    fn an_empty_pool_refuses_a_slot_write() {
        let mut pool = WarmPool::empty();
        let _ = pool.slot_mut(0, 0);
    }

    #[test]
    #[should_panic(expected = "span [0, 1) is past the pool's 0 slots")]
    fn an_empty_pool_refuses_a_span() {
        let mut pool = WarmPool::empty();
        let _ = pool.span_mut(0, 1);
    }

    #[test]
    fn a_wrapping_slot_count_is_refused_before_allocating() {
        // `num_slots * slot_size` wraps to a small number; the checked multiply
        // must answer `None` rather than pin that small buffer and then hand
        // out `num_slots` slots over it.
        assert!(WarmPool::try_alloc(usize::MAX / 4 + 1, 8).is_none());
        assert!(WarmPool::try_alloc(usize::MAX, 2).is_none());
    }

    /// The length bound, which needs real pinned memory to have a slot size at
    /// all — hence the device, without which `cuMemAllocHost` answers
    /// `NOT_INITIALIZED` and the test would pass by skipping the thing it
    /// exists to check. Skips only where there is no CUDA device: the bound is
    /// the property under test, not the machine's ability to pin.
    #[test]
    fn a_write_wider_than_its_slot_panics() {
        const SLOT: usize = 4096;
        let Ok(_device) = candle::Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return;
        };
        let Some(mut pool) = WarmPool::try_alloc(2, SLOT) else {
            eprintln!("[skip] could not pin {} B", 2 * SLOT);
            return;
        };
        // Exactly a slot is fine, and so is the last slot.
        assert_eq!(pool.slot_mut(0, SLOT).len(), SLOT);
        assert_eq!(pool.slot_mut(1, SLOT).len(), SLOT);
        assert_eq!(pool.span_mut(0, 2).len(), 2 * SLOT);

        // One byte past, and one slot past, both refused. `catch_unwind` rather
        // than `#[should_panic]` so the pool stays alive across both probes and
        // is freed once, by its own `Drop`, at the end of the test.
        let too_wide = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = pool.slot_mut(0, SLOT + 1);
        }));
        assert!(
            too_wide.is_err(),
            "a write one byte past the slot must panic, not run"
        );
        let too_far = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = pool.slot_mut(2, SLOT);
        }));
        assert!(
            too_far.is_err(),
            "a write to the slot after the last must panic, not run"
        );
    }
}
