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
    seed: u64,
) -> Vec<(usize, usize)> {
    if num_layers == 0 || experts_per_layer == 0 || warm_slots == 0 {
        return Vec::new();
    }
    let total = num_layers * experts_per_layer;
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
    // Per layer, the experts not yet drawn, partitioned so that the ones outside
    // the VRAM prefix are taken first.
    let mut decks: Vec<(Vec<usize>, Vec<usize>)> = (0..num_layers)
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
            for layer in 0..num_layers {
                if remaining == 0 {
                    break;
                }
                let deck = if pass == 0 {
                    &mut decks[layer].0
                } else {
                    &mut decks[layer].1
                };
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
                        gib = (slots * slot_size) as f64 / 1e9,
                        "warm tier: cuMemAllocHost refused; halving"
                    );
                    slots /= 2;
                }
            }
        }
        Self::empty()
    }

    fn try_alloc(num_slots: usize, slot_size: usize) -> Option<Self> {
        let total_size = num_slots * slot_size;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, total_size) };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return None;
        }
        // Feed the process-wide gauge: pinned memory is non-pageable, so the
        // host-RAM availability measurement must treat it as structural.
        candle::vram::note_host_pinned_alloc(total_size as u64);
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
    #[inline]
    pub(crate) fn slot_mut(&mut self, slot_idx: usize, len: usize) -> &mut [u8] {
        debug_assert!(slot_idx < self.num_slots);
        debug_assert!(len <= self.slot_size);
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
    #[inline]
    pub(crate) fn span_mut(&mut self, first: usize, n_slots: usize) -> &mut [u8] {
        debug_assert!(first + n_slots <= self.num_slots);
        unsafe {
            let ptr = self.base.add(first * self.slot_size);
            std::slice::from_raw_parts_mut(ptr, n_slots * self.slot_size)
        }
    }

    /// A shared byte slice for a slot — the source of every promotion.
    #[inline]
    pub(crate) fn slot_ref(&self, slot_idx: usize, len: usize) -> &[u8] {
        debug_assert!(slot_idx < self.num_slots);
        debug_assert!(len <= self.slot_size);
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
            candle::vram::note_host_pinned_free(self.total_size as u64);
        }
    }
}

// SAFETY: The pinned memory is allocated via cuMemAllocHost which returns
// a host-accessible pointer valid for any thread.  The pool is exclusively
// owned by the pipeline thread (no shared access).
#[cfg(feature = "cuda")]
unsafe impl Send for WarmPool {}

#[cfg(test)]
mod tests {
    use super::stratified_membership;
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
        let m = stratified_membership(48, 128, 48 * 40, 0, 0xC0FFEE);
        assert_eq!(m.len(), 48 * 40);
        assert!(per_layer_counts(&m, 48).iter().all(|&c| c == 40));
    }

    /// A total that does not divide evenly spreads the remainder one per layer,
    /// so the spread between the fullest and emptiest layer is never above one.
    #[test]
    fn a_remainder_is_spread_one_per_layer() {
        let m = stratified_membership(48, 128, 48 * 40 + 7, 0, 1);
        let counts = per_layer_counts(&m, 48);
        assert_eq!(m.len(), 48 * 40 + 7);
        assert_eq!(counts.iter().filter(|&&c| c == 41).count(), 7);
        assert_eq!(counts.iter().filter(|&&c| c == 40).count(), 41);
    }

    /// No expert may appear twice: two warm slots holding one expert would
    /// waste a slot and leave `ram` pointing at whichever was written last.
    #[test]
    fn no_expert_is_drawn_twice() {
        let m = stratified_membership(8, 16, 8 * 9, 0, 42);
        let unique: HashSet<(usize, usize)> = m.iter().copied().collect();
        assert_eq!(unique.len(), m.len());
    }

    /// The draw is reproducible from its seed, and a different seed moves it —
    /// so a warm-tier measurement is attributable to a change, not to luck.
    #[test]
    fn the_draw_is_deterministic_in_the_seed() {
        let a = stratified_membership(8, 64, 8 * 10, 0, 7);
        let b = stratified_membership(8, 64, 8 * 10, 0, 7);
        let c = stratified_membership(8, 64, 8 * 10, 0, 8);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// Asking for more than the model has yields the whole model once, not a
    /// list with repeats or an over-long one.
    #[test]
    fn asking_for_everything_yields_each_expert_once() {
        let m = stratified_membership(4, 8, 999, 0, 3);
        assert_eq!(m.len(), 32);
        let unique: HashSet<(usize, usize)> = m.iter().copied().collect();
        assert_eq!(unique.len(), 32);
    }

    /// A warm tier of zero is legal — the cold tier holds every expert — and
    /// must produce an empty membership rather than a panic.
    #[test]
    fn a_zero_sized_warm_tier_is_empty_not_an_error() {
        assert!(stratified_membership(48, 128, 0, 0, 1).is_empty());
        assert!(stratified_membership(0, 128, 100, 0, 1).is_empty());
        assert!(stratified_membership(48, 0, 100, 0, 1).is_empty());
    }

    /// The draw covers the whole expert range rather than clustering at the
    /// front — a sampler that always took `0..take` would pass every test above.
    #[test]
    fn the_draw_reaches_the_far_end_of_a_layer() {
        let m = stratified_membership(1, 128, 16, 0, 99);
        assert!(
            m.iter().any(|&(_, e)| e >= 64),
            "every drawn expert came from the first half: {m:?}"
        );
    }

    /// **The point of the VRAM prefix.** With room for exactly the complement of
    /// the startup VRAM fill, every warm slot goes to an expert VRAM will *not*
    /// hold — so the whole miss stream is covered and none of the tier is spent
    /// on guaranteed hits.
    #[test]
    fn the_complement_of_vram_is_taken_first() {
        // 4 layers × 8 experts; VRAM takes the first 12 (layers 0–1 entirely).
        let m = stratified_membership(4, 8, 20, 12, 5);
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
        let m = stratified_membership(4, 8, 28, 12, 5);
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
        let m = stratified_membership(4, 8, 9, 16, 11);
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
        let m = stratified_membership(4, 8, 22, 10, 3);
        assert_eq!(m.len(), 22);
        for &(layer, expert) in &m {
            assert!(layer * 8 + expert >= 10, "L{layer}E{expert} is inside");
        }
        assert_eq!(per_layer_counts(&m, 4)[1], 6);
    }
}
