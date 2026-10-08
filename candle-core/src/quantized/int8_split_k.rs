//! Split-K selection for the q8a128 int8 **dense** matmul.
//!
//! The dense kernel tiles the output as `ceil(M/16) × ceil(N/32)` blocks of 128 threads, each
//! walking the whole of K. That is the right shape for prefill, where M fills the grid. At decode
//! width with a narrow N it is not: the Qwen3.8-Flash-Next hyper-connection `down` projection
//! (N = 416, K = 10,240) launches 13 blocks on a 110-SM card, and `ncu` measured 8% achieved
//! occupancy and 7% of DRAM — 80 serialized K steps per block, with nearly the whole card idle.
//!
//! # One summation order, whatever the launch
//!
//! Every dense int8 launch sums K the same way: one K tile's fold at a time, in tile order,
//! `((0 + f₀) + f₁) + …`. A split launch cuts K into `splits` slices, one block each; each block
//! stores every tile's fold as its own F32 partial (`TILE_PARTIALS` in the kernel), and the last
//! block to finish adds them in tile order — the same chain, so it produces **exactly** the
//! unsplit kernel's bits, at any split count. Two things follow:
//!
//! - a row's output does not depend on how many other rows its wave carried — the split
//!   decision depends on M, the numbers do not;
//! - the split count is purely a speed decision, so this rule can change without moving a
//!   single output bit.
//!
//! The unsplit kernel carries nothing for this: its accumulator already is the chain, so its
//! arithmetic is the plain tile-by-tile fold with no split-K term in it. MXFP4 is the exception
//! the launcher handles: its per-sub fold accumulates straight into the sum, so it never splits.
//!
//! # The rule
//!
//! ```text
//! base   = ceil(M/16) · ceil(N/32)                              // the unsplit grid
//! want   = ceil(SPLIT_TARGET_BLOCKS_PER_SM · SMs / base)        // depth that fills the card
//! splits = ceil(K_tiles / ceil(K_tiles / min(want, K_tiles)))   // no empty slice
//! split  ⇔  M ≤ MAX_SPLIT_ROWS  ∧  2·base ≤ SPLIT_TARGET_BLOCKS_PER_SM · SMs
//!           ∧ splits > 1  ∧  the scratch holds K_tiles·M·N partials and base counters
//! ```
//!
//! Prefill shapes never qualify — their unsplit grid already fills the card — and neither does a
//! shape whose partials would not fit the fixed scratch; both run unsplit, with the same bits.
//!
//! # The narrow launch
//!
//! At up to eight rows a shape that splits runs the **narrow** kernel instead
//! ([`q8a128_dense_plan`]): one block per 8-row output tile, K walked by the block's warps and
//! summed in shared memory — the same tile-ordered chain, so the same bits, with no partials in
//! global memory, no counters and no last-block tail. It runs at the most warps per block
//! ([`narrow_warps`]) at which every block is resident in one wave; a shape for which not even
//! four warps give one wave — or whose fold slots (one per K tile past the first warp's range)
//! would not fit [`NARROW_SMEM_CAP`] — stays split-K.

/// The mode-1 token tile the split kernel shares.
const M_TILE: usize = 16;
/// The output tile: rows per block.
const N_TILE: usize = 32;
/// The K tile: one activation block, one weight chunk.
const K_TILE: usize = 128;

/// Blocks per SM a split launch aims for, and below which an unsplit launch is worth splitting.
///
/// Tuned with the KO projection bench (`gr_hyper_bench ko`) on the RTX PRO 5000 (110 SMs).
pub const SPLIT_TARGET_BLOCKS_PER_SM: usize = 4;

/// The widest wave a launch splits at: two token tiles. A decode wave of one to a few rows per
/// sequence, and a narrow verify, sit under it; wider launches fill the card unsplit.
pub const MAX_SPLIT_ROWS: usize = 32;

/// F32 partials the per-stream scratch holds: 16 MiB.
pub const SPLIT_SCRATCH_PARTIALS: usize = 1 << 22;

/// Per-tile counters the per-stream scratch holds.
pub const SPLIT_SCRATCH_COUNTERS: usize = 1024;

/// The K tiles of a `k`-wide contraction — one F32 partial each in a split launch.
pub fn dense_k_tiles(k: usize) -> usize {
    k / K_TILE
}

/// The grid depth a request for `splits` slices of a `k`-wide contraction launches at: the kernel
/// cuts K into slices of `ceil(K_tiles / splits)` tiles, so this is the count of non-empty ones.
pub fn dense_k_split_depth(k: usize, splits: usize) -> usize {
    let tiles = dense_k_tiles(k);
    if tiles == 0 || splits == 0 {
        return 1;
    }
    tiles.div_ceil(tiles.div_ceil(splits.min(tiles)))
}

/// Whether a split launch of `[m, k] × [n, k]ᵀ` fits the fixed per-stream scratch.
pub fn dense_k_split_fits(m: usize, n: usize, k: usize) -> bool {
    dense_k_tiles(k) * m * n <= SPLIT_SCRATCH_PARTIALS
        && m.div_ceil(M_TILE) * n.div_ceil(N_TILE) <= SPLIT_SCRATCH_COUNTERS
}

/// How many K slices the dense int8 matmul of `[m, k] × [n, k]ᵀ` runs in on a part with
/// `sm_count` SMs: enough to fill the card when splitting pays, `1` (the unsplit kernel)
/// otherwise. Every count produces the same bits.
pub fn q8a128_dense_k_splits(m: usize, n: usize, k: usize, sm_count: usize) -> usize {
    if m == 0 || n == 0 || sm_count == 0 || m > MAX_SPLIT_ROWS {
        return 1;
    }
    let base = m.div_ceil(M_TILE) * n.div_ceil(N_TILE);
    let target = SPLIT_TARGET_BLOCKS_PER_SM * sm_count;
    if 2 * base > target || !dense_k_split_fits(m, n, k) {
        return 1;
    }
    dense_k_split_depth(k, target.div_ceil(base))
}

/// The most rows a narrow launch carries: the int8 MMA's first eight token rows.
pub const NARROW_MAX_ROWS: usize = 8;

/// The dynamic shared memory a narrow launch may use, mirroring `NARROW_SMEM_CAP` in
/// `quantized/narrow_layout.cuh`.
pub const NARROW_SMEM_CAP: usize = 96 * 1024;

/// The widest weight ring one narrow warp keeps, across every format it runs: Q8_KO's four
/// 1,056-byte chunks (`narrow_ahead` in `narrow_layout.cuh` keeps every other format at or
/// under it).
const NARROW_RING_BYTES_PER_WARP: usize = 4 * 1056;

/// Bytes of one K tile's fold slot: a float2 per lane.
const NARROW_SLOT_BYTES_PER_TILE: usize = 32 * 8;

/// The most dynamic shared memory a narrow launch of a `k`-wide contraction at `warps` warps
/// uses, in any format: the weight rings, and a fold slot per K tile past warp 0's range.
pub fn narrow_smem_bound(k: usize, warps: usize) -> usize {
    let tiles = dense_k_tiles(k);
    let per = tiles.div_ceil(warps.max(1));
    warps * NARROW_RING_BYTES_PER_WARP + tiles.saturating_sub(per) * NARROW_SLOT_BYTES_PER_TILE
}

/// How one dense int8 matmul launches. Every plan produces the same bits — each sums K one
/// tile's fold at a time, in tile order — so the plan is purely a speed decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensePlan {
    /// The tile grid over the whole of K, one block per (token tile, 32 rows).
    Unsplit,
    /// K cut into this many slices across blocks, partials summed by the last block.
    SplitK(usize),
    /// One block per 8 output rows, K across its `warps` warps, summed in shared memory.
    Narrow { warps: usize },
}

/// The fewest and the most warps a narrow block runs at; the most mirrors `NARROW_MAX_WARPS`
/// in `quantized/narrow_layout.cuh`.
pub const NARROW_MIN_WARPS: usize = 4;
pub const NARROW_MAX_WARPS: usize = 16;

/// Shared memory one SM gives its resident blocks, on every part the kernels are built for
/// (sm_86, sm_89 and sm_120 all carry 100 KiB).
const SM_SMEM_BYTES: usize = 100 * 1024;

/// Shared memory the runtime reserves for each resident block on those parts.
const SMEM_RESERVED_PER_BLOCK: usize = 1024;

/// Narrow warps one SM holds at once: the kernel compiles to at most 64 registers a thread
/// (60–64 across the formats and parts), so the 64K-register file holds 32 warps of it.
const NARROW_WARPS_PER_SM: usize = 32;

/// Narrow blocks of a `k`-wide contraction at `warps` warps one SM holds at once: the lesser
/// of what its shared memory and its register file admit.
pub fn narrow_blocks_per_sm(k: usize, warps: usize) -> usize {
    let smem = narrow_smem_bound(k, warps) + SMEM_RESERVED_PER_BLOCK;
    (SM_SMEM_BYTES / smem).min(NARROW_WARPS_PER_SM / warps.max(1))
}

/// The warps of a narrow launch over an `n`-wide output and a `k`-wide contraction on
/// `sm_count` SMs: the most — between [`NARROW_MIN_WARPS`] and [`NARROW_MAX_WARPS`], and no
/// more than K has tiles — at which every one of the `n / 8` blocks is resident in the same
/// wave and the shared memory fits [`NARROW_SMEM_CAP`]. `None` when not even the fewest warps
/// give one wave: the narrow kernel's blocks walk their K in one pass, so a second wave would
/// pay the whole latency chain again, and that shape stays split-K.
///
/// Measured with `bench_narrow_against_split_on_decode_projections` (RTX PRO 5000, 110 SMs):
/// more warps keep more weight chunks in flight and win until the blocks stop fitting one
/// wave — the 52-block hyper-connection `down` is fastest at 12–16 warps, while the 320-block
/// out-projections lose 10–30% going from a one-wave count to a two-wave one.
pub fn narrow_warps(n: usize, k: usize, sm_count: usize) -> Option<usize> {
    let blocks = n.div_ceil(8);
    let top = dense_k_tiles(k).clamp(NARROW_MIN_WARPS, NARROW_MAX_WARPS);
    (NARROW_MIN_WARPS..=top).rev().find(|&w| {
        narrow_smem_bound(k, w) <= NARROW_SMEM_CAP
            && narrow_blocks_per_sm(k, w) * sm_count >= blocks
    })
}

/// How the dense int8 matmul of `[m, k] × [n, k]ᵀ` launches on `sm_count` SMs. `per_tile`
/// is whether the weight format folds one K tile at a time (every affine KO format; not
/// MXFP4, whose per-sub fold has no per-tile partial) — only those split or run narrow.
/// A shape that splits runs narrow instead at up to [`NARROW_MAX_ROWS`] rows, when
/// [`narrow_warps`] finds a one-wave launch for it.
pub fn q8a128_dense_plan(
    m: usize,
    n: usize,
    k: usize,
    sm_count: usize,
    per_tile: bool,
) -> DensePlan {
    if !per_tile {
        return DensePlan::Unsplit;
    }
    let splits = q8a128_dense_k_splits(m, n, k, sm_count);
    if splits == 1 {
        return DensePlan::Unsplit;
    }
    if m <= NARROW_MAX_ROWS {
        if let Some(warps) = narrow_warps(n, k, sm_count) {
            return DensePlan::Narrow { warps };
        }
    }
    DensePlan::SplitK(splits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SM: usize = 110; // RTX PRO 5000 Blackwell.

    #[test]
    fn the_split_depth_never_launches_an_empty_slice() {
        // 80 tiles asked for 34 slices: 3 tiles each, so 27 slices cover K.
        assert_eq!(dense_k_split_depth(10_240, 34), 27);
        assert_eq!(dense_k_split_depth(10_240, 40), 40);
        assert_eq!(
            dense_k_split_depth(10_240, 500),
            80,
            "never more slices than tiles"
        );
        assert_eq!(
            dense_k_split_depth(384, 2),
            2,
            "three tiles: two slices, the last one short"
        );
        assert_eq!(dense_k_split_depth(128, 8), 1);
        for k in [128usize, 384, 2560, 10_240] {
            for s in 1..=100 {
                let d = dense_k_split_depth(k, s);
                let tiles = dense_k_tiles(k);
                let w = tiles.div_ceil(s.min(tiles));
                assert!((d - 1) * w < tiles, "slice {d} of K={k} at {s} is empty");
            }
        }
    }

    /// The shape this exists for: the hyper-connection `down` splits to fill the card at every
    /// decode width up to two token tiles.
    #[test]
    fn the_hyper_connection_down_splits_at_decode_width() {
        // One token tile: 13 blocks unsplit, 440 wanted → 34 slices → 27 non-empty.
        for m in [1usize, 5, 16] {
            assert_eq!(q8a128_dense_k_splits(m, 416, 10_240, SM), 27, "{m} rows");
        }
        // Two token tiles: 26 blocks unsplit → 17 slices of 5 tiles → 16 non-empty.
        for m in [17usize, 32] {
            assert_eq!(q8a128_dense_k_splits(m, 416, 10_240, SM), 16, "{m} rows");
        }
    }

    #[test]
    fn prefill_widths_never_split() {
        assert_eq!(q8a128_dense_k_splits(33, 416, 10_240, SM), 1);
        assert_eq!(q8a128_dense_k_splits(512, 416, 10_240, SM), 1);
        assert_eq!(q8a128_dense_k_splits(2048, 10_240, 384, SM), 1);
    }

    #[test]
    fn a_wide_output_never_splits() {
        assert_eq!(q8a128_dense_k_splits(1, 10_240, 384, SM), 1);
        assert_eq!(q8a128_dense_k_splits(1, 248_320, 2560, SM), 1);
    }

    #[test]
    fn a_single_tile_does_not_split() {
        assert_eq!(q8a128_dense_k_splits(1, 416, 128, SM), 1);
    }

    #[test]
    fn degenerate_shapes_do_not_split() {
        assert_eq!(q8a128_dense_k_splits(0, 416, 10_240, SM), 1);
        assert_eq!(q8a128_dense_k_splits(1, 416, 10_240, 0), 1);
    }

    /// The narrow kernel's shared-memory bound: a Q8_KO ring of 4,224 bytes per warp, and 256
    /// bytes per K tile past the first warp's range.
    #[test]
    fn the_narrow_smem_bound_counts_rings_and_slotted_tiles() {
        // 80 tiles: 20 per warp at 4 warps (60 slotted), 10 at 8 (70 slotted).
        assert_eq!(narrow_smem_bound(10_240, 4), 4 * 4224 + 60 * 256);
        assert_eq!(narrow_smem_bound(10_240, 8), 8 * 4224 + 70 * 256);
        // 5 tiles at 4 warps: 2 per warp, 3 slotted; at 8: 1 per warp, 4 slotted.
        assert_eq!(narrow_smem_bound(640, 4), 4 * 4224 + 3 * 256);
        assert_eq!(narrow_smem_bound(640, 8), 8 * 4224 + 4 * 256);
        // One warp slots nothing.
        assert_eq!(narrow_smem_bound(10_240, 1), 4224);
        // The largest K a narrow launch takes at eight warps: 192 tiles → 168 slotted, 76,800
        // bytes; 256 tiles → 224 slotted, 91,136 — both under the cap; 384 tiles is over it.
        assert!(narrow_smem_bound(24_576, 8) <= NARROW_SMEM_CAP);
        assert!(narrow_smem_bound(32_768, 8) <= NARROW_SMEM_CAP);
        assert!(narrow_smem_bound(49_152, 8) > NARROW_SMEM_CAP);
    }

    /// How many narrow blocks one SM holds: shared memory against 100 KiB (1 KiB reserved per
    /// block), and the register file's 32 warps.
    #[test]
    fn narrow_residency_is_the_lesser_of_smem_and_registers() {
        // 16 warps over 80 tiles: 86,784 + 1,024 bytes → one block (and two by registers).
        assert_eq!(narrow_blocks_per_sm(10_240, 16), 1);
        // 5 warps over 48 tiles: 30,848 + 1,024 → three blocks (six by registers).
        assert_eq!(narrow_blocks_per_sm(6144, 5), 3);
        // 6 warps over 48 tiles: 35,584 + 1,024 → two.
        assert_eq!(narrow_blocks_per_sm(6144, 6), 2);
        // 4 warps over 5 tiles: 17,664 + 1,024 → five by memory, eight by registers.
        assert_eq!(narrow_blocks_per_sm(640, 4), 5);
        // One tile slots nothing: 16 rings (67,584 + 1,024) hold one block, 4 rings five.
        assert_eq!(narrow_blocks_per_sm(128, 16), 1);
        assert_eq!(narrow_blocks_per_sm(128, 4), 5);
    }

    /// The Flash-Next decode projections at five rows on the 110-SM card: each shape that
    /// splits runs narrow, at the most warps whose blocks all fit one wave.
    #[test]
    fn decode_width_projections_run_narrow() {
        // Hyper-connection `down`: 52 blocks, one per SM at 16 warps.
        assert_eq!(
            q8a128_dense_plan(5, 416, 10_240, SM, true),
            DensePlan::Narrow { warps: 16 }
        );
        // 320 blocks need three per SM: the DeltaNet out-proj (48 tiles) fits five warps, the
        // attention out-proj (32 tiles) six, and the shared-expert down has five tiles to give.
        for (k, warps) in [(6144usize, 5usize), (4096, 6), (640, 5)] {
            assert_eq!(
                q8a128_dense_plan(5, 2560, k, SM, true),
                DensePlan::Narrow { warps },
                "K = {k}"
            );
        }
        // The stacked router | shared gate_up | gate: 1,824 rows → 228 blocks, three per SM
        // at six warps.
        assert_eq!(
            q8a128_dense_plan(5, 1824, 2560, SM, true),
            DensePlan::Narrow { warps: 6 }
        );
        for m in 1..=NARROW_MAX_ROWS {
            assert_eq!(
                q8a128_dense_plan(m, 416, 10_240, SM, true),
                DensePlan::Narrow { warps: 16 },
                "{m} rows"
            );
        }
    }

    /// A shape whose narrow blocks cannot all be resident at once stays split-K: 512 blocks of
    /// a 4,096-wide output need five per SM, and four warps over 32 tiles give four.
    #[test]
    fn a_shape_past_one_narrow_wave_stays_split() {
        assert_eq!(narrow_warps(4096, 4096, SM), None);
        assert_eq!(
            q8a128_dense_plan(5, 4096, 4096, SM, true),
            DensePlan::SplitK(4)
        );
    }

    /// Past eight rows a shape that splits stays split-K, at the depth the split rule picks.
    #[test]
    fn wider_decode_waves_split() {
        for m in [9usize, 16] {
            assert_eq!(
                q8a128_dense_plan(m, 416, 10_240, SM, true),
                DensePlan::SplitK(27),
                "{m} rows"
            );
        }
        assert_eq!(
            q8a128_dense_plan(32, 416, 10_240, SM, true),
            DensePlan::SplitK(16)
        );
    }

    /// A shape that does not split today does not run narrow either, and a format without a
    /// per-tile fold (MXFP4) runs unsplit whatever its shape.
    #[test]
    fn narrow_needs_a_shape_that_splits_and_a_per_tile_format() {
        assert_eq!(
            q8a128_dense_plan(1, 10_240, 384, SM, true),
            DensePlan::Unsplit
        );
        assert_eq!(q8a128_dense_plan(1, 416, 128, SM, true), DensePlan::Unsplit);
        assert_eq!(
            q8a128_dense_plan(512, 416, 10_240, SM, true),
            DensePlan::Unsplit
        );
        assert_eq!(
            q8a128_dense_plan(5, 416, 10_240, SM, false),
            DensePlan::Unsplit
        );
        assert_eq!(
            q8a128_dense_plan(0, 416, 10_240, SM, true),
            DensePlan::Unsplit
        );
    }

    /// A K whose fold slots would not fit the cap even at four warps stays split-K: 512 tiles
    /// (384 slotted, 98,304 bytes, plus the rings) at five rows of a 416-wide output.
    #[test]
    fn a_k_past_the_narrow_cap_stays_split() {
        let k = 65_536;
        assert!(narrow_smem_bound(k, NARROW_MIN_WARPS) > NARROW_SMEM_CAP);
        assert!(dense_k_split_fits(5, 416, k));
        assert_eq!(
            q8a128_dense_plan(5, 416, k, SM, true),
            DensePlan::SplitK(q8a128_dense_k_splits(5, 416, k, SM))
        );
    }

    /// Every narrow plan the rule picks fits the shared-memory cap, carries at most eight rows,
    /// and puts every block in one wave — swept over M, N, K and part size.
    #[test]
    fn every_narrow_plan_fits() {
        for sm in [46usize, 76, 82, 110, 170] {
            for m in 1..=MAX_SPLIT_ROWS {
                for n in (32..=16_384).step_by(32) {
                    for k in [128usize, 384, 1024, 2560, 10_240, 65_536] {
                        if let DensePlan::Narrow { warps } = q8a128_dense_plan(m, n, k, sm, true) {
                            assert!(m <= NARROW_MAX_ROWS);
                            assert!((NARROW_MIN_WARPS..=NARROW_MAX_WARPS).contains(&warps));
                            assert!(narrow_smem_bound(k, warps) <= NARROW_SMEM_CAP);
                            assert!(narrow_blocks_per_sm(k, warps) * sm >= n.div_ceil(8));
                            assert!(q8a128_dense_k_splits(m, n, k, sm) > 1);
                        }
                    }
                }
            }
        }
    }

    /// Every shape that splits fits the fixed scratch and launches no empty slice — swept over
    /// M, N, K and part size.
    #[test]
    fn every_split_fits_the_scratch() {
        for sm in [46usize, 76, 82, 110, 170] {
            for m in 1..=MAX_SPLIT_ROWS {
                for n in (32..=16_384).step_by(32) {
                    for k in [128usize, 384, 1024, 2560, 10_240, 65_536] {
                        let s = q8a128_dense_k_splits(m, n, k, sm);
                        if s == 1 {
                            continue;
                        }
                        assert_eq!(s, dense_k_split_depth(k, s));
                        assert!(dense_k_tiles(k) * m * n <= SPLIT_SCRATCH_PARTIALS);
                        assert!(m.div_ceil(M_TILE) * n.div_ceil(N_TILE) <= SPLIT_SCRATCH_COUNTERS);
                    }
                }
            }
        }
    }
}
