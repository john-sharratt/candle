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
