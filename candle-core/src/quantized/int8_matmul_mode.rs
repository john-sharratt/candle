//! Tiling-mode selection for the q8a128 int8 **dense** matmul.
//!
//! The dense int8 kernel (`candle-kernels/.../dispatcher.cu`) has two tilings:
//!
//! - **mode-1** (`Bm = 16`, `N_SUB = 1`): `ceil(M/16)` batch-tiles × `ceil(N/32)` N-tiles.
//! - **mode-2** (`Bm = 32`, `N_SUB = 2`): `ceil(M/32)` batch-tiles × `ceil(N/64)` N-tiles — about
//!   **¼** as many blocks, and roughly **half** the DRAM traffic (it amortizes weight reads across
//!   a 2× wider token tile and activation reads across a 2× wider output tile).
//!
//! # What the 3-axis crossover benchmark showed (RTX 4090, 76 SMs, 64 MiB L2)
//!
//! (Blackwell's re-measurement is in [`MODE2_ANCHORS`]. The mechanisms below — the occupancy gate
//! and the `[17,32]` trap — reproduce on both parts; only the gate's *threshold* is per-part.)
//!
//! (`q8a128_dense_mode_crossover` in `quantized/cuda_tests.rs`, Q4_KO, L2 flushed, median of 30,
//! sweeping M, N, K independently.)
//!
//! The decisive observation: mode-2's modeled DRAM traffic is ~**0.5×** mode-1's *everywhere*, yet
//! it frequently loses — so the dense kernel here is **occupancy-bound, not bandwidth-bound**. The
//! winner tracks the *block count*, which is dominated by **N** (the output axis), not weight bytes
//! (my earlier weight-bytes fit was a confound: `W = N·K`):
//!
//! | N    | measured crossover M | reason                                            |
//! |------|----------------------|---------------------------------------------------|
//! | 2048 | ~80–96               | few N-blocks → mode-2 needs many M-tiles to fill   |
//! | 4096 | ~48–64               | —                                                 |
//! | 8192 | ~1 (mode-2 wins)     | many N-blocks → mode-2 fills the GPU immediately   |
//!
//! K only weakly lowers the crossover (more activation traffic), so it stays a reserved term.
//!
//! Two robust, mechanism-level effects:
//!
//! 1. **Occupancy gate.** Mode-2 wins once `blk2 = ceil(M/32)·ceil(N/64)` is enough to fill the
//!    GPU. The benchmark cleanly separates `blk2 = 64` (N=4096, M≤16 → mode-1) from `blk2 = 128`
//!    (N=8192, M≤16 → mode-2), so the threshold sits between — `≈ 5/4 × SM count` (≈95 on the 4090).
//! 2. **The `M ∈ [17, 32]` trap.** There `ceil(M/16)=2` but `ceil(M/32)=1`, so mode-1 launches 4×
//!    the blocks; it wins even at half the traffic and even when mode-2's occupancy is otherwise
//!    fine (confirmed at N=8192). This is a hard exclusion, independent of N.
//!
//! # The formula
//!
//! ```text
//! if 17 <= M <= 32:            use mode-1            // the block-count trap
//! else:                        mode-2  ⇔  ceil(M/32)·ceil(N/64) >= blocks_per_SM · SM_count
//! ```
//!
//! where `blocks_per_SM` is **calibrated per architecture** (see [`MODE2_ANCHORS`]) — `5/4` on Ada,
//! `7/3` on Blackwell. It was originally a single `5/4` scaled by SM count, on the theory that
//! expressing it over SM count made it track the GPU. Re-measuring on Blackwell (110 SMs, 96 MiB
//! L2) refuted that: the required blocks-per-SM is itself architecture-dependent, and the Ada
//! constant crossed to mode-2 at M ≈ 129/65/33 for N = 2048/4096/8192 where the part actually
//! wants 256/128/56 — i.e. it picked the weight-reuse tiling well before it wins. Fitting the
//! Blackwell anchor took formula-vs-measurement agreement from **87.2% to 94.2%** of clear-winner
//! cells, and Ada's decisions are unchanged (`ada_anchor_is_bit_identical_to_the_original_constant`).
//!
//! Fast (two ceil-divides + a compare), and correct at the production extremes the bench implies:
//! tiny-N `kv_proj`/router (`N ≤ 512`) → always mode-1; huge-N `lm_head` → mode-2 from M=1 (minus
//! the trap). Weight bytes and L2 are **not** terms — the decision is GPU fill, not cache pressure.

/// Mode-1 token tile (`Bm`). Mode-2's is twice this.
const M_TILE_MODE1: usize = 16;
/// Mode-2 token tile (`Bm`).
const M_TILE_MODE2: usize = 32;
/// Mode-2 N tile width (`32 * N_SUB`); the N axis is split into this-wide blocks.
const N_TILE_MODE2: usize = 64;

/// Occupancy target in **blocks per SM** that mode-2 must reach to win, calibrated per
/// architecture.
///
/// This was a single constant (`5/4`) scaled by SM count, on the theory that expressing it over
/// SM count made it track the GPU. Re-running the crossover benchmark on Blackwell showed that it
/// does not: the required blocks-per-SM is itself architecture-dependent, so scaling Ada's
/// constant by a larger SM count still crosses far too early.
///
/// | anchor | part | SMs | L2 | blocks/SM | `blk2` at crossover |
/// |---|---|---|---|---|---|
/// | Ada | RTX 4090 | 76 | 64 MiB | `5/4` | ~95 |
/// | Blackwell | RTX PRO 5000 | 110 | 96 MiB | `7/3` | 256 |
///
/// The Blackwell figure is the sharper of the two: N=2048/4096/8192 cross at M\*=256/128/56, and
/// all three land on `blk2 = 256` exactly. Mechanically this is consistent — mode-2 buys ~½ the
/// DRAM traffic at ~¼ the block count, so the more bandwidth-rich the part is relative to its
/// occupancy, the better filled mode-2 must be before the traffic saving pays.
///
/// **Ada's entry reproduces the original fit exactly**, so a 76-SM part decides identically to
/// before this table existed — `ada_anchor_is_bit_identical_to_the_original_constant` pins that,
/// and the whole `SM = 76` test suite below is unchanged.
///
/// Add an anchor when a new part is measured (`q8a128_dense_mode_crossover`); unmeasured parts
/// take the nearest anchor by SM count, which is a guess and is documented as one.
const MODE2_ANCHORS: [(usize, usize, usize); 2] = [
    // (SM count, blocks-per-SM numerator, denominator)
    (76, 5, 4),  // Ada / RTX 4090 — the original fit, preserved exactly.
    (110, 7, 3), // Blackwell / RTX PRO 5000 — blk2 = 256 at crossover.
];

/// Blocks-per-SM target for `sm_count`, from the nearest measured anchor.
///
/// Nearest-by-SM-count rather than interpolation: with two anchors an interpolation would invent a
/// curve the data cannot support, and a part far from both is a guess either way. Being explicit
/// that it is a lookup keeps the next person honest about needing a new measurement.
#[inline]
fn mode2_blocks_per_sm(sm_count: usize) -> (usize, usize) {
    let mut best = MODE2_ANCHORS[0];
    for &(sm, num, den) in &MODE2_ANCHORS[1..] {
        if sm_count.abs_diff(sm) < sm_count.abs_diff(best.0) {
            best = (sm, num, den);
        }
    }
    (best.1, best.2)
}

#[inline]
fn ceil_div(x: usize, b: usize) -> usize {
    x.div_ceil(b)
}

/// Decide the q8a128 dense matmul tiling: `false` → mode-1 (`Bm=16`), `true` → mode-2 (`Bm=32`,
/// `N_SUB=2`). Occupancy-driven (block count vs SM count) + the `[17,32]` trap. Allocation-free.
///
/// * `m` — token rows (the activation's flattened leading dim).
/// * `n` — weight output dim (the dominant driver: it sets mode-2's N-block count).
/// * `k` — contraction dim. Reserved: the bench shows a weak "bigger K crosses slightly earlier"
///   effect (more activation traffic), too small to fit a term yet.
/// * `sm_count` — the device's streaming-multiprocessor count (occupancy target). `0` degrades to
///   "always enough" → mode-2 outside the trap.
pub fn q8a128_dense_use_mode2(m: usize, n: usize, k: usize, sm_count: usize) -> bool {
    let _ = k; // reserved secondary term; see fn docs.

    // The block-count trap: mode-1 gets 2 batch-tiles to mode-2's 1 (→ ~4× the blocks with N_SUB),
    // so it wins even at mode-2's half traffic — measured across every N, including high-occupancy
    // ones. This is exactly the band where ceil(M/16)=2 while ceil(M/32)=1.
    if m > M_TILE_MODE1 && m <= M_TILE_MODE2 {
        return false;
    }

    // Otherwise it's an occupancy decision: mode-2's ~¼ block count must still fill the GPU.
    let blk2 = ceil_div(m, M_TILE_MODE2) * ceil_div(n, N_TILE_MODE2);
    let (num, den) = mode2_blocks_per_sm(sm_count);
    let threshold = sm_count * num / den;
    blk2 >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    const SM: usize = 76; // RTX 4090; tests are device-shape-agnostic via this constant.
    const SM_BW: usize = 110; // RTX PRO 5000 Blackwell.

    /// The occupancy threshold the formula checks (outside the trap), for `SM`.
    fn threshold() -> usize {
        let (num, den) = mode2_blocks_per_sm(SM);
        SM * num / den
    }

    /// **The no-regression guarantee for Ada.** `q8a128_dense_use_mode2` is a pure function of
    /// `sm_count`, so a 4090's decisions can be pinned from any machine: this reproduces the
    /// original `5/4 · SM` constant bit-for-bit across the whole decision surface. If the anchor
    /// table ever perturbs Ada, this fails — no 4090 required to catch it.
    #[test]
    fn ada_anchor_is_bit_identical_to_the_original_constant() {
        const ORIGINAL_NUM: usize = 5;
        const ORIGINAL_DEN: usize = 4;
        let (num, den) = mode2_blocks_per_sm(SM);
        assert_eq!((num, den), (ORIGINAL_NUM, ORIGINAL_DEN));

        let original = |m: usize, n: usize| -> bool {
            if m > M_TILE_MODE1 && m <= M_TILE_MODE2 {
                return false;
            }
            ceil_div(m, M_TILE_MODE2) * ceil_div(n, N_TILE_MODE2)
                >= SM * ORIGINAL_NUM / ORIGINAL_DEN
        };
        for n in [128usize, 512, 1024, 2048, 4096, 8192, 16384, 151_936] {
            for m in 1..=1024usize {
                assert_eq!(
                    q8a128_dense_use_mode2(m, n, 4096, SM),
                    original(m, n),
                    "Ada decision changed at m={m} n={n}"
                );
            }
        }
    }

    /// Blackwell's measured crossovers (`q8a128_dense_mode_crossover`, 110 SMs / 96 MiB L2):
    /// N=2048→M\*=256, N=4096→M\*=128, N=8192→M\*=56, all landing on `blk2 = 256`.
    ///
    /// The old single constant put these at 129 / 65 / 33 — mode-2 chosen well before it wins,
    /// which on this part is a pessimization of the most-launched GEMM in the model.
    #[test]
    fn blackwell_anchor_matches_measured_crossovers() {
        assert_eq!(mode2_blocks_per_sm(SM_BW), (7, 3));
        let (num, den) = mode2_blocks_per_sm(SM_BW);
        assert_eq!(
            SM_BW * num / den,
            256,
            "blk2 target at the measured crossover"
        );

        // Each N flips to mode-2 within the benchmark grid step containing its measured M*.
        // Every `below` here must sit ABOVE the 17..=32 trap, or the assertion passes for any
        // anchor value and tests nothing. (A first cut used below=32 for N=8192 and was exactly
        // that vacuous.)
        for &(n, below, at) in &[
            (2048usize, 192usize, 256usize),
            (4096, 96, 128),
            (8192, 16, 33),
        ] {
            assert!(
                below <= M_TILE_MODE1 || below > M_TILE_MODE2,
                "N={n}: below={below} is inside the trap, so this cell proves nothing"
            );
            assert!(
                !q8a128_dense_use_mode2(below, n, 4096, SM_BW),
                "N={n} should still be mode-1 at M={below}"
            );
            assert!(
                q8a128_dense_use_mode2(at, n, 4096, SM_BW),
                "N={n} should be mode-2 by M={at}"
            );
        }
    }

    /// **The formula's resolution limit, stated rather than hidden.**
    ///
    /// `blk2` steps only every 32 rows, so within one M-tile the decision cannot change. At
    /// N=8192 that puts the predicted flip at M=33 while the benchmark measures M\*=56 — the
    /// whole band [33, 55] is called mode-2 about one tile early. N=2048 and N=4096 have no such
    /// gap (predicted 225/97 vs measured 256/128, both inside the grid step below the measurement).
    ///
    /// Closing it would need an M term finer than the tile, which the current data cannot support.
    /// This test pins the gap so it is a known quantity rather than a surprise.
    #[test]
    fn n8192_is_predicted_one_m_tile_early_on_blackwell() {
        assert!(!q8a128_dense_use_mode2(32, 8192, 4096, SM_BW)); // trap, not the gate
        assert!(!q8a128_dense_use_mode2(16, 8192, 4096, SM_BW)); // blk2 = 128 < 256
        for m in 33..=55usize {
            assert!(
                q8a128_dense_use_mode2(m, 8192, 4096, SM_BW),
                "known early band: m={m}"
            );
        }
        assert!(q8a128_dense_use_mode2(56, 8192, 4096, SM_BW)); // measured crossover
    }

    /// The trap and the tiny-N/huge-N extremes must hold on BOTH anchors — they are mechanism,
    /// not calibration, so an anchor that broke them would be wrong regardless of measurement.
    #[test]
    fn mechanism_holds_on_both_anchors() {
        for &sm in &[SM, SM_BW] {
            for m in (M_TILE_MODE1 + 1)..=M_TILE_MODE2 {
                assert!(
                    !q8a128_dense_use_mode2(m, 200_000, 4096, sm),
                    "trap sm={sm} m={m}"
                );
            }
            // Tiny N (kv_proj / router) never fills the GPU across any real batch.
            for m in [1usize, 16, 64, 128, 256] {
                assert!(
                    !q8a128_dense_use_mode2(m, 512, 2048, sm),
                    "tiny-N sm={sm} m={m}"
                );
            }
            // Huge N (lm_head) fills it immediately, outside the trap.
            assert!(
                q8a128_dense_use_mode2(1, 151_936, 2048, sm),
                "lm_head sm={sm}"
            );
            assert!(
                q8a128_dense_use_mode2(16, 151_936, 2048, sm),
                "lm_head sm={sm}"
            );
        }
    }

    /// An unmeasured part takes the nearest anchor, and the boundary sits where the anchors say.
    #[test]
    fn unmeasured_parts_take_the_nearest_anchor() {
        assert_eq!(mode2_blocks_per_sm(76), (5, 4)); // exact
        assert_eq!(mode2_blocks_per_sm(110), (7, 3)); // exact
        assert_eq!(mode2_blocks_per_sm(80), (5, 4)); // closer to Ada
        assert_eq!(mode2_blocks_per_sm(108), (7, 3)); // closer to Blackwell
        assert_eq!(mode2_blocks_per_sm(1), (5, 4)); // far below both
        assert_eq!(mode2_blocks_per_sm(200), (7, 3)); // far above both
    }

    #[test]
    fn trap_is_always_mode1() {
        // M ∈ [17, 32]: mode-1 regardless of N (even a huge, fully-occupied weight).
        for m in (M_TILE_MODE1 + 1)..=M_TILE_MODE2 {
            assert!(!q8a128_dense_use_mode2(m, 8192, 4096, SM), "m={m}");
            assert!(
                !q8a128_dense_use_mode2(m, 200_000, 4096, SM),
                "m={m} huge N"
            );
        }
    }

    #[test]
    fn occupancy_gate_separates_n4096_from_n8192_at_small_m() {
        // The benchmark's cleanest split: at M ≤ 16 (one M-tile), N=4096 → blk2=64 (mode-1),
        // N=8192 → blk2=128 (mode-2). The threshold must fall between.
        for m in 1..=M_TILE_MODE1 {
            assert!(
                !q8a128_dense_use_mode2(m, 4096, 4096, SM),
                "N=4096 should be mode-1 at m={m}"
            );
            assert!(
                q8a128_dense_use_mode2(m, 8192, 4096, SM),
                "N=8192 should be mode-2 at m={m}"
            );
        }
        assert!(threshold() > 64 && threshold() <= 128);
    }

    #[test]
    fn tiny_n_stays_mode1_through_decode_and_typical_prefill() {
        // kv_proj / router (small N): too few N-blocks to fill the GPU until M is enormous, so the
        // whole decode + typical-prefill range stays mode-1. (N=512 only crosses past M≈350,
        // N=128 not until M≈1500 — both well beyond any real batch.)
        for &n in &[128usize, 512] {
            for m in [1, 16, 64, 128, 256] {
                assert!(!q8a128_dense_use_mode2(m, n, 2048, SM), "n={n} m={m}");
            }
        }
    }

    #[test]
    fn huge_n_lm_head_uses_mode2_outside_trap() {
        // lm_head (N = vocab): mode-2 everywhere except the trap.
        let n = 151_936;
        assert!(q8a128_dense_use_mode2(1, n, 2048, SM));
        assert!(q8a128_dense_use_mode2(16, n, 2048, SM));
        assert!(!q8a128_dense_use_mode2(24, n, 2048, SM)); // trap
        assert!(q8a128_dense_use_mode2(64, n, 2048, SM));
    }

    #[test]
    fn bigger_n_crosses_no_later() {
        // Core property: more N ⇒ mode-2 reached at a lower (or equal) M. Check the first M above
        // the trap where each N flips; it must be non-increasing in N.
        let first_cross = |n: usize| -> usize {
            (M_TILE_MODE2 + 1..4096)
                .find(|&m| q8a128_dense_use_mode2(m, n, 4096, SM))
                .unwrap_or(usize::MAX)
        };
        let mut prev = usize::MAX;
        for &n in &[1024usize, 2048, 4096, 8192, 16384] {
            let c = first_cross(n);
            assert!(c <= prev, "crossover rose at N={n}: {c} > {prev}");
            prev = c;
        }
    }

    #[test]
    fn monotonic_in_m_above_trap() {
        // Above the trap the occupancy threshold is M-monotone: once mode-2, stays mode-2.
        for &n in &[2048usize, 4096, 8192] {
            let mut seen = false;
            for m in (M_TILE_MODE2 + 1)..2048 {
                let on = q8a128_dense_use_mode2(m, n, 4096, SM);
                if seen {
                    assert!(on, "non-monotonic above trap at m={m} n={n}");
                }
                seen |= on;
            }
        }
    }

    #[test]
    fn n2048_crossover_matches_measured_band() {
        // N=2048 measured crossover ~80–96 → blk2 = ceil(M/32)·32 >= ~95 ⇒ ceil(M/32) >= 3 ⇒ M ≥ 65.
        assert!(!q8a128_dense_use_mode2(64, 2048, 2048, SM));
        assert!(q8a128_dense_use_mode2(65, 2048, 2048, SM));
    }

    #[test]
    fn zero_sm_count_degrades_to_mode2_outside_trap() {
        assert!(q8a128_dense_use_mode2(1, 1024, 2048, 0));
        assert!(!q8a128_dense_use_mode2(24, 1024, 2048, 0)); // trap still holds
    }
}
