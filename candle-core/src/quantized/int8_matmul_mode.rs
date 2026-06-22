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
//! else:                        mode-2  ⇔  ceil(M/32)·ceil(N/64) >= (5/4)·SM_count
//! ```
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

/// Occupancy target, as a fraction of SM count, that mode-2 must reach to win: `blk2 >= 5/4 · SMs`.
/// Fit to the benchmark, which separates `blk2 = 64` (mode-1) from `blk2 = 128` (mode-2) on a
/// 76-SM part. Expressed over SM count so it tracks the GPU rather than hard-coding the 4090.
const MODE2_BLOCKS_PER_SM_NUM: usize = 5;
const MODE2_BLOCKS_PER_SM_DEN: usize = 4;

#[inline]
fn ceil_div(x: usize, b: usize) -> usize {
    (x + b - 1) / b
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
    let threshold = sm_count * MODE2_BLOCKS_PER_SM_NUM / MODE2_BLOCKS_PER_SM_DEN;
    blk2 >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    const SM: usize = 76; // RTX 4090; tests are device-shape-agnostic via this constant.

    /// `blk2 >= ceil(5/4 · SM)` — the occupancy threshold the formula checks (outside the trap).
    fn threshold() -> usize {
        SM * MODE2_BLOCKS_PER_SM_NUM / MODE2_BLOCKS_PER_SM_DEN
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
