//! Chooses the grid axis order and row-tile height of a grouped expert launch.
//!
//! A grouped launch tiles its rows along two axes, and whichever runs fast decides
//! whether the activation is streamed from DRAM once or once per pass. When the
//! activation and the pass's output columns cannot both stay in L2, row tiles run
//! fast; when they can, token tiles run fast, which keeps consecutive blocks on
//! the same weight rows. Both orders compute identical bits, so the choice is
//! purely a performance band. The token-tile mode of an int8 grouped launch is
//! picked from the average rows per expert, because the tile width sets how often
//! each expert's weights are reused.

use super::*;

/// Grid axis order for a grouped launch (see `quantized_matmul_grouped_entry`):
/// row-tiles-fast whenever the stacked activation cannot stay L2-resident
/// through one row-tile pass — token-tiles-fast would then re-stream the whole
/// activation from DRAM once per pass (measured 6× DRAM traffic amplification
/// at prefill scale). A pass streams the activation AND the output columns of
/// its row tile (`out_pass_bytes`: every row × the row tile's width), so both
/// count against the L2: with the 128-row prefill tile a 7500-token `down`
/// pass writes 38 MB of F32 beside its 48 MB activation, and token-fast measured
/// 2.07 ms against row-fast's 1.36 ms (`bench_prefill_grouped`, ncu: L2 hit 54%,
/// the activation re-read every pass). When both fit comfortably, residency is
/// free under either order and token-tiles-fast keeps consecutive blocks on the
/// same weight rows instead. The third-of-L2 threshold leaves the rest for the
/// weight stream and the replacement policy's slack; it classifies every band
/// `bench_prefill_grouped` and `moe_layer_gemm_bench` measured (96 MiB L2):
/// token-fast wins at decode (2.7 MB, +5%) and at a 2048-token 64-row-tile
/// `down` (+5%); row-fast wins every 128-row-tile prefill and everything from
/// 100 MB up (+20..50%). Mis-choosing costs ~5% on the token-fast side of the
/// line and up to 50% on the row-fast side, so the threshold sits well below the
/// row-fast danger zone. Both orders are bit-identical (schedule only) — the
/// crossover is a pure performance band.
pub(crate) fn grouped_grid_row_fast(
    act_bytes: usize,
    out_pass_bytes: usize,
    device: &CudaDevice,
) -> bool {
    (act_bytes + out_pass_bytes) * 3 > cached_l2_bytes(device)
}

/// Rows per block of the FP grouped kernels.
pub(super) const GROUPED_FP_ROW_TILE: usize = 32;

/// Rows per block of the grouped int8 kernels at token-tile mode `n_sub` — the
/// row-tile height `run_grouped_quantized_matmul` sizes its grid with (mode-2: the
/// 32-row tile; mode-4: 128 rows, RN = 4; mode-8: 64 rows, RN = 2).
pub(crate) fn grouped_int8_row_tile(n_sub: usize) -> usize {
    match n_sub {
        4 => 128,
        8 => 64,
        _ => 32,
    }
}

/// [`grouped_grid_row_fast`] for a grouped int8 launch of `total_batch` rows over `ncols`
/// (K) and `nrows` (N) at token-tile mode `n_sub`: q8a128 activations are ~1 B/elem (int8
/// qs + per-128 scales), the output F32.
pub(crate) fn grouped_int8_row_fast(
    total_batch: usize,
    ncols: usize,
    nrows: usize,
    n_sub: usize,
    device: &CudaDevice,
) -> bool {
    let row_tile = grouped_int8_row_tile(n_sub).min(nrows);
    grouped_grid_row_fast(total_batch * ncols, total_batch * row_tile * 4, device)
}

/// INT8 grouped matmul on PRE-QUANTIZED q8a128 activations
/// (`block_q8a128[total_batch][K/128]`). The caller quantizes once (amortized
/// across gate/up). Tensor-core only (the q8a128 input has no FP fallback);
/// output is F32. Routes through the unified `run_grouped_quantized_matmul`
/// with ytype = Q8A128. The grouped/MoE path has no mode-2 kernel, so it always
/// runs mode-1 (Q8A128V) regardless of M. Internal building block for the `Int8`
/// arm of [`grouped_qmatmul`].
/// Token width of one mode-2 (`Bm=32`, `N_SUB=2`) grouped-GEMM expert tile — the
/// decode regime's width, where 32 is already the full reuse win, and the width
/// device-table callers that do not choose a mode build at. The expert cache's live
/// dispatch and the host tile builder (`grouped_matmul_gemx_q8a128_with_mode`) choose
/// per launch instead ([`grouped_int8_n_sub`]: `16·n_sub`, up to `Bm=128` for
/// prefill-scale rows-per-expert). A builder/launch width divergence silently
/// mis-strides the kernel's batch slices, so each launch site pairs its table width
/// with its `n_sub`.
pub const GROUPED_GEMM_TILE_W: usize = 32;

/// The token-tile mode (`n_sub`; tile width `16·n_sub`) for a grouped int8 launch
/// whose experts average `avg_rows` rows each — shared by the host tile builder and
/// every device-table launch, so both pick the same width for the same load.
///
/// The grouped int8 kernel loads and dequants each expert weight chunk ONCE per tile
/// and sweeps the m16n8k32 core across the tile's 16-row sub-tiles, so the tile width
/// IS the weight-reuse factor: at decode's 1–32 rows/expert the 32-wide mode-2 tile is
/// already optimal (a partial tile costs the same weight traffic), but at PREFILL's
/// ~100–300 rows/expert it re-streams and re-dequants every expert 4×+ per launch —
/// measured as the flat ~0.87 ms/token marginal prefill cost. Wide modes exist for the
/// KO formats only: a launch whose `dtypes` (every weight format sharing the tile table)
/// are not all KO runs mode-2.
///
/// The wide modes are also tall (mode-4: Bm 64 × 128 rows; mode-8: Bm 128 × 64 rows —
/// see `grouped_int8_row_tile`), and which one wins depends on the fold, so the bands are
/// per format. Device time, `bench_prefill_grouped` (RTX PRO 5000):
/// - **Affine KO** (Q2..Q8_KO — one fold per 128-K tile): mode-4 wins at every prefill
///   width — Flash-Next Q4_KO experts at 40 / 80 / 146 rows/expert, gate/up 440 / 798 /
///   1291 µs against mode-8's 624 / 865 / 1584, `down` 583 / 889 / 1461 against 658 /
///   954 / 1724.
/// - **MXFP4_KO** (a fold per 32-K sub, four times the fold work and registers): mode-4
///   wins at 48 rows/expert (DeepSeek-V4-Flash gate/up 3.99 ms against 4.53), mode-8 from
///   64 up (91 rows: 5.90 against 7.33 ms; 192 rows: 11.65 against 13.21 ms).
pub fn grouped_int8_n_sub(avg_rows: usize, dtypes: &[GgmlDType]) -> usize {
    if avg_rows < 32 || dtypes.is_empty() || !dtypes.iter().all(|d| d.is_ko()) {
        2
    } else if avg_rows >= 64 && dtypes.contains(&GgmlDType::MXFP4_KO) {
        8
    } else {
        4
    }
}
