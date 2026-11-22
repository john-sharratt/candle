//! Bounds the scratch a scoring pass may carve, so a wave's plan can price it
//! before the forward runs.
//!
//! Scores are tiled to a byte budget. A live tail wide enough to cross the
//! crossover is scored by one GEMM rather than by the paged scorer, and the
//! crossover is counted in cells because the paged route's time follows cells.
//! The bounds mirror the two routes exactly, so a price is never short of what
//! a layer carves, whether or not the cache it describes exists yet.

use super::*;

/// Rows of scores computed in one tile.
///
/// The score matrix is `[rows × heads, blocks]`, which at depth is the largest
/// thing in the selection path — 4 heads × 4 bytes × one block per 4 tokens,
/// per query. Tiling the query rows bounds it; the tile is chosen so a tile's
/// scores stay under this many bytes.
pub(super) const SCORE_TILE_BYTES: usize = 128 << 20;

/// The live-tail cells — query rows × live-tail blocks — from which the tail is
/// scored by cuBLAS rather than by the paged scorer.
///
/// Below it a span's scores come from one paged-scorer launch over its pages
/// and its live tail, every key rotated as it loads. From it up — a prefill
/// chunk over a long unplaced turn — the live tail goes to cuBLAS instead,
/// rotated once into a scratch buffer that lives for the launch.
///
/// **A cell count, not a row count**, because that is what the paged route's
/// time follows: `tests/qsa_score_rot_harness.rs`'s sweep puts it at ~0.78 ms
/// per 2²⁵ cells whether they are 4096 rows over 8K tokens or 128 over 256K.
/// The GEMM has a fixed cost the paged route does not — the scratch and its
/// rotation, ~0.6 ms once the tail is wide. At 2²⁵ cells it is ahead by about
/// 0.1 ms in three of the four splits the sweep measured and behind by 0.08 in
/// the fourth; at twice that it wins by 1.4–1.8×. Below it the GEMM is never
/// more than 60 µs ahead.
pub const GEMM_TAIL_MIN_CELLS: usize = 1 << 25;

/// How a span's live tail is scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailRoute {
    /// As one more page of the paged scorer, rotated on load.
    Paged,
    /// Rotated once into a scratch buffer, then one cuBLAS GEMM.
    Gemm,
}

impl TailRoute {
    /// The route for `rows` query rows over `live_cols` live-tail blocks —
    /// [`GEMM_TAIL_MIN_CELLS`].
    pub fn for_span(rows: usize, live_cols: usize) -> Self {
        if rows.saturating_mul(live_cols) < GEMM_TAIL_MIN_CELLS {
            Self::Paged
        } else {
            Self::Gemm
        }
    }
}

/// The query side of a scoring bound: `t` query rows whose deepest sees
/// `cand_max` candidate blocks, scored by `heads` heads of width `d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoreRows {
    pub t: usize,
    pub cand_max: usize,
    pub heads: usize,
    pub d: usize,
}

/// The most bytes [`IndexCache::score_rows`] carves for `rows` over a cache whose
/// placed pages span `page_cols` candidate columns in `page_chunks()` descriptor
/// chunks and whose live tail holds `live_blocks` blocks once the wave's appends
/// land — each carve rounded up to `align`, so the sum is a bound.
///
/// One formula for a cache that exists ([`IndexCache::score_bound`]) and for one
/// that does not yet ([`fresh_score_bound`]), so a wave priced before its
/// sequence has run is priced by the same arithmetic as the wave itself. Mirrors
/// the two routes: the paged scorer's page, offset and count tables, and — when
/// the live tail goes to cuBLAS — the rotated tail, its page table and the one
/// tile's scoring product every tile is written into. `page_chunks` is asked only
/// when the paged scorer runs.
pub fn score_bound_for(
    page_cols: usize,
    page_chunks: impl FnOnce() -> Result<usize>,
    live_blocks: usize,
    rows: ScoreRows,
    align: usize,
) -> Result<usize> {
    use candle_kernels::simple::qsa_score_paged::PAGE_WORDS;

    let ScoreRows {
        t,
        cand_max,
        heads,
        d,
    } = rows;
    let a = |b: usize| b.div_ceil(align) * align;
    let live_cols = cand_max.saturating_sub(page_cols);
    let tail_in_paged = live_cols > 0 && TailRoute::for_span(t, live_cols) == TailRoute::Paged;
    let paged_cols = if tail_in_paged { cand_max } else { page_cols };
    let mut bytes = 0usize;
    if paged_cols > 0 {
        let mut chunks = page_chunks()?;
        if tail_in_paged {
            chunks += live_blocks.div_ceil(PAGE_BLOCKS);
        }
        bytes += a(chunks * PAGE_WORDS * 8) + a((chunks + 1) * 4) + a(t * 4);
    }
    if !tail_in_paged && live_cols > 0 {
        bytes += a(live_cols.div_ceil(PAGE_BLOCKS) * 8) + a(live_cols * d * 4);
        // One tile's product, which every tile is written into.
        let per_tile = (SCORE_TILE_BYTES / (heads * live_cols * 4)).clamp(1, t);
        bytes += a(per_tile * heads * live_cols * 4);
    }
    Ok(bytes)
}

/// [`score_bound_for`] over a cache with no placed pages whose live tail holds
/// `live_blocks` blocks once the wave's appends land — the bound for a sequence
/// that forwards its whole prefix and has no cache yet to ask.
pub fn fresh_score_bound(live_blocks: usize, rows: ScoreRows, align: usize) -> usize {
    score_bound_for(0, || Ok(0), live_blocks, rows, align)
        .expect("a cache with no placed pages has no descriptor to fail on")
}
