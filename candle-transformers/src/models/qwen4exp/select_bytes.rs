//! What one attention layer's sparse selection carves on its phase — the
//! figure the wave plan prices as `WaveBuffer::QsaSelection`.
//!
//! The plan cannot derive it from the wave's width: the score buffer and the
//! scoring products are as wide as the candidate blocks the deepest row can
//! see, so they grow with context depth, and the page tables follow how a
//! sequence's index was assembled. So the model states it, from the caches
//! the wave is about to score against, mirroring [`select_layer`]'s carves in
//! order. Every carve is rounded up to the bump's alignment, which makes the
//! sum a bound: the phase is never priced short of what it carves, and an
//! overrun fails the wave by name rather than reaching the pool.
//!
//! [`select_layer`]: super::indexer::select_layer

use std::collections::HashMap;

use candle::Result;
use candle_kernels::simple::qsa_topk::SPLIT_KEYS;

use super::config::IndexerConfig;
use super::indexer::{selection_engages, selection_stride, widest_candidates, IndexCache};
use crate::models::delta_net::SeqSpan;

/// The bump alignment every carve starts on.
pub const CARVE_ALIGN: usize = 256;

/// The most bytes [`super::indexer::select_layer`] carves on its ticket's arena
/// for one KV layer `kv` selecting at `ratio` over `spans` — the wave's rows,
/// `offsets[i]` the first position of `spans[i]`.
pub fn select_layer_bytes(
    kv: usize,
    ratio: usize,
    spans: &[SeqSpan],
    offsets: &[usize],
    idx_map: &HashMap<usize, Vec<IndexCache>>,
    cfg: &IndexerConfig,
) -> Result<usize> {
    if ratio == 0 {
        return Ok(0);
    }
    let a = |b: usize| b.div_ceil(CARVE_ALIGN) * CARVE_ALIGN;
    let (h, d) = (cfg.n_heads, cfg.head_dim);
    let rows: usize = spans.iter().map(|s| s.len).sum();

    // The raw index keys, every wave. The append's job and carry tables go
    // through the device's staging scratch, not this span.
    let mut bytes = a(rows * d * 4);

    let engages = spans
        .iter()
        .zip(offsets)
        .any(|(s, &off)| selection_engages(off + s.len, cfg, ratio));
    if !engages {
        return Ok(bytes);
    }

    // The selection table — its rows as wide as the deepest row's stratified
    // selection can be, through the same sizing the table itself uses.
    let stride = selection_stride(
        ratio,
        cfg.top_k,
        widest_candidates(spans, offsets, idx_map, kv, ratio),
        &cfg.strata,
    )?;
    bytes += a(rows * stride * 4) + a(rows * 4);
    // The selection kernel's split scratch, a fixed size.
    bytes += a(SPLIT_KEYS * 8);
    // The queries: the projection, the norm's chain (the square, the mean, the
    // epsilon, the root, the divide, the gain), the rotation's position and
    // rung tables, and the rotated rows.
    let q = a(rows * h * d * 4);
    let per_head = a(rows * h * 4);
    bytes += 4 * q + 3 * per_head + 2 * a(rows * 4) + q;

    // The score buffer, as wide as the deepest row's candidates, and every
    // span's scoring.
    let mut widest = 1usize;
    let mut needs_pages = false;
    let mut prefixes = 0usize;
    for (s, &off) in spans.iter().zip(offsets) {
        let Some(cache) = idx_map.get(&s.seq).and_then(|c| c.get(kv)) else {
            continue;
        };
        let last = off + s.len;
        let cand = cache.candidates_at(last.saturating_sub(1), ratio);
        widest = widest.max(cand);
        bytes += cache.score_bound(s.len, cand, s.len / ratio + 1, h, d, CARVE_ALIGN)?;
        needs_pages |= cache.has_pages();
        prefixes += cache.page_prefixes().len();
    }
    bytes += a(rows * widest * 4);
    // The selection kernel's packed row metadata: candidates, position, tail
    // and prompt span.
    bytes += a(4 * rows * 4);
    // The page layout the attention walks, when any sequence holds a prefix.
    if needs_pages {
        // A sequence without a cache contributes the degenerate two-word layout.
        bytes += a((prefixes + 2 * spans.len()) * 4) + a(rows * 2 * 4);
    }
    Ok(bytes)
}
