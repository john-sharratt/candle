//! QSA — Qwen Sparse Attention block selection, reference implementation.
//!
//! Semantics from `qwen4exp.cpp` `build_qsa_top_k` and
//! `llama_memory_hybrid_idx_context::set_input_qsa`
//! (`docs/qwen38_flash_next.md` §12.5):
//!
//! - The indexer caches **raw** keys (`index_k_proj · x`); pooling (mean over
//!   `ratio` cells), RMS norm and rotation are applied at read time, with the
//!   block's **first position** as the rope position.
//! - Score is `ReLU(q · k̄)` summed over the query heads — no scale; only the
//!   rank matters.
//! - A complete block's cells all inherit the block score. An **incomplete**
//!   block is never scored (−inf) — except the query's own trailing tail
//!   (`positions ≥ (q+1)/r·r`), which is forced in with a large positive
//!   bias. The token-level causal mask then drops future cells, so a "forced"
//!   future cell never survives.
//! - Per query token, the best `top_k + ratio − 1` cells (2051) are kept and
//!   the rest masked to −inf. At `n_kv ≤ 2051` selection is the identity, so
//!   this module returns `None` and the layer attends densely — the same
//!   arithmetic, skipping the scoring launch.

use candle::{LiveTensor, Result, Tensor};

use super::config::IndexerConfig;
use super::qsa_select::{
    entry_block, entry_cells, selected_width, selection_entries, RowSelection,
};
use crate::models::qwen35::attention::RopeTables;

/// The QSA indexer's per-layer weights.
#[derive(Debug, Clone)]
pub struct IndexerWeights {
    /// `[n_heads · head_dim, hidden]`.
    pub q_proj: Tensor,
    /// `[head_dim, hidden]` — one shared key head.
    pub k_proj: Tensor,
    /// `[head_dim]` each.
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

/// One sequence's carried index cache: the raw (unpooled, unnormed, unroped)
/// indexer keys, appended like KV.
#[derive(Debug, Default)]
pub struct IndexState {
    /// `[total, head_dim]`.
    pub keys: Option<Tensor>,
}

impl IndexState {
    pub fn empty() -> Self {
        Self { keys: None }
    }

    pub fn seq_len(&self) -> usize {
        self.keys
            .as_ref()
            .map(|k| k.dim(0).unwrap_or(0))
            .unwrap_or(0)
    }
}

/// RMS norm over the last axis, as the indexer applies it to both its pooled
/// block keys and its queries.
///
/// The engine's index cache calls this same function rather than the fused
/// `candle_nn::ops::rms_norm`: the two agree to rounding, and rounding is
/// exactly what decides a rank at the selection's cut. Both sides of an
/// oracle-vs-engine comparison have to normalise the same way for the
/// comparison to mean anything, and neither tensor is large enough for the
/// fused kernel to matter.
pub(crate) fn rms_norm_last<'w>(
    x: &LiveTensor<'w>,
    weight: &Tensor,
    eps: f64,
) -> Result<LiveTensor<'w>> {
    let ms = x.sqr()?.mean_keepdim(candle::D::Minus1)?;
    x.broadcast_div(&(ms + eps)?.sqrt()?)?.broadcast_mul(weight)
}

/// Score this segment's queries against the (updated) index cache and build
/// the additive selection mask.
///
/// `x` is the sequence's `[T, hidden]` block input (the same rows the
/// attention projections read), `past` the cells already cached. Appends the
/// segment's raw keys to `state`, then returns `Some([T, past+T])` with `0`
/// on selected cells and `−inf` elsewhere — or `None` when every visible cell
/// is selected anyway.
#[allow(clippy::too_many_arguments)]
pub fn qsa_selection_mask(
    x: &Tensor,
    w: &IndexerWeights,
    state: &mut IndexState,
    rope: &RopeTables,
    ratio: usize,
    cfg: &IndexerConfig,
    rms_eps: f64,
) -> Result<Option<Tensor>> {
    let (t, _hidden) = x.dims2()?;
    let d = cfg.head_dim;
    let past = state.seq_len();
    let total = past + t;

    // Cache the raw keys first — selection and caching share the append.
    let k_raw = x.matmul(&w.k_proj.t()?)?; // [T, d]
    let all_keys = match &state.keys {
        Some(prev) => Tensor::cat(&[prev, &k_raw], 0)?,
        None => k_raw,
    };
    state.keys = Some(all_keys.clone());

    // Whole blocks plus the tail: the reference's selected width.
    let width = selected_width(cfg.top_k, ratio);
    if total <= width {
        return Ok(None);
    }

    let r = ratio;
    let n_complete = total / r; // blocks with all `r` cells present

    // Pooled block keys: mean over each complete block's raw cells, then
    // norm, then rope at the block's first position. Incomplete blocks are
    // never scored, so only the complete ones are pooled.
    let pooled = all_keys
        .narrow(0, 0, n_complete * r)?
        .reshape((n_complete, r, d))?
        .mean(1)?; // [n_complete, d]
    let pooled = rms_norm_last(&pooled, &w.k_norm, rms_eps)?;
    // Each block key rotates at its block's FIRST position, `b·r` — a
    // stride-`r` walk, hence the explicit-positions rope form.
    let pooled = rope.apply_at_positions(
        &pooled.reshape((n_complete, 1, d))?,
        &(0..n_complete).map(|b| b * r).collect::<Vec<_>>(),
    )?;
    let pooled = pooled.reshape((n_complete, d))?;

    // Queries: [T, H, d], normed, roped at token positions.
    let q = x.matmul(&w.q_proj.t()?)?.reshape((t, cfg.n_heads, d))?;
    let q = rms_norm_last(&q, &w.q_norm, rms_eps)?;
    let q = rope.apply(&q, past)?; // positions past..past+t

    // ReLU(q·k̄) summed over heads → [T, n_complete].
    let scores = q
        .reshape((t * cfg.n_heads, d))?
        .matmul(&pooled.t()?)?
        .relu()?
        .reshape((t, cfg.n_heads, n_complete))?
        .sum(1)?;
    let scores: Vec<Vec<f32>> = scores.to_vec2()?;

    // Per query token: the shared selection (`qsa_select`), expanded into this
    // row's additive mask. A row too short for the budget attends everything
    // visible — the same identity the whole-segment early return above takes,
    // reached per row because a segment may straddle the threshold.
    let mut mask = vec![f32::NEG_INFINITY; t * total];
    let mut entries: Vec<u32> = Vec::new();
    for (i, row) in scores.iter().enumerate() {
        let qpos = past + i;
        match selection_entries(row, qpos, r, cfg.top_k, &mut entries) {
            RowSelection::Dense => {
                for j in 0..=qpos {
                    mask[i * total + j] = 0.0;
                }
            }
            RowSelection::Entries(_) => {
                for &e in &entries {
                    let base = entry_block(e) * r;
                    for c in 0..entry_cells(e) {
                        mask[i * total + base + c] = 0.0;
                    }
                }
            }
        }
    }
    Ok(Some(Tensor::from_vec(mask, (t, total), x.device())?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn dev() -> Device {
        Device::Cpu
    }

    fn lcg_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed;
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::from_vec(vals, shape, dev).unwrap()
    }

    fn tiny(hidden: usize, cfg: &IndexerConfig, dev: &Device) -> IndexerWeights {
        IndexerWeights {
            q_proj: lcg_tensor(&[cfg.n_heads * cfg.head_dim, hidden], 61, dev)
                .affine(0.3, 0.)
                .unwrap(),
            k_proj: lcg_tensor(&[cfg.head_dim, hidden], 62, dev)
                .affine(0.3, 0.)
                .unwrap(),
            q_norm: lcg_tensor(&[cfg.head_dim], 63, dev)
                .affine(0.1, 1.0)
                .unwrap(),
            k_norm: lcg_tensor(&[cfg.head_dim], 64, dev)
                .affine(0.1, 1.0)
                .unwrap(),
        }
    }

    #[test]
    fn short_context_is_dense() {
        let dev = dev();
        let cfg = IndexerConfig {
            n_heads: 2,
            head_dim: 8,
            top_k: 8,
        };
        let w = tiny(6, &cfg, &dev);
        let rope = RopeTables::new(4, 1e6, 64, &dev).unwrap();
        let mut st = IndexState::empty();
        let x = lcg_tensor(&[5, 6], 65, &dev);
        // 5 ≤ 8+4−1: dense, but the keys are still cached.
        let m = qsa_selection_mask(&x, &w, &mut st, &rope, 4, &cfg, 1e-6).unwrap();
        assert!(m.is_none());
        assert_eq!(st.seq_len(), 5);
    }

    #[test]
    fn selection_keeps_the_budget_and_always_the_tail() {
        let dev = dev();
        let cfg = IndexerConfig {
            n_heads: 2,
            head_dim: 8,
            top_k: 4,
        };
        let r = 4usize;
        let width = cfg.top_k + r - 1; // 7
        let w = tiny(6, &cfg, &dev);
        let rope = RopeTables::new(4, 1e6, 128, &dev).unwrap();
        let mut st = IndexState::empty();
        let t = 26usize;
        let x = lcg_tensor(&[t, 6], 66, &dev);
        let m = qsa_selection_mask(&x, &w, &mut st, &rope, r, &cfg, 1e-6)
            .unwrap()
            .expect("26 > 7: selection must engage");
        assert_eq!(m.dims(), &[t, t]);
        let rows: Vec<Vec<f32>> = m.to_vec2().unwrap();
        for (i, row) in rows.iter().enumerate() {
            let selected: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|(_, &v)| v == 0.0)
                .map(|(j, _)| j)
                .collect();
            // Never more than the budget, never a future cell.
            assert!(
                selected.len() <= width,
                "row {i}: {} selected",
                selected.len()
            );
            assert!(
                selected.iter().all(|&j| j <= i),
                "row {i} selected a future cell"
            );
            // The query's own tail cells are always in.
            let tail_start = (i + 1) / r * r;
            for (j, &v) in row.iter().enumerate().take(i + 1).skip(tail_start) {
                assert!(v == 0.0, "row {i}: tail cell {j} not selected");
            }
            // Early rows with few candidates select everything visible.
            if i < width {
                assert_eq!(selected.len(), i + 1, "row {i} under-selected");
            }
        }
    }

    #[test]
    fn selected_cells_come_in_whole_blocks_when_the_tail_is_full() {
        // The +r−1 slack accommodates the tail, so the cut lands on a block
        // boundary exactly when the tail holds its full r−1 cells — a query
        // at position ≡ r−2 (mod r). There, top_k is a whole number of
        // blocks and every selected complete block is selected in full. (At
        // other phases the fixed width splits the boundary block; the
        // reference behaves the same, its "cuts on a block boundary" comment
        // holding at this phase.)
        let dev = dev();
        let cfg = IndexerConfig {
            n_heads: 2,
            head_dim: 8,
            top_k: 8,
        };
        let r = 4usize;
        let w = tiny(6, &cfg, &dev);
        let rope = RopeTables::new(4, 1e6, 256, &dev).unwrap();
        let mut st = IndexState::empty();
        let t = 40usize;
        let x = lcg_tensor(&[t, 6], 67, &dev);
        let m = qsa_selection_mask(&x, &w, &mut st, &rope, r, &cfg, 1e-6)
            .unwrap()
            .unwrap();
        let rows: Vec<Vec<f32>> = m.to_vec2().unwrap();
        // qpos = 38 ≡ r−2 (mod 4): tail = {36, 37, 38}, three cells, so the
        // block budget is exactly top_k = two whole blocks.
        let qpos = 38usize;
        let row = &rows[qpos];
        let tail_start = (qpos + 1) / r * r; // 36
        for (j, &v) in row.iter().enumerate().take(qpos + 1).skip(tail_start) {
            assert_eq!(v, 0.0, "tail cell {j} not selected");
        }
        let mut whole_blocks = 0usize;
        for b in 0..tail_start / r {
            let cells: Vec<bool> = (b * r..(b + 1) * r).map(|j| row[j] == 0.0).collect();
            let any = cells.iter().any(|&c| c);
            let all = cells.iter().all(|&c| c);
            assert!(
                !any || all,
                "block {b} partially selected at the full-tail phase: {cells:?}"
            );
            whole_blocks += usize::from(all);
        }
        assert_eq!(
            whole_blocks,
            cfg.top_k / r,
            "block budget not spent in whole blocks"
        );
    }

    #[test]
    fn decode_after_prefill_scores_the_same_cache() {
        // Append-then-select across segments: a one-token decode step against
        // a prefilled cache must see all past cells as candidates.
        let dev = dev();
        let cfg = IndexerConfig {
            n_heads: 2,
            head_dim: 8,
            top_k: 4,
        };
        let r = 4usize;
        let w = tiny(6, &cfg, &dev);
        let rope = RopeTables::new(4, 1e6, 128, &dev).unwrap();
        let mut st = IndexState::empty();
        let x = lcg_tensor(&[20, 6], 68, &dev);
        let _ = qsa_selection_mask(&x, &w, &mut st, &rope, r, &cfg, 1e-6).unwrap();
        assert_eq!(st.seq_len(), 20);
        let x1 = lcg_tensor(&[1, 6], 69, &dev);
        let m = qsa_selection_mask(&x1, &w, &mut st, &rope, r, &cfg, 1e-6)
            .unwrap()
            .expect("21 > 7");
        assert_eq!(m.dims(), &[1, 21]);
        assert_eq!(st.seq_len(), 21);
        let row: Vec<f32> = m.to_vec2::<f32>().unwrap().remove(0);
        let n_sel = row.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(n_sel, cfg.top_k + r - 1);
    }
}
