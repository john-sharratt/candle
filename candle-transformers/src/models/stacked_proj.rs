//! Running a group of input projections that share one operand, and splitting
//! the result without a copy per part.
//!
//! Several projections in a block are the same contraction against the same
//! activation, differing only in output width — a DeltaNet layer's `[Q|K|V]`,
//! `z`, `β` and `α`, or an attention block's `q`, `k` and `v`. A loader that can
//! row-concatenate their weights turns N GEMMs into one.
//!
//! # The split is the whole problem, and it has exactly one good answer
//!
//! A stacked weight's output is **token-major**: `[t, Σrows]`, with each token's
//! parts adjacent and every part row-strided down the buffer. The consumers all
//! index by their own width, so each part has to be made dense.
//!
//! Three ways to do that, and the arithmetic of the first two is why this module
//! looks the way it does:
//!
//! * **One `contiguous()` per part.** N allocations and N copy launches. Tried,
//!   measured: `dn:proj` came out flat (106.4 → 107.5 ms) because the copies ate
//!   the saved GEMM launches exactly, and attention lost outright
//!   (`prefill:qkv_proj` 22.1 → 25.6 ms).
//! * **Don't stack at all.** Separate GEMMs write separate wave-arena
//!   allocations, which a bump allocator already lays down adjacent and dense —
//!   zero copies, but N launches, and N narrower GEMMs.
//! * **Stack, then split with ONE ragged scatter.** `rows_scatter` takes a
//!   descriptor table of `{src, src row stride, dst, dst row stride, rows,
//!   words}` and runs every copy in a single launch (hot-path invariant 2b), and
//!   its `row_geometry` accepts any row stride so long as the channel stride is
//!   1 — which is exactly what a narrowed view of a stacked output is. The
//!   destination is **one** bump allocation the parts carve up.
//!
//! The third is what this does: **one GEMM launch, one copy launch, one
//! allocation**, against the unstacked path's N GEMMs and N allocations. The
//! bytes copied are unchanged from the per-part form — the win is launches and
//! allocations, which is what costs at decode, where the parts are a handful of
//! rows and the GEMM does almost no work.
//!
//! # Why a *list* of weights, not one
//!
//! Stacking is a byte append over the GGUF block layout, so it must happen
//! before `QMatMul::from_qtensor_with_mode` repacks to the lane-major KO twin
//! (`QTensor::concat_rows_cuda` refuses a KO input by name). A loader applying a
//! *per-tensor* narrowing schedule therefore cannot stack the tensors it narrows
//! differently: qwen35's streaming schedule narrows `attn_qkv` to `Q4_KO` and
//! leaves `attn_gate`, `ssm_beta` and `ssm_alpha` alone, and one stacked weight
//! takes one target. So the packing is the loader's decision, and this is the
//! one code path that serves either — not a fallback and not a feature flag. A
//! weight holding exactly one part hands it over untouched, so the unstacked
//! case pays nothing for the generality and issues no scatter at all.

use candle::{DType, Device, LiveTensor, Result};

use crate::models::latent_moe::scatter::{rows_scatter_inline, RowRun};
use crate::models::quantized_matmul::QMatMul;

/// Project `x` through `weights` and split the result into parts of `widths`.
///
/// `weights` covers `widths` in order and each weight must end on a part
/// boundary — one weight per part (unstacked), one weight for all of them
/// (fully stacked), or any grouping between.
pub fn project_grouped<'w>(
    x: &LiveTensor<'w>,
    weights: &[QMatMul],
    widths: &[usize],
    out_dtype: DType,
    what: &str,
) -> Result<Vec<LiveTensor<'w>>> {
    if weights.is_empty() {
        candle::bail!("{what}: no projection weights");
    }
    let mut outs = Vec::with_capacity(weights.len());
    for w in weights {
        outs.push(w.forward_live_as(x, out_dtype)?);
    }
    split_group(outs, widths, what)
}

/// The splitting half of [`project_grouped`], for callers that produce the
/// group's outputs themselves.
///
/// The attention block is one: its three projections share a **pre-quantized**
/// int8 activation, so they go through `forward_dynamic` rather than
/// `forward_live_as` and the outputs arrive already computed. The split is the
/// same either way, and there is one definition of it so the two cannot drift.
pub fn split_group<'w>(
    outs: Vec<LiveTensor<'w>>,
    widths: &[usize],
    what: &str,
) -> Result<Vec<LiveTensor<'w>>> {
    if outs.is_empty() {
        candle::bail!("{what}: no projection outputs");
    }
    // **The output axis is the LAST one, not dimension 1.** These come back at
    // whatever rank their operand had — `[rows, out]` from a flat activation,
    // `[batch, seq, out]` from a batched one — so splitting on dimension 1 would
    // cut a batched projection along its *sequence*. Caught rather than assumed:
    // the width check below is what turned that into "the projection group is
    // 134 wide but its parts need 13312" instead of three tensors of plausible
    // shape holding the wrong rows.
    let axis = outs[0].rank() - 1;
    let mut have = 0usize;
    for (i, o) in outs.iter().enumerate() {
        if o.rank() - 1 != axis {
            candle::bail!(
                "{what}: output {i} is rank {} but output 0 is rank {} — a group's projections \
                 share an operand, so they cannot differ in rank",
                o.rank(),
                axis + 1
            );
        }
        have += o.dim(axis)?;
    }
    let want: usize = widths.iter().sum();
    if have != want {
        candle::bail!(
            "{what}: the projection group is {have} wide but its parts need {want} \
             ({widths:?}) — the stacked weight was built for a different geometry"
        );
    }

    // Fast path with nothing to scatter: every weight holds exactly one part, so
    // each output already *is* a part. This is the unstacked loader, and it must
    // not pay a launch for a copy that would be the identity.
    //
    // **Per-part widths are checked here, not just the total.** Returning `outs`
    // asserts that output `i` IS part `i`, which the sum above does not
    // establish: a group whose parts were stacked differently can match on count
    // and on total while every part is the wrong one. Nothing downstream would
    // notice — several groups in this tree carry same-width parts (β and α, for
    // one), so a swap type-checks, produces plausible shapes, and shows up only
    // as a wrong number much later.
    if outs.len() == widths.len() {
        for (i, (o, &w)) in outs.iter().zip(widths.iter()).enumerate() {
            let got = o.dim(axis)?;
            if got != w {
                candle::bail!(
                    "{what}: part {i} of the projection group is {got} wide but the geometry \
                     says {w} ({widths:?}) — the parts total correctly, so this is a group \
                     stacked in a different order rather than a different size"
                );
            }
        }
        return Ok(outs);
    }

    // The leading axes, flattened: `rows_scatter` walks rows, and a rank-3
    // `[b, s, W]` is `[b·s, W]` with row stride `W` — a view, not a copy,
    // because a contiguous output's leading strides are exactly that product.
    let lead: Vec<usize> = outs[0].dims()[..axis].to_vec();
    let rows: usize = lead.iter().product();
    let dev = outs[0].device().clone();
    if !matches!(dev, Device::Cuda(_)) {
        candle::bail!("{what}: a stacked projection group is CUDA-only");
    }

    // **ONE destination for every part, and nothing else allocated.**
    // `empty_beside` relays the source's wave ticket, so this is a pointer bump
    // in the arena the projections already live in — not a pool allocation, and
    // not a memset: the scatter writes every byte of it, which is exactly the
    // condition invariant 6 requires for `alloc_uninit`.
    let block = outs[0].empty_beside(rows * want, outs[0].dtype())?;

    let mut runs = Vec::with_capacity(widths.len());
    let mut parts = Vec::with_capacity(widths.len());
    let mut oi = 0usize;
    let mut off = 0usize;
    let mut dst_off = 0usize;
    for (pi, &w) in widths.iter().enumerate() {
        let cur = &outs[oi];
        let cw = cur.dim(axis)?;
        if off + w > cw {
            candle::bail!(
                "{what}: part {pi} ({w} wide) straddles the end of weight {oi} ({cw} wide, \
                 {off} consumed) — a stacked weight must end on a part boundary, or the split \
                 would cut a projection in half"
            );
        }
        // Both sides are VIEWS. The source is the stacked output read where it
        // lies — a row stride wider than the part is what `row_geometry`
        // explicitly permits — and the destination is this part's slice of the
        // one block. No tensor is copied to build the run; `RowRun` holds `Arc`
        // handles, which is why it takes the wave lifetime.
        let src = cur.flatten_to(axis - 1)?.narrow(1, off, w)?;
        let dst = block.narrow(0, dst_off, rows * w)?.reshape((rows, w))?;
        runs.push(RowRun::new(src, &dst, 0));
        let mut shape = lead.clone();
        shape.push(w);
        parts.push(dst.reshape(shape)?);
        dst_off += rows * w;
        off += w;
        if off == cw {
            oi += 1;
            off = 0;
        }
    }

    // Every part in ONE launch, with the descriptor in kernel parameters rather
    // than staged — so this path opens no generation, takes no global stager
    // mutex, and puts nothing on the bus for the device to read back.
    rows_scatter_inline(&runs)?;
    Ok(parts)
}
