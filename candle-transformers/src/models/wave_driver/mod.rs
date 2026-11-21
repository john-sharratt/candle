//! The model-agnostic half of a co-batched wave: everything `forward_wave` does
//! apart from the layer sweep itself.
//!
//! A model supplies a `WaveSweep` for its layers and receives `drive_wave` for the
//! rest. That covers bounding a forward's token count, folding one-token prefills,
//! permuting rows between caller order and internal order, rolling back on
//! failure, and advancing the decode rows once the head has run. The attention
//! metadata is built inside the sweep, not here, because a model may move its
//! arena before its layers read it. Each ordering rule in this module records a
//! failure that was diagnosed the hard way, and each holds for every model that
//! drives a wave.
//!
//! - `drive_step`: one co-batched step over its segments.
//! - `forward_core`: one forward, from its entry to its rollback or advance.
//! - `step_rows`: a step's row groups, its creep and its segments.
//! - `wave_sweep`: the trait a model implements for its layer sweep.
//! - `tests_step`: tests for segment cutting and the head's row mapping.

use std::time::Instant;

use candle::cuda_backend::wave_provenance::WaveTicket;
use candle::{DType, Device, Error, Result, Tensor};

use super::batched_inference::{
    pack_prefill_slabs, prefill_slack_cap, BatchedInferenceSession, PendingGlue, WaveResult,
    WaveStep,
};
use super::batched_model::{WaveGuard, WavePhase};
use super::held_tensor::hold_rows;
use super::kv_cache_utils::SequenceContext;
use super::profile::{begin_gpu_span_forward, gpu_span};
use super::residual_rows::{
    move_rows, segment_parts, CreepResidual, HandedOn, Part, ResidualIn, RowMap, RowStore, SpanRows,
};
use candle::quantized::pinned_staging::Generation;
use candle_nn::kv_cache::PREFILL_KV_STAGE_MIN_Q_LEN;
use candle_nn::kv_cache::{
    begin_wave, end_wave_transient, plan_wave_transient, provision_held_ground, tier_footprint,
    LayerPhase,
};

/// What a sweep does with the forward it is handed (`docs/wave_step.md` §2.1).
#[derive(Debug, Clone, Copy)]
pub enum SweepMode {
    /// A forward on its own: hand back the last tier, move the boundary, admit,
    /// place this forward's tier, open its forward span, sweep.
    Whole,
    /// One segment of a step, before the step's tier is placed: the prologue and
    /// the admit, and the segment's per-phase price returned as
    /// [`WavePhase::Priced`] — nothing placed, nothing swept.
    Prepare,
    /// One segment of a step whose tier is placed and whose forward span is
    /// open: no prologue, no admit, no placement, and every forward-span carve
    /// from the step's generation, named by this ticket. `None` for a model that
    /// priced its segments at nothing — it keeps no buffer on the tier, so the
    /// step places none and opens no span for it.
    InStep(Option<WaveTicket>),
}

/// Everything the layer sweep needs that the driver assembled for it.
///
/// The contexts are passed alongside rather than inside, because the sweep
/// takes them mutably and this struct is borrowed from the same scope.
pub struct WaveGroups<'a> {
    /// Contexts `[0, n_decode)` — one row each, decode-kernel layout.
    pub n_decode: usize,
    /// Contexts `[n_decode, n_decode + n_prefill)` — ragged, prefill kernel.
    pub n_prefill: usize,
    /// The scheduler's sequence id of each context, in context order.
    ///
    /// A [`SequenceContext`] carries a sequence's KV and its offset but not its
    /// identity, which is all a uniform transformer needs. A model with
    /// per-sequence state outside the paged cache — a recurrent mixer's `S` and
    /// conv tail — has to key that state by something, and this is it.
    pub seq_ids: &'a [usize],
    /// One input tensor per context, in the same internal
    /// `[decode | prefill | glue]` order as `seq_ids` (the driver has already
    /// folded 1-token prefills into the decode group).
    ///
    /// The sweep needs these twice: to assemble its [`SequenceContext`]s
    /// (via [`assemble_wave_contexts`]) and to embed a fresh wave's rows. They
    /// are here rather than inside pre-built contexts because the sweep also
    /// needs the SESSION — for its attention-metadata build, which must run at
    /// the sweep's own phase order (see the module docs) — and the contexts
    /// borrow the session mutably, so the driver cannot hand over both.
    pub inputs: &'a [Tensor],
    /// Raw glue scatter descriptors staged on the session, one per glue
    /// sequence in order — taken off the session by the driver (which also
    /// drops stale staging, loudly) and handed through untranslated: what shape
    /// the kernel metadata takes (flat device tensors, per-run host slices) is
    /// the sweep's call. `None` when the wave carries no glue rows or nothing
    /// was staged.
    pub pending_glue: Option<Vec<PendingGlue>>,
    /// Pinned-stager generation guarding this wave's kernel metadata uploads.
    pub generation: &'a Generation,
    pub layer_start: usize,
    pub layer_end: usize,
    /// The residual a resumed forward starts from. The sweep carves its own
    /// residual on its forward span and gathers this into it
    /// ([`ResidualIn::gather_into`]) — internal order is baked into the gather —
    /// so it never writes what it read. `None` for a forward that embeds.
    ///
    /// A forward that stops short returns that buffer, still on its span, with
    /// the span's guard; the driver moves it on.
    pub x_in: Option<ResidualIn<'a>>,
    /// Whole forward, step preparation, or step segment.
    pub mode: SweepMode,
    /// The width every activation in this wave flows in.
    ///
    /// Decided ONCE, here, from the session's declared activation dtype, and
    /// carried rather than re-derived. The embedding emits it and every kernel
    /// downstream emits what its consumer reads, so no conversion appears on
    /// the path (hot-path invariant 1); where two operands must agree, the
    /// consumer VALIDATES with `expect_dtype` rather than rewriting them
    /// (invariant 1b).
    ///
    /// It is on the wave rather than derived inside the sweep because the sweep
    /// used to read it off the live KV cache — which answers a different
    /// question. A quantized backing reports F16 because the arena really is
    /// F16 (K in `R16`, V in plain F16); that is a fact about KV STORAGE and
    /// says nothing about the width the model computes in. The two agreed only
    /// until a model declared otherwise, and then they silently disagreed: the
    /// norms were materialised BF16 from the session while the activations
    /// arrived F16 from the cache.
    pub act_dtype: DType,
    /// The LoRA adapter every sequence in this wave decodes through, by name,
    /// or `None` for the base model.
    ///
    /// One value for the wave, not one per sequence: the adapter alters the
    /// projections the whole batch flows through together, so a wave is
    /// adapter-homogeneous by construction and the scheduler loops waves by
    /// adapter rather than batching across them.
    /// [`BatchedInferenceSession::wave_adapter`] derives it and refuses a mixed
    /// wave.
    ///
    /// A model that does not implement adapters ignores this; a model that does
    /// resolves the name against what it has loaded, and fails if it cannot —
    /// running unadapted because a name did not match is the failure mode this
    /// is shaped to prevent.
    pub adapter: Option<&'a str>,
}

/// Drive one co-batched wave: the whole of `forward_wave`, bar the sweep.
///
/// `contexts` are ordered `[decode… | prefill… | glue…]`. When the range
/// reaches the head, the result carries logits for the **decode + prefill**
/// rows only, in the caller's order — glue rows scatter K/V and carry none.
///
/// A range that starts past layer 0 resumes from `residual`'s committed rows; a
/// range that stops short of the head writes its residual to `residual`'s
/// pending half and commits it. Both are in caller order, one row per input
/// token (`docs/wave_step.md` §3.2).
#[allow(clippy::too_many_arguments)]
pub fn drive_wave<S: WaveSweep + ?Sized>(
    model: &S,
    session: &mut BatchedInferenceSession,
    decode_seqs: &[usize],
    decode_inputs: &[Tensor],
    prefill_seqs: &[usize],
    prefill_inputs: &[Tensor],
    glue_seqs: &[usize],
    glue_inputs: &[Tensor],
    layer_start: usize,
    layer_end: usize,
    residual: Option<&mut CreepResidual>,
) -> Result<WaveResult> {
    if decode_inputs.len() != decode_seqs.len()
        || prefill_inputs.len() != prefill_seqs.len()
        || glue_inputs.len() != glue_seqs.len()
    {
        candle::bail!("forward_wave: input/seq length mismatch");
    }
    provision_held(model);
    // Whether this forward's GPU spans are sampled (`set_gpu_span_period`): a
    // forward carrying a prompt chunk always is.
    let bulk = prefill_inputs
        .iter()
        .any(|t| t.dims().last().copied().unwrap_or(0) >= PREFILL_KV_STAGE_MIN_Q_LEN);
    begin_gpu_span_forward(bulk);

    // Bound a single forward's token count: a PURE prefill full sweep whose
    // total tokens exceed the budget is split into token-bounded sub-forwards
    // and its logits concatenated. Only applies with no decode/glue rows, a
    // full `[0, N)` sweep, and no resumed residual — the co-batched /
    // re-entrant paths are bounded by the scheduler's admission window + OOM
    // retry instead.
    let num_layers = model.num_layers();
    if decode_seqs.is_empty()
        && glue_seqs.is_empty()
        && residual.is_none()
        && layer_start == 0
        && layer_end == num_layers
    {
        let lens: Vec<usize> = prefill_inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(1))
            .collect();
        let total: usize = lens.iter().sum();
        let max_len = lens.iter().copied().max().unwrap_or(1);
        // Two ceilings, for unrelated reasons, so the narrower one wins.
        //
        // `MAX_PREFILL_TOKENS` is where the kernels stop caring: compute
        // saturates around it, so a wider forward buys no throughput. The
        // plan's bound is what the FFN span can physically hold, which is a
        // correctness limit — exceed it and the expert chain spills to the
        // pool, silently, one allocation at a time.
        //
        // They were previously decided apart, and the arena was the one that
        // lost: it ran at ~100% of its span while the slicer sized waves
        // against a constant that knows nothing about model geometry. A dense
        // model and a MoE model at the same token count need wildly different
        // spans — `expert_rows` multiplies by `experts_per_tok` — so only the
        // plan can answer this.
        let width_cap = model.prefill_width_cap(session.activation_dtype());
        // The entry check uses the SLACK ceiling, not the bare cap: a fleet
        // within 25% of the cap runs as a single wave (the straggler a
        // bare-cap split would produce costs the full fixed per-wave sweep for
        // its few tokens), and a slab the packer emitted WITH slack must not
        // re-slice itself when this function recurses on it.
        if total > prefill_slack_cap(width_cap) && max_len > 1 && prefill_seqs.len() > 1 {
            let mut all_logits: Vec<Tensor> = Vec::with_capacity(prefill_seqs.len());
            for (start, end) in pack_prefill_slabs(&lens, width_cap) {
                let step = drive_wave(
                    model,
                    session,
                    &[],
                    &[],
                    &prefill_seqs[start..end],
                    &prefill_inputs[start..end],
                    &[],
                    &[],
                    0,
                    num_layers,
                    None,
                )?;
                let lg = step
                    .logits
                    .as_ref()
                    .ok_or_else(|| Error::Msg("forward_wave slice: no logits".into()))?;
                // Copied off the span, not moved off it. Each slice is a whole
                // forward and reclaims its own forward span when `step` drops
                // at the end of this iteration — so a borrowed logits row would
                // be reading recycled bytes by the time the next slice ran. (It
                // could not even get that far: the span refuses a second live
                // generation, so slice two would fail to open one while slice
                // one still held it.)
                //
                // This is the sanctioned escape and it really copies — into
                // held ground, one block per slice. It is confined to the
                // slicing path, which is already paying for N forwards, and it
                // is why the value returned below is owned.
                if !lg.is_empty() {
                    let block = hold_rows(lg)?;
                    for (i, row) in lg.iter().enumerate() {
                        all_logits.push(block.narrow(0, i, 1)?.reshape(row.shape())?);
                    }
                }
            }
            return Ok(WaveResult::owned(WaveStep {
                logits: Some(all_logits),
            }));
        }
    }

    // The residual is checked against this forward's rows before anything
    // runs: past the sweep, a mismatch would surface only after the window's
    // recurrent and KV state had committed.
    let rows = caller_rows(decode_inputs) + caller_rows(prefill_inputs) + caller_rows(glue_inputs);
    match residual.as_deref() {
        Some(r) => check_residual(model, session, r, rows)?,
        None if layer_start > 0 || layer_end < num_layers => candle::bail!(
            "forward_wave: layers [{layer_start}, {layer_end}) of {num_layers} need a residual to \
             resume from or write to"
        ),
        None => {}
    }
    let pending_glue = take_pending_glue(session, glue_inputs);
    // A range past layer 0 resumes from the residual's committed rows — all of
    // them, in caller order, one per input row.
    let input = match residual.as_deref() {
        Some(r) if layer_start > 0 => Some(vec![Part {
            store: r.committed(),
            layout: r.layout(),
            start: 0,
            len: r.rows(),
        }]),
        _ => None,
    };
    let out = forward_core(
        model,
        session,
        GroupRows {
            decode_seqs,
            decode_inputs,
            prefill_seqs,
            prefill_inputs,
            glue_seqs,
            glue_inputs,
        },
        (layer_start, layer_end),
        pending_glue,
        input,
        SweepMode::Whole,
    )?;
    match out {
        // The head's outputs sit on the forward span, so the guard goes back
        // with them: `WaveResult` is what stops the span being reclaimed while
        // the caller still holds the logits.
        CoreOut::Logits { step, guard } => Ok(WaveResult::on_span(step, guard)),
        CoreOut::Residual {
            working,
            layout,
            guard,
        } => {
            let Some(r) = residual else {
                candle::bail!(
                    "forward_wave: a range ending at layer {layer_end} of {num_layers} stops short \
                     of the head and was given no residual to write"
                );
            };
            let width = r.committed().width();
            move_rows(
                &SpanRows::new(&working, width)?,
                &layout,
                0,
                r.pending(),
                &r.layout(),
                0,
                layout.rows(),
            )?;
            // The copy is queued behind the sweep on the same stream; the guard
            // goes once it is.
            drop(guard);
            r.commit();
            Ok(WaveResult::owned(WaveStep { logits: None }))
        }
        CoreOut::Priced(_) => candle::bail!("forward_wave: a whole forward came back priced"),
    }
}

mod drive_step;
mod forward_core;
mod step_rows;
#[cfg(test)]
mod tests_step;
mod wave_sweep;

pub use drive_step::*;
pub(crate) use forward_core::*;
pub use step_rows::*;
pub use wave_sweep::*;
