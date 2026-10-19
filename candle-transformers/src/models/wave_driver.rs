//! The model-agnostic half of a co-batched wave.
//!
//! A wave forward splits cleanly in two. The **outer** half — bounding a
//! forward's token count, routing 1-token prefills to the decode kernel,
//! permuting tokens between caller order and internal order, rolling the KV
//! back when the sweep fails, and advancing the decode rows once the head has
//! run — depends on nothing about the model except its depth and its device.
//! The **inner** half, the layer sweep itself, is where a model's architecture
//! actually lives.
//!
//! Attention metadata is deliberately the SWEEP'S to build, not the driver's.
//! The slot headers serialize each sequence's live arena state, and a model may
//! legitimately move that state before its layer loop reads it — a sliding
//! window ring evicts front chunks and re-bases offsets, a prefill commits its
//! whole write range up front, a speculative verify commits block lengths. A
//! header built out here would describe the arena as it stood before any of
//! that, which on a slid ring resolves ABSOLUTE offsets past the resident span.
//! So the driver hands the sweep the session and the raw inputs, and the sweep
//! builds its headers at the point in its own phase order where they are true.
//!
//! Only the inner half is per-model. This module is the outer half, written
//! once: a model supplies [`WaveSweep`] and gets [`drive_wave`], which is the
//! whole of `ManagedBatchedModel::forward_wave`. A uniform transformer's sweep
//! is `BatchedInference::forward_wave_contexts`; a hybrid's dispatches on layer
//! kind and carries recurrent state; neither restates the bookkeeping around
//! it.
//!
//! The ordering rules encoded here are the expensive ones — each comment below
//! marks a failure that was diagnosed the hard way, and they hold for every
//! model that drives a wave.

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
    begin_wave, end_wave_transient, plan_wave_transient, provision_held_ground, LayerPhase,
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

/// The per-model half of a wave: run one layer range over assembled contexts.
pub trait WaveSweep {
    fn device(&self) -> &Device;

    /// Transformer depth — what bounds a wave's layer range. On a hybrid this
    /// is the trunk depth, not the KV-layer count.
    fn num_layers(&self) -> usize;

    /// Widest prefill this model runs in one forward, in tokens.
    fn prefill_width_cap(&self, act_dtype: DType) -> usize;

    /// One row of this model's residual stream: its elements and their type —
    /// what a creep's arena holds per row (`docs/wave_step.md` §3). Asked
    /// before a step places its tier, since the arena is claimed then.
    fn residual_row(&self, session: &BatchedInferenceSession) -> Result<(usize, DType)>;

    /// The **KV-cache** index range a trunk-layer range writes to.
    ///
    /// A session allocates one paged cache per layer that attends, so on a
    /// uniform transformer the two coincide and this is the identity. On a
    /// hybrid three quarters of the layers own no KV, and a rollback driven by
    /// trunk indices would index past the end of the cache vector — so the
    /// translation lives here, where the driver can ask for it, rather than
    /// being assumed.
    fn kv_layer_range(&self, layer_start: usize, layer_end: usize) -> (usize, usize) {
        (layer_start, layer_end)
    }

    /// Forward-entry invariant: every member's logical offset must equal the
    /// token count its live block table covers. The varlen metadata is built
    /// from the offsets while the slot headers are built from the block tables
    /// — the attention kernels resolve every `[0, kv_len)` position through the
    /// table, so any divergence walks them past the slot's span in the packed
    /// staged uploads (garbage slice indices → wild record pointers →
    /// CUDA_ERROR_ILLEGAL_ADDRESS, or silent cross-slot reads). Offsets run
    /// AHEAD of the backing when a projection drops sections it could not lift
    /// under VRAM pressure; they run BEHIND after glue reserves gap chunks the
    /// wave didn't reflect. Positions are slot-relative (slice ropes), so the
    /// backing length is also the correct RoPE base either way.
    ///
    /// **A model hook because the invariant itself is a model property.** The
    /// default states the uniform/hybrid contract, `offset == backing`. A model
    /// whose arena is a sliding window ring keeps ABSOLUTE offsets against a
    /// RESIDENT backing (`offset == base_pos + backing`), and the default's
    /// clamp would silently re-base every slid sequence — so such a model
    /// overrides this with its own reconciliation (or a no-op, where its wave
    /// entry re-derives lengths from the session itself).
    fn reconcile_entry_offsets(
        &self,
        session: &mut BatchedInferenceSession,
        seq_groups: [&[usize]; 3],
    ) -> Result<()> {
        for ids in seq_groups {
            for &i in ids {
                let off = session.sequence_offset(i).unwrap_or(0);
                let backing = session.sequence_backing_tokens(i).unwrap_or(off);
                if backing != off {
                    if backing < off {
                        tracing::warn!(
                            seq = i,
                            offset = off,
                            backing,
                            "sequence offset AHEAD of backing at forward entry — \
                             clamped down (projection dropped un-liftable sections)"
                        );
                    } else {
                        tracing::debug!(
                            seq = i,
                            offset = off,
                            backing,
                            "sequence offset behind backing at forward entry — advanced"
                        );
                    }
                    session.set_sequence_offset(i, backing)?;
                }
            }
        }
        Ok(())
    }

    /// Advance the decode rows' per-layer KV lengths after a completed step —
    /// once per delivered token, every layer at once, called by the driver
    /// only after the head ran and the logits were copied off.
    ///
    /// The default is the uniform/hybrid contract: every cache to
    /// `offset + 1`, with a failure at row `i` unwound by truncating rows
    /// `0..=i` back to their entry lengths (leaving layers `0..k` at
    /// `offset+1` against the rest at `offset` is exactly the per-layer
    /// divergence this consolidation exists to prevent, and the wave rollback
    /// cannot reach it).
    ///
    /// A model whose decode write-length is committed **on-device** by its
    /// decode kernel — and re-committed from the session at every wave entry —
    /// overrides this as a no-op: the default's `set_current_seq_len` stamps
    /// the session's ABSOLUTE offset into the backing, which a sliding window
    /// ring addresses in RESIDENT terms.
    fn advance_decode_rows(&self, contexts: &mut [SequenceContext], n_decode: usize) -> Result<()> {
        for i in 0..n_decode {
            let offset = contexts[i].offset;
            let mut advance = || -> Result<()> {
                for cache in contexts[i].kv_caches.caches.iter_mut() {
                    cache.set_current_seq_len(offset + 1)?;
                }
                Ok(())
            };
            if let Err(e) = advance() {
                // Every layer of every touched row, even past a failure. The
                // `try_for_each` this replaced stopped at the first error and
                // discarded it, so an unwind that hit a bad layer left the rows
                // *before* it truncated and the rest advanced — reproducing, from
                // the recovery path, the exact per-layer skew the unwind exists to
                // erase, and saying nothing about it.
                for c in contexts[..=i].iter_mut() {
                    let off = c.offset;
                    for (li, cache) in c.kv_caches.caches.iter_mut().enumerate() {
                        if let Err(te) = cache.truncate_to_offset(off) {
                            tracing::warn!(
                                layer = li,
                                offset = off,
                                "decode advance unwind: layer truncate failed; continuing \
                                 so the remaining layers are not left advanced: {te}"
                            );
                        }
                    }
                }
                return Err(e);
            }
        }
        Ok(())
    }

    /// Undo a failed wave's per-layer KV bookkeeping, so the scheduler's retry
    /// is a retry rather than a decode against per-layer token windows.
    ///
    /// The default truncates every row on every KV layer of the range back to
    /// its entry offset ([`super::wave_admit::rollback_wave_kv`]). A model
    /// that re-derives every layer's writer length from the session at wave
    /// entry overrides this to match — the default's absolute-offset truncate
    /// would mis-address a sliding window ring's resident arena.
    fn rollback_wave(
        &self,
        contexts: &mut [SequenceContext],
        kv_start: usize,
        kv_end: usize,
    ) -> Result<()> {
        super::wave_admit::rollback_wave_kv(contexts, kv_start, kv_end)
    }

    /// Fail the segment just swept if the device gave up part of it — an expert
    /// cache whose workers lost a cold expert (`ExpertCache::take_fault`). The
    /// driver calls it after every segment, residual or logits, and on an error
    /// rolls the segment back exactly as a failed sweep, so a faulted segment
    /// never commits its KV or hands its residual on.
    ///
    /// The fault word is complete only once every launch that could claim it
    /// is, so a model whose device can give work up waits for the segment's
    /// launches first (`ExpertCache::take_segment_fault`). Only a cache that can
    /// leave a worker waiting on a cold expert can: one whose every expert is
    /// in VRAM or pinned host memory never waits, and its check is free. The
    /// wait is not: nothing before it has synchronised the segment — the logits
    /// stay on the device — so it stalls the host until the device drains, and
    /// taken on every wave it cost Flash-Next's single-session decode ~6 ms a
    /// step on a card that could never fault.
    ///
    /// Required, not defaulted: every sweep states whether its device can give
    /// work up, so a model gaining an expert cache cannot silently skip it.
    fn take_device_fault(&self) -> Result<()>;

    /// Run `[layer_start, layer_end)` over the wave, returning the residual
    /// (range stopped short of the head) or the logits (range reached it) —
    /// or, under [`SweepMode::Prepare`], the segment's price.
    ///
    /// **A residual comes back on the forward span, with the span's guard.**
    /// The driver moves it where it goes next — a step's next segment, a
    /// creep's arena — while the guard still holds it.
    ///
    /// The sweep owns the session for its duration. It builds its own attention
    /// metadata (see the module docs for why the driver cannot), assembles its
    /// [`SequenceContext`]s via [`assemble_wave_contexts`] when its layer body
    /// works through per-sequence `KvCache`s, and is free to skip that entirely
    /// when its KV lives behind the session's per-layer backings. Order matters
    /// inside: metadata builds borrow the session shared, the contexts borrow
    /// it mutably, so headers are built before contexts are taken.
    fn sweep(
        &self,
        session: &mut BatchedInferenceSession,
        wave: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)>;
}

/// Assemble the per-sequence contexts for a wave, in `seq_ids` order.
///
/// `inputs` is one tensor per sequence in the same order. Fails loudly when a
/// named sequence has no live slot — `caches_for_sequences_mut` silently skips
/// those, which used to surface ~100 lines later as a `checked_sub` underflow
/// naming neither the fault nor the sequence.
///
/// Used by the sweeps on the way in, and by [`drive_wave`] after the sweep
/// returns — the KV rollback of a failed wave and the decode advance of a
/// completed one need the same borrows, and re-assembling is legal because the
/// sweep's mutable borrow of the session ends when it returns (nothing inside a
/// wave moves the session's sequence offsets; the scheduler advances them after
/// the forward).
pub(crate) fn assemble_wave_contexts<'s>(
    session: &'s mut BatchedInferenceSession,
    seq_ids: &[usize],
    inputs: &'s [Tensor],
) -> Result<Vec<SequenceContext<'s>>> {
    if inputs.len() != seq_ids.len() {
        candle::bail!(
            "assemble_wave_contexts: {} inputs against {} sequences",
            inputs.len(),
            seq_ids.len()
        );
    }
    let caches_data = session.caches_for_sequences_mut(seq_ids);
    if caches_data.len() != seq_ids.len() {
        let live: std::collections::HashSet<usize> =
            caches_data.iter().map(|(i, _, _)| *i).collect();
        let missing: Vec<usize> = seq_ids
            .iter()
            .copied()
            .filter(|s| !live.contains(s))
            .collect();
        candle::bail!(
            "forward_wave: {} sequences requested but only {} have live slots \
             (missing/duplicated: {missing:?}) — the wave group named a sequence \
             the scheduler has since released",
            seq_ids.len(),
            caches_data.len(),
        );
    }
    let mut contexts: Vec<SequenceContext<'s>> = Vec::with_capacity(seq_ids.len());
    for ((_seq_idx, offset, caches), input) in caches_data.into_iter().zip(inputs.iter()) {
        contexts.push(SequenceContext {
            offset,
            kv_caches: caches,
            input_ids: input,
            input_len: input.dims().get(1).copied().unwrap_or(1),
        });
    }
    Ok(contexts)
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

/// A forward's caller-order rows: decode, prefill and glue, each with its inputs.
#[derive(Clone, Copy)]
pub(crate) struct GroupRows<'a> {
    pub decode_seqs: &'a [usize],
    pub decode_inputs: &'a [Tensor],
    pub prefill_seqs: &'a [usize],
    pub prefill_inputs: &'a [Tensor],
    pub glue_seqs: &'a [usize],
    pub glue_inputs: &'a [Tensor],
}

/// What one forward produced.
pub(crate) enum CoreOut {
    /// [`SweepMode::Prepare`]: the forward's per-phase price.
    Priced([usize; 3]),
    /// The range stopped short: the forward's residual, still on its span, in
    /// the forward's internal layout, with what holds the span.
    Residual {
        working: Tensor,
        layout: RowMap,
        guard: Option<WaveGuard>,
    },
    /// The range reached the head.
    Logits {
        step: WaveStep,
        guard: Option<WaveGuard>,
    },
}

/// The glue scatter descriptors staged for a forward that carries glue rows.
///
/// Staged glue is consumed only by a wave that carries glue rows. The
/// descriptors are staged immediately before the gap-fill forward they describe
/// — but that forward can die before reaching this point, and the staging then
/// sits on the session for whatever wave comes next, which is how a dead wave's
/// leftovers killed the titler twice. Stale staging is dropped, loudly: the
/// reproject that staged it re-stages when its own retry runs.
fn take_pending_glue(
    session: &mut BatchedInferenceSession,
    glue_inputs: &[Tensor],
) -> Option<Vec<PendingGlue>> {
    if glue_inputs.is_empty() {
        if let Some(stale) = session.take_pending_glue() {
            tracing::warn!(
                n = stale.len(),
                "dropping glue descriptors staged by a wave that never ran its \
                 gap-fill forward — this wave carries no glue rows"
            );
        }
        None
    } else {
        session.take_pending_glue()
    }
}

/// One forward over `[layer_start, layer_end)`: everything the driver does
/// around the sweep, in `mode`.
///
/// `input` is the resumed residual's parts in caller order; the forward gathers
/// them into its own buffer in its internal layout.
fn forward_core<S: WaveSweep + ?Sized>(
    model: &S,
    session: &mut BatchedInferenceSession,
    rows: GroupRows<'_>,
    (layer_start, layer_end): (usize, usize),
    pending_glue: Option<Vec<PendingGlue>>,
    input: Option<Vec<Part<'_>>>,
    mode: SweepMode,
) -> Result<CoreOut> {
    let GroupRows {
        decode_seqs,
        decode_inputs,
        prefill_seqs,
        prefill_inputs,
        glue_seqs,
        glue_inputs,
    } = rows;
    let num_layers = model.num_layers();
    let n_decode_in = decode_seqs.len();
    let n_prefill_in = prefill_seqs.len();

    let stager_generation = session.begin_stager_generation();

    // Forward-entry offset reconciliation — the choke point every forward
    // passes through (wave steps, deferred projection gap-fills, probes). The
    // invariant and its enforcement are the model's
    // ([`WaveSweep::reconcile_entry_offsets`]): uniform/hybrid stacks clamp
    // `offset` to the backing length, a sliding-window-ring model keeps its
    // absolute offsets.
    model.reconcile_entry_offsets(session, [decode_seqs, prefill_seqs, glue_seqs])?;

    // Per-group query lengths.
    let dev = model.device();
    let input_len = |ins: &[Tensor]| -> Vec<usize> {
        ins.iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(1))
            .collect()
    };

    // A single-token prefill is operationally a decode — one new token over a
    // prefix — and the paged prefill kernel DIVERGES from the canonical decode
    // kernel for `q_len == 1` (GPU-verified: cos ~0.94, argmax flips). Route
    // every 1-token prefill row through the DECODE path (the correct
    // single-token attention), keeping multi-token prefills on the prefill
    // kernel. `single` / `multi` hold the original prefill indices of each
    // class, so the caller's `[decode | prefill]` output order is restored by a
    // stable inverse permutation at the end. `single` empty ⇒ no-op fast path.
    let pre_lens_in = input_len(prefill_inputs);
    let single: Vec<usize> = (0..n_prefill_in).filter(|&i| pre_lens_in[i] == 1).collect();
    let multi: Vec<usize> = (0..n_prefill_in).filter(|&i| pre_lens_in[i] != 1).collect();

    let mut proc_decode_seqs: Vec<usize> = decode_seqs.to_vec();
    let mut proc_decode_inputs: Vec<Tensor> = decode_inputs.to_vec();
    for &i in &single {
        proc_decode_seqs.push(prefill_seqs[i]);
        proc_decode_inputs.push(prefill_inputs[i].clone());
    }
    let proc_prefill_seqs: Vec<usize> = multi.iter().map(|&i| prefill_seqs[i]).collect();
    let proc_prefill_inputs: Vec<Tensor> =
        multi.iter().map(|&i| prefill_inputs[i].clone()).collect();
    let n_decode = proc_decode_seqs.len();
    let n_prefill = proc_prefill_seqs.len();

    let glue_lens = input_len(glue_inputs);

    // Assemble the combined sequence list in [decode | prefill | glue] order.
    let mut all_seqs: Vec<usize> = Vec::with_capacity(n_decode + n_prefill + glue_seqs.len());
    all_seqs.extend_from_slice(&proc_decode_seqs);
    all_seqs.extend_from_slice(&proc_prefill_seqs);
    all_seqs.extend_from_slice(glue_seqs);
    // A sequence id must appear in exactly ONE group: `caches_for_sequences_mut`
    // yields one entry per unique id, so a duplicate COLLAPSES the context list
    // and shifts every later member's cache against the group varlen metadata
    // built above — slot headers then describe a different sequence than the
    // kernel's cu_seqlens/kv_lens entry, and the kernel walks past the
    // (shorter) slot's staged state into neighboring uploads. Fail loudly.
    {
        let mut seen = std::collections::HashSet::with_capacity(all_seqs.len());
        for &id in &all_seqs {
            if !seen.insert(id) {
                candle::bail!(
                    "forward wave: sequence {id} appears in more than one group \
                     (decode {proc_decode_seqs:?} | prefill {proc_prefill_seqs:?} \
                     | glue {glue_seqs:?}) — the context list would collapse and \
                     desync every later member's cache from its metadata"
                );
            }
        }
    }
    // One OWNED input per context, internal order. `WaveGroups` borrows these
    // for the sweep, and the post-sweep rollback/advance re-borrows them to
    // re-assemble contexts (`Tensor` clones are refcount bumps).
    let all_inputs: Vec<Tensor> = proc_decode_inputs
        .iter()
        .cloned()
        .chain(proc_prefill_inputs.iter().cloned())
        .chain(glue_inputs.iter().cloned())
        .collect();

    // The wave's declared activation width; it belongs to the session, not to
    // the caches.
    let act_dtype = session.activation_dtype();
    // Same reason, and one more: this refuses a wave whose sequences disagree
    // about their adapter. The projections run once over the whole batch, so
    // there is no such thing as half an adapted wave — better to fail here than
    // to give one group's conversations another group's fine-tune.
    let wave_adapter = session.wave_adapter(&all_seqs)?;

    // The forward's layout (`residual_rows::row_map`): the sweep packs rows in
    // internal order, and every residual that crosses into or out of it is in
    // caller order. A resumed residual is gathered straight into internal order
    // and one handed on is moved out of it, so nothing is reordered through a
    // buffer of its own. With no single-token prefill the two coincide.
    let layout = RowMap::wave(n_decode_in, &pre_lens_in, glue_lens.iter().sum());
    let x_in = match input {
        Some(parts) => Some(ResidualIn::new(parts, layout.clone())?),
        None => None,
    };
    let preparing = matches!(mode, SweepMode::Prepare);
    // The layer sweep plus the head, so the forward's stream time divides into
    // "the model" and "everything the driver and its caller do around it". The
    // gap between this and the caller's own forward span is where a per-row copy
    // off the wave arena hid 76% of a 128-slot decode step.
    let g_sweep = gpu_span("wv:sweep", dev);
    // The sweep runs as a chain of graphs (`docs/decode_graphs.md`): its
    // launches are recorded and handed to the driver one segment at a time,
    // each segment ending where the forward meets the device eagerly — the MoE
    // host protocol, an upload, a readback. A failed sweep drops the capture,
    // discarding the segment it was recording; the rollback below runs eagerly.
    // A segment being prepared records nothing: it admits and prices, and
    // launches no layer.
    let capture = match dev {
        Device::Cuda(cuda) if !preparing => Some(cuda.begin_wave_capture()?),
        _ => None,
    };
    let wave = model.sweep(
        session,
        WaveGroups {
            n_decode,
            n_prefill,
            adapter: wave_adapter.as_deref(),
            seq_ids: &all_seqs,
            inputs: &all_inputs,
            pending_glue,
            generation: &stager_generation,
            layer_start,
            layer_end,
            x_in,
            mode,
            act_dtype,
        },
    );
    let wave = match (wave, capture) {
        (Ok(v), Some(capture)) => capture.finish().map(|()| v),
        (wave, _) => wave,
    };
    g_sweep.end();
    // **A failed wave leaves no trace.** The layer sweep advances each layer's
    // usage as that layer completes, so an error anywhere in it — and the relief
    // design treats failing a wave as routine — leaves the early layers one
    // token ahead of the rest. The rollback restores every row to its entry
    // length on every layer of the range, which is what makes the retry a retry
    // rather than a decode against per-layer token windows. This is the single
    // choke point every wave goes through; the sweep itself stays free to
    // advance eagerly, because whatever it did is undone here on the way out.
    // The sweep's mutable session borrow ended with it, so the rollback
    // re-assembles the contexts it needs. An assembly failure here is the same
    // unrecoverable shape as a failed rollback: report both.
    let fail = |session: &mut BatchedInferenceSession, e: Error| -> Error {
        let (kv_start, kv_end) = model.kv_layer_range(layer_start, layer_end);
        let rolled = assemble_wave_contexts(session, &all_seqs, &all_inputs)
            .and_then(|mut contexts| model.rollback_wave(&mut contexts, kv_start, kv_end));
        match rolled {
            Ok(()) => e,
            Err(rb) => Error::Msg(format!(
                "wave failed ({e}) and the KV rollback that keeps that failure \
                 recoverable also failed ({rb}) — the affected sequences may hold \
                 per-layer token windows"
            )),
        }
    };
    let (phase, head_span) = match wave {
        Ok(v) => v,
        // A launch refused over an untaken fault fails the sweep before any
        // logits exist, so the fault is taken here too — or it would stay set
        // and refuse every later wave. It is the cause, so it is the error.
        Err(e) => {
            let e = match model.take_device_fault() {
                Ok(()) => e,
                Err(f) => Error::Msg(format!("{f} (the sweep stopped on: {e})")),
            };
            return Err(fail(session, e));
        }
    };
    // Output ordering. The single-token prefills were folded into the decode
    // group, so the internal row order is
    // `[orig-decode | single-prefills | multi-prefills | glue]`.
    //
    // - Logits (final, head ran) are restored to the caller's
    //   `[decode | prefill-in-caller-order]` so `pf_logits[k]` aligns with the
    //   caller's `pf_seqs[k]`.
    // - The intermediate residual stays in internal order, on the forward span,
    //   with the forward's layout beside it: whoever moves it on moves it
    //   through the layout.
    let out = match phase {
        WavePhase::Priced(per_phase) => return Ok(CoreOut::Priced(per_phase)),
        // A glue-only forward that reached the head scored nothing: its rows
        // scattered K/V and that was the whole of it.
        WavePhase::Residual(_) if layer_end == num_layers => CoreOut::Logits {
            step: WaveStep { logits: None },
            guard: head_span,
        },
        WavePhase::Residual(x) => {
            // A segment that faulted must not commit: the next segment would
            // carry its residual on and roll back only its own layers.
            if let Err(e) = model.take_device_fault() {
                return Err(fail(session, e));
            }
            CoreOut::Residual {
                working: x.to_tensor(),
                layout,
                guard: head_span,
            }
        }
        WavePhase::Logits(l) => {
            // **The decode rows' usage advances here — once per step, every
            // layer at once — and nowhere else.** The per-layer advance used to
            // live inside the decode attention, which meant a step split across
            // creep segments held layers on both sides of the segment boundary
            // at different lengths; each later segment's metadata rebuild then
            // read its own step's half-done bookkeeping as per-layer corruption,
            // and the repair for *real* corruption truncated the freshly-written
            // token off the swept layers — token duplication in the visible text
            // was the symptom.
            //
            // The head having run is the definition of "the step completed":
            // logits exist only when the final segment reached layer N, so this
            // fires exactly once per delivered token — and never for a failed
            // wave, which leaves nothing for the rollback to undo on these rows.
            // In-step attention never needed the advance; it reads the new token
            // through the position map's write slot, built against the pre-step
            // usage.
            //
            // `n_decode` is the internal group — caller decode rows plus the
            // folded single-token prefills, which advance by their one token the
            // same way. Glue rows must not advance and are past
            // `n_decode + n_prefill`; multi-token prefills advance per layer
            // inside their own sweep because a creep cohort's layers are
            // *legitimately* at different lengths across waves.
            //
            // **The logits copy comes FIRST.** `into_vec` is the last fallible
            // step of the arm — an async CUDA fault surfaces on this sync — and
            // it sits outside the wave-error rollback (the `match wave` above
            // already resolved Ok). Advancing before it would leave every layer
            // advanced for a token the caller never receives; the retry then
            // writes into offset+1 with a stale KV row at offset. Copy first,
            // advance after, and nothing is advanced for an undelivered token.
            let lg = l.into_vec()?;
            // Anything the device gave up in the wave is failed and rolled back
            // like a failed sweep — its logits were computed from what was
            // given up.
            if let Err(e) = model.take_device_fault() {
                return Err(fail(session, e));
            }
            // The advance itself is per layer with no transaction, so a failure
            // at layer k is unwound here — truncating the advanced layers back
            // to their entry length, the same idempotent operation admit
            // performs — before the error propagates. Left as-is, layers 0..k at
            // offset+1 against the rest at offset is exactly the per-layer
            // divergence this consolidation exists to prevent, and the wave
            // rollback cannot reach it.
            let mut contexts = assemble_wave_contexts(session, &all_seqs, &all_inputs)?;
            model.advance_decode_rows(&mut contexts, n_decode)?;
            let out = if single.is_empty() {
                lg
            } else {
                // Original-prefill index → its logit position (one per row).
                let mut pre_logit_idx = vec![0usize; n_prefill_in];
                for (r, &j) in single.iter().enumerate() {
                    pre_logit_idx[j] = n_decode_in + r;
                }
                for (r, &j) in multi.iter().enumerate() {
                    pre_logit_idx[j] = n_decode + r;
                }
                let mut o: Vec<Tensor> = Vec::with_capacity(lg.len());
                o.extend_from_slice(&lg[0..n_decode_in]);
                for j in 0..n_prefill_in {
                    o.push(lg[pre_logit_idx[j]].clone());
                }
                o
            };
            CoreOut::Logits {
                step: WaveStep { logits: Some(out) },
                guard: head_span,
            }
        }
    };
    // Read this wave's assert slots and start a fresh epoch, so each report
    // describes one wave rather than the run so far.
    //
    // This is the ONE synchronisation the instrument costs, and it sits after
    // every launch of the sweep is already enqueued — the caller syncs a moment
    // later to sample anyway, so the wave's work is complete or nearly so by
    // the time this waits on it. That is the whole difference from a probe that
    // reads a scalar back per checkpoint: one fence at the end of a wave rather
    // than a hundred inside it.
    #[cfg(feature = "tensor-assert")]
    {
        let dev = decode_inputs
            .first()
            .or_else(|| prefill_inputs.first())
            .or_else(|| glue_inputs.first())
            .map(|t| t.device().clone());
        if let Some(dev) = dev {
            let bad = candle::tensor_assert::report(&dev)?;
            if !bad.is_empty() {
                tracing::error!(
                    target: "candle_transformers::wave_driver",
                    layer_start, layer_end,
                    decode = decode_seqs.len(),
                    prefill = prefill_seqs.len(),
                    glue = glue_seqs.len(),
                    origin = %bad[0].name,
                    "tensor_assert: this wave produced non-finite values"
                );
            }
            candle::tensor_assert::epoch(&dev)?;
        }
    }

    Ok(out)
}

/// One group of a step's rows: its sequences and their inputs.
#[derive(Clone, Copy)]
pub struct StepRows<'a> {
    pub seqs: &'a [usize],
    pub inputs: &'a [Tensor],
}

impl StepRows<'_> {
    pub const NONE: StepRows<'static> = StepRows {
        seqs: &[],
        inputs: &[],
    };

    /// Caller rows: one per token.
    fn tokens(&self) -> usize {
        caller_rows(self.inputs)
    }
}

/// Caller rows of a group's inputs: one per token.
fn caller_rows(inputs: &[Tensor]) -> usize {
    inputs
        .iter()
        .map(|t| t.dims().get(1).copied().unwrap_or(1))
        .sum()
}

/// Refuses a residual that cannot hold `rows` of this model's residual rows —
/// the wrong row count, width or dtype.
fn check_residual<S: WaveSweep + ?Sized>(
    model: &S,
    session: &BatchedInferenceSession,
    residual: &CreepResidual,
    rows: usize,
) -> Result<()> {
    let (width, dtype) = model.residual_row(session)?;
    let held = residual.committed();
    if !residual_fits(residual, rows, width, dtype) {
        candle::bail!(
            "wave residual: {} rows of {} × {:?} held for {rows} rows of {width} × {dtype:?}",
            residual.rows(),
            held.width(),
            held.dtype()
        );
    }
    Ok(())
}

/// The residual holds exactly `rows` rows of `width` × `dtype`.
fn residual_fits(residual: &CreepResidual, rows: usize, width: usize, dtype: DType) -> bool {
    residual.rows() == rows
        && residual.committed().width() == width
        && residual.committed().dtype() == dtype
}

/// The creep riding a step: its rows, the layers it crosses this step, and the
/// arena its residual is held in between steps.
pub struct StepCreep<'a> {
    pub rows: StepRows<'a>,
    /// First layer this step's window runs.
    pub cursor: usize,
    /// Layer the window stops at.
    pub win_end: usize,
    /// Read when the window resumes (`cursor > 0`); written and committed when
    /// it stops short of the head. A step that needs one and finds none claims
    /// it, sized to the creep's rows, before it places its tier.
    pub residual: &'a mut Option<CreepResidual>,
}

/// One co-batched step: rows that ride every layer, and an optional creep.
pub struct StepSpec<'a> {
    /// One row per sequence, decode kernels.
    pub decode: StepRows<'a>,
    /// Full-sweep rows in the prefill slot: speculative verify blocks.
    pub verify: StepRows<'a>,
    /// Gap-fill rows, which scatter K/V and score nothing.
    pub glue: StepRows<'a>,
    pub creep: Option<StepCreep<'a>>,
}

/// What a step did.
///
/// [`drive_step`] returns `Err` only for a failure of the creep's own segment —
/// its admit or its sweep — after which the creep's window has not run and its
/// arena's committed half is exactly as it was. A failure of a segment that
/// carried only the full-sweep rows lands in [`Self::head`], with
/// [`Self::creep_advanced`] saying whether the creep's window had committed.
pub struct StepOutcome {
    /// The creep's window ran and its residual — or its head — committed.
    pub creep_advanced: bool,
    /// Host wall-clock of the creep's segment, microseconds: the forward the
    /// creep actually rode, apart from the full-sweep rows' segments around it.
    pub creep_us: u64,
    /// The head: logits `[decode | verify | creep]` when the creep reached the
    /// head with the rest, `[decode | verify]` when it stopped short, and none
    /// when nothing rode every layer.
    pub head: Result<WaveResult>,
}

/// **Between forwards:** fit the held ground's arenas to what recent forwards
/// took, so the buffers this forward takes inside itself — a slot-state slab
/// promoted at a 32-token boundary, a sampler pack growing — find room already
/// claimed (`docs/held_ground_elimination.md` §4). Before admission, so the
/// claims stand below the frontier the forward's tier is placed against.
fn provision_held<S: WaveSweep + ?Sized>(model: &S) {
    if let Device::Cuda(cuda) = model.device() {
        provision_held_ground(cuda);
    }
}

/// One forward of a step: its layer range, and whether the creep rides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Segment {
    pub start: usize,
    pub end: usize,
    pub creep: bool,
}

/// A step's forwards, in layer order: the full-sweep rows alone up to the
/// creep's window, with the creep across it, alone again after it. Without
/// full-sweep rows, the creep's window alone.
pub(crate) fn segments(n: usize, cursor: usize, win_end: usize, full: bool) -> Vec<Segment> {
    let seg = |start, end, creep| Segment { start, end, creep };
    if !full {
        return vec![seg(cursor, win_end, true)];
    }
    let mut out = Vec::with_capacity(3);
    if cursor > 0 {
        out.push(seg(0, cursor, false));
    }
    out.push(seg(cursor, win_end, true));
    if win_end < n {
        out.push(seg(win_end, n, false));
    }
    out
}

/// The tier a step places for its segments: each layer phase's span at the
/// widest any segment needs — they reset per layer phase, so the segments
/// share them — and the forward span at the sum, since the step's forward
/// generation does not reset between segments (`docs/wave_step.md` §2.1).
pub(crate) fn step_plan(prices: &[[usize; 3]]) -> [usize; 3] {
    prices.iter().fold([0; 3], |acc, p| {
        [acc[0].max(p[0]), acc[1].max(p[1]), acc[2] + p[2]]
    })
}

/// Drive one co-batched step (`docs/wave_step.md` §2).
///
/// The full-sweep rows cross every layer; the creep crosses its window. Every
/// segment's KV is admitted before the step's tier is placed, the tier is
/// placed once, and the segments run inside one forward span, the full-sweep
/// rows' residual handed from segment to segment on it.
pub fn drive_step<S: WaveSweep + ?Sized>(
    model: &S,
    session: &mut BatchedInferenceSession,
    spec: StepSpec<'_>,
) -> Result<StepOutcome> {
    let StepSpec {
        decode,
        verify,
        glue,
        creep,
    } = spec;
    let n = model.num_layers();
    let full = !decode.seqs.is_empty() || !verify.seqs.is_empty() || !glue.seqs.is_empty();
    // A step without a creep is one forward, which provisions for itself.
    if creep.is_some() {
        provision_held(model);
    }
    let Some(creep) = creep else {
        let head = drive_wave(
            model,
            session,
            decode.seqs,
            decode.inputs,
            verify.seqs,
            verify.inputs,
            glue.seqs,
            glue.inputs,
            0,
            n,
            None,
        );
        return Ok(StepOutcome {
            creep_advanced: false,
            creep_us: 0,
            head,
        });
    };
    let StepCreep {
        rows: creep_rows,
        cursor,
        win_end,
        residual,
    } = creep;
    if cursor >= win_end || win_end > n {
        candle::bail!("wave step: a creep window [{cursor}, {win_end}) of {n} layers");
    }
    let head = decode.seqs.len() + verify.tokens();
    let creep_tok = creep_rows.tokens();
    let glue_tok = glue.tokens();
    let stops_short = win_end < n;
    let (width, dtype) = model.residual_row(session)?;
    // A creep starting at layer 0 reads nothing from its arena, so one left by
    // an earlier attempt whose rows no longer match — a group that re-formed —
    // is given back rather than carried into the move-out.
    if cursor == 0
        && residual
            .as_ref()
            .is_some_and(|r| !residual_fits(r, creep_tok, width, dtype))
    {
        *residual = None;
    }
    // The creep's arena, claimed now if the window needs one: between forwards.
    // The previous forward's tier is handed back first — it stands past its
    // guards until the next forward's prologue — so the claim is measured
    // against a pool no forward holds any of (`docs/wave_step.md` §3).
    if stops_short && residual.is_none() {
        if let Device::Cuda(cuda) = model.device() {
            end_wave_transient(&cuda.cuda_stream());
        }
        *residual = Some(CreepResidual::claim(
            model.device(),
            creep_tok,
            width,
            dtype,
        )?);
    }
    // Every shape the move-out depends on is settled here, before any segment
    // runs: past its sweep, a refusal would come after the window's recurrent
    // and KV state had committed.
    match residual.as_ref() {
        Some(r) => check_residual(model, session, r, creep_tok)?,
        None if cursor > 0 => {
            candle::bail!("wave step: a creep resuming at layer {cursor} holds no residual")
        }
        None => {}
    }

    let segs = segments(n, cursor, win_end, full);
    let mid_seqs: Vec<usize> = verify.seqs.iter().chain(creep_rows.seqs).copied().collect();
    let mid_inputs: Vec<Tensor> = verify
        .inputs
        .iter()
        .chain(creep_rows.inputs)
        .cloned()
        .collect();
    let rows_of = |s: &Segment| GroupRows {
        decode_seqs: decode.seqs,
        decode_inputs: decode.inputs,
        prefill_seqs: if s.creep { &mid_seqs } else { verify.seqs },
        prefill_inputs: if s.creep { &mid_inputs } else { verify.inputs },
        glue_seqs: glue.seqs,
        glue_inputs: glue.inputs,
    };
    let pending_glue = take_pending_glue(session, glue.inputs);
    // A failure before the creep's window has run, in a segment that carried
    // only the full-sweep rows: reported in the outcome's head, not as the
    // creep's failure.
    let full_sweep_failed = |e: Error| StepOutcome {
        creep_advanced: false,
        creep_us: 0,
        head: Err(e),
    };

    // ── Prepare: every segment's prologue and admit, priced. ──
    let mut prices = Vec::with_capacity(segs.len());
    for s in &segs {
        match forward_core(
            model,
            session,
            rows_of(s),
            (s.start, s.end),
            pending_glue.clone(),
            None,
            SweepMode::Prepare,
        ) {
            Ok(CoreOut::Priced(p)) => prices.push(p),
            Ok(_) => candle::bail!("wave step: a segment being prepared ran"),
            Err(e) if s.creep => return Err(e),
            Err(e) => return Ok(full_sweep_failed(e)),
        }
    }

    // ── Place the step's tier and open its forward span. ──
    //
    // Not for a model that priced every segment at nothing: it keeps no buffer
    // on the tier, and a live span would refuse the arena claims it makes inside
    // its own sweeps. A placement the partition refuses fails the creep's window
    // with the rest: nothing has run.
    let plan = step_plan(&prices);
    let step_span = match model.device() {
        Device::Cuda(cuda) if plan != [0; 3] => {
            let stream = cuda.cuda_stream();
            // The refusal names the partition; what the step's segments priced
            // — each `[layer range): [attention, ffn, forward]` — is what says
            // which part of the step the admission that composed it missed.
            plan_wave_transient(&stream, plan).map_err(|e| {
                let segments: Vec<String> = segs
                    .iter()
                    .zip(&prices)
                    .map(|(s, p)| {
                        format!(
                            "[{}, {}){}: {:?}",
                            s.start,
                            s.end,
                            if s.creep { " creep" } else { "" },
                            p
                        )
                    })
                    .collect();
                // And who holds the span: every slot tenant's regions and the
                // held ground's, with how high it reaches — a frontier that rose
                // under a sweep while holes stood below it rose by someone's
                // contiguous claim, and this names whose.
                let census: Vec<String> = candle_nn::kv_cache::SlotTenant::ALL
                    .iter()
                    .map(|&t| {
                        format!(
                            "{}={}",
                            t.label(),
                            candle_nn::kv_cache::arena_regions(model.device(), t)
                        )
                    })
                    .collect();
                let held = candle_nn::kv_cache::held_ground_stats(stream.context().ordinal())
                    .map_or(String::new(), |h| {
                        format!(
                            ", held ground {} arenas {} B reaching region {}",
                            h.arenas, h.arena_bytes, h.arena_top
                        )
                    });
                candle::Error::Msg(format!(
                    "{e} — for a step of {} segments {}, placed {plan:?} B; tenants {}{held}",
                    segs.len(),
                    segments.join(" "),
                    census.join(" "),
                ))
            })?;
            Some(begin_wave(&stream, LayerPhase::Forward)?)
        }
        _ => None,
    };
    let ticket = step_span.as_ref().map(|g| g.ticket());

    // ── Run: each segment inside the step's span. ──
    let mut prev: Option<(SpanRows, RowMap)> = None;
    let mut creep_advanced = false;
    let mut creep_us = 0;
    for (i, s) in segs.iter().enumerate() {
        let last = i + 1 == segs.len();
        let width = |t: &Tensor, layout: &RowMap| t.elem_count() / layout.rows().max(1);
        let result = {
            let creep_in = residual
                .as_ref()
                .filter(|_| s.creep && s.start > 0)
                .map(|r| Part {
                    store: r.committed() as &dyn RowStore,
                    layout: r.layout(),
                    start: 0,
                    len: creep_tok,
                });
            let handed_on = prev.as_ref().map(|(rows, layout)| HandedOn {
                store: rows as &dyn RowStore,
                layout,
                carried_creep: segs[i - 1].creep,
            });
            let input = segment_parts(handed_on, creep_in, head, creep_tok, glue_tok);
            let t = Instant::now();
            let r = forward_core(
                model,
                session,
                rows_of(s),
                (s.start, s.end),
                pending_glue.clone(),
                input,
                SweepMode::InStep(ticket),
            );
            if s.creep {
                creep_us = t.elapsed().as_micros() as u64;
            }
            r
        };
        let out = match result {
            Ok(out) => out,
            Err(e) if s.creep => return Err(e),
            Err(e) => {
                return Ok(StepOutcome {
                    creep_advanced,
                    creep_us,
                    head: Err(e),
                })
            }
        };
        match out {
            CoreOut::Residual {
                working, layout, ..
            } => {
                // A full-sweep segment's failure is the head's, as in its sweep.
                let rows = match SpanRows::new(&working, width(&working, &layout)) {
                    Ok(rows) => rows,
                    Err(e) if s.creep => return Err(e),
                    Err(e) => {
                        return Ok(StepOutcome {
                            creep_advanced,
                            creep_us,
                            head: Err(e),
                        })
                    }
                };
                if s.creep {
                    // The window stopped short: its rows go to the pending half,
                    // and the window commits.
                    let r = residual
                        .as_mut()
                        .expect("claimed above for a window that stops short");
                    move_rows(&rows, &layout, head, r.pending(), &r.layout(), 0, creep_tok)?;
                    r.commit();
                    creep_advanced = true;
                }
                if last {
                    return Ok(StepOutcome {
                        creep_advanced,
                        creep_us,
                        head: Ok(WaveResult::owned(WaveStep { logits: None })),
                    });
                }
                prev = Some((rows, layout));
            }
            CoreOut::Logits { step, guard } => {
                if s.creep {
                    // The creep reached the head with the rest: its sweep is
                    // done, and its arena goes with it.
                    *residual = None;
                    creep_advanced = true;
                }
                // The logits are on the step's span; the result holds it.
                return Ok(StepOutcome {
                    creep_advanced,
                    creep_us,
                    head: Ok(WaveResult::on_span(step, step_span.or(guard))),
                });
            }
            CoreOut::Priced(_) => candle::bail!("wave step: a running segment came back priced"),
        }
    }
    candle::bail!("wave step: no segment ran")
}

#[cfg(test)]
mod step_tests {
    use super::*;

    fn seg(start: usize, end: usize, creep: bool) -> Segment {
        Segment { start, end, creep }
    }

    #[test]
    fn a_creep_mid_stack_splits_the_step_in_three() {
        assert_eq!(
            segments(40, 10, 20, true),
            vec![seg(0, 10, false), seg(10, 20, true), seg(20, 40, false)]
        );
    }

    #[test]
    fn a_creep_from_layer_zero_to_the_head_is_one_segment() {
        assert_eq!(segments(40, 0, 40, true), vec![seg(0, 40, true)]);
    }

    #[test]
    fn a_creep_reaching_the_head_ends_the_step() {
        assert_eq!(
            segments(40, 30, 40, true),
            vec![seg(0, 30, false), seg(30, 40, true)]
        );
    }

    #[test]
    fn a_creep_alone_runs_only_its_window() {
        assert_eq!(segments(40, 10, 20, false), vec![seg(10, 20, true)]);
    }

    /// Layer-phase spans at the widest segment; the forward span at the sum.
    #[test]
    fn the_step_tier_shares_layer_phases_and_sums_forward_spans() {
        assert_eq!(
            step_plan(&[[10, 20, 5], [30, 15, 7], [5, 5, 3]]),
            [30, 20, 15]
        );
        assert_eq!(step_plan(&[]), [0, 0, 0]);
    }
}
