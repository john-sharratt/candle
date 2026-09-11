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

use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::WaveWidth;

use super::batched_inference::{
    pack_prefill_slabs, prefill_slack_cap, BatchedInferenceSession, PendingGlue, WaveResult,
    WaveStep,
};
use super::batched_model::{WaveGuard, WavePhase};
use super::kv_cache_utils::SequenceContext;
use super::profile::gpu_span;
use super::tensor_cat::TensorCat;
use candle::quantized::pinned_staging::Generation;

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
    /// A paused wave's residual stream, already permuted into internal order.
    pub x_in: Option<TensorCat>,
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

    /// Widest prefill this model runs in one forward, in tokens, with `head`
    /// already in the wave ahead of it (decode rows and verify blocks, which
    /// share the same transient tier) and `tier_budget` bytes of ground the
    /// tier may be priced against.
    ///
    /// `head` is a whole [`WaveWidth`] rather than a row count because the
    /// phases it widens are not the same ones the prefill widens: its decode
    /// rows price the decode chain, its scored rows price the head, and neither
    /// is a number the prefill's own rows could stand in for.
    fn prefill_width_cap(&self, act_dtype: DType, head: WaveWidth, tier_budget: usize) -> usize;

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

    /// Run `[layer_start, layer_end)` over the wave, returning the residual
    /// (range stopped short of the head) or the logits (range reached it).
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
    residual_in: Option<Tensor>,
) -> Result<WaveResult> {
    if decode_inputs.len() != decode_seqs.len()
        || prefill_inputs.len() != prefill_seqs.len()
        || glue_inputs.len() != glue_seqs.len()
    {
        candle::bail!("forward_wave: input/seq length mismatch");
    }

    // Bound a single forward's token count: a PURE prefill full sweep whose
    // total tokens exceed the budget is split into token-bounded sub-forwards
    // and its logits concatenated. Only applies with no decode/glue rows, a
    // full `[0, N)` sweep, and no resumed residual — the co-batched /
    // re-entrant paths are bounded by the scheduler's admission window + OOM
    // retry instead.
    let num_layers = model.num_layers();
    if decode_seqs.is_empty()
        && glue_seqs.is_empty()
        && residual_in.is_none()
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
        // No head rows: this branch is entered only when the wave carries no
        // decode or glue rows, so the whole tier is the prefill's. The budget
        // is the one the fill that admitted these prefills priced them against,
        // so a group the fill composed is not re-sliced here.
        let width_cap = model.prefill_width_cap(
            session.activation_dtype(),
            WaveWidth::default(),
            session.tier_budget_bytes(),
        );
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
                    .ok_or_else(|| candle::Error::Msg("forward_wave slice: no logits".into()))?;
                // Copied off the span, not moved off it. Each slice is a whole
                // forward and reclaims its own forward span when `step` drops
                // at the end of this iteration — so a borrowed logits row would
                // be reading recycled bytes by the time the next slice ran. (It
                // could not even get that far: the span refuses a second live
                // generation, so slice two would fail to open one while slice
                // one still held it.)
                //
                // This is the sanctioned escape and it really copies. It is
                // confined to the slicing path, which is already paying for N
                // forwards, and it is why the value returned below is owned.
                for t in lg {
                    all_logits.push(t.to_owned_tensor()?);
                }
            }
            return Ok(WaveResult::owned(WaveStep {
                residual: None,
                logits: Some(all_logits),
            }));
        }
    }

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

    // Glue staging: taken HERE — the one `&mut`-session step the sweep's shared
    // borrows could not perform — and handed through raw; the sweep decides
    // what kernel metadata to build from it. Staged glue is consumed only by a
    // wave that carries glue rows. The descriptors are staged immediately
    // before the gap-fill forward they describe — but that forward can die
    // before reaching this point, and the staging then sits on the session for
    // whatever wave comes next, which is how a dead wave's leftovers killed the
    // titler twice. Stale staging is dropped, loudly: the reproject that staged
    // it re-stages when its own retry runs.
    let pending_glue: Option<Vec<PendingGlue>> = if glue_lens.is_empty() {
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
    };

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

    // Residual token order. The sweep packs per-token hidden states in INTERNAL
    // order `[orig-decode | single-prefills | multi-prefills | glue]` (the
    // single-token prefills were folded into the decode group). The residual
    // crosses the API boundary in CALLER order `[decode | prefill (caller
    // order) | glue]` so a co-batched caller can split it by contiguous group —
    // decode, section, cohort, glue — which is what lets a creeping cohort be
    // held whole across a wave while the full-sweep members continue. We
    // reorder caller→internal on the way in and internal→caller on the way out;
    // the two permutations are exact inverses, so re-feeding the returned
    // residual on the next layer window round-trips. When there are no
    // single-token prefills the two orders coincide (the multis keep caller
    // order), so the permutation is identity and we skip it.
    let token_perm: Option<(Tensor, Tensor)> = if single.is_empty() {
        None
    } else {
        let mut single_rank = vec![usize::MAX; n_prefill_in];
        for (r, &i) in single.iter().enumerate() {
            single_rank[i] = r;
        }
        let mut multi_tok_start = vec![0usize; n_prefill_in];
        let mut acc = n_decode;
        for &i in &multi {
            multi_tok_start[i] = acc;
            acc += pre_lens_in[i];
        }
        let glue_internal_base = acc;
        let glue_tok: usize = glue_lens.iter().sum();
        let total_tok = glue_internal_base + glue_tok;
        let mut internal_of_caller: Vec<u32> = Vec::with_capacity(total_tok);
        for t in 0..n_decode_in {
            internal_of_caller.push(t as u32);
        }
        for j in 0..n_prefill_in {
            if single_rank[j] != usize::MAX {
                internal_of_caller.push((n_decode_in + single_rank[j]) as u32);
            } else {
                let start = multi_tok_start[j];
                for t in 0..pre_lens_in[j] {
                    internal_of_caller.push((start + t) as u32);
                }
            }
        }
        for t in 0..glue_tok {
            internal_of_caller.push((glue_internal_base + t) as u32);
        }
        let mut caller_of_internal = vec![0u32; total_tok];
        for (c, &k) in internal_of_caller.iter().enumerate() {
            caller_of_internal[k as usize] = c as u32;
        }
        let i2c = Tensor::from_vec(internal_of_caller, total_tok, dev)?;
        let c2i = Tensor::from_vec(caller_of_internal, total_tok, dev)?;
        Some((i2c, c2i))
    };

    let x_in = match (residual_in, token_perm.as_ref()) {
        (Some(t), Some((_, c2i))) => {
            // Caller order → internal order for the resume. Tokens are dim 1
            // (`[batch, tokens, hidden]`).
            Some(TensorCat::from_cat_tensor(t.index_select(c2i, 1)?, 0)?)
        }
        (Some(t), None) => Some(TensorCat::from_cat_tensor(t, 0)?),
        (None, _) => None,
    };
    // The layer sweep plus the head, so the forward's stream time divides into
    // "the model" and "everything the driver and its caller do around it". The
    // gap between this and the caller's own forward span is where a per-row copy
    // off the wave arena hid 76% of a 128-slot decode step.
    let g_sweep = gpu_span("wv:sweep", dev);
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
            act_dtype,
        },
    );
    g_sweep.end();
    // **A failed wave leaves no trace.** The layer sweep advances each layer's
    // usage as that layer completes, so an error anywhere in it — and the relief
    // design treats failing a wave as routine — leaves the early layers one
    // token ahead of the rest. The rollback restores every row to its entry
    // length on every layer of the range, which is what makes the retry a retry
    // rather than a decode against per-layer token windows. This is the single
    // choke point every wave goes through; the sweep itself stays free to
    // advance eagerly, because whatever it did is undone here on the way out.
    let (phase, head_span) = match wave {
        Ok(v) => v,
        Err(e) => {
            // The sweep's mutable session borrow ended with it, so the rollback
            // re-assembles the contexts it needs. An assembly failure here is
            // the same unrecoverable shape as a failed rollback: report both.
            let (kv_start, kv_end) = model.kv_layer_range(layer_start, layer_end);
            let rolled = assemble_wave_contexts(session, &all_seqs, &all_inputs)
                .and_then(|mut contexts| model.rollback_wave(&mut contexts, kv_start, kv_end));
            if let Err(rb) = rolled {
                candle::bail!(
                    "wave failed ({e}) and the KV rollback that keeps that failure \
                     recoverable also failed ({rb}) — the affected sequences may hold \
                     per-layer token windows"
                )
            }
            return Err(e);
        }
    };
    // Output ordering. The single-token prefills were folded into the decode
    // group, so the internal row order is
    // `[orig-decode | single-prefills | multi-prefills | glue]`.
    //
    // - Logits (final, head ran) are restored to the caller's
    //   `[decode | prefill-in-caller-order]` so `pf_logits[k]` aligns with the
    //   caller's `pf_seqs[k]`.
    // - The intermediate residual is returned in CALLER order (see the
    //   `token_perm` note above): internal→caller on the way out, caller→internal
    //   on the way back in, exact inverses that round-trip across layer windows.
    let step = match phase {
        WavePhase::Residual(x) => {
            // Internal order → caller order. Tokens are dim 1.
            let res = match token_perm.as_ref() {
                Some((i2c, _)) => x.to_tensor().index_select(i2c, 1)?,
                None => x.to_tensor(),
            };
            WaveStep {
                residual: Some(res),
                logits: None,
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
            WaveStep {
                residual: None,
                logits: Some(out),
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

    // The head's outputs sit on the forward span, so the guard goes back with
    // them: `WaveResult` is what stops the span being reclaimed while the caller
    // still holds the logits.
    #[cfg(feature = "cuda")]
    {
        Ok(WaveResult::on_span(step, head_span))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = head_span;
        Ok(WaveResult::owned(step))
    }
}
