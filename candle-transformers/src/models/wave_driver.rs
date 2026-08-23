//! The model-agnostic half of a co-batched wave.
//!
//! A wave forward splits cleanly in two. The **outer** half — bounding a
//! forward's token count, routing 1-token prefills to the decode kernel,
//! building the three groups' attention metadata, assembling the context list,
//! permuting tokens between caller order and internal order, rolling the KV
//! back when the sweep fails, and advancing the decode rows once the head has
//! run — depends on nothing about the model except its depth and its device.
//! The **inner** half, the layer sweep itself, is where a model's architecture
//! actually lives.
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

#[cfg(feature = "cuda")]
use super::batched_inference::build_glue_meta;
use super::batched_inference::{
    pack_prefill_slabs, prefill_slack_cap, BatchedInferenceSession, WaveResult, WaveStep,
};
use super::batched_layer::{BatchedPrefillMeta, DecodeHeaders};
use super::batched_model::{WaveGuard, WavePhase};
use super::kv_cache_utils::SequenceContext;
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
    pub decode_headers: DecodeHeaders,
    pub prefill_headers: DecodeHeaders,
    pub glue_headers: DecodeHeaders,
    /// Pinned-stager generation guarding this wave's kernel metadata uploads.
    pub generation: &'a Generation,
    pub layer_start: usize,
    pub layer_end: usize,
    /// A paused wave's residual stream, already permuted into internal order.
    pub x_in: Option<TensorCat>,
}

/// The per-model half of a wave: run one layer range over assembled contexts.
pub trait WaveSweep {
    fn device(&self) -> &Device;

    /// Transformer depth — what bounds a wave's layer range. On a hybrid this
    /// is the trunk depth, not the KV-layer count.
    fn num_layers(&self) -> usize;

    /// Widest prefill this model runs in one forward, in tokens.
    fn prefill_width_cap(&self, act_dtype: DType) -> usize;

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

    /// Run `[layer_start, layer_end)` over `contexts`, returning the residual
    /// (range stopped short of the head) or the logits (range reached it).
    fn sweep(
        &self,
        contexts: &mut [SequenceContext],
        groups: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)>;
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

    // Forward-entry invariant: every member's logical offset must equal the
    // token count its live block table covers. The varlen metadata
    // (`cu_seqlens`/`kv_lens`) below is built from the offsets, while the
    // per-layer slot headers are built from the block tables — the attention
    // kernels resolve every `[0, kv_len)` position through the table, so any
    // divergence walks them past the slot's span in the packed staged uploads
    // (garbage slice indices → wild record pointers → CUDA_ERROR_ILLEGAL_ADDRESS,
    // or silent cross-slot reads). Offsets run AHEAD of the backing when a
    // projection drops sections it could not lift to hot under VRAM pressure;
    // they run BEHIND after glue reserves gap chunks the wave didn't reflect.
    // Positions are slot-relative (slice ropes), so the backing length is also
    // the correct RoPE base either way. This is the choke point every forward
    // passes through — wave steps, deferred projection gap-fills, and probes
    // alike.
    for ids in [decode_seqs, prefill_seqs, glue_seqs] {
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

    // Per-group offsets + query lengths.
    let dev = model.device();
    let seq_off = |session: &BatchedInferenceSession, ids: &[usize]| -> Vec<usize> {
        ids.iter()
            .map(|&i| session.sequence_offset(i).unwrap_or(0))
            .collect()
    };
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

    let pre_off = seq_off(session, &proc_prefill_seqs);
    let glue_off = seq_off(session, glue_seqs);
    let pre_lens = input_len(&proc_prefill_inputs);
    let glue_lens = input_len(glue_inputs);

    // Build the three groups' attention headers. Decode gets its packed
    // SlotHeader buffer; prefill/glue get ragged cu_seqlens; glue additionally
    // carries the staged per-token scatter descriptors.
    #[cfg(feature = "cuda")]
    let (_pm_guard, decode_headers) = if n_decode > 0 {
        let (pm_guard, buf, stride) =
            session.build_decode_metadata(&proc_decode_seqs, &stager_generation)?;
        (pm_guard, DecodeHeaders::Decode { buf, stride })
    } else {
        (
            None,
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            },
        )
    };
    #[cfg(not(feature = "cuda"))]
    let decode_headers = DecodeHeaders::Decode {
        buf: None,
        stride: 0,
    };

    let prefill_headers =
        DecodeHeaders::Prefill(BatchedPrefillMeta::new_ragged(&pre_off, &pre_lens, dev)?);

    #[allow(unused_mut)]
    let mut glue_meta = BatchedPrefillMeta::new_ragged(&glue_off, &glue_lens, dev)?;
    // Staged glue is consumed only by a wave that carries glue rows. The
    // descriptors are staged immediately before the gap-fill forward they
    // describe — but that forward can die before reaching this point (its
    // decode metadata refuses first), and the staging then sits on the session
    // for whatever wave comes next. A glue-less wave that takes it fails
    // `build_glue_meta` with "N glue descriptors vs 0 input_lens", which is how
    // a dead wave's leftovers killed the titler twice. Stale staging is
    // dropped, loudly: the reproject that staged it re-stages when its own
    // retry runs.
    #[cfg(feature = "cuda")]
    if glue_lens.is_empty() {
        if let Some(stale) = session.take_pending_glue() {
            tracing::warn!(
                n = stale.len(),
                "dropping glue descriptors staged by a wave that never ran its \
                 gap-fill forward — this wave carries no glue rows"
            );
        }
    } else if let Some(pending) = session.take_pending_glue() {
        glue_meta.glue = build_glue_meta(pending, &glue_lens, dev)?;
    }
    let glue_headers = DecodeHeaders::Prefill(glue_meta);

    // Assemble the combined context list in [decode | prefill | glue] order.
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
    let mut all_inputs: Vec<&Tensor> = Vec::with_capacity(all_seqs.len());
    all_inputs.extend(proc_decode_inputs.iter());
    all_inputs.extend(proc_prefill_inputs.iter());
    all_inputs.extend(glue_inputs.iter());
    let all_lens: Vec<usize> = all_inputs
        .iter()
        .map(|t| t.dims().get(1).copied().unwrap_or(1))
        .collect();

    let mut caches_data = session.caches_for_sequences_mut(&all_seqs);
    // `caches_for_sequences_mut` SILENTLY skips slots that are `None`, so a
    // sequence released between wave-group formation and this forward yields a
    // short `contexts`. That used to surface ~100 lines later as
    // `group bounds exceed batch` — a `checked_sub` underflow — which names
    // neither the real fault nor the sequence responsible. Fail here instead,
    // where the missing slots are still known. (A duplicate index in `all_seqs`
    // collapses the same way, since the lookup is set-based; this catches that
    // too.)
    if caches_data.len() != all_seqs.len() {
        let live: std::collections::HashSet<usize> =
            caches_data.iter().map(|(i, _, _)| *i).collect();
        let missing: Vec<usize> = all_seqs
            .iter()
            .copied()
            .filter(|s| !live.contains(s))
            .collect();
        candle::bail!(
            "forward_wave: {} sequences requested but only {} have live slots \
             (missing/duplicated: {missing:?}) — the wave group named a sequence \
             the scheduler has since released",
            all_seqs.len(),
            caches_data.len(),
        );
    }
    let mut contexts: Vec<SequenceContext<'_>> = Vec::with_capacity(all_seqs.len());
    for (i, (_seq_idx, offset, caches)) in caches_data.iter_mut().enumerate() {
        contexts.push(SequenceContext {
            offset: *offset,
            kv_caches: caches,
            input_ids: all_inputs[i],
            input_len: all_lens[i],
        });
    }

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
    let wave = model.sweep(
        &mut contexts,
        WaveGroups {
            n_decode,
            n_prefill,
            seq_ids: &all_seqs,
            decode_headers,
            prefill_headers,
            glue_headers,
            generation: &stager_generation,
            layer_start,
            layer_end,
            x_in,
        },
    );
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
            let (kv_start, kv_end) = model.kv_layer_range(layer_start, layer_end);
            if let Err(rb) = super::wave_admit::rollback_wave_kv(&mut contexts, kv_start, kv_end) {
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
            for i in 0..n_decode {
                let offset = contexts[i].offset;
                let mut advance = || -> Result<()> {
                    for cache in contexts[i].kv_caches.caches.iter_mut() {
                        cache.set_current_seq_len(offset + 1)?;
                    }
                    Ok(())
                };
                if let Err(e) = advance() {
                    for c in contexts[..=i].iter_mut() {
                        let off = c.offset;
                        let _ = c
                            .kv_caches
                            .caches
                            .iter_mut()
                            .try_for_each(|cache| cache.truncate_to_offset(off));
                    }
                    return Err(e);
                }
            }
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
