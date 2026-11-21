//! Runs one layer range of a hybrid Gated-DeltaNet stack, dispatching each layer
//! on its kind and carrying recurrent state across waves.
//!
//! Each sequence's recurrent state is lifted out of the model for the sweep and
//! put back unconditionally, so a failed sweep cannot leave its sequences
//! stateless for every later wave. The admit phase claims chunks over the KV
//! layers only, since three quarters of the trunk layers own no cache. The
//! transient tier is priced for the rows this wave runs, not for the widest wave
//! the engine can take. The residual is checked on entry to a layer, after its
//! mixer and after its FFN, so a bad value is blamed on the layer it came from.

use super::*;

impl WaveSweep for HybridBatched {
    fn device(&self) -> &Device {
        HybridBatched::device(self)
    }

    fn verify_row_seqs(&self) -> Vec<usize> {
        HybridBatched::verify_row_seqs(self).unwrap_or_default()
    }

    fn num_layers(&self) -> usize {
        HybridBatched::num_layers(self)
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        <Self as ManagedBatchedModel>::prefill_width_cap(self, act_dtype)
    }

    /// The single-stream residual: `hidden` wide, in the wave's activation width.
    fn residual_row(&self, session: &BatchedInferenceSession) -> Result<(usize, DType)> {
        let dtype = session.activation_dtype();
        Ok((
            <Self as ManagedBatchedModel>::wave_geometry(self, dtype).hidden,
            dtype,
        ))
    }

    fn kv_layer_range(&self, layer_start: usize, layer_end: usize) -> (usize, usize) {
        HybridBatched::kv_layer_range(self, layer_start, layer_end)
    }

    fn take_device_fault(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        if let Some(c) = self.model().experts.as_ref() {
            c.take_segment_fault(HybridBatched::device(self))?;
        }
        Ok(())
    }

    /// Open the wave's recurrent state, sweep, then commit or roll back.
    ///
    /// The state is lifted out of the model's map for the duration and put back
    /// unconditionally: a sweep that failed with the stores still lifted would
    /// leave those sequences stateless for every later wave, which is a worse
    /// failure than the one that caused it.
    fn sweep(
        &self,
        session: &mut BatchedInferenceSession,
        wave: WaveGroups<'_>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        let seqs = wave.seq_ids.to_vec();
        let mode = wave.mode;

        // A segment of a step skips everything up to the recurrent bracket: the
        // step ran it — the tier hand-back, the boundary, the state claims — for
        // every segment before it placed the tier they share
        // (`docs/wave_step.md` §2.1), and a claim under that tier is refused.
        if !matches!(mode, SweepMode::InStep(_)) {
            let q_lens: Vec<usize> = wave
                .inputs
                .iter()
                .map(|t| t.dims().get(1).copied().unwrap_or(1))
                .collect();
            self.prologue(session, &seqs, &q_lens)?;
        }
        // A segment being prepared prices itself and runs nothing, so it opens
        // no recurrent wave and lifts no store.
        if matches!(mode, SweepMode::Prepare) {
            return sweep_layers(self, session, wave, &mut []);
        }
        self.begin_recurrent_wave(&seqs)?;
        let mut stores = match self.take_recurrent(&seqs) {
            Ok(s) => s,
            Err(e) => {
                // Nothing was lifted, so there is nothing to put back — but the
                // wave IS open and must be closed or the next one is refused.
                let _ = self.rollback_recurrent_wave(&seqs);
                return Err(e);
            }
        };

        let swept = {
            let mut refs: Vec<&mut RecurrentStateStore> = stores.iter_mut().collect();
            sweep_layers(self, session, wave, &mut refs)
        };

        self.put_recurrent(&seqs, stores)?;
        match &swept {
            // A close that fails commits nothing (`commit_stores`), so the wave
            // is rolled back like any other that failed, and leaves no trace.
            Ok(_) => {
                if let Err(e) = self.commit_recurrent_wave(&seqs) {
                    self.rollback_recurrent_wave(&seqs)?;
                    return Err(e);
                }
            }
            Err(_) => self.rollback_recurrent_wave(&seqs)?,
        }
        swept
    }
}

/// The layer sweep proper.
pub(super) fn sweep_layers(
    model: &HybridBatched,
    session: &mut BatchedInferenceSession,
    wave: WaveGroups<'_>,
    stores: &mut [&mut RecurrentStateStore],
) -> Result<(WavePhase, Option<WaveGuard>)> {
    let WaveGroups {
        n_decode,
        n_prefill,
        seq_ids,
        inputs,
        pending_glue,
        generation,
        layer_start,
        layer_end,
        x_in,
        mode,
        act_dtype,
        adapter,
    } = wave;
    if seq_ids.is_empty() {
        candle::bail!("qwen35 wave: empty batch");
    }
    let preparing = matches!(mode, SweepMode::Prepare);
    let in_step = matches!(mode, SweepMode::InStep(_));
    let q = model.model();
    // Resolved once for the sweep, not per layer: an unknown name fails here,
    // before any layer has run, rather than after the wave has half-decoded.
    let adapter = model.adapter(adapter)?;
    let num_layers = q.cfg.num_layers;
    if layer_start > layer_end || layer_end > num_layers {
        candle::bail!(
            "qwen35 wave: bad layer range [{layer_start}, {layer_end}) over {num_layers} layers"
        );
    }
    let n_glue = seq_ids
        .len()
        .checked_sub(n_decode + n_prefill)
        .ok_or_else(|| candle::Error::Msg("qwen35 wave: group bounds exceed batch".into()))?;
    // The gap-fill kernel is `head_dim 128` only and the float prefill fallback
    // carries no glue masking, so a glue row would be attended as an ordinary
    // prefill token — a wrong answer, not a slow one. `pending_glue` can only
    // arrive with glue rows, so the one check covers both.
    if n_glue > 0 || pending_glue.is_some() {
        candle::bail!(
            "qwen35 wave: {n_glue} glue rows — reprojection glue is not implemented at \
             head_dim {}; this stack must recompute rather than gap-fill",
            q.cfg.attn_head_dim
        );
    }

    // Offsets + query lengths from the session, BEFORE the contexts borrow it.
    let offsets: Vec<usize> = seq_ids
        .iter()
        .map(|&s| session.sequence_offset(s).unwrap_or(0))
        .collect();
    let q_lens: Vec<usize> = inputs
        .iter()
        .map(|t| t.dims().get(1).copied().unwrap_or(1))
        .collect();
    let (dec_off, pre_off) = offsets.split_at(n_decode);
    let (dec_q, pre_q) = q_lens.split_at(n_decode);
    let pre_rows: usize = pre_q.iter().sum();
    let total_rows = n_decode + pre_rows;

    // Attention metadata, from the session's shared borrow — the hybrid's arena
    // state does not move before the layer loop reads it, so building here
    // matches the order the wave driver used when it built these.
    //
    // Spanned because it is the widest-scaling piece of the sweep's setup: one
    // slot header per decode sequence, so it is ~1 ms at a handful of slots and
    // ~20 ms at 128.
    let accept_in_place = session.accept_in_place();
    let stencil_rows = session.stencil_rows();
    let g_meta = crate::models::profile::gpu_span("fwd:meta", model.device());
    // A segment being prepared runs no layer, so it builds no header — but it
    // claims the decode rows' write chunks the build would, here, before the
    // step's span stands.
    #[cfg(feature = "cuda")]
    if preparing {
        session.ensure_decode_writes(&seq_ids[..n_decode])?;
    }
    #[cfg(feature = "cuda")]
    let decode_headers = if n_decode > 0 && !preparing {
        let (buf, stride) = session.build_decode_metadata(&seq_ids[..n_decode], generation)?;
        DecodeHeaders::Decode { buf, stride }
    } else {
        DecodeHeaders::Decode {
            buf: None,
            stride: 0,
        }
    };
    #[cfg(not(feature = "cuda"))]
    let decode_headers = DecodeHeaders::Decode {
        buf: None,
        stride: 0,
    };
    g_meta.end();

    let mut contexts = assemble_wave_contexts(session, seq_ids, inputs)?;
    let contexts = contexts.as_mut_slice();

    // The wave's declared width — carried, not re-derived. See
    // `WaveGroups::act_dtype`. The embedding emits this and every kernel after
    // it emits what its consumer reads, so nothing on the path converts.
    let embed_dtype = act_dtype;
    let dev = model.device();
    // The concrete device the same-pass checkpoints launch and fence on. Bound
    // once here rather than matched at each site, so adding a checkpoint is one
    // line.
    #[cfg(feature = "tensor-assert")]
    let capture_dev = match dev {
        Device::Cuda(d) => d,
        _ => candle::bail!("qwen35 wave: tensor-assert checkpoints require a CUDA device"),
    };

    // **Phase 1: admit** — claim every KV chunk this wave will write before a
    // byte of it computes, so the arena frontier is final when the transient
    // tier is placed against it. Over the **KV** range: three quarters of the
    // trunk range owns no cache at all, and the draft head's layer is one past
    // the end of it (see [`HybridBatched::kv_layer_range`]).
    //
    // A segment of a step was admitted when the step prepared it, before the
    // tier its segments share was placed (`docs/wave_step.md` §2.1).
    let (kv_start, kv_end) = model.kv_layer_range(layer_start, layer_end);
    let head_kv = model.mtp_kv_layer().filter(|_| layer_end == num_layers);
    if !in_step {
        admit_wave_kv(contexts, n_decode, n_prefill, kv_start, kv_end)?;
    }

    // **Phase 2: price and reserve this wave's transient tier**, sized to this
    // wave rather than to the widest one the engine can run. A segment being
    // prepared returns its price for the step to place; a segment running in
    // the step places nothing.
    #[cfg(feature = "cuda")]
    if total_rows > 0 && !in_step {
        if let Device::Cuda(d) = dev {
            let geometry = model.wave_geometry(embed_dtype);
            let plan = WavePlan::new(geometry);
            // **Scored rows are not all rows.** The head runs every decode row
            // and the LAST row of each prefill span — a verifying span excepted,
            // where each row is a prediction to compare a proposal against.
            // Pricing `HeadLogits` at `total_rows × vocab` buys ~1 GB of tier
            // for a 248 KB logits block on a 2,048-row prefill, taken from the
            // weight side every wave.
            //
            // A wave that stops short of the last layer returns its residual and
            // runs no head, so it scores no rows at all.
            let verify_seqs = model.verify_row_seqs()?;
            let scored_prefill: usize = pre_q
                .iter()
                .enumerate()
                .map(|(k, &l)| {
                    if verify_seqs.contains(&seq_ids[n_decode + k]) {
                        l
                    } else {
                        1
                    }
                })
                .sum();
            let scored_rows = if layer_end == num_layers {
                n_decode + scored_prefill
            } else {
                0
            };
            let width = WaveWidth {
                prefill_rows: pre_rows,
                decode_rows: n_decode,
                scored_rows,
                // The caller's accept walk selects on the head's span only when
                // it reads the logits there.
                accept_rows: if accept_in_place { scored_rows } else { 0 },
                stencil_rows,
                // This lineage's attention is dense: no sparse selection.
                qsa_bytes: 0,
                // Its MoE layers apply no look-ahead routers.
                look_ahead_hops: 0,
                // A one-row prefill group takes the decode kernels, so it carves
                // no span-table entry and, alone, no scan transient.
                prefill_spans: pre_q.iter().filter(|&&l| l > 1).count(),
                // The prefill launch's carve: up to one key chunk of each bulk
                // chunk's sequence, and the carry beside it.
                kv_stage: PrefillKvStageLayout::new(
                    pre_q,
                    pre_off,
                    geometry.n_head,
                    geometry.n_kv_head,
                    geometry.head_dim,
                    PrefillLaunchBounds::PRODUCTION,
                ),
            };
            let per_phase = [
                plan.phase_bytes(LayerPhase::Attention, width),
                plan.phase_bytes(LayerPhase::Ffn, width),
                plan.phase_bytes(LayerPhase::Forward, width),
            ];
            if preparing {
                return Ok((WavePhase::Priced(per_phase), None));
            }
            plan_wave_transient(&d.cuda_stream(), per_phase)?;
        }
    }
    if preparing {
        return Ok((WavePhase::Priced([0; 3]), None));
    }

    // From here to the end of this function the forward owns the partition —
    // after phase 2 in both directions: admit creates arenas through the same
    // gate the sealing thread uses (opening earlier had the forward waiting on
    // itself), and the tier placement may buy ground from the weight side,
    // whose `set_weight_floor` refuses while a forward is open.
    #[cfg(feature = "cuda")]
    let _forward_open = match dev {
        Device::Cuda(d) => Some(begin_forward(&d.cuda_stream())),
        _ => None,
    };

    // The forward's span, open from here to the head: the setup every layer
    // reads — the token ids and the ragged prefill tables — is laid on it
    // first, and the head's norm and logits are carved from it after the last
    // layer. It is reset per *forward*, the lifetime all of them have. A
    // segment of a step carves from the step's span, which outlives it.
    #[cfg(feature = "cuda")]
    let head_span = match (dev, mode) {
        (_, SweepMode::InStep(_)) => None,
        (Device::Cuda(d), _) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Forward)?),
        _ => None,
    };
    #[cfg(feature = "cuda")]
    let fwd_ticket = match mode {
        SweepMode::InStep(ticket) => ticket,
        _ => head_span.as_ref().map(|g| g.ticket()),
    };
    #[cfg(not(feature = "cuda"))]
    let (head_span, fwd_ticket): (Option<WaveGuard>, Option<WaveTicket>) = (None, None);

    let g_meta = crate::models::profile::gpu_span("fwd:meta", dev);
    let prefill_headers = DecodeHeaders::Prefill(BatchedPrefillMeta::new_ragged(
        pre_off, pre_q, dev, fwd_ticket,
    )?);
    g_meta.end();

    // Every row's id in one buffer on the span — one upload for ids handed over
    // on the host, a copy for ids already on the device — read by the embedding
    // and, when the wave reaches it, by the draft head.
    let ids = {
        let inputs: Vec<&Tensor> = contexts.iter().map(|c| c.input_ids).collect();
        device_token_ids(&inputs, dev, fwd_ticket)?
    };

    // Combined residual: embed every row flat `[1, total, hidden]`, or resume a
    // paused wave from the residual it was handed. Either way it is a buffer of
    // this forward's own on the forward span (`WaveBuffer::ForwardResidual`):
    // a resumed one is gathered into it in this forward's row order, leaving
    // what it was gathered from as it was (`docs/wave_step.md` §3.1).
    let g_embed = crate::models::profile::gpu_span("fwd:embed", dev);
    let mut x = match x_in {
        Some(resume) => {
            let buf = wave_empty_ticketed(
                (1, total_rows, resume.width()),
                embed_dtype,
                dev,
                fwd_ticket,
            )?;
            resume.gather_into(&buf)?;
            TensorCat::from_cat_tensor(buf, 0)?
        }
        None => TensorCat::from_cat_tensor(embed_rows(q, &ids, embed_dtype, fwd_ticket)?, 0)?,
    };
    g_embed.end();

    // Per-group split cos/sin over this wave's own positions, for the
    // non-paged paths; the paged kernels rotate from `model.rope()`.
    let theta = q.cfg.rope_theta;
    let rot = model.rotary();
    let dec_pos: Vec<u32> = dec_off.iter().map(|&o| o as u32).collect();
    let mut pre_pos: Vec<u32> = Vec::with_capacity(pre_rows);
    for (&o, &l) in pre_off.iter().zip(pre_q) {
        for i in 0..l {
            pre_pos.push((o + i) as u32);
        }
    }
    let rope_dtype = if embed_dtype == DType::F8E4M3 {
        DType::BF16
    } else {
        embed_dtype
    };
    let dec_rope = LazyRope::new(|| rot.rope_cos_sin(&dec_pos, theta, rope_dtype, dev, None));
    // Prefill's activation is the flat batch-of-one `[1, total, …]`, so its
    // tables carry the same leading axis.
    let half = q.cfg.attn_head_dim / 2;
    let pre_rope = LazyRope::new(|| {
        let (cos, sin) = rot.rope_cos_sin(&pre_pos, theta, rope_dtype, dev, None)?;
        Ok((
            cos.reshape((1, pre_rows, half))?,
            sin.reshape((1, pre_rows, half))?,
        ))
    });

    let dec_pm: RefCell<Option<SharedPm>> = RefCell::new(None);
    let pre_pm: RefCell<Option<SharedPm>> = RefCell::new(None);
    // NeoX half-split within the rotary width — the layout `RotaryLayout`
    // permutes the head dims into, never the interleaved GPT-J form.
    let interleaved = false;
    let dec_params = BatchedAttentionParams::new(
        &dec_rope,
        interleaved,
        model.rope(),
        decode_headers,
        dec_q,
        generation,
        &dec_pm,
    );
    let pre_params = BatchedAttentionParams::new(
        &pre_rope,
        interleaved,
        model.rope(),
        prefill_headers,
        pre_q,
        generation,
        &pre_pm,
    );

    // Where each sequence's rows sit in the packed buffer — what a recurrent
    // mixer needs to run one sequence at a time against its own state.
    let spans = seq_spans(seq_ids, &q_lens)?;
    let eps = q.cfg.rms_norm_eps;

    // The decode pointer table for the whole sweep — every DeltaNet layer's
    // state/tail addresses for every decode sequence, ONE host upload, built
    // here where the launch queue is still empty. Each layer takes its slice;
    // a per-layer upload would sync the stream mid-sweep and serialise the
    // pipeline. `None` when the wave carries no decode span.
    //
    // On the forward span (`WaveBuffer::DeltaNetDecodeTable`): built once and
    // read by every DeltaNet layer, which is what that span holds. The span is
    // open from before the embedding to the head, so a caller must drop the
    // previous wave's result — which holds that wave's span — before issuing
    // the next; `begin_wave` refuses a phase that is still open.
    //
    // Spanned for the same reason as `fwd:meta`: one table entry per DeltaNet
    // layer per decode sequence, so it is the other setup cost that grows with
    // the cohort rather than with the model.
    let g_dntab = crate::models::profile::gpu_span("fwd:dntab", dev);
    #[cfg(feature = "cuda")]
    let dn_table = build_wave_table(&spans, stores, fwd_ticket)?;
    g_dntab.end();

    let verify_seqs = model.verify_row_seqs()?;

    // The sweep's own input, before any layer has touched it. A fault reported
    // downstream of here is this sweep's; a fault reported AT here arrived
    // already formed, and the search belongs upstream of `layer_start`.
    x.as_cat_tensor().assert("qwen35.sweep_in");

    // The rows the routed experts' residency scores as decode: the decode rows,
    // the verify segments and each prompt's last row (`residency_rows`). A
    // drafting checkpoint decodes in verify waves, which carry no decode rows.
    let decode_like = residency_decode_rows(
        n_decode,
        seq_ids[n_decode..]
            .iter()
            .copied()
            .zip(pre_q.iter().copied()),
        |s| verify_seqs.contains(&s),
    );

    // Everything above — admission, the tier, the tables, the embedding — ran
    // eagerly. The layers and the head are recorded as a chain of graphs
    // (`docs/decode_graphs.md`), cut where a layer meets the host.
    dev.record_launches()?;

    for li in layer_start..layer_end {
        // The residual ENTERING this layer — the first checkpoint of the layer,
        // so it fires before anything inside can. Reaching layer N's body means
        // every layer before it produced a finite residual.
        #[cfg(feature = "tensor-assert")]
        crate::models::nan_capture::checkpoint(
            candle::tensor_assert::site("qwen35.layer_in.L", li),
            x.as_cat_tensor(),
            &[],
            capture_dev,
        )?;
        // The wave arrives at `li`: join its transfer if one is in flight, and
        // hold the handle for as long as its compute is being issued. On a
        // resident store this is an `Arc` bump.
        // Stream-elapsed for the WHOLE layer, so it can be compared against the
        // sum of the `dn:*` / `decode:*` sub-spans inside it. Those cover the
        // projections, the mixer and the attention kernel but nothing between
        // them, and at width the forward's stream time is ~5.6× their sum — this
        // span says whether the difference sits inside layers (unspanned norms,
        // residuals, quantize/convert kernels) or between them (a starved stream
        // waiting on host-side per-layer work).
        let g_layer = crate::models::profile::gpu_span("fwd:layer", dev);
        let layer = q.layers.ensure(li)?;
        // This layer's adapter pairs, or all-`None` when the wave is unadapted
        // or the adapter does not reach this layer. Seven hash lookups per
        // layer per wave against a wave that is thousands of matmuls long.
        let layer_lora = adapter.map(|a| a.layer(li)).unwrap_or_default();
        // On `cfg.layer_kinds` rather than the layer's own `mix`: a streamed
        // store has no `q.layers[li]` to match against until `ensure` has
        // produced one, and the schedule is the same answer without a fetch.
        match q.cfg.layer_kinds[li] {
            LayerKind::Attention => {
                let kv = model.kv_map().kv_index(li).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "qwen35 wave: layer {li} attends but has no KV index"
                    ))
                })?;
                let mut cache_refs: Vec<&mut KvCache> = contexts
                    .iter_mut()
                    .map(|c| &mut c.kv_caches.caches[kv])
                    .collect();
                let (dec_c, pre_c) = cache_refs.split_at_mut(n_decode);
                let mut wave_groups: Vec<WaveAttnGroup> = Vec::with_capacity(2);
                if n_decode > 0 {
                    wave_groups.push(WaveAttnGroup {
                        caches: dec_c,
                        offsets: dec_off,
                        params: &dec_params,
                        rows: n_decode,
                        decode_layout: true,
                        // Dense attention: the block-sparse selection belongs
                        // to Qwen3.8-Flash-Next's indexer, not this lineage.
                        qsa: None,
                    });
                }
                if n_prefill > 0 {
                    wave_groups.push(WaveAttnGroup {
                        caches: pre_c,
                        offsets: pre_off,
                        params: &pre_params,
                        rows: pre_rows,
                        decode_layout: false,
                        qsa: None,
                    });
                }
                let layer = Qwen35AttentionLayer {
                    layer: &layer,
                    n_head: q.cfg.num_attention_heads,
                    n_kv_head: q.cfg.num_kv_heads,
                    head_dim: q.cfg.attn_head_dim,
                    rotary: model.rotary(),
                    lora: layer_lora,
                };
                // `layer_idx` names the KV layer, not the trunk layer: it is
                // what the per-layer arena bookkeeping inside the mixed
                // dispatch indexes by, and that bookkeeping is per cache.
                forward_layer_batched_mixed(
                    &layer,
                    &mut wave_groups,
                    &mut x,
                    embed_dtype,
                    kv,
                    &decode_like,
                )?;
                // The attention arm's mixer output, which had no site of its
                // own — only the DeltaNet arm was covered, so an attention
                // layer could only ever be blamed at `layer_out`, after its FFN
                // had already been folded in.
                #[cfg(feature = "tensor-assert")]
                crate::models::nan_capture::checkpoint(
                    candle::tensor_assert::site("qwen35.attn_out.L", li),
                    x.as_cat_tensor(),
                    &[],
                    capture_dev,
                )?;
            }
            LayerKind::DeltaNet => {
                let orig = x.dtype();
                #[cfg(feature = "cuda")]
                let layer_table = match &dn_table {
                    Some(t) => Some(t.layer_slice(li)?),
                    None => None,
                };
                #[cfg(not(feature = "cuda"))]
                let layer_table = None;
                // The recurrent state as it stands ENTERING this layer's scan.
                //
                // This is the one value on the path that survives between waves:
                // everything else is rebuilt from the residual each time, so a
                // NaN here was written by an earlier wave and has been carried,
                // not produced. A report where `layer_in` is clean and this is
                // not is the whole answer — and it is unreachable from the
                // residual-only sites, which is why it needs its own.
                #[cfg(feature = "tensor-assert")]
                for st in stores.iter() {
                    let Ok(state) = st.layer_state(li) else {
                        continue;
                    };
                    crate::models::nan_capture::checkpoint(
                        candle::tensor_assert::site("dn.state_in.s.L", li),
                        &state.s,
                        &[("conv_tail", &state.conv_tail)],
                        capture_dev,
                    )?;
                }
                delta_net_mix_wave(
                    q,
                    &layer,
                    li,
                    &spans,
                    &mut x,
                    stores,
                    eps,
                    layer_table.as_ref(),
                )?;
                // The scan writes no live state — its advance lands in a
                // timeline or the prefill scratch, applied by the fold or the
                // commit — so the state's exit is checked there
                // (`gdn.exit` / `gdn.fold`), not here.
                //
                // Sampled BETWEEN the mixer and the FFN: "layer N is bad" is
                // ambiguous while the expert FFN is inside the same layer, and
                // a routed expert arriving late is as good a NaN candidate as
                // the recurrent scan. Splitting them is the difference between
                // naming a subsystem and naming a layer.
                #[cfg(feature = "tensor-assert")]
                crate::models::nan_capture::checkpoint(
                    candle::tensor_assert::site("qwen35.mixer_out.L", li),
                    x.as_cat_tensor(),
                    &[],
                    capture_dev,
                )?;
                quantized_delta_net_ffn(
                    &layer,
                    &mut x,
                    embed_dtype,
                    orig,
                    layer_lora,
                    &decode_like,
                )?;
            }
        }
        // The layer's result. Reaching this on a bad value means the mixer's
        // own checkpoint passed, so the FFN produced it.
        #[cfg(feature = "tensor-assert")]
        crate::models::nan_capture::checkpoint(
            candle::tensor_assert::site("qwen35.layer_out.L", li),
            x.as_cat_tensor(),
            &[],
            capture_dev,
        )?;
        // Bisect *when* a narrowed weight changes, not merely *that* it did.
        #[cfg(feature = "tensor-assert")]
        crate::models::nan_capture::watch_layer(capture_dev, li);
        // This layer's compute is issued, so the copy stream can overlap the
        // next layers' transfers with it rather than serialising in front of
        // them. Safe to evict from here: a load launches the recording segment
        // and orders every copy behind an event recorded on the compute stream
        // after it, so a slot whose GEMMs are still running cannot be
        // overwritten under them. A resident store, or a plan with nothing to
        // load, returns without touching anything.
        //
        // **After the checkpoints, not before.** They read `x`, not the layer,
        // but the prefetch may evict the slot `layer` borrows — so releasing it
        // first would end the borrow the assert path is still inside.
        q.layers.prefetch()?;
        g_layer.end();
        // A segment per layer, as the uniform sweep cuts it: recorded whole, the
        // forward would be one segment and the GPU would idle until the host had
        // recorded every layer. A routed checkpoint is cut the same way: its
        // expert dispatch flushes only on `expert_lre::flush_schedule`'s doubling
        // cadence, which was measured for Flash-Next's own sweep, not this one.
        dev.flush_launches()?;
    }

    // A window that stops short hands its residual on as it stands, on the
    // forward span, with the span's guard: the driver moves it to the next
    // segment or the creep's arena while the guard still holds it.
    if layer_end < num_layers {
        return Ok((WavePhase::Residual(x), head_span));
    }

    let xt = x.to_tensor();
    let hidden = xt.dim(2)?;
    let x_flat = xt.reshape((xt.dim(1)?, hidden))?;

    // **The draft head's layer.** One more attention pass over the same rows at
    // the same positions, against the KV layer past every trunk one, so the
    // head's history stays exactly as long as the sequence it drafts for. It
    // runs here rather than inside the sweep because its input is the trunk's
    // OUTPUT, not the residual stream. See [`super::draft`].
    if let (Some(head), Some(kv_layer)) = (q.mtp.as_ref(), head_kv) {
        head_wave_pass(
            model,
            head,
            contexts,
            &x_flat,
            &ids.reshape((1, ids.elem_count()))?,
            &HeadWave {
                n_decode,
                pre_rows,
                dec_off,
                pre_off,
                dec_params: &dec_params,
                pre_params: &pre_params,
                spans: &spans,
                kv_layer,
                act_dtype: embed_dtype,
                forward: fwd_ticket,
            },
        )?;
    }

    // Head over the rows that need logits: every decode row (one token each,
    // flat positions `0..n_decode`) and the last token of every prefill row —
    // except a **verifying** span, where every row is a prediction to compare a
    // proposal against, so all of them are scored.
    let mut idx: Vec<u32> = Vec::with_capacity(n_decode + pre_rows);
    for (d, _) in seq_ids.iter().take(n_decode).enumerate() {
        idx.push(d as u32);
    }
    let mut acc = n_decode as u32;
    for (k, &l) in pre_q.iter().enumerate() {
        let seq = seq_ids[n_decode + k];
        if verify_seqs.contains(&seq) {
            for t in 0..l as u32 {
                idx.push(acc + t);
            }
        } else {
            idx.push(acc + l as u32 - 1);
        }
        acc += l as u32;
    }
    if idx.is_empty() {
        // On the forward span, so the span's guard goes back with it.
        return Ok((WavePhase::Residual(x), head_span));
    }
    let g_head = crate::models::profile::gpu_span("fwd:head", dev);
    let pre_norm = select_head_rows(&x_flat, idx, 0, fwd_ticket)?;
    // The head runs on the forward's span, opened before the embedding. Seeded
    // from `wave_root`, which yields a ticket rather than a borrow, so the
    // logits stay `'static`-typed and physically on the span; what makes that
    // sound is handing the guard back with them.
    let logits = {
        #[cfg(feature = "cuda")]
        {
            let acts = q.final_norm.forward_dynamic(
                &pre_norm,
                q.lm_head.int8mode(),
                wave_root_ticketed(fwd_ticket),
            )?;
            q.lm_head
                .forward_dynamic(acts.as_dynamic(), pre_norm.dtype())?
        }
        #[cfg(not(feature = "cuda"))]
        {
            use candle_nn::Module;
            q.lm_head.forward(&q.final_norm.forward(&pre_norm)?)?
        }
    };
    g_head.end();

    Ok((
        WavePhase::Logits(TensorCat::from_cat_tensor(logits, 0)?),
        head_span,
    ))
}
