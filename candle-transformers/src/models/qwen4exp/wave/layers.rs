//! The layer sweep of a wave: the attention, MoE and draft-head work over one packed set of rows, with each phase's transients on its own arena span.
//!
//! The residual is one buffer that every combine updates in place, and each layer's attention and FFN run as two phases that share their ground in turn. A window that stops short of the last layer hands its residual back, and the draft head runs behind the window that completes the trunk. Numerical faults are reported through the wave's drain rather than read back here, since a readback is the fence those faults stop reproducing under. Look-ahead votes for the next rows' experts are cast beside each layer's own routing, so the expert cache can copy them before they are needed.

use super::*;
use crate::models::qwen4exp::sum_scale::ACT_SUM_SCALE;

/// Later rows a MoE layer's router look-ahead votes on, for a wave of
/// `total_rows`: [`LOOK_AHEAD_HOPS`] at decode width, for the device's
/// read-ahead, and [`STAGED_HOPS`] at prefill width, whose voted sets the
/// expert stager reads off the drive (`PipelineState::stage_voted`). One answer
/// for the forward that builds the routers and the plan that prices their
/// columns ([`WaveWidth::look_ahead_hops`]).
pub(super) fn look_ahead_hops(total_rows: usize) -> usize {
    if total_rows <= LOOK_AHEAD_MAX_TOKENS {
        LOOK_AHEAD_HOPS
    } else {
        STAGED_HOPS
    }
}

impl Qwen4ExpBatched {
    /// Bytes a forward's tier can be placed in: the KV side's free ground —
    /// counting what the standing tier blocks, which the next forward releases
    /// first — plus what the weight side could still concede above its floor.
    /// Never less than [`WAVE_SPAN_BYTES`], the tier the partition always
    /// leaves room for. `None` where there is no span to measure (a CPU
    /// device, a test).
    pub(super) fn placeable_tier_bytes(&self) -> Option<usize> {
        let stats = region_stats(0)?;
        let free = (stats.free + stats.blocked).saturating_mul(REGION_BYTES);
        Some(
            free.saturating_add(self.model.experts.cedeable_span_bytes())
                .max(WAVE_SPAN_BYTES),
        )
    }

    /// One full-attention layer's QSA work: append every sequence's index
    /// keys, then build the wave's selection table if any row is past the
    /// budget.
    ///
    /// `kv` is the layer's index among the full-attention layers, which is
    /// also its index into a sequence's caches. Returns `None` when the layer
    /// attends densely — either the checkpoint declares it dense
    /// (`compress_ratio == 0`) or no row has more visible cells than the
    /// budget, where the selection is the identity and computing it would be
    /// arithmetic with no effect (§12.5).
    ///
    /// The keys are appended for EVERY wave regardless, because the cache is
    /// what a later, deeper wave scores against.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn layer_selection(
        &self,
        kv: usize,
        compress_ratio: usize,
        indexer: &IndexerWeights,
        rope: &FactoredRope,
        h: &Tensor,
        spans: &[SeqSpan],
        offsets: &[usize],
        idx_map: &mut HashMap<usize, Vec<IndexCache>>,
        capture: Option<&mut SpecCapture>,
        total_rows: usize,
        fwd_ticket: Option<candle::cuda_backend::wave_provenance::WaveTicket>,
    ) -> Result<Option<QsaSelection>> {
        let selected = select_layer(
            kv,
            compress_ratio,
            indexer,
            rope,
            h,
            spans,
            offsets,
            idx_map,
            capture,
            total_rows,
            &self.model.cfg.indexer,
            self.model.cfg.rms_norm_eps,
            &self.model.device,
            &self.qsa_rows,
            fwd_ticket,
        )?;
        let (sel, rows) = selected.unzip();
        // At a sampled layer, what the selection chose is counted into its
        // rounds (`docs/hit_driven_projection.md` §6.2), the stride's last
        // queries are kept, and the first wave after a boundary scores the
        // kept queries against the pooled rounds first (§6.3) — nothing read
        // back. Not under a forced recent span: every row names those blocks
        // whatever they score, so they would count as read by the indexer when
        // nothing chose them.
        let strata = &self.model.cfg.indexer.strata;
        let forced = strata.recent == Recent::Forced && strata.recent_blocks > 0;
        if let (Some(sel), Some(rows)) = (sel.as_ref().filter(|_| !forced), rows.as_ref()) {
            if let Some(layer) = self.hit_kv.iter().position(|&k| k == kv) {
                let hits = self
                    .hits
                    .read()
                    .map_err(|_| candle::Error::Msg("hits lock poisoned".into()))?;
                if !hits.is_empty() {
                    // The wave's last sampled layer publishes what every
                    // sampled layer counted, for the next boundary to read.
                    let publish = layer + 1 == self.hit_kv.len();
                    let mut table = self
                        .hit_table
                        .lock()
                        .map_err(|_| candle::Error::Msg("hit table lock poisoned".into()))?;
                    let scan = ShadowScan {
                        geom: self.shadow_geometry(compress_ratio),
                        rope,
                    };
                    count_hits(
                        sel,
                        rows,
                        spans,
                        offsets,
                        &hits,
                        &mut table,
                        layer,
                        publish,
                        &scan,
                        &self.model.device,
                    )?;
                }
            }
        }
        Ok(sel)
    }

    /// The layer sweep proper — the oracle's algebra over the wave's packed
    /// rows. See the module docs for the mixer substitutions.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sweep_layers(
        &self,
        contexts: &mut [crate::models::kv_cache_utils::SequenceContext],
        seq_ids: &[usize],
        inputs: &[Tensor],
        n_decode: usize,
        offs: (&[usize], &[usize], &[usize], &[usize]),
        rows: (usize, usize),
        range: (usize, usize),
        x_in: Option<ResidualIn<'_>>,
        headers: (DecodeHeaders, DecodeHeaders),
        generation: &candle::quantized::pinned_staging::Generation,
        eps: f64,
        // The forward-scoped span's ticket — the home for every per-wave table
        // built below. `None` off CUDA, or when no tier was placed.
        fwd_ticket: Option<candle::cuda_backend::wave_provenance::WaveTicket>,
    ) -> Result<(WavePhase, Option<WaveGuard>)> {
        let (dec_off, dec_q, pre_off, pre_q) = offs;
        let (total_rows, pre_rows) = rows;
        let (layer_start, layer_end) = range;
        let (decode_headers, prefill_headers) = headers;
        let m = &self.model;
        let cfg = &m.cfg;
        let dev = &m.device;
        let hc = cfg.hc.count;
        let n_embd = cfg.hidden_size;
        let num_layers = cfg.num_layers;
        let q_lens: Vec<usize> = dec_q.iter().chain(pre_q.iter()).copied().collect();

        // ── Row embeddings: the trunk's entry into the wide stream, and the
        // draft head's `embed(t)` half. ──
        //
        // Gathered once and shared by both. A resumed wave takes its residual
        // from `x_in` and would never gather these, but if it is the window
        // that reaches the last trunk layer then the head runs behind it and
        // needs them — so the condition is "either consumer wants them", not
        // "the residual is fresh".
        //
        // One launch fills both: the fresh residual is the embedding repeated
        // across the `hc` streams, written by the gather itself, and the head
        // reads the bare rows. Both are fully written (invariant 6), and both
        // are on the forward span (`WaveBuffer::HyperResidual`,
        // `WaveBuffer::HeadRowEmbeds`).
        let fresh = x_in.is_none();
        let reaches_head = layer_end == num_layers;
        let head_embeds = reaches_head && m.mtp.is_some();
        let res_dims = (total_rows, hc, n_embd);
        // The residual's home, fresh or resumed: the forward span, which
        // outlives every layer phase of the sweep.
        let residual_buffer = || wave_empty_ticketed(res_dims, DType::F32, dev, fwd_ticket);
        let entry = if fresh {
            Some(residual_buffer()?)
        } else {
            None
        };
        let row_embeds = if head_embeds {
            Some(wave_empty_ticketed(
                (total_rows, n_embd),
                DType::F32,
                dev,
                fwd_ticket,
            )?)
        } else {
            None
        };
        // Per-sequence host token ids, read once: the embedding gather below and
        // the PLE hash side both take them from here. The buffer they were read
        // through stays on the forward span as every row's id in wave order —
        // what the gather reads — so only a wave with a host-side input uploads
        // its ids again (`WaveBuffer::RowTokenIds` prices both).
        let ids = host_token_ids(inputs, fwd_ticket)?;
        let mut seq_tokens: Vec<Vec<u32>> = ids.per_input;
        // An image's rows arrive under virtual ids: resolved here, the hash
        // side handed `<|image_pad|>` in their place, and their values the
        // gather's second table.
        let images = take_wave_images(
            &mut seq_tokens,
            m.embed.n_rows(),
            cfg.ple.image_token_id,
            n_embd,
            dev,
            fwd_ticket,
        )?;
        if fresh || head_embeds {
            let ids = match (&images, ids.device) {
                (Some(w), _) => {
                    wave_from_vec_ticketed(w.gather_ids.clone(), (total_rows,), dev, fwd_ticket)?
                }
                (None, Some(t)) => t,
                (None, None) => {
                    let flat_ids: Vec<u32> = seq_tokens.iter().flatten().copied().collect();
                    wave_from_vec_ticketed(flat_ids, (total_rows,), dev, fwd_ticket)?
                }
            };
            m.embed.gather_into(
                &ids,
                images.as_ref().map(|w| &w.extra),
                entry.as_ref(),
                row_embeds.as_ref(),
            )?;
        }

        // ── Residual: the fresh rows as gathered, or resume. ──
        //
        // **One buffer for the whole forward, updated in place.** `hc_combine`
        // adds each block's scatter into `res` where it stands, so the wide
        // `[rows, hc, hidden]` stream is this one buffer, not one per combine.
        // It crosses every layer phase's reset, so it is carved from the
        // forward span, which outlives them all. A window that stops short
        // hands it back as `WavePhase::Residual`, still on the span, and the
        // driver moves it on (`docs/wave_step.md`).
        //
        // **A resumed residual is gathered in, never combined into where it
        // lies.** What `x_in` names is the caller's: a creep's committed half,
        // the last segment's rows. Combining into it would rewrite it under its
        // owner, and a wave that failed part-way would leave it advanced by the
        // layers that did run. The gather lands the rows in this forward's own
        // buffer in its row order, which is the copy a resume costs.
        let mut res = match x_in {
            Some(resume) => {
                let dst = residual_buffer()?;
                resume.gather_into(&dst)?;
                dst
            }
            None => entry.expect("allocated whenever the residual is fresh"),
        };

        // Row spans, beside the host token ids above (the PLE hash side).
        let spans = seq_spans(seq_ids, &q_lens)?;

        // ── Model-side RoPE for the non-paged paths; the paged kernels rotate
        // from `self.rope`, each sequence at its own rung. ──
        let theta = cfg.rope_theta;
        let dec_pos: Vec<u32> = dec_off.iter().map(|&o| o as u32).collect();
        let mut pre_pos: Vec<u32> = Vec::with_capacity(pre_rows);
        for (&o, &l) in pre_off.iter().zip(pre_q) {
            for i in 0..l {
                pre_pos.push((o + i) as u32);
            }
        }
        let dec_rope = LazyRope::new(|| {
            m.rotary
                .rope_cos_sin(&dec_pos, theta, DType::F32, dev, fwd_ticket)
        });
        let half = cfg.attn_head_dim / 2;
        let pre_rope = LazyRope::new(|| {
            let (cos, sin) = m
                .rotary
                .rope_cos_sin(&pre_pos, theta, DType::F32, dev, fwd_ticket)?;
            Ok((
                cos.reshape((1, pre_rows, half))?,
                sin.reshape((1, pre_rows, half))?,
            ))
        });
        let dec_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let pre_pm: std::cell::RefCell<Option<SharedPm>> = std::cell::RefCell::new(None);
        let dec_params = BatchedAttentionParams::new(
            &dec_rope,
            false,
            &self.rope,
            decode_headers,
            dec_q,
            generation,
            &dec_pm,
        );
        // The prefill group's image rows, by their row in the group: image ids
        // arrive in prompts, which are prefill rows, after the decode group.
        let image_rotation = match &images {
            Some(w) => {
                let runs: Vec<ImageRun> = w
                    .runs
                    .iter()
                    .map(|r| ImageRun {
                        row: r.row - n_decode,
                        ..*r
                    })
                    .collect();
                Some(ImageRotation::new(
                    runs,
                    cfg.attn_head_dim,
                    cfg.rope_dim,
                    f64::from(theta),
                    |v, s| wave_from_vec_ticketed(v, s, dev, fwd_ticket),
                )?)
            }
            None => None,
        };
        let pre_params = BatchedAttentionParams::new(
            &pre_rope,
            false,
            &self.rope,
            prefill_headers,
            pre_q,
            generation,
            &pre_pm,
        )
        .with_image_rotation(image_rotation.as_ref());

        // QSA: the indexer's rotation table and this wave's index caches.
        // Absolute positions per row, in the wave's packed order — decode
        // rows first (one each), then each prefill sequence's span.
        let index_rope = self.index_rope().clone();
        let mut offsets_all: Vec<usize> = Vec::with_capacity(total_rows);
        offsets_all.extend_from_slice(dec_off);
        offsets_all.extend_from_slice(pre_off);
        let mut idx_map = self
            .index
            .write()
            .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;

        let mut rec = self
            .recurrent
            .write()
            .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
        // Held for the sweep beside the stores, and each span's absolute position
        // beside its rows — what an armed span is named by.
        let mut arms = self
            .arms
            .write()
            .map_err(|_| candle::Error::Msg("arms lock poisoned".into()))?;
        let span_at: Vec<usize> = dec_off.iter().chain(pre_off).copied().collect();
        let mut ple_map = self
            .ple
            .write()
            .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
        // Held for the sweep, like the three state maps: the capture sites are
        // inside the layer loop and a per-layer lock would be a lock per layer
        // per wave for a value that is `None` on every plain decode.
        let mut verify_guard = self
            .verify
            .write()
            .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;
        let cap_map: &mut Option<SpecCapture> = &mut verify_guard;

        // The decode pointer table for the whole sweep — every DeltaNet layer's
        // tail, state and timeline addresses for every decode sequence, ONE host
        // upload here, before the first layer's launches, where the queue is
        // still empty. Each DeltaNet layer takes its slice, so no layer uploads a
        // table of its own behind the launches already queued. Valid for this
        // forward only: compaction may move a slot between forwards.
        #[cfg(feature = "cuda")]
        let dn_table = {
            let at: HashMap<usize, usize> =
                spans.iter().enumerate().map(|(i, s)| (s.seq, i)).collect();
            let mut slot: Vec<Option<&mut RecurrentStateStore>> =
                spans.iter().map(|_| None).collect();
            for (seq, store) in rec.iter_mut() {
                if let Some(&i) = at.get(seq) {
                    slot[i] = Some(store);
                }
            }
            let stores: Vec<&mut RecurrentStateStore> = spans
                .iter()
                .zip(slot)
                .map(|(span, st)| {
                    st.ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "qwen4exp: sequence {} has no recurrent store in this wave",
                            span.seq
                        ))
                    })
                })
                .collect::<Result<_>>()?;
            build_wave_table(&spans, &stores, fwd_ticket)?
        };

        // Sub-block finiteness probes for the layer bisect — sync readbacks,
        // so they exist only in `tensor-assert` diagnostic builds.
        #[cfg(feature = "tensor-assert")]
        /// Fold one layer value into its assert slot.
        ///
        /// **Asynchronous, and it reports through the drain like everything else.**
        /// This was an `eprintln!` of `max(abs(t))`, which cost a `to_scalar` — a
        /// synchronous readback, per value, per layer, per wave — and printed to
        /// stderr where nothing correlates it with the wave that produced it. Three
        /// separate problems: the readback is the fence these faults stop
        /// reproducing under, the print cannot say which site went bad *first*
        /// because stderr has no ordering against the device, and a human reading
        /// magnitudes is not a check.
        ///
        /// `assert_tensor` is one reduction kernel with no readback and no fence,
        /// and `wave_driver`'s per-wave drain then reports every bad site ordered by
        /// the kernel's own ticket — which is what makes "the first bad value in
        /// this wave" a question with an answer.
        fn probe(name: &'static str, t: &LiveTensor<'_>) {
            candle::tensor_assert::assert_tensor(t, name);
        }
        #[cfg(feature = "tensor-assert")]
        use candle::tensor_assert::site;
        // Where an armed attention capture fences, reads back and dumps.
        #[cfg(feature = "tensor-assert")]
        let Device::Cuda(capture_dev) = dev
        else {
            candle::bail!("qwen4exp wave runs on CUDA");
        };
        // A decode row per sequence — built once per wave, not per layer.
        #[cfg(feature = "tensor-assert")]
        let decode_rows = vec![1usize; n_decode];

        // The rows the routed experts' residency scores as decode: the decode
        // rows, the verify segments and each prompt's last row
        // (`residency_rows`).
        // From the capture this sweep already holds — `self.verify` is
        // write-locked for the whole sweep.
        let verify_seqs: HashSet<usize> = cap_map
            .as_ref()
            .map(|c| c.seqs.keys().copied().collect())
            .unwrap_or_default();
        let decode_like = residency_decode_rows(
            n_decode,
            seq_ids[n_decode..]
                .iter()
                .copied()
                .zip(pre_q.iter().copied()),
            |s| verify_seqs.contains(&s),
        );
        // The prefill-shaped attention serves two very different workloads: a
        // bulk prompt chunk (thousands of rows over the whole history) and the
        // speculative verify (a few rows per sequence, every decode step). One
        // span over both averaged a 200 ms bulk layer with hundreds of sub-ms
        // verify layers into a number that described neither.
        // The routed experts likewise: a wave carrying a prompt chunk is a
        // different MoE workload from a decode step.
        let bulk_prefill = seq_ids[n_decode..]
            .iter()
            .zip(pre_q)
            .any(|(s, &l)| l > 1 && !verify_seqs.contains(s));
        let (attn_prefill_span, moe_span) = if bulk_prefill {
            ("q4e:attn_prefill", "q4e:moe_routed:bulk")
        } else {
            ("q4e:attn_verify", "q4e:moe_routed:step")
        };

        dev.record_launches()?;

        for li in layer_start..layer_end {
            let layer = &m.layers[li];
            // The forward thread's time issuing this layer, by mixer kind.
            let _host_layer = span(if matches!(cfg.layer_kinds[li], LayerKind::Attention) {
                "q4e:host:attn_layer"
            } else {
                "q4e:host:dn_layer"
            });

            // ── PLE, before this layer's mixer (§12.4). ──
            let g_ple = if li == cfg.ple.layer {
                Some(crate::models::profile::gpu_span("q4e:ple", dev))
            } else {
                None
            };
            if li == cfg.ple.layer {
                // Every sequence's rows in one pass — one table gather, the
                // injection and the conv over the whole wave, the residual
                // updated in place of a per-sequence concatenation. A verifying
                // span captures the rows it appends to the conv history; every
                // other span captures nothing.
                let at: HashMap<usize, usize> =
                    spans.iter().enumerate().map(|(i, s)| (s.seq, i)).collect();
                let mut slot: Vec<Option<&mut PleState>> = spans.iter().map(|_| None).collect();
                for (seq, st) in ple_map.iter_mut() {
                    if let Some(&i) = at.get(seq) {
                        slot[i] = Some(st);
                    }
                }
                let mut ple_spans = Vec::with_capacity(spans.len());
                for ((span, toks), st) in spans.iter().zip(&seq_tokens).zip(slot) {
                    let state = st.ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "qwen4exp: sequence {} has no PLE state in this wave",
                            span.seq
                        ))
                    })?;
                    ple_spans.push(PleSpan {
                        start: span.start,
                        len: span.len,
                        tokens: toks,
                        state,
                        capture: cap_map
                            .as_ref()
                            .is_some_and(|c| c.seqs.contains_key(&span.seq)),
                    });
                }
                // Its own phase, opened and closed here before the mixer opens
                // the same generation: every PLE transient is consumed by the
                // two fused launches, and the residual they write is the
                // wave's own buffer.
                let Device::Cuda(cuda) = dev else {
                    candle::bail!("qwen4exp: the PLE block runs on CUDA");
                };
                let ple_wave = begin_wave(&cuda.cuda_stream(), LayerPhase::Attention)?;
                let captured = ple_apply_spans_fused(
                    &mut res,
                    &mut ple_spans,
                    m.ple_table.as_ref(),
                    &m.ple_w,
                    &cfg.ple,
                    eps,
                    Some(ple_wave.ticket()),
                )?;
                drop(ple_spans);
                // Kept before the phase closes: the captured rows are views on
                // it, copied into the cohort's kept-row buffer at each
                // sequence's capture row.
                if let Some(c) = cap_map.as_mut() {
                    let SpecCapture {
                        seqs, rows: kept, ..
                    } = c;
                    for (span, rows) in spans.iter().zip(captured) {
                        if let (Some(s), Some(rows)) = (seqs.get_mut(&span.seq), rows) {
                            s.ple_rows = Some(CaptureRows::keep(&kept.ple, s.row, &rows)?);
                        }
                    }
                }
                drop(ple_wave);
            }
            if let Some(g) = g_ple {
                g.end();
            }

            // ── Token mixer under the first HC module. ──
            //
            // The mixer's transients — the Gated Residual's wide intermediates,
            // the projections' q8a128 operands, the DeltaNet scan buffers —
            // belong to this phase's arena span. Opened here and dropped before
            // the FFN phase below, because `begin_wave` refuses a phase that is
            // already open: the two spans are reused in turn, not held at once.
            #[cfg(feature = "cuda")]
            let mix_wave = match dev {
                Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
                _ => None,
            };
            let g_pre = crate::models::profile::gpu_span("q4e:gr_pre", dev);
            // A DeltaNet layer whose four input projections run int8 reads the
            // mix's own q8a128 operand, written as the collapse stores `h` — the
            // bytes a quantize of `h` would produce, without the launch, provided
            // the mix quantizes under the projections' `Σx` convention (checked
            // where the operand is handed over). Any other layer's projections
            // take the float: an attention layer quantizes through its own hook.
            #[cfg(feature = "cuda")]
            let (h, inject, h_q8) = match &layer.mix {
                GpuLayerMix::DeltaNet(w) if w.input_mode().is_int8() => hc_mix_with_operand(
                    &res,
                    &layer.hc_attn,
                    eps,
                    mix_wave.as_ref().map(|g| g.ticket()),
                )?,
                _ => {
                    let (h, inject) = hc_mix(
                        &res,
                        &layer.hc_attn,
                        eps,
                        mix_wave.as_ref().map(|g| g.ticket()),
                    )?;
                    (h, inject, None)
                }
            };
            #[cfg(not(feature = "cuda"))]
            let (h, inject) = hc_mix(&res, &layer.hc_attn, eps, None)?;
            let inject = inject.expect("layer HC modules carry an inject");
            g_pre.end();
            #[cfg(feature = "tensor-assert")]
            {
                probe(site("q4e.hc_mix.h.L", li), &h);
                probe(site("q4e.hc_mix.inject.L", li), &inject);
            }

            // The mixer's block output for a combine over every row, or `None`
            // when the mixer has already combined each of its row groups into
            // its own rows of the residual.
            let y: Option<LiveTensor<'_>> = match &layer.mix {
                GpuLayerMix::DeltaNet(w) => {
                    let mut seqs: Vec<DeltaNetSeq<'_>> = Vec::with_capacity(spans.len());
                    for (span, &at) in spans.iter().zip(&span_at) {
                        let hook = arms.hook(span.seq, li as u32, at, span.len);
                        let store =
                            rec.get_mut(&span.seq).expect("ensured") as *mut RecurrentStateStore;
                        // SAFETY: each span names a distinct sequence (the
                        // driver refuses duplicates), so the mutable borrows
                        // are disjoint.
                        let store = unsafe { &mut *store };
                        // The live state the span reads, and where its work
                        // lands: the layer's timeline for a short span — a
                        // verify block's included, which the rewind folds — or
                        // the prefill scratch for a long one.
                        let (state, write) = store.layer_span_mut(li, span.len)?;
                        seqs.push(DeltaNetSeq {
                            start: span.start,
                            len: span.len,
                            state,
                            write,
                            hook,
                        });
                    }
                    let acts = match h_q8 {
                        Some(op) if op.sum_scale == w.input_sum_scale() => {
                            DynamicActs::Int8(op.with_lead(vec![total_rows]))
                        }
                        Some(op) => candle::bail!(
                            "layer {li}: the hyper-connection mix quantized under {:?} but the \
                             DeltaNet projections read {:?} — the operand would be the wrong \
                             bytes for them",
                            op.sum_scale,
                            w.input_sum_scale()
                        ),
                        None => DynamicActs::Float(h.clone()),
                    };
                    #[cfg(feature = "cuda")]
                    let layer_table = match &dn_table {
                        Some(t) => Some(t.layer_slice(li)?),
                        None => None,
                    };
                    #[cfg(not(feature = "cuda"))]
                    let layer_table = None;
                    let mixed = quantized_delta_net_layer_forward_spans(
                        &acts,
                        h.dtype(),
                        w,
                        &cfg.delta_net,
                        &mut seqs,
                        eps,
                        layer_table.as_ref(),
                        ZGate::Sigmoid,
                        #[cfg(feature = "cuda")]
                        mix_wave.as_ref(),
                        #[cfg(not(feature = "cuda"))]
                        None,
                    )?;
                    drop(seqs);
                    Some(mixed)
                }
                GpuLayerMix::Attention {
                    w,
                    indexer,
                    compress_ratio,
                } => {
                    let kv = kv_index(&cfg.layer_kinds, li)?;
                    // ── QSA (§3.1): cache this segment's index keys, then
                    // select. The keys are cached at every depth — a wave that
                    // crosses the budget scores blocks the waves below it
                    // built — and the selection engages only once some row has
                    // more visible cells than the budget, which is where it
                    // stops being the identity (§12.5).
                    let g_sel = crate::models::profile::gpu_span("q4e:qsa_select", dev);
                    let qsa = self.layer_selection(
                        kv,
                        *compress_ratio,
                        indexer,
                        &index_rope,
                        &h,
                        &spans,
                        &offsets_all,
                        &mut idx_map,
                        cap_map.as_mut(),
                        total_rows,
                        // **The mixer's span, not the forward's.** These tables
                        // are rebuilt for every attention layer and dead by the
                        // end of it, so the forward-scoped span — sized for the
                        // handful of kilobytes a wave builds ONCE — filled up
                        // partway through the sweep and the rest fell back to
                        // the pool. A per-layer span resets under them, which is
                        // exactly their lifetime.
                        #[cfg(feature = "cuda")]
                        mix_wave.as_ref().map(|g| g.ticket()),
                        #[cfg(not(feature = "cuda"))]
                        None,
                    )?;
                    g_sel.end();
                    let dec_sel = match &qsa {
                        Some(s) if n_decode > 0 => Some(s.rows_slice(0, n_decode)?),
                        _ => None,
                    };
                    let pre_sel = match &qsa {
                        Some(s) if pre_rows > 0 => Some(s.rows_slice(n_decode, pre_rows)?),
                        _ => None,
                    };
                    let alayer = Qwen4ExpAttentionLayer {
                        w,
                        n_head: cfg.num_attention_heads,
                        n_kv_head: cfg.num_kv_heads,
                        head_dim: cfg.attn_head_dim,
                        rotary: &m.rotary,
                    };
                    // The two branches' outputs stay on the mixer's span. They
                    // are read once, by the combine below, inside the same
                    // phase — so owning them was a full `[rows, n_embd]` copy
                    // into a fresh pool allocation, per branch, per layer, per
                    // wave, to hand a kernel bytes it only reads. A pure-decode
                    // wave now copies nothing at all here; a mixed one pays only
                    // the `cat`.
                    //
                    // The generation is handed to the projection rather than
                    // left as `None`: the attention phase's plan already prices
                    // this layer's Q/K/V splits and out-projection output
                    // (`WaveBuffer::{QSplit, KSplit, VContiguous, OProjOutput}`,
                    // all of which name `LayerPhase::Attention`), so passing
                    // nothing spent that ground on the pool instead. It is also
                    // what binds `'w` here — without it these outputs type as
                    // `'static` while the provenance they inherit from `h` puts
                    // them on the span regardless, which is a lifetime the
                    // compiler cannot police.
                    //
                    // Each group's output is combined straight into its own row
                    // range of the residual (`hc_combine` takes a row view), so a
                    // mixed wave never concatenates the two (hot-path invariant
                    // 2). The groups read `h`, never `res`, so combining the
                    // decode rows before the prefill group runs changes nothing
                    // it sees.
                    let mut cache_refs: Vec<&mut KvCache> = contexts
                        .iter_mut()
                        .map(|c| &mut c.kv_caches.caches[kv])
                        .collect();
                    let (dec_c, pre_c) = cache_refs.split_at_mut(n_decode);
                    if n_decode > 0 {
                        // A row range of `h` is dense, so the group reads it in place.
                        let x_g = TensorCat::from_cat_tensor(
                            h.narrow(0, 0, n_decode)?.reshape((n_decode, 1, n_embd))?,
                            0,
                        )?;
                        expect_dense_view(x_g.as_cat_tensor(), "q4e decode attention rows")?;
                        // The whole attention block, not just its kernel.
                        // `decode:kernel` and `prefill:kernel` are reported by the
                        // kernel wrappers themselves, so the projections, rope, KV
                        // append and out-proj around them would otherwise go
                        // unattributed. The difference
                        // between this span and the kernel row inside it is that
                        // surrounding work.
                        let g_attn = crate::models::profile::gpu_span("q4e:attn_decode", dev);
                        let out = forward_attn_batched(
                            &alayer,
                            dec_c,
                            &x_g,
                            dec_off,
                            &dec_params,
                            kv,
                            dec_sel.as_ref(),
                            mix_wave.as_ref(),
                        )?;
                        g_attn.end();
                        #[cfg(feature = "tensor-assert")]
                        attention_checkpoint(
                            site("q4e.mix.y.L", li),
                            &out,
                            x_g.as_cat_tensor(),
                            &decode_rows,
                            dec_c,
                            |c| {
                                forward_attn_batched(
                                    &alayer,
                                    c,
                                    &x_g,
                                    dec_off,
                                    &dec_params,
                                    kv,
                                    dec_sel.as_ref(),
                                    mix_wave.as_ref(),
                                )
                            },
                            capture_dev,
                        )?;
                        combine_rows(
                            &mut res,
                            &out.reshape((n_decode, n_embd))?,
                            &inject,
                            0,
                            n_decode,
                            dev,
                        )?;
                    }
                    if pre_rows > 0 {
                        let x_g = TensorCat::from_cat_tensor(
                            h.narrow(0, n_decode, pre_rows)?
                                .reshape((1, pre_rows, n_embd))?,
                            0,
                        )?;
                        // Starts at row `n_decode` of `h`; the projections
                        // address it from its own first element.
                        expect_dense_view(x_g.as_cat_tensor(), "q4e prefill attention rows")?;
                        let g_attn = crate::models::profile::gpu_span(attn_prefill_span, dev);
                        let out = forward_attn_batched(
                            &alayer,
                            pre_c,
                            &x_g,
                            pre_off,
                            &pre_params,
                            kv,
                            pre_sel.as_ref(),
                            mix_wave.as_ref(),
                        )?;
                        g_attn.end();
                        #[cfg(feature = "tensor-assert")]
                        attention_checkpoint(
                            site("q4e.mix.y.L", li),
                            &out,
                            x_g.as_cat_tensor(),
                            pre_q,
                            pre_c,
                            |c| {
                                forward_attn_batched(
                                    &alayer,
                                    c,
                                    &x_g,
                                    pre_off,
                                    &pre_params,
                                    kv,
                                    pre_sel.as_ref(),
                                    mix_wave.as_ref(),
                                )
                            },
                            capture_dev,
                        )?;
                        combine_rows(
                            &mut res,
                            &out.reshape((pre_rows, n_embd))?,
                            &inject,
                            n_decode,
                            pre_rows,
                            dev,
                        )?;
                    }
                    None
                }
            };
            if let Some(y) = y {
                #[cfg(feature = "tensor-assert")]
                probe(site("q4e.mix.y.L", li), &y);
                combine_rows(&mut res, &y, &inject, 0, total_rows, dev)?;
            }
            #[cfg(feature = "tensor-assert")]
            probe(site("q4e.post_mix.res.L", li), &res);
            // The mixer's span is done with: `res` is the residual, which lives
            // outside the tier, and nothing below reads a mixer transient.
            //
            // `y` borrows this guard, and that borrow is what the compiler
            // polices — not by forcing a drop here (the borrow ends at `y`'s
            // last use, which is the combine above), but by refusing any *later*
            // read of it. Moving the combine below this line does not compile,
            // which is the property that matters: a phase output cannot be named
            // after the generation that reclaims its range.
            #[cfg(feature = "cuda")]
            drop(mix_wave);

            // ── MoE under the second HC module. The machinery consumes the
            // flat `[1, rows, hidden]` activation layout every FFN path feeds
            // it (the fused SwiGLU kernels are written for it). ──
            #[cfg(feature = "cuda")]
            let ffn_wave = match dev {
                Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Ffn)?),
                _ => None,
            };
            let g_pre2 = crate::models::profile::gpu_span("q4e:gr_pre_ffn", dev);
            // The FFN's input arrives already quantized when the module runs int8:
            // the collapse writes the operand as it stores `h2`.
            let (h2, inject2, h2_q8) = hc_mix_with_operand(
                &res,
                &layer.hc_ffn,
                eps,
                ffn_wave.as_ref().map(|g| g.ticket()),
            )?;
            let inject2 = inject2.expect("layer HC modules carry an inject");
            g_pre2.end();
            let candle::Device::Cuda(cuda) = dev else {
                candle::bail!("qwen4exp wave runs on CUDA");
            };
            // **The gap the first hunt could not see into.** `post_mix.res` was
            // clean and `moe.shared_gated` was the first bad site in the wave, and
            // everything between them — the pre-FFN HC module's output and its
            // inject — was uninstrumented. The MoE's two halves share exactly one
            // input, and they went non-finite with identical counts, which is what
            // a bad input looks like and not what bad expert weights look like.
            // So this is where the answer is.
            #[cfg(feature = "tensor-assert")]
            {
                probe(site("q4e.hc_ffn.h2.L", li), &h2);
                probe(site("q4e.hc_ffn.inject2.L", li), &inject2);
            }
            let h2_3d = h2.reshape((1, total_rows, n_embd))?;
            // Quantized ONCE, in the session's mode, into the one q8a128
            // operand every consumer of the FFN input reads — the shared
            // expert, its gate, the router, and the routed experts, whose tile
            // gather copies the 20 tiles of each 2560-wide row it routes. Handed
            // over as float, the experts gathered float rows and quantized the
            // stacked `rows × top_k` block themselves, once per layer.
            // Σx by amax (`sum_scale::ACT_SUM_SCALE`): this block's sums pass
            // f16 after a full-attention layer.
            let g_acts = crate::models::profile::gpu_span("q4e:moe_acts", dev);
            let acts = match h2_q8 {
                // The mix's own operand, at the flat layout's leading dims.
                Some(op) => DynamicActs::Int8(op.with_lead(vec![1, total_rows])),
                None => to_dynamic(&h2_3d, m.lm_head.int8mode(), cuda, ACT_SUM_SCALE)?,
            };
            g_acts.end();
            #[cfg(feature = "cuda")]
            let moe_wave = ffn_wave.as_ref();
            #[cfg(not(feature = "cuda"))]
            let moe_wave = None;
            // On the FFN span, like the mixer's output above: the combine below
            // reads it inside the same phase, so owning it was a full
            // `[rows, n_embd]` copy into a pool allocation on *every* layer.
            //
            // **The MoE block's device time, on its own span.** 512 experts per
            // layer — shared expert, router, bucketize, gather, the grouped expert
            // GEMMs (including any worker copies of non-VRAM experts) and the
            // scatter — is the largest block of device work in the model. Nothing
            // here waits on the host, so this event span is the one place that
            // device time is attributed.
            let g_moe = crate::models::profile::gpu_span(moe_span, dev);
            // The next rows' routers, applied to this layer's input in its own
            // stacked projection: their votes predict those rows' experts for the
            // expert cache's read-ahead and staging, while the cache copies
            // experts often enough to repay them (`ExpertCache::reads_ahead`).
            // A decode-width launch votes [`LOOK_AHEAD_HOPS`] rows ahead for the
            // device's read-ahead; a prefill-width one votes on the next row
            // only, whose voted set the stager reads off the drive while this
            // layer computes — a prompt routes about half of every row, not all
            // of it, so the next row's own router is what says which half.
            let look_ahead: Vec<&QMatMul> = if layer.moe.routed.cache.reads_ahead() {
                m.layers[li + 1..(li + 1 + look_ahead_hops(total_rows)).min(num_layers)]
                    .iter()
                    .map(|l| &l.moe.routed.gate)
                    .collect()
            } else {
                Vec::new()
            };
            // The layer's output in its three parts: the shared expert's gate is
            // applied by the combine below, which reads the block output anyway,
            // rather than by three launches of its own.
            let parts =
                layer
                    .moe
                    .forward_parts(acts, DType::F32, &decode_like, moe_wave, &look_ahead)?;
            let routed = parts.routed.reshape((total_rows, n_embd))?;
            g_moe.end();
            // **The shared expert is probed where the combine reads it**, not
            // computed a second time to be probed: an F32 recompute through
            // ordinary tensor ops allocated inside the recording wave and freed
            // its temporaries there, a free the capture refuses — and the
            // refused capture wedged the device's stream for every thread.
            #[cfg(feature = "tensor-assert")]
            probe(site("q4e.moe.shared.L", li), &parts.shared.y);
            #[cfg(feature = "tensor-assert")]
            probe(site("q4e.moe.routed.L", li), &routed);
            let g_comb2 = crate::models::profile::gpu_span("q4e:gr_combine_ffn", dev);
            hc_combine_gated(&mut res, &routed, &parts.shared, &inject2)?;
            g_comb2.end();
            #[cfg(feature = "tensor-assert")]
            probe(site("q4e.post_moe.res.L", li), &res);
            // Same reasoning as the mixer's drop above: `res` has left the span,
            // `y2`'s borrow ended at the combine, and the next layer's mixer
            // phase needs this one closed.
            #[cfg(feature = "cuda")]
            drop(ffn_wave);
        }
        // ── The draft head, in the same wave over the same rows. ──
        //
        // Only when the sweep reached the last trunk layer, because the head's
        // input is the trunk's *finished* residual — a windowed sweep that
        // stops early hands its residual on and the head runs behind the window
        // that completes it. Before the maps are dropped: the head selects over
        // its own index cache and needs the same borrow the trunk layers took.
        if layer_end == num_layers {
            if let Some(head) = &m.mtp {
                let mut seeds = self
                    .seeds
                    .write()
                    .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
                let mut spares = self
                    .seed_spares
                    .write()
                    .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
                let hw = HeadWave {
                    n_decode,
                    pre_rows,
                    dec_off,
                    pre_off,
                    dec_params: &dec_params,
                    pre_params: &pre_params,
                    spans: &spans,
                    offsets_all: &offsets_all,
                    index_rope: &index_rope,
                    kv_layer: cfg
                        .mtp_kv_layer()
                        .ok_or_else(|| candle::Error::msg("a head means a head KV layer"))?,
                    generation,
                };
                let embeds = row_embeds
                    .as_ref()
                    .expect("gathered whenever the head runs behind this window");
                self.head_wave_pass(
                    head,
                    contexts,
                    &mut idx_map,
                    cap_map.as_mut(),
                    &mut seeds,
                    &mut spares,
                    &res,
                    embeds,
                    &hw,
                    eps,
                )?;
            }
        }

        drop(rec);
        drop(ple_map);
        drop(idx_map);

        if layer_end < num_layers {
            let flat = res.reshape((1, total_rows, hc * n_embd))?;
            return Ok((
                WavePhase::Residual(TensorCat::from_cat_tensor(flat, 0)?),
                None,
            ));
        }

        // ── Head: the final mix IS the output norm; score decode rows + each
        // prefill's last row through the LM head in one GEMM. ──
        //
        // Decode rows, then each prefill span's LAST row — except a verifying
        // span, where EVERY row is scored.
        //
        // A prefill is normally scored once because only its final position
        // continues the sequence; the rows before it are context. A verify
        // block is the opposite: each of its positions is a proposal, and what
        // checks a proposal is the model's own prediction at the position
        // before it. Scoring only the last row would hand the accept walk one
        // row where it needs `len`, which is exactly what the driver's row
        // count catches.
        let mut sel: Vec<u32> = (0..n_decode as u32).collect();
        let mut acc = n_decode as u32;
        for (i, &l) in pre_q.iter().enumerate() {
            let verifying = cap_map.as_ref().is_some_and(|c| {
                seq_ids
                    .get(n_decode + i)
                    .is_some_and(|s| c.seqs.contains_key(s))
            });
            if verifying {
                sel.extend(acc..acc + l as u32);
            } else {
                sel.push(acc + l as u32 - 1);
            }
            acc += l as u32;
        }
        // **The rows are chosen BEFORE the mix, not after.** The mix is
        // row-wise, so mixing only the scored rows gives the same bits, and a
        // prefill wave scores a handful of its thousands of rows: mixing the
        // whole residual first ran the head's norm, its two low-rank GEMMs and
        // the collapse over every row to keep a few. A wave that scores every
        // row (all decode) takes the residual as it stands.
        //
        // The mix runs on the forward span, where the plan prices it at the
        // scored rows (`WaveBuffer::HyperHead*`). A scattered selection gathers
        // beside the residual, which is on that span too, with its index
        // (`WaveBuffer::HeadScoredResidual`, `WaveBuffer::HeadScoredRows`).
        let r_total = sel.len();
        let scored_res = select_head_rows(&res, sel, 0, fwd_ticket)?;
        // **The head is probed like a layer.** A NaN refused at the logits
        // while the last recurrent layer's state stayed finite was born at or
        // after the last attention layer — and the trunk's last probe is that
        // layer's residual, so the head's own steps are what tell the attention
        // from the mix and the projection after it.
        #[cfg(feature = "tensor-assert")]
        probe(site("q4e.head.res.L", num_layers), &scored_res);
        let (scored, _) = hc_mix(&scored_res, &m.out_hc, eps, fwd_ticket)?;
        #[cfg(feature = "tensor-assert")]
        probe(site("q4e.head.mix.L", num_layers), &scored);
        let acts = {
            let candle::Device::Cuda(cuda) = dev else {
                candle::bail!("qwen4exp wave runs on CUDA");
            };
            to_dynamic(&scored, m.lm_head.int8mode(), cuda, m.lm_head.sum_scale())?
        };
        let logits = m
            .lm_head
            .forward_dynamic(acts.as_dynamic(), DType::F32)?
            .reshape((r_total, cfg.vocab_size))?;
        #[cfg(feature = "tensor-assert")]
        probe(site("q4e.head.logits.L", num_layers), &logits);
        // Keeping the session's per-layer lengths in step after the head is
        // the DRIVER's job (the default advance hook) — nothing more here.
        Ok((
            WavePhase::Logits(TensorCat::from_cat_tensor(logits, 0)?),
            None,
        ))
    }
}

/// `hc_combine` over rows `[start, start + len)` of the residual: the block
/// output covers exactly those rows, and the combine writes them in place
/// through a row view, reading the matching rows of `inject`.
pub(super) fn combine_rows(
    res: &mut Tensor,
    block_out: &LiveTensor<'_>,
    inject: &Tensor,
    start: usize,
    len: usize,
    dev: &Device,
) -> Result<()> {
    let g_comb = crate::models::profile::gpu_span("q4e:gr_combine", dev);
    if start == 0 && len == res.dim(0)? {
        hc_combine(res, block_out, inject)?;
    } else {
        let mut rows = res.narrow(0, start, len)?;
        hc_combine(&mut rows, block_out, &inject.narrow(0, start, len)?)?;
    }
    g_comb.end();
    Ok(())
}

/// The KV-cache index of attention layer `li` (three quarters of the trunk
/// owns no cache).
pub(super) fn kv_index(kinds: &[LayerKind], li: usize) -> Result<usize> {
    if !matches!(kinds[li], LayerKind::Attention) {
        candle::bail!("layer {li} is not an attention layer");
    }
    Ok(kinds[..li]
        .iter()
        .filter(|k| matches!(k, LayerKind::Attention))
        .count())
}
