//! The `WaveSweep` implementation for Qwen4ExpBatched: what a sweep claims, prices, folds and rolls back around the layers it runs.
//!
//! The KV layers a sweep writes are one set everywhere they are named: admission claims them, the failure rollback restores them and the sweep writes them, since a claim short of the sweep allocates inside the forward and a rollback short of it leaves the draft head a token ahead. The tier is priced and placed before the forward opens, and the carried state is bracketed so a failed wave leaves it as it entered. A successful speculative wave hands its entry snapshots to the verify capture rather than copying them. A device fault the sweep gave up on is taken before the commit, so no store keeps state written after the fault.

use super::*;

impl WaveSweep for Qwen4ExpBatched {
    fn device(&self) -> &Device {
        &self.model.device
    }

    /// The armed cohort: the head scores every row of a sequence the verify
    /// capture holds (`SpecCapture::seqs`).
    fn verify_row_seqs(&self) -> Vec<usize> {
        self.verify
            .read()
            .ok()
            .and_then(|g| g.as_ref().map(|c| c.seqs.keys().copied().collect()))
            .unwrap_or_default()
    }

    fn take_device_fault(&self) -> Result<()> {
        self.model.experts.take_segment_fault(&self.model.device)
    }

    fn num_layers(&self) -> usize {
        self.model.cfg.num_layers
    }

    fn prefill_width_cap(&self, act_dtype: DType) -> usize {
        <Self as ManagedBatchedModel>::prefill_width_cap(self, act_dtype)
    }

    /// The Gated Residual's wide stream: `hc` streams of `hidden`, in F32.
    fn residual_row(&self, _session: &BatchedInferenceSession) -> Result<(usize, DType)> {
        let cfg = &self.model.cfg;
        Ok((cfg.hc.count * cfg.hidden_size, DType::F32))
    }

    /// Caches are indexed by KV layer: three quarters of the trunk owns no
    /// paged cache, so the driver's rollback range must be translated.
    ///
    /// **Plus the draft head's layer when the window reaches the last trunk
    /// layer**, because that is when the head's pass runs and writes it. One
    /// answer, three callers — admission claims this range, the failure
    /// rollback restores it, and the sweep writes it — and they must name the
    /// same set. A claim short of the sweep is a chunk allocated from inside
    /// the forward that owns the partition (hot-path invariant 7); a rollback
    /// short of the sweep leaves the head's layer a token ahead of the trunk's
    /// after a failed wave, which the next wave "heals" by truncating a token
    /// the caller was already given.
    fn kv_layer_range(&self, layer_start: usize, layer_end: usize) -> (usize, usize) {
        let cfg = &self.model.cfg;
        let kinds = &cfg.layer_kinds;
        let count = |n: usize| {
            kinds[..n.min(kinds.len())]
                .iter()
                .filter(|k| matches!(k, LayerKind::Attention))
                .count()
        };
        let (start, mut end) = (count(layer_start), count(layer_end));
        if cfg.mtp_kv_layer().is_some() && layer_end == cfg.num_layers {
            end += 1;
        }
        (start, end)
    }

    fn sweep(
        &self,
        session: &mut BatchedInferenceSession,
        wave: WaveGroups<'_>,
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
            act_dtype: _,
            adapter,
        } = wave;
        if seq_ids.is_empty() {
            candle::bail!("qwen4exp wave: empty batch");
        }
        let preparing = matches!(mode, SweepMode::Prepare);
        let in_step = matches!(mode, SweepMode::InStep(_));
        let accept_in_place = session.accept_in_place();
        let stencil_rows = session.stencil_rows();
        // Flash-Next runs its layers under the Gated Residual, which carries no
        // adapter plumbing, so `project_qkv_gated` is called with an empty
        // `LayerLora`. Refused rather than ignored: dropping the name here would
        // serve the BASE model under an adapter's name, which reads as a bad
        // fine-tune rather than as an unsupported architecture.
        if let Some(name) = adapter {
            candle::bail!(
                "qwen4exp wave: Flash-Next has no LoRA support, so adapter `{name}` \
                 cannot be applied"
            );
        }
        let m = &self.model;
        let cfg = &m.cfg;
        let eps = cfg.rms_norm_eps;
        let num_layers = cfg.num_layers;
        if layer_start > layer_end || layer_end > num_layers {
            candle::bail!("qwen4exp wave: bad layer range [{layer_start}, {layer_end})");
        }
        // Host time, in three parts. A decode step at depth is host-bound — the
        // readback after it waits ~1.6 ms — so what the forward thread spends
        // issuing the sweep is the step, and the GPU spans cannot say which part.
        let preamble_span = span("q4e:host:preamble");
        // The preamble's parts, each on its own host span: at depth the
        // preamble is hundreds of milliseconds a forward with the GPU idle, and
        // the whole-preamble span cannot say which part.
        let t_pre = profile_now();

        // Hand back the PREVIOUS wave's transient tier, here rather than at the
        // end of the wave that placed it. `end_wave_transient` gates on
        // `live_generations`, a host-side count: when a sweep returns its
        // generations are dropped but its kernels are still in flight, so
        // releasing there would hand ground back while the GPU is still reading
        // it. What makes this point safe is the work in between, which drains
        // the wave that placed it. It also has to precede `ensure_seq_state`
        // below, which claims span regions for any store it creates — and a
        // claim under a standing tier is refused outright.
        //
        // A segment of a step skips this, the boundary move, the admit and the
        // state claims below: the step ran all four for every segment before it
        // placed the tier they share, and its forward span is live
        // (`docs/wave_step.md` §2.1).
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(d), false) = (&m.device, in_step) {
            end_wave_transient(&d.cuda_stream());
        }
        pipeline_record("q4e:pre:end_tier", t_pre);

        // The KV↔expert boundary's GROWING direction, in the one gap it is legal
        // in: between forwards, on the line after the transient tier goes back, so
        // no tier stands and the region ceiling is the pool's own size — which is
        // what `growth_policy`'s occupancy arithmetic is written against.
        //
        // Without this the boundary only ever moves toward KV (`request_kv_ground`
        // buys on the spot) and expert residency ratchets down across a long run:
        // spare KV regions above the KV side's recent high-water never come back as
        // resident-expert slots. The shrink direction needs no call here — a KV claim
        // that runs out buys its own ground. Mirrors `latent_moe`'s wave and the
        // blanket `BatchedModel` wave's phase 0; Flash-Next runs its own engine
        // rather than `BatchedModelCore`, so it does not inherit either.
        let t_pre = profile_now();
        if !in_step {
            m.experts.reclaim_spare_ground();
        }
        pipeline_record("q4e:pre:reclaim", t_pre);

        let n_glue = seq_ids.len() - n_decode - n_prefill;
        if n_glue > 0 || pending_glue.is_some() {
            candle::bail!(
                "qwen4exp wave: glue rows — reprojection glue is not implemented at \
                 head_dim {}; this stack must recompute rather than gap-fill",
                cfg.attn_head_dim
            );
        }

        // Offsets + query lengths from the session, BEFORE the contexts borrow.
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

        // Attention metadata from the session's shared borrow. The group is the
        // session's STREAM layers — the draft head's layer is stepped by the
        // head's own pass, and `build_decode_metadata` scopes itself to the
        // stream for exactly that reason.
        // A segment being prepared runs no layer, so it builds no header — but
        // it claims the decode rows' write chunks the build would, here, before
        // the step's span stands.
        let t_pre = profile_now();
        if preparing {
            session.ensure_decode_writes(&seq_ids[..n_decode])?;
        }
        let decode_headers = if n_decode > 0 && !preparing {
            let (buf, stride) = session.build_decode_metadata(&seq_ids[..n_decode], generation)?;
            DecodeHeaders::Decode { buf, stride }
        } else {
            DecodeHeaders::Decode {
                buf: None,
                stride: 0,
            }
        };
        pipeline_record("q4e:pre:headers", t_pre);
        // Read before the contexts borrow the session: the tier pricing below
        // needs it, and `assemble_wave_contexts` holds a mutable borrow across
        // everything that follows.
        let tier_act_dtype = session.activation_dtype();
        let t_pre = profile_now();
        let mut contexts = assemble_wave_contexts(session, seq_ids, inputs)?;
        let contexts = contexts.as_mut_slice();
        pipeline_record("q4e:pre:contexts", t_pre);

        // Admit: claim every KV chunk this wave writes, over the KV range.
        let t_pre = profile_now();
        let (kv_start, kv_end) = self.kv_layer_range(layer_start, layer_end);
        if !in_step {
            admit_wave_kv(contexts, n_decode, n_prefill, kv_start, kv_end)?;
        }
        pipeline_record("q4e:pre:admit", t_pre);

        // ── Carried state: ensure (reset at offset 0), open the GDN wave,
        // snapshot the PLE states for the failure bracket. ──
        //
        // **A claim, so it belongs with the other claims — before the tier is
        // placed and before the forward opens.** A sequence entering the wave
        // without a store builds one here, and a store takes state-arena slots —
        // which, when no arena of its geometry has room, claims a reservation region
        // through the same arena window `admit_wave_kv` does. Run after
        // `begin_forward` that claim asks
        // the partition for ground while holding the window that decides who
        // gets it, and the refusal is the unrecoverable one — the thread that
        // would have to end the wave is the thread asking.
        //
        // It sat below `begin_forward` and was invisible for as long as every
        // sequence in a wave already had its store: the lazy branch only fires
        // for a slot that has never run one, or one starting over. A deeper
        // `repo_map` walk (`--max-depth 3`) put the boot priming chain's own
        // ingest into exactly that shape, and the daemon failed to load with
        // "creating a KV arena from inside the forward that owns the partition".
        //
        // Above `plan_wave_transient` as well as `begin_forward`, and that order
        // is load-bearing in both directions: the tier is placed against the
        // arena frontier as it stands, so a region claimed after the placement
        // moves the frontier under a tier already standing on it (hot-path
        // invariant 7).
        let t_pre = profile_now();
        if !in_step {
            for ((&seq, &off), &q) in seq_ids.iter().zip(&offsets).zip(&q_lens) {
                self.ensure_seq_state(seq, off, off + q, layer_start)?;
            }
        }
        pipeline_record("q4e:pre:ensure_state", t_pre);

        // **Price and reserve this wave's transient tier**, sized to this wave
        // rather than to the widest one the engine can run — after the KV claim
        // above, so the arena frontier is final when the tier is placed against
        // it, and before `begin_forward` below, because the placement may buy
        // ground from the weight side and `set_weight_floor` refuses while a
        // forward is open.
        //
        // A segment being prepared returns its price for the step to place; a
        // segment running in the step places nothing — the step's tier stands.
        let t_pre = profile_now();
        #[cfg(feature = "cuda")]
        if total_rows > 0 && !in_step {
            if let Device::Cuda(d) = &m.device {
                let geometry = self.wave_geometry(tier_act_dtype);
                let plan = WavePlan::new(geometry);
                // The wave's composition, not just its row count: the chains
                // price a prefill row, a decode row and a scored row
                // differently, and the Gated Residual's streams are carried by
                // `ModelGeometry::hyper` so its wide intermediates are priced
                // rather than overflowing into the pool.
                //
                // **Scored rows are not all rows.** The head runs every decode
                // row and the LAST row of each prefill span — a verifying span
                // excepted, where every row is a prediction to compare a
                // proposal against (see the head section below). Pricing
                // `HeadLogits` at `total_rows × vocab` buys roughly a gigabyte
                // of tier for a logits block three orders of magnitude smaller,
                // and takes it from the weight side every wave.
                //
                // A wave that stops short of the last layer returns its residual
                // and runs no head, so it scores no rows at all.
                let verify_seqs: Vec<usize> = self
                    .verify
                    .read()
                    .ok()
                    .and_then(|g| g.as_ref().map(|c| c.seqs.keys().copied().collect()))
                    .unwrap_or_default();
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
                    // A one-row prefill group takes the decode kernels, so it
                    // carves no span-table entry and, alone, no scan transient.
                    prefill_spans: pre_q.iter().filter(|&&l| l > 1).count(),
                    // The caller's accept walk selects on the head's span only
                    // when it reads the logits there.
                    accept_rows: if accept_in_place { scored_rows } else { 0 },
                    stencil_rows,
                    qsa_bytes: self.wave_qsa_bytes(
                        seq_ids,
                        &q_lens,
                        &offsets,
                        layer_start,
                        layer_end,
                    )?,
                    // The prefill launch's carve: up to one key chunk of each
                    // bulk chunk's sequence, and the carry beside it.
                    kv_stage: PrefillKvStageLayout::new(
                        pre_q,
                        pre_off,
                        geometry.n_head,
                        geometry.n_kv_head,
                        geometry.head_dim,
                        PrefillLaunchBounds::PRODUCTION,
                    ),
                    look_ahead_hops: look_ahead_hops(total_rows),
                    // The rows the gather's second table holds: one per
                    // virtual id among the wave's inputs.
                    image_rows: count_image_rows(inputs)?,
                };
                let per_phase = [
                    plan.phase_bytes(LayerPhase::Attention, width),
                    plan.phase_bytes(LayerPhase::Ffn, width),
                    plan.phase_bytes(LayerPhase::Forward, width),
                ];
                if preparing {
                    return Ok((WavePhase::Priced(per_phase), None));
                }
                // The refusal names the partition; the wave it was refused for
                // is what says which of the wave's parts to look at.
                plan_wave_transient(&d.cuda_stream(), per_phase).map_err(|e| {
                    candle::Error::Msg(format!("{e} — for a wave of {width:?}, {per_phase:?} B"))
                })?;
            }
        }
        pipeline_record("q4e:pre:plan_tier", t_pre);
        if preparing {
            return Ok((WavePhase::Priced([0; 3]), None));
        }

        // From here the forward owns the span partition: the tier is placed and
        // must not move under it.
        let t_pre = profile_now();
        #[cfg(feature = "cuda")]
        let _forward_open = match &m.device {
            Device::Cuda(d) => Some(begin_forward(&d.cuda_stream())),
            _ => None,
        };
        pipeline_record("q4e:pre:begin_forward", t_pre);

        // **The forward-scoped span**, held for the whole sweep rather than per
        // layer: it carries the metadata a forward builds once and every layer
        // reads — ragged prefill offsets, page and candidate tables, RoPE
        // tables, gathered position ids — which is exactly what
        // `WAVE_FORWARD_BYTES` describes. Its ticket travels instead of the
        // guard, because the builders are deep in the sweep and a `Copy`
        // coordinate reaches them without changing any signature's lifetime.
        // A segment of a step carves from the step's span, which outlives it.
        #[cfg(feature = "cuda")]
        let fwd_span = match (&m.device, mode) {
            (_, SweepMode::InStep(_)) => None,
            (Device::Cuda(d), _) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Forward)?),
            _ => None,
        };
        #[cfg(feature = "cuda")]
        let fwd_ticket = match mode {
            SweepMode::InStep(ticket) => ticket,
            _ => fwd_span.as_ref().map(|g| g.ticket()),
        };
        #[cfg(not(feature = "cuda"))]
        let fwd_ticket: Option<candle::cuda_backend::wave_provenance::WaveTicket> = None;

        // Built AFTER the span opens, so its three tables land on it. They are
        // the wave's ragged prefill offsets — read by every layer, written once
        // — which is the forward span's stated purpose. Nothing above needs
        // them, so moving the construction down costs only this comment.
        let t_pre = profile_now();
        let prefill_headers = DecodeHeaders::Prefill(BatchedPrefillMeta::new_ragged(
            pre_off, pre_q, &m.device, fwd_ticket,
        )?);
        pipeline_record("q4e:pre:meta", t_pre);

        let t_pre = profile_now();
        let index_snapshot: Vec<(usize, Vec<IndexSnapshot>)> = {
            let idx = self
                .index
                .read()
                .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
            seq_ids
                .iter()
                .map(|&s| {
                    let caches = idx.get(&s).expect("ensured above");
                    let snaps = caches
                        .iter()
                        .map(IndexCache::snapshot)
                        .collect::<Result<Vec<_>>>()?;
                    Ok((s, snaps))
                })
                .collect::<Result<Vec<_>>>()?
        };
        let ple_snapshot: Vec<(usize, PleState)> = {
            let ple = self
                .ple
                .read()
                .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
            seq_ids
                .iter()
                .map(|&s| {
                    let st = ple.get(&s).expect("ensured above");
                    Ok((s, st.clone()))
                })
                .collect::<Result<_>>()?
        };
        // The head's seed belongs in the same bracket as the state above.
        // `head_wave_pass` writes it at the end of a sweep that reaches the
        // head, so a wave that then failed — the LM-head GEMM, a relief-driven
        // failure, both routine here — left every other carried state rolled
        // back to `off` and the seed describing `off + q − 1`. The retried
        // wave's head pass would take that as row 0's `h(t−1)` and write the
        // head's committed K/V from it.
        // One row per sequence, and the storage is shared rather than copied:
        // a seed is already owned storage no generation reclaims, and the head
        // pass *replaces* the entry rather than writing through it.
        let seed_snapshot: Vec<(usize, Option<Tensor>)> = {
            let seeds = self
                .seeds
                .read()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            seq_ids
                .iter()
                .map(|&s| (s, seeds.get(&s).cloned()))
                .collect()
        };
        {
            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
            for &s in seq_ids {
                rec.get_mut(&s).expect("ensured above").begin_wave()?;
            }
        }
        pipeline_record("q4e:pre:snapshots", t_pre);

        drop(preamble_span);
        let layers_span = span("q4e:host:layers");
        let swept = self.sweep_layers(
            contexts,
            seq_ids,
            inputs,
            n_decode,
            (dec_off, dec_q, pre_off, pre_q),
            (total_rows, pre_rows),
            (layer_start, layer_end),
            x_in,
            (decode_headers, prefill_headers),
            generation,
            eps,
            fwd_ticket,
        );
        drop(layers_span);
        let _tail_span = span("q4e:host:tail");

        // Close the wave: commit on success, rewind everything on failure. A
        // rollback that itself fails leaves the sequence's state unaccounted
        // — report both, exactly as the driver's KV rollback does.
        //
        // The close folds every short span a sequence outside the armed verify
        // cohort ran — all of its rows, in ONE launch over every such sequence
        // and layer — then commits (`commit_stores`). The armed cohort's layers
        // stay pending for the rewind, which alone knows how many rows each
        // keeps. A close that fails commits nothing, so the wave is failed and
        // rolled back like any other, carried state and all.
        //
        // **And a sweep the device gave part of up fails here, before the
        // commit, not after it.** A wait that gives an expert up does not stop
        // the forward: the later layers run on what the given-up experts left,
        // and the fault is read when the segment ends (`take_device_fault`).
        // Read only by the driver, after this close, it found the stores
        // already committed — each long span's scratch exchanged in as the live
        // state — and rolled back the K/V alone. On WhiteWorld (c3ebac8bf,
        // 2026-10-19) a creep window whose CPU experts were given up at MoE row
        // 36 left its member's recurrent state non-finite from layer 37 on; the
        // retry swept over it, and the turn decoded NaN and was discarded. Taken
        // here, the fault fails the sweep and everything it wrote is rewound —
        // the stores, the PLE histories and the index caches alike — and the
        // driver, finding the fault already taken, fails it on the sweep's own
        // error.
        let swept = swept.and_then(|v| {
            WaveSweep::take_device_fault(self)?;
            Ok(v)
        });
        let swept = {
            let armed: Vec<usize> = self
                .verify
                .read()
                .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?
                .as_ref()
                .map(|c| c.seqs.keys().copied().collect())
                .unwrap_or_default();
            let mut rec = self
                .recurrent
                .write()
                .map_err(|_| candle::Error::Msg("recurrent lock poisoned".into()))?;
            let closed = match &swept {
                Ok(_) => {
                    let mut stores: Vec<(&mut RecurrentStateStore, bool)> = rec
                        .iter_mut()
                        .filter(|(s, _)| seq_ids.contains(s))
                        .map(|(s, store)| (store, !armed.contains(s)))
                        .collect();
                    commit_stores(&mut stores)
                }
                Err(_) => Ok(()),
            };
            let swept = match closed {
                Ok(()) => swept,
                Err(e) => Err(e),
            };
            {
                let mut arms = self
                    .arms
                    .write()
                    .map_err(|_| candle::Error::Msg("arms lock poisoned".into()))?;
                for &s in seq_ids {
                    match &swept {
                        Ok(_) => arms.commit(s),
                        Err(_) => arms.rollback(s),
                    }
                }
            }
            if let Err(e) = &swept {
                for (s, store) in rec.iter_mut() {
                    if !seq_ids.contains(s) {
                        continue;
                    }
                    if let Err(rb) = store.rollback_wave() {
                        candle::bail!(
                            "wave failed ({e}) and sequence {s}'s recurrent rollback also \
                             failed ({rb})"
                        );
                    }
                }
            }
            swept
        };
        if swept.is_err() {
            let mut ple = self
                .ple
                .write()
                .map_err(|_| candle::Error::Msg("ple lock poisoned".into()))?;
            for (s, st) in ple_snapshot {
                ple.insert(s, st);
            }
            let mut idx = self
                .index
                .write()
                .map_err(|_| candle::Error::Msg("index lock poisoned".into()))?;
            for (s, snaps) in &index_snapshot {
                if let Some(caches) = idx.get_mut(s) {
                    for (cache, snap) in caches.iter_mut().zip(snaps) {
                        cache.restore(snap)?;
                    }
                }
            }
            let mut seeds = self
                .seeds
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            let mut spares = self
                .seed_spares
                .write()
                .map_err(|_| candle::Error::Msg("seed lock poisoned".into()))?;
            for (s, seed) in seed_snapshot {
                // The carry may already have flipped: the seed being put back
                // would then also be the spare, and the next carry would write
                // the seed it reads.
                spares.remove(&s);
                match seed {
                    // Restore what the sequence entered the wave with — which
                    // for a sequence that had none is nothing, so the entry is
                    // removed rather than left holding this wave's write.
                    Some(prev) => {
                        seeds.insert(s, prev);
                    }
                    None => {
                        seeds.remove(&s);
                    }
                }
            }
        } else {
            // **The successful path hands the same snapshots to the verify
            // capture instead of dropping them.**
            //
            // A rewind needs the state the block was ENTERED with, and that is
            // exactly what the failure bracket above already took — so a
            // speculative wave pays no snapshot of its own. Moved rather than
            // cloned: on a wave that committed, nothing else reads them.
            let mut g = self
                .verify
                .write()
                .map_err(|_| candle::Error::Msg("verify lock poisoned".into()))?;
            if let Some(cap) = g.as_mut() {
                let mut idx_snaps: HashMap<usize, Vec<IndexSnapshot>> =
                    index_snapshot.into_iter().collect();
                for (s, ple) in ple_snapshot {
                    // **Never an empty stand-in.** Both snapshots are built from
                    // this wave's `seq_ids`, so a sequence present in one and
                    // absent from the other is an inconsistency — and defaulting
                    // it to an empty vector does not paper over the gap, it
                    // silently disarms the rewind's QSA restore while leaving its
                    // re-append running, which puts the index a whole block past
                    // the K/V for the rest of the sequence.
                    let qsa = idx_snaps.remove(&s).ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "qwen4exp: sequence {s} has a PLE entering snapshot but no index \
                             one — the rewind would re-append the accepted rows onto a cache \
                             it never rolled back"
                        ))
                    })?;
                    cap.take_entering(s, ple, qsa);
                }
            }
        }
        // The head's logits are carved from the forward span (`hc_mix` and the
        // head GEMM run on its ticket), so a wave that reached the head hands
        // the span's guard back with them: `WaveResult` keeps the span alive
        // while the caller reads, instead of the head copying the `[R, vocab]`
        // block off it first. A window that stopped short hands its residual
        // back on the same span, and the guard goes with it for the same reason.
        #[cfg(feature = "cuda")]
        let swept = swept.map(|(phase, guard)| (phase, guard.or(fwd_span)));
        swept
    }
}
