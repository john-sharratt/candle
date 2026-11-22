//! Runs one forward over a layer range: its rows in internal order, the sweep, and
//! the rollback or advance around it.
//!
//! A failed wave leaves no trace. Every layer's usage is rolled back to its entry
//! length on the way out, so a retry is a retry and not a decode against per-layer
//! windows. The decode rows advance once per step, after the head has run and its
//! logits have been copied off, so a token the caller never receives advances
//! nothing. A single-token prefill runs on the decode kernel, since the paged
//! prefill kernel diverges from the decode kernel at one token. The caller's row
//! order is restored at the end.

use super::*;

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
pub(super) fn take_pending_glue(
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
pub(super) fn forward_core<S: WaveSweep + ?Sized>(
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
        // The failed forward's asserts go with it: drained by the next wave
        // instead, a cold expert's given-up rows read as that wave's NaN.
        #[cfg(feature = "tensor-assert")]
        if let Some(dev) = Some(model.device().clone()).filter(|d| d.is_cuda()) {
            if let Err(d) = candle::tensor_assert::discard(&dev) {
                tracing::warn!(
                    target: "candle_transformers::wave_driver",
                    "tensor_assert: the failed forward's findings could not be discarded: {d}"
                );
            }
        }
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
                // **Back to the caller's order, each member with all of its
                // scored rows.** A folded single-token prefill scored its one
                // row among the decode rows; a multi-token member scored its
                // last row — or, verifying, every row of its block — after them;
                // the glue rows trail. Taking one row per member handed a verify
                // block beside a folded prefill a single row and shifted every
                // member after it onto another's logits.
                let verifying = model.verify_row_seqs();
                let scored: Vec<usize> = (0..n_prefill_in)
                    .map(|j| {
                        if verifying.contains(&prefill_seqs[j]) {
                            pre_lens_in[j]
                        } else {
                            1
                        }
                    })
                    .collect();
                let order = caller_order_rows(n_decode_in, &single, &multi, &scored, lg.len())?;
                let mut o: Vec<Tensor> = Vec::with_capacity(lg.len());
                for rows in order {
                    o.extend_from_slice(&lg[rows]);
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
    //
    // **The slots are read on the model's device, not an input's.** The
    // inputs are token ids, which live on the host (the embedding lookup is a
    // CPU `index_select`), so a drain of the first input's device drained the
    // CPU: it read no sites, reset none, and every probe of every wave on
    // WhiteWorld folded into slots nothing ever looked at while NaN logits
    // were refused downstream of them.
    #[cfg(feature = "tensor-assert")]
    {
        let dev = Some(model.device().clone()).filter(|d| d.is_cuda());
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

/// Refuses a residual that cannot hold `rows` of this model's residual rows —
/// the wrong row count, width or dtype.
pub(super) fn check_residual<S: WaveSweep + ?Sized>(
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
pub(super) fn residual_fits(
    residual: &CreepResidual,
    rows: usize,
    width: usize,
    dtype: DType,
) -> bool {
    residual.rows() == rows
        && residual.committed().width() == width
        && residual.committed().dtype() == dtype
}
