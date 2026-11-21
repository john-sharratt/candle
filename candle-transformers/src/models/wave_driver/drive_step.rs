//! Drives one co-batched step: its full-sweep rows across every layer, and its
//! creep across the layers of its window, inside one forward span.
//!
//! Every segment's KV is admitted before the step's tier is placed, and the tier
//! is placed once. The segments then run on one forward span, and the residual of
//! the full-sweep rows is handed from one segment to the next on it. A creep that
//! starts at layer zero reads nothing from an arena left by an earlier attempt,
//! so that arena is given back rather than carried forward. Every shape the
//! move-out depends on is settled before any segment runs, because a refusal
//! after the window's recurrent and KV state had committed could not be undone.

use super::*;

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
            // A wide window's placement, with the span it stood on: a frontier
            // that climbs gigabytes over a sweep while little more is live is
            // read straight off these, window by window.
            let placed = tier_footprint(plan);
            if placed >= 1 << 30 {
                if let Some(rs) = candle_nn::kv_cache::region_stats(stream.context().ordinal()) {
                    tracing::info!(
                        target: "candle_transformers::wave_step",
                        layers = format!("[{}, {})", segs.first().map_or(0, |s| s.start), segs.last().map_or(0, |s| s.end)),
                        placed_mib = placed >> 20,
                        live = rs.live,
                        frontier = rs.live_watermark,
                        total = rs.total,
                        "wide step placed its tier"
                    );
                }
            }
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
        // A buffer past what its phase was priced at names its call site; what
        // says which segment of the step was priced short is the step's own
        // composition — every segment's priced phases, the one that ran marked.
        // Measured on WhiteWorld (b4ea6e33b, 2026-10-19): a 230 MiB fresh
        // residual asked of a 16 MiB forward-phase arena, a segment priced for
        // a decode width running a cohort's rows.
        let result = result.map_err(|e| {
            let msg = e.to_string();
            if !msg.contains("wave arena exhausted") {
                return e;
            }
            let segments: Vec<String> = segs
                .iter()
                .zip(&prices)
                .enumerate()
                .map(|(k, (g, p))| {
                    format!(
                        "{}[{}, {}){}: {:?}",
                        if k == i { ">" } else { "" },
                        g.start,
                        g.end,
                        if g.creep { " creep" } else { "" },
                        p
                    )
                })
                .collect();
            // And the domain as it stands: the plan recorded, each arena's laid-out
            // capacity and the generations live — whether the spans were laid out
            // for the plan this step ran.
            let layout = match model.device() {
                Device::Cuda(cuda) => candle_nn::kv_cache::wave_domain_layout(&cuda.cuda_stream())
                    .map_or_else(
                        || "; no plan recorded".to_string(),
                        |(planned, laid, live)| {
                            format!(
                                "; domain planned {planned:?} B, laid out {laid:?} B, {live} live"
                            )
                        },
                    ),
                _ => String::new(),
            };
            candle::Error::Msg(format!(
                "{msg} — in segment {i} (marked >) of a step of {} segments {}{layout}",
                segs.len(),
                segments.join(" ")
            ))
        });
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
