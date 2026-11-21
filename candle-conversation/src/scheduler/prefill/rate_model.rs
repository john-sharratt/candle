//! Prices prefills, creeps and decode steps with the learned rate model, and
//! sizes the cohorts it forms and the residency they leave behind.
//!
//! The model learns forward time from forwards this machine has measured, and
//! the figures it publishes are the ones a gateway places turns by. Each figure
//! is priced at the width a placed turn would run at and the residency that
//! width leaves, so the numbers describe the load a placement would send. A
//! cohort is traded against the running decodes the way admission trades it,
//! and a gallery's keep time is set from the expert misses it would spare.
//! Figures are priced only when a reader takes them, since pricing on every
//! step cost more than the steps it measured.

use super::*;

/// How often the learned rates are logged — and, with no reader registered
/// (`phase_ring::rates_read_every`), how often they are priced at all.
pub(in super::super) const RATES_LOG_EVERY: std::time::Duration =
    std::time::Duration::from_secs(60);

/// The rows a cohort of `rows` that needs `need` bytes of tier and KV shrinks
/// to, to fit `ground`: scaled by the shortfall and a margin for the staircase
/// a tier's regions make — or `None` when it already fits, or is one chunk and
/// can shrink no further. Always fewer rows than it had, never under a chunk.
pub(super) fn fitted_rows(rows: usize, need: usize, ground: usize) -> Option<usize> {
    /// Under the scaled width, for the staircase a tier's regions make.
    const FIT_MARGIN: f64 = 0.97;
    if need <= ground || rows <= CHUNK_SIZE {
        return None;
    }
    let scaled = (rows as f64 * ground as f64 / need as f64 * FIT_MARGIN) as usize;
    Some(scaled.clamp(CHUNK_SIZE, rows - 1))
}

/// The longest a decode-less creep step runs: what a prompt arriving while
/// nothing decodes waits, at most, before the creep yields to it at the step's
/// end. Short against a whole 8K-row sweep at depth (~2.6 s), long against the
/// step's fixed cost, so the sweep runs in a handful of windows, not dozens.
pub(in super::super) const DECODE_LESS_SLICE: std::time::Duration =
    std::time::Duration::from_millis(250);

/// Layers a creep advances per wave, out of `n`, at airtime ratio `ratio`:
/// `ceil(n / ratio)`, never zero.
pub(super) fn creep_layer_budget(n: usize, ratio: u32) -> usize {
    n.div_ceil(ratio.max(1) as usize).max(1)
}

impl Scheduler {
    /// Set the gallery's keep time `T*` from the trade it stands for
    /// (`provenance::gallery_arena::eviction`): a turn earns its ground while a
    /// scan uses it more often than its re-upload costs, per byte, over what
    /// the same ground would save the weights per byte per second.
    ///
    /// **The weights' side is the miss stall one more slot would spare.** A
    /// routed expert not resident streams its image over the link — served from
    /// pinned memory, so a miss need not evict anything, and a zone held small
    /// and saturated can stall on every layer with its evictions near zero. So
    /// the measure is the misses, not the evictions: one more slot keeps one
    /// more of the experts not resident, and with routing spread over them it is
    /// `1 / (experts − resident slots)` of the misses that then hit, each
    /// sparing an image's link time, `slot bytes / link rate`. Per byte of the
    /// slot, the slot's size cancels:
    ///
    /// ```text
    /// misses/s × 1e6 / (link bytes/s × (experts − resident slots))   µs/s per byte
    /// ```
    ///
    /// No misses, no pressure: the gallery keeps what it holds.
    pub(in super::super) fn update_gallery_keep(&mut self) {
        /// Time constant of the miss rate's average, in seconds.
        const RATE_TAU_SECS: f64 = 10.0;
        let Some(arena) = self.gallery_arena.clone() else {
            return;
        };
        let Some((misses, link, total, slots)) = self.model.expert_stats_shared().and_then(|s| {
            s.lock().ok().map(|s| {
                let slots = s.zone_bytes.checked_div(s.expert_slot_bytes).unwrap_or(0);
                (
                    s.expert_misses,
                    s.link_bytes_per_sec,
                    s.total_experts,
                    slots,
                )
            })
        }) else {
            return;
        };
        // **The link the scheduler measured, where the pipeline measured none.**
        // The pipeline probes its link only for read-ahead or the CPU class, and
        // a card running neither publishes zero — which priced every weight byte
        // at nothing, so the gallery kept all it held: on WhiteWorld (e119fd7d7,
        // 2026-10-19) 4.16 GB of it beside a 3.7 GB expert zone, `link_gb_s=0.00`.
        // The rate model probes the same bus at start and raises it on the
        // forwards that beat it.
        let link = if link > 0.0 {
            link
        } else {
            self.wave_rate
                .as_ref()
                .map_or(0.0, |r| r.link_bytes_per_s())
        };
        let now = Instant::now();
        if let Some((then, seen)) = self.gallery_misses_seen {
            let secs = now.duration_since(then).as_secs_f64();
            if secs > 0.0 {
                // The tally restarts each interval: a count below the last one
                // read is all new.
                let new = if misses >= seen {
                    misses - seen
                } else {
                    misses
                };
                let w = 1.0 - (-secs / RATE_TAU_SECS).exp();
                self.gallery_miss_rate += w * (new as f64 / secs - self.gallery_miss_rate);
            }
        }
        self.gallery_misses_seen = Some((now, misses));
        let not_resident = total.saturating_sub(slots);
        let weight_us_per_byte_sec = if link > 0.0 && not_resident > 0 {
            self.gallery_miss_rate * 1e6 / (link * not_resident as f64)
        } else {
            0.0
        };
        // The rebuild a scan waits for, as timed on the scans that rebuilt;
        // before one has, its least part — the copy over the link — so a cold
        // gallery is priced rather than kept whole.
        let upload = match arena.upload_us_per_byte() {
            u if u > 0.0 => u,
            _ if link > 0.0 => 1e6 / link,
            _ => 0.0,
        };
        let keep =
            crate::provenance::gallery_arena::eviction::keep_within(upload, weight_us_per_byte_sec);
        arena.set_keep_within(keep);
        // The trade's inputs, once a minute: a gallery that never gives a turn
        // up is either priced to keep everything or scanned within its keep
        // time throughout, and nothing else says which. On WhiteWorld
        // (6ebf6205a, 2026-10-19) 4.16 GB of it stood beside a 3.7 GB expert
        // zone with no eviction ever logged.
        if self
            .gallery_logged_at
            .is_none_or(|t| now.duration_since(t) >= std::time::Duration::from_secs(60))
        {
            self.gallery_logged_at = Some(now);
            let (turns, pinned) = arena.residency_counts();
            tracing::info!(
                target: "candle_conversation::provenance",
                miss_per_s = format!("{:.0}", self.gallery_miss_rate),
                link_gb_s = format!("{:.2}", link / 1e9),
                not_resident,
                upload_ns_per_kib = (upload * 1024.0 * 1000.0) as u64,
                keep_ms = keep.map(|k| k.as_millis() as u64),
                turns,
                pinned,
                "gallery keep trade"
            );
        }
        // And applied as it moves, not only when the gallery next uploads or
        // relief next runs: a gallery nothing scans holds its ground until one
        // of those comes, and on WhiteWorld (9940d0ffc, 2026-10-19) 4 GB of it
        // stood minutes past its keep time while heavy-decode's opening prompts
        // were refused at the floor. Every five seconds: the pass walks every
        // resident turn under the residency lock scans take — ~109k of them on
        // a member after its preload — and a turn past a keep time of minutes
        // loses nothing by going seconds later. With the dead-turn sweep
        // (`gallery_sweep::SWEEP_EVERY`) it cost BlackWorld's wave thread ~2.0 s
        // a minute at once a second, ~0.8 s at five (2026-10-30).
        if self
            .gallery_evicted_at
            .is_none_or(|t| now.duration_since(t) >= std::time::Duration::from_secs(5))
        {
            self.gallery_evicted_at = Some(now);
            arena.evict_past_keep();
        }
    }

    /// Publish what the rate model holds now (`phase_ring::learned_rates`): its
    /// prefill rate at the width the held cohort runs — or, with none held, the
    /// width admission would offer — and its decode step with and without one
    /// more decode.
    pub(in super::super) fn publish_learned_rates(&mut self) {
        if self.wave_rate.is_none() {
            return;
        }
        // What a placed turn would wait behind: the `High` prompts ahead of it,
        // not the background work it goes ahead of. On WhiteWorld (b4ea6e33b,
        // 2026-10-19) a 68–92K-token backlog of file ingest read as every
        // placed turn's wait, and the card took 2 of 48 dialogue turns.
        let high = |id: SequenceId| self.decode_priority_or_high(id).is_live();
        let high_backlog = self
            .active_prefills
            .iter()
            .filter(|p| high(p.work.sequence_id))
            .map(|p| p.work.tokens.len().saturating_sub(p.offset))
            .chain(
                self.prefill_queue
                    .iter()
                    .filter(|w| high(w.sequence_id))
                    .map(|w| w.tokens.len()),
            )
            .chain(
                self.pending_submits
                    .iter()
                    .filter(|p| high(p.parent()))
                    .map(|p| p.work.tokens.len()),
            )
            .sum::<usize>() as u64;
        // A prompt above `Low` that admission has held this long is what a placed
        // turn would stand behind too, whatever its band: 220 `Normal` turns
        // waiting 86 minutes read as no backlog at all, and the gateway went on
        // placing a person's requests on the card that held them (WhiteWorld,
        // 2026-10-31).
        let stalled = |band: DecodePriority, at: Instant| {
            band == DecodePriority::Normal && at.elapsed() >= STALLED_PROMPT
        };
        let stalled_backlog = self
            .prefill_queue
            .iter()
            .filter(|w| stalled(self.decode_priority_or_high(w.sequence_id), w.submitted_at))
            .map(|w| w.tokens.len())
            .chain(
                self.pending_submits
                    .iter()
                    .filter(|p| {
                        stalled(
                            self.decode_priority_or_high(p.parent()),
                            p.work.submitted_at,
                        )
                    })
                    .map(|p| p.work.tokens.len()),
            )
            .sum::<usize>() as u64;
        // **The rate a placed turn's prompt drains at, measured while a `High`
        // backlog stands** — the backlog it joins and the decodes beside it
        // included, which the model's forward throughput is not (`high_drain`).
        // Metered every step, for the airtime split reads the backlog per step.
        self.high_backlog_rows = high_backlog as usize;
        // The turn-content forwards since the last tick, before the meter moves
        // on: they ran under the state the last tick saw.
        let content = super::super::TURN_CONTENT_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
        let since = content.saturating_sub(self.turn_content_seen);
        self.turn_content_seen = content;
        self.high_drain.drained(since as usize);
        let now = Instant::now();
        self.high_drain.advance(now, high_backlog > 0);
        // **Priced as often as it is read, never per step.** The figures below
        // re-trade the cohort and compose its tier for every width they probe;
        // their one reader takes them once a period (`phase_ring::rates_read_every`),
        // and with no reader only the minute's log line does. Priced on every
        // step, the work nobody read sat between the steps: on BlackWorld
        // (41bd79ae3, 2026-10-20) heavy-read stepped 19–26 s of each minute
        // with 34–43 s outside any step.
        let period = super::super::phase_ring::rates_read_period().unwrap_or(RATES_LOG_EVERY);
        if self
            .rates_priced_at
            .is_some_and(|t| now.duration_since(t) < period)
        {
            return;
        }
        self.rates_priced_at = Some(now);
        self.observe_idle_slots(now);
        // **At the width a placed turn would run at, not the width running
        // now.** A forward streams its experts whatever its width, so the rate
        // at a narrow width says almost nothing about the card under the load
        // a placement sends it: on WhiteWorld (359d3bf58, 2026-10-19) a 177-row
        // width read as 40 tok/s, the gateway priced every turn's prefill at
        // that, and placed none here through two patterns. So the width is
        // what this card has waiting to prefill — every admitted prefill's
        // remaining rows and every queued turn's — and one turn more, up to the
        // width admission would offer.
        //
        // **That turn is the size this card's turns are.** A fixed nominal
        // prompt priced every workload as one: heavy-decode's short prompts
        // were charged as 2,048-row ones, and a card's wide prefill rate bit
        // where its real prompts would not (BlackWorld, 437000551, 2026-10-20).
        // The mean prompt of the turns begun here, published so the gateway
        // prices with it too; the gateway's former nominal seeds it until one
        // has begun.
        const SEED_PROMPT_ROWS: usize = 2048;
        let placed_prompt = self
            .prompt_mean
            .tokens()
            .map_or(SEED_PROMPT_ROWS, |t| t.round().max(1.0) as usize);
        let waiting: usize = self
            .active_prefills
            .iter()
            .map(|p| p.work.tokens.len().saturating_sub(p.offset))
            .chain(self.prefill_queue.iter().map(|w| w.tokens.len()))
            .sum();
        let rows = waiting
            .saturating_add(placed_prompt)
            .min(self.prefill_pass_ceiling().max(1));
        let rate = self.wave_rate.as_ref().expect("checked above");
        let (learned, samples, decode_samples) =
            (rate.learned(rows), rate.samples(), rate.decode_samples());
        // **The step a placed decode would get, not the step of the wave now.**
        // With a cohort held, one more decode re-trades it (`traded_width`):
        // the cohort narrows to what its rows still pay for beside that decode,
        // and the decode steps at the residency the narrower cohort leaves.
        // Published at the wave as it stands, a background ingest's width set
        // the figure and no decode was ever placed to narrow it — a fixed point
        // that kept WhiteWorld idle for dialogue (c673b63c9, 2026-10-19).
        let placed_step = self.wave_rate.as_ref().and_then(|r| {
            let q_lens: Vec<usize> =
                self.wave_prefill_members
                    .iter()
                    .map(|m| match m {
                        WaveMember::Prefill { advance, .. }
                        | WaveMember::Section { advance, .. } => *advance,
                    })
                    .collect();
            let seqs: Vec<usize> = self
                .wave_prefill_members
                .iter()
                .map(|m| m.seq_id())
                .collect();
            let offsets: Vec<usize> = seqs
                .iter()
                .map(|&s| self.session.sequence_offset(s).unwrap_or(0))
                .collect();
            let full: usize = q_lens.iter().sum();
            if full == 0 {
                return None;
            }
            let zone = super::super::interleave::effective_weight_zone_bytes(0)?;
            let weights_at = self.memo_weights(&q_lens, &offsets, &seqs, zone);
            // One decode more, and a burst's width more: what the gateway
            // extrapolates the step from as it places turns here, each searched
            // from the width it traded to last time.
            let at = |more, hint| {
                r.placed_decode_step_near(
                    hint,
                    self.decode_width(),
                    more,
                    full,
                    super::super::admit_ground::PREFILL_MIN_ADVANCE,
                    CHUNK_SIZE,
                    q_lens.len() as f64,
                    &weights_at,
                )
            };
            let [_, one_hint, wide_hint, _] = self.rates_width_hints;
            // The width now as well: the figures above it are published as
            // its multiples, so the residency the solve prices them at — the
            // zone less the composed cohort's tier — is the same for all three.
            Some((
                at(0, one_hint),
                at(1, one_hint),
                at(super::super::phase_ring::PLACEMENT_WIDE, wide_hint),
            ))
        });
        // With no backlog standing for long enough to say, the model's figure.
        let drain = self.high_drain.rate();
        // **Before the drain can say, the width the trade would give.** A
        // pattern's opening burst is placed inside the drain's first seconds,
        // and the dedicated forward at the pass ceiling took all of it: on
        // BlackWorld (a965e3a07, 2026-10-19) heavy-read's 24 opening turns went
        // to a member advertising 2,390 tok/s that drained a few hundred. So the
        // cohort a placed turn would join — every waiting prefill and one turn
        // more — is traded against the running decodes as admission will trade
        // it, and runs in a forward that carries the decodes' rows beside it.
        // With no decode the trade keeps the whole width, and the wide rate is
        // what such a card gives.
        let traded = self.wave_rate.as_ref().and_then(|r| {
            let mut q_lens = Vec::new();
            let mut offsets = Vec::new();
            let mut seqs = Vec::new();
            for p in &self.active_prefills {
                q_lens.push(p.work.tokens.len().saturating_sub(p.offset));
                offsets.push(
                    self.session
                        .sequence_offset(p.work.sequence_id.0)
                        .unwrap_or(0),
                );
                seqs.push(p.work.sequence_id.0);
            }
            for w in &self.prefill_queue {
                q_lens.push(w.tokens.len());
                offsets.push(self.session.sequence_offset(w.sequence_id.0).unwrap_or(0));
                seqs.push(w.sequence_id.0);
            }
            // The placed turn itself: no sequence holds it yet, so it is put at
            // the depth this card's prompts have been attending — a tool loop's
            // next prompt continues a conversation, it does not start one.
            q_lens.push(placed_prompt);
            offsets.push(r.context_per_row() as usize);
            seqs.push(usize::MAX);
            let zone = super::super::interleave::effective_weight_zone_bytes(0)?;
            let weights_at = self.memo_weights(&q_lens, &offsets, &seqs, zone);
            let stage = self.decode_stage();
            let decode_rows: usize = stage.iter().map(|&(_, _, b)| b.max(1)).sum();
            let decode_pairs: f64 = stage
                .iter()
                .map(|&(_, at, b)| at as f64 * b.max(1) as f64)
                .sum();
            // The cohort's mean depth per row: each member's rows attend from
            // its offset to its end.
            let cohort_rows: usize = q_lens.iter().sum();
            let context = q_lens
                .iter()
                .zip(&offsets)
                .map(|(&q, &o)| q as f64 * (o as f64 + q as f64 / 2.0))
                .sum::<f64>()
                / cohort_rows.max(1) as f64;
            let at = |more: usize, hint: Option<usize>| {
                // `more` placed turns decoding beside the cohort, each a row at
                // the depth this card's prompts attend.
                r.placed_prefill_near(
                    hint,
                    rows,
                    super::super::admit_ground::PREFILL_MIN_ADVANCE,
                    CHUNK_SIZE,
                    q_lens.len() as f64,
                    self.decode_width() + more,
                    decode_rows + more,
                    context,
                    decode_pairs + more as f64 * r.context_per_row(),
                    &weights_at,
                )
            };
            // The rate now, and the rate a burst's own turns leave once they
            // decode here: what the gateway bends a burst's prefill by, as it
            // bends its step by the wide step.
            Some((
                at(0, self.rates_width_hints[0]),
                at(
                    super::super::phase_ring::PLACEMENT_WIDE,
                    self.rates_width_hints[3],
                ),
            ))
        });
        // The widths the trades settled at, each the next pricing's start.
        self.rates_width_hints = [
            traded.map(|((_, w), _)| w),
            placed_step.map(|(_, (_, w), _)| w),
            placed_step.map(|(_, _, (_, w))| w),
            traded.map(|(_, (_, w))| w),
        ];
        let traded_wide = traded.map(|(_, (rate, _))| rate).filter(|&rate| rate > 0.0);
        let traded = traded.map(|((rate, _), _)| rate).filter(|&rate| rate > 0.0);
        // **The wide rate in the published rate's own currency.** Under a
        // backlog the rate published is the drain measured; the model's wide
        // rate beside it read faster than the measurement (BlackWorld,
        // da11a3717, 2026-10-20: 233–1,796 wide against 182–375 drained), so
        // the burst was clamped flat and went unpriced. The model gives the
        // shape — how much a burst's decodes slow the prefill — and the
        // measurement the scale.
        let prefill_wide = match (drain, traded, traded_wide) {
            (Some(d), Some(now), Some(wide)) => Some(d * wide / now),
            (None, _, wide) => wide,
            _ => None,
        };
        let placed_step = placed_step.map(|((now, _), (one, _), (wide, _))| (now, one, wide));
        let learned = learned.map(|r| super::super::phase_ring::LearnedRates {
            prefill_tok_s: drain.or(traded).unwrap_or(r.prefill_tok_s),
            prefill_tok_s_plus_wide: prefill_wide.unwrap_or(r.prefill_tok_s_plus_wide),
            high_backlog_tokens: high_backlog + stalled_backlog,
            prompt_tokens: self.prompt_mean.tokens().unwrap_or(0.0),
            reply_tokens: self.segment_mean.tokens().unwrap_or(0.0),
            tokens_per_step: self.tokens_per_step.tokens().unwrap_or(0.0),
            // **The placed figures as multiples of the step now.** The solve
            // prices them at the zone less the composed cohort's tier, a
            // residency the step now is not read at, and the hit falls steeply
            // there: WhiteWorld (da11a3717, 2026-10-20) published 37.7 ms now,
            // 438.7 one more and 1,751.5 a burst, while stepping 12 decodes at
            // about 100 ms. Taken as ratios of the same solve, the shape is the
            // model's and the scale the step's.
            decode_step_ms_plus_one: placed_step.map_or(
                r.decode_step_ms_plus_one,
                |(now, one, _)| {
                    if now > 0.0 {
                        r.decode_step_ms * one / now
                    } else {
                        one
                    }
                },
            ),
            decode_step_ms_plus_wide: placed_step.map_or(
                r.decode_step_ms_plus_wide,
                |(now, _, wide)| {
                    if now > 0.0 {
                        r.decode_step_ms * wide / now
                    } else {
                        wide
                    }
                },
            ),
            ..r
        });
        // **Never slower than the card has stepped.** A step at a width bounds
        // every narrower one from above (`step_census::WidthSteps`): a creep
        // riding it only adds. The fit, with no decode-only step to learn from
        // while every step co-batches, froze on heavy-read's last ones and
        // published 0.7–3.3 s against co-batched steps of about 0.3 s
        // (BlackWorld, dcba2cc3b, 2026-10-20). So the step now is held to the
        // fastest measured at its width or wider, the whole curve scaled by the
        // same factor to keep the model's shape around what was measured — and
        // each wider figure is then held to its own width's measurement too,
        // where one has been taken.
        let width = self.decode_width();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        let bound_at = |w: usize| self.width_steps.bound(w, now, RATES_LOG_EVERY).map(ms);
        let (bound, bound_one, bound_wide) = (
            bound_at(width),
            bound_at(width + 1),
            bound_at(width + super::super::phase_ring::PLACEMENT_WIDE),
        );
        let learned = learned.map(|r| {
            let f = match bound {
                Some(b) if r.decode_step_ms > b && r.decode_step_ms > 0.0 => b / r.decode_step_ms,
                _ => 1.0,
            };
            let held = |v: f64, at: Option<f64>| at.map_or(v * f, |b| (v * f).min(b));
            super::super::phase_ring::LearnedRates {
                decode_step_ms: held(r.decode_step_ms, bound),
                decode_step_ms_plus_one: held(r.decode_step_ms_plus_one, bound_one),
                decode_step_ms_plus_wide: held(r.decode_step_ms_plus_wide, bound_wide),
                ..r
            }
        });
        if let Some(rates) = learned {
            super::super::phase_ring::publish_learned_rates(rates);
        }
        // What this member advertises, once a minute — and that it advertises
        // nothing, which a gateway reads as unlearned: the figure a placement
        // that never sends this card a turn is to be checked against.
        if self
            .rates_logged_at
            .is_none_or(|t| now.duration_since(t) >= RATES_LOG_EVERY)
        {
            self.rates_logged_at = Some(now);
            self.log_pinned_working_sets();
            // And the model behind them, for the next start (`export_learned`).
            if let Some(r) = self.wave_rate.as_ref() {
                super::super::phase_ring::keep_learned_model(&r.export_learned());
            }
            let ((copy, compute, attend), context, (hit_model, fit)) = self
                .wave_rate
                .as_ref()
                .map_or(((0.0, 0.0, 0.0), 0.0, (0.0, None)), |r| {
                    (
                        r.forward_shares(rows),
                        r.context_per_row(),
                        r.decode_terms(),
                    )
                });
            let depth = self.wave_rate.as_ref().and_then(|r| r.decode_depth());
            let decode_fit = fit.map(|(a, b, c, d)| {
                let (e, ctx) = depth.unwrap_or((0.0, 0.0));
                format!(
                    "a={:.3}ms b={:.4}ms e={:.3}us/Kpos@{:.0} c={:.2}us d={:.3}us/Mpair",
                    a * 1e3,
                    b * 1e3,
                    e * 1e9,
                    ctx,
                    c * 1e6,
                    d * 1e12
                )
            });
            // What the fit has seen, answered or not: whether a missing answer
            // is the data's (too few steps, too little spread) or the solver's.
            let fit_seen = self.wave_rate.as_ref().map(|r| {
                let (n, [m, q, r_sd, p], why) = r.decode_fit_report();
                format!(
                    "n={n} sd_m={m:.2} sd_q={q:.2} sd_r={r_sd:.1} sd_p={p:.2}{}",
                    why.map_or(String::new(), |w| format!(" none={w:?}"))
                )
            });
            // Its own line: at the end of the rates line it was past what a
            // log reader shows of one.
            let steps = self.step_census.take(now);
            tracing::info!(
                target: "candle_conversation::scheduler::admission",
                "step census {steps}"
            );
            let idle = self.idle_census(now);
            match learned {
                Some(r) => tracing::info!(
                    target: "candle_conversation::scheduler::admission",
                    rows,
                    prefill_tok_s = r.prefill_tok_s.round() as u64,
                    prefill_tok_s_plus_wide = r.prefill_tok_s_plus_wide.round() as u64,
                    high_backlog_tokens = r.high_backlog_tokens,
                    prompt_tokens = r.prompt_tokens.round() as u64,
                    reply_tokens = r.reply_tokens.round() as u64,
                    tokens_per_step = format!("{:.2}", r.tokens_per_step),
                    decode_step_ms = format!("{:.1}", r.decode_step_ms),
                    decode_step_ms_plus_one = format!("{:.1}", r.decode_step_ms_plus_one),
                    decode_step_ms_plus_wide = format!("{:.1}", r.decode_step_ms_plus_wide),
                    step_bound_ms = format!(
                        "{}/{}/{}",
                        bound.map_or("-".into(), |b| format!("{b:.1}")),
                        bound_one.map_or("-".into(), |b| format!("{b:.1}")),
                        bound_wide.map_or("-".into(), |b| format!("{b:.1}")),
                    ),
                    prefill_from = if drain.is_some() {
                        "high drain"
                    } else if traded.is_some() {
                        "traded width"
                    } else {
                        "model"
                    },
                    high_airtime_ratio = self.high_airtime.map(|r| format!("{r:.3}")),
                    step_debt_ratio = format!("{:.3}", self.wave_step_ratio),
                    segment_tokens = self.segment_mean.tokens().map(|t| t.round() as u64),
                    creep_rows_per_s = self.creep_airtime.per_s().map(|r| r.round() as u64),
                    forward_shares = format!("copy={copy:.2} compute={compute:.2} attend={attend:.2}"),
                    context_per_row = context.round() as u64,
                    hit_model = format!("{hit_model:.3}"),
                    decode_fit,
                    fit_seen,
                    idle,
                    prefill_samples = samples,
                    decode_samples,
                    "learned rates published"
                ),
                None => tracing::info!(
                    target: "candle_conversation::scheduler::admission",
                    prefill_samples = samples,
                    decode_samples,
                    "learned rates not yet published: the rate model has not learned both halves"
                ),
            }
        }
    }

    /// The widest a cohort of these members may run beside `decodes` decodes
    /// routing to `routed` experts a layer, by the trade admission judges by
    /// (`WaveRate::traded_width`), and the residency that width leaves. `None`
    /// when there is nothing to trade: no rate model, or no member.
    ///
    /// **The width a held cohort is re-fitted to.** A cohort admitted once
    /// re-forms every sweep, and fitted only to the ground it reaches it kept
    /// its width beside any decode that arrived later: on WhiteWorld
    /// (c673b63c9, 2026-10-19) a background ingest held the weights down so
    /// that the card's decode stepped at 0.5–0.8 s with one turn on it.
    /// Re-traded at each fit, it narrows to the width its prefill still pays
    /// for, and keeps advancing.
    pub(in super::super) fn traded_width(
        &self,
        q_lens: &[usize],
        offsets: &[usize],
        seqs: &[usize],
        decodes: usize,
        routed: usize,
    ) -> Option<(usize, u64)> {
        let rate = self.wave_rate.as_ref()?;
        let full: usize = q_lens.iter().sum();
        if full == 0 {
            return None;
        }
        let zone = super::super::interleave::effective_weight_zone_bytes(0)?;
        let weights_at = |rows| self.weights_at_width(q_lens, offsets, seqs, zone, rows);
        let width = rate.traded_width(
            full,
            super::super::admit_ground::PREFILL_MIN_ADVANCE,
            CHUNK_SIZE,
            q_lens.len() as f64,
            decodes,
            routed,
            weights_at,
        );
        Some((
            width,
            self.weights_at_width(q_lens, offsets, seqs, zone, width),
        ))
    }

    /// The residency a cohort of these members leaves at `rows` wide: the zone
    /// less the tier their shares of `rows` ([`Self::cohort_shares`]) place,
    /// beside the running decodes.
    pub(super) fn weights_at_width(
        &self,
        q_lens: &[usize],
        offsets: &[usize],
        seqs: &[usize],
        zone: u64,
        rows: usize,
    ) -> u64 {
        self.weights_at_width_beside(q_lens, offsets, seqs, zone, rows, &self.decode_stage())
    }

    /// [`Self::weights_at_width`] for one cohort at every width a search asks
    /// about, each width composed once: a trade's search re-asks the widths
    /// beside each probe and the searches after it re-ask the same ones, and
    /// each answer is a tier carved over every layer of every sequence — on
    /// BlackWorld (036bc5cd3, 2026-10-20) one pricing of the rates took about
    /// 330 ms of it at 150K depth.
    pub(super) fn memo_weights<'s>(
        &'s self,
        q_lens: &'s [usize],
        offsets: &'s [usize],
        seqs: &'s [usize],
        zone: u64,
    ) -> impl Fn(usize) -> u64 + 's {
        let stage = self.decode_stage();
        let memo = std::cell::RefCell::new(std::collections::HashMap::new());
        move |rows| {
            if let Some(&w) = memo.borrow().get(&rows) {
                return w;
            }
            let w = self.weights_at_width_beside(q_lens, offsets, seqs, zone, rows, &stage);
            memo.borrow_mut().insert(rows, w);
            w
        }
    }

    /// [`Self::weights_at_width`] beside a decode stage the caller built.
    pub(super) fn weights_at_width_beside(
        &self,
        q_lens: &[usize],
        offsets: &[usize],
        seqs: &[usize],
        zone: u64,
        rows: usize,
        stage: &[(usize, usize, usize)],
    ) -> u64 {
        let shares = self.cohort_shares(seqs, q_lens, rows);
        let mut adv = Vec::new();
        let mut off = Vec::new();
        let mut sq = Vec::new();
        for ((&s, &o), &q) in shares.iter().zip(offsets).zip(seqs) {
            if s > 0 {
                adv.push(s);
                off.push(o);
                sq.push(q);
            }
        }
        let width: usize = adv.iter().sum();
        let tier = self
            .model
            .wave_tier_bytes(
                width,
                adv.len().max(1),
                &adv,
                &off,
                &sq,
                stage,
                self.session.activation_dtype(),
            )
            .unwrap_or(0);
        zone.saturating_sub(tier)
    }

    /// The rows each member of a cohort — `seqs`, with `remaining` rows each,
    /// laid out shorts, gang, the rest — advances in a forward of `budget`
    /// rows: [`gang::gang_shares`] over the members of [`Self::prefill_gang`].
    /// The one division a cohort is formed, narrowed and priced by, so a gang
    /// stays in step however its width is cut.
    pub(in super::super) fn cohort_shares(
        &self,
        seqs: &[usize],
        remaining: &[usize],
        budget: usize,
    ) -> Vec<usize> {
        let in_gang: Vec<bool> = seqs.iter().map(|s| self.prefill_gang.contains(s)).collect();
        gang::gang_shares(remaining, &in_gang, budget, CHUNK_SIZE)
    }

    /// The gang a group forms with (`docs/prefill_gangs.md`): the live one —
    /// those of its members still prefilling — or, in a `background` group with
    /// none live, a new one from the long prompts not `paused`, the highest
    /// band's in arrival order, as wide as [`Self::gang_width`] allows. A long
    /// prompt is one a pass cannot finish.
    pub(super) fn prefill_gang_for(&self, background: bool, paused: &HashSet<usize>) -> Vec<usize> {
        let prefilling = |p: &ActivePrefill| {
            p.error.is_none() && p.finished.is_none() && p.offset < p.work.tokens.len()
        };
        let live: Vec<usize> = self
            .prefill_gang
            .iter()
            .copied()
            .filter(|&s| {
                self.active_prefills
                    .iter()
                    .any(|p| p.work.sequence_id.0 == s && prefilling(p))
            })
            .collect();
        if !live.is_empty() || !background {
            return live;
        }
        let cap = self.max_prefill_pass_tokens;
        let longs: Vec<(usize, usize)> = self
            .active_prefills
            .iter()
            .filter(|p| {
                prefilling(p)
                    && !paused.contains(&p.work.sequence_id.0)
                    && p.work.tokens.len() - p.offset > cap
            })
            .map(|p| {
                (
                    p.work.sequence_id.0,
                    rank(self.decode_priority_or_high(p.work.sequence_id)),
                )
            })
            .collect();
        let width = self.gang_width(longs.len());
        let formed = gang::form_gang(&longs, width);
        if !formed.is_empty() {
            let remaining: usize = self
                .active_prefills
                .iter()
                .filter(|p| formed.contains(&p.work.sequence_id.0))
                .map(|p| p.work.tokens.len() - p.offset)
                .sum();
            tracing::info!(
                target: "candle_conversation::scheduler::creep",
                members = formed.len(),
                width,
                longs = longs.len(),
                remaining,
                "prefill gang formed"
            );
        }
        formed
    }

    /// How wide a gang of up to `longs` prompts may run: the widest whose decode
    /// step, by the learned decode model at the weight zone's residency, costs
    /// no more than twice a lone decode's ([`gang::gang_width`]). `1` — no gang
    /// — while the decode model is unlearned.
    pub(super) fn gang_width(&self, longs: usize) -> usize {
        let Some(rate) = self.wave_rate.as_ref().filter(|r| r.decode_learned()) else {
            return 1;
        };
        let Some(zone) = super::super::interleave::effective_weight_zone_bytes(0) else {
            return 1;
        };
        let per = rate.experts_per_decode(0);
        gang::gang_width(longs, |n| {
            rate.decode_step_secs(n.saturating_mul(per), zone)
        })
    }
}
