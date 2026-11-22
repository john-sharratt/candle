//! Emits a window's summary line and phase breakdown, then resets the counters for
//! the next window.
//!
//! Each flush reads the pool's accounting and the device's residency, and it
//! carves projection, sealing and eviction out of the blocked remainder so that
//! each has a band of its own. Admission outcomes and residency against the
//! prefill floor are logged, so a run that starves shows which rule held it. An
//! idle window is logged at debug level, while a window with work waiting and no
//! forward stays at info, so a stall still shows its phase split. `carve_ms`
//! draws from its buckets in priority order and never takes more than they hold,
//! so the stacked decomposition cannot exceed the window it describes.

use super::*;

impl WaveStats {
    /// Emit the wave summary + phase breakdown, then reset. `kv_vram` is
    /// `(pool_budget_available, pool_used)` in bytes — the pool-accounting
    /// numbers our eviction gate keys on (NOT the driver's raw free, which the
    /// pool's retained reservation pins low). `None` on non-CUDA / query miss.
    /// `backlog` is the point-in-time prefill backlog in tokens
    /// ([`Scheduler::pending_prefill_tokens`]); the line reports
    /// *executed* forwards, so this is the only view of pending *work*.
    /// Call only when [`Self::due`] — windows with NO forwards still flush so
    /// stalls surface their phase split. `bands` is the whole card, cut by what
    /// holds each byte, for the VRAM panel.
    pub(super) fn flush(
        &mut self,
        kv_vram: Option<(usize, usize)>,
        backlog: u64,
        bands: VramBands,
        slots: (u32, u32, u32, u32),
        experts: Option<ExpertCounters>,
        graphs: Option<CaptureStats>,
        trades: Option<admit::rate::TradeWindow>,
        stores: (usize, usize),
    ) {
        let elapsed = self.window_start.elapsed();
        let avg = |sum: u64, n: u64| if n > 0 { sum as f64 / n as f64 } else { 0.0 };
        let phase_sum =
            self.drain_ms + self.promote_ms + self.decode_ms + self.prefill_ms + self.section_ms;
        let unaccounted = (elapsed.as_millis() as u64).saturating_sub(phase_sum);
        // Only emit channels that actually ran this window — an all-zero
        // `decode`/`section` block during a pure-prefill insert was just noise.
        // `tok total` is the tokens fed this window; `kv/fwd avg` is the
        // attended-KV length each forward swept — watch it climb to spot a
        // context-growth slowdown (vs. a fixed paged-glue prefix staying flat).
        let mut parts: Vec<String> = Vec::new();
        let p = &self.prefill;
        if p.fwds > 0 {
            parts.push(format!(
                "prefill fwds={} seqs avg={:.1} max={} tok/fwd avg={:.0} tok total={} kv/fwd avg={:.0} kv total={} fwd avg={:.0}ms",
                p.fwds, avg(p.seq_sum, p.fwds), p.seq_max,
                avg(p.tok_sum, p.fwds), p.tok_sum, avg(p.kv_sum, p.fwds), p.kv_sum, avg(p.ms_sum, p.fwds),
            ));
        }
        let d = &self.decode;
        if d.fwds > 0 {
            parts.push(format!(
                "decode fwds={} seqs avg={:.1} max={} kv/fwd avg={:.0} fwd avg={:.0}ms",
                d.fwds,
                avg(d.seq_sum, d.fwds),
                d.seq_max,
                avg(d.kv_sum, d.fwds),
                avg(d.ms_sum, d.fwds),
            ));
        }
        let s = &self.section;
        if s.fwds > 0 {
            parts.push(format!(
                "section fwds={} seqs avg={:.1} max={} tok/fwd avg={:.0} tok total={} kv/fwd avg={:.0} kv total={} fwd avg={:.0}ms",
                s.fwds, avg(s.seq_sum, s.fwds), s.seq_max, avg(s.tok_sum, s.fwds), s.tok_sum, avg(s.kv_sum, s.fwds), s.kv_sum, avg(s.ms_sum, s.fwds),
            ));
        }
        let ran_forwards = !parts.is_empty();
        let body = if !ran_forwards {
            "(no forwards)".to_string()
        } else {
            parts.join(" | ")
        };
        // The pool budget our eviction defends and our pool_used — watch
        // `budget` fall toward the band (eviction fires) and `used` cap out
        // rather than climb unbounded.
        let vram = match kv_vram {
            Some((budget, used)) => format!(
                " | kv-vram budget={}MiB used={}MiB",
                budget / (1 << 20),
                used / (1 << 20),
            ),
            None => String::new(),
        };
        // Prefill backlog (pending, not-yet-executed tokens) — the large-batch
        // trigger signal. Only emitted when non-zero to keep the line quiet at
        // idle.
        let backlog_str = if backlog > 0 {
            format!(" | backlog={backlog}tok")
        } else {
            String::new()
        };
        // This window's expert activations and how many were already resident —
        // a miss crosses the link inside its layer's GEMMs, and the one number
        // that says whether decode is bound by the card or by the link; and how
        // many experts the window promoted into VRAM. `cold` is the share of the
        // misses that were in no pinned slot and waited on the stager's pack
        // read — the slow kind, which a miss count alone cannot tell apart —
        // and `cpu` the share the host's expert pool computed, which crossed no
        // link: the rest of the misses are what the link carried. `ahead` is
        // what read-ahead copied in ahead of its row, the share of passes it
        // read ahead through, and how often its gate changed its mind — the
        // refill that brings a conceded zone back, and whether it ran.
        let experts_str = match (experts, self.experts_prev) {
            (Some(now), Some(then)) if now.hits + now.misses > then.hits + then.misses => {
                let d = now.since(&then);
                format!(
                    " | experts hit={:.2}% miss={} cold={} cpu={} promoted={} ahead={} on={:.0}% flips={}",
                    100.0 * d.hits as f64 / (d.hits + d.misses) as f64,
                    d.misses,
                    d.cold,
                    d.cpu,
                    d.promoted,
                    d.ahead_claims,
                    100.0 * d.ahead_passes_on as f64 / d.ahead_passes.max(1) as f64,
                    d.ahead_switches,
                )
            }
            _ => String::new(),
        };
        self.experts_prev = experts;
        // This window's wave captures: how many forwards ran as a recorded chain,
        // into how many graph launches they were cut, and how many of those had
        // to be instantiated afresh. Decode steps that outnumber captured waves
        // ran eager — launch-bound at decode width — and a high instantiation
        // share is host time spent building graphs instead of replaying them.
        let graphs_str = match (graphs, self.graphs_prev) {
            (Some(g), Some(g0)) => graph_window(&g0, &g),
            _ => String::new(),
        };
        self.graphs_prev = graphs;
        // This window's prefill-against-decode trades: how many turns were
        // judged on the trade and admitted, the mean weights each side carried,
        // and the mean shares of its own rate each side moved by — the figures
        // the formula is tuned from (`WaveRate::try_widen_trading`).
        // And every admission decision of the window by its outcome — what was
        // admitted (and how much of it outranked the running work), what was
        // refused and by which rule, and where residency stood against the
        // prefill floor — so a run that starves shows which rule held it.
        let trades_str = match trades {
            Some(t) => {
                let mut s = String::new();
                let offered = t.offers_admitted
                    + t.refused_floor
                    + t.refused_cap
                    + t.refused_rate
                    + t.refused_full;
                if offered > 0 {
                    // `res` against the prefill floor admission refuses at, and
                    // the residency the decodes are held to beside it: the
                    // prefill floor alone read 18.7–26.8 GB "above" 3.8 GB while
                    // the zone stood 6.4 GB under its 28.5 GB hold (BlackWorld,
                    // e00cb4300, 2026-10-20).
                    s.push_str(&format!(
                        " | admit ok={} out={} floor={} cap={} rate={} full={} res={}MiB/{}MiB hold={}MiB",
                        t.offers_admitted,
                        t.outranking,
                        t.refused_floor,
                        t.refused_cap,
                        t.refused_rate,
                        t.refused_full,
                        t.last_after >> 20,
                        t.last_floor >> 20,
                        interleave::optimal_weight_bytes().unwrap_or(0) >> 20,
                    ));
                }
                if t.judged > 0 {
                    let n = t.judged as f64;
                    s.push_str(&format!(
                        " | trades n={} ok={} pw={:.1} dw={:.1} dP={:.4} dD={:.4}",
                        t.judged,
                        t.admitted,
                        t.prefill_weight / n,
                        t.decode_weight / n,
                        t.prefill_gain / n,
                        t.decode_change / n,
                    ));
                }
                s
            }
            None => String::new(),
        };
        // The recurrent stores on the device against the slots the scheduler
        // holds, and the parked ones — so a window shows whether idle
        // conversations are giving their stores back.
        let stores_str = if stores.0 + stores.1 > 0 {
            format!(
                " | stores held={} parked={} slots={}",
                stores.0, stores.1, slots.0
            )
        } else {
            String::new()
        };
        // An idle window — no forward, no backlog, nothing decoding, prefilling
        // or ingesting — is a debug line: an idle card closes one every 2 s,
        // and at info those lines would rotate the log's quieter sources out
        // of the files. A window with work waiting and no forward stays at
        // info, so a stall still shows its phase split. The telemetry sample
        // below is pushed either way.
        let idle = !ran_forwards && backlog == 0 && slots.1 == 0 && slots.2 == 0 && slots.3 == 0;
        let line = format!(
            "wave {:.1}s: {body}{vram}{backlog_str}{experts_str}{graphs_str}{trades_str}{stores_str}",
            elapsed.as_secs_f64()
        );
        if idle {
            tracing::debug!("{line}");
        } else {
            tracing::info!("{line}");
        }
        // Phase breakdown: where the wall-clock went on the scheduler thread.
        // `drain` rising over the run ⇒ per-turn reprojection/elevate growing;
        // `reproj` rising ⇒ continuous-reproject (provenance scan/glue) growing;
        // `unaccounted` large ⇒ blocked off-thread (persistence thread / lock).
        // Detailed per-wave breakdown — the live GUI panels carry the same numbers,
        // so this stays at trace and the `wave {}s` heartbeat above is the info line.
        tracing::trace!(
            target: "candle_conversation::scheduler::timing",
            drain_ms = self.drain_ms,
            promote_ms = self.promote_ms,
            decode_ms = self.decode_ms,
            prefill_ms = self.prefill_ms,
            section_ms = self.section_ms,
            reproj_ms = self.reproj_ms,
            unaccounted_ms = unaccounted,
            // Sub-slices INSIDE prefill_ms (not additive to the phases above):
            // where the non-forward per-scope seal cost goes during code-read.
            seal_count = self.seal_count,
            seal_snapshot_ms = self.seal_snapshot_us / 1000,
            seal_sig_ms = self.seal_sig_us / 1000,
            seal_flush_ms = self.seal_flush_us / 1000,
            // seal_sig split: host pointer-resolve / GPU kernel(HtoD+launch+D2H) / host fold.
            prov_resolve_ms =
                PROV_RESOLVE_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000,
            prov_kernel_ms = PROV_KERNEL_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000,
            prov_assemble_ms =
                PROV_ASSEMBLE_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000,
            "wave phase breakdown (scheduler-thread wall-clock; watch which grows)"
        );
        // Swap in this window's accumulated GPU/persistence wait for the Sync phase,
        // plus the drain sub-timers (prefill re-attributed to Prefill; elevate/glue
        // decompose the drain).
        self.wait_ms = WAIT_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        self.maint_ms = MAINT_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        self.drain_prefill_ms =
            DRAIN_PREFILL_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        self.drain_prefill_tokens =
            DRAIN_PREFILL_TOKENS.swap(0, std::sync::atomic::Ordering::Relaxed);
        self.drain_elevate_ms =
            DRAIN_ELEVATE_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        self.drain_glue_ms = DRAIN_GLUE_US.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000;
        // Feed the live GUI's phase timeline: one measurement per phase that ran
        // this window. Volume is tokens for the inference phases and bytes for the
        // memory phases; the GUI colors by kind and shows the flip over time.
        self.push_phase_window(elapsed.as_millis() as u32);
        // Feed the instrumented generic panels (VRAM / throughput / backlog / wave
        // latency / arenas) — the same numbers the log line carries, straight from
        // the ring so the dashboard needs no log tail.
        let tok = self.prefill.tok_sum
            + self.decode.tok_sum
            + self.section.tok_sum
            + self.drain_prefill_tokens;
        let fwds = self.prefill.fwds + self.decode.fwds + self.section.fwds;
        let fwd_sum = self.prefill.ms_sum + self.decode.ms_sum + self.section.ms_sum;
        let fwd_ms = fwd_sum.checked_div(fwds).unwrap_or(0);
        let (budget_mib, used_mib) = kv_vram
            .map(|(b, u)| ((b / (1 << 20)) as u64, (u / (1 << 20)) as u64))
            .unwrap_or((0, 0));
        // Projection decomposition. The drain-path PREFILL is re-attributed to the
        // Prefill phase (below), so the projection band's drain contribution is
        // `drain_ms - drain_prefill`, decomposed into elevate / glue / other.
        // Reproject decomposes into scan / glue / layout / other.
        let use_ms = |a: &std::sync::atomic::AtomicU64| {
            a.swap(0, std::sync::atomic::Ordering::Relaxed) / 1000
        };
        let pdrain = self.drain_ms.saturating_sub(self.drain_prefill_ms);
        let proj = (
            pdrain,
            self.drain_elevate_ms.min(pdrain),
            self.drain_glue_ms.min(pdrain),
            self.reproj_ms,
            use_ms(&REPROJ_SCAN_US),
            use_ms(&REPROJ_GLUE_US),
            use_ms(&REPROJ_LAYOUT_US),
        );
        let timing = self.timing.take();
        phase_ring::push_wave(phase_ring::wave_sample(
            elapsed.as_millis() as u32,
            fwd_ms as u32,
            tok,
            budget_mib,
            used_mib,
            backlog,
            bands,
            proj,
            slots,
            (timing.waves, timing.wall_us, timing.overhead_us),
        ));
        self.window_start = std::time::Instant::now();
        self.prefill = WaveChannel::default();
        self.decode = WaveChannel::default();
        self.section = WaveChannel::default();
        self.drain_ms = 0;
        self.promote_ms = 0;
        self.decode_ms = 0;
        self.prefill_ms = 0;
        self.section_ms = 0;
        self.reproj_ms = 0;
        self.seal_snapshot_us = 0;
        self.seal_sig_us = 0;
        self.seal_flush_us = 0;
        self.seal_count = 0;
        self.evict_bytes = 0;
        self.evict_count = 0;
        self.evict_ms = 0;
        self.idle_ms = 0;
        self.requests_us = 0;
        self.housekeeping_us = 0;
        self.quantize_us = 0;
        self.compaction_us = 0;
        self.warm_us = 0;
        self.compaction_split = compaction_split::CompactionSplit::default();
        self.admission_split = admission_split::AdmissionSplit::default();
        self.telemetry_us = 0;
        self.mid_promote_us = 0;
        self.wait_ms = 0;
        self.maint_ms = 0;
        self.drain_prefill_ms = 0;
        self.drain_prefill_tokens = 0;
        self.drain_elevate_ms = 0;
        self.drain_glue_ms = 0;
        self.cobatched_prefill_us = 0;
        self.cobatched_section_us = 0;
        self.decode_committed = 0;
    }

    /// Emit this window's per-phase measurements to the GUI ring as a **disjoint**
    /// decomposition of the wave's wall-clock — the phase durations sum to
    /// `window_ms`, so the GUI can stack them as "time in phase" without
    /// over-counting.
    ///
    /// The reliable top-level split is the five timed quanta (drain, promote,
    /// decode, prefill, section) plus the leftover "blocked". Two costs are
    /// SUB-SLICES that run *inside* those quanta and would double-count if drawn as
    /// their own segment on top:
    /// - `reproj_ms` runs inside the decode quantum → carved out of decode into
    ///   Projection.
    /// - `seal_ms` runs inside prefill and decode (`perform_seal_and_write`)
    ///   → carved out of those into Sealing.
    /// - `evict_ms` runs partly in the flush block (reclaim/demote → the blocked
    ///   remainder) and partly inside the quanta (relief) → carved out, blocked
    ///   first, into Eviction.
    ///
    /// Carving redistributes ms between buckets (never adds), so the sum is
    /// preserved at `window_ms`.
    pub(super) fn push_phase_window(&self, window_ms: u32) {
        use phase_ring::{PhaseKind, PhaseMeasure};

        // Disjoint base buckets (sum to window_ms). The drain-path prefill (the
        // ingest turn's own content forward) is real prefill work that happens
        // inside `drain_submissions`, so move its time OUT of the projection band
        // and INTO Prefill — with its tokens (below) — so ingest throughput isn't
        // hidden as token-less projection time.
        let drain_prefill = self.drain_prefill_ms.min(self.drain_ms);
        let mut proj_dur = self.drain_ms.saturating_sub(drain_prefill) + self.reproj_ms;
        // Continuous-fair-wave co-batching folds the prefill cohort and section
        // chunks INTO the decode quantum — one shared forward per wave — so their
        // own quanta (`prefill_ms`/`section_ms`) hold little and their time sits in
        // `decode_ms`. Each decode step's creep share — the step past a
        // decode-only one (`close_decode_step`) — is moved out of decode into those phases, so
        // every band is wall-clock the window actually spent and the bands sum to
        // it. Charging the shared forward to every class instead stacked one sweep
        // two and three times: a 5.3 s window drew a 9.9 s bar, its 65-token decode
        // credited with the whole 5.3 s.
        let cobatched_prefill = self.cobatched_prefill_us / 1000;
        let cobatched_section = self.cobatched_section_us / 1000;
        let mut decode_dur = self
            .decode_ms
            .saturating_sub(self.reproj_ms)
            .saturating_sub(cobatched_prefill + cobatched_section);
        let mut prefill_dur = self.prefill_ms + cobatched_prefill + drain_prefill;
        let mut section_dur = self.section_ms + cobatched_section;
        // The loop's own admission pass, plus the one run inside the decode
        // quantum — carved out of decode, where its wall clock already is.
        let alloc_dur =
            self.promote_ms + carve_ms(self.mid_promote_us / 1000, &mut [&mut decode_dur]);
        let accounted =
            self.drain_ms + self.promote_ms + self.decode_ms + self.prefill_ms + self.section_ms;
        let mut blocked_dur = (window_ms as u64).saturating_sub(accounted);

        // Housekeeping runs entirely outside the five quanta, so its time is only
        // ever in the remainder. Carved BEFORE sealing, because the turn-reprefill
        // seal runs inside it and sealing must be able to take from it.
        let mut housekeeping_dur = carve_ms(self.housekeeping_us / 1000, &mut [&mut blocked_dur]);
        // The gap quantize runs between forwards, outside every quantum, so its
        // time is only ever in the remainder too.
        let quantize_dur = carve_ms(self.quantize_us / 1000, &mut [&mut blocked_dur]);
        let compaction_dur = carve_ms(self.compaction_us / 1000, &mut [&mut blocked_dur]);
        let telemetry_dur = carve_ms(self.telemetry_us / 1000, &mut [&mut blocked_dur]);
        // The background scan runs between quanta, so its time is only ever in
        // the remainder.
        let warm_dur = carve_ms(self.warm_us / 1000, &mut [&mut blocked_dur]);

        // Sealing runs in two places: the compression-turn seal inside
        // HOUSEKEEPING (`complete_compression_turn` → `perform_seal_and_write`,
        // reached from `promote_finished_prefills_to_decodes`), and the turn seals
        // inside prefill and decode. Housekeeping first — that is where the
        // per-turn K/V snapshot actually runs, and carving only from prefill/decode
        // left it stranded in the housekeeping band no matter how well it was timed.
        let seal_ms = (self.seal_snapshot_us + self.seal_sig_us + self.seal_flush_us) / 1000;
        let seal_dur = carve_ms(
            seal_ms,
            &mut [&mut housekeeping_dur, &mut prefill_dur, &mut decode_dur],
        );
        // Eviction lives in the flush block (blocked) + relief inside the quanta;
        // carve blocked first, then the quanta.
        let evict_dur = carve_ms(
            self.evict_ms,
            &mut [&mut blocked_dur, &mut decode_dur, &mut prefill_dur],
        );
        // Sync (deliberate GPU `synchronize` + persistence `flush_blocking` waits)
        // happens both in the flush block (→ blocked remainder) and inside the
        // quanta (relief mid-decode/prefill) → carve blocked first, then the
        // compute quanta, so a multi-second stall reads as Sync rather than
        // inflating decode/prefill or hiding in blocked.
        let mut sync_dur = carve_ms(
            self.wait_ms,
            &mut [
                &mut blocked_dur,
                &mut decode_dur,
                &mut prefill_dur,
                &mut section_dur,
            ],
        );
        // Persistence-maintenance stall (a segment compaction blocking seal writes)
        // is off-thread blocked time, so it carves ONLY from the blocked remainder
        // — never spilling into decode/prefill/section like an on-thread GPU sync
        // can. Any excess over this window's blocked is dropped rather than wrongly
        // shrinking a compute phase's duration (which would inflate its tok/s).
        sync_dur += carve_ms(self.maint_ms, &mut [&mut blocked_dur]);
        // Idle (loop blocked on `rx.recv()` with no work) is unaccounted time, so
        // it sits in the blocked remainder — carve it out so a scheduler sitting
        // idle between requests reads as Idle, not Blocked. What's left in
        // `blocked_dur` is genuinely unattributed remainder (lock contention, the
        // cheap on-thread flush-block housekeeping).
        let idle_dur = carve_ms(self.idle_ms, &mut [&mut blocked_dur]);
        // Request handling is a SUB-SLICE, and of two different buckets — so it is
        // carved, never added, or it double-counts (see this function's contract).
        // Most of it runs inside `drain_submissions`, whose wall-clock is already
        // in `drain_ms` and therefore in the projection band; the rest runs on the
        // idle path in `run()`, outside every quantum, and is in the remainder.
        // Carve projection first, spilling to blocked, so each source is taken
        // from the bucket that actually holds it.
        let requests_dur = carve_ms(
            self.requests_us / 1000,
            &mut [&mut proj_dur, &mut blocked_dur],
        );

        let mut phases: Vec<PhaseMeasure> = Vec::new();
        // The decode phase's tokens are the ones its steps committed, not the
        // rows they scored: a speculative step scores a drafted block per
        // sequence, and counting it whole credited decode with every draft it
        // rejected.
        if self.decode.fwds > 0 || decode_dur > 0 {
            phases.push(PhaseMeasure {
                kind: PhaseKind::Decode,
                dur_ms: decode_dur as u32,
                tokens: self.decode_committed,
                bytes: 0,
                seqs: self.decode.seq_max as u32,
                count: self.decode.fwds as u32,
            });
        }
        // One prefill band for every prefill the window ran: the async-queue
        // dialogue prefill, the drain-path ingest prefill, and section ingest —
        // which is prefill of a section's tokens, run by the same wave step from
        // the prefill quantum. Charted apart, a section sweep's time sat in
        // prefill while its tokens went to section: a 2.1 s, 93-token sweep drew
        // a 3 ms section band at 31,000 tok/s beside a token-less prefill band.
        let prefill_dur = prefill_dur + section_dur;
        let prefill_tok = self.prefill.tok_sum + self.drain_prefill_tokens + self.section.tok_sum;
        let prefill_seqs = self
            .prefill
            .seq_max
            .max(self.section.seq_max)
            .max(if drain_prefill > 0 { 1 } else { 0 });
        let prefill_fwds = self.prefill.fwds + self.section.fwds;
        if prefill_fwds > 0 || prefill_dur > 0 || prefill_tok > 0 {
            phases.push(PhaseMeasure {
                kind: PhaseKind::Prefill,
                dur_ms: prefill_dur as u32,
                tokens: prefill_tok,
                bytes: 0,
                seqs: prefill_seqs as u32,
                count: prefill_fwds as u32,
            });
        }
        let push = |phases: &mut Vec<PhaseMeasure>, kind, dur: u64, bytes, count| {
            phases.push(PhaseMeasure {
                kind,
                dur_ms: dur as u32,
                tokens: 0,
                bytes,
                seqs: 0,
                count,
            });
        };
        if proj_dur > 0 {
            push(&mut phases, PhaseKind::Projection, proj_dur, 0, 0);
        }
        if self.seal_count > 0 || seal_dur > 0 {
            push(
                &mut phases,
                PhaseKind::Sealing,
                seal_dur,
                0,
                self.seal_count as u32,
            );
        }
        if self.evict_count > 0 || self.evict_bytes > 0 {
            push(
                &mut phases,
                PhaseKind::Eviction,
                evict_dur,
                self.evict_bytes,
                self.evict_count as u32,
            );
        }
        if alloc_dur > 0 {
            push(&mut phases, PhaseKind::Allocation, alloc_dur, 0, 0);
            // Which part of admission the band went to: the chart draws one
            // segment, and making it faster needs the part.
            let s = &self.admission_split;
            tracing::info!(
                target: "candle_conversation::scheduler::admission",
                total_ms = alloc_dur,
                passes = s.passes,
                parts = %s.parts_ms(),
                "admission split"
            );
        }
        if sync_dur > 0 {
            push(&mut phases, PhaseKind::Sync, sync_dur, 0, 0);
        }
        if requests_dur > 0 {
            push(&mut phases, PhaseKind::Requests, requests_dur, 0, 0);
        }
        if quantize_dur > 0 {
            push(&mut phases, PhaseKind::Quantize, quantize_dur, 0, 0);
        }
        if warm_dur > 0 {
            push(&mut phases, PhaseKind::Warm, warm_dur, 0, 0);
        }
        if compaction_dur > 0 {
            push(&mut phases, PhaseKind::Compaction, compaction_dur, 0, 0);
            // Which pass the band went to: the chart draws one segment, and
            // making it faster needs the part.
            let s = &self.compaction_split;
            tracing::info!(
                target: "candle_conversation::scheduler::compaction",
                total_ms = compaction_dur,
                gate_ms = s.gate_us / 1000,
                kv_ms = s.kv_us / 1000,
                kv_passes = s.kv_passes,
                kv_moves = s.kv_moves,
                kv_stages = %s.kv_stages_ms(),
                kv_frontier_drop = s.kv_frontier_drop,
                tenants_ms = s.tenants_us / 1000,
                tenant_passes = s.tenant_passes,
                tenant_parts = %s.tenant_parts_ms(),
                tenants_frontier_drop = s.tenants_frontier_drop,
                "compaction split"
            );
        }
        if telemetry_dur > 0 {
            push(&mut phases, PhaseKind::Telemetry, telemetry_dur, 0, 0);
        }
        // Drained EVERY window, unconditionally — the sub-timers accumulate on
        // every wave, but the band they decompose is carved out of `blocked`,
        // which is routinely 0 in a busy window. Draining only when the band is
        // non-zero let the counters carry across windows, so the eventual log
        // reported a multi-window sum against one window's total and `other_ms`
        // saturated to 0 — destroying exactly the attribution the split exists
        // for.
        let (promote, admit, demote, gpu) = run::take_housekeeping_split();
        let (finalise, compression) = run::take_promote_split();
        if housekeeping_dur > 0 {
            push(&mut phases, PhaseKind::Housekeeping, housekeeping_dur, 0, 0);
            // Decompose the band for the log: the dashboard draws one
            // housekeeping segment, but which of the five steps is responsible is
            // what a regression needs.
            tracing::info!(
                target: "candle_conversation::scheduler::housekeeping",
                // `total_ms` is the band AFTER the sealing carve took its share,
                // but the sub-timers below are raw measurements taken before it.
                // `raw_total_ms` is what they actually decompose — subtracting
                // them from the carved total made `other_ms` saturate to 0
                // whenever the reprefill seal was material.
                total_ms = housekeeping_dur,
                raw_total_ms = self.housekeeping_us / 1000,
                promote_ms = promote,
                admit_ms = admit,
                demote_ms = demote,
                gpu_drain_ms = gpu,
                other_ms = (self.housekeeping_us / 1000)
                    .saturating_sub(promote + admit + demote + gpu),
                // Promote dominates the band, so it carries its own split.
                promote_finalise_ms = finalise,
                promote_compression_ms = compression,
                "housekeeping split"
            );
        }
        if idle_dur > 0 {
            push(&mut phases, PhaseKind::Idle, idle_dur, 0, 0);
        }
        if blocked_dur > 0 {
            push(&mut phases, PhaseKind::Blocked, blocked_dur, 0, 0);
        }
        phase_ring::push_window(window_ms, phases);
    }
}

/// Carve a sub-slice of `amt` ms out of `buckets` in priority order, draining each
/// before moving to the next. Returns the amount actually carved — bounded by the
/// total the buckets hold, so a carved-out phase segment can never push the stacked
/// decomposition past the window. Redistributes ms (subtracts from a bucket, hands
/// it to the caller's segment), so the overall sum across buckets + segment is
/// preserved.
pub(super) fn carve_ms(amt: u64, buckets: &mut [&mut u64]) -> u64 {
    let mut rem = amt;
    let mut taken = 0;
    for b in buckets.iter_mut() {
        let t = rem.min(**b);
        **b -= t;
        rem -= t;
        taken += t;
        if rem == 0 {
            break;
        }
    }
    taken
}
