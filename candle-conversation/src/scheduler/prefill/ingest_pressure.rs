//! Keeps bulk ingest from pinning the hot KV tier, shedding its warm-backed
//! turns first and throttling admission only when host RAM is over budget.
//!
//! Ingest KV is never attended until query time and reloads from warm on
//! demand, so shedding it costs nothing. Once the KV side is past a fraction of
//! its regions, sealed ingest turns beyond a small rolling window are demoted to
//! the RAM tier, least recently active first, and a turn a live slot is
//! attending is never demoted. Admission is not paced by backlog, because the
//! rate model already judges each offer; the one cut left is a host-RAM guard on
//! the warm tier. Watermarks multiply before they divide, since a region count
//! divided first truncates to zero and would demote the whole tail every wave.

use super::*;

/// Capacity fraction (%) at which cold **ingest** KV starts demoting to the warm
/// (RAM) tier — gentle and early, well before the free-region setpoint is
/// approached at all. Ingest KV is zero-reload-cost (never
/// re-attended until query time; it re-elevates warm→hot on demand), so it is the
/// cheapest relief and sheds first.
pub(super) const INGEST_DEMOTE_PCT: usize = 50;

/// Slack the warm PIPELINE may hold above the standing budget: hot→warm output
/// that exists only while the drain moves it to cold. On a zero-budget machine
/// this is the only warm residency there ever is, and cutting admission for it
/// would recreate the ratchet-to-the-floor failure — the drain clears it in a
/// pass. The throttle fires only when `resident + pending` exceeds
/// `budget + slack`, i.e. when the drain is genuinely not keeping up. 1 GiB.
pub(in super::super) const WARM_PIPELINE_SLACK_BYTES: u64 = 1024 * 1024 * 1024;

/// Minimum spacing between OS memory probes for host-RAM backpressure —
/// `sysinfo` is a syscall, so the scheduler caches the reading between waves.
pub(super) const HOST_RAM_PROBE_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(1000);

/// Sealed ingest turns kept hot per timeline (the rolling window) before the
/// gentle-early demote sheds the rest to RAM.
///
/// **Must cover the ingest projection's gather width.** With the tool-round-trip
/// ingest, each scope's summary decode projects the `scopes` group (`top_k` turns)
/// — i.e. an actively-ingesting conversation RE-ATTENDS its own recent turns every
/// scope. If this window is narrower than that gather, the demote sheds turns the
/// very next projection re-elevates: a warm↔hot churn that stalls the decode batch.
/// The scopes group is `top_k: 4`, so a scope's projected working set is ~4 turns
/// (2 coupled turns × ~2 scopes); 8 keeps a couple of scopes of margin resident so
/// the active working set never leaves hot.
pub(super) const INGEST_HOT_WINDOW: usize = 8;

impl Scheduler {
    /// Gentle-early ingest relief, run per-wave and long before the setpoint is
    /// approached. Once the KV side is more than [`ingest_demote_pct`] occupied
    /// (~50 % of its regions), shed the sealed, warm-backed KV of append-only
    /// ingest timelines down to a small rolling hot window
    /// ([`ingest_hot_window`]).
    ///
    /// Zero reload cost: ingest KV is never re-attended until query time, when
    /// it re-elevates warm→hot on demand. So it is the cheapest thing to shed
    /// and it sheds first, which is what keeps a bulk repo ingest from pinning
    /// a whole corpus hot until real pressure forces a much more expensive
    /// eviction of turns that are actually being attended.
    ///
    /// The watermark used to be `pool_used` against a fraction of the card.
    /// That reading no longer describes KV at all — the pool holds the model,
    /// the expert cache and a few scratches, so it sits at a high, flat
    /// fraction of C forever and the gate would fire on every wave regardless
    /// of how much ingest is resident. Occupancy of the KV span is the same
    /// question asked of the right counter.
    pub(in super::super) fn demote_cold_ingest_if_pressured(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        let Some(stats) = self.kv_regions() else {
            return;
        };
        // Multiply before dividing. The same expression read `capacity / 100 *
        // pct` when `capacity` was bytes (~1.6e10), where the truncation was
        // invisible; `stats.total` is a region *count* in the hundreds, so
        // dividing first quantises the watermark to whole percent-of-100 steps
        // — and on any span below 100 regions it truncates to **zero**, which
        // the `live <= watermark` early-return below can never satisfy. That
        // turns the gentle-early rung into an unconditional full demote of the
        // ingest tail on every wave.
        let watermark = stats.total * INGEST_DEMOTE_PCT / 100;
        if stats.live <= watermark {
            return;
        }
        let used = stats.live.saturating_mul(region_bytes() as usize);
        let watermark = watermark.saturating_mul(region_bytes() as usize);
        let window = INGEST_HOT_WINDOW;
        // Relieve back to the watermark, no further: `target` bounds the LRU walk
        // so the demote sheds the least-recently-active ingest tail just enough to
        // clear the pressure, never the whole hot working set.
        let target_bytes = used.saturating_sub(watermark) as u64;
        // Shed whatever is already warm-backed — free, no migration. A demote
        // takes hot copies off their residences directly and waits on nothing,
        // so it runs here, every wave it is due; this is the device's own
        // thread, so the copies no forward still holds are freed at once.
        tracing::trace!(
            target: "candle_conversation::scheduler::vram_relief",
            used_mib = used / (1 << 20),
            watermark_mib = watermark / (1 << 20),
            target_mib = target_bytes >> 20,
            window,
            "cold-ingest demote"
        );
        let t = Instant::now();
        let report = self.demote_ingest(window, target_bytes);
        // Feed the GUI's phase timeline: the gentle-rung ingest demotion.
        self.wave_stats.add_evict(
            report.bytes,
            report.count as u64,
            t.elapsed().as_millis() as u64,
        );
        self.nudge_if_warm_starved(report, target_bytes);
    }

    /// After a demote that shed `report` toward `target_bytes`: nudge the
    /// persistence drain when it fell short, and release what it emptied.
    pub(super) fn nudge_if_warm_starved(&mut self, report: EvictionReport, target_bytes: u64) {
        // warm-copy production (the async persistence pass) lags the ingest seal
        // rate, so the cold backlog is hot-without-warm and not yet demotable.
        // NUDGE the persistence thread to run its hot→warm drain (non-blocking),
        // and let the next demote shed the freshly-warmed backlog. Never
        // `flush_blocking` here: this runs on the scheduler thread, and under
        // sustained pressure the persist thread is already mid-pass — a blocking
        // wait would stall the scheduler for the full timeout while draining
        // nothing sooner. A `fire()` is a no-op when a pass is already queued.
        // The test is whether the demote *could* shed what it needed to —
        // `report.bytes` against `target_bytes` — not the CUDA pool, whose `used`
        // holds the model, the expert cache and the scratches and so read over
        // the watermark on every wave.
        let nudged = if report.bytes < target_bytes {
            self.persist_triggers.fire();
            true
        } else {
            false
        };
        if report.count > 0 {
            // Freed hot arenas → release, so their regions return to the free
            // list where the pressure signal can see them.
            let _ = self.session.release_empty_arenas();
            tracing::debug!(
                target: "candle_conversation::scheduler::vram_relief",
                ingest_timelines = self.ingest_timelines.len(),
                turns = report.count,
                freed_mib = report.bytes / (1 << 20),
                target_mib = target_bytes >> 20,
                nudged,
                "cold-ingest demote (gentle-early)"
            );
        }
    }

    /// Size the ingest admission window to the **hot→warm drain backlog** — the
    /// leading backpressure signal that keeps `used` off the warm-starved climb
    /// (see the pool-footprint dashboard). The persistence thread publishes its
    /// live backlog via [`PersistenceTriggers::pending_warm_bytes`]; when it
    /// exceeds the target the drain is behind the seal rate, so narrow the AIMD
    /// window (fewer concurrent scopes → lower seal rate → drain catches up);
    /// when it falls below half the target, reopen. `vram_under_pressure` stays
    /// the hard floor beneath this (its per-admission shrinks still fire on a
    /// true VRAM spike). No-op when nothing is ingesting — chat keeps the
    /// per-iteration AIMD recovery in the run loop. Runs at the ~2 s wave
    /// cadence, matching how often the backlog signal refreshes.
    /// Refresh the cached `sysinfo` reading at most once per
    /// [`HOST_RAM_PROBE_INTERVAL`] — never a per-wave syscall — and return the
    /// cached `(available, total)`. `(0, 0)` until the first probe.
    pub(in super::super) fn host_ram_reading(&mut self) -> (u64, u64) {
        let stale = self
            .host_ram_probe
            .map(|(t, _, _)| t.elapsed() >= HOST_RAM_PROBE_INTERVAL)
            .unwrap_or(true);
        if stale {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            self.host_ram_probe = Some((
                std::time::Instant::now(),
                sys.available_memory(),
                sys.total_memory(),
            ));
        }
        self.host_ram_probe
            .map(|(_, a, t)| (a, t))
            .unwrap_or((0, 0))
    }

    /// Whether the warm KV tier has outgrown its host-RAM budget PLUS the drain
    /// pipeline's slack — the condition under which slowing admission actually
    /// helps (less sealing → less hot→warm output). This replaced the absolute
    /// available-RAM floor, which our own resident weights held permanently
    /// true on any machine whose model fills RAM: an untestable condition that
    /// ratcheted the setpoint to the floor against structure, not pressure.
    pub(in super::super) fn warm_over_budget(&mut self) -> bool {
        let (_, total) = self.host_ram_reading();
        if total == 0 {
            return false;
        }
        let budget = candle::vram::host_ram_budget(total);
        let usage = self
            .persist_triggers
            .warm_resident_bytes()
            .saturating_add(self.persist_triggers.pending_warm_bytes());
        usage
            > budget
                .kv_warm_budget_bytes
                .saturating_add(WARM_PIPELINE_SLACK_BYTES)
    }

    /// **Ingest is not paced. The engine's own admission governs it.**
    ///
    /// This used to run an AIMD controller against the hot→warm backlog: cut the
    /// admission budget when the drain fell behind the seal rate, reopen a quantum
    /// when it caught up. That is a throttle on the *ingest*, and it was the wrong
    /// place to put one. Whether a prefill belongs in the next wave is a question
    /// the rate model already answers per offer, in the currency that decides it —
    /// expert bytes over the bus against rows gained (`admit::rate`). The backlog
    /// controller sat in front of that, cutting the width the model was about to
    /// judge, on a signal (drain lag) that says nothing about whether the wave gets
    /// faster. Measured effect: prefill spent long stretches running
    /// single-sequence mini-forwards at a fraction of batched throughput while the
    /// model would have admitted more.
    ///
    /// What remains is the one condition that is **not** a pacing decision: the
    /// warm KV tier outgrowing its host-RAM budget. That is a host-OOM guard — it
    /// has already aborted one full overnight load — and it is a hard cut, not a
    /// controller. Slowing admission genuinely relieves it (less sealing → less
    /// hot→warm output), and nothing else in the engine defends host memory.
    ///
    /// The reopen stays evidence-based and is no longer gated on the backlog:
    /// forwards completing out-of-memory-free are proof the current width is
    /// sustainable, and that is the only proof this controller needs. A device OOM
    /// or an eviction survival still cuts instantly through
    /// [`Self::cut_admit_budget_leveled`], which resets the streak.
    pub(in super::super) fn regulate_ingest_admission(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        if self.warm_over_budget() {
            self.cut_admit_budget_leveled(ThrottleReason::WarmOverBudget);
            return;
        }
        // "Is there room to reopen?" is asked against the STATIC bound, not the
        // live ceiling: this runs every wave, and the live ceiling costs a device
        // query plus a walk of the registered relievers. The live clamp still
        // happens where it matters — inside `raise_admit_budget`, and again at
        // admission time in `promote_new_prefills`.
        let ceiling = Self::max_admit_budget();
        // Volume-floored progress: a tick certifies the current width, which a
        // trickle of tiny forwards cannot (see `EVIDENCE_MIN_PREFILL_TOKENS`).
        // Sub-floor volume accumulates — `admit_ok_tokens_seen` advances only
        // when the floor is cleared.
        let ok_tokens = super::super::PREFILL_OK_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
        let progressed =
            ok_tokens >= self.admit_ok_tokens_seen + super::super::EVIDENCE_MIN_PREFILL_TOKENS;
        if progressed {
            self.admit_ok_tokens_seen = ok_tokens;
        }
        let (grow, streak) = evidence_admit_grow(
            self.admit_budget,
            ceiling,
            progressed,
            self.admit_grow_streak,
            // Cost scales with the budget already held, so the climb slows as it
            // nears the budget that last collapsed instead of charging it at
            // constant speed.
            evidence_ticks_for(budget_notches(self.admit_budget, admit_quantum())),
        );
        self.admit_grow_streak = streak;
        if grow {
            self.raise_admit_budget(ThrottleReason::Throughput);
        }
    }

    /// One LRU-smart cold-ingest demote, toward `target_bytes`, on every live
    /// substrate. `demote_cold_ingest` self-filters to the timelines a
    /// substrate owns (a non-matching id is a no-op), walks its hot turns
    /// least recently touched first so the least-recently-active tail sheds
    /// before an active window, and is idempotent (already-demoted turns hold
    /// no hot copy and are skipped). The protect-list is every live slot's
    /// working set — mirrors [`Self::evict_cold_tail`] — and every residence a
    /// working set pins. Each substrate gets the whole target: ingest
    /// timelines are one tenant's, so in practice one substrate sheds.
    pub(super) fn demote_ingest(&self, window: usize, target_bytes: u64) -> EvictionReport {
        // Protect the active working set of every live slot (what in-flight
        // prefills/decodes are attending) — the same union `evict_cold_tail`
        // builds, so an actively-ingesting conversation's gathered turns are never
        // demoted out from under the next projection.
        let mut keep_sections: Vec<SectionId> = Vec::new();
        let mut keep_turns: Vec<TurnKey> = Vec::new();
        for st in self.slot_projection_state.values() {
            keep_sections.extend(st.working_set.sections.iter().copied());
            keep_turns.extend(st.working_set.turns.iter().copied());
        }
        let mut report = EvictionReport { count: 0, bytes: 0 };
        for conv in self.distinct_substrates() {
            let r = conv.read().demote_cold_ingest(
                &self.ingest_timelines,
                &keep_turns,
                &keep_sections,
                window,
                target_bytes,
            );
            report.count += r.count;
            report.bytes += r.bytes;
        }
        report
    }
}
