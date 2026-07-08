use super::*;
use std::time::Instant;

/// Maximum decode steps per decode quantum (matches `CHUNK_SIZE`).
const DECODE_BUDGET: usize = 32;
/// Maximum prefill passes per prefill quantum.
const PREFILL_BUDGET: usize = 1;

impl Scheduler {
    /// Number of currently-active decode sequences (including summary probes).
    /// Used as the "is there decode work to run" guard.
    fn decode_width(&self) -> usize {
        self.active_decodes.values().filter(|s| !s.finished).count()
    }

    /// Active *foreground* decode sequences — excludes compression passes.
    ///
    /// Count the active foreground dialogue decodes that feed the
    /// prefill/decode flip heuristic. Compression half-passes ride
    /// `active_decodes` and the decode wave like any decode, but are excluded
    /// here so they never hold the loop in decode-first mode at the expense of
    /// dialogue prefills (they are off the critical path).
    fn foreground_decode_width(&self) -> usize {
        self.active_decodes
            .iter()
            .filter(|(_, s)| {
                !s.finished && !matches!(s.seal_action, super::SealAction::CompressionPass { .. })
            })
            .count()
    }

    /// Run decode steps until the budget is reached or decode is empty.
    ///
    /// We deliberately do **not** yield mid-quantum on width comparisons —
    /// the outer dispatcher already chose this phase to run, and the budget
    /// is what guarantees the other phase gets airtime.
    fn run_decode_until_budget(&mut self) {
        for _ in 0..DECODE_BUDGET {
            if self.decode_width() == 0 {
                // No live decode work, but there may be sequences inserted as
                // finished during the prefill phase (EOS on first token) that
                // the decode loop never had a chance to clean up.
                self.cleanup_finished();
                return;
            }
            // Inject any pending tool-call static runs (Layer 3) before the
            // decode forward, so a `Static` run costs one prefill rather than N
            // decode steps.  No-op when no sequence has an active stencil.
            self.inject_stencil_prefills();
            self.batch_decode_step();
            // Drain any continuous-re-projection swaps queued during the
            // batch.  Must run BEFORE cleanup_finished so a swap that
            // re-keys an active_decodes entry doesn't race with finalize.
            // Timed separately (a sub-slice of the decode quantum) because the
            // provenance scan + glue gap-fill here is a prime "grows over time" suspect.
            let t_reproj = Instant::now();
            self.drain_pending_reprojections();
            self.wave_stats
                .add_phase(WavePhase::Reproject, t_reproj.elapsed().as_millis() as u64);
            self.cleanup_finished();
        }
    }

    /// Run prefill passes until the budget is reached or prefill is empty.
    fn run_prefill_until_budget(&mut self) {
        for _ in 0..PREFILL_BUDGET {
            if self.prefill_width() == 0 {
                return;
            }
            self.run_one_prefill_pass();
            self.promote_finished_prefills_to_decodes();
        }
    }

    /// Run section ingest chunks until the budget is reached or all ingests
    /// are done. Uses the same PREFILL_BUDGET constant so section ingests and
    /// turn prefills each get one chunk per loop iteration.
    fn run_section_ingest_until_budget(&mut self) {
        for _ in 0..PREFILL_BUDGET {
            if self.section_ingest_width() == 0 {
                return;
            }
            self.run_one_section_ingest_chunk();
            self.finalize_done_section_ingests();
        }
    }

    /// Main scheduler loop. Runs on the scheduler thread until shutdown.
    ///
    /// Each iteration runs one prefill quantum (PREFILL_BUDGET passes) and
    /// one decode quantum (DECODE_BUDGET steps). The wider phase runs first
    /// so freshly-arrived prefills don't have to wait an entire decode
    /// budget when there's no decode work yet.
    pub fn run(&mut self) {
        tracing::info!("scheduler started");

        loop {
            // 1. Drain pending submissions (non-blocking). This synchronously
            // handles SubmitTurn (projection + elevate + apply_segments gap-fill
            // + view create) on the scheduler thread — a prime suspect for the
            // wall-clock that is NOT a forward, so time it.
            let t_drain = Instant::now();
            let cont = self.drain_submissions();
            self.wave_stats
                .add_phase(WavePhase::Drain, t_drain.elapsed().as_millis() as u64);
            if !cont {
                break; // Shutdown requested or channel closed.
            }

            // 2. Promote queued PrefillWork → ActivePrefill (up to cap).
            let t_promote = Instant::now();
            self.promote_new_prefills();
            self.wave_stats
                .add_phase(WavePhase::Promote, t_promote.elapsed().as_millis() as u64);

            // 3. If idle, block waiting for work.
            if self.active_decodes.is_empty()
                && self.active_prefills.is_empty()
                && self.prefill_queue.is_empty()
                && self.active_section_ingests.is_empty()
            {
                match self.rx.recv() {
                    Ok(req) => {
                        if !self.handle_request(req) {
                            break;
                        }
                    }
                    Err(_) => break, // Engine dropped.
                }
                continue;
            }

            // 4. Always run all quanta each iteration; order by current width.
            // Summary decodes are excluded from the flip count so they never
            // hold the loop in decode-first mode at the expense of prefills.
            let dw = self.foreground_decode_width();
            let pw = self.prefill_width();
            let sw = self.section_ingest_width();
            if dw >= pw.max(sw) {
                self.timed_decode();
                self.timed_prefill();
                self.timed_section();
            } else if sw >= pw {
                self.timed_section();
                self.timed_prefill();
                self.timed_decode();
            } else {
                self.timed_prefill();
                self.timed_section();
                self.timed_decode();
            }

            // Flush the wave summary + phase breakdown if its 2 s window
            // elapsed — even when no forward ran this iteration, so stalls still
            // surface their phase split. (The expert-DMA delta and cumulative
            // op-profile dumps were removed once measurement ruled the expert
            // path out — dma_loads stays 0; the prefill cost is the attention
            // kernel, seen in the per-forward `code-read prefill` breakdown.)
            if self.wave_stats.due() {
                // Our eviction gate's own view of VRAM: the pool budget we
                // defend (vram_budget_available) and pool_used — queried only
                // on the wave we emit, not every iteration.
                let kv_vram = self
                    .session
                    .vram_budget_available()
                    .zip(self.session.vram_pool_stats())
                    .map(|(budget, (used, _reserved))| (budget, used));
                self.wave_stats.flush(kv_vram);
            }
        }

        tracing::info!("scheduler shut down");
    }

    /// Run the decode quantum, attributing its wall-clock to [`WavePhase::Decode`]
    /// (the reprojection drain inside it is separately attributed to
    /// [`WavePhase::Reproject`] as a sub-slice).
    fn timed_decode(&mut self) {
        let t = Instant::now();
        self.run_decode_until_budget();
        self.wave_stats
            .add_phase(WavePhase::Decode, t.elapsed().as_millis() as u64);
    }

    /// Run the prefill quantum, attributing its wall-clock to [`WavePhase::Prefill`].
    fn timed_prefill(&mut self) {
        let t = Instant::now();
        self.run_prefill_until_budget();
        self.wave_stats
            .add_phase(WavePhase::Prefill, t.elapsed().as_millis() as u64);
    }

    /// Run the section-ingest quantum, attributing its wall-clock to
    /// [`WavePhase::Section`].
    fn timed_section(&mut self) {
        let t = Instant::now();
        self.run_section_ingest_until_budget();
        self.wave_stats
            .add_phase(WavePhase::Section, t.elapsed().as_millis() as u64);
    }
}
