use super::*;

/// Maximum decode steps per decode quantum (matches `CHUNK_SIZE`).
const DECODE_BUDGET: usize = 32;
/// Maximum prefill chunks per prefill quantum.
const PREFILL_BUDGET: usize = 1;

impl Scheduler {
    /// Number of currently-active decode sequences.
    fn decode_width(&self) -> usize {
        self.active_decodes.values().filter(|s| !s.finished).count()
    }

    /// Run decode steps until the budget is reached or decode is empty.
    ///
    /// We deliberately do **not** yield mid-quantum on width comparisons —
    /// the outer dispatcher already chose this phase to run, and the budget
    /// is what guarantees the other phase gets airtime.
    fn run_decode_until_budget(&mut self) {
        for _ in 0..DECODE_BUDGET {
            if self.decode_width() == 0 {
                return;
            }
            self.batch_decode_step();
            // Drain any continuous-re-projection swaps queued during the
            // batch.  Must run BEFORE cleanup_finished so a swap that
            // re-keys an active_decodes entry doesn't race with finalize.
            self.drain_pending_reprojections();
            self.cleanup_finished();
        }
    }

    /// Run prefill chunks until the budget is reached or prefill is empty.
    fn run_prefill_until_budget(&mut self) {
        for _ in 0..PREFILL_BUDGET {
            if self.prefill_width() == 0 {
                return;
            }
            self.run_one_prefill_chunk();
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
    /// Each iteration runs one prefill quantum (PREFILL_BUDGET chunks) and
    /// one decode quantum (DECODE_BUDGET steps). The wider phase runs first
    /// so freshly-arrived prefills don't have to wait an entire decode
    /// budget when there's no decode work yet.
    pub fn run(&mut self) {
        tracing::info!("scheduler started");

        loop {
            // 1. Drain pending submissions (non-blocking).
            if !self.drain_submissions() {
                break; // Shutdown requested or channel closed.
            }

            // 2. Promote queued PrefillWork → ActivePrefill (up to cap).
            self.promote_new_prefills();

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
            let dw = self.decode_width();
            let pw = self.prefill_width();
            let sw = self.section_ingest_width();
            if dw >= pw.max(sw) {
                self.run_decode_until_budget();
                self.run_prefill_until_budget();
                self.run_section_ingest_until_budget();
            } else if sw >= pw {
                self.run_section_ingest_until_budget();
                self.run_prefill_until_budget();
                self.run_decode_until_budget();
            } else {
                self.run_prefill_until_budget();
                self.run_section_ingest_until_budget();
                self.run_decode_until_budget();
            }
        }

        tracing::info!("scheduler shut down");
    }
}
