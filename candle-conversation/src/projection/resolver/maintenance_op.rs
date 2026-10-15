//! A segment-maintenance op run in steps, so the persistence thread can serve a
//! flush between them.
//!
//! An op relocates one batch at a time under its own hold of the persistence
//! lock, so a seal write or a cold load waits for at most one batch. A flush is
//! different: it is work for the persistence thread itself — the hot→warm
//! migration that lets relief evict — and the thread that would do it is the
//! one running the op. Run to the end in one call, an op made relief's flush
//! wait out every batch of it: a 2.1 s compaction of npcd's store was a 2.4 s
//! relief wait in the middle of a zend prefill, all of it on the time to the
//! reply's first token.
//!
//! So the background op is a value the thread holds across passes
//! ([`MaintenanceOp`]). Each pass advances it ([`Conversation::advance_maintenance`])
//! and steps aside between batches the moment a flush is waiting; the next pass
//! serves the flush in its hot→warm phase and then picks the op up where it
//! stopped. The relocation watch the op started at its plan already covers
//! every write made while it is paused — it exists for the writes that land
//! between batches, and a paused op is that, for longer.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::Conversation;
use crate::persistence::maintenance::{MaintenancePlan, MaintenanceRun, RELOCATION_BATCH_BYTES};
use crate::scheduler::note_persistence_maint_us;

/// Which maintenance op a pass may pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pass {
    /// The background pass: only past the age and dead-ratio gates.
    Gated,
    /// The manual trigger: the gates waived.
    Forced,
    /// A full volume's pass: dropping a dead segment, nothing that relocates.
    DropOnly,
}

/// Which batches an op has still to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// The resident re-emission and the targets' chunks.
    Relocate,
    /// Tokens, snapshots, branch checkpoints, singletons, characters, file
    /// events and blobs.
    Tail,
}

/// A planned op and how far it has got: its plan, its run, and the timings its
/// closing log line reports.
pub struct MaintenanceOp {
    plan: MaintenancePlan,
    run: MaintenanceRun,
    stage: Stage,
    dir: PathBuf,
    plan_ms: u64,
    started: Instant,
    held: Duration,
    batches: u32,
    longest: Duration,
}

impl Conversation {
    /// Plan an op of the kind `pass` allows and start it: the plan under a
    /// brief read + persistence lock, with the relocation watch started in the
    /// same hold so every write after the plan is known to the relocation.
    /// `None` when there is nothing to do, or an op is already in flight — an
    /// op releases the lock between its batches, so a second pass (the forced
    /// `POST /v1/substrate/zend/maintenance` beside the background thread's)
    /// can arrive mid-op. It plans nothing and leaves the status alone: the op
    /// in flight owns `running`.
    pub(super) fn begin_maintenance_op(&self, pass: Pass) -> candle::Result<Option<MaintenanceOp>> {
        let t_plan = Instant::now();
        let (plan, dir) = {
            let mut p = self.persistence.lock_behind_waiters().unwrap();
            if p.relocation_in_flight() {
                tracing::debug!(
                    target: "candle_conversation::persistence::maintenance",
                    "a maintenance op is already relocating — this pass plans none"
                );
                return Ok(None);
            }
            let substrate = self.read();
            let plan = match pass {
                Pass::Gated => p.plan_maintenance(&substrate, false),
                Pass::Forced => p.plan_maintenance(&substrate, true),
                Pass::DropOnly => p.plan_drop(&substrate),
            }
            .map_err(|e| candle::Error::Msg(format!("substrate maintenance plan: {e}")))?;
            if plan.is_some() {
                p.watch_relocation();
            }
            (plan, p.dir().to_path_buf())
        };
        let Some(plan) = plan else {
            self.refresh_maintenance_status();
            return Ok(None);
        };
        let plan_ms = t_plan.elapsed().as_millis() as u64;
        // "In progress" for the GUI's spinner, across every pass the op spans.
        self.maintenance.lock().unwrap().2 = true;
        let mut held = Duration::ZERO;
        let run = match self.maintenance_hold(&mut held, |p| p.begin_maintenance(&plan)) {
            Ok(run) => run,
            Err(e) => {
                self.refresh_maintenance_status();
                return Err(e);
            }
        };
        Ok(Some(MaintenanceOp {
            plan,
            run,
            stage: Stage::Relocate,
            dir,
            plan_ms,
            started: Instant::now(),
            held,
            batches: 0,
            longest: Duration::ZERO,
        }))
    }

    /// Run `op`'s batches until it is done or `yield_now` says a flush is
    /// waiting — at least one batch per call, so an op is never starved by a
    /// stream of flushes. Answers whether the op completed: relocated, applied
    /// to the index and its sources unlinked. An op that errs has ended (each
    /// step ends the relocation watch on failure) and is not run again.
    ///
    /// Relocation runs under the persistence lock only — the substrate lock is
    /// released, so decode's in-RAM projection proceeds during the read,
    /// re-append and fsync — and the lock is taken per batch: the resident
    /// re-emit, the chunks, and the tail, each `RELOCATION_BATCH_BYTES` at a
    /// time. Held across a whole phase it stalled what waited for it: a 3.6 GB
    /// segment's chunks stalled an elevate — and the decode behind it — for
    /// 37 s, and a ~35,700-record re-emit held every finishing turn's token
    /// write for 44–52 s. Only the time the lock is actually held is charged
    /// to the scheduler's Sync bucket.
    pub fn advance_maintenance(
        &self,
        op: &mut MaintenanceOp,
        yield_now: &dyn Fn() -> bool,
    ) -> candle::Result<bool> {
        let advanced = self.relocate_until(op, yield_now);
        match advanced {
            Ok(false) => Ok(false),
            Ok(true) => {
                let finished = self.complete_maintenance_op(op);
                self.refresh_maintenance_status();
                finished.map(|()| true)
            }
            Err(e) => {
                self.refresh_maintenance_status();
                Err(e)
            }
        }
    }

    /// Let `op` go without finishing it: the relocation watch ends and the
    /// status clears. What it relocated stays, as copies a later op reclaims;
    /// its sources were never unlinked, so nothing it planned is lost. For a
    /// thread shutting down, whose drain must not queue behind the op.
    pub fn abandon_maintenance(&self, op: MaintenanceOp) {
        drop(op);
        self.persistence.lock().unwrap().end_relocation_watch();
        self.refresh_maintenance_status();
    }

    /// The batches of [`Self::advance_maintenance`]: `true` once relocation is
    /// through, `false` when it stepped aside with batches still to run.
    fn relocate_until(
        &self,
        op: &mut MaintenanceOp,
        yield_now: &dyn Fn() -> bool,
    ) -> candle::Result<bool> {
        loop {
            let before = op.held;
            let more = match op.stage {
                Stage::Relocate => self.maintenance_hold(&mut op.held, |p| {
                    p.relocate_batch(&op.plan, &mut op.run, RELOCATION_BATCH_BYTES)
                })?,
                Stage::Tail => self.maintenance_hold(&mut op.held, |p| {
                    p.relocate_tail_batch(&op.plan, &mut op.run, RELOCATION_BATCH_BYTES)
                })?,
            };
            op.batches += 1;
            op.longest = op.longest.max(op.held - before);
            if !more {
                match op.stage {
                    Stage::Relocate => op.stage = Stage::Tail,
                    Stage::Tail => return Ok(true),
                }
            }
            if yield_now() {
                return Ok(false);
            }
        }
    }

    /// Repoint the index at the relocated records under a brief write lock,
    /// then unlink the drained sources under the persistence lock, and log the
    /// op's timings.
    fn complete_maintenance_op(&self, op: &mut MaintenanceOp) -> candle::Result<()> {
        let result = op.run.take_result();
        note_persistence_maint_us(op.held.as_micros() as u64);
        let exec_ms = op.started.elapsed().as_millis() as u64;
        let t_apply = Instant::now();
        {
            let mut substrate = self.write();
            result.apply_to_substrate(&mut substrate);
        }
        let apply_ms = t_apply.elapsed().as_millis() as u64;
        let t_finish = Instant::now();
        {
            let mut p = self.persistence.lock_behind_waiters().unwrap();
            p.finish_maintenance(&op.plan)
                .map_err(|e| candle::Error::Msg(format!("substrate maintenance drop: {e}")))?;
        }
        let finish_ms = t_finish.elapsed().as_millis() as u64;
        // **Every phase's hold, not only the relocation's.** The plan holds the
        // persistence lock and a substrate read lock together; the apply holds
        // the substrate write lock; the finish holds the persistence lock across
        // the unlinks. Any of them stalls a seal behind it, and
        // `longest_hold_ms` covers only the relocation batches. At `info`, once
        // per op beside the plan's own line: an op that stalls the tenants
        // behind it is named by its slow phase here, and a process hosting
        // several substrates says whose it was. `exec_ms` runs from the op's
        // start to the end of its relocation, pauses for flushes included.
        tracing::info!(
            target: "candle_conversation::persistence::maintenance",
            dir = %op.dir.display(),
            plan_ms = op.plan_ms,
            exec_ms,
            held_ms = op.held.as_millis() as u64,
            batches = op.batches,
            longest_hold_ms = op.longest.as_millis() as u64,
            apply_ms,
            finish_ms,
            "segment-maintenance: plan under both locks, relocation one batch \
             at a time, apply under the substrate write lock, finish under the \
             persistence lock"
        );
        Ok(())
    }

    /// Refresh the status cache from the persistence layer — this also clears
    /// `running`. The persistence lock is held only for two O(1) reads (never
    /// across I/O); the status endpoint reads the separate `maintenance` lock,
    /// so it never blocks on a compaction.
    fn refresh_maintenance_status(&self) {
        let view = {
            let p = self.persistence.lock().unwrap();
            (p.segment_count(), p.last_maintenance(), false)
        };
        *self.maintenance.lock().unwrap() = view;
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::path::Path;

    use super::*;
    use crate::persistence::maintenance::RELOCATION_BATCH_RECORDS;
    use crate::persistence::{SubstratePersistence, SUBSTRATE_DIR};
    use crate::substrate::Substrate;

    /// Section tombstones enough for three resident batches.
    const TOMBSTONES: u64 = 2 * RELOCATION_BATCH_RECORDS as u64 + 5;

    fn sealed_segment_1(dir: &Path) -> PathBuf {
        dir.join(SUBSTRATE_DIR).join(format!("seg-{:010}.log", 1))
    }

    /// A store whose one sealed segment holds [`TOMBSTONES`] section
    /// tombstones and a superseded debug id — dead weight a forced op compacts.
    fn store() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut substrate = Substrate::new();
        let mut p =
            SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        for i in 0..TOMBSTONES {
            p.write_section_tombstone(0x5000 + i, None, None).unwrap();
        }
        p.write_debug_id(7, "first", p.mint()).unwrap();
        p.write_debug_id(7, "second", p.mint()).unwrap();
        p.commit().unwrap();
        p.seal_active().unwrap();
        dir
    }

    fn open(dir: &Path) -> Conversation {
        let mut substrate = Substrate::new();
        let p = SubstratePersistence::open_in_with_substrate(dir, &mut substrate).unwrap();
        Conversation::from_parts(substrate, p)
    }

    /// **A waiting flush steps the op aside after one batch, and the next pass
    /// finishes it.** Run to the end in one call, an op made relief's flush
    /// wait out every batch of it — 2.4 s of a prefill on the live daemon.
    #[test]
    fn a_waiting_flush_steps_the_op_aside_and_the_next_pass_finishes_it() {
        let dir = store();
        let conv = open(dir.path());
        let mut op = conv.begin_maintenance_op(Pass::Forced).unwrap();
        assert!(
            op.is_some(),
            "the superseded debug id is dead weight to compact"
        );
        assert!(conv.maintenance_status().2, "running from the plan on");

        let asked = Cell::new(0u32);
        let flush_waiting = || {
            asked.set(asked.get() + 1);
            true
        };
        assert!(!conv
            .step_background_maintenance(&mut op, false, &flush_waiting)
            .unwrap());
        assert_eq!(asked.get(), 1, "asked once, after its first batch");
        let paused = op.as_ref().expect("the op is kept for the next pass");
        assert_eq!(paused.batches, 1);
        assert!(
            sealed_segment_1(dir.path()).exists(),
            "nothing unlinked yet"
        );
        assert!(conv.maintenance_status().2, "still running while paused");

        // Another pass with a flush waiting runs one more batch, no more.
        assert!(!conv
            .step_background_maintenance(&mut op, false, &flush_waiting)
            .unwrap());
        assert_eq!(op.as_ref().unwrap().batches, 2);

        // A pass with nothing waiting finishes it.
        assert!(conv
            .step_background_maintenance(&mut op, false, &|| false)
            .unwrap());
        assert!(op.is_none());
        assert!(
            !sealed_segment_1(dir.path()).exists(),
            "segment 1 compacted away"
        );
        assert!(!conv.maintenance_status().2, "the status clears");
        drop(conv);

        let mut substrate = Substrate::new();
        let _p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        assert_eq!(
            substrate.ended_unstamped_sections().count() as u64,
            TOMBSTONES,
            "every section stays ended across the paused op and a reload"
        );
    }

    /// **An op let go keeps everything it planned.** Its sources were never
    /// unlinked, the watch ends, the status clears, and the next op plans.
    #[test]
    fn an_abandoned_op_loses_nothing_and_frees_the_next() {
        let dir = store();
        let conv = open(dir.path());
        let mut op = conv.begin_maintenance_op(Pass::Forced).unwrap();
        assert!(!conv
            .step_background_maintenance(&mut op, false, &|| true)
            .unwrap());
        assert!(
            conv.begin_maintenance_op(Pass::Forced).unwrap().is_none(),
            "one op at a time"
        );
        conv.abandon_maintenance(op.take().unwrap());
        assert!(sealed_segment_1(dir.path()).exists());
        assert!(!conv.maintenance_status().2);
        assert!(
            conv.begin_maintenance_op(Pass::Forced).unwrap().is_some(),
            "the watch ended, so a new op plans"
        );
    }
}
