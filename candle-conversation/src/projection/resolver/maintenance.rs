//! Runs the redo log's durability, compaction and segment maintenance for a
//! workspace.
//!
//! Group commit flushes staged records, and shutdown drains the log's owner first,
//! so an in-flight turn survives a crash and a process exit does not drop queued
//! writes. Compaction runs in steps, so its slow relocation I/O never makes a
//! decode wait, and it yields to a flush that is waiting. Maintenance holds off
//! while replication applies to the same log, since every apply waits on the same
//! owner.

use super::*;

impl Conversation {
    /// Durably flush the persistence redo log — the group-commit point.
    /// `fsync`s every staged record so an in-flight turn survives a crash.
    pub fn commit_persistence(&self) -> candle::Result<()> {
        self.persistence
            .call(|p| p.commit())
            .map_err(|e| candle::Error::Msg(format!("persist commit: {e}")))
    }

    /// Drain the log's owner: append every queued durable write (warm→cold
    /// KV, tokens, sigs, stream-decls) and commit — the point a shutdown waits
    /// for, after every enqueuer (the scheduler, the persistence thread's final
    /// drain, the summariser) has stopped. Without it, the daemon's
    /// `process::exit` would drop everything still queued (the whole point of
    /// the drain). The owner keeps serving calls afterward.
    pub fn flush_writer(&self) {
        self.persistence.wait_written(self.persistence.ticket());
        if let Err(e) = self.commit_persistence() {
            tracing::error!("persist commit at the writer flush failed: {e}");
        }
    }

    /// Give the substrate directory up to the next writer
    /// ([`SubstratePersistence::release_writer`]): commit, refuse every write
    /// from here on, and drop the directory's lock while this conversation is
    /// still alive. The last step of an engine's shutdown, after
    /// [`Self::flush_writer`].
    pub fn release_writer(&self) -> candle::Result<()> {
        self.persistence
            .call(|p| p.release_writer())
            .map_err(|e| candle::Error::Msg(format!("persist release: {e}")))
    }

    /// Like [`Self::commit_persistence`] but a no-op when nothing is
    /// staged. Returns `Ok(true)` when an `fsync` actually happened.
    /// Used by the daemon's 5-second flush task so a quiescent
    /// workspace doesn't issue pointless syscalls.
    pub fn commit_persistence_if_pending(&self) -> candle::Result<bool> {
        self.persistence
            .call(|p| p.commit_if_pending())
            .map_err(|e| candle::Error::Msg(format!("persist commit_if_pending: {e}")))
    }

    /// Force a full redo-log compaction — the whole-file dead-record rewrite
    /// (§5.8). Ignores the dead-ratio threshold: the operator opted in
    /// explicitly via the daemon's startup flag. `progress` reports coarse
    /// phase progress (0..=5) for the loading screen. A read-only substrate
    /// is never compacted — the store belongs to the process that writes it —
    /// so this returns without touching it.
    pub fn compact_substrate(
        &self,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> candle::Result<()> {
        if self.read_only {
            return Ok(());
        }
        // Behind every write already queued, so the rewrite carries every
        // record whose in-memory change came before it.
        self.persistence
            .call_after_writes(|p| {
                p.commit()?;
                p.compact_shared(&self.inner, progress)
            })
            .map_err(|e| candle::Error::Msg(format!("persist compaction: {e}")))
    }

    /// Replication is busy applying to this substrate, or not: segment
    /// maintenance defers to it (`persistence::replication_load`). Said once a
    /// second by the member replicating it.
    pub fn note_replication_busy(&self, busy: bool) {
        self.persistence.replication_load().note_now(busy);
    }

    /// Advance the persistence thread's **background maintenance** on the
    /// segmented redo log — drop a fully-dead segment, compact a mostly-dead
    /// one, or combine two small adjacent ones
    /// (`docs/archived/segmented_substrate_log.md` §6). Continues the op in
    /// `op` when there is one; otherwise, when `start`, plans one past the age
    /// and dead-ratio gates (a cheap per-segment liveness scan on the common
    /// no-op path). Steps aside between batches when `yield_now` says a flush
    /// is waiting, leaving the op in `op` for the next pass
    /// ([`MaintenanceOp`]).
    ///
    /// Returns `true` when an op completed.
    pub fn step_background_maintenance(
        &self,
        op: &mut Option<MaintenanceOp>,
        start: bool,
        yield_now: &dyn Fn() -> bool,
    ) -> candle::Result<bool> {
        // While replication applies here, no op starts and one running steps
        // aside after its next batch: every apply waits on the same owner
        // (`persistence::replication_load`, bounded by its `MAX_DEFER`).
        let replication = self.persistence.replication_load();
        let deferring = replication.defers_now();
        if op.is_none() && start && !deferring {
            *op = self.begin_maintenance_op(Pass::Gated)?;
        }
        let Some(running) = op.as_mut() else {
            return Ok(false);
        };
        let yield_now = || yield_now() || replication.defers_now();
        match self.advance_maintenance(running, &yield_now) {
            Ok(false) => Ok(false),
            done => {
                *op = None;
                done
            }
        }
    }

    /// Run one maintenance op that drops a fully-dead segment, and nothing
    /// that relocates: what a member whose volume is full may run
    /// (`docs/replicated_substrate.md` §7.7). Returns whether one ran.
    pub fn drop_dead_segment(&self) -> candle::Result<bool> {
        self.run_maintenance_pass(Pass::DropOnly)
    }

    /// Force one maintenance op **now**, waiving the age/ratio gates — the manual
    /// `POST /v1/substrate/zend/maintenance` trigger. Seals the active segment first, so a
    /// conversation just archived/distilled this session (whose now-dead records
    /// sit in the active, which maintenance never touches) moves into a sealed
    /// segment and gets compacted away. Runs in the same steps, so decode keeps
    /// running through the relocation. Returns whether an op ran.
    pub fn force_compact_persistence(&self) -> candle::Result<bool> {
        self.persistence.call(|p| {
            p.commit()
                .map_err(|e| candle::Error::Msg(format!("substrate maintenance commit: {e}")))?;
            p.seal_active()
                .map_err(|e| candle::Error::Msg(format!("substrate maintenance seal: {e}")))
        })?;
        self.run_maintenance_pass(Pass::Forced)
    }

    /// Run one step of a maintenance op on the log's owner as background work,
    /// adding the time the step took to `held`. A step that fails ends the
    /// op's relocation watch, since no later step of it will run.
    ///
    /// Background, so every call and write waiting on the owner — a cold load,
    /// a seal write — goes ahead of the next step.
    pub(super) fn maintenance_hold<T: Send>(
        &self,
        held: &mut Duration,
        step: impl FnOnce(&mut SubstratePersistence) -> PersistenceResult<T> + Send,
    ) -> candle::Result<T> {
        let (out, took) = self.persistence.call_behind(|p| {
            let t = Instant::now();
            let out = step(p);
            if out.is_err() {
                p.end_relocation_watch();
            }
            (out, t.elapsed())
        });
        *held += took;
        out.map_err(|e| candle::Error::Msg(format!("substrate maintenance exec: {e}")))
    }

    /// One maintenance op run to the end in steps, so the slow relocation I/O
    /// never makes a decode wait — see [`MaintenanceOp`]. A forced pass waives the
    /// age/ratio gates (`pick_maintenance_op`); a drop-only pass relocates
    /// nothing (`pick_drop`). Returns whether an op ran.
    pub(super) fn run_maintenance_pass(&self, pass: Pass) -> candle::Result<bool> {
        let Some(mut op) = self.begin_maintenance_op(pass)? else {
            return Ok(false);
        };
        self.advance_maintenance(&mut op, &|| false)
    }

    /// The segmented redo log's maintenance state for the daemon status / GUI
    /// compaction indicator: `(segment_count, last_op, running)` where `last_op`
    /// is `(label, unix_secs)` of the most recent drop/compact/combine this
    /// session (or `None`) and `running` is `true` while an op's I/O is in
    /// flight. One atomic load, so `/v1/status` never waits on the log's owner.
    pub fn maintenance_status(&self) -> MaintenanceStatus {
        MaintenanceStatus::clone(&self.maintenance.load())
    }

    /// The redo log's dead-byte ratio as the last liveness scan found it — the
    /// measure the auto-compaction trigger and maintenance's plan read — for a
    /// viewer. Reads what the scan left rather than scanning again: the scan
    /// walks the whole substrate, up to 0.86 s on BlackWorld. `None` before the
    /// first scan.
    pub fn dead_ratio(&self) -> Option<f32> {
        self.persistence.facts().dead_ratio()
    }

    /// Run `f` against the persistence layer's current manifest snapshot, on
    /// the log's owner. Read-only accessor for callers that need to inspect
    /// the redo log's stream/chunk locations (sizes, formats, offsets) without
    /// rebuilding the whole substrate boundary.
    pub fn with_persistence_manifest<R: Send>(
        &self,
        f: impl FnOnce(&crate::persistence::manifest::Manifest) -> R + Send,
    ) -> R {
        self.persistence.call(|p| f(p.manifest()))
    }
}
