//! Holding the mind for the runtime and reacting when it arrives or changes.
//!
//! The mind is read through the runtime, and a mind pending at start is handed to
//! every reader once it arrives, before the follower's first round so that nothing
//! reads it twice. A change to the mind's schema asks for a relaunch, because the
//! grammar is built from the schema once. A change to documents is reconciled into
//! the substrate at the next start. Each handle the runtime needs is installed once
//! at startup, and installing a second cast is refused, so there is only ever one
//! writable handle to the log.

use super::*;

impl Runtime {
    /// The mind's files as they stand now, or `None` with no mind.
    pub fn mind_files(&self) -> Option<Arc<MindFiles>> {
        self.mind_handle.files()
    }

    /// Restart the process, when the load has said how: the mind's schema
    /// moved, and the grammar is built from it once.
    pub(super) fn ask_relaunch(&self) {
        match self.relaunch.get() {
            Some(stopping) => {
                stopping.ask(Ending::Relaunch);
            }
            None => tracing::warn!("mind: a restart is needed to take the new schema"),
        }
    }

    /// Supply what reads the authored state again when a mind pending at
    /// start arrives. Called once at startup, before the load runs.
    pub fn set_mind_arrived(&self, hook: MindArrived) {
        *self.mind_arrived.write().unwrap() = Some(hook);
    }

    /// The mind pending at start arrived as `files`: every reader moves onto
    /// it and the benches are pointed at it. Before the mind is followed, so
    /// the follower's first round finds it already here.
    pub(super) fn arrive(&self, files: Arc<MindFiles>) {
        tracing::info!(commit = ?files.commit(), "mind: arrived — taken in without a restart");
        self.mind_handle.replace(files);
        self.hosted.set_bench_root(self.mind_handle.clone());
    }

    /// The authored state read again from the mind that arrived — what that
    /// gives back, when there is a hook to ask. On the loader thread, which
    /// drives the hook itself, as it drives the ingest. Taken, not cloned: it
    /// runs once, and holding it on would keep the authored state it closes
    /// over alive with the runtime.
    pub(super) fn read_again(&self) -> Option<anyhow::Result<Arrived>> {
        let hook = self.mind_arrived.write().unwrap().take()?;
        Some(block_on(hook()))
    }

    /// What the mind's move from `old` to `new` changed, reported as the folder
    /// watcher reports an edit: the documents added, changed and removed. They
    /// are read into the substrate at the next start, as a watched folder's
    /// edits are (`engine::watcher`).
    pub(super) fn reconcile_mind(&self, old: &MindFiles, new: &MindFiles) {
        let changes = old.diff(new);
        let ingestible =
            |paths: &[String]| paths.iter().filter(|p| is_ingestible(Path::new(p))).count();
        tracing::info!(
            "mind reloaded: +{} ~{} -{}",
            ingestible(&changes.added),
            ingestible(&changes.changed),
            ingestible(&changes.removed)
        );
    }

    /// The async runtime character tasks and world work run on; `None` in a
    /// test that stood the runtime up outside one.
    pub(crate) fn spawner(&self) -> Option<Handle> {
        self.spawner.clone()
    }

    /// The daemon's substrate, once it has been handed one — see
    /// [`Self::set_substrate`]. A runtime stood up without one keeps its
    /// worlds in memory alone.
    pub(crate) fn substrate(&self) -> Option<SharedSubstrate> {
        self.substrate.read().unwrap().clone()
    }

    /// Supply the already-open substrate for the engine to adopt. Called once
    /// at startup, before [`start`], with the handle
    /// [`crate::npcs::Npcs::load`] opened.
    ///
    /// **Not optional in practice.** Without it the engine opens `--data`
    /// itself, and this daemon then holds two writable handles to one log —
    /// which loses characters rather than failing. [`load`] refuses to build an
    /// engine when this is unset, so a future wiring mistake is a startup error
    /// instead of a slow leak.
    pub fn set_substrate(&self, shared: SharedSubstrate) {
        self.ops.set_substrate(shared.clone());
        *self.substrate.write().unwrap() = Some(shared);
    }

    /// Adopt the shared cast. Called once at startup with the very handle
    /// [`crate::api::Authored`] holds — never a second `Npcs` — so the effector
    /// device's `/self` reads and its agency writes go through the one cast and
    /// the one writable substrate handle behind it (effector design §9.2,
    /// Appendix E "The bridge").
    ///
    /// Once: a second cast would be the second writable handle this exists to
    /// rule out, so installing one over another panics.
    ///
    /// The scheduler takes the cast's suspended characters from it here,
    /// before any character is woken, so one suspended before a restart is
    /// suspended on its first moment after it.
    pub fn set_npcs(&self, npcs: Arc<Npcs>) {
        assert!(
            self.npcs.set(npcs).is_ok(),
            "the cast is installed once, at startup"
        );
        self.follow_suspended();
    }

    /// Bring the scheduler's suspended characters to what the cast's records
    /// say — after this member writes a suspension or a state edit, and after
    /// the cast takes in records written on other members, so every member
    /// holds the same characters still.
    ///
    /// A character suspended here also lets its conversation go
    /// ([`Minds::let_go`]): its engine slot and resident K/V are freed for
    /// other work, and the turn it takes once resumed rejoins the conversation
    /// from the log. Kept open, a suspended cast would take no turns and still
    /// hold the card.
    pub fn follow_suspended(&self) {
        let Some(npcs) = self.npcs() else {
            return;
        };
        let minds = self.minds.read().unwrap().clone();
        for (npc, suspended) in self.scheduler.follow_suspended(npcs.view().suspended()) {
            if suspended {
                let freed = minds.as_ref().is_some_and(|m| m.let_go(npc));
                tracing::info!(
                    npc,
                    freed,
                    "character suspended: it takes no turn until it is resumed"
                );
            } else {
                tracing::info!(npc, "character resumed: it thinks again");
            }
        }
    }

    /// Whether `npc`'s own record says it is suspended — asked as each turn
    /// is about to begin, so no turn begins on the scheduler's copy of the
    /// cast's suspensions alone. A character the record suspends that the copy
    /// had missed is suspended there now, and its conversation let go, as
    /// [`Self::follow_suspended`] would have done.
    pub fn held_by_record(&self, npc: u64) -> bool {
        let held = self
            .npcs()
            .is_some_and(|n| n.view().payload(npc).is_some_and(|p| p.state == SUSPENDED));
        if held && self.scheduler.suspend(npc, true) {
            let minds = self.minds.read().unwrap().clone();
            let freed = minds.as_ref().is_some_and(|m| m.let_go(npc));
            tracing::warn!(
                npc,
                freed,
                "character suspended in its record but not held: held at its turn"
            );
        }
        held
    }

    /// The bodies in world `world` whose characters are suspended.
    pub fn suspended_bodies(&self, world: &str) -> BTreeSet<String> {
        self.scheduler
            .suspended()
            .into_iter()
            .filter_map(|npc| self.bodies.bound(npc))
            .filter(|at| at.world == world)
            .map(|at| at.body)
            .collect()
    }

    /// The shared cast, for the effector bridge. `None` on a test path that
    /// stood the runtime up without one — a caller degrades to an empty result
    /// rather than failing the read.
    pub fn npcs(&self) -> Option<Arc<Npcs>> {
        self.npcs.get().cloned()
    }

    /// Whether the engine is loaded and the cast is thinking.
    pub fn is_ready(&self) -> bool {
        self.progress.is_ready()
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    /// Latch the shutdown flag: the metronome callbacks stop moving worlds,
    /// and a supervisor whose attempt dies during shutdown stops restarting.
    /// Cancelling and awaiting the character tasks themselves is
    /// [`Runtime::stop_characters`]'s job, ordered after this.
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Release);
    }

    pub fn stopping(&self) -> bool {
        self.stopping.load(Ordering::Acquire)
    }

    /// Shut the engine down, when it has been published: drain its persistence
    /// thread and writer, detach this substrate — and stop the scheduler when
    /// this is the engine's last attachment — then release the substrate's
    /// writer lock. Blocking: it joins threads. Run after
    /// [`Runtime::stop_characters`], so nothing of the cast still submits.
    pub fn shutdown_engine(&self) -> anyhow::Result<()> {
        let minds = self.minds.read().unwrap().clone();
        let Some(minds) = minds else {
            return Ok(());
        };
        minds
            .engine()
            .shutdown()
            .map_err(|e| anyhow::anyhow!("engine shutdown: {e}"))
    }
}
