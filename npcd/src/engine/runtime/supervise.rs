//! Supervising each character's loop as a task that restarts after a panic and
//! stops when cancelled.
//!
//! Each character's loop runs as its own task, so a thousand idle characters cost
//! little, and a panic in one attempt restarts only that character, after an
//! exponential backoff. The backoff stops a down engine from being hammered by
//! every character retrying at once. Cancelling a character aborts both the
//! supervisor and the attempt it is running, because aborting only the wrapper
//! would leave the attempt's loop holding its decode slot.

use super::*;

/// One supervised character: the supervisor wrapper's handle, and the abort
/// handle of whichever attempt is currently running the loop.
///
/// Both are needed to cancel, because aborting only the wrapper DETACHES the
/// attempt — a dropped [`tokio::task::JoinHandle`] lets its task run on — and
/// a detached loop keeps its decode's slot and races the character's
/// re-creation. That was a live bug: deletion looked instant while the old
/// loop thought on.
pub(super) struct CharacterTask {
    pub(super) supervisor: tokio::task::JoinHandle<()>,
    pub(super) attempt: Arc<Mutex<AttemptSlot>>,
}

/// The supervisor's current attempt, or the tombstone that tells a supervisor
/// racing [`CharacterTask::stop`] to put its fresh attempt down: the stop may
/// land between the attempt's spawn and its registration here, and without
/// the tombstone that attempt would be the same detached zombie.
pub(super) enum AttemptSlot {
    Live(Option<tokio::task::AbortHandle>),
    Stopped,
}

impl CharacterTask {
    /// Cancel the supervisor and whatever attempt it is running, and tombstone
    /// the slot so an attempt registered after this abort is cancelled too.
    pub(super) fn stop(&self) {
        self.supervisor.abort();
        let mut slot = self.attempt.lock().unwrap();
        if let AttemptSlot::Live(Some(handle)) = std::mem::replace(&mut *slot, AttemptSlot::Stopped)
        {
            handle.abort();
        }
    }
}

/// Spawn every woken character's task — the supervisor's initial roster.
///
/// One task per character, not one thread: a task parked in its wait costs a
/// few hundred bytes of state, so a thousand idle minds stay as cheap as the
/// scheduler's design promises.
///
/// Called once, at the end of the load, after the cast has been woken into the
/// scheduler; a character created later is spawned by the create path, and a
/// deleted one is stopped by the delete path.
pub fn supervise(rt: &Arc<Runtime>) {
    let ids = rt.scheduler.population_ids();
    let n = ids.len();
    for id in ids {
        rt.spawn_character(id);
    }
    tracing::info!("cast supervisor running — {n} character task(s) spawned");
}

/// The supervisor's restart pacing: how long a died character task waits
/// before its retry. Doubles from a floor to a ceiling, and a task that ran a
/// healthy stretch before dying starts from the floor again. The backoff
/// exists for the systemic case: an engine that is down fails every character
/// at once, and a hundred tasks retrying hot would be a stampede on a
/// scheduler that is trying to come back.
pub(super) struct Backoff {
    pub(super) next: Duration,
}

impl Backoff {
    pub(super) const FLOOR: Duration = Duration::from_secs(1);
    pub(super) const CEIL: Duration = Duration::from_secs(60);
    /// A task that has run this long without panicking is healthy, and its
    /// next failure starts the backoff from the floor again.
    pub(super) const HEALTHY_AFTER: Duration = Duration::from_secs(300);

    pub(super) fn new() -> Self {
        Self { next: Self::FLOOR }
    }

    /// The wait before this retry, given how long the attempt that just died
    /// had been running.
    pub(super) fn after(&mut self, ran_for: Duration) -> Duration {
        if ran_for >= Self::HEALTHY_AFTER {
            self.next = Self::FLOOR;
        }
        let wait = self.next;
        self.next = (self.next * 2).min(Self::CEIL);
        wait
    }
}

impl Runtime {
    /// Put one character's loop on the runtime, supervised: a panic restarts
    /// it with exponential backoff, a clean exit (retired, or the daemon
    /// stopping) ends it. Idempotent — a character whose task is already
    /// running keeps it.
    pub fn spawn_character(self: &Arc<Self>, npc_id: u64) {
        // A drone at rest is held before anything of its is scheduled: its
        // conversations answer nobody, and its loop takes no turn.
        self.settle_rest(npc_id);
        // What waited on its pair threads while it ran nowhere, or elsewhere:
        // a `Told` pulse is not kept, so nothing else would answer it.
        self.rescan_talks(npc_id);
        let Some(handle) = self.spawner.clone() else {
            // The unit tests exercise scheduling data without a runtime; the
            // daemon always has one.
            return;
        };
        let mut tasks = self.tasks.lock().unwrap();
        if tasks
            .get(&npc_id)
            .is_some_and(|t| !t.supervisor.is_finished())
        {
            return;
        }
        let rt = Arc::clone(self);
        let supervisor = handle.clone();
        let attempt_slot = Arc::new(Mutex::new(AttemptSlot::Live(None)));
        let slot = Arc::clone(&attempt_slot);
        let task = handle.spawn(async move {
            let mut backoff = Backoff::new();
            loop {
                let ran = Instant::now();
                // The loop's own panics are contained by the spawned attempt,
                // so one bad turn cannot take the supervisor down with it.
                let attempt = supervisor.spawn(character_loop(Arc::clone(&rt), npc_id));
                {
                    // Registered before this task's next await, so a stop that
                    // aborts the supervisor either finds the handle here or
                    // has already tombstoned the slot — in which case this
                    // attempt is the one it could not see, and dies here.
                    let mut slot = slot.lock().unwrap();
                    match &mut *slot {
                        AttemptSlot::Stopped => {
                            attempt.abort();
                            return;
                        }
                        AttemptSlot::Live(handle) => *handle = Some(attempt.abort_handle()),
                    }
                }
                match attempt.await {
                    Ok(()) => return,
                    Err(e) if e.is_cancelled() => return,
                    Err(e) => {
                        if rt.stopping() {
                            return;
                        }
                        let wait = backoff.after(ran.elapsed());
                        tracing::error!(
                            "npc {npc_id}: character task died ({e}) — restarting in {wait:?}"
                        );
                        tokio::time::sleep(wait).await;
                    }
                }
            }
        });
        tasks.insert(
            npc_id,
            CharacterTask {
                supervisor: task,
                attempt: attempt_slot,
            },
        );
    }

    /// Stop a character's task — the deletion path. Cancels the supervisor AND
    /// its current attempt (see [`CharacterTask`]); the abort lands at the
    /// attempt's next await, and a decode it was waiting on is dropped with
    /// it, which stops that decode ([`candle_conversation::TurnHandle`]'s
    /// contract).
    pub fn stop_character(&self, npc_id: u64) {
        if let Some(task) = self.tasks.lock().unwrap().remove(&npc_id) {
            task.stop();
        }
    }

    /// Shutdown's ordered half: cancel every character task — parked or
    /// mid-await — and await the supervisors out, so every dropped turn future
    /// has released its slot before whatever runs after this touches the
    /// engine. [`Runtime::stop`] should be called first so the metronome
    /// callbacks and restart guards read the latch.
    pub async fn stop_characters(&self) {
        let tasks = std::mem::take(&mut *self.tasks.lock().unwrap());
        for task in tasks.values() {
            task.stop();
        }
        for (_, task) in tasks {
            let _ = task.supervisor.await;
        }
    }
}
