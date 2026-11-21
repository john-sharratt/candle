//! Each character's wait and its tick: when it becomes due, and the three phases
//! of a turn around the model decode.
//!
//! The deadline is the truth and the waker is a doorbell, so a delivery, a pause
//! or a park takes effect at the next check rather than at the end of a stale
//! sleep. Perception lands in the window before the decode reads it, and the
//! acts land after. The decode runs with no lock held, so it never blocks other
//! characters' deliveries.

use super::*;

impl Scheduler {
    /// Sleep until this character is due to think — its own task's wait.
    ///
    /// **`due_at` is the truth; the waker is the doorbell.** The wait re-reads
    /// the deadline after every wake, so anything that moves it — a delivery
    /// pulling it to now, a pause pushing it out, a tick parking it at `NEVER`
    /// — takes effect at once rather than at the end of a stale sleep. A wake
    /// sent while the task is mid-think is a stored permit, read the moment the
    /// task comes back here.
    ///
    /// Returns `false` when the character has been retired: there is no inbox
    /// left to be due, and the task exits.
    ///
    /// A suspended character waits on its waker alone, whatever its deadline:
    /// it is not due until it is resumed, and resuming rings it.
    pub async fn wait_due(&self, npc_id: u64) -> bool {
        loop {
            let (due_at, waker, suspended) = {
                let inboxes = self.inboxes.lock().unwrap();
                let Some(inbox) = inboxes.get(&npc_id) else {
                    return false;
                };
                (inbox.due_at, Arc::clone(&inbox.waker), self.is_held(npc_id))
            };
            // Registered before the deadline check could race a notify: a
            // permit stored between the read above and this await completes
            // the wait immediately.
            let woken = waker.notified();
            if suspended {
                woken.await;
                continue;
            }
            let now = self.now_ms();
            if due_at <= now {
                return true;
            }
            if due_at == NEVER {
                woken.await;
            } else {
                let remaining = Duration::from_millis(due_at - now);
                tokio::select! {
                    _ = woken => {}
                    _ = tokio::time::sleep(remaining) => {}
                }
            }
        }
    }

    /// Every character due at `now_ms`, soonest deadline first — the
    /// synchronous read of the state each [`Self::wait_due`] sleeps on, for
    /// tests that drive the scheduler without standing the tasks up. A
    /// suspended character is never among them.
    pub fn due_now(&self, now_ms: u64) -> Vec<u64> {
        let inboxes = self.inboxes.lock().unwrap();
        let suspended = self.suspended.lock().unwrap();
        let resting = self.resting.lock().unwrap();
        let held = |id: &u64| suspended.contains(id) || resting.contains_key(id);
        let mut due: Vec<(u64, u64)> = inboxes
            .values()
            .filter(|i| i.due_at <= now_ms && !held(&i.npc_id))
            .map(|i| (i.due_at, i.npc_id))
            .collect();
        due.sort_unstable();
        due.into_iter().map(|(_, id)| id).collect()
    }

    /// Whether this one character is due at `now_ms` — [`Self::due_now`]
    /// narrowed to the inbox a test is watching.
    #[cfg(test)]
    pub(super) fn is_due(&self, npc_id: u64, now_ms: u64) -> bool {
        let inboxes = self.inboxes.lock().unwrap();
        let suspended = self.suspended.lock().unwrap();
        let resting = self.resting.lock().unwrap();
        inboxes.get(&npc_id).is_some_and(|i| {
            i.due_at <= now_ms && !suspended.contains(&npc_id) && !resting.contains_key(&npc_id)
        })
    }

    /// Run one character's tick.
    ///
    /// `act` is handed the drained events as prose and returns the acts the
    /// character took. It is a closure so the scheduler can be tested — and
    /// reasoned about — without a GPU: the scheduling *is* the thing under test
    /// here, and a decode inside it would make every test a model test.
    ///
    /// [`Self::begin_tick`] then [`Self::end_tick`], with the closure between:
    /// the live character task uses the two halves directly so it can *await*
    /// its decode there, and this composition is what keeps the tests and the
    /// task on one code path.
    pub fn tick<F>(&self, npc_id: u64, now_ms: u64, world_ms: u64, act: F) -> Option<TickRecord>
    where
        F: FnOnce(&[Event], &Window) -> Vec<String>,
    {
        let start = self.begin_tick(npc_id)?;
        let acts = act(&start.events, &start.window);
        self.end_tick(npc_id, now_ms, world_ms, start, acts)
    }

    /// The tick's first two phases: drain the inbox and land perception in the
    /// window, handing back the snapshot the decode reads. `None` when there is
    /// nothing to think about, or no such character — and going quiet is
    /// *recorded*, not merely returned (see the body).
    ///
    /// The decode between this and [`Self::end_tick`] runs under no lock here:
    /// it can take seconds, and holding the inbox map across it would block
    /// every other character's `deliver` for that whole time.
    ///
    /// `None`, with nothing drained, for a suspended character: one suspended
    /// after its task left [`Self::wait_due`] takes no turn, and its inbox
    /// stays whole for the turn it takes once resumed.
    pub fn begin_tick(&self, npc_id: u64) -> Option<TickStart> {
        if self.is_held(npc_id) {
            return None;
        }
        // The drain takes the lock and gives it straight back.
        let (events, cause, owed, replayed) = {
            let mut inboxes = self.inboxes.lock().unwrap();
            let inbox = inboxes.get_mut(&npc_id)?;
            // Readiness FIRST. `drain` clears the preempt flag, so reading it
            // afterwards reports `Blocked` for every tick and the Pulse feed
            // loses the one column that says why a character woke.
            let cause = inbox.readiness();
            let owed = std::mem::take(&mut inbox.owed);
            let replayed = std::mem::take(&mut inbox.replayed);
            (inbox.drain(), cause, owed, replayed)
        };

        /* **Nothing happened, so there is nothing to think about.**
         *
         * This used to synthesise a heartbeat event and think anyway, on the
         * reasoning that "nothing arrived" must not mean "dead forever". The
         * cost of that was not the decode — it was what the decode *wrote*.
         *
         * An embodied character is paced `WORKING`, which floors its heartbeat
         * at four seconds and never relaxes, so a character standing in a quiet
         * room produced an act every four seconds for as long as it stood
         * there. Each act is appended to its own window, and within a couple of
         * minutes the twenty-four-turn window held nothing but copies of the
         * character's last gesture. At that point the next token is certain: a
         * `top_p` nucleus of one, sampled identically whatever the RNG says.
         *
         * That is the collapse, and no sampler can undo it — a penalty of a
         * fifth of a logit against p ≈ 1 is nothing. The evidence has to not
         * pile up, and the way it does not pile up is that a character with
         * nothing to react to does not speak.
         *
         * Liveness is somebody else's question now. "Waiting on input" and
         * "dead" are told apart by asking the scheduler, not by spending a
         * decode a minute per character to have it say so in prose. */
        /* **And going quiet has to be *recorded*, not merely returned.**
         *
         * This returned straight out, leaving `due_at` at whatever made the
         * character due — which is `0`. That looks harmless and is a permanent
         * wedge, because `deliver` reads `due_at == 0` as "an entry for this
         * character is already standing in the heap, so do not push another".
         * The entry that made it due has just been consumed by the pop that led
         * here, so the character is left claiming to be queued while nothing
         * anywhere will ever run it: events accumulate, `preempted` stays set,
         * and the tick count never moves again.
         *
         * It is reachable whenever a tick ends `scheduled` with an empty queue
         * — `think_again` is the common way, since an act that answered sets
         * `due_at = 0` and the answer arrives in the outcome rather than as a
         * queued event. Measured live: three characters wedged this way inside
         * a minute of starting, with inboxes at sixty and climbing, and an
         * URGENT broadcast that preempts everything could not move them.
         *
         * Setting it back to `NEVER` is what the module already says an empty
         * queue means — unscheduled until the world says otherwise — and it is
         * what lets the next `deliver` see a character that needs queueing. */
        /* **Unless a turn is owed.** An act that answered left its answer
         * pending on the conversation, and the turn that reads it has nothing
         * in the queue by construction — see `Inbox::think_again`. */
        if events.is_empty() && !owed {
            let mut inboxes = self.inboxes.lock().unwrap();
            if let Some(inbox) = inboxes.get_mut(&npc_id) {
                inbox.due_at = NEVER;
            }
            return None;
        }

        // **Three phases, because the middle one is a model decode.**
        //
        // Perception lands in the window before the decode sees it, the decode reasons over
        // the window, and the acts land after — but only the first and third need the lock.
        // Holding it across `act` is what the note above says must not happen, and it did:
        // every `deliver`, `broadcast`, `window_of`, `census`, `wake` and `retire` blocked on
        // this `Mutex` for the length of a generation, and the HTTP ones do it from `async fn`s
        // that park a tokio worker while they wait.
        //
        // The decode reads a *snapshot*. Anything delivered while it runs lands in the real
        // window and is seen by the next tick, which is the same guarantee as an event that
        // arrived a moment after the lock was released.
        let perceived: Vec<String> = events.iter().map(|e| e.prose()).collect();
        let snapshot = {
            let mut inboxes = self.inboxes.lock().unwrap();
            let inbox = inboxes.get_mut(&npc_id)?;
            // A retried turn's events lead the queue and are in the window
            // already — see [`Scheduler::retry`].
            for e in events.iter().skip(replayed) {
                inbox.window.push_event(e);
            }
            inbox.window.clone()
        };

        Some(TickStart {
            events,
            perceived,
            cause,
            window: snapshot,
        })
    }

    /// The tick's third phase: land the acts, settle the next deadline, and
    /// record the row. `None` when the character was retired while the decode
    /// ran — its acts have nowhere to land, and nothing downstream wants a
    /// record for a character that is gone.
    pub fn end_tick(
        &self,
        npc_id: u64,
        now_ms: u64,
        world_ms: u64,
        start: TickStart,
        acts: Vec<String>,
    ) -> Option<TickRecord> {
        let TickStart {
            perceived, cause, ..
        } = start;
        let mut inboxes = self.inboxes.lock().unwrap();
        let inbox = inboxes.get_mut(&npc_id)?;
        for a in &acts {
            inbox.window.push_npc(a.clone(), world_ms);
        }
        // Only when it actually did something. A tick that decided on nothing is
        // a character that thought, and "how long since it last acted" has to
        // keep counting through those or it answers a different question.
        if !acts.is_empty() {
            inbox.last_act_at = Some(self.born.elapsed());
            inbox.last_act_ms = inbox.last_act_ms.max(world_ms);
        }

        inbox.ticks += 1;
        /* **What happens next is decided by the world, not by a timer.**
         *
         * A character is scheduled when something reaches it — see
         * [`Scheduler::deliver`] — and not otherwise. So the only thing to
         * settle here is whether anything arrived *during* the decode, which
         * takes seconds and is exactly when it is most likely to.
         *
         * Empty means unscheduled, and unscheduled means asleep until the world
         * says otherwise. That is what stops a quiet character writing its own
         * window full.
         *
         * **Unless this tick already decided.** The act loop runs inside the
         * closure above, so a `pause` or an act that answered has set the
         * deadline before this line is reached — and without the flag this
         * overwrites it with `NEVER`, on the empty queue both of those cases
         * have by definition. For a pause that is not "wait two minutes" but
         * "never think again"; for an answer it is a character holding what it
         * just read with no turn in which to use it. */
        match inbox.scheduled {
            true => inbox.scheduled = false,
            false => {
                inbox.due_at = match inbox.queue.iter().any(Event::wakes) {
                    false => NEVER,
                    true => 0,
                }
            }
        }
        let record = TickRecord {
            npc_id,
            tick: self.ticks.fetch_add(1, AtomicOrdering::Relaxed),
            at_ms: now_ms,
            world_ms,
            perceived,
            cause,
            acts,
            heartbeat_ms: inbox.heartbeat.as_millis() as u64,
            inbox_after: inbox.depth(),
            at: self.born.elapsed(),
            // Filled in on the way out — see `recent`.
            ms_ago: 0,
        };
        // Nothing to queue: the character's own task returns to `wait_due`
        // after this and reads the fresh `due_at` — `NEVER` parks it on its
        // waker, a deadline sleeps it, zero runs it again at once.
        drop(inboxes);

        let mut recent = self.recent.lock().unwrap();
        recent.push_back(record.clone());
        while recent.len() > self.recent_cap {
            recent.pop_front();
        }
        Some(record)
    }

    /// Decide and record a day roll-over. `None` when the character is still in
    /// the day it thinks it is.
    ///
    /// **The turns the roll-over clears are kept by the journal, because
    /// nothing else will ever see them.** The window is the journal's only
    /// source, and the new day begins with none of yesterday in it; the
    /// guardian asks about them like any other stretch.
    pub fn roll_day(&self, npc_id: u64, world_ms: u64) -> Option<Rollover> {
        let mut inboxes = self.inboxes.lock().unwrap();
        let inbox = inboxes.get_mut(&npc_id)?;
        match inbox.day.evaluate(world_ms) {
            DayAction::Continue => None,
            DayAction::RollOver { from, to } => {
                inbox.journal.lock().unwrap().strand(&inbox.window);
                inbox.day.rolled_over_to(to);
                inbox.window.roll_over();
                Some(Rollover { from, to })
            }
        }
    }
}
