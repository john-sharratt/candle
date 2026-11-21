//! Which characters the scheduler holds, and which of them are held back from
//! taking a turn.
//!
//! A suspended or resting character is never due, and what reaches it waits in
//! its inbox, bounded per source when it is let go. The two holds are kept apart
//! because different owners lift them, and a character held both ways stays held
//! until neither holds it. A pace change schedules no thought, so it cannot put a
//! quiet character on the schedule.

use super::*;

impl Scheduler {
    pub fn new(recent_cap: usize) -> Self {
        Self {
            inboxes: Mutex::new(HashMap::new()),
            seq: AtomicU64::new(0),
            ticks: AtomicU64::new(0),
            recent: Mutex::new(VecDeque::new()),
            recent_cap: recent_cap.max(1),
            born: Instant::now(),
            suspended: Mutex::new(HashSet::new()),
            resting: Mutex::new(HashMap::new()),
            whole: Mutex::new(HashMap::new()),
            talk: OnceLock::new(),
        }
    }

    /// Hand every character's conversation events to `talk`.
    pub fn set_talk(&self, talk: Talk) {
        if self.talk.set(talk).is_err() {
            tracing::warn!("the scheduler's conversations were already set");
        }
    }

    /// Suspend or resume one character. Returns whether that changed anything.
    ///
    /// A suspended character is never due: [`Self::wait_due`] parks its task
    /// on its waker, [`Self::due_now`] leaves it out, and [`Self::begin_tick`]
    /// refuses it a turn. Everything delivered meanwhile stays in its inbox.
    /// Resuming bounds that backlog per source ([`held::bound`]) and rings the
    /// waker, so a character with something waiting thinks about it at once,
    /// on the newest of each source and a count of the rest.
    pub fn suspend(&self, npc_id: u64, suspended: bool) -> bool {
        let changed = {
            let mut set = self.suspended.lock().unwrap();
            if suspended {
                set.insert(npc_id)
            } else {
                set.remove(&npc_id)
            }
        };
        if changed && !suspended && !self.is_held(npc_id) {
            self.resume(npc_id);
        }
        changed
    }

    /// Put one character at rest, or wake it. Returns whether that changed
    /// anything.
    ///
    /// **A character at rest is held exactly as a suspended one is**: never
    /// due, refused a turn, and everything delivered meanwhile kept in its
    /// inbox, bounded per source when it wakes. What differs is who holds it —
    /// the engine, for a character with no work to carry, where a suspension
    /// is its owner's — so the two are kept apart, and a character both
    /// suspended and at rest stays held until neither holds.
    pub fn rest(&self, npc_id: u64, resting: bool) -> bool {
        let changed = {
            let mut set = self.resting.lock().unwrap();
            match resting {
                true if set.contains_key(&npc_id) => false,
                true => {
                    set.insert(npc_id, self.now_ms());
                    true
                }
                false => set.remove(&npc_id).is_some(),
            }
        };
        if changed && !resting && !self.is_held(npc_id) {
            self.resume(npc_id);
        }
        changed
    }

    /// Deliver `seen` — what a mind that is its world sees of the whole of it
    /// — as its situation, when it differs from what it was last carried.
    /// Returns whether it was delivered.
    ///
    /// The view is composed every moment and is the same for hours; a
    /// situation carried again unchanged is a turn's worth of prompt rewritten
    /// for nothing. At idle salience: a view is there to be read when the mind
    /// next thinks, and is no reason to think.
    pub fn carry_whole(&self, npc_id: u64, at_ms: u64, seen: String) -> bool {
        let mut h = DefaultHasher::new();
        seen.hash(&mut h);
        let digest = h.finish();
        if self.whole.lock().unwrap().insert(npc_id, digest) == Some(digest) {
            return false;
        }
        self.deliver(
            npc_id,
            at_ms,
            Salience::IDLE,
            EventKind::Situation { text: seen },
        );
        true
    }

    /// Whether this character is at rest.
    pub fn is_resting(&self, npc_id: u64) -> bool {
        self.resting.lock().unwrap().contains_key(&npc_id)
    }

    /// When this character's rest began, on the scheduler's clock — `None`
    /// for one that is not at rest.
    pub fn resting_since(&self, npc_id: u64) -> Option<u64> {
        self.resting.lock().unwrap().get(&npc_id).copied()
    }

    /// Whether this character is held — suspended or at rest — and so takes no
    /// turn.
    pub fn is_held(&self, npc_id: u64) -> bool {
        self.is_suspended(npc_id) || self.is_resting(npc_id)
    }

    /// Hold exactly `suspended` suspended — the cast's suspended characters,
    /// as its records say — and return each character whose standing changed,
    /// with whether it is suspended now.
    ///
    /// What follows the records: a character suspended or resumed on another
    /// member arrives here as its record, and this is how the scheduler comes
    /// to agree with it. Each character resumed is let go as
    /// [`Self::suspend`] lets it go.
    pub fn follow_suspended(&self, suspended: HashSet<u64>) -> Vec<(u64, bool)> {
        let changed: Vec<(u64, bool)> = {
            let mut set = self.suspended.lock().unwrap();
            let mut changed: Vec<(u64, bool)> = suspended
                .difference(&set)
                .map(|id| (*id, true))
                .chain(set.difference(&suspended).map(|id| (*id, false)))
                .collect();
            changed.sort_unstable();
            *set = suspended;
            changed
        };
        for (id, now_suspended) in &changed {
            if !now_suspended && !self.is_held(*id) {
                self.resume(*id);
            }
        }
        changed
    }

    /// Whether this character is suspended.
    pub fn is_suspended(&self, npc_id: u64) -> bool {
        self.suspended.lock().unwrap().contains(&npc_id)
    }

    /// Every character suspended now.
    pub fn suspended(&self) -> HashSet<u64> {
        self.suspended.lock().unwrap().clone()
    }

    /// Let a character go: bound what queued while it was held, then wake its
    /// task. Nothing for a character with no inbox here.
    ///
    /// The head of the queue a failed turn handed back is already in the
    /// window ([`Self::retry`]) and is left whole; what follows it is bounded
    /// to [`held::PER_SOURCE`] per source.
    pub(super) fn resume(&self, npc_id: u64) {
        let waker = {
            let mut inboxes = self.inboxes.lock().unwrap();
            let Some(inbox) = inboxes.get_mut(&npc_id) else {
                return;
            };
            let waiting: Vec<Event> = inbox.queue.drain(..).collect();
            let (replayed, held) = waiting.split_at(inbox.replayed.min(waiting.len()));
            let bounded = held::bound(held.to_vec(), held::PER_SOURCE, || self.next_seq());
            inbox.queue = replayed.iter().cloned().chain(bounded).collect();
            Arc::clone(&inbox.waker)
        };
        waker.notify_one();
    }

    /// Bring a character into the scheduler. Idempotent — a character already
    /// present keeps its inbox, which is what makes this safe to call from a
    /// mind-file reload.
    pub fn wake(&self, npc_id: u64, _now_ms: u64, world_ms: u64) {
        let mut inboxes = self.inboxes.lock().unwrap();
        if inboxes.contains_key(&npc_id) {
            return;
        }
        let mut inbox = Inbox::new(npc_id);
        inbox.day.start_at(world_ms);
        /* **Waking is not a reason to think.**
         *
         * A character was given a first thought here, staggered by id so a
         * hundred of them coming up at once did not stampede one GPU. There is
         * nothing to stagger any more: a character that has just woken has an
         * empty inbox, and thinking about an empty inbox is what this whole
         * change removed.
         *
         * It still gets going the moment there is anything to get going about —
         * `embody` hands it its situation, and the world's sweep hands it
         * whatever is happening — and each of those schedules it on arrival. A
         * cast that boots into a quiet world simply costs nothing until the
         * world does something. */
        inbox.due_at = NEVER;
        inboxes.insert(npc_id, inbox);
    }

    /// Set how alert a character is allowed to be.
    ///
    /// **This no longer schedules anything, and that is the change.** It used
    /// to pull a character forward to the new floor, on the reasoning that a
    /// Maker given its working pace should not sit out an interval it is no
    /// longer allowed to have. That reasoning held only while a pace *was* a
    /// schedule — and it is what made being in a world mean thinking every four
    /// seconds into a quiet room, which filled the character's window with its
    /// own output until it could not say anything else.
    ///
    /// A character now thinks when something reaches it. The pace is the ceiling
    /// on how alert it reads as, not a promise of a thought.
    ///
    /// `false` if there is no such character.
    pub fn set_pace(&self, npc_id: u64, pace: Pace, _now_ms: u64) -> bool {
        let mut inboxes = self.inboxes.lock().unwrap();
        let Some(inbox) = inboxes.get_mut(&npc_id) else {
            return false;
        };
        inbox.pace = pace;
        inbox.heartbeat = inbox.heartbeat.min(pace.slowest());
        true
    }

    /// What pace a character is on, if it is in the scheduler.
    pub fn pace_of(&self, npc_id: u64) -> Option<Pace> {
        self.inboxes.lock().unwrap().get(&npc_id).map(|i| i.pace())
    }

    /// How much is waiting for a character, if it is in the scheduler.
    ///
    /// What a caller with something optional to add asks first: a standing
    /// instruction is worth restating to a character with nothing to think
    /// about, and is an interruption to one in the middle of a conversation.
    pub fn inbox_depth(&self, npc_id: u64) -> Option<usize> {
        self.inboxes.lock().unwrap().get(&npc_id).map(|i| i.depth())
    }

    /// How long this character has gone without news, at `world_ms`.
    /// `None` for a character with no inbox. See [`Inbox::quiet_for`].
    pub fn quiet_for(&self, npc_id: u64, world_ms: u64) -> Option<u64> {
        self.inboxes
            .lock()
            .unwrap()
            .get(&npc_id)
            .map(|i| i.quiet_for(world_ms))
    }

    /// Every character in the scheduler, for a sweep that must not hold the
    /// map while it works — `answer_waits` reads the world per character, and
    /// holding the inbox map across that would block every `deliver` behind it.
    pub fn population_ids(&self) -> Vec<u64> {
        self.inboxes.lock().unwrap().keys().copied().collect()
    }

    /// Stop a character for a while. See [`Inbox::pause_for`].
    ///
    /// Returns whether there was a character to stop.
    pub fn pause_for(&self, npc_id: u64, now_ms: u64, for_ms: u64) -> bool {
        let waker = {
            let mut g = self.inboxes.lock().unwrap();
            let Some(i) = g.get_mut(&npc_id) else {
                return false;
            };
            i.pause_for(now_ms, for_ms);
            Arc::clone(&i.waker)
        };
        // Woken so the character's task re-reads its deadline: a pause set from
        // outside the task's own tick would otherwise not shorten a longer
        // sleep already in flight.
        waker.notify_one();
        true
    }

    /// Remove a character — deleted, or no longer in the cast after a reload.
    ///
    /// The character's own task is woken so it finds the inbox gone and exits;
    /// see [`Self::wait_due`].
    pub fn retire(&self, npc_id: u64) {
        let gone = self.inboxes.lock().unwrap().remove(&npc_id);
        if let Some(inbox) = gone {
            inbox.waker.notify_one();
        }
    }

    pub fn population(&self) -> usize {
        self.inboxes.lock().unwrap().len()
    }
}
