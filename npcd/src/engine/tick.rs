//! The tick scheduler — one loop per character, all of them, all the time.
//!
//! # Every character, not just the ones being watched
//!
//! This is the property that makes the world feel inhabited rather than staged.
//! A character nobody is talking to still perceives, still thinks, still acts —
//! the quartermaster is counting sacks whether or not a player is in the yard.
//! An engine that ticked only the characters a user had open would produce a
//! world that springs to life when observed, which is exactly the thing the
//! substrate exists to avoid.
//!
//! # Idle costs nothing
//!
//! The mind design's central scheduling claim: an NPC waiting on an empty inbox
//! burns no decode and is not in the batch. So the population scales with *what
//! is happening*, not with how many characters exist. A thousand characters
//! standing still cost a thousand entries in a map and no GPU at all.
//!
//! That is why the scheduler is a queue over *ready* characters rather than a
//! sweep over all of them. A sweep is the obvious implementation and it silently
//! makes idle cost O(population) per tick, which is the one thing this design
//! cannot afford.
//!
//! # Salience decides when, never whether
//!
//! An arrival at or above [`Salience::PREEMPT_AT`] forces a tick now. Everything
//! else waits for the character's own heartbeat, whose interval is its idle
//! metabolism — a guard at his post ticks slowly, the same guard who just heard
//! something ticks tight until it settles. Nothing is ever dropped for being
//! low-salience; it simply waits.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;

use crate::engine::event::{Event, EventKind, Salience};
use crate::engine::sleep::{DayAction, DayTracker};
use crate::engine::waiting::Waiting;
use crate::engine::window::Window;

/// The slowest a wholly idle character thinks. Long, because a character with
/// nothing happening genuinely has nothing to think about, and the cost of a
/// wake is a decode.
pub const IDLE_HEARTBEAT: Duration = Duration::from_secs(120);

/// The fastest a character settles back to after something happened.
pub const ALERT_HEARTBEAT: Duration = Duration::from_secs(4);

/// How much of the way back toward idle a character relaxes per quiet tick.
/// Below 1.0 so alertness decays rather than snapping back — a character that
/// heard something stays tight for a while, which is what reads as alert.
const RELAX: f32 = 1.6;

/// The slowest a character is allowed to become.
///
/// The relax curve is right for a character whose life is *reactive*: nothing
/// happening means nothing to think about, and settling to a two-minute
/// heartbeat is the whole reason a thousand of them are affordable.
///
/// It is wrong for a character whose work is continuous. A Maker with a
/// standing task always has something to do, so its quiet turns are not empty —
/// they carry the task forward. Letting it settle to two minutes would not save
/// anything; it would stall the work and read as a character that had forgotten
/// what it was doing.
///
/// So the floor is per character. Everything else about the curve — tighten on
/// a preempt, relax while quiet — is unchanged; this only says how far the
/// relaxing is allowed to go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pace {
    slowest: Duration,
}

impl Pace {
    /// A character that thinks when something happens to it.
    pub const AMBIENT: Pace = Pace {
        slowest: IDLE_HEARTBEAT,
    };

    /// A character with work in front of it, which never truly goes quiet.
    pub const WORKING: Pace = Pace {
        slowest: ALERT_HEARTBEAT,
    };

    /// Thinks at least this often.
    ///
    /// Clamped to [`ALERT_HEARTBEAT`] at the fast end: below that a character
    /// thinks faster than the world moves, and spends decodes discovering that
    /// nothing has changed since the last one.
    pub fn at_most(gap: Duration) -> Pace {
        Pace {
            slowest: gap.max(ALERT_HEARTBEAT),
        }
    }

    pub fn slowest(self) -> Duration {
        self.slowest
    }
}

impl Default for Pace {
    fn default() -> Self {
        Pace::AMBIENT
    }
}

/// A character's readiness to think, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Readiness {
    /// Inbox empty, waiting on the heartbeat. Costs nothing.
    Blocked,
    /// Events waiting; will tick at its scheduled moment.
    Pending,
    /// A high-salience arrival is forcing a tick now.
    Preempted,
}

/// One character's loop state.
#[derive(Debug)]
pub struct Inbox {
    pub npc_id: u64,
    queue: VecDeque<Event>,
    /// Current heartbeat interval — the character's idle metabolism.
    heartbeat: Duration,
    /// How slow that metabolism is allowed to get.
    pace: Pace,
    /// Monotonic scheduler time (ms) at which this character next thinks.
    due_at: u64,
    preempted: bool,
    pub window: Window,
    pub day: DayTracker,
    /// Lifetime counters, for the Pulse view.
    pub ticks: u64,
    pub events_seen: u64,
    /// World time (ms) of the last **news** this character received.
    ///
    /// What the standing task is gated on. `0` for a character none has ever
    /// reached, which reads as "quiet since the beginning" — correct, because a
    /// character that has never been told anything is exactly the one the
    /// standing task is for.
    last_news_ms: u64,
    /// What this character has stopped to wait for, if anything.
    ///
    /// Scheduling state, beside `heartbeat` and `due_at`, because that is what
    /// a wait *is*: the character is not thinking until the thing happens or
    /// the patience runs out. See [`crate::engine::waiting`].
    waiting: Option<Waiting>,
    /// Who was in the room when the wait was set.
    ///
    /// Arrivals and departures reach a character as one line of narrated prose,
    /// so "did somebody come in" cannot be answered from the event stream — it
    /// is answered by comparing the room then against the room now, and this is
    /// *then*.
    waiting_room: Vec<String>,
    /// World time (ms) the standing task was last restated to this character.
    ///
    /// **The task has to be quiet about itself too.** It is not news, so it
    /// cannot stamp `last_news_ms` — which left the gate, once open, open
    /// forever: a character with nobody talking to it was handed the task again
    /// on every tick, and because it supersedes in its own band each copy took
    /// the most recent position in the window. Identical instruction, identical
    /// context, identical decode; the character repeated one act until something
    /// outside it changed.
    last_nudge_ms: u64,
}

impl Inbox {
    pub fn new(npc_id: u64) -> Self {
        Self {
            npc_id,
            queue: VecDeque::new(),
            heartbeat: IDLE_HEARTBEAT,
            pace: Pace::AMBIENT,
            due_at: 0,
            preempted: false,
            window: Window::with_default_cap(),
            day: DayTracker::new(),
            ticks: 0,
            events_seen: 0,
            waiting: None,
            waiting_room: Vec::new(),
            last_news_ms: 0,
            last_nudge_ms: 0,
        }
    }

    /// Stop and wait. The character goes quiet until the wait is answered or
    /// its patience runs out — whichever the world decides first.
    pub fn begin_waiting(&mut self, waiting: Waiting, room: Vec<String>, now_ms: u64) {
        // The deadline *is* the heartbeat while a wait stands. A character that
        // kept its ordinary beat would go on thinking every four seconds about
        // a thing it had already decided to wait for, which is the busy-loop the
        // typed wait exists to end.
        self.due_at = now_ms.saturating_add(
            waiting
                .until_ms
                .saturating_sub(self.waiting_now_ms(&waiting)),
        );
        self.waiting_room = room;
        self.waiting = Some(waiting);
    }

    /// The world clock the wait was stamped against, recovered from its own
    /// deadline. Keeps `begin_waiting` honest when scheduler time and world
    /// time are not the same clock — which they are not.
    fn waiting_now_ms(&self, w: &Waiting) -> u64 {
        w.until_ms.saturating_sub(crate::engine::waiting::PATIENCE_MS)
    }

    /// What this character is waiting for, if anything.
    pub fn waiting(&self) -> Option<&Waiting> {
        self.waiting.as_ref()
    }

    /// Who was here when the wait was set.
    pub fn waiting_room(&self) -> &[String] {
        &self.waiting_room
    }

    /// The wait is over. Returns whether one was actually standing, so a caller
    /// can tell "answered" from "there was nothing to answer".
    pub fn stop_waiting(&mut self) -> bool {
        self.waiting_room.clear();
        self.waiting.take().is_some()
    }

    /// Whether anything queued is speech `f` accepts.
    fn heard_speech(&self, f: &impl Fn(&str) -> bool) -> bool {
        self.queue.iter().any(|e| match &e.kind {
            EventKind::Speech { speaker, .. } => f(speaker),
            _ => false,
        })
    }

    /// Think on the next pass rather than on the next heartbeat.
    ///
    /// The same flag an arriving high-salience event sets, reached from the
    /// other direction: there the world decided this was worth interrupting
    /// for, here the character said so in advance.
    fn wake_now(&mut self) {
        self.preempted = true;
        self.due_at = 0;
    }

    pub fn depth(&self) -> usize {
        self.queue.len()
    }

    /// Whether an event is **news** — something that happened, as against
    /// something that is merely true.
    ///
    /// The distinction the standing task turns on. A character's inbox is never
    /// empty while it has a body: the environment refreshes where it is standing
    /// every moment, and that situation is a fact about the room rather than an
    /// event asking for a response. Speech and things that happened nearby are
    /// news; where you are, time passing, and the standing task itself are not.
    fn is_news(kind: &EventKind) -> bool {
        matches!(
            kind,
            EventKind::Speech { .. }
                | EventKind::Description { .. }
                | EventKind::Entity { .. }
                | EventKind::Operator { .. }
                | EventKind::Wake { .. }
                | EventKind::Sleep { .. }
        )
    }

    /// How long this character has gone without news, at `world_ms`.
    ///
    /// # Why a duration and not "is the queue empty of news"
    ///
    /// The standing task is for a character with **nothing going on**, and that
    /// is a question about a stretch of time, not about this instant. Asked
    /// instantaneously it is true in every gap between two things happening —
    /// and a conversation is mostly gaps. A character speaks, its companion
    /// hears it a moment later and answers a moment after that, and in between
    /// every one of those the queue holds no news.
    ///
    /// So the standing task was landing *inside* conversations. Two characters
    /// alternated perfectly for a hundred turns: heard the other speak, then
    /// were told "nothing has been asked of you", then heard the other speak,
    /// then were told it again. And because the task supersedes in its own band
    /// it is always the most recent thing in the window — the position attention
    /// weights hardest — so each of them was being told, more recently than
    /// anything their companion had actually said, that nothing was going on.
    ///
    /// This question has now been asked three ways. `depth()` was wrong in both
    /// directions; `has_news()` fixed the false negative and left this false
    /// positive; a stretch of quiet is the thing the task was always described
    /// as waiting for.
    pub fn quiet_for(&self, world_ms: u64) -> u64 {
        // Anything queued and unread is news happening right now, whatever the
        // clock says — it arrived after the last tick and has not been answered.
        if self.queue.iter().any(|e| Self::is_news(&e.kind)) {
            return 0;
        }
        world_ms.saturating_sub(self.last_news_ms)
    }

    /// Whether the standing task is due: `after_ms` without news, **and**
    /// `after_ms` since it was last restated.
    ///
    /// The second half is what makes this a cadence rather than a latch. On the
    /// news clock alone the gate opens once and never shuts — the task is not
    /// news, so nothing it does moves that clock — and a character nobody is
    /// talking to gets it again every tick, at the most recent position in the
    /// window every time. Restating it is meant to keep a long quiet run on
    /// course, not to be the run.
    pub fn nudge_due(&self, world_ms: u64, after_ms: u64) -> bool {
        self.quiet_for(world_ms) >= after_ms
            && world_ms.saturating_sub(self.last_nudge_ms) >= after_ms
    }

    pub fn readiness(&self) -> Readiness {
        if self.preempted {
            Readiness::Preempted
        } else if self.queue.is_empty() {
            Readiness::Blocked
        } else {
            Readiness::Pending
        }
    }

    /// Accept an event. Never refuses: the mind design is explicit that a filter
    /// dropping evidence before it lands makes a delusion permanent.
    ///
    /// Returns whether this arrival forces a tick now.
    pub fn push(&mut self, event: Event) -> bool {
        self.events_seen += 1;
        // Stamped on arrival rather than on drain: the clock the standing task
        // is gated on is "when did something last happen to this character",
        // and that is when it reached them, not when they got round to it.
        if Self::is_news(&event.kind) {
            self.last_news_ms = self.last_news_ms.max(event.at_ms);
        }
        // Stamped here rather than at the delivery site so it cannot be
        // forgotten by a second caller: every path that hands a character the
        // standing task goes through this one.
        if matches!(event.kind, EventKind::Nudge { .. }) {
            self.last_nudge_ms = self.last_nudge_ms.max(event.at_ms);
        }
        let preempts = event.preempts();
        if preempts {
            self.preempted = true;
            // Something happened. Tighten the metabolism — this is the whole of
            // the alertness model, and it needs no combat branch anywhere.
            self.heartbeat = ALERT_HEARTBEAT;
        }
        self.queue.push_back(event);
        preempts
    }

    /// Take everything waiting. A busy character drains a fat batch — one
    /// better-informed thinking step rather than several thrashing ones.
    pub fn drain(&mut self) -> Vec<Event> {
        self.preempted = false;
        self.queue.drain(..).collect()
    }

    /// Relax toward idle after a quiet tick, as far as this character's pace
    /// allows.
    fn relax(&mut self) {
        let next = self.heartbeat.mul_f32(RELAX);
        self.heartbeat = next.min(self.pace.slowest());
    }

    /// How slow this character may become, and how slow it currently is.
    pub fn pace(&self) -> Pace {
        self.pace
    }
}

/// A scheduled wake, ordered so the earliest is popped first.
///
/// `BinaryHeap` is a max-heap, so the comparison is reversed — the classic place
/// to get this backwards and end up with a scheduler that runs the furthest-out
/// character first and looks like a hang.
#[derive(Debug, PartialEq, Eq)]
struct Due {
    at: u64,
    npc_id: u64,
}

impl Ord for Due {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at
            .cmp(&self.at)
            .then_with(|| other.npc_id.cmp(&self.npc_id))
    }
}

impl PartialOrd for Due {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// What one tick did — the Pulse view's row.
#[derive(Clone, Debug, Serialize)]
pub struct TickRecord {
    /// **Serialised as a string, and it has to be.**
    ///
    /// Character ids are minted across the whole `u64` range — a real one is
    /// `6817662845163923144`, comfortably past the 2^53 where a JavaScript
    /// number stops being exact. As a JSON number it arrives at the console
    /// rounded, so a tick's id would not match the same character's id anywhere
    /// else on the page: the cast strip would colour one character and the feed
    /// another, and clicking a row would filter to a character that does not
    /// exist. The character record itself already serialises its id this way;
    /// this is the same decision, and missing it here is invisible until an id
    /// happens to be large.
    #[serde(serialize_with = "id_as_string")]
    pub npc_id: u64,
    /// Monotonic within a daemon run.
    pub tick: u64,
    /// Scheduler time this tick ran at.
    pub at_ms: u64,
    /// The world-clock instant the character reasoned about.
    pub world_ms: u64,
    /// What it perceived, as prose — exactly what went to the model.
    pub perceived: Vec<String>,
    /// Why it ran now.
    pub cause: Readiness,
    /// Acts that came out, rendered. Empty until the decode path is live.
    pub acts: Vec<String>,
    /// Heartbeat after this tick, in ms — the character's current alertness.
    pub heartbeat_ms: u64,
    pub inbox_after: usize,
}

/// The scheduler: every character's loop, and the order they run in.
pub struct Scheduler {
    inboxes: Mutex<HashMap<u64, Inbox>>,
    due: Mutex<BinaryHeap<Due>>,
    /// Monotonic event sequence, shared by every character so the Pulse view can
    /// order arrivals across the whole cast.
    seq: AtomicU64,
    /// Monotonic tick counter, same reason.
    ticks: AtomicU64,
    /// Recent ticks, newest last. Bounded — this is an instrument, not a log.
    recent: Mutex<VecDeque<TickRecord>>,
    recent_cap: usize,
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(512)
    }
}

impl Scheduler {
    pub fn new(recent_cap: usize) -> Self {
        Self {
            inboxes: Mutex::new(HashMap::new()),
            due: Mutex::new(BinaryHeap::new()),
            seq: AtomicU64::new(0),
            ticks: AtomicU64::new(0),
            recent: Mutex::new(VecDeque::new()),
            recent_cap: recent_cap.max(1),
        }
    }

    /// Bring a character into the scheduler. Idempotent — a character already
    /// present keeps its inbox, which is what makes this safe to call from a
    /// mind-file reload.
    pub fn wake(&self, npc_id: u64, now_ms: u64, world_ms: u64) {
        let mut inboxes = self.inboxes.lock().unwrap();
        if inboxes.contains_key(&npc_id) {
            return;
        }
        let mut inbox = Inbox::new(npc_id);
        inbox.day.start_at(world_ms);
        // Staggered by id rather than all due at once: a hundred characters
        // waking on the same millisecond is a thundering herd against a single
        // GPU, and the stagger costs nothing.
        inbox.due_at = now_ms + (npc_id % IDLE_HEARTBEAT.as_millis() as u64);
        let due = Due {
            at: inbox.due_at,
            npc_id,
        };
        inboxes.insert(npc_id, inbox);
        drop(inboxes);
        self.due.lock().unwrap().push(due);
    }

    /// Set how slow a character is allowed to become.
    ///
    /// Takes effect immediately in both directions: quickening pulls a
    /// character that was settled into a long wait forward to the new floor,
    /// rather than leaving it to sit out an interval it is no longer allowed to
    /// have. Without that, giving a Maker its working pace could leave it two
    /// minutes from its next thought — which is exactly the state the pace was
    /// set to prevent.
    ///
    /// `false` if there is no such character.
    pub fn set_pace(&self, npc_id: u64, pace: Pace, now_ms: u64) -> bool {
        let mut inboxes = self.inboxes.lock().unwrap();
        let Some(inbox) = inboxes.get_mut(&npc_id) else {
            return false;
        };
        inbox.pace = pace;
        let slowest = pace.slowest().as_millis() as u64;
        inbox.heartbeat = inbox.heartbeat.min(pace.slowest());

        let sooner = now_ms + slowest;
        if inbox.due_at > sooner {
            inbox.due_at = sooner;
            let due = Due { at: sooner, npc_id };
            drop(inboxes);
            // The entry for the old, later moment stays in the heap and will
            // pop in its own time. Harmless: an early wake with an empty inbox
            // is a heartbeat, which relaxes and re-queues. Being woken sooner
            // than necessary costs a cheap tick; being woken later than the
            // pace allows is the thing this exists to prevent.
            self.due.lock().unwrap().push(due);
        }
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

    /// Whether anything queued for this character is speech `f` accepts.
    ///
    /// Peeked, not drained: the character is about to be woken *for* this, and
    /// it has to still be there when it reads its batch.
    pub fn heard_speech(&self, npc_id: u64, f: impl Fn(&str) -> bool) -> bool {
        self.inboxes
            .lock()
            .unwrap()
            .get(&npc_id)
            .is_some_and(|i| i.heard_speech(&f))
    }

    /// Stop a character to wait. See [`Inbox::begin_waiting`].
    pub fn begin_waiting(&self, npc_id: u64, waiting: Waiting, room: Vec<String>, now_ms: u64) {
        if let Some(i) = self.inboxes.lock().unwrap().get_mut(&npc_id) {
            i.begin_waiting(waiting, room, now_ms);
        }
    }

    /// What a character is waiting for, and who was here when it started.
    pub fn waiting(&self, npc_id: u64) -> Option<(Waiting, Vec<String>)> {
        let g = self.inboxes.lock().unwrap();
        let i = g.get(&npc_id)?;
        Some((i.waiting()?.clone(), i.waiting_room().to_vec()))
    }

    /// End a wait and wake the character **now**.
    ///
    /// The preempt is the point. A wait that was answered and then sat until the
    /// next heartbeat would have the character replying to a question four
    /// seconds after it was asked, with the answer buried in whatever else
    /// arrived meanwhile — which is the batching working exactly against the one
    /// case where the character has said in advance what it cares about.
    pub fn answer_wait(&self, npc_id: u64) -> bool {
        {
            let mut g = self.inboxes.lock().unwrap();
            let Some(i) = g.get_mut(&npc_id) else {
                return false;
            };
            if !i.stop_waiting() {
                return false;
            }
            i.wake_now();
        }
        // Queued explicitly: the entry this character was carrying was consumed
        // by whichever `due_now` deferred it back, or by the tick that began the
        // wait, so clearing `due_at` alone would leave it due and unqueued —
        // awake with nothing to notice it.
        self.due.lock().unwrap().push(Due { at: 0, npc_id });
        true
    }

    /// Whether this character is due the standing task. See [`Inbox::nudge_due`].
    pub fn nudge_due(&self, npc_id: u64, world_ms: u64, after_ms: u64) -> bool {
        self.inboxes
            .lock()
            .unwrap()
            .get(&npc_id)
            .is_some_and(|i| i.nudge_due(world_ms, after_ms))
    }

    /// Remove a character — deleted, or no longer in the cast after a reload.
    pub fn retire(&self, npc_id: u64) {
        self.inboxes.lock().unwrap().remove(&npc_id);
        // The heap entry is left to expire: popping it is O(n) and the tick loop
        // already skips ids with no inbox. A stale entry costs one comparison.
    }

    pub fn population(&self) -> usize {
        self.inboxes.lock().unwrap().len()
    }

    /// Entries standing in the due heap.
    ///
    /// Not the population: a retired character's entry is left to expire, and a
    /// character is entered once per wake it is owed. The number the duplicate
    /// -wake test reads, since the scheduler behaves identically either way and
    /// only the heap shows the difference.
    #[cfg(test)]
    fn due_len(&self) -> usize {
        self.due.lock().unwrap().len()
    }

    /// Deliver an event to a character. `false` if there is no such character.
    pub fn deliver(&self, npc_id: u64, world_ms: u64, salience: Salience, kind: EventKind) -> bool {
        let seq = self.seq.fetch_add(1, AtomicOrdering::Relaxed);
        let event = Event::new(seq, world_ms, salience, kind);
        let mut inboxes = self.inboxes.lock().unwrap();
        let Some(inbox) = inboxes.get_mut(&npc_id) else {
            return false;
        };
        let preempts = inbox.push(event);
        if preempts {
            // Due immediately — but only *entered* as due once. `due_at == 0`
            // already means an at-zero entry is standing in the heap and has not
            // been served, so a second one would be a duplicate wake for a
            // character that is already at the front of the queue. A burst of
            // twenty preempting arrivals between two passes pushed twenty
            // entries, all naming the same character, all popped by the same
            // pass — work proportional to the burst to schedule one tick that
            // drains the whole burst anyway.
            //
            // An arrival *during* a decode still gets its entry: the tick set
            // `due_at` to its next heartbeat before decoding, so this reads
            // non-zero and the character is re-woken the moment it finishes.
            let already_queued = inbox.due_at == 0;
            inbox.due_at = 0;
            drop(inboxes);
            if !already_queued {
                self.due.lock().unwrap().push(Due { at: 0, npc_id });
            }
        }
        true
    }

    /// Deliver to every character at once — a world event nobody is exempt from.
    ///
    /// `world_ms` is a function rather than a value: characters live in
    /// different worlds at different paces, so there is no single instant to
    /// stamp them all with. Thunder over two worlds happens at two times.
    pub fn broadcast<F>(&self, world_ms: F, salience: Salience, kind: EventKind) -> usize
    where
        F: Fn(u64) -> u64,
    {
        // Ids collected first, and the lock released, because `deliver` takes it
        // again per character.
        let ids: Vec<u64> = self.inboxes.lock().unwrap().keys().copied().collect();
        ids.iter()
            .filter(|id| self.deliver(**id, world_ms(**id), salience, kind.clone()))
            .count()
    }

    /// Read a character's window under the lock, without cloning it out.
    ///
    /// A closure rather than a getter: the window lives inside the inbox map, and
    /// handing out a clone of every turn to answer one request would copy the
    /// whole tail per poll of the Pulse view.
    pub fn window_of<T, F>(&self, npc_id: u64, f: F) -> Option<T>
    where
        F: FnOnce(&Window) -> T,
    {
        let inboxes = self.inboxes.lock().unwrap();
        inboxes.get(&npc_id).map(|i| f(&i.window))
    }

    /// The characters due to think at or before `now_ms`, soonest first.
    ///
    /// Pops from the heap rather than sweeping the population, so an idle cast
    /// costs nothing per pass.
    pub fn due_now(&self, now_ms: u64) -> Vec<u64> {
        let mut ready = Vec::new();
        let mut deferred: Vec<Due> = Vec::new();
        let mut due = self.due.lock().unwrap();
        let inboxes = self.inboxes.lock().unwrap();
        while let Some(top) = due.peek() {
            if top.at > now_ms {
                break;
            }
            let Due { npc_id, .. } = due.pop().expect("peeked");
            // Retired, or a stale duplicate from a preempt. Either way, drop it.
            let Some(inbox) = inboxes.get(&npc_id) else {
                continue;
            };
            // **`due_at` is the truth; a heap entry is a hint.**
            //
            // An entry is pushed when a tick ends and cannot be withdrawn, so
            // anything that moves a character's next thought *later* — a wait
            // beginning after the entry was queued — leaves a stale entry that
            // would drag it back. Re-queue it where it now belongs rather than
            // running it early: a waiting character woken by its own heartbeat
            // is the busy-loop the typed wait exists to end.
            if inbox.due_at > now_ms {
                deferred.push(Due {
                    at: inbox.due_at,
                    npc_id,
                });
                continue;
            }
            if !ready.contains(&npc_id) {
                ready.push(npc_id);
            }
        }
        for d in deferred {
            due.push(d);
        }
        ready
    }

    /// Run one character's tick.
    ///
    /// `act` is handed the drained events as prose and returns the acts the
    /// character took. It is a closure so the scheduler can be tested — and
    /// reasoned about — without a GPU: the scheduling *is* the thing under test
    /// here, and a decode inside it would make every test a model test.
    pub fn tick<F>(&self, npc_id: u64, now_ms: u64, world_ms: u64, act: F) -> Option<TickRecord>
    where
        F: FnOnce(&[Event], &Window) -> Vec<String>,
    {
        // The drain takes the lock and gives it straight back. The decode below
        // can take seconds, and holding the inbox map across it would block
        // every other character's `deliver` for that whole time — an event
        // arriving for a second character would wait on the first one thinking.
        let (events, cause) = {
            let mut inboxes = self.inboxes.lock().unwrap();
            let inbox = inboxes.get_mut(&npc_id)?;
            // Readiness FIRST. `drain` clears the preempt flag, so reading it
            // afterwards reports `Blocked` for every tick and the Pulse feed
            // loses the one column that says why a character woke.
            let cause = inbox.readiness();
            (inbox.drain(), cause)
        };

        // Nothing waiting: this is the heartbeat firing on a quiet character. It
        // still thinks — that is what stops "nothing arrived" meaning "dead
        // forever" — but on a synthetic event rather than on emptiness.
        let events = if events.is_empty() {
            let seq = self.seq.fetch_add(1, AtomicOrdering::Relaxed);
            vec![Event::new(
                seq,
                world_ms,
                Salience::IDLE,
                EventKind::Heartbeat,
            )]
        } else {
            events
        };

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
            for e in &events {
                inbox
                    .window
                    .push_world(e.prose(), e.at_ms, e.kind.replaces());
            }
            inbox.window.clone()
        };

        let acts = act(&events, &snapshot);

        let mut inboxes = self.inboxes.lock().unwrap();
        // Retired while the decode ran. Its acts have nowhere to land and nothing downstream
        // wants a record for a character that is gone.
        let inbox = inboxes.get_mut(&npc_id)?;
        for a in &acts {
            inbox.window.push_npc(a.clone(), world_ms);
        }

        // A tick where nothing arrived is a quiet one; relax toward idle.
        if events
            .iter()
            .all(|e| matches!(e.kind, EventKind::Heartbeat))
        {
            inbox.relax();
        }
        inbox.ticks += 1;
        // **A wait begun during this tick keeps its own deadline.**
        //
        // The act loop arms the wait inside `act`, and this line used to
        // overwrite it a moment later with the ordinary heartbeat — so a
        // character that had just decided to wait was scheduled to think again
        // in four seconds, which is precisely the busy-loop the typed wait
        // replaced. While a wait stands, its patience is the heartbeat.
        if inbox.waiting.is_none() {
            inbox.due_at = now_ms + inbox.heartbeat.as_millis() as u64;
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
        };
        let next = Due {
            at: inbox.due_at,
            npc_id,
        };
        drop(inboxes);
        self.due.lock().unwrap().push(next);

        let mut recent = self.recent.lock().unwrap();
        recent.push_back(record.clone());
        while recent.len() > self.recent_cap {
            recent.pop_front();
        }
        Some(record)
    }

    /// Decide and record a day roll-over. `None` when the character is still in
    /// the day it thinks it is.
    pub fn roll_day(&self, npc_id: u64, world_ms: u64) -> Option<(u64, u64)> {
        let mut inboxes = self.inboxes.lock().unwrap();
        let inbox = inboxes.get_mut(&npc_id)?;
        match inbox.day.evaluate(world_ms) {
            DayAction::Continue => None,
            DayAction::RollOver { from, to } => {
                inbox.day.rolled_over_to(to);
                inbox.window.roll_over();
                Some((from, to))
            }
        }
    }

    /// Recent ticks, newest last. The Pulse view's feed.
    pub fn recent(&self, limit: usize) -> Vec<TickRecord> {
        let r = self.recent.lock().unwrap();
        let skip = r.len().saturating_sub(limit);
        r.iter().skip(skip).cloned().collect()
    }

    /// A snapshot of every character's loop state.
    pub fn census(&self) -> Vec<Census> {
        let inboxes = self.inboxes.lock().unwrap();
        let mut v: Vec<Census> = inboxes
            .values()
            .map(|i| Census {
                npc_id: i.npc_id,
                readiness: i.readiness(),
                inbox_depth: i.depth(),
                heartbeat_ms: i.heartbeat.as_millis() as u64,
                ticks: i.ticks,
                events_seen: i.events_seen,
                window_turns: i.window.len(),
                window_cap: i.window.cap(),
                faded: i.window.faded(),
                day: i.day.current(),
            })
            .collect();
        v.sort_by_key(|c| c.npc_id);
        v
    }
}

/// A `u64` id on the wire as a string. See [`TickRecord::npc_id`].
fn id_as_string<S: serde::Serializer>(id: &u64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&id.to_string())
}

/// One character's loop state, for the roster and the Pulse header.
#[derive(Clone, Debug, Serialize)]
pub struct Census {
    #[serde(serialize_with = "id_as_string")]
    pub npc_id: u64,
    pub readiness: Readiness,
    pub inbox_depth: usize,
    pub heartbeat_ms: u64,
    pub ticks: u64,
    pub events_seen: u64,
    pub window_turns: usize,
    pub window_cap: usize,
    pub faded: u64,
    pub day: Option<u64>,
}

/// Shared handle.
pub type Shared = Arc<Scheduler>;

#[cfg(test)]
mod tests {
    use super::*;

    fn sched() -> Scheduler {
        Scheduler::new(8)
    }

    fn say(text: &str) -> EventKind {
        EventKind::Description { text: text.into() }
    }

    #[test]
    fn a_woken_character_joins_the_population() {
        let s = sched();
        s.wake(1, 0, 0);
        s.wake(2, 0, 0);
        assert_eq!(s.population(), 2);
        s.retire(1);
        assert_eq!(s.population(), 1);
    }

    /// Waking twice must not reset an existing character's inbox — a mind-file
    /// reload calls this for the whole cast.
    #[test]
    fn waking_an_existing_character_is_idempotent() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("something"));
        s.wake(1, 0, 0);
        assert_eq!(s.population(), 1);
        assert_eq!(s.census()[0].inbox_depth, 1, "the reload emptied the inbox");
    }

    #[test]
    fn an_event_for_an_unknown_character_is_refused() {
        let s = sched();
        assert!(!s.deliver(99, 0, Salience::NORMAL, say("x")));
    }

    /// Idle means blocked, and blocked is the state that costs nothing.
    #[test]
    fn an_empty_inbox_reads_blocked() {
        let s = sched();
        s.wake(1, 0, 0);
        assert_eq!(s.census()[0].readiness, Readiness::Blocked);
        s.deliver(1, 0, Salience::NORMAL, say("a noise"));
        assert_eq!(s.census()[0].readiness, Readiness::Pending);
    }

    /// **The tick must record why it ran, not what it looks like afterwards.**
    /// `drain` clears the preempt flag, so reading readiness after it reports
    /// `Blocked` for every tick — and the Pulse feed's cause column, the one
    /// thing that says why a character woke, silently becomes a constant.
    #[test]
    fn a_ticks_recorded_cause_is_its_state_before_the_drain() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::URGENT, say("a bolt"));
        let rec = s.tick(1, 0, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.cause, Readiness::Preempted);

        s.deliver(1, 0, Salience::NORMAL, say("footsteps"));
        let rec = s.tick(1, 0, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.cause, Readiness::Pending);
    }

    /// A high-salience arrival forces a tick now rather than waiting for the
    /// heartbeat.
    #[test]
    fn a_high_salience_event_preempts_and_becomes_due_immediately() {
        let s = sched();
        s.wake(1, 10_000, 0);
        assert!(s.due_now(10_000).is_empty(), "not due yet");
        s.deliver(1, 0, Salience::URGENT, say("the beam gives"));
        assert_eq!(s.census()[0].readiness, Readiness::Preempted);
        assert_eq!(s.due_now(10_000), vec![1]);
    }

    /// **A burst of urgent arrivals is one wake, not one wake each.**
    ///
    /// A character already at the front of the queue cannot be moved further
    /// forward, so every preempting arrival after the first used to push a heap
    /// entry that named a character already standing there — work proportional
    /// to the burst, to schedule the single tick that drains the whole burst.
    /// The tick itself is unaffected either way, which is what kept this
    /// invisible: the character behaves correctly and the heap does the work.
    #[test]
    fn a_burst_of_preempting_arrivals_queues_one_wake() {
        let s = sched();
        s.wake(1, 10_000, 0);
        for _ in 0..20 {
            s.deliver(1, 0, Salience::URGENT, say("the beam gives"));
        }
        // Two: the heartbeat entry `wake` stood up, and one at-zero wake for the
        // whole burst. Twenty-one is the bug — an entry per arrival.
        assert_eq!(s.due_len(), 2, "the burst queued one wake per arrival");

        // And the one wake still drains all twenty.
        assert_eq!(s.due_now(10_000), vec![1]);
        let rec = s.tick(1, 10_000, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.perceived.len(), 20);

        // An arrival during the decode is a different case and must still wake
        // the character: the tick had already set its next heartbeat, so this is
        // the first at-zero entry rather than a duplicate of a standing one.
        s.deliver(1, 0, Salience::URGENT, say("and again"));
        assert_eq!(s.due_now(10_000), vec![1]);
    }

    /// Ordinary traffic waits for the character's own heartbeat. It is not
    /// dropped — nothing ever is — it simply does not jump the queue.
    #[test]
    fn ordinary_traffic_waits_without_being_dropped() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("a rumour"));
        assert_eq!(s.census()[0].inbox_depth, 1);
        assert_eq!(s.census()[0].readiness, Readiness::Pending);
    }

    /// **The heap is a max-heap and the ordering is reversed.** Getting this
    /// backwards produces a scheduler that runs the furthest-out character first,
    /// which presents as a hang rather than as a wrong order.
    #[test]
    fn the_soonest_character_is_due_first() {
        let s = sched();
        // Staggered by id, so 1 is due before 300.
        s.wake(1, 0, 0);
        s.wake(300, 0, 0);
        let ready = s.due_now(1_000_000);
        assert_eq!(ready.first(), Some(&1), "the heap ran the later one first");
    }

    /// A busy character drains everything at once — one better-informed step,
    /// not several thrashing ones.
    #[test]
    fn a_tick_drains_the_whole_batch() {
        let s = sched();
        s.wake(1, 0, 0);
        for i in 0..5 {
            s.deliver(1, 0, Salience::NORMAL, say(&format!("event {i}")));
        }
        let rec = s
            .tick(1, 0, 0, |events, _| {
                assert_eq!(events.len(), 5, "the tick did not drain the batch");
                vec!["waits".into()]
            })
            .expect("ticked");
        assert_eq!(rec.perceived.len(), 5);
        assert_eq!(rec.inbox_after, 0);
        assert_eq!(s.census()[0].readiness, Readiness::Blocked);
    }

    /// "Nothing arrived" must not mean "dead forever": the heartbeat ticks on a
    /// synthetic event rather than on emptiness.
    #[test]
    fn a_quiet_character_still_thinks_on_its_heartbeat() {
        let s = sched();
        s.wake(1, 0, 0);
        let rec = s
            .tick(1, 0, 0, |events, _| {
                assert_eq!(events.len(), 1);
                assert!(matches!(events[0].kind, EventKind::Heartbeat));
                vec![]
            })
            .expect("ticked");
        assert_eq!(rec.perceived.len(), 1);
        assert_eq!(rec.cause, Readiness::Blocked);
    }

    /// Alertness with no combat branch: something happens, the metabolism
    /// tightens, and it decays back rather than snapping.
    #[test]
    fn alertness_tightens_on_a_preempt_and_decays_when_quiet() {
        let s = sched();
        s.wake(1, 0, 0);
        assert_eq!(
            s.census()[0].heartbeat_ms,
            IDLE_HEARTBEAT.as_millis() as u64
        );

        s.deliver(1, 0, Salience::URGENT, say("a bolt"));
        s.tick(1, 0, 0, |_, _| vec![]);
        let alert = s.census()[0].heartbeat_ms;
        assert_eq!(alert, ALERT_HEARTBEAT.as_millis() as u64);

        // Quiet ticks relax it, monotonically, and never past idle.
        let mut prev = alert;
        for _ in 0..20 {
            s.tick(1, 0, 0, |_, _| vec![]);
            let now = s.census()[0].heartbeat_ms;
            assert!(now >= prev, "alertness went the wrong way");
            prev = now;
        }
        assert_eq!(prev, IDLE_HEARTBEAT.as_millis() as u64, "never settled");
    }

    /// A character with work in front of it never goes as quiet as one that is
    /// only reacting. The relax curve is the same; how far it is allowed to run
    /// is not.
    #[test]
    fn a_working_character_never_settles_as_far_as_an_ambient_one() {
        let s = sched();
        s.wake(1, 0, 0);
        s.wake(2, 0, 0);
        assert!(s.set_pace(2, Pace::WORKING, 0));

        s.deliver(1, 0, Salience::URGENT, say("a bolt"));
        s.deliver(2, 0, Salience::URGENT, say("a bolt"));
        s.tick(1, 0, 0, |_, _| vec![]);
        s.tick(2, 0, 0, |_, _| vec![]);

        for _ in 0..20 {
            s.tick(1, 0, 0, |_, _| vec![]);
            s.tick(2, 0, 0, |_, _| vec![]);
        }
        let beat = |id: u64| {
            s.census()
                .into_iter()
                .find(|c| c.npc_id == id)
                .expect("in the cast")
                .heartbeat_ms
        };
        assert_eq!(beat(1), IDLE_HEARTBEAT.as_millis() as u64);
        assert_eq!(beat(2), ALERT_HEARTBEAT.as_millis() as u64);
        assert!(beat(2) < beat(1), "the pace floor did nothing");
    }

    #[test]
    fn a_character_is_ambient_until_it_is_told_otherwise() {
        let s = sched();
        s.wake(1, 0, 0);
        assert_eq!(s.pace_of(1), Some(Pace::AMBIENT));
        assert_eq!(s.pace_of(2), None, "an absent character has a pace");
        assert!(
            !s.set_pace(2, Pace::WORKING, 0),
            "paced an absent character"
        );
    }

    /// Quickening has to take effect now. A Maker given its working pace while
    /// settled two minutes out would sit through the wait it was just told it
    /// may not have — the exact state the pace exists to prevent.
    #[test]
    fn quickening_a_character_pulls_its_next_thought_forward() {
        let s = sched();
        s.wake(1, 0, 0);
        // Waking staggers a character by its id so a cast does not all think on
        // the same millisecond. That entry has to be taken off before the queue
        // says anything about the pace.
        s.due_now(u64::MAX);
        for _ in 0..20 {
            s.tick(1, 0, 0, |_, _| vec![]);
        }

        let far = IDLE_HEARTBEAT.as_millis() as u64;
        assert!(s.due_now(far / 2).is_empty(), "settled sooner than idle");

        s.set_pace(1, Pace::WORKING, far / 2);
        let soon = far / 2 + ALERT_HEARTBEAT.as_millis() as u64;
        assert_eq!(s.due_now(soon), vec![1], "it kept the wait it was denied");
    }

    /// Slowing a character must not drag it forward, and must not leave it
    /// due at a moment it has already passed.
    #[test]
    fn slowing_a_character_leaves_its_next_thought_where_it_was() {
        let s = sched();
        s.wake(1, 0, 0);
        s.set_pace(1, Pace::WORKING, 0);
        for _ in 0..5 {
            s.tick(1, 0, 0, |_, _| vec![]);
        }
        s.set_pace(1, Pace::AMBIENT, 0);
        assert_eq!(s.pace_of(1), Some(Pace::AMBIENT));
        // It relaxes the rest of the way now that it is allowed to.
        for _ in 0..20 {
            s.tick(1, 0, 0, |_, _| vec![]);
        }
        assert_eq!(
            s.census()[0].heartbeat_ms,
            IDLE_HEARTBEAT.as_millis() as u64
        );
    }

    /// A pace cannot be set faster than the world moves — below that a
    /// character spends decodes discovering nothing has changed.
    #[test]
    fn a_pace_cannot_be_set_faster_than_the_world_moves() {
        assert_eq!(Pace::at_most(Duration::from_millis(1)), Pace::WORKING);
        assert_eq!(Pace::at_most(ALERT_HEARTBEAT), Pace::WORKING);
        assert_eq!(Pace::at_most(IDLE_HEARTBEAT), Pace::AMBIENT);
        assert_eq!(
            Pace::at_most(Duration::from_secs(30)).slowest(),
            Duration::from_secs(30)
        );
    }

    /// A tick with real events must not relax — the character heard something.
    #[test]
    fn a_tick_with_real_events_does_not_relax() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::URGENT, say("a bolt"));
        s.tick(1, 0, 0, |_, _| vec![]);
        let alert = s.census()[0].heartbeat_ms;
        s.deliver(1, 0, Salience::NORMAL, say("footsteps"));
        s.tick(1, 0, 0, |_, _| vec![]);
        assert_eq!(
            s.census()[0].heartbeat_ms,
            alert,
            "relaxed despite an event"
        );
    }

    /// Perception lands in the window before the decode reads it, and acts land
    /// after — the transcript has to be in the order it happened.
    #[test]
    fn perception_precedes_the_decode_and_acts_follow_it() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("the gate groans open"));
        s.tick(1, 0, 100, |_, window| {
            assert_eq!(
                window.len(),
                1,
                "the decode could not see what it perceived"
            );
            vec!["steps back".into()]
        });
        let c = &s.census()[0];
        assert_eq!(c.window_turns, 2, "the act did not land in the window");
    }

    #[test]
    fn a_broadcast_reaches_everyone() {
        let s = sched();
        for id in 1..=4 {
            s.wake(id, 0, 0);
        }
        // The clock is per character: thunder over two worlds happens at two
        // times, and each character's event must carry its own world's instant.
        assert_eq!(
            s.broadcast(|id| id * 100, Salience::NORMAL, say("thunder")),
            4
        );
        for c in s.census() {
            assert_eq!(c.inbox_depth, 1);
        }
    }

    #[test]
    fn the_day_rolls_over_once_and_clears_the_window() {
        use crate::engine::sleep::DAY_MS;
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("yesterday"));
        s.tick(1, 0, 0, |_, _| vec![]);
        assert!(s.census()[0].window_turns > 0);

        assert_eq!(s.roll_day(1, DAY_MS + 5), Some((0, 1)));
        assert_eq!(s.census()[0].window_turns, 0, "the tail survived the day");
        assert_eq!(s.census()[0].day, Some(1));
        assert_eq!(s.roll_day(1, DAY_MS + 6), None, "rolled over twice");
    }

    /// **The gap between two turns of a conversation is not idleness.**
    ///
    /// The bug this is here for: asked as "is there news queued right now", the
    /// answer is *no* in every gap between two things happening — and a
    /// conversation is mostly gaps. Two characters alternated for a hundred
    /// turns, each hearing the other speak and then, in the pause before the
    /// reply, being told that nothing had been asked of it.
    #[test]
    fn a_pause_in_a_conversation_does_not_read_as_a_quiet_character() {
        let s = sched();
        s.wake(1, 0, 0);

        // Somebody speaks at t=0 and the character reads it.
        s.deliver(
            1,
            0,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Maker-02".into(),
                text: "that the redoubt burned twice".into(),
                to: crate::engine::event::Addressed::Room,
            },
        );
        assert_eq!(s.quiet_for(1, 0), Some(0), "unread news is news");
        s.tick(1, 0, 0, |_, _| Vec::new());

        // Its companion is now thinking. Several heartbeats pass with nothing
        // arriving — which is what waiting for a reply looks like from here.
        for gap in [4_000, 8_000, 20_000, 45_000] {
            assert_eq!(
                s.quiet_for(1, gap),
                Some(gap),
                "the clock runs from the last thing said, not from the last drain"
            );
        }

        // The situation and the standing task keep arriving throughout, and
        // neither is something that happened — so neither may reset the clock,
        // or the task would hold its own gate open forever.
        s.deliver(
            1,
            46_000,
            Salience::IDLE,
            EventKind::Situation {
                text: "You are in the green room.".into(),
            },
        );
        s.deliver(
            1,
            46_000,
            Salience::IDLE,
            EventKind::Nudge {
                text: "go and look around".into(),
            },
        );
        assert_eq!(s.quiet_for(1, 50_000), Some(50_000));

        // And a genuinely abandoned character does eventually come back round.
        assert!(s.quiet_for(1, 600_000).unwrap() > 45_000);
    }

    /// **The standing task is a cadence, not a latch.**
    ///
    /// It is not news, so it cannot move the news clock — which left the gate,
    /// once open, open forever. A character nobody was talking to got the task
    /// again on every tick, each copy superseding the last into the most recent
    /// position in the window. Identical instruction, identical context,
    /// identical decode: the character repeated one act until something outside
    /// it changed.
    #[test]
    fn the_standing_task_is_restated_on_a_cadence_rather_than_every_tick() {
        const AFTER: u64 = 90_000;
        // The world clock is an absolute time, not a counter that starts at
        // zero with the character — so a fresh one is already long past any
        // threshold, which is what makes it due immediately.
        const T0: u64 = 1_000_000;
        let s = sched();
        s.wake(1, 0, T0);

        // Never told anything: due at once, which is right — a character with
        // nothing ever asked of it is the one the task exists for.
        assert!(s.nudge_due(1, T0, AFTER));
        s.deliver(
            1,
            T0,
            Salience::IDLE,
            EventKind::Nudge {
                text: "go and look around".into(),
            },
        );
        s.tick(1, 0, T0, |_, _| Vec::new());

        // Still nothing happening — and it is NOT due again on the next tick,
        // nor the one after, nor a minute later.
        for gap in [4_000, 8_000, 45_000, 89_000] {
            assert!(
                !s.nudge_due(1, T0 + gap, AFTER),
                "restated {gap}ms after the last one"
            );
        }
        // A full cadence later, it is.
        assert!(s.nudge_due(1, T0 + AFTER, AFTER));

        // And news resets the other clock, so a character that gets spoken to
        // is not handed the task on the very next quiet tick.
        s.deliver(
            1,
            T0 + AFTER,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Maker-02".into(),
                text: "that the redoubt burned twice".into(),
                to: crate::engine::event::Addressed::Room,
            },
        );
        s.tick(1, 0, T0 + AFTER, |_, _| Vec::new());
        assert!(
            !s.nudge_due(1, T0 + AFTER + 10_000, AFTER),
            "ten seconds after being spoken to"
        );
    }

    /// **A waiting character stops thinking**, which is the whole difference
    /// between this and the act it replaced.
    ///
    /// The old `wait` left the heartbeat alone, so the character woke four
    /// seconds later and decided to wait again, and again — a busy-loop wearing
    /// the word "wait". Here the deadline *is* the heartbeat: nothing is due
    /// until the wait is answered or its patience runs out.
    #[test]
    fn a_wait_takes_the_character_out_of_the_schedule() {
        let s = sched();
        s.wake(1, 0, 0);
        assert!(s.due_now(10_000).contains(&1), "a woken character is due");

        s.begin_waiting(
            1,
            Waiting::new(crate::engine::waiting::Kind::SomeoneSpeaks, None, 0),
            Vec::new(),
            0,
        );
        assert!(
            !s.due_now(10_000).contains(&1),
            "it kept its heartbeat while waiting"
        );
        assert!(s.waiting(1).is_some());
    }

    /// And the answer wakes it **now**, not on the next heartbeat.
    ///
    /// A character replying four seconds after the question, with the answer
    /// buried in whatever else arrived meanwhile, is the batching working
    /// against the one case where the character said in advance what it cared
    /// about.
    #[test]
    fn answering_a_wait_wakes_the_character_immediately() {
        let s = sched();
        s.wake(1, 0, 0);
        s.begin_waiting(
            1,
            Waiting::new(crate::engine::waiting::Kind::SomeoneSpeaks, None, 0),
            Vec::new(),
            0,
        );
        assert!(!s.due_now(10_000).contains(&1));

        assert!(s.answer_wait(1), "there was a wait to answer");
        assert!(s.due_now(0).contains(&1), "it did not wake at once");
        assert!(s.waiting(1).is_none(), "the wait outlived its answer");
        // Answering twice is not an error, it is a no-op — two things can
        // notice the same speech on the same pass.
        assert!(!s.answer_wait(1));
    }

    /// Speech is matched by speaker where it arrives structured, so a wait on
    /// one person is not ended by somebody else talking.
    #[test]
    fn a_named_wait_is_not_ended_by_the_wrong_voice() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(
            1,
            0,
            Salience::NORMAL,
            EventKind::Speech {
                speaker: "Orion Vance".into(),
                text: "something".into(),
                to: crate::engine::event::Addressed::Room,
            },
        );
        assert!(s.heard_speech(1, |sp| sp == "Orion Vance"));
        assert!(!s.heard_speech(1, |sp| sp == "Perrin Vastwood"));
    }

    /// The feed is an instrument, not a log. It must stay bounded under load.
    #[test]
    fn the_recent_feed_is_bounded() {
        let s = Scheduler::new(4);
        s.wake(1, 0, 0);
        for _ in 0..20 {
            s.tick(1, 0, 0, |_, _| vec![]);
        }
        let r = s.recent(100);
        assert_eq!(r.len(), 4);
        // Newest last, and contiguous.
        assert!(r.windows(2).all(|w| w[1].tick == w[0].tick + 1));
    }

    /// **An id past 2^53 must survive the wire.**
    ///
    /// Character ids use the whole `u64` range. As a JSON number a large one
    /// arrives at the console rounded, so the same character would carry
    /// different ids in the feed and the census — the cast strip colouring one
    /// row and the feed another, and a click filtering to a character that does
    /// not exist. Invisible until an id happens to be large, which is why it is
    /// asserted against a real one.
    #[test]
    fn a_large_character_id_reaches_the_wire_exactly() {
        let big = 6_817_662_845_163_923_144u64;
        assert!(big > (1u64 << 53), "the test's id is not past the danger");

        let s = sched();
        s.wake(big, 0, 0);
        s.tick(big, 0, 0, |_, _| vec![]);

        let json = serde_json::to_string(&s.recent(1)[0]).unwrap();
        assert!(
            json.contains(&format!("\"npc_id\":\"{big}\"")),
            "a tick's id is not a string: {json}"
        );
        let census = serde_json::to_string(&s.census()[0]).unwrap();
        assert!(
            census.contains(&format!("\"npc_id\":\"{big}\"")),
            "a census row's id is not a string: {census}"
        );

        // And it round-trips: the exact value comes back, not a rounded one.
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["npc_id"].as_str().unwrap().parse::<u64>().unwrap(), big);
    }

    /// A retired character's stale heap entries must not resurrect it.
    #[test]
    fn a_retired_character_is_not_scheduled() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::URGENT, say("x"));
        s.retire(1);
        assert!(s.due_now(u64::MAX).is_empty());
        assert!(s.tick(1, 0, 0, |_, _| vec![]).is_none());
    }

    /// Due lists must not contain a character twice — a preempt pushes a second
    /// heap entry, and ticking the same character twice in one pass would drain
    /// an inbox that the first tick already emptied.
    #[test]
    fn a_preempted_character_appears_once_in_the_due_list() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::URGENT, say("a"));
        s.deliver(1, 0, Salience::URGENT, say("b"));
        let due = s.due_now(u64::MAX);
        assert_eq!(due.iter().filter(|id| **id == 1).count(), 1);
    }
}
