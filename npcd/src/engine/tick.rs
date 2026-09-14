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
//!
//! # Each character waits on its own inbox
//!
//! There is no shared due-queue and nothing polls. Every character is an async
//! task that sleeps on [`Scheduler::wait_due`] — a wait on its own inbox's
//! waker, bounded by its own deadline — so an idle cast costs a parked future
//! apiece and a delivery wakes exactly the character it names, at the moment it
//! lands rather than at the next poll.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::Notify;

use crate::engine::event::{Event, EventKind, Salience};
use crate::engine::sleep::{DayAction, DayTracker};
use crate::engine::window::Window;

/// The slowest a wholly idle character thinks. Long, because a character with
/// nothing happening genuinely has nothing to think about, and the cost of a
/// wake is a decode.
pub const IDLE_HEARTBEAT: Duration = Duration::from_secs(120);

/// The fastest a character settles back to after something happened.
pub const ALERT_HEARTBEAT: Duration = Duration::from_secs(4);

/// A character that is not scheduled to think at all.
///
/// **The ordinary state of a quiet character**, and the whole of what "no idle
/// events" means: nothing is due, so nothing thinks, until the world puts
/// something in its inbox. Spelled as a deadline past any clock rather than as
/// an `Option`, so the one comparison in [`Scheduler::wait_due`] answers both
/// "is it time" and "is it scheduled".
const NEVER: u64 = u64::MAX;

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
    /// Inbox empty. Costs nothing, and is where a character spends nearly all
    /// of its life.
    ///
    /// **It was called `Blocked`, and that was a lie an instrument told.** Every
    /// view of the cast reported the healthiest possible state with a word that
    /// means stuck — and since idle ticks were removed this is the *resting*
    /// state of every quiet character, not a rare one. It read as a fault to
    /// everybody who saw it, including the people who wrote it.
    ///
    /// Its doc named a heartbeat too. There is no heartbeat: a character with an
    /// empty inbox is waiting on the world, not on a clock.
    Quiet,
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
    /// When this character last did something, measured from
    /// [`Scheduler::born`].
    ///
    /// Only a tick that produced **acts** sets it: a character can think and
    /// decide to do nothing, and the question this answers is how long it has
    /// been since it last *acted*, not since it last ran.
    ///
    /// `None` until the first one, which is a different fact from "a long time
    /// ago" and is shown as such.
    last_act_at: Option<Duration>,
    /// Set when the tick that is still running has already decided when the
    /// next one is — by pausing, or by asking the world something it must come
    /// back to read.
    ///
    /// It exists only to survive the line at the end of [`Scheduler::tick`],
    /// which otherwise puts the deadline back to `NEVER` on the empty queue
    /// both of those cases have by definition. Cleared there.
    scheduled: bool,
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
    /// What the character's own task sleeps on — see [`Scheduler::wait_due`].
    /// A delivery, a cut-short pause, and a retirement each notify it; the
    /// permit is stored if the task is mid-think, so a wake sent while the
    /// character is busy is read the moment it returns to its wait.
    waker: Arc<Notify>,
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
            last_act_at: None,
            scheduled: false,
            last_news_ms: 0,
            last_nudge_ms: 0,
            waker: Arc::new(Notify::new()),
        }
    }

    /// Stop for a while and then come back to it.
    ///
    /// **The whole of what a pause is.** It moves the next think out by `for_ms`
    /// and does nothing else — no condition to satisfy, no patience to run out,
    /// nothing to answer. Anything arriving meanwhile still wakes the character
    /// at once, because that is already true of every character with an empty
    /// queue, and it is why the pause needs no rousing rule of its own.
    ///
    /// # How long is the act's to say, not this function's
    ///
    /// It comes from [`crate::engine::cooldown::Cost::stall`], where it sits
    /// beside the same act's cooldown so the two are chosen against each other.
    /// `Runtime::arm_pause` reads it and passes it here; nothing in this file
    /// restates the number, so there is no second copy to drift.
    ///
    /// **Chosen by the character, not imposed on it** — the difference between
    /// this and the idle heartbeat that was removed. A heartbeat wakes
    /// everybody on a timer whether or not there is anything to think about; a
    /// stall only ever runs because a character looked at its situation and
    /// decided there was nothing to do this second.
    ///
    /// What the table charges `reflect` is two minutes, which is what the old
    /// wait's patience was and for the same reason: long enough that a
    /// companion who is thinking or walking a stop is worth waiting for, short
    /// enough that a character which stopped when it should not have is doing
    /// something else within the minute after. It is also the ceiling on how
    /// often a character can stop in a loop — at two minutes, thirty decodes an
    /// hour spent standing still.
    pub fn pause_for(&mut self, now_ms: u64, for_ms: u64) {
        self.due_at = now_ms.saturating_add(for_ms);
        self.scheduled = true;
    }

    /// Come straight back, because the last act answered with something.
    ///
    /// **An act that told the character something has to be followed by a turn
    /// in which it can use it.** A Maker that reads the muster board gets the
    /// board's contents in its outcome and nowhere else — nothing in the world
    /// perceives a document being read, so no sweep will ever deliver it — and
    /// then went to sleep with `due_at` at `NEVER`. It had the answer written
    /// into its window and no turn in which to act on it, so the next thing it
    /// did, whenever the room finally woke it, was read the board again.
    ///
    /// Narrow on purpose: only the acts in [`crate::engine::body::ANSWERS`].
    /// Giving *every* act a free follow-up is the treadmill — a character that
    /// speaks and is immediately asked again will speak again, into the same
    /// silence, and its own window fills with its own voice until the nucleus
    /// collapses. Asking the world a question is different from talking: the
    /// world said something back.
    pub fn think_again(&mut self) {
        self.due_at = 0;
        self.scheduled = true;
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
            Readiness::Quiet
        } else {
            Readiness::Pending
        }
    }

    /// Accept an event. Never refuses: the mind design is explicit that a filter
    /// dropping evidence before it lands makes a delusion permanent.
    ///
    /// **Every arrival is a wake.** Nothing polls, so an event that does not
    /// rouse the character's task is an event nobody ever reads — it would sit
    /// in the queue until something louder happened to arrive behind it. The
    /// salience still decides how *urgently* the character reads it
    /// (`readiness` keeps the Pulse feed's "why did it wake" column); a
    /// character mid-think elsewhere simply finds the wake stored on its waker
    /// when it comes back.
    pub fn push(&mut self, event: Event) {
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
        // **Anything arriving cuts a pause short**, and nearly all of that is
        // free: a pause is only a due time, and `deliver` moves the character's
        // due time to now. That is the whole benefit over the subscription it
        // replaced — being messaged mid-pause is read at once rather than two
        // minutes later.
        //
        // The one line it does cost is this. `scheduled` tells the end of a tick
        // to leave the deadline alone, and a pause that has been cut short no
        // longer has a deadline worth keeping — so without clearing it here the
        // *next* tick skips its reset too, leaves `due_at` at zero, and the
        // character ticks flat out on an empty queue. Which is precisely the
        // busy-loop the old wait's rousing rule existed to close, arriving by a
        // different door.
        self.scheduled = false;
        if preempts {
            self.preempted = true;
            // Something happened. Tighten the metabolism — this is the whole of
            // the alertness model, and it needs no combat branch anywhere.
            self.heartbeat = ALERT_HEARTBEAT;
        }
        self.queue.push_back(event);
    }

    /// Take everything waiting. A busy character drains a fat batch — one
    /// better-informed thinking step rather than several thrashing ones.
    pub fn drain(&mut self) -> Vec<Event> {
        self.preempted = false;
        self.queue.drain(..).collect()
    }

    /// There is no `relax`, because there are no quiet ticks to relax on.
    ///
    /// Alertness used to decay over ticks where nothing arrived — which was the
    /// only kind of tick a quiet character had. With those gone the decay has
    /// nothing to run on: a character is either answering something or it is
    /// not scheduled at all. `heartbeat` now records only that something
    /// recently happened, which is what the Pulse feed reports.
    ///
    /// How slow this character may become, and how slow it currently is.
    pub fn pace(&self) -> Pace {
        self.pace
    }
}

/// A tick's first two phases, handed across the decode to [`Scheduler::end_tick`]:
/// what was drained, why the character woke, and the window snapshot the decode
/// reads. Anything delivered while the decode runs lands in the real window and
/// is seen by the next tick — the same guarantee as an event that arrived a
/// moment after the drain.
pub struct TickStart {
    pub events: Vec<Event>,
    /// The raw events as prose. The record reports the narrator's curated
    /// rendering instead when one is produced (the tick loop overrides this
    /// before `end_tick`); this is the fallback for a tick with nothing to
    /// narrate. See `crate::engine::narrator`.
    pub perceived: Vec<String>,
    pub cause: Readiness,
    pub window: Window,
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
    /// What it perceived, as prose — the narrator's curated third-person
    /// rendering of the events, which is what went to the model, or the raw
    /// events on a tick with nothing to narrate.
    pub perceived: Vec<String>,
    /// Why it ran now.
    pub cause: Readiness,
    /// Acts that came out, rendered. Empty until the decode path is live.
    pub acts: Vec<String>,
    /// Heartbeat after this tick, in ms — the character's current alertness.
    pub heartbeat_ms: u64,
    pub inbox_after: usize,
    /// When this tick ran, against [`Scheduler::born`]. Never serialised: it is
    /// the raw material for `ms_ago`, and a duration on the wire would only
    /// hand the reader the clock problem this field exists to solve.
    #[serde(skip)]
    at: Duration,
    /// How long ago this tick ran, in milliseconds.
    ///
    /// **Filled in by [`Scheduler::recent`], not at construction**, because a
    /// record's age is not a property of the record — it changes every second it
    /// sits in the ring. It is an age rather than a timestamp for the same
    /// reason [`Census::acted_ms_ago`] is: the reader is a browser on another
    /// machine, and `at_ms` beside it is the *driver's* clock, which starts
    /// minutes after this one.
    pub ms_ago: u64,
}

/// The scheduler: every character's loop state, and the wakes that run them.
pub struct Scheduler {
    inboxes: Mutex<HashMap<u64, Inbox>>,
    /// Monotonic event sequence, shared by every character so the Pulse view can
    /// order arrivals across the whole cast.
    seq: AtomicU64,
    /// Monotonic tick counter, same reason.
    ticks: AtomicU64,
    /// Recent ticks, newest last. Bounded — this is an instrument, not a log.
    recent: Mutex<VecDeque<TickRecord>>,
    recent_cap: usize,
    /// The scheduler's own clock, for answering *how long ago*.
    ///
    /// Its own rather than the driver's: `drive` starts an `Instant` when the
    /// loop begins, minutes after `Runtime::new`, so a duration measured against
    /// one and read against the other is out by however long loading took. And
    /// its own rather than the wall clock, because the answer is handed to a
    /// browser whose clock is not this machine's — an age computed here is
    /// right whatever the reader's clock says, where a timestamp is not.
    born: Instant,
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
            seq: AtomicU64::new(0),
            ticks: AtomicU64::new(0),
            recent: Mutex::new(VecDeque::new()),
            recent_cap: recent_cap.max(1),
            born: Instant::now(),
        }
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

    /// Bring a character straight back, because its last act answered with
    /// something. See [`Inbox::think_again`].
    pub fn think_again(&self, npc_id: u64) -> bool {
        let waker = {
            let mut g = self.inboxes.lock().unwrap();
            let Some(i) = g.get_mut(&npc_id) else {
                return false;
            };
            i.think_again();
            Arc::clone(&i.waker)
        };
        waker.notify_one();
        true
    }

    /// Complete an act's row with the result that came back after it was
    /// recorded — in the character's window and in the Pulse ring both, so the
    /// feed and the transcript say the same thing. See [`Window::amend_npc`].
    ///
    /// Returns whether the row was found in the window.
    pub fn amend_act(&self, npc_id: u64, from: &str, to: String) -> bool {
        let found = self
            .inboxes
            .lock()
            .unwrap()
            .get_mut(&npc_id)
            .is_some_and(|i| i.window.amend_npc(from, to.clone()));
        let mut recent = self.recent.lock().unwrap();
        if let Some(act) = recent
            .iter_mut()
            .rev()
            .filter(|r| r.npc_id == npc_id)
            .flat_map(|r| r.acts.iter_mut())
            .find(|a| a.as_str() == from)
        {
            *act = to;
        }
        found
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

    /// Deliver an event to a character. `false` if there is no such character.
    pub fn deliver(&self, npc_id: u64, world_ms: u64, salience: Salience, kind: EventKind) -> bool {
        let seq = self.seq.fetch_add(1, AtomicOrdering::Relaxed);
        let event = Event::new(seq, world_ms, salience, kind);
        let waker = {
            let mut inboxes = self.inboxes.lock().unwrap();
            let Some(inbox) = inboxes.get_mut(&npc_id) else {
                return false;
            };
            inbox.push(event);
            // Due immediately: preempted whatever it planned, or roused out of
            // a wait. The waker coalesces — a burst of twenty arrivals stores
            // one permit, and the one wake drains the whole burst. An arrival
            // *during* a decode stores its permit too, so the character is
            // re-woken the moment its task returns to its wait.
            inbox.due_at = 0;
            Arc::clone(&inbox.waker)
        };
        waker.notify_one();
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

    /// The scheduler's own monotonic clock, in milliseconds — the `now_ms`
    /// every character task ticks against, so a tick's `at_ms` and its age in
    /// the Pulse ring are measured on one clock.
    pub fn now_ms(&self) -> u64 {
        self.born.elapsed().as_millis() as u64
    }

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
    pub async fn wait_due(&self, npc_id: u64) -> bool {
        loop {
            let (due_at, waker) = {
                let inboxes = self.inboxes.lock().unwrap();
                let Some(inbox) = inboxes.get(&npc_id) else {
                    return false;
                };
                (inbox.due_at, Arc::clone(&inbox.waker))
            };
            let now = self.now_ms();
            if due_at <= now {
                return true;
            }
            // Registered before the deadline check could race a notify: a
            // permit stored between the read above and this await completes
            // the wait immediately.
            let woken = waker.notified();
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
    /// tests that drive the scheduler without standing the tasks up.
    pub fn due_now(&self, now_ms: u64) -> Vec<u64> {
        let inboxes = self.inboxes.lock().unwrap();
        let mut due: Vec<(u64, u64)> = inboxes
            .values()
            .filter(|i| i.due_at <= now_ms)
            .map(|i| (i.due_at, i.npc_id))
            .collect();
        due.sort_unstable();
        due.into_iter().map(|(_, id)| id).collect()
    }

    /// Whether this one character is due at `now_ms` — [`Self::due_now`]
    /// narrowed to the inbox a test is watching.
    #[cfg(test)]
    fn is_due(&self, npc_id: u64, now_ms: u64) -> bool {
        self.inboxes
            .lock()
            .unwrap()
            .get(&npc_id)
            .is_some_and(|i| i.due_at <= now_ms)
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
    pub fn begin_tick(&self, npc_id: u64) -> Option<TickStart> {
        // The drain takes the lock and gives it straight back.
        let (events, cause) = {
            let mut inboxes = self.inboxes.lock().unwrap();
            let inbox = inboxes.get_mut(&npc_id)?;
            // Readiness FIRST. `drain` clears the preempt flag, so reading it
            // afterwards reports `Blocked` for every tick and the Pulse feed
            // loses the one column that says why a character woke.
            let cause = inbox.readiness();
            (inbox.drain(), cause)
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
        if events.is_empty() {
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
            for e in &events {
                inbox
                    .window
                    .push_world(e.prose(), e.at_ms, e.kind.replaces());
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
                inbox.due_at = match inbox.queue.is_empty() {
                    true => NEVER,
                    false => 0,
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
        // Once, outside the walk, so every row in one page is measured against
        // the same instant.
        let now = self.born.elapsed();
        let r = self.recent.lock().unwrap();
        let skip = r.len().saturating_sub(limit);
        r.iter()
            .skip(skip)
            .cloned()
            .map(|mut t| {
                t.ms_ago = now.saturating_sub(t.at).as_millis() as u64;
                t
            })
            .collect()
    }

    /// A snapshot of every character's loop state.
    pub fn census(&self) -> Vec<Census> {
        // Read once, outside the map, so every row in one census is measured
        // against the same instant — otherwise a large cast reports ages that
        // disagree by however long the walk took.
        let now = self.born.elapsed();
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
                acted_ms_ago: i
                    .last_act_at
                    // `saturating_sub`: a tick that landed between the reading
                    // above and this line would otherwise be an age in the
                    // future, which renders as a very large number.
                    .map(|at| now.saturating_sub(at).as_millis() as u64),
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
    /// How long ago this character last **acted**, in milliseconds.
    ///
    /// An age rather than a timestamp, computed here against
    /// [`Scheduler::born`]. A timestamp would have to be read against a clock,
    /// and the reader is a browser on somebody else's machine — so "at
    /// 23:51:02" is a question about clock skew where "nine seconds ago" is
    /// just true.
    ///
    /// `None` for a character that has not acted yet, which reads as *never*
    /// rather than as a very large number.
    pub acted_ms_ago: Option<u64>,
}

/// Shared handle.
pub type Shared = Arc<Scheduler>;

#[cfg(test)]
mod tests {
    use super::*;

    fn sched() -> Scheduler {
        Scheduler::new(8)
    }

    /// A stall long enough to be visible against the heartbeat.
    ///
    /// **A number of this module's own, on purpose.** What is under test here
    /// is the scheduler's stall *mechanism* — that a stalled character leaves
    /// the schedule, comes back on time, and is woken early by anything that
    /// arrives. Which acts stall, and for how long, is a question for the cost
    /// table and is answered by its own tests. Reading the table from here
    /// coupled the two: when `reflect` stopped stalling, seven scheduler tests
    /// failed for a reason that had nothing to do with scheduling.
    fn pause_ms() -> u64 {
        30_000
    }

    fn say(text: &str) -> EventKind {
        EventKind::Description { text: text.into() }
    }

    /// **A character that runs out of things to think about must stay
    /// reachable.**
    ///
    /// The wedge this closes, in the order it happens live:
    ///
    /// 1. An act that answered calls `think_again` — `due_at = 0`, `scheduled`
    ///    set — and nothing new arrives, so the tick ends with an empty queue
    ///    and `due_at` still `0`.
    /// 2. The next pass pops the character's heap entry and ticks it. The drain
    ///    yields nothing, so the tick returns early.
    /// 3. That early return used to leave `due_at` at `0` with the entry now
    ///    consumed — so `deliver`'s `already_queued` check (`due_at == 0`, read
    ///    as "an entry is standing in the heap") suppressed every future push.
    ///
    /// From there the character is unschedulable for the life of the daemon:
    /// events accumulate, `readiness` reports `Preempted`, and nothing runs it.
    /// Measured on the live daemon — three characters wedged inside a minute,
    /// inboxes past sixty, and an URGENT broadcast could not move them.
    #[test]
    fn a_tick_that_finds_nothing_leaves_the_character_reachable() {
        let s = sched();
        s.wake(1, 0, 0);

        // An act that answered: due now, and no event behind it.
        s.deliver(1, 0, Salience::NORMAL, say("the board says three names"));
        assert!(s.is_due(1, 1_000));
        s.tick(1, 1_000, 0, |_, _| vec!["read the board".into()]);
        s.think_again(1);

        // The pass that finds an empty queue.
        assert!(s.is_due(1, 2_000), "it was not even due");
        assert!(
            s.tick(1, 2_000, 0, |_, _| vec![]).is_none(),
            "there was nothing to drain, so there is no record"
        );

        // The character must now be schedulable again by an ordinary arrival.
        s.deliver(1, 0, Salience::NORMAL, say("somebody comes in"));
        assert!(
            s.is_due(1, 3_000),
            "the character was left claiming to be queued while nothing could \
             ever run it — every later arrival is silently dropped on the floor"
        );
        let rec = s.tick(1, 3_000, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.perceived.len(), 1);
    }

    /// **A character thinking somewhere else loses nothing that arrives
    /// meanwhile.** Its task is awaiting a reflection's answer rather than
    /// sitting in [`Scheduler::wait_due`], so nothing drains its inbox — and
    /// every arrival stores a wake on its waker, so the moment the task
    /// returns to its wait it thinks at once, on all of it.
    #[tokio::test]
    async fn what_arrives_while_a_character_thinks_elsewhere_is_read_on_return() {
        let s = sched();
        s.wake(1, 0, 0);

        // The character's task is elsewhere: nothing calls wait_due or tick.
        s.deliver(1, 0, Salience::NORMAL, say("somebody comes in"));
        s.deliver(1, 0, Salience::URGENT, say("an alarm"));
        assert_eq!(s.inbox_depth(1), Some(2), "something was dropped");

        // Back from the other conversation: the stored wake completes at once…
        let due = tokio::time::timeout(Duration::from_secs(1), s.wait_due(1))
            .await
            .expect("the stored wake was lost");
        assert!(due);
        // …and the one turn reads everything that queued up meanwhile.
        let now = s.now_ms();
        let rec = s.tick(1, now, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.perceived.len(), 2, "on everything it missed");
    }

    /// **An act recorded before its result is completed with it.** A reflect
    /// goes into the window as the act alone and is answered seconds later by
    /// its own conversation; the row has to end up reading what came back, not
    /// stay a call with nothing after it.
    #[test]
    fn an_act_recorded_before_its_result_is_completed_with_it() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("a box sags"));
        assert!(s.is_due(1, 1_000));
        s.tick(1, 1_000, 0, |_, _| vec!["reflect — it gave up".into()]);

        let done = "reflect — it gave up → I keep counting what gives up".to_string();
        assert!(s.amend_act(1, "reflect — it gave up", done.clone()));
        let window: Vec<String> = s
            .window_of(1, |w| w.turns().map(|t| t.text.clone()).collect())
            .unwrap();
        assert!(window.contains(&done), "{window:?}");
        assert!(
            !window.iter().any(|t| t == "reflect — it gave up"),
            "{window:?}"
        );

        assert!(
            !s.amend_act(1, "nothing like it", "x".into()),
            "nothing to amend"
        );
    }

    /// The neighbouring path, which does **not** wedge — and is here to keep it
    /// that way.
    ///
    /// A pause leaves `due_at` at a real deadline rather than at `0`, so
    /// `deliver`'s `already_queued` check reads false and an arrival still
    /// queues it. That is the whole reason `think_again` was the one that broke:
    /// the two differ only in the value they park `due_at` at, and one of those
    /// values happens to be the sentinel the queue check keys on. Anything that
    /// makes a pause park at zero brings the wedge back here.
    #[test]
    fn a_character_that_wakes_from_a_pause_to_an_empty_queue_stays_reachable() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("a light goes out"));
        s.tick(1, 1_000, 0, |_, _| vec!["reflect".into()]);
        s.pause_for(1, 1_000, pause_ms());

        // Its deadline comes round with nothing waiting.
        assert!(s.is_due(1, 1_000 + pause_ms() + 1));
        assert!(s
            .tick(1, 1_000 + pause_ms() + 1, 0, |_, _| vec![])
            .is_none());

        s.deliver(1, 0, Salience::URGENT, say("a door slams"));
        assert!(
            s.is_due(1, 1_000 + pause_ms() + 2_000),
            "an urgent arrival could not wake it"
        );
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

    /// An empty inbox is **quiet**, not blocked — the state that costs nothing
    /// and the one a character is in nearly all the time.
    #[test]
    fn an_empty_inbox_reads_quiet() {
        let s = sched();
        s.wake(1, 0, 0);
        assert_eq!(s.census()[0].readiness, Readiness::Quiet);
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
        assert!(!s.is_due(1, 10_000), "not due yet");
        s.deliver(1, 0, Salience::URGENT, say("the beam gives"));
        assert_eq!(s.census()[0].readiness, Readiness::Preempted);
        assert!(s.is_due(1, 10_000));
    }

    /// **A burst of urgent arrivals is one wake, not one wake each.** The
    /// waker coalesces — twenty deliveries store one permit — and the single
    /// tick that permit buys drains the whole burst.
    #[tokio::test]
    async fn a_burst_of_preempting_arrivals_is_one_wake_that_drains_it_all() {
        let s = sched();
        s.wake(1, 10_000, 0);
        for _ in 0..20 {
            s.deliver(1, 0, Salience::URGENT, say("the beam gives"));
        }

        assert!(tokio::time::timeout(Duration::from_secs(1), s.wait_due(1))
            .await
            .expect("the burst stored no wake"));
        let rec = s.tick(1, 10_000, 0, |_, _| vec![]).expect("ticked");
        assert_eq!(rec.perceived.len(), 20);

        // An arrival during the decode is a different case and must still wake
        // the character: its permit was stored while the task was ticking, so
        // the next wait completes at once.
        s.deliver(1, 0, Salience::URGENT, say("and again"));
        assert!(s.is_due(1, 10_000));
        assert!(tokio::time::timeout(Duration::from_secs(1), s.wait_due(1))
            .await
            .expect("the mid-decode arrival's wake was lost"));
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
        assert_eq!(s.census()[0].readiness, Readiness::Quiet);
    }

    /// **A character with nothing to react to does not think.**
    ///
    /// It used to: an empty inbox synthesised a heartbeat event and the
    /// character took a turn on it, so that "nothing arrived" could not mean
    /// "dead forever". The cost was not the decode, it was what the decode
    /// wrote — an act, appended to the character's own window, every beat. An
    /// embodied character is floored at four seconds, so a quiet room filled a
    /// twenty-four-turn window with copies of one gesture inside two minutes,
    /// and after that the next token was certain and the character could not
    /// say anything else.
    #[test]
    fn a_quiet_character_does_not_think_at_all() {
        let s = sched();
        s.wake(1, 0, 0);
        assert!(
            s.tick(1, 0, 0, |_, _| panic!(
                "thought with nothing to think about"
            ))
            .is_none(),
            "an empty inbox produced a turn"
        );
    }

    /// And it is not *scheduled*, either — nothing to serve, so nothing is due
    /// however long you wait. Being quiet costs no decodes and no heap.
    #[test]
    fn a_quiet_character_is_never_due() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("something"));
        assert!(s.is_due(1, 0));
        s.tick(1, 0, 0, |_, _| vec!["speak — yes".into()])
            .expect("ticked");

        for at in [0, 10_000, 600_000, u64::MAX - 1] {
            assert!(
                !s.is_due(1, at),
                "a character with an empty inbox came due at {at}"
            );
        }
    }

    /// Alertness with no combat branch: something happens, the metabolism
    /// tightens, and it decays back rather than snapping.
    #[test]
    fn alertness_tightens_on_a_preempt() {
        let s = sched();
        s.wake(1, 0, 0);
        assert_eq!(
            s.census()[0].heartbeat_ms,
            IDLE_HEARTBEAT.as_millis() as u64
        );

        s.deliver(1, 0, Salience::URGENT, say("a bolt"));
        s.tick(1, 0, 0, |_, _| vec![]);
        assert_eq!(
            s.census()[0].heartbeat_ms,
            ALERT_HEARTBEAT.as_millis() as u64
        );
        // It does not decay from here. Alertness used to relax over the ticks a
        // quiet character took, and a quiet character no longer takes any — so
        // what this number records is that something recently happened, which
        // is what the Pulse feed shows. Nothing schedules from it.
    }

    /// **A pace no longer schedules anything**, and this is what that means.
    ///
    /// `Pace::WORKING` floored an embodied character's heartbeat at four
    /// seconds and never let it relax, so *being in a world* meant thinking
    /// every four seconds whether or not anything had happened. That floor was
    /// the engine of the repetition: not the sampler, not the grammar — a
    /// character scheduled to speak fifteen times a minute into a room where
    /// nothing was going on.
    ///
    /// Neither pace makes a character due now. Whether it thinks is decided by
    /// its inbox, and both of these are empty.
    #[test]
    fn no_pace_makes_a_quiet_character_think() {
        let s = sched();
        s.wake(1, 0, 0);
        s.wake(2, 0, 0);
        assert!(s.set_pace(2, Pace::WORKING, 0));

        for at in [0, 4_000, 120_000, 600_000] {
            assert!(!s.is_due(1, at), "the ambient one was due at {at}");
            assert!(
                !s.is_due(2, at),
                "a working pace still put a quiet character on the schedule at {at}"
            );
        }
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

    /// **Changing a pace must not conjure a thought.**
    ///
    /// It used to pull a character forward to the new floor, which is how
    /// `embody` — which sets `WORKING` — put every character in a world onto a
    /// four-second treadmill. Setting a pace now says how alert the character
    /// reads as and nothing else.
    #[test]
    fn changing_a_pace_does_not_schedule_a_thought() {
        let s = sched();
        s.wake(1, 0, 0);
        // A freshly-woken character starts unscheduled; a tick on something
        // real parks it there again, and neither pace change below moves it.
        s.deliver(1, 0, Salience::NORMAL, say("something"));
        s.tick(1, 0, 0, |_, _| vec![]);

        s.set_pace(1, Pace::WORKING, 0);
        assert_eq!(s.pace_of(1), Some(Pace::WORKING));
        for at in [0, 4_000, 120_000] {
            assert!(
                !s.is_due(1, at),
                "quickening scheduled a character with an empty inbox at {at}"
            );
        }

        s.set_pace(1, Pace::AMBIENT, 0);
        assert_eq!(s.pace_of(1), Some(Pace::AMBIENT));
        assert!(!s.is_due(1, u64::MAX - 1));
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

    /// **A paused character stops thinking**, which is the whole of what a
    /// pause is for.
    ///
    /// The free-text `wait` it began as left the heartbeat alone, so the
    /// character woke four seconds later and decided to wait again, and again —
    /// a busy-loop wearing the word. Here the deadline *is* the schedule.
    #[test]
    fn a_pause_takes_the_character_out_of_the_schedule() {
        let s = sched();
        s.wake(1, 0, 0);
        // Due because something reached it, not because it woke.
        s.deliver(1, 0, Salience::NORMAL, say("anything"));
        assert!(s.is_due(1, 10_000), "an arrival did not schedule it");
        s.tick(1, 0, 0, |_, _| vec![]);

        assert!(
            s.pause_for(1, 0, pause_ms()),
            "there was a character to stop"
        );
        assert!(
            !s.is_due(1, pause_ms() - 1),
            "it came back before the pause was up"
        );
        assert!(
            s.is_due(1, pause_ms() + 1),
            "the pause never ended, so it will never think again"
        );
    }

    #[test]
    fn pausing_a_character_that_is_not_there_says_so() {
        let s = sched();
        assert!(!s.pause_for(99, 0, pause_ms()));
    }

    /// **Anything that reaches a paused character brings it straight back**,
    /// and nothing had to be written to make it so.
    ///
    /// This is the whole benefit over the typed wait it replaced. That was a
    /// subscription to one named thing, so everything *else* the world did —
    /// being messaged, being spoken to by the wrong person, an operator saying
    /// something in the room — landed in the inbox and sat there until the
    /// patience ran out, two minutes after whoever did it had gone. It needed a
    /// rousing bar, tuned against a ladder, to get half of that back.
    ///
    /// A pause is only a due time, and `deliver` moves every character's due
    /// time to now, so all of it comes back at once — including the quiet
    /// traffic, which under the old rule was deliberately left out and is the
    /// one case here that changed.
    #[test]
    fn anything_arriving_at_all_cuts_a_pause_short() {
        for (what, salience, kind) in [
            (
                "being spoken to",
                Salience::URGENT,
                EventKind::Speech {
                    speaker: "Orion Vance".into(),
                    text: "look at me".into(),
                    to: crate::engine::event::Addressed::You,
                },
            ),
            (
                "a message",
                Salience::URGENT,
                EventKind::Message {
                    thread: "Johnathan Sharratt".into(),
                    from: "Johnathan Sharratt".into(),
                    text: "where are you".into(),
                },
            ),
            (
                "somebody talking past you",
                Salience::from(npc_map::Weight::Note),
                EventKind::Speech {
                    speaker: "Orion Vance".into(),
                    text: "to somebody else".into(),
                    to: crate::engine::event::Addressed::Other {
                        who: "Maker-03".into(),
                    },
                },
            ),
            (
                "the room changing under you",
                Salience::IDLE,
                EventKind::Situation {
                    text: "You are in the green room.".into(),
                },
            ),
        ] {
            let s = sched();
            s.wake(1, 0, 0);
            s.pause_for(1, 0, pause_ms());
            assert!(!s.is_due(1, 1), "it did not stop");

            s.deliver(1, 0, salience, kind);
            assert!(s.is_due(1, 1), "{what} did not bring it back");
        }
    }

    /// **A pause cut short goes back to the ordinary schedule.**
    ///
    /// The path that used to spin: `deliver` sets `due_at = 0` for an arrival,
    /// and the end of `tick` puts the deadline back. A pause that survived its
    /// own interruption left the character due at zero for ever, ticking flat
    /// out on an empty queue, and every view of it read "healthy".
    #[test]
    fn a_pause_cut_short_does_not_leave_the_character_spinning() {
        let s = sched();
        s.wake(1, 0, 0);
        s.pause_for(1, 0, pause_ms());
        s.deliver(
            1,
            0,
            Salience::URGENT,
            EventKind::Speech {
                speaker: "Orion Vance".into(),
                text: "look at me".into(),
                to: crate::engine::event::Addressed::You,
            },
        );
        assert!(s.is_due(1, 0), "it did not wake");
        s.tick(1, 0, 0, |_, _| vec!["speak — yes".to_string()])
            .expect("it ticked");
        assert!(
            !s.is_due(1, 0),
            "due again at the same instant — the busy-loop is back"
        );
    }

    /// **A pause taken during a tick survives the end of that tick.**
    ///
    /// The act loop arms the pause from inside the closure, and the line at the
    /// end of `tick` runs afterwards. Without the flag it overwrites the
    /// deadline with `NEVER` — on the empty queue a paused character has by
    /// definition — which is not "wait two minutes" but "never think again",
    /// and reads from every angle as a healthy quiet character.
    #[test]
    fn a_pause_taken_mid_tick_is_not_overwritten_by_the_end_of_it() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("something to answer"));
        s.tick(1, 0, 0, |_, _| {
            s.pause_for(1, 0, pause_ms());
            vec!["reflect — that nothing here needs me".to_string()]
        })
        .expect("it ticked");

        assert!(!s.is_due(1, pause_ms() - 1), "the pause was cut short");
        assert!(
            s.is_due(1, pause_ms() + 1),
            "the pause became a character that never thinks again"
        );
    }

    /// **An act that answered brings the character straight back.**
    ///
    /// What a board says exists in that act's outcome and nowhere else —
    /// nothing in the world perceives a document being read — so a character
    /// left unscheduled after asking is one holding the answer with no turn in
    /// which to use it. It read the board, went to sleep, and read the board
    /// again the next time anything woke it.
    #[test]
    fn an_act_that_answered_gets_a_turn_to_use_the_answer() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("go and read the board"));
        s.tick(1, 0, 0, |_, _| {
            s.think_again(1);
            vec!["read — the muster board".to_string()]
        })
        .expect("it ticked");

        assert!(
            s.is_due(1, 0),
            "it was told something and given nowhere to put it"
        );
    }

    /// **How long since it last acted**, which is the figure the cast strip
    /// leads with — and the one that says whether a world is alive at all.
    ///
    /// An age rather than a timestamp, measured against the scheduler's own
    /// clock. Neither of the other two would do: the driver's starts minutes
    /// after `Scheduler::new`, so a duration written by one and read against the
    /// other is out by however long loading took, and the wall clock belongs to
    /// this machine while the reader is a browser on somebody else's.
    #[test]
    fn a_character_reports_how_long_since_it_last_acted() {
        let s = sched();
        s.wake(1, 0, 0);
        // Never, not "a very long time" — a character that has not acted is a
        // different fact from one that has gone quiet, and the strip says so.
        assert_eq!(s.census()[0].acted_ms_ago, None);

        // Thinking is not acting. A tick that decided on nothing leaves the
        // question where it was, or it answers "how long since it last ran".
        s.deliver(1, 0, Salience::NORMAL, say("something"));
        s.tick(1, 0, 0, |_, _| Vec::new()).expect("it ticked");
        assert_eq!(
            s.census()[0].acted_ms_ago,
            None,
            "a turn that produced nothing counted as an act"
        );

        s.deliver(1, 0, Salience::NORMAL, say("something else"));
        s.tick(1, 0, 0, |_, _| vec!["tell — anything".to_string()])
            .expect("it ticked");
        let age = s.census()[0]
            .acted_ms_ago
            .expect("it acted and reported nothing");
        assert!(
            age < 5_000,
            "the age is measured against the wrong clock: {age}"
        );
    }

    /// And every *other* act leaves it unscheduled, which is the half that
    /// stops the treadmill: a character that speaks and is immediately asked
    /// again speaks again, into the same silence, until its own window holds
    /// nothing but its own voice.
    #[test]
    fn an_ordinary_act_does_not_buy_another_turn() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::NORMAL, say("anything"));
        s.tick(1, 0, 0, |_, _| vec!["tell — something back".to_string()])
            .expect("it ticked");

        assert!(
            !s.is_due(1, u64::MAX - 1),
            "speaking scheduled its own next turn"
        );
    }

    /// The feed is an instrument, not a log. It must stay bounded under load.
    #[test]
    fn the_recent_feed_is_bounded() {
        let s = Scheduler::new(4);
        s.wake(1, 0, 0);
        for i in 0..20 {
            // Something to think about each time: a character with an empty
            // inbox does not tick, so a bare `tick` produces no record at all.
            s.deliver(1, i, Salience::NORMAL, say("something"));
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
        s.deliver(big, 0, Salience::NORMAL, say("something to answer"));
        s.tick(big, 0, 0, |_, _| vec![]).expect("ticked");

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

    /// A retired character is gone: not due, not tickable — and its own task,
    /// wherever it is waiting, is woken to find that out and exit.
    #[tokio::test]
    async fn a_retired_character_is_not_scheduled_and_its_wait_ends() {
        let s = sched();
        s.wake(1, 0, 0);
        s.deliver(1, 0, Salience::URGENT, say("x"));
        s.retire(1);
        assert!(!s.is_due(1, u64::MAX));
        assert!(s.tick(1, 0, 0, |_, _| vec![]).is_none());
        assert!(
            !tokio::time::timeout(Duration::from_secs(1), s.wait_due(1))
                .await
                .expect("the retired character's wait never ended"),
            "a wait on a retired character claimed it was due"
        );
    }

    /// A character parked at `NEVER` sleeps on its waker alone: the wait is
    /// still pending after real time has passed, and one delivery ends it.
    #[tokio::test]
    async fn an_unscheduled_character_sleeps_until_something_arrives() {
        let s = Arc::new(sched());
        s.wake(1, 0, 0);

        let waiting = tokio::spawn({
            let s = Arc::clone(&s);
            async move { s.wait_due(1).await }
        });
        // Genuinely parked: nothing has arrived, so the wait must not end.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!waiting.is_finished(), "a quiet character's wait returned");

        s.deliver(1, 0, Salience::NORMAL, say("something"));
        let due = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("the delivery did not wake the wait")
            .expect("the waiting task panicked");
        assert!(due);
    }
}
