//! A character's loop state: its inbox, its deadline for the next thought,
//! and what it is still owed.
//!
//! An empty inbox is the resting state of a quiet character, so readiness
//! reports it as quiet, not blocked. An arrival that asks for a response wakes
//! the character at once; information that asks nothing waits for its next
//! wake. The standing task is gated on a quiet clock that counts only news, so
//! a conversation in progress is not mistaken for idleness.

use super::*;

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
    /// Events are waiting, and the character will think about them at its next
    /// scheduled beat — the ordinary way a tick comes due. Not stuck: simply
    /// waiting for its beat.
    Waiting,
    /// A high-salience arrival — usually being spoken to — is interrupting the
    /// schedule to force a tick right now. From the character's side it has been
    /// interrupted, which is what the name reads as.
    Interrupted,
}

/// One character's loop state.
#[derive(Debug)]
pub struct Inbox {
    pub npc_id: u64,
    pub(super) queue: VecDeque<Event>,
    /// Current heartbeat interval — the character's idle metabolism.
    pub(super) heartbeat: Duration,
    /// How slow that metabolism is allowed to get.
    pub(super) pace: Pace,
    /// Monotonic scheduler time (ms) at which this character next thinks.
    pub(super) due_at: u64,
    pub(super) preempted: bool,
    pub window: Window,
    /// What this character's journal holds and when its next draft is due.
    /// Shared, because a draft runs on a task of its own, off the character's
    /// line, and settles its own state when it ends.
    pub(super) journal: Arc<Mutex<JournalState>>,
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
    pub(super) last_news_ms: u64,
    /// When this character last did something, measured from
    /// [`Scheduler::born`].
    ///
    /// Only a tick that produced **acts** sets it: a character can think and
    /// decide to do nothing, and the question this answers is how long it has
    /// been since it last *acted*, not since it last ran.
    ///
    /// `None` until the first one, which is a different fact from "a long time
    /// ago" and is shown as such.
    pub(super) last_act_at: Option<Duration>,
    /// World time (ms) of the last tick that produced acts — the other half
    /// of what [`Inbox::quiet_for`] measures quiet from.
    pub(super) last_act_ms: u64,
    /// Set when the tick that is still running has already decided when the
    /// next one is — by pausing, or by asking the world something it must come
    /// back to read.
    ///
    /// It exists only to survive the line at the end of [`Scheduler::tick`],
    /// which otherwise puts the deadline back to `NEVER` on the empty queue
    /// both of those cases have by definition. Cleared there.
    pub(super) scheduled: bool,
    /// Owed a turn on what its last act came back with — see
    /// [`Inbox::think_again`]. Taken by the next [`Scheduler::begin_tick`],
    /// which runs that turn even with nothing new in the queue.
    pub(super) owed: bool,
    /// How many events at the head of the queue a failed turn handed back,
    /// already written into the window — see [`Scheduler::retry`].
    pub(super) replayed: usize,
    /// World time (ms) the standing task was last restated to this character.
    ///
    /// **The task has to be quiet about itself too.** It is not news, so it
    /// cannot stamp `last_news_ms` — which left the gate, once open, open
    /// forever: a character with nobody talking to it was handed the task again
    /// on every tick, and because it supersedes in its own band each copy took
    /// the most recent position in the window. Identical instruction, identical
    /// context, identical decode; the character repeated one act until something
    /// outside it changed.
    pub(super) last_nudge_ms: u64,
    /// World time (ms) a `rest` of this character's last landed: a mind with
    /// no body stopping, by its own act, until something reaches it — see
    /// [`Inbox::nudge_due`]. `None` for one that has never rested.
    pub(super) rested_ms: Option<u64>,
    /// What a person asked of this character that it has not yet told them
    /// the outcome of — see [`Scheduler::ask`].
    pub(super) asked: Option<Asked>,
    /// What the character's own task sleeps on — see [`Scheduler::wait_due`].
    /// A delivery, a cut-short pause, and a retirement each notify it; the
    /// permit is stored if the task is mid-think, so a wake sent while the
    /// character is busy is read the moment it returns to its wait.
    pub(super) waker: Arc<Notify>,
}

/// How long a request stands with nothing told of it before it is dropped.
pub const ASKED_FOR_MS: u64 = 20 * 60 * 1000;

/// A request a person made of a character on the phone, standing until the
/// character has worked at it and told them what came of it.
#[derive(Debug, Clone)]
pub(super) struct Asked {
    /// Who asked, by the name the character calls them.
    pub(super) party: String,
    /// What they asked, as the conversation's journal entry wrote it.
    pub(super) request: String,
    /// World time (ms) it was asked.
    pub(super) at_ms: u64,
    /// Whether an act of the character's has come back with something since:
    /// only then is telling them an answer and not a promise of one.
    pub(super) worked: bool,
}

impl Inbox {
    /// Begin a draft over the turns not yet written about, whether or not one
    /// is due. `None` while one is running or when nothing is new.
    pub(super) fn begin_draft(&mut self) -> Option<Due> {
        let span = self.journal.lock().unwrap().begin(&self.window)?;
        Some(Due {
            state: Arc::clone(&self.journal),
            span,
        })
    }

    pub fn new(npc_id: u64) -> Self {
        Self {
            npc_id,
            queue: VecDeque::new(),
            heartbeat: IDLE_HEARTBEAT,
            pace: Pace::AMBIENT,
            due_at: 0,
            preempted: false,
            window: Window::with_default_cap(),
            journal: Arc::new(Mutex::new(JournalState::new(npc_id))),
            day: DayTracker::new(),
            ticks: 0,
            events_seen: 0,
            last_act_at: None,
            last_act_ms: 0,
            scheduled: false,
            owed: false,
            replayed: 0,
            last_news_ms: 0,
            last_nudge_ms: 0,
            rested_ms: None,
            asked: None,
            waker: Arc::new(Notify::new()),
        }
    }

    /// What a person asked of this character and has not been told the
    /// outcome of, while it still stands at `world_ms`: who, and what was
    /// asked. One left standing past [`ASKED_FOR_MS`] is dropped — a request
    /// nothing came of does not hold a character for ever.
    pub fn asked(&mut self, world_ms: u64) -> Option<(String, String)> {
        let standing = self
            .asked
            .as_ref()
            .is_some_and(|a| world_ms.saturating_sub(a.at_ms) < ASKED_FOR_MS);
        if !standing {
            self.asked = None;
        }
        self.asked
            .as_ref()
            .map(|a| (a.party.clone(), a.request.clone()))
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
    ///
    /// **The turn is owed, and runs on an empty queue.** What the act came back
    /// with rides on the conversation as a pending answer, not as an event in
    /// the queue — so a follow-up that waited for an event never ran: the
    /// character came due, found nothing queued, and went quiet with the answer
    /// unread until something else in the world happened to wake it. While the
    /// building spoke every half minute that was hidden; a Maker at its desk,
    /// spared the building's noise, stopped for minutes after every read.
    pub fn think_again(&mut self) {
        self.due_at = 0;
        self.scheduled = true;
        self.owed = true;
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
    ///
    /// **What a conversation settled is news.** It is what a person asked for
    /// on the phone, told to the mind that does it. Left out, it moved no
    /// clock: Zen's mind was handed "survey the house network and report" and,
    /// in the same turn and again ninety seconds on, its idle thought —
    /// "Nothing calls for you this moment… rest" — and it rested.
    pub(super) fn is_news(kind: &EventKind) -> bool {
        matches!(
            kind,
            EventKind::Speech { .. }
                | EventKind::Concluded { .. }
                | EventKind::Description { .. }
                | EventKind::Announcement { .. }
                | EventKind::Terminal { .. }
                | EventKind::Entity { .. }
                | EventKind::Operator { .. }
                | EventKind::MindControl { .. }
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
        // **A character that has just acted has something going on.** What an
        // act came back with rides on the conversation, not in the queue, so a
        // character working through a request one act at a time received no
        // news at all while it worked — and was told mid-task that nothing
        // called for it.
        world_ms.saturating_sub(self.last_news_ms.max(self.last_act_ms))
    }

    /// Whether this character has acted since news last reached it — what a
    /// mind with no body is asked before it may `rest`. An act of any kind
    /// counts, a refused one too: the question is whether it turned to what
    /// arrived, not whether it did it well.
    pub fn acted_since_news(&self) -> bool {
        self.last_act_ms >= self.last_news_ms
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
    ///
    /// **A mind that rested is not woken to be told to rest.** The task rouses
    /// a character gone quiet; one that stopped by its own `rest`, with
    /// nothing reaching it since, has not gone quiet — it is waiting for its
    /// round, its terminals or a message, each of which is news and wakes it.
    /// Zen's idle thought tells it to rest, so the task woke it every ninety
    /// seconds, all night, to read that and rest again.
    pub fn nudge_due(&self, world_ms: u64, after_ms: u64) -> bool {
        let resting = self.rested_ms.is_some_and(|r| r >= self.last_news_ms);
        !resting
            && self.quiet_for(world_ms) >= after_ms
            && world_ms.saturating_sub(self.last_nudge_ms) >= after_ms
    }

    /// Note that this character's `rest` landed at `world_ms`.
    pub fn rested(&mut self, world_ms: u64) {
        self.rested_ms = Some(self.rested_ms.map_or(world_ms, |r| r.max(world_ms)));
    }

    pub fn readiness(&self) -> Readiness {
        if self.preempted {
            Readiness::Interrupted
        } else if self.queue.is_empty() {
            Readiness::Quiet
        } else {
            Readiness::Waiting
        }
    }

    /// Accept an event. Never refuses: the mind design is explicit that a filter
    /// dropping evidence before it lands makes a delusion permanent.
    ///
    /// **Every arrival that asks something is a wake.** Nothing polls, so an
    /// event that does not rouse the character's task is an event nobody ever
    /// reads — it would sit in the queue until something louder happened to
    /// arrive behind it. The salience still decides how *urgently* the
    /// character reads it (`readiness` keeps the Pulse feed's "why did it wake"
    /// column); a character mid-think elsewhere simply finds the wake stored on
    /// its waker when it comes back. The one exception is information that
    /// asks nothing ([`Event::wakes`]): queued, read with the next wake, and
    /// leaving the character's schedule exactly as it was.
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
        if event.wakes() {
            self.scheduled = false;
        }
        if preempts {
            self.preempted = true;
            // Something happened. Tighten the metabolism — this is the whole of
            // the alertness model, and it needs no combat branch anywhere.
            self.heartbeat = ALERT_HEARTBEAT;
        }
        // **An event that supersedes its band retires the queued one it
        // supersedes**, as the window does when it lands (`EventKind::replaces`,
        // `window::Window::push`). A situation is what is true now, so a second
        // one waiting beside the first is the first gone stale, never news. Kept,
        // every copy rode the next tick: a mind that is its world is carried its
        // whole view each time anything in it moves, so seven characters on
        // missions queued a run of near-identical "Everything inside you" views
        // between two of Zen's turns, each one listed in the tick Pulse shows.
        // The held path collapses bands the same way (`held`).
        if let Some(band) = event.kind.replaces() {
            self.queue
                .retain(|queued| queued.kind.replaces().as_deref() != Some(band.as_str()));
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
