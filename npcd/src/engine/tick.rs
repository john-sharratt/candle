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
    /// Monotonic scheduler time (ms) at which this character next thinks.
    due_at: u64,
    preempted: bool,
    pub window: Window,
    pub day: DayTracker,
    /// Lifetime counters, for the Pulse view.
    pub ticks: u64,
    pub events_seen: u64,
}

impl Inbox {
    pub fn new(npc_id: u64) -> Self {
        Self {
            npc_id,
            queue: VecDeque::new(),
            heartbeat: IDLE_HEARTBEAT,
            due_at: 0,
            preempted: false,
            window: Window::with_default_cap(),
            day: DayTracker::new(),
            ticks: 0,
            events_seen: 0,
        }
    }

    pub fn depth(&self) -> usize {
        self.queue.len()
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

    /// Relax toward idle after a quiet tick.
    fn relax(&mut self) {
        let next = self.heartbeat.mul_f32(RELAX);
        self.heartbeat = next.min(IDLE_HEARTBEAT);
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
        let mut due = self.due.lock().unwrap();
        let inboxes = self.inboxes.lock().unwrap();
        while let Some(top) = due.peek() {
            if top.at > now_ms {
                break;
            }
            let Due { npc_id, .. } = due.pop().expect("peeked");
            // Retired, or a stale duplicate from a preempt. Either way, drop it.
            if !inboxes.contains_key(&npc_id) {
                continue;
            }
            if !ready.contains(&npc_id) {
                ready.push(npc_id);
            }
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
        inbox.due_at = now_ms + inbox.heartbeat.as_millis() as u64;
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
