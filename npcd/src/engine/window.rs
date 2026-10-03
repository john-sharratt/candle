//! The bounded conversation a character thinks inside for one day.
//!
//! # Why a window at all, in an unbounded-context engine
//!
//! It looks like a contradiction and is not. The substrate holds everything —
//! that is what the redo log and the provenance gather are for, and nothing here
//! deletes any of it. What the window bounds is the **turn tail carried
//! verbatim** into the next decode: the literal recent transcript, which is the
//! one part of the working set that grows with wall-clock time rather than with
//! relevance.
//!
//! Without a bound, a character that has been awake for six hours arrives at its
//! next tick carrying six hours of "time passes quietly" verbatim. Those turns
//! are not *forgotten* when they leave the window — they are in the substrate,
//! and the gather can pull any of them back the moment something makes them
//! relevant. They simply stop being pasted in unconditionally.
//!
//! That is the mind design's *soft fade by non-selection*: reversible, and
//! cue-resurfaceable. The hard forget belongs to the sleep fold, which is
//! `engine::sleep`'s business, not this module's.
//!
//! # Sizing
//!
//! [`DEFAULT_TURNS`] is the tail length. It is small on purpose: an NPC's
//! continuity is supposed to come from the gather, and a long verbatim tail is
//! precisely the crutch that would hide a gather that is not working. If a
//! character only behaves coherently with a hundred turns of transcript pasted
//! in, the retrieval is broken and the window is concealing it.

use std::collections::VecDeque;

use serde::Serialize;

use crate::engine::event::{Event, EventKind, Salience};

/// Turns of verbatim tail carried into the next decode.
///
/// Counts *turns*, not exchanges: an event and the act it produced are two.
///
/// A character's continuity is supposed to come from the gather, and a long
/// verbatim tail is the crutch that would hide a gather that is not working — a
/// character only coherent with a hundred turns pasted in has broken retrieval
/// and a window concealing it. So the ceiling below is not a formality; it is
/// the thing that keeps that crutch from being reached for quietly.
///
/// It was 24 while the act loop was still being closed, and the gather was not
/// the only thing that had to be proved. Now it is 64, matching
/// [`crate::engine::mind::CONTEXT_WINDOW_TURNS`]'s 32 exchanges: a character
/// gets the better part of an hour of its own conduct in front of it, which is
/// what a cast that has to remember a conversation across a dozen interruptions
/// needs. The ceiling moved with it rather than being deleted, because the
/// reason for having one did not change.
pub const DEFAULT_TURNS: usize = 64;

const _: () = assert!(
    DEFAULT_TURNS <= 64,
    "a long verbatim tail lets a broken gather look like a working one"
);

/// Which side of the conversation a turn came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Speaker {
    /// The world, the narrator, an operator — anything the character perceives.
    World,
    /// The character itself: its acts, rendered.
    Npc,
}

/// What produced a turn, kept beside its text so a later reader can tell a
/// thing that happened from a line of scenery without parsing prose.
///
/// The journal cites turns by id and decides what a claim rests on from this
/// tag alone. It never reads the wording: the wording is the model's own, or
/// the narrator's, and neither is a reliable witness to what kind of thing it
/// was.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// A world event, by its [`EventKind::tag`]. `ambient` is the room idling:
    /// a description arriving at idle salience, which is the stir's own flavour
    /// and says nothing happened.
    Event { tag: &'static str, ambient: bool },
    /// The character's own act, by tool name. Empty when the line is the world's
    /// refusal of a call that never parsed, which names no tool.
    Act { verb: String },
}

/// Event kinds that are standing state or machinery rather than something
/// that happened, and so are never a thing a claim can be said to rest on.
const UNCITABLE_EVENTS: &[&str] = &["nudge", "heartbeat", "sleep", "wake"];

impl Origin {
    pub fn of_event(e: &Event) -> Self {
        Origin::Event {
            tag: e.kind.tag(),
            ambient: matches!(e.kind, EventKind::Description { .. })
                && e.salience.get() <= Salience::IDLE.get(),
        }
    }

    /// The tool an act line names. Every act line is `tool` or `tool — what was
    /// asked …`, so the verb is whatever precedes the first separator; anything
    /// that is not a bare tool name there is a refusal's sentence, not a verb.
    pub fn of_act(line: &str) -> Self {
        let head = line.split(" — ").next().unwrap_or_default().trim();
        let verb = match !head.is_empty()
            && head
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            true => head.to_string(),
            false => String::new(),
        };
        Origin::Act { verb }
    }

    /// Whether a claim may rest on a turn from here. Scenery, standing
    /// instructions, machinery, a character's own reflection and a refusal
    /// that named no tool are all things a journal entry may sit beside and may
    /// not cite.
    pub fn citable(&self) -> bool {
        match self {
            Origin::Event { tag, ambient } => !ambient && !UNCITABLE_EVENTS.contains(tag),
            Origin::Act { verb } => !verb.is_empty() && verb != "reflect",
        }
    }
}

/// One turn in the tail.
#[derive(Clone, Debug, Serialize)]
pub struct Turn {
    /// Stable for the life of the window: assigned once, never reused, and not
    /// reset at the day boundary, so a journal entry's citation of it can never
    /// come to mean a different turn.
    pub id: u64,
    pub speaker: Speaker,
    pub origin: Origin,
    pub text: String,
    /// World-clock milliseconds when this landed.
    pub at_ms: u64,
    /// The band this turn supersedes within, if any — see
    /// [`crate::engine::event::EventKind::replaces`]. A new turn with the same
    /// band retires the older one wherever it sits in the window, which is the
    /// engine's one departure from pure append-only.
    pub replaces: Option<String>,
}

/// A character's rolling transcript.
///
/// `Clone` so a tick can take a snapshot to decode against and release the inbox lock first —
/// see `Scheduler::tick`. The cost is one `VecDeque<Turn>` of at most `cap` entries, against a
/// model decode measured in seconds; holding the lock instead stalls every `deliver`,
/// `census` and `window` call in the daemon, on tokio worker threads.
#[derive(Clone, Debug)]
pub struct Window {
    turns: VecDeque<Turn>,
    cap: usize,
    /// Turns that have left the window this run. Not a loss counter — they are
    /// all still in the substrate — but the number an operator wants when asking
    /// whether a character's continuity is coming from the gather or from the
    /// tail.
    faded: u64,
    /// The id the next turn takes. Starts at 1 so 0 is never a turn, and is not
    /// reset by [`Window::roll_over`].
    next_id: u64,
}

impl Window {
    pub fn new(cap: usize) -> Self {
        Self {
            // A zero cap would make `land` drop everything it was just handed,
            // producing a character with no present tense at all and no error to
            // explain it. One turn is the smallest thing that is still a
            // conversation.
            cap: cap.max(1),
            turns: VecDeque::with_capacity(cap.max(1)),
            faded: 0,
            next_id: 1,
        }
    }

    pub fn with_default_cap() -> Self {
        Self::new(DEFAULT_TURNS)
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn faded(&self) -> u64 {
        self.faded
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Add a turn, evicting the oldest if that puts us over the cap.
    ///
    /// A turn carrying a `replaces` band first retires any turn already in the
    /// window with the same band — wherever it sits, not just at the tail. A
    /// superseded map three turns back is exactly as misleading as one at the
    /// end.
    ///
    /// Returns the id the turn took.
    fn land(
        &mut self,
        speaker: Speaker,
        origin: Origin,
        text: String,
        at_ms: u64,
        replaces: Option<String>,
    ) -> u64 {
        if let Some(band) = &replaces {
            // Superseded, not faded: it was replaced by better information about
            // the same thing, which is a different event from falling out of the
            // window, and conflating them would make `faded` unreadable.
            self.turns.retain(|t| t.replaces.as_ref() != Some(band));
        }
        let id = self.next_id;
        self.next_id += 1;
        self.turns.push_back(Turn {
            id,
            speaker,
            origin,
            text,
            at_ms,
            replaces,
        });
        while self.turns.len() > self.cap {
            self.turns.pop_front();
            self.faded += 1;
        }
        id
    }

    /// Land a world event: its prose, its band, and what kind of thing it was.
    pub fn push_event(&mut self, e: &Event) -> u64 {
        self.land(
            Speaker::World,
            Origin::of_event(e),
            e.prose(),
            e.at_ms,
            e.kind.replaces(),
        )
    }

    /// Land one of the character's own act lines. The origin is read off the
    /// line, which is built in one place and always leads with the tool name.
    pub fn push_npc(&mut self, text: impl Into<String>, at_ms: u64) -> u64 {
        let text = text.into();
        let origin = Origin::of_act(&text);
        self.land(Speaker::Npc, origin, text, at_ms, None)
    }

    /// The turn with this id, if it is still in the window.
    pub fn get(&self, id: u64) -> Option<&Turn> {
        self.turns.iter().find(|t| t.id == id)
    }

    /// Every turn after `id`, oldest first. `after(0)` is the whole window.
    pub fn after(&self, id: u64) -> impl Iterator<Item = &Turn> {
        self.turns.iter().filter(move |t| t.id > id)
    }

    /// The id of the newest turn, or 0 when nothing has landed yet.
    pub fn newest_id(&self) -> u64 {
        self.next_id - 1
    }

    /// The id of the oldest turn still held, if any.
    pub fn oldest_id(&self) -> Option<u64> {
        self.turns.front().map(|t| t.id)
    }

    /// Rewrite the most recent of the character's own turns reading `from`.
    ///
    /// For an act whose result arrives after the act is recorded — a reflect,
    /// answered by a conversation of its own. The row goes in as the act alone
    /// and is completed here once there is something to complete it with.
    /// Returns whether there was such a turn: one that has already faded out of
    /// the window has nothing left to amend.
    pub fn amend_npc(&mut self, from: &str, to: String) -> bool {
        match self
            .turns
            .iter_mut()
            .rev()
            .find(|t| t.speaker == Speaker::Npc && t.text == from)
        {
            Some(turn) => {
                turn.text = to;
                true
            }
            None => false,
        }
    }

    pub fn turns(&self) -> impl Iterator<Item = &Turn> {
        self.turns.iter()
    }

    /// Empty the window without touching the fade counter's meaning.
    ///
    /// Called at the day boundary: the new day's conversation starts with no
    /// verbatim tail, because yesterday is now memory rather than transcript.
    /// The turns are not lost — `engine::sleep` has already written them.
    pub fn roll_over(&mut self) {
        self.turns.clear();
    }
}

impl Default for Window {
    fn default() -> Self {
        Self::with_default_cap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(w: &mut Window, text: &str, band: Option<&str>) -> u64 {
        let origin = Origin::Event {
            tag: "description",
            ambient: false,
        };
        w.land(
            Speaker::World,
            origin,
            text.to_string(),
            0,
            band.map(str::to_string),
        )
    }

    fn world(w: &mut Window, text: &str) {
        put(w, text, None);
    }

    fn event(kind: EventKind, salience: Salience) -> Event {
        Event::new(1, 0, salience, kind)
    }

    fn describe(text: &str) -> EventKind {
        EventKind::Description { text: text.into() }
    }

    #[test]
    fn a_window_holds_at_most_its_cap() {
        let mut w = Window::new(3);
        for i in 0..10 {
            world(&mut w, &format!("turn {i}"));
        }
        assert_eq!(w.len(), 3);
        assert_eq!(w.faded(), 7);
    }

    /// Oldest out, newest in. A window that dropped the newest would be a
    /// character that stops perceiving once busy.
    #[test]
    fn the_oldest_turn_is_the_one_that_leaves() {
        let mut w = Window::new(2);
        world(&mut w, "first");
        world(&mut w, "second");
        world(&mut w, "third");
        let texts: Vec<&str> = w.turns().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["second", "third"]);
    }

    /// A zero cap would silently produce a character with no present tense.
    #[test]
    fn a_zero_cap_still_keeps_one_turn() {
        let mut w = Window::new(0);
        world(&mut w, "the only thing I know");
        assert_eq!(w.len(), 1);
        assert_eq!(w.cap(), 1);
    }

    /// A superseded map three turns back misleads exactly as much as one at the
    /// end, so replacement scans the whole window rather than the tail.
    #[test]
    fn a_replacing_turn_retires_its_band_anywhere_in_the_window() {
        let mut w = Window::new(10);
        put(&mut w, "in band one", Some("situation"));
        world(&mut w, "something happened");
        world(&mut w, "something else happened");
        put(&mut w, "in the green room", Some("situation"));

        let texts: Vec<&str> = w.turns().map(|t| t.text.as_str()).collect();
        assert_eq!(
            texts,
            vec![
                "something happened",
                "something else happened",
                "in the green room"
            ],
            "the stale situation survived"
        );
    }

    /// Bands are independent: replacement is scoped to the band and never
    /// widens to superseding turns in general.
    #[test]
    fn replacement_is_per_band() {
        let mut w = Window::new(10);
        put(&mut w, "in band one", Some("situation"));
        put(&mut w, "finish the redoubt", Some("task"));
        put(&mut w, "in the green room", Some("situation"));
        let texts: Vec<&str> = w.turns().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["finish the redoubt", "in the green room"]);
    }

    /// Superseding is not fading. Conflating them makes the fade counter — the
    /// one number that says whether continuity comes from the gather — unreadable.
    #[test]
    fn superseding_a_turn_does_not_count_as_a_fade() {
        let mut w = Window::new(10);
        put(&mut w, "in band one", Some("situation"));
        put(&mut w, "in the green room", Some("situation"));
        assert_eq!(w.faded(), 0);
        assert_eq!(w.len(), 1);
    }

    /// **The join.** `replaces()` and `push_event` are each tested at their own
    /// end; this is the only place the chain the scheduler actually runs is
    /// exercised whole — an event, through its own band, into a real window.
    ///
    /// Broken, it fails silently and in the worst direction: the character
    /// carries every situation it has ever been in, with the oldest reading as
    /// current if the newest has faded out of the cap.
    #[test]
    fn a_situation_event_leaves_exactly_one_of_itself_in_a_real_window() {
        let mut w = Window::new(10);
        for room in ["band one", "the north run", "the green room"] {
            let e = event(
                EventKind::Situation {
                    text: format!("You are in {room}."),
                },
                Salience::IDLE,
            );
            w.push_event(&e);
            // Something happens between them, so the retirement has to scan
            // past it rather than only checking the tail.
            world(&mut w, "somebody came in");
        }

        let situations: Vec<&str> = w
            .turns()
            .map(|t| t.text.as_str())
            .filter(|t| t.starts_with("You are"))
            .collect();
        assert_eq!(
            situations,
            vec!["You are in the green room."],
            "the character is standing in more than one place"
        );
        // And the things that happened all survived — only the present replaces.
        assert_eq!(
            w.turns().filter(|t| t.text == "somebody came in").count(),
            3
        );
    }

    #[test]
    fn both_speakers_are_kept_in_order() {
        let mut w = Window::new(4);
        w.push_event(&event(describe("the gate opens"), Salience::NORMAL));
        w.push_npc("you step back", 11);
        let t: Vec<(Speaker, &str)> = w.turns().map(|t| (t.speaker, t.text.as_str())).collect();
        assert_eq!(
            t,
            vec![
                (Speaker::World, "the gate opens"),
                (Speaker::Npc, "you step back")
            ]
        );
    }

    /// The day boundary starts a fresh transcript. The turns are already written
    /// to the substrate by then, so this drops the tail and nothing else.
    #[test]
    fn rolling_over_empties_the_tail() {
        let mut w = Window::new(4);
        world(&mut w, "yesterday");
        w.roll_over();
        assert!(w.is_empty());
    }

    #[test]
    fn ids_start_at_one_and_rise_by_one() {
        let mut w = Window::new(8);
        assert_eq!(w.newest_id(), 0);
        let a = put(&mut w, "a", None);
        let b = put(&mut w, "b", None);
        let c = w.push_npc("speak — hello", 0);
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(w.newest_id(), 3);
    }

    /// A citation of an id that has since faded must find nothing, never a
    /// different turn that happened to take the slot.
    #[test]
    fn an_id_is_never_reused_after_it_fades() {
        let mut w = Window::new(2);
        let first = put(&mut w, "first", None);
        world(&mut w, "second");
        world(&mut w, "third");
        assert!(w.get(first).is_none());
        assert_eq!(w.oldest_id(), Some(2));
        assert_eq!(w.get(3).map(|t| t.text.as_str()), Some("third"));
    }

    /// Superseding retires a turn without renumbering its neighbours.
    #[test]
    fn a_superseded_turn_leaves_its_neighbours_ids_alone() {
        let mut w = Window::new(8);
        put(&mut w, "old room", Some("situation"));
        let between = put(&mut w, "a thing happened", None);
        let new = put(&mut w, "new room", Some("situation"));
        assert_eq!(between, 2);
        assert_eq!(new, 3);
        assert!(w.get(1).is_none());
        assert_eq!(w.get(2).map(|t| t.text.as_str()), Some("a thing happened"));
    }

    /// The day boundary clears the tail but the ids carry on, so yesterday's
    /// citations cannot alias today's turns.
    #[test]
    fn ids_survive_the_day_rollover() {
        let mut w = Window::new(8);
        world(&mut w, "yesterday");
        w.roll_over();
        assert_eq!(put(&mut w, "today", None), 2);
        assert!(w.get(1).is_none());
    }

    #[test]
    fn after_returns_the_turns_past_an_id_oldest_first() {
        let mut w = Window::new(8);
        for t in ["a", "b", "c", "d"] {
            world(&mut w, t);
        }
        let texts: Vec<&str> = w.after(2).map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["c", "d"]);
        assert_eq!(w.after(0).count(), 4);
        assert_eq!(w.after(4).count(), 0);
    }

    #[test]
    fn amending_an_act_keeps_its_id_and_origin() {
        let mut w = Window::new(8);
        let id = w.push_npc("reflect — the quiet", 0);
        assert!(w.amend_npc(
            "reflect — the quiet",
            "reflect — the quiet → I feel calm".into()
        ));
        let t = w.get(id).unwrap();
        assert_eq!(t.text, "reflect — the quiet → I feel calm");
        assert_eq!(
            t.origin,
            Origin::Act {
                verb: "reflect".into()
            }
        );
    }

    #[test]
    fn an_event_is_tagged_with_its_kind() {
        let mut w = Window::new(8);
        let said = event(
            EventKind::Speech {
                speaker: "Pax".into(),
                text: "the fabricator is gone".into(),
                to: crate::engine::event::Addressed::You,
            },
            Salience::NORMAL,
        );
        let id = w.push_event(&said);
        assert_eq!(
            w.get(id).unwrap().origin,
            Origin::Event {
                tag: "speech",
                ambient: false
            }
        );
    }

    /// The stir's idle flavour arrives as a description at idle salience. That is
    /// the one shape that is scenery, and a real description at ordinary
    /// salience is not.
    #[test]
    fn only_an_idle_description_is_ambient() {
        let idle = Origin::of_event(&event(describe("A pipe ticks."), Salience::IDLE));
        let real = Origin::of_event(&event(describe("The door slides open."), Salience::NORMAL));
        let nudge = Origin::of_event(&event(
            EventKind::Nudge { text: "x".into() },
            Salience::IDLE,
        ));
        assert_eq!(
            idle,
            Origin::Event {
                tag: "description",
                ambient: true
            }
        );
        assert_eq!(
            real,
            Origin::Event {
                tag: "description",
                ambient: false
            }
        );
        assert_eq!(
            nudge,
            Origin::Event {
                tag: "nudge",
                ambient: false
            }
        );
    }

    #[test]
    fn the_verb_is_whatever_precedes_the_first_separator() {
        let verb = |s: &str| match Origin::of_act(s) {
            Origin::Act { verb } => verb,
            other => panic!("not an act: {other:?}"),
        };
        assert_eq!(
            verb("speak — Pax; that it is done → You tell Pax."),
            "speak"
        );
        assert_eq!(verb("pause"), "pause");
        assert_eq!(verb("invoke — the fabricator ✗ out of reach"), "invoke");
        assert_eq!(verb("move_to — the command room"), "move_to");
        // A refusal's sentence names no tool.
        assert_eq!(verb("Nothing in the world answers that."), "");
        assert_eq!(verb(""), "");
    }

    #[test]
    fn what_a_claim_may_rest_on() {
        let ev = |tag: &'static str, ambient: bool| Origin::Event { tag, ambient };
        let act = |verb: &str| Origin::Act { verb: verb.into() };
        for cites in [
            ev("speech", false),
            ev("message", false),
            ev("description", false),
            ev("situation", false),
            ev("announcement", false),
            ev("entity", false),
            act("invoke"),
            act("speak"),
        ] {
            assert!(cites.citable(), "{cites:?} should be citable");
        }
        for refuses in [
            ev("description", true),
            ev("nudge", false),
            ev("heartbeat", false),
            ev("sleep", false),
            ev("wake", false),
            act("reflect"),
            act(""),
        ] {
            assert!(!refuses.citable(), "{refuses:?} should not be citable");
        }
    }

    /// A default-capped window is the shape every character actually gets — the
    /// constant's bound is held at compile time above.
    #[test]
    fn the_default_window_uses_the_default_cap() {
        assert_eq!(Window::default().cap(), DEFAULT_TURNS);
    }
}
