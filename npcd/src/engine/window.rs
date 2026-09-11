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

/// One turn in the tail.
#[derive(Clone, Debug, Serialize)]
pub struct Turn {
    pub speaker: Speaker,
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
}

impl Window {
    pub fn new(cap: usize) -> Self {
        Self {
            // A zero cap would make `push` drop everything it was just handed,
            // producing a character with no present tense at all and no error to
            // explain it. One turn is the smallest thing that is still a
            // conversation.
            cap: cap.max(1),
            turns: VecDeque::with_capacity(cap.max(1)),
            faded: 0,
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
    pub fn push(&mut self, turn: Turn) {
        if let Some(band) = &turn.replaces {
            let before = self.turns.len();
            self.turns.retain(|t| t.replaces.as_ref() != Some(band));
            // Superseded, not faded: it was replaced by better information about
            // the same thing, which is a different event from falling out of the
            // window, and conflating them would make `faded` unreadable.
            let _ = before;
        }
        self.turns.push_back(turn);
        while self.turns.len() > self.cap {
            self.turns.pop_front();
            self.faded += 1;
        }
    }

    pub fn push_world(&mut self, text: impl Into<String>, at_ms: u64, replaces: Option<String>) {
        self.push(Turn {
            speaker: Speaker::World,
            text: text.into(),
            at_ms,
            replaces,
        });
    }

    pub fn push_npc(&mut self, text: impl Into<String>, at_ms: u64) {
        self.push(Turn {
            speaker: Speaker::Npc,
            text: text.into(),
            at_ms,
            replaces: None,
        });
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

    fn world(w: &mut Window, text: &str) {
        w.push_world(text, 0, None);
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
        w.push_world("in band one", 0, Some("situation".into()));
        world(&mut w, "something happened");
        world(&mut w, "something else happened");
        w.push_world("in the green room", 0, Some("situation".into()));

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
        w.push_world("in band one", 0, Some("situation".into()));
        w.push_world("finish the redoubt", 0, Some("task".into()));
        w.push_world("in the green room", 0, Some("situation".into()));
        let texts: Vec<&str> = w.turns().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["finish the redoubt", "in the green room"]);
    }

    /// Superseding is not fading. Conflating them makes the fade counter — the
    /// one number that says whether continuity comes from the gather — unreadable.
    #[test]
    fn superseding_a_turn_does_not_count_as_a_fade() {
        let mut w = Window::new(10);
        w.push_world("in band one", 0, Some("situation".into()));
        w.push_world("in the green room", 0, Some("situation".into()));
        assert_eq!(w.faded(), 0);
        assert_eq!(w.len(), 1);
    }

    /// **The join.** `replaces()` and `push_world` are each tested at their own
    /// end; this is the only place the chain the scheduler actually runs is
    /// exercised whole — an event, through its own band, into a real window.
    ///
    /// Broken, it fails silently and in the worst direction: the character
    /// carries every situation it has ever been in, with the oldest reading as
    /// current if the newest has faded out of the cap.
    #[test]
    fn a_situation_event_leaves_exactly_one_of_itself_in_a_real_window() {
        use crate::engine::event::{Event, EventKind, Salience};

        let mut w = Window::new(10);
        for room in ["band one", "the north run", "the green room"] {
            let e = Event::new(
                1,
                0,
                Salience::IDLE,
                EventKind::Situation {
                    text: format!("You are in {room}."),
                },
            );
            w.push_world(e.prose(), e.at_ms, e.kind.replaces());
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
        w.push_world("the gate opens", 10, None);
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

    /// A default-capped window is the shape every character actually gets — the
    /// constant's bound is held at compile time above.
    #[test]
    fn the_default_window_uses_the_default_cap() {
        assert_eq!(Window::default().cap(), DEFAULT_TURNS);
    }
}
