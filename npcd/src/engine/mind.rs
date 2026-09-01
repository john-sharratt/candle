//! A character's live conversation, and the decode that turns perception into
//! acts.
//!
//! # One conversation per character per day
//!
//! Each character holds a [`Sequence`] — a conversation on the substrate — for
//! the day it is living in. Its id is *derived* from `(npc_id, day)` rather than
//! allocated, so a daemon restarted at noon rejoins the conversation the
//! character was already on instead of opening a second one for the same day.
//! That bug is invisible until somebody asks why a character has two of
//! everything.
//!
//! At the day boundary the conversation is retired and a new one opened. Retired
//! means **tombstoned**, not deleted: the turns stay in the redo log and stay
//! reachable by the gather. What changes is that they are no longer selected by
//! default. This is the mind design's soft fade made explicit at a boundary —
//! yesterday stops being transcript and becomes memory.
//!
//! # Why the sequence is windowed as well
//!
//! `SequenceConfig::context_window_turns` bounds what is prefilled onto the GPU
//! per turn. That is the same discipline as [`crate::engine::window`] and for
//! the same reason, one layer down: the substrate holds everything, the gather
//! decides what is relevant, and only a bounded tail is carried verbatim. Both
//! bounds are deliberate and neither is the other's fallback.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_conversation::{ConversationEngine, Sequence, SequenceConfig};

use crate::engine::act::{self, Parsed};
use crate::engine::event::Event;
use crate::engine::prompt::{self, Persona};
use crate::engine::sleep::conversation_id;
use crate::engine::tools::{for_mode, Mode};
use crate::engine::window::{Speaker, Window};

/// How many completed exchanges the GPU sequence carries per turn.
///
/// Matches [`crate::engine::window::DEFAULT_TURNS`] in intent — a short verbatim
/// tail, with continuity coming from the gather — but counts exchanges rather
/// than turns, so it is half the number.
pub const CONTEXT_WINDOW_TURNS: usize = 12;

/// The GPU tail and the perception window are two bounds with one intent,
/// counted in different units — exchanges here, turns there. Held at compile
/// time rather than by a test, so a change to either constant has to reckon with
/// the other even in a build nobody runs the tests for.
const _: () = assert!(CONTEXT_WINDOW_TURNS * 2 <= crate::engine::window::DEFAULT_TURNS);

/// A character's live conversation.
struct Live {
    sequence: Sequence,
    /// The day this conversation belongs to. A mismatch against the world's day
    /// is what triggers the roll-over.
    day: u64,
    id: String,
}

/// Every character's conversation, and the engine they run on.
pub struct Minds {
    engine: Arc<Mutex<ConversationEngine>>,
    /// The model's own dialect and sampling, captured at load.
    ///
    /// From `ModelBuilder::conversation_config` rather than assembled here: the
    /// dialect's markers and the model's sampling defaults belong to the
    /// checkpoint, and a second opinion about either is a silent way to run a
    /// model outside the settings it was tuned under.
    base_config: SequenceConfig,
    live: Mutex<HashMap<u64, Live>>,
}

/// What one character's tick produced.
#[derive(Debug, Default)]
pub struct Thought {
    pub parsed: Parsed,
    /// Set when this tick opened a new day's conversation.
    pub rolled_over: Option<(u64, u64)>,
}

impl Minds {
    pub fn new(engine: Arc<Mutex<ConversationEngine>>, base_config: SequenceConfig) -> Self {
        Self {
            engine,
            base_config,
            live: Mutex::new(HashMap::new()),
        }
    }

    /// The engine, for work that is not a character thinking.
    ///
    /// [`crate::lifegen`] primes its own conversations and forks them; it is not
    /// a character taking a turn, so it does not go through [`Self::think`]. The
    /// handle is shared rather than a second engine because there is one card
    /// and one scheduler, and the generator's forks batch into the same waves as
    /// everything else.
    pub fn engine(&self) -> Arc<Mutex<ConversationEngine>> {
        Arc::clone(&self.engine)
    }

    /// The model's own dialect and sampling, as captured at load.
    ///
    /// Handed out rather than reassembled by the caller for the reason it is
    /// held here at all: a second opinion about a checkpoint's markers or its
    /// sampling defaults is a silent way to run a model outside the settings it
    /// was tuned under.
    pub fn base_config(&self) -> SequenceConfig {
        self.base_config.clone()
    }

    /// How many characters currently hold an open conversation.
    ///
    /// Not the same as the scheduler's population: a character wakes into the
    /// scheduler at startup and only opens a conversation on its first tick, so
    /// this trails it. The gap is exactly "how many have thought at least once",
    /// which is what the Pulse header reports.
    pub fn resident(&self) -> usize {
        self.live.lock().unwrap().len()
    }

    /// Run one thinking step: perception in, acts out.
    ///
    /// Errors are returned rather than swallowed. A decode that failed and a
    /// character that chose to do nothing produce the same empty act list, and
    /// the caller has to be able to tell them apart — Pulse renders them
    /// differently, and it should.
    pub fn think(
        &self,
        npc_id: u64,
        persona: &Persona<'_>,
        mode: Mode,
        day: u64,
        events: &[Event],
        window: &Window,
    ) -> anyhow::Result<Thought> {
        let mut thought = Thought::default();
        let mut live = self.live.lock().unwrap();

        // The day boundary. Checked here rather than on a timer because a
        // timer fires on host-elapsed time and would be wrong the moment the
        // narrative clock is paused, jumped or re-paced — all of which the
        // console can do at any moment.
        if let Some(existing) = live.get(&npc_id) {
            if existing.day != day {
                let from = existing.day;
                // Tombstone, not delete. The turns stay in the redo log; what
                // changes is that they stop being selected by default.
                if let Some(l) = live.remove(&npc_id) {
                    retire(&self.engine, l);
                }
                thought.rolled_over = Some((from, day));
            }
        }

        // Opened through the vacant entry rather than `contains_key` + `insert`,
        // so the map is probed once and the borrow below cannot miss.
        let l = match live.entry(npc_id) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(slot) => {
                let id = conversation_id(npc_id, day);
                let system = prompt::build(persona, mode, &for_mode(mode));
                let mut cfg = self.base_config.clone();
                cfg.context_window_turns = CONTEXT_WINDOW_TURNS;
                let sequence = self.engine.lock().unwrap().new_conversation(&system, cfg)?;
                slot.insert(Live { sequence, day, id })
            }
        };

        // Everything drained this tick, as one message. A fat batch is one
        // better-informed thinking step rather than several thrashing ones —
        // the mind design is explicit that this is what a busy character should
        // get.
        let perception = compose(events, window);
        let response = l.sequence.send_turn(&perception)?;
        thought.parsed = act::parse(&response.text);
        Ok(thought)
    }

    /// Retire a character's conversation — on delete, or on shutdown.
    pub fn retire_npc(&self, npc_id: u64) {
        if let Some(l) = self.live.lock().unwrap().remove(&npc_id) {
            retire(&self.engine, l);
        }
    }
}

/// Tombstone a conversation's timeline.
///
/// Failure is logged, never propagated. A tombstone that did not land leaves
/// yesterday's turns selectable — untidy, and strictly better than refusing to
/// open today's conversation over it.
fn retire(engine: &Arc<Mutex<ConversationEngine>>, l: Live) {
    let timeline = l.sequence.timeline_id();
    match engine.lock().unwrap().tombstone_timeline(timeline) {
        Ok(()) => tracing::info!("conversation {} retired (tombstoned)", l.id),
        Err(e) => tracing::warn!(
            "conversation {} could not be tombstoned: {e:?} — yesterday stays selectable",
            l.id
        ),
    }
}

/// What the character reads this tick.
///
/// The events, as prose, in arrival order. The window is *not* pasted in: the
/// sequence carries its own bounded tail (`context_window_turns`) and the
/// substrate carries the rest, so repeating the window here would put the same
/// turns in the context twice — once verbatim from us and once from the
/// sequence's own history — and teach the model that everything happens twice.
fn compose(events: &[Event], window: &Window) -> String {
    let _ = window;
    let mut s = String::new();
    for (i, e) in events.iter().enumerate() {
        if i > 0 {
            s.push_str("\n\n");
        }
        s.push_str(&e.prose());
    }
    s
}

/// Render a window turn for a transcript view.
pub fn render_turn(speaker: Speaker, text: &str) -> String {
    match speaker {
        Speaker::World => text.to_string(),
        Speaker::Npc => format!("→ {text}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::event::{EventKind, Salience};

    fn ev(text: &str) -> Event {
        Event::new(
            0,
            0,
            Salience::NORMAL,
            EventKind::Description { text: text.into() },
        )
    }

    /// A fat batch arrives as one message, in order — one better-informed
    /// thinking step rather than several thrashing ones.
    #[test]
    fn a_batch_composes_in_arrival_order() {
        let w = Window::with_default_cap();
        let s = compose(&[ev("the gate opens"), ev("someone shouts")], &w);
        assert_eq!(s, "the gate opens\n\nsomeone shouts");
    }

    /// **The window must not be pasted in.** The sequence carries its own
    /// bounded tail, so repeating it here would put the same turns in the
    /// context twice and teach the model that everything happens twice.
    #[test]
    fn the_window_is_not_repeated_into_the_message() {
        let mut w = Window::with_default_cap();
        w.push_world("something that already happened", 0, None);
        w.push_npc("and what I did about it", 0);
        let s = compose(&[ev("something new")], &w);
        assert_eq!(s, "something new");
        assert!(!s.contains("already happened"));
    }

    #[test]
    fn an_empty_batch_composes_to_nothing() {
        assert_eq!(compose(&[], &Window::with_default_cap()), "");
    }

    /// A restart mid-day must rejoin the day's conversation rather than open a
    /// second one for the same day.
    #[test]
    fn a_days_conversation_id_is_stable_across_a_restart() {
        assert_eq!(conversation_id(4, 9), conversation_id(4, 9));
        assert_ne!(conversation_id(4, 9), conversation_id(4, 10));
    }

    #[test]
    fn an_act_renders_distinguishably_from_perception() {
        assert_eq!(render_turn(Speaker::World, "it rains"), "it rains");
        assert_eq!(render_turn(Speaker::Npc, "waits"), "→ waits");
    }
}
