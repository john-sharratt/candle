//! The records a tick leaves behind for the Pulse view and the journal.
//!
//! Character ids use the whole u64 range, so they are written as strings: a
//! JSON number would be rounded in the browser and stop matching the same
//! character elsewhere on the page. Ages are computed against the scheduler's
//! own clock rather than stored as timestamps, so a reader on another machine
//! sees a duration that is true whatever its clock says.

use super::*;

/// A journal draft that has been started: the character's journal state, which
/// the draft settles when it ends, and the stretch it is about. See
/// [`Scheduler::journal_begin`].
#[derive(Debug)]
pub struct Due {
    pub state: Arc<Mutex<JournalState>>,
    pub span: Span,
}

/// A day roll-over: the days it crossed.
#[derive(Debug)]
pub struct Rollover {
    pub from: u64,
    pub to: u64,
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
    /// **Serialised as a base-36 string ([`npcs::npc_id_wire`]), and it has to
    /// be a string.**
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
    pub(super) at: Duration,
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

/// A `u64` id on the wire as a string. See [`TickRecord::npc_id`].
pub(super) fn id_as_string<S: serde::Serializer>(id: &u64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&npcs::npc_id_wire(*id))
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
    /// Journal entries this character has kept, over every run of the daemon.
    pub journal_written: u64,
    /// Items its journal holds open.
    pub journal_open: usize,
    /// Whether an entry is being drafted now.
    pub journal_drafting: bool,
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
    /// The world-clock instant of the last news it was given — what the quiet
    /// gate on its standing task measures from.
    pub last_news_ms: u64,
    /// The world-clock instant its standing task was last restated — what the
    /// summons gate measures from. One ahead of the world's clock shuts both
    /// gates until the clock reaches it.
    pub last_nudge_ms: u64,
}
