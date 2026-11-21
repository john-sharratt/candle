//! The two records an act leaves behind: the line in the feed and the verdict the
//! character reads.
//!
//! A person watching and the character that acted need different sentences, so
//! one act produces both. The feed shows the act and what came of it, with a mark
//! between the two that the console uses to set them in different faces. The
//! character is given only the world's verdict, since its own arguments handed
//! back as an outcome would read as one. A reflect is recorded as the thought it
//! asked for, and the row is completed once the reflection gives its answer.

use super::*;

/// What separates an act from what came of it in a feed line, when it landed.
///
/// A single character with spaces around it, because the console splits on it
/// to set the two halves in different faces — see `pulse.js::splitAct`. A word
/// would appear inside the prose on either side of it and the split would land
/// in the middle of a sentence.
pub const LANDED: &str = "→";

/// The same, when the world refused. Distinct from [`LANDED`] so a refusal is
/// legible as one at a glance rather than by reading the sentence.
pub const REFUSED: &str = "✗";

/// What one act left behind, for the two readers that want different things.
///
/// **A person watching and the character that acted need different sentences.**
/// The feed wants the act — `move_to — the command room` — in a scannable
/// column beside every other act. The character wants the world's verdict —
/// "You got to the command room." — because that is the only thing that says
/// whether what it tried happened.
///
/// These were one string while the feed was the only channel. Handing that
/// string to the model as a `<tool_response>` sent it its own arguments back
/// as though they were an outcome. See [`Runtime::record_act`].
#[derive(Debug, Clone)]
pub struct Recorded {
    /// The Pulse line, and the character's window.
    pub feed: String,
    /// What the character is told came of the act.
    pub answer: String,
    /// Whether the world took the act — the verdict `answer` is written from.
    /// What decides whether a `reflect` goes on to a reflection: one the world
    /// refused was not a character stopping to think.
    pub landed: bool,
    /// Whether the act was a line called after somebody leaving the room, so the
    /// narration voices the parting rather than a word spoken face to face.
    pub departing: bool,
}

/// What a turn with a reflection in it still owes, held until the reflection's
/// first question is answered — see [`Runtime::spawn_reflection`].
#[derive(Clone, Debug)]
pub struct Owed {
    /// Every answer the turn owes, one per call, in call order. The reflect's
    /// holds the world's own line until the reflection replaces it.
    pub answers: Vec<String>,
    /// Which of `answers` is the reflect's.
    pub slot: usize,
    /// The reflect's row as recorded: the act without its result. Completed
    /// with the result once there is one — see [`Scheduler::amend_act`].
    pub row: String,
}

/// What came of saying something to a character on its handset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageSent {
    /// What the character is called, by the world.
    pub with: String,
    /// How much is now waiting for it, unread.
    pub waiting_for_them: usize,
    /// Whether it carries a handset, and so can answer at all.
    pub can_reply: bool,
}
