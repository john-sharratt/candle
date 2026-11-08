//! What the night looks like now, for the tools and the screens: published
//! whole by the brain each time it changes, read without a lock.

use serde::Serialize;

use crate::hands::Holder;

/// One deck as shown.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DeckView {
    pub number: u8,
    pub title: String,
    pub artist: String,
    pub bpm: f32,
    pub beat: Option<f64>,
    pub bars_left: Option<f64>,
    pub playing: bool,
    pub on_air: bool,
}

/// The next track as shown.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct NextView {
    pub title: String,
    pub artist: String,
    pub why: String,
    pub requested_by: Option<String>,
    /// How it will be mixed, once planned.
    pub style: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Live {
    pub holder: Holder,
    pub decks: Vec<DeckView>,
    pub next: Option<NextView>,
    /// The requests waiting: `who: artist — title`.
    pub requests: Vec<String>,
    pub automix: bool,
    /// The last fault, while it stands.
    pub fault: Option<String>,
    /// Why the deck reader cannot see, when it cannot.
    pub blind: Option<String>,
    pub target_energy: f32,
    pub landing: f32,
    /// Seconds to midnight, negative after it.
    pub to_midnight_s: i64,
    /// The line for the screens, newest: a choice's reason, a dedication.
    pub banner: Option<String>,
    /// Whether music is playing.
    pub live: bool,
    /// Whether rekordbox is running: while it is, this machine is reserved
    /// for the decks, and keep's guard reads so (`http::keep_status`).
    pub rekordbox: bool,
}
