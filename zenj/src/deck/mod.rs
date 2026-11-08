//! Live deck state, read from rekordbox's memory (§4.2): the offsets for its
//! release ([`offsets`]), reading another process ([`memory`]), what a
//! sample is believed to say ([`state`]), and the thread that reads it
//! ([`reader`]).

pub mod memory;
pub mod offsets;
pub mod reader;
pub mod state;

pub use state::{anlz_key, Deck, Snapshot, Tracks};
