//! What a track is: its key on the Camelot wheel ([`key`]) and its profile
//! — grid, phrases, cues, energy, whose it is ([`track`]).

pub mod key;
pub mod track;

pub use key::{Camelot, KeyFit};
pub use track::{Grid, HotCue, Liked, Phrase, PhraseKind, Profile, TrackId};
