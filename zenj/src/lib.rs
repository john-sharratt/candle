//! ZenJ — Zen as DJ (`docs/zenj.md`): the brain and hands above rekordbox.
//!
//! | Module | What |
//! |---|---|
//! | [`music`] | A track: its key on the Camelot wheel, its profile |
//! | [`control`] | Every control a mixer has, by one name |
//! | [`plan`] | A transition plan: moves at beats |
//! | [`brain`] | The set plan, the shortlist, Zen's choice, transition plans |
//! | [`midi`] | The wire, controller maps, the ports |
//! | [`deck`] | rekordbox's live deck state, read from its memory |
//! | [`rekordbox`] | rekordbox's library and analysis files |
//! | [`oauth`] | Signing in to Spotify and TIDAL |
//! | [`spotify`] | Taste and requests |
//! | [`tidal`] | The audio's catalogue and ZenJ's playlists |
//! | [`library`] | The tracks ZenJ plays from, built ahead of the night |

pub mod brain;
pub mod control;
pub mod deck;
pub mod library;
pub mod midi;
pub mod music;
pub mod oauth;
pub mod plan;
pub mod rekordbox;
pub mod spotify;
pub mod tidal;
