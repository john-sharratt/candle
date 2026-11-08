//! ZenJ — Zen as DJ (`docs/zenj.md`): the brain and hands above rekordbox.
//!
//! | Module | What |
//! |---|---|
//! | [`music`] | A track: its key on the Camelot wheel, its profile |
//! | [`control`] | Every control a mixer has, by one name |
//! | [`plan`] | A transition plan: moves at beats |
//! | [`brain`] | The set plan, the shortlist, Zen's choice, transition plans |
//! | [`conductor`] | The per-beat loop: plans run, loads confirmed, the watchdog |
//! | [`hands`] | A person and ZenJ on one mixer: modes, taking back, the walk back |
//! | [`events`] | What happened, in order |
//! | [`midi`] | The wire, controller maps, the ports |
//! | [`deck`] | rekordbox's live deck state, read from its memory |
//! | [`rekordbox`] | rekordbox's library and analysis files |
//! | [`room`] | The FLX4's output captured, its levels, how it is landing |
//! | [`oauth`] | Signing in to Spotify and TIDAL |
//! | [`spotify`] | Taste and requests |
//! | [`tidal`] | The audio's catalogue and ZenJ's playlists |
//! | [`library`] | The tracks ZenJ plays from, built ahead of the night |
//! | [`live`] | What the night looks like now, for the tools and screens |
//! | [`zen`] | Asking Zen, through the gateway, as the service `zenj` |

pub mod brain;
pub mod conductor;
pub mod control;
pub mod deck;
pub mod events;
pub mod hands;
pub mod library;
pub mod live;
pub mod midi;
pub mod music;
pub mod oauth;
pub mod plan;
pub mod rekordbox;
pub mod room;
pub mod spotify;
pub mod tidal;
pub mod zen;
