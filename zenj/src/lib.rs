//! ZenJ — Zen as DJ (`docs/zenj.md`): the brain and hands above rekordbox.
//!
//! | Module | What |
//! |---|---|
//! | [`music`] | A track: its key on the Camelot wheel, its profile |
//! | [`control`] | Every control a mixer has, by one name |
//! | [`plan`] | A transition plan: moves at beats |
//! | [`beat`] | The beat as a clock: predicted, coasting, and when it jumps |
//! | [`brain`] | The set plan, the shortlist, Zen's choice, transition plans |
//! | [`conductor`] | The per-beat loop: plans run, loads confirmed, the watchdog |
//! | [`hands`] | A person and ZenJ on one mixer: who holds it, taking back, the walk back |
//! | [`events`] | What happened, in order |
//! | [`midi`] | The wire, controller maps, the ports |
//! | [`deck`] | rekordbox's live deck state, read from its memory |
//! | [`rekordbox`] | rekordbox's library and analysis files; opening and closing it |
//! | [`bus`] | The house's asks of rekordbox, over the gateway's bus |
//! | [`lights`] | The WiZ bulbs on the beat: fixtures, scenes, the rate gate, the thread |
//! | [`room`] | The FLX4's output captured, its levels, how it is landing |
//! | [`oauth`] | Signing in to Spotify and TIDAL |
//! | [`spotify`] | Taste and requests |
//! | [`tidal`] | The audio's catalogue and ZenJ's playlists |
//! | [`library`] | The tracks ZenJ plays from, built ahead of the night |
//! | [`live`] | What the night looks like now, for the tools and screens |
//! | [`tools`] | The tools on the `/mcp` surface |
//! | [`http`] | The routes: tools, status, events, keep's guard |
//! | [`presence`] | The cluster socket, as the app `zenj` |
//! | [`screens`] | The scenes and ZenJ's layer on every output |
//! | [`config`] | The machine's file |
//! | [`zen`] | Asking Zen, through the gateway, as the service `zenj` |

pub mod beat;
pub mod brain;
pub mod bus;
pub mod conductor;
pub mod config;
pub mod control;
pub mod deck;
pub mod events;
pub mod hands;
pub mod http;
pub mod library;
pub mod lights;
pub mod live;
pub mod midi;
pub mod music;
pub mod oauth;
pub mod plan;
pub mod presence;
pub mod rekordbox;
pub mod room;
pub mod screens;
pub mod spotify;
pub mod tidal;
pub mod tools;
pub mod zen;
