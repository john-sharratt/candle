//! TIDAL, the audio (§4.1): the API v2 client that finds tracks and keeps
//! ZenJ's playlists, which rekordbox shows in its TIDAL tree.

pub mod api;

pub use api::{Item, Tidal, Track};
