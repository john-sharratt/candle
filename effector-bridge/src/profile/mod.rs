//! The device profiles (`docs/zen_moon_mind.md` §15): one module per kind of
//! device the world can work, each built on the shared protocol libraries
//! ([`crate::proto`]) and thin because of it. Each exposes a `work` that takes
//! an action and its arguments and does it, so [`crate::work`] can reach any
//! profile the same way.

pub mod androidtv;
pub mod cast;
pub mod receiver;
pub mod sonos;
pub mod webos;
pub mod wiz;
