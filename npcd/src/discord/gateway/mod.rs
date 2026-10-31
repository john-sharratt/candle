//! A bot's connection to Discord's gateway (`docs/npcd_discord.md` §6): how
//! Discord hands a bot its DMs and its servers' channels.
//!
//! Written on `tokio-tungstenite` rather than a Discord library, because the
//! one thing it must do beyond the protocol is point at a fake gateway in a
//! test, and a bot needs only `HELLO`, `IDENTIFY`, heartbeats, `READY`,
//! `RESUME`, the close codes, `MESSAGE_CREATE`, and the server and channel
//! events.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`protocol`] | the payloads, the intents, what a close code means |
//! | [`guild`] | a server and its channels, as the events carry them |
//! | [`shard`] | one bot's connection, kept open, and what it hears |
//!
//! A test's fake of the gateway is `fake`, built only for tests.

#[cfg(test)]
pub mod fake;
pub mod guild;
pub mod protocol;
pub mod shard;
