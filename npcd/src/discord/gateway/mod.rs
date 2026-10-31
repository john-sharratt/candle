//! A bot's connection to Discord's gateway (`docs/npcd_discord.md` §6): how
//! Discord hands a bot its DMs.
//!
//! Written on `tokio-tungstenite` rather than a Discord library, because the
//! one thing it must do beyond the protocol is point at a fake gateway in a
//! test, and a bot that only hears its DMs needs only `HELLO`, `IDENTIFY`,
//! heartbeats, `READY`, `MESSAGE_CREATE`, `RESUME` and the close codes.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`protocol`] | the payloads, the intents, what a close code means |
//! | [`shard`] | one bot's connection, kept open, and what it hears |
//!
//! A test's fake of the gateway is `fake`, built only for tests.

#[cfg(test)]
pub mod fake;
pub mod protocol;
pub mod shard;
