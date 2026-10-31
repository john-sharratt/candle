//! A character's own Discord bot account (`docs/npcd_discord.md` §4.3).
//!
//! Discord has no API that makes a bot, so a character's owner makes one by
//! hand in the Developer Portal and pastes its token into the console. npcd
//! does the rest: it checks the token, names and pictures the bot as the
//! character, seals the token and keeps it as the replicated record
//! `discord/bot/<npc_id>`, so every member can take the bot over with the
//! world.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`seal`] | sealing and opening a bot token |
//! | [`store`] | the record of a character's bot |
//! | [`client`] | the Discord REST calls a connect makes |
//! | [`error`] | what a Discord call can answer instead |
//! | [`profile`] | the name and avatar a bot is given from its character |
//! | [`invite`] | the link that adds a bot to a server, with its permissions |
//! | [`routes`] | `GET`/`PUT`/`DELETE /v1/npc/:nid/discord` |
//!
//! A test's fake of Discord's API is `fake`, built only for tests.

pub mod client;
pub mod error;
#[cfg(test)]
mod fake;
pub mod invite;
pub mod profile;
pub mod routes;
pub mod seal;
pub mod store;
