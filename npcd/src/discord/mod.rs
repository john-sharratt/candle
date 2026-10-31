//! A character's own Discord bot (`docs/npcd_discord.md`).
//!
//! Discord has no API that makes a bot, so a character's owner makes one by
//! hand in the Developer Portal and pastes its token into the console. npcd
//! does the rest: it checks the token, names and pictures the bot as the
//! character, seals the token and keeps it as the replicated record
//! `discord/bot/<npc_id>`, so every member can take the bot over with the
//! world. The member that is the authority of the character's world keeps the
//! bot online, and a person who has linked their Discord account DMs the
//! character there.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`seal`] | sealing and opening a bot token |
//! | [`store`] | the record of a character's bot |
//! | [`state`] | whether a bot is online, offline or refused |
//! | [`client`] | the Discord REST calls a bot makes |
//! | [`error`] | what a Discord call can answer instead |
//! | [`limit`] | the host's share of Discord's invalid-request limit |
//! | [`message`] | a Discord message, as the gateway and the API carry it |
//! | [`profile`] | the name and avatar a bot is given from its character |
//! | [`invite`] | the link that adds a bot to a server, with its permissions |
//! | [`routes`] | `GET`/`PUT`/`DELETE /v1/npc/:nid/discord` |
//! | [`gateway`] | a bot's gateway connection |
//! | [`link`] | a person's Discord account, linked to their account |
//! | [`dm`] | DMs with a character's bot, both ways |
//! | [`bots`] | every bot this member runs, and their beat |
//!
//! A test's fake of Discord's API is `fake`, built only for tests.

pub mod bots;
pub mod client;
pub mod dm;
pub mod error;
#[cfg(test)]
mod fake;
pub mod gateway;
pub mod invite;
pub mod limit;
pub mod link;
pub mod message;
pub mod profile;
pub mod routes;
pub mod seal;
pub mod state;
pub mod store;
