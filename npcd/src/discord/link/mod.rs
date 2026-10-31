//! A person's Discord account, linked to their account here
//! (`docs/npcd_discord.md` §3.2) — so a DM they send a character's bot is
//! heard in the world as them.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`index`] | `discord/user/<id>`: which account a Discord user is |
//! | [`code`] | `discord/link/<code>`: the six digits a person links with |
//! | [`linking`] | the account's field and the index, written together |
//! | [`routes`] | `POST /v1/me/discord/link`, `GET`/`DELETE /v1/me/discord` |

pub mod code;
pub mod index;
pub mod linking;
pub mod routes;
