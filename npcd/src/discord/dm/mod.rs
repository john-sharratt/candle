//! DMs with a character's own bot (`docs/npcd_discord.md` §5, the DM halves):
//! a person's DM heard in the world, the character's answers posted back.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`record`] | `discord/dm/<npc_id>/<account>`: a DM, as the world keeps it |
//! | [`told`] | who a bot has answered instead of hearing |
//! | [`inbound`] | a DM heard in the world |
//! | [`outbound`] | the character's answers posted to its DMs |
//! | [`gap`] | what was written while no member ran the bot |

pub mod gap;
pub mod inbound;
pub mod outbound;
pub mod record;
pub mod told;
