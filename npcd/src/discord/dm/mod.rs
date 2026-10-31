//! DMs with a character's own bot (`docs/npcd_discord.md` §5, the DM halves):
//! a person's DM heard in the world, the character's answers posted back.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`record`] | `discord/dm/<npc_id>/<account>`: a DM, as the world keeps it |
//! | [`online`] | a bot, as its DMs need it |
//! | [`told`] | who a bot has answered instead of hearing |
//! | [`inbound`] | a DM heard in the world |
//! | [`outbound`] | the character's answers posted to its DMs |
//! | [`split`] | a message cut into posts Discord will take |
//! | [`gap`] | what was written while no member ran the bot |

pub mod gap;
pub mod inbound;
pub mod online;
pub mod outbound;
pub mod record;
pub mod split;
pub mod told;

use crate::discord::error::DiscordError;

/// Discord rejected the bot's token mid-DM (a `401`): the bot is refused,
/// and its shard is to stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rejected;

impl Rejected {
    /// `e`, from `what` of character `npc`'s bot: a rejected token said so,
    /// anything else logged and gone past.
    pub fn or_logged(npc: u64, what: &str, e: DiscordError) -> Result<(), Rejected> {
        match e {
            DiscordError::TokenRejected => Err(Rejected),
            e => {
                tracing::warn!(npc, "discord: {what} was not sent: {e}");
                Ok(())
            }
        }
    }
}
