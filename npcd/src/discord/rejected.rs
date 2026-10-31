//! A bot whose token Discord rejected mid-call.

use crate::discord::error::DiscordError;

/// Discord rejected the bot's token mid-call (a `401`): the bot is refused,
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
