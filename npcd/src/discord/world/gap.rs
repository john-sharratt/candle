//! What was written in the world's server while no member ran its listener
//! (`docs/npcd_discord.md` §5.3): once the listener's shard reaches `READY`,
//! each mirrored channel is read after the last message taken in from it —
//! one page of a hundred — and what is there is heard, oldest first, as if it
//! had just arrived.

use std::sync::Arc;

use super::inbound::{hear, Heard};
use super::listened;
use crate::api::Authored;
use crate::discord::online::Online;
use crate::discord::rejected::Rejected;

/// Hear what the mirrored channels of the server `bot` listens to missed;
/// what became of each message heard. Nothing for a bot that listens to none.
pub async fn fetch(a: &Arc<Authored>, bot: &Online) -> Result<Vec<Heard>, Rejected> {
    let Some((hosted, guild)) = listened(a, bot) else {
        return Ok(Vec::new());
    };
    let mut heard = Vec::new();
    for (_, mirrored) in hosted.mirror(|m| m.threads()) {
        let missed = match a
            .discord
            .messages_after(&bot.token, mirrored.channel_id, mirrored.last_seen_discord)
            .await
        {
            Ok(missed) => missed,
            Err(e) => {
                Rejected::or_logged(bot.npc, "a mirrored channel's read", e)?;
                continue;
            }
        };
        // A message read back over REST does not name its server, as one
        // dispatched on the gateway does: it is the world's.
        for mut m in missed {
            m.guild_id.get_or_insert(guild.guild_id);
            heard.push(hear(a, bot, &hosted, &m).await);
        }
    }
    Ok(heard)
}
