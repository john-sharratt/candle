//! What was written in the world's server while no member ran its listener
//! (`docs/npcd_discord.md` §5.3): once the listener's shard reaches `READY`,
//! each mirrored channel is read after the last message taken in from it —
//! one page of a hundred — and what is there is heard, oldest first, as if it
//! had just arrived.

use std::sync::Arc;

use super::inbound::{hear, Heard};
use super::listened;
use super::record::Kind;
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
    // Each channel once, from its own cursor: a DM shares its character's
    // channel, whose cursor is the character's.
    let channels: Vec<(u64, u64)> = hosted.mirror(|m| {
        let threads = m
            .threads()
            .into_iter()
            .filter(|(_, t)| t.kind != Kind::Dm)
            .map(|(_, t)| (t.channel_id, t.last_seen_discord));
        let own = m
            .owns()
            .into_iter()
            .map(|(_, o)| (o.channel_id, o.last_seen_discord));
        threads.chain(own).collect()
    });
    for (channel_id, last_seen) in channels {
        let missed = match a
            .discord
            .messages_after(&bot.token, channel_id, last_seen)
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
