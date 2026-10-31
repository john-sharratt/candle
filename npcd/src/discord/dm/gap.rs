//! What was written to a bot while no member ran it (`docs/npcd_discord.md`
//! §5.3, DMs): once its shard reaches `READY`, each of its DMs is read after
//! the last message taken in from it (`GET /channels/{id}/messages?after=`),
//! and what is there is heard, oldest first, as if it had just arrived.

use std::sync::Arc;

use super::inbound::{hear, Heard};
use super::online::Online;
use super::told::Told;
use super::Rejected;
use crate::api::Authored;

/// Hear what `bot`'s DMs missed; what became of each message heard.
pub async fn fetch(
    a: &Arc<Authored>,
    bot: &Online,
    told: &mut Told,
) -> Result<Vec<Heard>, Rejected> {
    let Some(hosted) = a.runtime.as_ref().and_then(|rt| rt.hosted.get(&bot.world)) else {
        return Ok(Vec::new());
    };
    let mut heard = Vec::new();
    for (_, dm) in hosted.dms(|d| d.of(bot.npc)) {
        let missed = match a
            .discord
            .messages_after(&bot.token, dm.channel_id, dm.last_seen_discord)
            .await
        {
            Ok(missed) => missed,
            Err(e) => {
                Rejected::or_logged(bot.npc, "a DM's read", e)?;
                continue;
            }
        };
        for m in &missed {
            heard.push(hear(a, bot, told, m).await?);
        }
    }
    Ok(heard)
}
