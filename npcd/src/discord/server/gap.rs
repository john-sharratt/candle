//! What was written in a bot's servers' channels while no member ran it
//! (`docs/npcd_discord.md` §5.2–5.3): once its shard reaches `READY`, each
//! recorded channel with an open thread is read after the last message taken
//! in from it — one page of a hundred — and what is there is heard, oldest
//! first, as if it had just arrived.

use std::sync::Arc;

use super::inbound::{hear, Heard};
use super::thread;
use crate::api::Authored;
use crate::discord::online::Online;
use crate::discord::rejected::Rejected;

/// Hear what `bot`'s channels missed; what became of each message heard.
pub async fn fetch(a: &Arc<Authored>, bot: &Online) -> Result<Vec<Heard>, Rejected> {
    let Some(hosted) = a.runtime.as_ref().and_then(|rt| rt.hosted.get(&bot.world)) else {
        return Ok(Vec::new());
    };
    let mut heard = Vec::new();
    for (channel_id, record) in hosted.servers(|s| s.channels_of(bot.npc)) {
        if !thread::is_open(&hosted, bot.npc, channel_id) {
            continue;
        }
        let missed = match a
            .discord
            .messages_after(&bot.token, channel_id, record.last_seen_discord)
            .await
        {
            Ok(missed) => missed,
            Err(e) => {
                Rejected::or_logged(bot.npc, "a channel's read", e)?;
                continue;
            }
        };
        // A message read back over REST does not name its server, as one
        // dispatched on the gateway does: it is the record's.
        for mut m in missed {
            m.guild_id.get_or_insert(record.guild_id);
            heard.push(hear(a, bot, &m).await);
        }
    }
    Ok(heard)
}
