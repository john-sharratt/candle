//! The channels a world is mirrored into, found or made, and their persona
//! webhooks (`docs/npcd_discord.md` §3, §3.1).
//!
//! The world's open channel is `#the-channel`. The listener finds it among
//! the server's channels by name, or makes it, and records it as the
//! channel's mirror — with its cursor at the thread's end, so what the world
//! said before the server was set is not posted, and its `last_seen_discord`
//! the lowest snowflake of that moment, so what the channel held before is
//! not read back into the world. Each mirrored channel has a webhook, made by
//! the listener and named for the world, that a character without a bot of
//! its own speaks through; its token is sealed onto the world's record.
//!
//! A channel already recorded is not looked for again. Every call is the
//! listener's; a failure is said in its trace, and the beat tries again.

use std::sync::Arc;

use super::record::{hook_name, Guild, Hook, Kind, Mirrored};
use crate::api::{now_ms, Authored};
use crate::discord::client::{Overwrite, Webhook};
use crate::discord::message::snowflake_at;
use crate::discord::online::Online;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::trace::{failed, note};
use crate::sim::phone::CHANNEL;
use crate::world::Hosted;

/// What `#the-channel` is called on Discord.
pub const CHANNEL_NAME: &str = "the-channel";

/// Discord's type for a text channel.
const TEXT: u8 = 0;

/// The world's open channel's thread, when anybody is on it.
pub fn channel_thread(hosted: &Hosted) -> Option<String> {
    hosted.sim(|s| {
        s.threads
            .iter()
            .find(|t| t.permanent && t.name == CHANNEL)
            .map(|t| t.id.clone())
    })
}

/// The seq of thread `thread`'s last message; `0` for none.
pub fn last_seq(hosted: &Hosted, thread: &str) -> u64 {
    hosted.sim(|s| {
        s.threads
            .get(thread)
            .and_then(|t| t.messages.last())
            .map_or(0, |m| m.seq)
    })
}

/// Find or make every mirrored channel of `guild`, and their webhooks.
/// `false` when a call failed: the beat posts nothing until they stand.
pub async fn ensure(a: &Arc<Authored>, bot: &Online, hosted: &Hosted, guild: &Guild) -> bool {
    let Some(thread) = channel_thread(hosted) else {
        return true;
    };
    let channel_id = match hosted.mirror(|m| m.thread(&thread)) {
        Some(mirrored) => mirrored.channel_id,
        None => {
            let Some(channel_id) = find_or_make_with(a, bot, guild, CHANNEL_NAME, &[]).await else {
                return false;
            };
            let mirrored = Mirrored::new(
                channel_id,
                Kind::Channel,
                last_seq(hosted, &thread),
                snowflake_at(now_ms()),
            );
            hosted.with_mirror(|m| m.put_thread(&thread, mirrored));
            note(
                a,
                bot.npc,
                Note::new(now_ms(), Direction::Out, "mirror", "channel_recorded")
                    .guild(guild.guild_id)
                    .channel(channel_id)
                    .thread(&thread),
            );
            channel_id
        }
    };
    hook(a, bot, hosted, guild, channel_id).await.is_some()
}

/// Channel `channel_id`'s webhook, made when the world has none for it;
/// `None` when it could not be made.
pub async fn hook(
    a: &Authored,
    bot: &Online,
    hosted: &Hosted,
    guild: &Guild,
    channel_id: u64,
) -> Option<Webhook> {
    if let Some(held) = hosted.mirror(|m| m.hook(channel_id)) {
        match a
            .discord_tokens
            .open_for(&hook_name(channel_id), &held.token)
        {
            Ok(token) => {
                return Some(Webhook {
                    id: held.webhook_id,
                    token,
                })
            }
            Err(why) => tracing::warn!(
                channel_id,
                "discord: a webhook's token does not open here ({why:?}); making another"
            ),
        }
    }
    let made = match a
        .discord
        .create_webhook(&bot.token, channel_id, &guild.name)
        .await
    {
        Ok(made) => made,
        Err(e) => {
            failed(
                a,
                bot.npc,
                Note::new(
                    now_ms(),
                    Direction::Out,
                    "mirror",
                    &format!("webhook not made: {e}"),
                )
                .channel(channel_id),
            );
            return None;
        }
    };
    let sealed = a
        .discord_tokens
        .seal_for(&hook_name(channel_id), &made.token);
    hosted.with_mirror(|m| {
        m.put_hook(
            channel_id,
            Hook {
                webhook_id: made.id,
                token: sealed,
            },
        )
    });
    note(
        a,
        bot.npc,
        Note::new(now_ms(), Direction::Out, "mirror", "webhook_made").channel(channel_id),
    );
    Some(made)
}

/// The id of `guild`'s text channel called `name`, made with `overwrites`
/// when it has none; `None` when Discord answered neither.
pub async fn find_or_make_with(
    a: &Authored,
    bot: &Online,
    guild: &Guild,
    name: &str,
    overwrites: &[Overwrite],
) -> Option<u64> {
    let fail = |what: &str, e: String| {
        failed(
            a,
            bot.npc,
            Note::new(now_ms(), Direction::Out, "mirror", &format!("{what}: {e}"))
                .guild(guild.guild_id),
        );
    };
    let channels = match a.discord.guild_channels(&bot.token, guild.guild_id).await {
        Ok(channels) => channels,
        Err(e) => {
            fail("channels not read", e.to_string());
            return None;
        }
    };
    if let Some(found) = channels.iter().find(|c| c.kind == TEXT && c.name == name) {
        return Some(found.id);
    }
    match a
        .discord
        .create_channel(&bot.token, guild.guild_id, name, overwrites)
        .await
    {
        Ok(id) => Some(id),
        Err(e) => {
            fail("channel not made", e.to_string());
            None
        }
    }
}
