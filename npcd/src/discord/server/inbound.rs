//! A message in one of a bot's servers' channels, heard in the world
//! (`docs/npcd_discord.md` §3.4, §5.2).
//!
//! In order, a message is:
//!
//! 1. dropped if it is not a server's, or was written by any bot or webhook
//!    — this bot's own post among them — so two bots cannot talk each other
//!    into a loop;
//! 2. dropped if its channel has no open thread: one that is not a text
//!    channel the bot was told of (a thread, a voice channel's chat), or one
//!    closed since;
//! 3. dropped if it is already in the world: at or below the channel's
//!    `last_seen_discord`, or recorded as an origin on its thread;
//! 4. otherwise `WorldOp::ChannelMessage`, from the name its author speaks
//!    by when linked and `<display name> (@<username>)` when not, its
//!    mentions written out by name and its attachments as link lines, said
//!    to the character when it mentions or answers the bot or names the
//!    character — and the channel's `last_seen_discord` advanced.
//!
//! Anyone the server lets write there is heard, linked or not: the server's
//! members chose to invite the bot.

use std::sync::Arc;

use super::author::{unlinked_name, written_out};
use super::{embodied, thread};
use crate::accounts::public_id;
use crate::api::{now_ms, Authored};
use crate::authority::ops::WorldOp;
use crate::discord::link::index;
use crate::discord::message::DiscordMessage;
use crate::discord::online::Online;
use crate::discord::trace::note;
use crate::discord::trace::note::{Direction, Note};
use crate::engine::speaking_as;
use crate::sim::phone::Origin;

/// What became of a message, for the trace and the tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Heard {
    /// The bot's own post, come back.
    Echo,
    /// Not the bot's to hear: not a server's, or another bot's or a
    /// webhook's.
    Ignored,
    /// Its channel is not one the character talks in.
    NoThread,
    /// It was in the world already.
    Again,
    /// It is in the world now, from this name, as message `seq` of thread
    /// `thread`.
    Applied {
        from: String,
        thread: String,
        seq: u64,
    },
    /// The world did not take it; the gap fetch will offer it again.
    Unapplied,
}

impl Heard {
    /// What became of it, as the trace says it.
    pub fn outcome(&self) -> &'static str {
        match self {
            Heard::Echo => "echo",
            Heard::Ignored => "bot_or_webhook_dropped",
            Heard::NoThread => "no_thread",
            Heard::Again => "duplicate",
            Heard::Applied { .. } => "applied",
            Heard::Unapplied => "unapplied",
        }
    }
}

/// Hear `m`, a server message on `bot`'s shard, and say what became of it.
pub async fn hear(a: &Arc<Authored>, bot: &Online, m: &DiscordMessage) -> Heard {
    let heard = judge(a, bot, m).await;
    let mut n = Note::new(now_ms(), Direction::In, "channel", heard.outcome())
        .message(m.id)
        .channel(m.channel_id)
        .author(m.author.id);
    if let Heard::Applied { thread, seq, .. } = &heard {
        n = n.thread(thread).seq(*seq);
    }
    note(a, bot.npc, n);
    heard
}

/// What becomes of `m`.
async fn judge(a: &Arc<Authored>, bot: &Online, m: &DiscordMessage) -> Heard {
    if m.author.id == bot.user_id {
        return Heard::Echo;
    }
    if m.guild_id.is_none() || m.webhook_id.is_some() || m.author.bot {
        return Heard::Ignored;
    }
    let Some((hosted, character)) = embodied(a, bot) else {
        return Heard::Unapplied;
    };
    let npc = bot.npc;
    let Some(record) = hosted
        .servers(|s| s.channel(npc, m.channel_id))
        .filter(|_| thread::is_open(&hosted, npc, m.channel_id))
    else {
        return Heard::NoThread;
    };
    let origin = Origin::Discord {
        message: m.id,
        channel: m.channel_id,
    };
    let recorded = hosted.sim(|s| {
        s.threads
            .get(&record.thread)
            .is_some_and(|t| t.messages.iter().any(|x| x.origin == Some(origin)))
    });
    if m.id <= record.last_seen_discord || recorded {
        return Heard::Again;
    }

    let from = author_name(a, m).await;
    let text = written_out(&m.text(), &m.mentions, bot.user_id, &character);
    if !text.is_empty() {
        let op = WorldOp::ChannelMessage {
            npc,
            channel: m.channel_id,
            from: from.clone(),
            text,
            origin: Some(origin),
            to: m
                .addresses(bot.user_id, &character)
                .then(|| character.clone()),
        };
        let rt = a.runtime.as_ref().expect("a body is the runtime's");
        if let Err(e) = rt.world_write(&bot.world, op).await {
            tracing::warn!(
                npc,
                "discord: a channel message did not reach the world: {e:#}"
            );
            return Heard::Unapplied;
        }
    }
    hosted.with_servers(|s| {
        if let Some(mut record) = s.channel(npc, m.channel_id) {
            record.last_seen_discord = record.last_seen_discord.max(m.id);
            s.put_channel(npc, m.channel_id, record);
        }
    });
    let seq = hosted.sim(|s| {
        s.threads
            .get(&record.thread)
            .and_then(|t| t.messages.last())
            .map_or(0, |m| m.seq)
    });
    Heard::Applied {
        from,
        thread: record.thread,
        seq,
    }
}

/// What `m`'s author is called in the world: the name their account speaks
/// by, when their Discord account is linked to one.
async fn author_name(a: &Authored, m: &DiscordMessage) -> String {
    let linked = index::account_of(&a.npcs.substrate(), &m.author.id.to_string());
    let Some(account) = linked else {
        return unlinked_name(m);
    };
    let identity = a.accounts.read().await.identity(&account);
    match identity {
        Some(identity) => speaking_as(&identity, &public_id(&account), &a.roles),
        None => unlinked_name(m),
    }
}
