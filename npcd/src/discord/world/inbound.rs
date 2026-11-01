//! A message in a mirrored channel of the world's server, heard in the world
//! (`docs/npcd_discord.md` §5.2).
//!
//! Heard on the listener's shard alone; every other bot in the server drops
//! the server's messages ([`super::is_world_server`]). In order, a message is:
//!
//! 1. dropped if any bot or webhook wrote it — the bridge's own posts among
//!    them, which is what keeps a post from coming back as a message;
//! 2. dropped if its channel mirrors no thread;
//! 3. dropped if it is already in the world: at or below the channel's
//!    `last_seen_discord`, or recorded as an origin on its thread;
//! 4. dropped if its author is linked to no account, or to one without the
//!    Creator role — the role the console's own route asks of anyone
//!    speaking on the open channel (§3.2);
//! 5. otherwise said on the open channel as that person, as the console's
//!    route says it, with the Discord message as its origin.
//!
//! The channel's `last_seen_discord` moves past every message judged past
//! the first three steps, so the gap fetch reads none of them again.

use std::sync::Arc;

use super::record::Kind;
use crate::accounts::public_id;
use crate::api::{now_ms, Authored};
use crate::authority::ops::WorldOp;
use crate::discord::link::index;
use crate::discord::message::DiscordMessage;
use crate::discord::online::Online;
use crate::discord::server::author::written_out;
use crate::discord::server::embodied;
use crate::discord::trace::note;
use crate::discord::trace::note::{Direction, Note};
use crate::engine::speaking_as;
use crate::sim::phone::Origin;
use crate::world::Hosted;

/// What became of a message, for the trace and the tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Heard {
    /// A bot's or a webhook's — the bridge's own posts among them.
    Ignored,
    /// Its channel mirrors no thread.
    NoThread,
    /// It was in the world already.
    Again,
    /// Its author is linked to no account.
    Unlinked,
    /// Its author's account may not speak on the open channel.
    NotCreator,
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
            Heard::Ignored => "bot_or_webhook_dropped",
            Heard::NoThread => "no_thread",
            Heard::Again => "duplicate",
            Heard::Unlinked => "unlinked_dropped",
            Heard::NotCreator => "not_creator_dropped",
            Heard::Applied { .. } => "applied",
            Heard::Unapplied => "unapplied",
        }
    }
}

/// Hear `m`, a message of `hosted`'s server on `bot`'s shard — the
/// listener's — and say what became of it.
pub async fn hear(a: &Arc<Authored>, bot: &Online, hosted: &Hosted, m: &DiscordMessage) -> Heard {
    let heard = judge(a, bot, hosted, m).await;
    let mut n = Note::new(now_ms(), Direction::In, "mirror", heard.outcome())
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
async fn judge(a: &Arc<Authored>, bot: &Online, hosted: &Hosted, m: &DiscordMessage) -> Heard {
    if m.webhook_id.is_some() || m.author.bot {
        return Heard::Ignored;
    }
    let Some((thread, mirrored)) = hosted.mirror(|x| x.by_channel(m.channel_id)) else {
        return Heard::NoThread;
    };
    let origin = Origin::Discord {
        message: m.id,
        channel: m.channel_id,
    };
    let recorded = hosted.sim(|s| {
        s.threads
            .get(&thread)
            .is_some_and(|t| t.messages.iter().any(|x| x.origin == Some(origin)))
    });
    if m.id <= mirrored.last_seen_discord || recorded {
        return Heard::Again;
    }
    let heard = speak(a, bot, &thread, mirrored.kind, m, origin).await;
    if heard != Heard::Unapplied {
        hosted.with_mirror(|x| {
            if let Some(mut now) = x.thread(&thread) {
                now.last_seen_discord = now.last_seen_discord.max(m.id);
                x.put_thread(&thread, now);
            }
        });
    }
    match heard {
        Heard::Applied { from, .. } => {
            let seq = hosted.sim(|s| {
                s.threads
                    .get(&thread)
                    .and_then(|t| t.messages.last())
                    .map_or(0, |m| m.seq)
            });
            Heard::Applied { from, thread, seq }
        }
        other => other,
    }
}

/// Say `m` on `thread`, a mirror of `kind`, as its author.
async fn speak(
    a: &Arc<Authored>,
    bot: &Online,
    thread: &str,
    kind: Kind,
    m: &DiscordMessage,
    origin: Origin,
) -> Heard {
    let Some(account) = index::account_of(&a.npcs.substrate(), &m.author.id.to_string()) else {
        return Heard::Unlinked;
    };
    let Some(identity) = a.accounts.read().await.identity(&account) else {
        return Heard::Unlinked;
    };
    if !a.roles.of(Some(&identity)).is_creator() {
        return Heard::NotCreator;
    }
    let from = speaking_as(&identity, &public_id(&account), &a.roles);
    let character = embodied(a, bot).map(|(_, name)| name).unwrap_or_default();
    let text = written_out(&m.text(), &m.mentions, bot.user_id, &character);
    if text.is_empty() {
        return Heard::Applied {
            from,
            thread: thread.to_string(),
            seq: 0,
        };
    }
    let op = match kind {
        Kind::Channel => WorldOp::Channel {
            from: from.clone(),
            text,
            origin: Some(origin),
        },
    };
    let rt = a.runtime.as_ref().expect("a listener is the runtime's");
    if let Err(e) = rt.world_write(&bot.world, op).await {
        tracing::warn!(
            npc = bot.npc,
            "discord: a message of the world's server did not reach the world: {e:#}"
        );
        return Heard::Unapplied;
    }
    Heard::Applied {
        from,
        thread: thread.to_string(),
        seq: 0,
    }
}
