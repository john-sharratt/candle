//! A DM to a character's bot, heard in the world (`docs/npcd_discord.md`
//! §5.2, the DM half).
//!
//! In order, a message is:
//!
//! 1. dropped if it was written in a server (a server's channels are heard
//!    by [`crate::discord::server::inbound`]), by a bot or a webhook, or by
//!    this bot itself — the echo of its own post;
//! 2. a link, if it is exactly a live link code: the author's Discord account
//!    is linked to the code's account, and they are told so;
//! 3. answered once and dropped if its author is linked to no account, or to
//!    one that may not see the character — who may DM a character is who may
//!    see it, the check `owned_by` makes;
//! 4. dropped if it is already in the world: at or below the DM's
//!    `last_seen_discord`, or recorded as an origin on the thread;
//! 5. otherwise the `WorldOp::Message` the console's `post_message` makes,
//!    from the name the author speaks by, with the Discord message as its
//!    `origin` — and the DM's record written or advanced.
//!
//! What it does to the world it does as a write to the world's authority:
//! this runs on the authority, so the write is made in place.

use std::sync::Arc;

use web::auth::Role;

use super::record::Dm;
use super::told::Told;
use crate::accounts::{public_id, DiscordUser};
use crate::api::{now_ms, Authored};
use crate::authority::ops::WorldOp;
use crate::discord::link::{code, index, linking};
use crate::discord::message::DiscordMessage;
use crate::discord::online::Online;
use crate::discord::rejected::Rejected;
use crate::discord::trace::note;
use crate::discord::trace::note::{Direction, Note};
use crate::engine::speaking_as;
use crate::sim::phone::Origin;

/// What became of a message, for the trace and the tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Heard {
    /// The bot's own post, come back.
    Echo,
    /// Not the bot's to hear here: a server's, another bot's, a webhook's.
    Ignored,
    /// Its author was linked, as the account speaking by this name.
    Linked { name: String },
    /// Its author is linked to no account; told how to link if `told`.
    Unlinked { told: bool },
    /// Its author may not talk with the character; told so if `told`.
    Refused { told: bool },
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
            Heard::Linked { .. } => "link_code",
            Heard::Unlinked { told: true } => "unlinked_told",
            Heard::Unlinked { told: false } => "unlinked_dropped",
            Heard::Refused { told: true } => "refused_visibility_told",
            Heard::Refused { told: false } => "refused_visibility_dropped",
            Heard::Again => "duplicate",
            Heard::Applied { .. } => "applied",
            Heard::Unapplied => "unapplied",
        }
    }
}

/// Hear `m`, a message on `bot`'s shard, and say what became of it.
pub async fn hear(
    a: &Arc<Authored>,
    bot: &Online,
    told: &mut Told,
    m: &DiscordMessage,
) -> Result<Heard, Rejected> {
    let heard = judge(a, bot, told, m).await?;
    let mut n = Note::new(now_ms(), Direction::In, "dm", heard.outcome())
        .message(m.id)
        .channel(m.channel_id)
        .author(m.author.id);
    if let Heard::Applied { thread, seq, .. } = &heard {
        n = n.thread(thread).seq(*seq);
    }
    note(a, bot.npc, n);
    Ok(heard)
}

/// What becomes of `m`.
async fn judge(
    a: &Arc<Authored>,
    bot: &Online,
    told: &mut Told,
    m: &DiscordMessage,
) -> Result<Heard, Rejected> {
    if m.author.id == bot.user_id {
        return Ok(Heard::Echo);
    }
    if m.guild_id.is_some() || m.webhook_id.is_some() || m.author.bot {
        return Ok(Heard::Ignored);
    }
    let shared = a.npcs.substrate();
    let author = m.author.id.to_string();
    let content = m.content.trim();

    if code::is_code(content) {
        let redeemed = code::redeem(&shared, content, now_ms()).unwrap_or_else(|e| {
            tracing::error!(
                npc = bot.npc,
                "discord: a link code was not redeemed: {e:#}"
            );
            None
        });
        if let Some(account) = redeemed {
            let user = DiscordUser {
                id: author.clone(),
                username: m.author.username.clone(),
                linked_ms: now_ms(),
            };
            let name = {
                let accounts = a.accounts.read().await;
                if let Err(e) = linking::link(&accounts, &shared, &account, &user) {
                    tracing::error!(npc = bot.npc, "discord: a link was not written: {e:#}");
                    return Ok(Heard::Unapplied);
                }
                accounts
                    .identity(&account)
                    .map(|id| speaking_as(&id, &public_id(&account), &a.roles))
                    .unwrap_or_default()
            };
            told.forget(m.author.id);
            tracing::info!(npc = bot.npc, user = %author, "discord: an account was linked");
            reply(
                a,
                bot,
                m.channel_id,
                &format!("Linked — you are {name} here."),
            )
            .await?;
            return Ok(Heard::Linked { name });
        }
    }

    let Some(account) = index::account_of(&shared, &author) else {
        let first = told.first_unlinked(m.author.id);
        if first {
            let console = a.bots.console();
            let how = format!(
                "I don't know you yet. Open {console}/#/profile, choose Link Discord, and send \
                 me the code."
            );
            reply(a, bot, m.channel_id, &how).await?;
        }
        return Ok(Heard::Unlinked { told: first });
    };
    let identity = a.accounts.read().await.identity(&account);
    let handle = public_id(&account);
    let may = identity.as_ref().is_some_and(|id| {
        a.roles.of(Some(id)).at_least(Role::User)
            && a.npcs.view().visible_to(bot.npc, &handle).is_some()
    });
    let Some(identity) = identity.filter(|_| may) else {
        let first = told.first_refused(m.author.id);
        if first {
            reply(a, bot, m.channel_id, "I can't talk with you").await?;
        }
        return Ok(Heard::Refused { told: first });
    };

    let Some((hosted, body)) = a
        .runtime
        .as_ref()
        .and_then(|rt| rt.body_of(bot.npc))
        .filter(|(hosted, _)| hosted.id() == bot.world)
    else {
        return Ok(Heard::Unapplied);
    };
    let from = speaking_as(&identity, &handle, &a.roles);
    let Some(character) = hosted.read(|w| w.actor(&body).map(|a| a.name.clone())) else {
        return Ok(Heard::Unapplied);
    };
    let origin = Origin::Discord {
        message: m.id,
        channel: m.channel_id,
    };
    let before = hosted.dms(|d| d.get(bot.npc, &account));
    let recorded = hosted.sim(|s| {
        s.threads
            .direct_between(&from, &character)
            .is_some_and(|t| t.messages.iter().any(|x| x.origin == Some(origin)))
    });
    if before.is_some_and(|dm| m.id <= dm.last_seen_discord) || recorded {
        return Ok(Heard::Again);
    }

    let text = m.text();
    if !text.is_empty() {
        let op = WorldOp::Message {
            npc: bot.npc,
            from: from.clone(),
            text,
            origin: Some(origin),
        };
        let rt = a.runtime.as_ref().expect("a body is the runtime's");
        if let Err(e) = rt.world_write(&bot.world, op).await {
            tracing::warn!(
                npc = bot.npc,
                "discord: a DM did not reach the world: {e:#}"
            );
            return Ok(Heard::Unapplied);
        }
    }
    // A DM newly recorded starts its mirror at the thread's end: what was
    // said in the console before is the console's.
    let (thread, through) = hosted.sim(|s| {
        s.threads
            .direct_between(&from, &character)
            .map(|t| (t.id.clone(), t.messages.last().map_or(0, |last| last.seq)))
            .unwrap_or_default()
    });
    let dm = Dm {
        channel_id: m.channel_id,
        user_id: m.author.id,
        mirrored_through: before.map_or(through, |dm| dm.mirrored_through),
        last_seen_discord: m.id,
    };
    hosted.with_dms(|d| d.put(bot.npc, &account, dm));
    Ok(Heard::Applied {
        from,
        thread,
        seq: through,
    })
}

/// Answer in `channel` as `bot`.
async fn reply(a: &Authored, bot: &Online, channel: u64, text: &str) -> Result<(), Rejected> {
    match a.discord.create_message(&bot.token, channel, text).await {
        Ok(_) => Ok(()),
        Err(e) => Rejected::or_logged(bot.npc, "a reply", e),
    }
}
