//! A character's answers, posted to its DMs (`docs/npcd_discord.md` §5.1, the
//! DM half).
//!
//! Each beat of the bots, for each DM of a bot: every message of the
//! character's direct thread with the person past the DM's
//! `mirrored_through` that came from no outside origin is posted, in order —
//! the character's as the text it is, the person's own from the console as a
//! quoted line (`> Wren (console): …`), since a bot cannot post as a person —
//! each cut to Discord's 2,000 characters. The cursor moves past a message
//! only once all of it was posted. A `429` waits `retry_after` and tries
//! again; anything else leaves the cursor for the next beat, so a gap is
//! retried, never skipped.
//!
//! A linked person the character writes to with no DM yet gets one opened
//! (`POST /users/@me/channels`): what the character has written to them
//! since the bot came online is posted there. Discord opens one only to a
//! person the bot shares a server with; refused that (`50278`), what was
//! written stays unposted, and Discord is not asked again until the
//! character writes again.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::time::sleep;

use super::online::Online;
use super::record::Dm;
use super::split::{split, LIMIT};
use super::Rejected;
use crate::accounts::public_id;
use crate::api::{now_ms, Authored};
use crate::discord::error::DiscordError;
use crate::discord::link::index;
use crate::discord::message::snowflake_at;
use crate::engine::speaking_as;
use crate::sim::phone::Message;
use crate::world::Hosted;

/// How many times one post is tried against a `429` in a beat.
const RATE_TRIES: usize = 3;

/// The longest a post waits on one `429` before the next beat takes it.
const RATE_WAIT_MAX: Duration = Duration::from_secs(30);

/// For each linked person with no DM yet, the last seq of the character's
/// thread with them that Discord would not open a DM for.
pub type Unreachable = HashMap<String, u64>;

/// Post what is waiting in `bot`'s DMs.
pub async fn post(
    a: &Arc<Authored>,
    bot: &Online,
    unreachable: &mut Unreachable,
) -> Result<(), Rejected> {
    let Some((hosted, body)) = a
        .runtime
        .as_ref()
        .and_then(|rt| rt.body_of(bot.npc))
        .filter(|(hosted, _)| hosted.id() == bot.world)
    else {
        return Ok(());
    };
    let Some(character) = hosted.read(|w| w.actor(&body).map(|a| a.name.clone())) else {
        return Ok(());
    };

    for (account, dm) in hosted.dms(|d| d.of(bot.npc)) {
        let Some(person) = speaking_name(a, &account).await else {
            continue;
        };
        let waiting = waiting(&hosted, &person, &character, dm.mirrored_through);
        deliver(a, bot, &hosted, &account, dm, &character, waiting).await?;
    }

    for (discord_id, account) in index::all(&a.npcs.substrate()) {
        if hosted.dms(|d| d.get(bot.npc, &account)).is_some() {
            continue;
        }
        let Ok(user_id) = discord_id.parse::<u64>() else {
            continue;
        };
        let Some(person) = speaking_name(a, &account).await else {
            continue;
        };
        let written: Vec<Message> = waiting(&hosted, &person, &character, 0)
            .into_iter()
            .filter(|m| m.from == character && m.at_ms >= bot.since_ms)
            .collect();
        let (Some(first), Some(last)) = (written.first(), written.last()) else {
            continue;
        };
        // Asked once for what the character has written so far; asked again
        // only once it writes more.
        let tried = unreachable.get(&account).copied().unwrap_or(0);
        if last.seq <= tried {
            continue;
        }
        let channel_id = match a.discord.open_dm(&bot.token, user_id).await {
            Ok(channel_id) => channel_id,
            Err(e) if e.no_shared_server() => {
                tracing::info!(
                    npc = bot.npc,
                    user = user_id,
                    "discord: no DM opens with a person the bot shares no server with"
                );
                unreachable.insert(account, last.seq);
                continue;
            }
            Err(e) => {
                Rejected::or_logged(bot.npc, "a DM's opening", e)?;
                continue;
            }
        };
        unreachable.remove(&account);
        let dm = Dm {
            channel_id,
            user_id,
            mirrored_through: first.seq - 1,
            // Anything the person writes from now is newer than this.
            last_seen_discord: snowflake_at(now_ms()),
        };
        hosted.with_dms(|d| d.put(bot.npc, &account, dm));
        let waiting = waiting(&hosted, &person, &character, dm.mirrored_through);
        deliver(a, bot, &hosted, &account, dm, &character, waiting).await?;
    }
    Ok(())
}

/// The name account `account`'s person speaks by in the world.
async fn speaking_name(a: &Authored, account: &str) -> Option<String> {
    let identity = a.accounts.read().await.identity(account)?;
    Some(speaking_as(&identity, &public_id(account), &a.roles))
}

/// The messages of `person`'s direct thread with `character` past seq
/// `through` that came from the world, oldest first.
fn waiting(hosted: &Hosted, person: &str, character: &str, through: u64) -> Vec<Message> {
    hosted.sim(|s| {
        s.threads
            .direct_between(person, character)
            .map(|t| {
                t.messages
                    .iter()
                    .filter(|m| m.seq > through && m.origin.is_none())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    })
}

/// Post `waiting` to `dm`'s channel in order, moving its cursor past each
/// message once all of it is posted; stopping at the first that is not.
async fn deliver(
    a: &Authored,
    bot: &Online,
    hosted: &Hosted,
    account: &str,
    mut dm: Dm,
    character: &str,
    waiting: Vec<Message>,
) -> Result<(), Rejected> {
    for m in waiting {
        let text = match m.from == character {
            true => m.intent.clone(),
            false => format!("> {} (console): {}", m.from, m.intent),
        };
        for part in split(&text, LIMIT) {
            if !posted(a, bot, dm.channel_id, &part).await? {
                return Ok(());
            }
        }
        dm.mirrored_through = m.seq;
        hosted.with_dms(|d| d.put(bot.npc, account, dm));
    }
    Ok(())
}

/// Post `text` to `channel` as `bot`: whether it was posted. A `429` is
/// waited out and tried again, a few times.
async fn posted(a: &Authored, bot: &Online, channel: u64, text: &str) -> Result<bool, Rejected> {
    for _ in 0..RATE_TRIES {
        match a.discord.create_message(&bot.token, channel, text).await {
            Ok(_) => return Ok(true),
            Err(DiscordError::RateLimited { retry_after_s }) => {
                sleep(Duration::from_secs_f64(retry_after_s.max(0.0)).min(RATE_WAIT_MAX)).await;
            }
            Err(e) => {
                Rejected::or_logged(bot.npc, "a DM post", e)?;
                return Ok(false);
            }
        }
    }
    Ok(false)
}
