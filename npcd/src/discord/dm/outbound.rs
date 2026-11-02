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
//! again; anything else leaves the cursor where it is, so a gap is retried,
//! never skipped — after a hold that doubles with each refusal in a row
//! (`crate::discord::world::hold`), so a refused DM is not asked again on
//! every beat.
//!
//! A linked person the character writes to with no DM yet gets one opened
//! (`POST /users/@me/channels`): what the character has written to them
//! since the bot came online is posted there. Discord opens one only to a
//! person the bot shares a server with; refused that (`50278`), what was
//! written stays unposted, and Discord is not asked again until the
//! character writes again.

use std::collections::HashMap;
use std::sync::Arc;

use super::record::Dm;
use crate::accounts::public_id;
use crate::api::{now_ms, Authored};
use crate::discord::link::index;
use crate::discord::message::snowflake_at;
use crate::discord::online::Online;
use crate::discord::post::split::{split, LIMIT};
use crate::discord::post::{post as send, posted, Post, Target};
use crate::discord::rejected::Rejected;
use crate::discord::trace::note;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::world::hold::{FAILED_FOR, REFUSED_FOR};
use crate::engine::reported::spoken;
use crate::engine::speaking_as;
use crate::sim::phone::Message;
use crate::world::Hosted;

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
            .messages
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
                note(
                    a,
                    bot.npc,
                    Note::new(now_ms(), Direction::Out, "dm_open", "no_shared_server")
                        .author(user_id)
                        .seq(last.seq),
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
pub(super) async fn speaking_name(a: &Authored, account: &str) -> Option<String> {
    let identity = a.accounts.read().await.identity(account)?;
    Some(speaking_as(&identity, &public_id(account), &a.roles))
}

/// What is waiting on a direct thread: its id, and its messages to post.
struct Waiting {
    thread: String,
    messages: Vec<Message>,
}

/// The messages of `person`'s direct thread with `character` past seq
/// `through` that came from the world, oldest first.
fn waiting(hosted: &Hosted, person: &str, character: &str, through: u64) -> Waiting {
    hosted.sim(|s| {
        s.threads
            .direct_between(person, character)
            .map(|t| Waiting {
                thread: t.id.clone(),
                messages: t
                    .messages
                    .iter()
                    .filter(|m| m.seq > through && m.origin.is_none())
                    .cloned()
                    .collect(),
            })
            .unwrap_or(Waiting {
                thread: String::new(),
                messages: Vec::new(),
            })
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
    waiting: Waiting,
) -> Result<(), Rejected> {
    // A DM Discord refused holds off as a world's mirror does
    // (`crate::discord::world::hold`): asked again on every beat, a person
    // who closed their DMs costs two invalid requests a second.
    let held = format!("dm/{}", dm.channel_id);
    if a.bots.mirror_held(&held) {
        return Ok(());
    }
    for m in waiting.messages {
        let text = match m.from == character {
            true => spoken(&m.intent),
            false => format!("> {} (console): {}", m.from, m.intent),
        };
        let to = Target {
            kind: "dm",
            channel: dm.channel_id,
            thread: &waiting.thread,
            seq: m.seq,
        };
        let parts = split(&text, LIMIT);
        let mut last = 0;
        for part in &parts {
            match send(a, bot, to, part).await? {
                Post::Posted(id) => last = id,
                Post::Forbidden | Post::Gone => {
                    a.bots.hold_mirror(&held, REFUSED_FOR);
                    return Ok(());
                }
                Post::Unposted => {
                    a.bots.hold_mirror(&held, FAILED_FOR);
                    return Ok(());
                }
            }
        }
        posted(a, bot, to, last, parts.len());
        dm.mirrored_through = m.seq;
        hosted.with_dms(|d| d.put(bot.npc, account, dm));
        a.bots.mirror_cleared(&held);
    }
    Ok(())
}
