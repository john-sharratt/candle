//! A world's own Discord server, and the world mirrored into it
//! (`docs/npcd_discord.md` §3.1, §4.4, §5).
//!
//! A world may have a server, set in the console by a Creator
//! ([`routes`]). Its **listener** — a character of the world with a bot of its
//! own — reads the server; the world's conversations are posted there. The
//! world's open channel is `#the-channel`, made by the listener when the server
//! has none ([`channels`]).
//!
//! Each beat of the bots, on the world's authority, while the listener is
//! online ([`tick`]):
//!
//! - **out** ([`outbound`]): every message of a mirrored thread past its
//!   cursor that came from no outside origin is posted — by the speaker's own
//!   bot when it has one online, and otherwise through the channel's persona
//!   webhook, under the speaker's name and with its portrait;
//! - **in** ([`inbound`]): what the listener's shard heard in a mirrored
//!   channel is said in the world: on the open channel by a linked person
//!   with the Creator role, as the console's route would say it; on a group
//!   by a linked member of it; in a DM's private thread by its person, as a
//!   message to the character.
//!
//! Every five minutes, and on the first beat after a member takes the world,
//! the reconcile ([`reconcile`]) opens what the world has that the server
//! does not — a group's channel ([`groups`]), a DM's private thread
//! ([`dms`]) — keeps each group's members in step, archives a group nobody
//! remains on, reads what the channels missed, and names and pictures the
//! bots.
//!
//! What was written there while no member ran the listener is read when it
//! comes online ([`gap`]). Every other bot in the world's server drops its
//! channels: they are the listener's to hear, and they are not threads of any
//! one character's.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`record`] | `discord/guild`, `discord/thread/…` and `discord/webhook/…`, as the world keeps them |
//! | [`routes`] | `GET`/`PUT`/`DELETE /v1/world/:wid/discord` |
//! | [`channels`] | the mirrored channels, found or made, and their webhooks |
//! | [`hold`] | how long the mirror holds off after Discord refused it |
//! | [`members`] | who a thread's members are on Discord, and what a channel lets them do |
//! | [`groups`] | the named groups as channels only their members see |
//! | [`dms`] | a person's DM with a character without a bot, as a private thread |
//! | [`backstage`] | a DM between two characters, read-only for their owners |
//! | [`reconcile`] | the five-minute pass: channels, members, gaps, bots |
//! | [`speaker`] | who posts a message, and as whom |
//! | [`outbound`] | what is said on a mirrored thread, posted to its channel |
//! | [`inbound`] | a mirrored channel's message, heard in the world |
//! | [`gap`] | what was written while no member ran the listener |

pub mod backstage;
pub mod channels;
pub mod dms;
pub mod gap;
pub mod groups;
pub mod hold;
pub mod inbound;
pub mod members;
pub mod outbound;
pub mod reconcile;
pub mod record;
pub mod routes;
pub mod speaker;

use std::collections::HashMap;
use std::sync::Arc;

use crate::api::Authored;
use crate::discord::online::Online;
use crate::world::Hosted;

use self::record::Guild;

/// The world `bot`'s character listens to the server of, and that server:
/// `None` when it is the listener of none.
pub fn listened(a: &Authored, bot: &Online) -> Option<(Arc<Hosted>, Guild)> {
    let hosted = a.runtime.as_ref()?.hosted.get(&bot.world)?;
    let guild = hosted.mirror(|m| m.guild().cloned())?;
    (guild.listener == bot.npc).then_some((hosted, guild))
}

/// Whether server `guild_id` is the server of `bot`'s world — whose channels
/// are the listener's to hear, and no thread of any one character's.
pub fn is_world_server(a: &Authored, bot: &Online, guild_id: u64) -> bool {
    a.runtime
        .as_ref()
        .and_then(|rt| rt.hosted.get(&bot.world))
        .and_then(|hosted| hosted.mirror(|m| m.guild().map(|g| g.guild_id)))
        == Some(guild_id)
}

/// One beat of every world whose listener is among `online`: its channels
/// found or made, the reconcile when it is due, then what was said in the
/// world posted.
pub async fn tick(a: &Arc<Authored>, online: &[Online]) {
    let by_npc: HashMap<u64, &Online> = online.iter().map(|b| (b.npc, b)).collect();
    for bot in online {
        let Some((hosted, guild)) = listened(a, bot) else {
            continue;
        };
        if a.bots.mirror_held(hosted.id()) {
            continue;
        }
        if let Err(e) = channels::ensure(a, bot, &hosted, &guild).await {
            a.bots.hold_mirror(hosted.id(), hold::after(&e));
            continue;
        }
        if a.bots.reconcile_due(hosted.id(), reconcile::EVERY) {
            reconcile::run(a, bot, &hosted, &guild, &by_npc).await;
        }
        if let Some(wait) = outbound::post(a, bot, &hosted, &guild, &by_npc).await {
            a.bots.hold_mirror(hosted.id(), wait);
        }
    }
}
