//! The reconcile (`docs/npcd_discord.md` §5.3): every five minutes, and on
//! the first beat after a member takes the world, on its authority.
//!
//! - **Channels.** Every named group has its channel and every channel's
//!   overwrites are its thread's members; a group nobody remains on is
//!   archived ([`super::groups`]). Every person's DM with a character
//!   without a bot has its private thread ([`super::dms`]).
//! - **Gaps.** Every mirrored channel is read after its `last_seen_discord`
//!   ([`super::gap`]) — what arrived while no member held the world.
//! - **Bots.** Every character of the world with a bot online here is named
//!   and pictured as it now is, and one whose bot is not in the server is
//!   said in its trace: a bot is added to a server only by its invite.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::record::Guild;
use super::{dms, gap, groups, members};
use crate::api::{now_ms, Authored};
use crate::discord::online::Online;
use crate::discord::profile;
use crate::discord::server::embodied;
use crate::discord::store;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::trace::{failed, note};
use crate::world::Hosted;

/// How often a world's server is reconciled.
pub const EVERY: Duration = Duration::from_secs(5 * 60);

/// Reconcile `hosted`'s server `guild`, `bot` its listener and `online`
/// every bot online here.
pub async fn run(
    a: &Arc<Authored>,
    bot: &Online,
    hosted: &Hosted,
    guild: &Guild,
    online: &HashMap<u64, &Online>,
) {
    let people = members::linked_people(a).await;
    groups::reconcile(a, bot, hosted, guild, &people).await;
    dms::reconcile(a, bot, hosted, guild, &people).await;
    if let Err(e) = gap::fetch(a, bot).await {
        tracing::warn!(
            npc = bot.npc,
            "discord: the reconcile's read was refused: {e:?}"
        );
    }
    for other in online.values().filter(|b| b.world == hosted.id()) {
        if hosted
            .servers(|s| s.server(other.npc, guild.guild_id))
            .is_none()
        {
            failed(
                a,
                other.npc,
                Note::new(now_ms(), Direction::Out, "mirror", "not_in_server")
                    .guild(guild.guild_id),
            );
        }
        pictured(a, other).await;
    }
    note(
        a,
        bot.npc,
        Note::new(now_ms(), Direction::Out, "mirror", "reconciled").guild(guild.guild_id),
    );
}

/// Name and picture `bot` as its character now is, sending only what
/// differs from its record: Discord allows a bot few renames an hour.
pub async fn pictured(a: &Authored, bot: &Online) {
    let shared = a.npcs.substrate();
    let Some(mut record) = store::get(&shared, bot.npc) else {
        return;
    };
    let Some((_, name)) = embodied(a, bot) else {
        return;
    };
    let rename = profile::username(&name).filter(|n| *n != record.username);
    let portrait = a
        .npcs
        .view()
        .payload(bot.npc)
        .and_then(|n| n.portrait_image_id.clone());
    let repicture = portrait
        .clone()
        .filter(|p| record.pictured.as_ref() != Some(p));
    let avatar = repicture
        .as_deref()
        .and_then(|iid| match a.images.get(iid) {
            Ok((bytes, mime)) => Some(profile::avatar(&bytes, mime)),
            Err(e) => {
                tracing::warn!(npc = bot.npc, error = %e, "discord: the portrait was not read");
                None
            }
        });
    if rename.is_none() && avatar.is_none() {
        return;
    }
    match a
        .discord
        .set_profile(&bot.token, rename.as_deref(), avatar.as_deref())
        .await
    {
        Ok(user) => {
            record.username = user.username;
            if avatar.is_some() {
                record.pictured = portrait;
            }
            if let Err(e) = store::put(&shared, bot.npc, &record) {
                tracing::error!(npc = bot.npc, "discord: {e:#}");
            }
            note(
                a,
                bot.npc,
                Note::new(now_ms(), Direction::Out, "profile", "renamed_or_pictured"),
            );
        }
        Err(e) => failed(
            a,
            bot.npc,
            Note::new(now_ms(), Direction::Out, "profile", &format!("failed: {e}")),
        ),
    }
}
