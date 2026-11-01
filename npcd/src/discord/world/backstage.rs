//! A DM between two characters, mirrored for their owners
//! (`docs/npcd_discord.md` §3, §8 step 12).
//!
//! What two characters say to each other is the world's, and private to
//! them; their owners may read it. In a world whose server says so
//! (`backstage` on its `discord/guild` record), each such DM is a private
//! thread under `#backstage`, named for the two of them, with the owners of
//! both added — each owner who has linked a Discord account. `#backstage`
//! lets everyone see it and nobody write in it or its threads: the thread is
//! read-only, and the character's lines are posted into it through
//! `#backstage`'s webhook under their names. What anybody does write there
//! is dropped ([`super::inbound`]).

use std::collections::{BTreeMap, BTreeSet};

use super::channels::{find_or_make_with, last_seq};
use super::members::{character, everyone, member, HISTORY, LISTENER, SEND, SEND_IN_THREADS, VIEW};
use super::record::{Guild, Kind, Mirrored};
use crate::accounts::public_id;
use crate::api::{now_ms, Authored};
use crate::discord::link::index;
use crate::discord::message::snowflake_at;
use crate::discord::online::Online;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::trace::{failed, note};
use crate::sim::phone::Kind as ThreadKind;
use crate::world::Hosted;

/// What the channel the backstage threads are made under is called.
pub const BACKSTAGE: &str = "backstage";

/// The longest name Discord takes on a thread.
const NAME_MAX: usize = 100;

/// Each owner's linked Discord user, by the user id characters are owned
/// under.
fn owners_on_discord(a: &Authored) -> BTreeMap<String, u64> {
    index::all(&a.npcs.substrate())
        .into_iter()
        .filter_map(|(discord_id, account)| Some((public_id(&account), discord_id.parse().ok()?)))
        .collect()
}

/// The world's DMs between two characters: `(thread, a, b, owners)`, the
/// owners those of both who are on Discord.
fn backstage(a: &Authored, hosted: &Hosted) -> Vec<(String, String, String, BTreeSet<u64>)> {
    let on_discord = owners_on_discord(a);
    let cast = a.npcs.view();
    let pairs: Vec<(String, Vec<String>)> = hosted.sim(|s| {
        s.threads
            .iter()
            .filter(|t| t.open && t.kind() == ThreadKind::Direct && t.members.len() == 2)
            .map(|t| (t.id.clone(), t.members.clone()))
            .collect()
    });
    pairs
        .into_iter()
        .filter_map(|(thread, members)| {
            let (one, two) = (&members[0], &members[1]);
            let npcs = [character(a, hosted, one)?, character(a, hosted, two)?];
            let owners = npcs
                .iter()
                .filter_map(|npc| cast.payload(*npc))
                .filter_map(|p| on_discord.get(&p.owner_id).copied())
                .collect();
            Some((thread, one.clone(), two.clone(), owners))
        })
        .collect()
}

/// Every DM between two characters of `hosted` as a backstage thread, when
/// `guild` mirrors them; `bot` the listener.
pub async fn reconcile(a: &Authored, bot: &Online, hosted: &Hosted, guild: &Guild) {
    if !guild.backstage {
        return;
    }
    let waiting: Vec<_> = backstage(a, hosted)
        .into_iter()
        .filter(|(thread, ..)| hosted.mirror(|m| m.thread(thread)).is_none())
        .collect();
    if waiting.is_empty() {
        return;
    }
    let Some(parent) = parent(a, bot, hosted, guild).await else {
        return;
    };
    for (thread, one, two, owners) in waiting {
        let name: String = format!("{one} & {two}").chars().take(NAME_MAX).collect();
        let channel_id = match a
            .discord
            .create_private_thread(&bot.token, parent, &name)
            .await
        {
            Ok(id) => id,
            Err(e) => {
                said(
                    a,
                    bot,
                    parent,
                    &thread,
                    &format!("backstage not opened: {e}"),
                );
                continue;
            }
        };
        for owner in owners {
            if let Err(e) = a
                .discord
                .add_thread_member(&bot.token, channel_id, owner)
                .await
            {
                said(
                    a,
                    bot,
                    channel_id,
                    &thread,
                    &format!("owner not added: {e}"),
                );
            }
        }
        let mut mirrored = Mirrored::new(
            channel_id,
            Kind::Backstage,
            last_seq(hosted, &thread),
            snowflake_at(now_ms()),
        );
        mirrored.parent = Some(parent);
        hosted.with_mirror(|m| m.put_thread(&thread, mirrored));
        note(
            a,
            bot.npc,
            Note::new(now_ms(), Direction::Out, "mirror", "backstage_opened")
                .channel(channel_id)
                .thread(&thread),
        );
    }
}

/// `#backstage`, found or made — everyone may see it, nobody may write in
/// it or its threads — and recorded.
async fn parent(a: &Authored, bot: &Online, hosted: &Hosted, guild: &Guild) -> Option<u64> {
    if let Some(id) = hosted.mirror(|m| m.parent(BACKSTAGE)) {
        return Some(id);
    }
    let overwrites = [
        everyone(guild.guild_id, VIEW | HISTORY, SEND | SEND_IN_THREADS),
        member(bot.user_id, LISTENER, 0),
    ];
    let id = find_or_make_with(a, bot, guild, BACKSTAGE, &overwrites)
        .await
        .ok()?;
    hosted.with_mirror(|m| m.put_parent(BACKSTAGE, id));
    Some(id)
}

fn said(a: &Authored, bot: &Online, channel_id: u64, thread: &str, outcome: &str) {
    failed(
        a,
        bot.npc,
        Note::new(now_ms(), Direction::Out, "mirror", outcome)
            .channel(channel_id)
            .thread(thread),
    );
}
