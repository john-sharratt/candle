//! The channel of each character that has no bot of its own, and the DMs
//! posted in it (`docs/npcd_discord.md` §3, §3.1).
//!
//! A webhook cannot DM, so a character without a bot is talked to in a
//! channel of the world's server named for it, `#<name>`, that everyone may
//! read and write in. Every linked person's DM with the character is posted
//! there — the character's lines through the channel's webhook, each naming
//! whom it answers, and a person's as their own — and a post there is a
//! message to the character (`WorldOp::Message`), as a DM to a bot is, from
//! whoever wrote it. In the world each person's DM stays their own: its own
//! thread, conversation and journal.
//!
//! **One channel for a character, not one per person.** A private thread per
//! DM, under a `#dms` channel, left the server a list of "Wick & Johnathan"
//! threads nobody else could join; a character's channel is open to anyone
//! who wants to talk to it. Its cursor starts at the end of what each DM
//! holds, so what was said before is not posted again.

use std::collections::BTreeMap;

use super::channels::{find_or_make_with, last_seq};
use super::members::{character, everyone, member, LISTENER, MEMBER};
use super::record::{Guild, Kind, Mirrored, Own};
use crate::api::{now_ms, Authored};
use crate::discord::message::snowflake_at;
use crate::discord::online::Online;
use crate::discord::store;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::trace::{failed, note};
use crate::sim::phone::{Kind as ThreadKind, Member};
use crate::world::Hosted;

/// The longest name Discord takes on a channel.
const NAME_MAX: usize = 100;

/// What character `name`'s channel is called, as Discord keeps a text
/// channel's name: lower case, each run of anything but a letter or a digit
/// one `-`, none at either end.
pub fn channel_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars().flat_map(char::to_lowercase) {
        match c.is_alphanumeric() {
            true => out.push(c),
            false if !out.is_empty() && !out.ends_with('-') => out.push('-'),
            false => {}
        }
    }
    out.trim_end_matches('-').chars().take(NAME_MAX).collect()
}

/// The characters of `hosted` without a bot of their own, by character, with
/// their names.
fn botless(a: &Authored, hosted: &Hosted) -> Vec<(u64, String)> {
    let Some(rt) = a.runtime.as_ref() else {
        return Vec::new();
    };
    let shared = a.npcs.substrate();
    rt.bodies
        .in_world(hosted.id())
        .into_iter()
        .filter(|(npc, body)| !body.starts_with("visitor:") && store::get(&shared, *npc).is_none())
        .filter_map(|(npc, body)| {
            let name = hosted.read(|w| w.actor(&body).map(|x| x.name.clone()))?;
            Some((npc, name))
        })
        .collect()
}

/// The world's direct threads between a linked person, of `people` (by
/// handle), and a character without a bot: `(thread, character)`.
fn dms(a: &Authored, hosted: &Hosted, people: &BTreeMap<String, u64>) -> Vec<(String, u64)> {
    let pairs: Vec<(String, Vec<Member>)> = hosted.sim(|s| {
        s.threads
            .iter()
            .filter(|t| t.open && t.kind() == ThreadKind::Direct && t.members.len() == 2)
            .map(|t| (t.id.clone(), t.members.clone()))
            .collect()
    });
    let shared = a.npcs.substrate();
    pairs
        .into_iter()
        .filter_map(|(thread, members)| {
            let who = members.iter().find(|m| people.contains_key(&m.id))?;
            let other = members.iter().find(|m| m.id != who.id)?;
            let npc = character(a, hosted, &other.id)?;
            store::get(&shared, npc).is_none().then_some((thread, npc))
        })
        .collect()
}

/// Every character of `hosted` without a bot has its channel, and every DM
/// with one is posted in it; `bot` the listener and `people` every linked
/// person.
pub async fn reconcile(
    a: &Authored,
    bot: &Online,
    hosted: &Hosted,
    guild: &Guild,
    people: &BTreeMap<String, u64>,
) {
    for (npc, name) in botless(a, hosted) {
        if hosted.mirror(|m| m.own(npc)).is_some() {
            continue;
        }
        let overwrites = [
            everyone(guild.guild_id, MEMBER, 0),
            member(bot.user_id, LISTENER, 0),
        ];
        let channel_id =
            match find_or_make_with(a, bot, guild, &channel_name(&name), &overwrites).await {
                Ok(id) => id,
                Err(e) => {
                    failed(
                        a,
                        bot.npc,
                        Note::new(
                            now_ms(),
                            Direction::Out,
                            "mirror",
                            &format!("character channel not made: {e}"),
                        )
                        .guild(guild.guild_id),
                    );
                    continue;
                }
            };
        let own = Own {
            channel_id,
            last_seen_discord: snowflake_at(now_ms()),
        };
        hosted.with_mirror(|m| m.put_own(npc, own));
        note(
            a,
            bot.npc,
            Note::new(now_ms(), Direction::Out, "mirror", "character_channel")
                .guild(guild.guild_id)
                .channel(channel_id),
        );
    }
    for (thread, npc) in dms(a, hosted, people) {
        let Some(own) = hosted.mirror(|m| m.own(npc)) else {
            continue;
        };
        let was = hosted.mirror(|m| m.thread(&thread));
        if was.is_some_and(|w| w.channel_id == own.channel_id && w.kind == Kind::Dm) {
            continue;
        }
        let from = was.map_or_else(|| last_seq(hosted, &thread), |w| w.mirrored_through);
        let mirrored = Mirrored::new(own.channel_id, Kind::Dm, from, own.last_seen_discord);
        hosted.with_mirror(|m| m.put_thread(&thread, mirrored));
        note(
            a,
            bot.npc,
            Note::new(
                now_ms(),
                Direction::Out,
                "mirror",
                "dm_in_character_channel",
            )
            .channel(own.channel_id)
            .thread(&thread),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A character's channel is named as Discord keeps a channel's name.**
    #[test]
    fn a_characters_channel_is_named_as_discord_keeps_it() {
        assert_eq!(channel_name("Wick"), "wick");
        assert_eq!(channel_name("  Ilsa Varn "), "ilsa-varn");
        assert_eq!(channel_name("Zenling (Stinger) #2"), "zenling-stinger-2");
        assert_eq!(channel_name(&"x".repeat(120)).chars().count(), 100);
    }
}
