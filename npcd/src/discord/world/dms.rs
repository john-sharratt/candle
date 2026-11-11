//! A person's DM with a character that has no bot of its own, as a private
//! thread in the world's server (`docs/npcd_discord.md` §3, §3.1).
//!
//! A webhook cannot DM, so a private thread is the nearest thing a server
//! has: under a `#dms` channel the listener finds or makes — everyone may see
//! it, nobody may write in it but in its threads — the listener opens a
//! private thread for each such DM, named for the two of them, that nobody
//! on it may invite to, and adds the person. The character's lines are posted
//! there through `#dms`'s webhook; the person's are their own, heard as a
//! message to the character (`WorldOp::Message`), as a DM to a bot is.
//!
//! A DM is opened once both are there to talk: the person linked, the
//! character in the world with no bot. Its cursor starts at the thread's
//! end, so what was said in the console before is not posted.

use std::collections::BTreeMap;

use super::channels::{find_or_make_with, last_seq};
use super::members::{character, everyone, member, HISTORY, LISTENER, SEND, VIEW};
use super::record::{Guild, Kind, Mirrored};
use crate::api::{now_ms, Authored};
use crate::discord::message::snowflake_at;
use crate::discord::online::Online;
use crate::discord::store;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::trace::{failed, note};
use crate::sim::phone::{Kind as ThreadKind, Member};
use crate::world::Hosted;

/// What the channel the DMs are made under is called.
pub const DMS: &str = "dms";

/// The longest name Discord takes on a thread.
const NAME_MAX: usize = 100;

/// What a DM's private thread is called: the character and the person.
pub fn thread_name(character: &str, person: &str) -> String {
    format!("{character} & {person}")
        .chars()
        .take(NAME_MAX)
        .collect()
}

/// The world's direct threads between a linked person, of `people` (by
/// handle), and a character without a bot: `(thread, character's name,
/// person's name, their user)`.
fn dms(
    a: &Authored,
    hosted: &Hosted,
    people: &BTreeMap<String, u64>,
) -> Vec<(String, String, String, u64)> {
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
            let (who, user) = members
                .iter()
                .find_map(|m| people.get(&m.id).map(|u| (m.clone(), *u)))?;
            let other = members.iter().find(|m| m.id != who.id)?.clone();
            let npc = character(a, hosted, &other.id)?;
            store::get(&shared, npc)
                .is_none()
                .then_some((thread, other.name, who.name, user))
        })
        .collect()
}

/// Every DM of `hosted` that should be a private thread, opened when it is
/// not one; `bot` the listener and `people` every linked person.
pub async fn reconcile(
    a: &Authored,
    bot: &Online,
    hosted: &Hosted,
    guild: &Guild,
    people: &BTreeMap<String, u64>,
) {
    let waiting: Vec<_> = dms(a, hosted, people)
        .into_iter()
        .filter(|(thread, ..)| hosted.mirror(|m| m.thread(thread)).is_none())
        .collect();
    if waiting.is_empty() {
        return;
    }
    let Some(parent) = parent(a, bot, hosted, guild).await else {
        return;
    };
    for (thread, character, person, user) in waiting {
        let opened = a
            .discord
            .create_private_thread(&bot.token, parent, &thread_name(&character, &person))
            .await;
        let channel_id = match opened {
            Ok(id) => id,
            Err(e) => {
                said(a, bot, parent, &thread, &format!("dm not opened: {e}"));
                continue;
            }
        };
        if let Err(e) = a
            .discord
            .add_thread_member(&bot.token, channel_id, user)
            .await
        {
            said(
                a,
                bot,
                channel_id,
                &thread,
                &format!("person not added: {e}"),
            );
            continue;
        }
        let mut mirrored = Mirrored::new(
            channel_id,
            Kind::Dm,
            last_seq(hosted, &thread),
            snowflake_at(now_ms()),
        );
        mirrored.parent = Some(parent);
        hosted.with_mirror(|m| m.put_thread(&thread, mirrored));
        note(
            a,
            bot.npc,
            Note::new(now_ms(), Direction::Out, "mirror", "dm_opened")
                .channel(channel_id)
                .thread(&thread),
        );
    }
}

/// `#dms`, found or made — everyone may see it and nobody may write in it
/// but in its threads — and recorded.
async fn parent(a: &Authored, bot: &Online, hosted: &Hosted, guild: &Guild) -> Option<u64> {
    if let Some(id) = hosted.mirror(|m| m.parent(DMS)) {
        return Some(id);
    }
    let overwrites = [
        everyone(guild.guild_id, VIEW | HISTORY, SEND),
        member(bot.user_id, LISTENER, 0),
    ];
    let id = find_or_make_with(a, bot, guild, DMS, &overwrites)
        .await
        .ok()?;
    hosted.with_mirror(|m| m.put_parent(DMS, id));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// **A DM's thread is named for the two of them**, cut to Discord's 100.
    #[test]
    fn a_dms_thread_is_named_for_the_two_of_them() {
        assert_eq!(thread_name("Wick", "Wren S"), "Wick & Wren S");
        assert_eq!(thread_name(&"x".repeat(95), "Wren S").chars().count(), 100);
    }
}
