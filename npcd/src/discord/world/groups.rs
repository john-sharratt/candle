//! A world's named groups as channels of its server (`docs/npcd_discord.md`
//! §3, §5.3).
//!
//! Each group thread — a group a character opened, or a DM that gained a
//! third member — is a text channel of its own, named for the group, whose
//! permission overwrites admit exactly its members: `@everyone` is denied
//! the channel, the listener is let in to mirror it, and each member on
//! Discord — a character's bot, a linked person — is let see, read and
//! write. The reconcile keeps the overwrites the thread's members: a new
//! member is let in, one who left is taken out. A group nobody remains on —
//! closed, or gone from the world — is **archived**: its members keep
//! seeing it and may no longer write, and nothing more is posted there. It
//! is never deleted.

use std::collections::{BTreeMap, BTreeSet};

use super::channels::last_seq;
use super::members::{discord_users, everyone, member, HISTORY, LISTENER, MEMBER, SEND, VIEW};
use super::record::{Guild, Kind, Mirrored};
use crate::api::{now_ms, Authored};
use crate::discord::client::Overwrite;
use crate::discord::message::snowflake_at;
use crate::discord::online::Online;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::trace::{failed, note};
use crate::sim::phone::Kind as ThreadKind;
use crate::world::Hosted;

/// The longest name Discord takes on a channel.
const NAME_MAX: usize = 100;

/// A group thread of the world.
struct Group {
    id: String,
    name: String,
    members: Vec<String>,
    open: bool,
}

/// What a group is called as a channel: Discord's own shape of a name —
/// lowercase, words joined by `-`, nothing but letters, digits, `-` and `_`
/// — and never empty.
pub fn channel_name(name: &str) -> String {
    let mut out = String::new();
    for c in name.trim().chars().flat_map(char::to_lowercase) {
        match c {
            c if c.is_alphanumeric() || c == '_' => out.push(c),
            _ if !out.ends_with('-') && !out.is_empty() => out.push('-'),
            _ => {}
        }
    }
    let out: String = out.trim_end_matches('-').chars().take(NAME_MAX).collect();
    match out.is_empty() {
        true => "group".to_string(),
        false => out,
    }
}

/// The world's group threads, open or not, and each mirrored one gone from
/// the world, as one that is closed.
fn groups(hosted: &Hosted) -> Vec<Group> {
    let mut found: Vec<Group> = hosted.sim(|s| {
        s.threads
            .iter()
            .filter(|t| t.kind() == ThreadKind::Group && !t.permanent)
            .map(|t| Group {
                id: t.id.clone(),
                name: t.name.clone(),
                members: t.members.iter().map(|m| m.id.clone()).collect(),
                open: t.open && !t.members.is_empty(),
            })
            .collect()
    });
    let standing: BTreeSet<String> = found.iter().map(|g| g.id.clone()).collect();
    for (id, mirrored) in hosted.mirror(|m| m.threads()) {
        if mirrored.kind == Kind::Group && !standing.contains(&id) {
            found.push(Group {
                id,
                name: String::new(),
                members: Vec::new(),
                open: false,
            });
        }
    }
    found
}

/// Every group of `hosted` as a channel of `guild`, with its overwrites the
/// thread's members, `bot` the listener and `people` every linked person.
pub async fn reconcile(
    a: &Authored,
    bot: &Online,
    hosted: &Hosted,
    guild: &Guild,
    people: &BTreeMap<String, u64>,
) {
    for g in groups(hosted) {
        let users: BTreeSet<u64> = discord_users(a, hosted, &g.members, people)
            .into_values()
            .filter(|u| *u != bot.user_id)
            .collect();
        match hosted.mirror(|m| m.thread(&g.id)) {
            None if g.open => open(a, bot, hosted, guild, &g, &users).await,
            None => {}
            Some(m) if m.archived => {}
            Some(m) if !g.open => archive(a, bot, hosted, &g.id, m).await,
            Some(m) => keep(a, bot, m.channel_id, &users).await,
        }
    }
}

/// Open group `g` as a channel only its members see.
async fn open(
    a: &Authored,
    bot: &Online,
    hosted: &Hosted,
    guild: &Guild,
    g: &Group,
    users: &BTreeSet<u64>,
) {
    let mut overwrites = vec![
        everyone(guild.guild_id, 0, VIEW),
        member(bot.user_id, LISTENER, 0),
    ];
    overwrites.extend(users.iter().map(|u| member(*u, MEMBER, 0)));
    // Always a channel of its own: one found by name could be anybody's, and
    // open to the whole server.
    let made = a
        .discord
        .create_channel(
            &bot.token,
            guild.guild_id,
            &channel_name(&g.name),
            &overwrites,
        )
        .await;
    let channel_id = match made {
        Ok(id) => id,
        Err(e) => {
            failed(
                a,
                bot.npc,
                Note::new(
                    now_ms(),
                    Direction::Out,
                    "mirror",
                    &format!("group not opened: {e}"),
                )
                .guild(guild.guild_id)
                .thread(&g.id),
            );
            return;
        }
    };
    let mirrored = Mirrored::new(
        channel_id,
        Kind::Group,
        last_seq(hosted, &g.id),
        snowflake_at(now_ms()),
    );
    hosted.with_mirror(|m| m.put_thread(&g.id, mirrored));
    note(
        a,
        bot.npc,
        Note::new(now_ms(), Direction::Out, "mirror", "group_opened")
            .guild(guild.guild_id)
            .channel(channel_id)
            .thread(&g.id),
    );
}

/// Keep channel `channel_id`'s member overwrites `users`, and the
/// listener's: let in who is missing, take out who left.
async fn keep(a: &Authored, bot: &Online, channel_id: u64, users: &BTreeSet<u64>) {
    let Some(standing) = overwrites(a, bot, channel_id).await else {
        return;
    };
    let members: BTreeSet<u64> = standing
        .iter()
        .filter(|o| o.kind == 1 && o.id != bot.user_id)
        .map(|o| o.id)
        .collect();
    for user in users.difference(&members) {
        let let_in = member(*user, MEMBER, 0);
        if let Err(e) = a
            .discord
            .put_overwrite(&bot.token, channel_id, &let_in)
            .await
        {
            said_failed(a, bot, channel_id, "member not let in", &e.to_string());
        }
    }
    for user in members.difference(users) {
        if let Err(e) = a
            .discord
            .delete_overwrite(&bot.token, channel_id, *user)
            .await
        {
            said_failed(a, bot, channel_id, "member not taken out", &e.to_string());
        }
    }
}

/// Archive group `thread`'s channel: every member it has may see it and no
/// longer write, and nothing more is posted there.
async fn archive(a: &Authored, bot: &Online, hosted: &Hosted, thread: &str, mut m: Mirrored) {
    let Some(standing) = overwrites(a, bot, m.channel_id).await else {
        return;
    };
    for o in standing
        .iter()
        .filter(|o| o.kind == 1 && o.id != bot.user_id)
    {
        let read_only = member(o.id, VIEW | HISTORY, SEND);
        if let Err(e) = a
            .discord
            .put_overwrite(&bot.token, m.channel_id, &read_only)
            .await
        {
            said_failed(a, bot, m.channel_id, "not archived", &e.to_string());
            return;
        }
    }
    m.archived = true;
    hosted.with_mirror(|x| x.put_thread(thread, m));
    note(
        a,
        bot.npc,
        Note::new(now_ms(), Direction::Out, "mirror", "group_archived")
            .channel(m.channel_id)
            .thread(thread),
    );
}

/// Channel `channel_id`'s overwrites, or `None` said in the trace.
async fn overwrites(a: &Authored, bot: &Online, channel_id: u64) -> Option<Vec<Overwrite>> {
    match a.discord.overwrites(&bot.token, channel_id).await {
        Ok(standing) => Some(standing),
        Err(e) => {
            said_failed(a, bot, channel_id, "overwrites not read", &e.to_string());
            None
        }
    }
}

fn said_failed(a: &Authored, bot: &Online, channel_id: u64, what: &str, why: &str) {
    failed(
        a,
        bot.npc,
        Note::new(
            now_ms(),
            Direction::Out,
            "mirror",
            &format!("{what}: {why}"),
        )
        .channel(channel_id),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A group's channel is named as Discord names one**: lowercase, its
    /// words joined by one `-`, its punctuation gone, cut to 100, and never
    /// empty.
    #[test]
    fn a_groups_channel_is_named_as_discord_names_one() {
        assert_eq!(channel_name("The Eastern Sweep"), "the-eastern-sweep");
        assert_eq!(channel_name("  ops: night  shift! "), "ops-night-shift");
        assert_eq!(channel_name("crew_7"), "crew_7");
        assert_eq!(channel_name("!!!"), "group");
        assert_eq!(channel_name(&"a".repeat(150)).len(), 100);
    }
}
