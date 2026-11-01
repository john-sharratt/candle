//! The servers a bot is in, followed as Discord says they change
//! (`docs/npcd_discord.md` §3.4).
//!
//! - **A server is learned of** by its `GUILD_CREATE`: every one `READY`
//!   named is handed over after it, and one the bot is added to arrives on
//!   its own. Each text channel gets its thread ([`thread::stand`]), a
//!   channel the server no longer has is closed, and the server is recorded
//!   by name. **A server with no record is a join**, told to the character
//!   with the channels it can talk in: one it was added to just now, or
//!   while it was offline, or before the bot first came online. One with a
//!   record is the replay of a server it was already in, or a server back
//!   from an outage, and is told nothing.
//! - **`READY`'s list against the records** says which servers the bot left
//!   while it was offline: each recorded server `READY` does not name is a
//!   leave.
//! - **A leave** — that, or a `GUILD_DELETE` that is not an outage — is told
//!   to the character, closes the server's threads, and forgets the server.
//!   Its channels' records stay, so a bot added back takes its threads up
//!   again.
//! - **A rename** of a server or a channel renames the threads; a channel
//!   made is stood up, and one deleted is closed and forgotten.

use std::collections::HashSet;

use super::notice::{channel_of, joined, removed, thread_name};
use super::record::Server;
use super::{embodied, tell, thread};
use crate::api::{now_ms, Authored};
use crate::discord::gateway::guild::{Guild, GuildChannel};
use crate::discord::online::Online;
use crate::discord::trace::note;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::world::is_world_server;
use crate::world::Hosted;

/// Server `g`, as a `GUILD_CREATE` hands it to `bot`.
///
/// **The world's own server is recorded and nothing more.** Its channels
/// mirror the world's threads (`crate::discord::world`) and are the
/// listener's to hear; a thread of one character's for each of them would be
/// a second copy of every conversation. The record is what says the bot may
/// post there as itself. A thread that stood for one of its channels before
/// the server became the world's is closed.
pub async fn learned(a: &Authored, bot: &Online, g: &Guild) {
    let Some((hosted, character)) = embodied(a, bot) else {
        return;
    };
    let npc = bot.npc;
    let joining = hosted.servers(|s| s.server(npc, g.id)).is_none();
    hosted.with_servers(|s| {
        s.put_server(
            npc,
            g.id,
            Server {
                name: g.name.clone(),
            },
        )
    });
    if is_world_server(a, bot, g.id) {
        for (channel_id, _) in hosted.servers(|s| s.channels_in(npc, g.id)) {
            gone(&hosted, npc, channel_id);
        }
        note(
            a,
            npc,
            Note::new(now_ms(), Direction::In, "server", "the_worlds").guild(g.id),
        );
        return;
    }
    let text: Vec<&GuildChannel> = g.text_channels().collect();
    for c in &text {
        thread::stand(&hosted, npc, &character, g.id, &g.name, c.id, &c.name);
    }
    let standing: HashSet<u64> = text.iter().map(|c| c.id).collect();
    for (channel_id, _) in hosted.servers(|s| s.channels_in(npc, g.id)) {
        if !standing.contains(&channel_id) {
            gone(&hosted, npc, channel_id);
        }
    }
    let outcome = match joining {
        true => "joined",
        false => "replayed",
    };
    note(
        a,
        npc,
        Note::new(now_ms(), Direction::In, "server", outcome).guild(g.id),
    );
    if joining {
        let names: Vec<String> = text.iter().map(|c| c.name.clone()).collect();
        tell(a, bot, joined(&g.name, &names)).await;
    }
}

/// `READY` said `bot` is in the servers `guilds`: each recorded server it
/// does not name, the bot left while it was offline.
pub async fn ready(a: &Authored, bot: &Online, guilds: &[u64]) {
    let Some((hosted, _)) = embodied(a, bot) else {
        return;
    };
    let now: HashSet<u64> = guilds.iter().copied().collect();
    for (guild_id, _) in hosted.servers(|s| s.servers_of(bot.npc)) {
        if !now.contains(&guild_id) {
            left(a, bot, guild_id).await;
        }
    }
}

/// `bot` was removed from server `guild_id`.
pub async fn left(a: &Authored, bot: &Online, guild_id: u64) {
    let Some((hosted, _)) = embodied(a, bot) else {
        return;
    };
    let npc = bot.npc;
    let Some(server) = hosted.with_servers(|s| s.remove_server(npc, guild_id)) else {
        return;
    };
    for (channel_id, _) in hosted.servers(|s| s.channels_in(npc, guild_id)) {
        thread::close(&hosted, npc, channel_id);
    }
    note(
        a,
        npc,
        Note::new(now_ms(), Direction::In, "server", "left").guild(guild_id),
    );
    tell(a, bot, removed(&server.name)).await;
}

/// Server `guild_id` of `bot`'s is called `name` now.
pub fn renamed(a: &Authored, bot: &Online, guild_id: u64, name: &str) {
    let Some((hosted, _)) = embodied(a, bot) else {
        return;
    };
    let npc = bot.npc;
    let Some(before) = hosted.servers(|s| s.server(npc, guild_id)) else {
        return;
    };
    if before.name == name {
        return;
    }
    hosted.with_servers(|s| s.put_server(npc, guild_id, Server { name: name.into() }));
    for (_, record) in hosted.servers(|s| s.channels_in(npc, guild_id)) {
        hosted.with_sim(|s| {
            let Some(t) = s.threads.get(&record.thread) else {
                return;
            };
            let Some(channel) = channel_of(&t.name, &before.name).map(str::to_string) else {
                return;
            };
            s.threads
                .rename(&record.thread, &thread_name(name, &channel));
        });
    }
}

/// Channel `c` was made or changed in one of `bot`'s servers.
pub fn channel(a: &Authored, bot: &Online, c: &GuildChannel) {
    let (Some(guild_id), true) = (c.guild_id, c.is_text()) else {
        return;
    };
    if is_world_server(a, bot, guild_id) {
        return;
    }
    let Some((hosted, character)) = embodied(a, bot) else {
        return;
    };
    let Some(server) = hosted.servers(|s| s.server(bot.npc, guild_id)) else {
        return;
    };
    thread::stand(
        &hosted,
        bot.npc,
        &character,
        guild_id,
        &server.name,
        c.id,
        &c.name,
    );
}

/// Channel `channel_id` of one of `bot`'s servers was deleted.
pub fn channel_gone(a: &Authored, bot: &Online, channel_id: u64) {
    let Some((hosted, _)) = embodied(a, bot) else {
        return;
    };
    gone(&hosted, bot.npc, channel_id);
}

/// A channel that no longer stands: its thread closed and its record
/// forgotten.
fn gone(hosted: &Hosted, npc: u64, channel_id: u64) {
    thread::close(hosted, npc, channel_id);
    hosted.with_servers(|s| s.remove_channel(npc, channel_id));
}
