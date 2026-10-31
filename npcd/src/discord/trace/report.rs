//! The trace of one bot (`docs/npcd_discord.md` §5.5): its state record, how
//! its shard stands, every DM, channel and server the world records for it
//! with how far each thread has been posted, and its last events.
//!
//! The shard and the events are the member's that runs the shard; any other
//! member answers what the replicated records say, with `member` naming
//! where the shard runs when its state says so.

use serde::Serialize;

use super::link::Link;
use super::note::Note;
use crate::accounts::public_id;
use crate::api::Authored;
use crate::discord::state::{self, BotState, Status};
use crate::discord::store;
use crate::engine::speaking_as;
use crate::sim::phone::Thread;

/// One bot's trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Report {
    /// Its `discord/state` record, as it stands.
    pub state: Option<BotState>,
    pub shard: Shard,
    pub dms: Vec<DmRow>,
    pub channels: Vec<ChannelRow>,
    pub servers: Vec<ServerRow>,
    /// Its last events, newest first.
    pub events: Vec<Note>,
}

/// How its shard stands, and the member running it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Shard {
    #[serde(flatten)]
    pub link: Link,
    /// The member running the shard, when one is known to.
    pub member: Option<u64>,
}

/// A DM with a linked person, and its thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DmRow {
    pub account: String,
    pub channel_id: u64,
    pub user_id: u64,
    pub mirrored_through: u64,
    pub last_seen_discord: u64,
    pub thread_id: Option<String>,
    pub thread_last_seq: Option<u64>,
    /// Messages on the thread past `mirrored_through`, from the world, that
    /// are still to be posted.
    pub unposted: usize,
}

/// A server channel, and its thread.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChannelRow {
    pub guild_id: u64,
    pub channel_id: u64,
    pub thread: String,
    pub name: Option<String>,
    pub mirrored_through: u64,
    pub last_seen_discord: u64,
    pub thread_last_seq: Option<u64>,
    pub unposted: usize,
    pub open: bool,
}

/// A server the bot is in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ServerRow {
    pub guild_id: u64,
    pub name: String,
}

/// A thread's last seq, and how many of its messages past `through` came
/// from the world: what a mirror still has to post.
fn tally(t: &Thread, through: u64) -> (Option<u64>, usize) {
    let unposted = t
        .messages
        .iter()
        .filter(|m| m.seq > through && m.origin.is_none())
        .count();
    (t.messages.last().map(|m| m.seq), unposted)
}

/// Character `npc`'s bot's trace, as this member sees it.
pub async fn report(a: &Authored, npc: u64) -> Report {
    let shared = a.npcs.substrate();
    let record = state::get(&shared, npc);
    let here = a.bots.trace().link(npc);
    let member = match &here {
        Some(_) => a.runtime.as_ref().and_then(|rt| rt.standing.node()),
        None => {
            let running = store::get(&shared, npc).is_some_and(|bot| {
                matches!(
                    state::status(&shared, npc, &bot),
                    Status::Online { .. } | Status::NeedsContentIntent
                )
            });
            record.as_ref().filter(|_| running).and_then(|s| s.member)
        }
    };
    let mut report = Report {
        state: record,
        shard: Shard {
            link: here.unwrap_or_default(),
            member,
        },
        dms: Vec::new(),
        channels: Vec::new(),
        servers: Vec::new(),
        events: a.bots.trace().notes(npc),
    };
    let Some((hosted, body)) = a.runtime.as_ref().and_then(|rt| rt.body_of(npc)) else {
        return report;
    };
    let character = hosted
        .read(|w| w.actor(&body).map(|a| a.name.clone()))
        .unwrap_or_default();

    for (account, dm) in hosted.dms(|d| d.of(npc)) {
        let person = a
            .accounts
            .read()
            .await
            .identity(&account)
            .map(|id| speaking_as(&id, &public_id(&account), &a.roles));
        let thread = person.and_then(|person| {
            hosted.sim(|s| {
                s.threads
                    .direct_between(&person, &character)
                    .map(|t| (t.id.clone(), tally(t, dm.mirrored_through)))
            })
        });
        let (thread_id, (thread_last_seq, unposted)) = match thread {
            Some((id, tally)) => (Some(id), tally),
            None => (None, (None, 0)),
        };
        report.dms.push(DmRow {
            account,
            channel_id: dm.channel_id,
            user_id: dm.user_id,
            mirrored_through: dm.mirrored_through,
            last_seen_discord: dm.last_seen_discord,
            thread_id,
            thread_last_seq,
            unposted,
        });
    }

    for (channel_id, c) in hosted.servers(|s| s.channels_of(npc)) {
        let seen = hosted.sim(|s| {
            s.threads
                .get(&c.thread)
                .map(|t| (t.name.clone(), t.open, tally(t, c.mirrored_through)))
        });
        let (name, open, (thread_last_seq, unposted)) = match seen {
            Some((name, open, tally)) => (Some(name), open, tally),
            None => (None, false, (None, 0)),
        };
        report.channels.push(ChannelRow {
            guild_id: c.guild_id,
            channel_id,
            thread: c.thread,
            name,
            mirrored_through: c.mirrored_through,
            last_seen_discord: c.last_seen_discord,
            thread_last_seq,
            unposted,
            open,
        });
    }

    report.servers = hosted
        .servers(|s| s.servers_of(npc))
        .into_iter()
        .map(|(guild_id, s)| ServerRow {
            guild_id,
            name: s.name,
        })
        .collect();
    report
}
