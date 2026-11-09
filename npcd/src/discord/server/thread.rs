//! A server channel's thread in the character's world, kept as the channel
//! is (`docs/npcd_discord.md` §3.4): opened when the bot learns of the
//! channel, named `<server> #<channel>` and described as the channel is,
//! renamed and described again with either, and closed when the channel is
//! deleted or the bot leaves the server.
//!
//! These run on the world's authority, in the bots' beat, so they write the
//! world's threads and the channel's record in place.

use super::notice::thread_name;
use super::record::Channel;
use crate::api::now_ms;
use crate::discord::message::snowflake_at;
use crate::sim::phone::Who;
use crate::world::Hosted;

/// The Discord channel a thread is stood up for: its server's id and name,
/// and its own id, name and description.
#[derive(Debug, Clone, Copy)]
pub struct Standing<'a> {
    pub guild_id: u64,
    pub server: &'a str,
    pub channel_id: u64,
    pub channel: &'a str,
    pub topic: Option<&'a str>,
}

/// Channel `c` stood up for character `npc` (called `character`): its thread
/// open, named and described as the channel now is and with the character on
/// it, and its record written. A channel recorded before keeps its thread and
/// its cursors; a thread reopened takes up its mirror from what it holds, so
/// nothing said before is posted.
pub fn stand(hosted: &Hosted, npc: u64, character: &Who, c: Standing<'_>) {
    let name = thread_name(c.server, c.channel);
    let recorded = hosted.servers(|s| s.channel(npc, c.channel_id));
    let kept = recorded.clone().and_then(|mut record| {
        let through = hosted.with_sim(|s| {
            let t = s.threads.get(&record.thread)?;
            let reopening = !t.open;
            let last = t.last_seq();
            if t.name != name {
                s.threads.rename(&record.thread, &name);
            }
            if reopening {
                s.threads.reopen(&record.thread);
            }
            s.threads.describe(&record.thread, c.topic);
            s.threads.join(&record.thread, character);
            Some(reopening.then_some(last))
        })?;
        if let Some(last) = through {
            record.mirrored_through = record.mirrored_through.max(last);
        }
        Some(record)
    });
    let record = kept.unwrap_or_else(|| {
        let thread = hosted.with_sim(|s| {
            let id = s.threads.open_standing(&name, &[character.clone()]);
            s.threads.describe(&id, c.topic);
            id
        });
        Channel {
            guild_id: c.guild_id,
            thread,
            mirrored_through: 0,
            mirrored_part: 0,
            // What was written in the channel before the bot learned of it
            // is the channel's, never read back into the world.
            last_seen_discord: recorded
                .map_or(0, |r| r.last_seen_discord)
                .max(snowflake_at(now_ms())),
        }
    });
    hosted.with_servers(|s| s.put_channel(npc, c.channel_id, record));
}

/// Close the thread of character `npc`'s channel `channel_id`, if it has
/// one: the character stops being offered it, and what was said there is
/// kept.
pub fn close(hosted: &Hosted, npc: u64, channel_id: u64) {
    let Some(record) = hosted.servers(|s| s.channel(npc, channel_id)) else {
        return;
    };
    hosted.with_sim(|s| s.threads.close(&record.thread));
}

/// Whether character `npc`'s channel `channel_id` has an open thread.
pub fn is_open(hosted: &Hosted, npc: u64, channel_id: u64) -> bool {
    let Some(record) = hosted.servers(|s| s.channel(npc, channel_id)) else {
        return false;
    };
    hosted.sim(|s| s.threads.get(&record.thread).is_some_and(|t| t.open))
}
