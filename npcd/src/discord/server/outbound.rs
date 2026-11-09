//! What is said on a server channel's thread, posted to the channel
//! (`docs/npcd_discord.md` §3.4, §5.1).
//!
//! Each beat of the bots, for each open channel thread of a bot: every
//! message past the channel's `mirrored_through` that came from no outside
//! origin is posted, in order — the character's own as the text it is, and
//! anybody else's on the thread, a character it invited, as a quoted line
//! (`> **Name:** …`), since the bot cannot post as them — each cut to
//! Discord's 2,000 characters. The cursor moves past a message only once all
//! of it was posted. A post Discord forbids (`403`: the bot may not write
//! there) closes the thread, so the character stops being offered it, and
//! it is not tried again; any other failure leaves the cursor for the next
//! beat.

use std::sync::Arc;

use super::{embodied, thread};
use crate::api::{now_ms, Authored};
use crate::discord::online::Online;
use crate::discord::post::split::{split, LIMIT};
use crate::discord::post::{post as send, posted, Post, Target};
use crate::discord::rejected::Rejected;
use crate::discord::trace::note;
use crate::discord::trace::note::{Direction, Note};
use crate::sim::phone::Message;

/// Post what is waiting on `bot`'s channels' threads.
pub async fn post(a: &Arc<Authored>, bot: &Online) -> Result<(), Rejected> {
    let Some((hosted, character)) = embodied(a, bot) else {
        return Ok(());
    };
    let npc = bot.npc;
    'channels: for (channel_id, mut record) in hosted.servers(|s| s.channels_of(npc)) {
        let waiting: Vec<(Message, String)> = hosted.sim(|s| {
            s.threads
                .get(&record.thread)
                .filter(|t| t.open)
                .map(|t| {
                    t.messages
                        .iter()
                        .filter(|m| m.seq > record.mirrored_through && m.origin.is_none())
                        .map(|m| {
                            let by = t.member(&m.from).map_or(m.from.clone(), |w| w.name.clone());
                            (m.clone(), by)
                        })
                        .collect()
                })
                .unwrap_or_default()
        });
        for (m, by) in waiting {
            let text = match m.from == character.id {
                true => m.text.clone(),
                false => format!("> **{by}:** {}", m.text),
            };
            let at = |part: u32| Target {
                kind: "channel",
                channel: channel_id,
                thread: &record.thread,
                seq: m.seq,
                part,
            };
            let parts = split(&text, LIMIT);
            let mut last = 0;
            for (i, part) in parts.iter().enumerate().skip(record.mirrored_part as usize) {
                match send(a, bot, at(i as u32), part).await? {
                    Post::Posted(id) => {
                        last = id;
                        record.mirrored_part = i as u32 + 1;
                        hosted.with_servers(|s| {
                            if let Some(mut now) = s.channel(npc, channel_id) {
                                now.mirrored_part = record.mirrored_part;
                                s.put_channel(npc, channel_id, now);
                            }
                        });
                    }
                    Post::Forbidden => {
                        note(
                            a,
                            npc,
                            Note::new(now_ms(), Direction::Out, "channel", "thread_closed")
                                .channel(channel_id)
                                .thread(&record.thread),
                        );
                        thread::close(&hosted, npc, channel_id);
                        continue 'channels;
                    }
                    // A deleted channel is closed by its `CHANNEL_DELETE`.
                    Post::Gone | Post::Unposted => continue 'channels,
                }
            }
            posted(a, bot, at(0), last, parts.len());
            record.mirrored_through = m.seq;
            record.mirrored_part = 0;
            hosted.with_servers(|s| {
                if let Some(mut now) = s.channel(npc, channel_id) {
                    now.mirrored_through = m.seq;
                    now.mirrored_part = 0;
                    s.put_channel(npc, channel_id, now);
                }
            });
        }
    }
    Ok(())
}
