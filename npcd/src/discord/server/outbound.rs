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
    'channels: for (channel_id, record) in hosted.servers(|s| s.channels_of(npc)) {
        let waiting: Vec<Message> = hosted.sim(|s| {
            s.threads
                .get(&record.thread)
                .filter(|t| t.open)
                .map(|t| {
                    t.messages
                        .iter()
                        .filter(|m| m.seq > record.mirrored_through && m.origin.is_none())
                        .cloned()
                        .collect()
                })
                .unwrap_or_default()
        });
        for m in waiting {
            let text = match m.from == character {
                true => m.intent.clone(),
                false => format!("> **{}:** {}", m.from, m.intent),
            };
            let to = Target {
                kind: "channel",
                channel: channel_id,
                thread: &record.thread,
                seq: m.seq,
            };
            let parts = split(&text, LIMIT);
            let mut last = 0;
            for part in &parts {
                match send(a, bot, to, part).await? {
                    Post::Posted(id) => last = id,
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
            posted(a, bot, to, last, parts.len());
            hosted.with_servers(|s| {
                if let Some(mut record) = s.channel(npc, channel_id) {
                    record.mirrored_through = m.seq;
                    s.put_channel(npc, channel_id, record);
                }
            });
        }
    }
    Ok(())
}
