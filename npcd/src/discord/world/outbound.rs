//! What is said on a mirrored thread, posted to its channel in the world's
//! server (`docs/npcd_discord.md` §5.1).
//!
//! Each beat, for each mirrored thread not archived: every message past its
//! `mirrored_through` that came from no outside origin is posted, in order,
//! by its speaker ([`speaker`]) — a character's own bot, or the channel's
//! webhook under the speaker's name, and for a private thread its parent
//! channel's webhook into the thread — each cut to Discord's 2,000
//! characters. The cursor moves past a message only once all of it was
//! posted; a failure leaves it for the next beat, so a gap is retried, never
//! skipped. A webhook Discord no longer has (`404`) is forgotten, and the
//! next beat makes another. A post Discord forbids leaves the thread where it is and is said
//! in the listener's trace: the server's settings, not the world, decide
//! whether the listener may write there.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use super::channels::hook;
use super::hold::{after, FAILED_FOR, REFUSED_FOR};
use super::record::Guild;
use super::speaker::{self, Speaker};
use crate::api::Authored;
use crate::discord::online::Online;
use crate::discord::post::split::{split, LIMIT};
use crate::discord::post::{post as send, post_hook, posted, Post, Target};
use crate::engine::reported::spoken;
use crate::sim::phone::Message;
use crate::world::Hosted;

/// Post what is waiting on every mirrored thread of `hosted`, `bot` its
/// listener and `online` every bot online here. How long the world's mirror
/// is to hold off when Discord refused a post ([`super::hold`]): a thread
/// left unposted is not asked again on the next beat.
pub async fn post(
    a: &Arc<Authored>,
    bot: &Online,
    hosted: &Hosted,
    guild: &Guild,
    online: &HashMap<u64, &Online>,
) -> Option<Duration> {
    let mut hold: Option<Duration> = None;
    let mut held = |wait: Duration| hold = Some(hold.map_or(wait, |h| h.max(wait)));
    // A character's line is its intent, posted as it would say it; a
    // person's words are posted as they wrote them.
    let cast: HashSet<String> = hosted.read(|w| w.actors().map(|a| a.name.clone()).collect());
    'threads: for (thread, mirrored) in hosted.mirror(|m| m.threads()) {
        if mirrored.archived {
            continue;
        }
        let waiting: Vec<Message> = hosted.sim(|s| {
            s.threads
                .get(&thread)
                .filter(|t| t.open)
                .map(|t| {
                    t.messages
                        .iter()
                        .filter(|m| m.seq > mirrored.mirrored_through && m.origin.is_none())
                        .cloned()
                        .collect()
                })
                .unwrap_or_default()
        });
        for m in waiting {
            let to = Target {
                kind: "mirror",
                channel: mirrored.channel_id,
                thread: &thread,
                seq: m.seq,
            };
            let who = match mirrored.parent {
                Some(_) => speaker::hooked(a, hosted, &m.from),
                None => speaker::of(a, hosted, guild.guild_id, &m.from, online),
            };
            let text = match cast.contains(&m.from) {
                true => spoken(&m.intent),
                false => m.intent.clone(),
            };
            let parts = split(&text, LIMIT);
            let mut last = 0;
            for part in &parts {
                let sent = match &who {
                    Speaker::Bot(speaker) => send(a, speaker, to, part).await,
                    Speaker::Hook {
                        username,
                        avatar_url,
                    } => {
                        let webhook =
                            match hook(a, bot, hosted, guild, mirrored.hook_channel()).await {
                                Ok(webhook) => webhook,
                                Err(e) => {
                                    held(after(&e));
                                    continue 'threads;
                                }
                            };
                        let as_who = (username.as_str(), avatar_url.as_deref());
                        let in_thread = mirrored.parent.is_some();
                        post_hook(a, bot, &webhook, as_who, in_thread, to, part).await
                    }
                };
                match sent {
                    Ok(Post::Posted(id)) => last = id,
                    // A webhook deleted in the server's settings: the next
                    // beat makes another.
                    Ok(Post::Gone) => {
                        if matches!(who, Speaker::Hook { .. }) {
                            hosted.with_mirror(|m| m.remove_hook(mirrored.hook_channel()));
                        }
                        continue 'threads;
                    }
                    Ok(Post::Forbidden) | Err(_) => {
                        held(REFUSED_FOR);
                        continue 'threads;
                    }
                    Ok(Post::Unposted) => {
                        held(FAILED_FOR);
                        continue 'threads;
                    }
                }
            }
            posted(a, bot, to, last, parts.len());
            hosted.with_mirror(|mirror| {
                if let Some(mut now) = mirror.thread(&thread) {
                    now.mirrored_through = m.seq;
                    mirror.put_thread(&thread, now);
                }
            });
        }
    }
    hold
}
