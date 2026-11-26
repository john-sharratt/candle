//! What is said on a mirrored thread, posted to its channel in the world's
//! server (`docs/npcd_discord.md` §5.1).
//!
//! Each beat, for each mirrored thread not archived: every message past its
//! `mirrored_through` that came from no outside origin is posted, in order,
//! by its speaker ([`speaker`]) — a character's own bot, or the channel's
//! webhook under the speaker's name — each cut to Discord's 2,000
//! characters. A DM shares its character's channel with the character's
//! other DMs, so the character's lines there begin by naming whom they
//! answer ([`addressed`]). The cursor moves past a message only once all of it was
//! posted; a failure leaves it for the next beat, so a gap is retried, never
//! skipped. A webhook Discord no longer has (`404`) is forgotten, and the
//! next beat makes another. A post Discord forbids leaves the thread where it is and is said
//! in the listener's trace: the server's settings, not the world, decide
//! whether the listener may write there.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::channels::hook;
use super::hold::{after, FAILED_FOR, REFUSED_FOR};
use super::members::linked_people;
use super::record::{Guild, Kind};
use super::speaker::{self, Speaker};
use crate::api::Authored;
use crate::discord::online::Online;
use crate::discord::post::split::{split, LIMIT};
use crate::discord::post::{post as send, post_hook, posted, Post, Target};
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
    let people = linked_people(a);
    'threads: for (thread, mirrored) in hosted.mirror(|m| m.threads()) {
        if mirrored.archived {
            continue;
        }
        // Every line as it was said — a character's voiced, a person's as
        // written — with the name of who said it and, in a DM, the Discord
        // user of the person a character's line answers.
        let waiting: Vec<(Message, String, Option<u64>)> = hosted.sim(|s| {
            s.threads
                .get(&thread)
                .filter(|t| t.open)
                .map(|t| {
                    t.messages
                        .iter()
                        .filter(|m| m.seq > mirrored.mirrored_through && m.origin.is_none())
                        .map(|m| {
                            let by = t.member(&m.from).map_or(m.from.clone(), |w| w.name.clone());
                            let answers = match mirrored.kind {
                                Kind::Dm => t
                                    .members
                                    .iter()
                                    .find(|w| w.id != m.from)
                                    .and_then(|w| people.get(&w.id).copied()),
                                _ => None,
                            };
                            (m.clone(), by, answers)
                        })
                        .collect()
                })
                .unwrap_or_default()
        });
        let mut part_from = mirrored.mirrored_part;
        for (m, by, answers) in waiting {
            let at = |part: u32| Target {
                kind: "mirror",
                channel: mirrored.channel_id,
                thread: &thread,
                seq: m.seq,
                part,
            };
            let who = speaker::of(a, hosted, guild.guild_id, (&m.from, &by), online);
            let parts = split(&addressed(&m.text, answers), LIMIT);
            let mut last = 0;
            // **Each part is cursored**, so a long line that failed part-way
            // resumes at the part it stopped at: a webhook post carries no
            // nonce, so this is what keeps it from posting the first parts
            // again.
            for (i, part) in parts.iter().enumerate().skip(part_from as usize) {
                let to = at(i as u32);
                let sent = match &who {
                    Speaker::Bot(speaker) => send(a, speaker, to, part).await,
                    Speaker::Hook {
                        username,
                        avatar_url,
                    } => {
                        let webhook = match hook(a, bot, hosted, guild, mirrored.channel_id).await {
                            Ok(webhook) => webhook,
                            Err(e) => {
                                held(after(&e));
                                continue 'threads;
                            }
                        };
                        let as_who = (username.as_str(), avatar_url.as_deref());
                        post_hook(a, bot, &webhook, as_who, to, part).await
                    }
                };
                match sent {
                    Ok(Post::Posted(id)) => {
                        last = id;
                        hosted.with_mirror(|mirror| {
                            if let Some(mut now) = mirror.thread(&thread) {
                                now.mirrored_part = i as u32 + 1;
                                mirror.put_thread(&thread, now);
                            }
                        });
                    }
                    // A webhook deleted in the server's settings: the next
                    // beat makes another.
                    Ok(Post::Gone) => {
                        if matches!(who, Speaker::Hook { .. }) {
                            hosted.with_mirror(|m| m.remove_hook(mirrored.channel_id));
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
            posted(a, bot, at(0), last, parts.len());
            part_from = 0;
            hosted.with_mirror(|mirror| {
                if let Some(mut now) = mirror.thread(&thread) {
                    now.mirrored_through = m.seq;
                    now.mirrored_part = 0;
                    mirror.put_thread(&thread, now);
                }
            });
        }
    }
    hold
}

/// `text` as posted in a channel several DMs share: led by a mention of
/// `answers`, the Discord user the line is said to, when it is said to one.
/// The webhook's mentions ping nobody; it shows whom the line answers.
fn addressed(text: &str, answers: Option<u64>) -> String {
    match answers {
        Some(user) => format!("<@{user}> {text}"),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A line said to someone in a shared channel names them first**; a
    /// line said to nobody in particular is posted as it is.
    #[test]
    fn a_line_to_someone_in_a_shared_channel_names_them() {
        assert_eq!(
            addressed("The ward is quiet.", Some(1566050542650658966)),
            "<@1566050542650658966> The ward is quiet."
        );
        assert_eq!(addressed("Hello.", None), "Hello.");
    }
}
