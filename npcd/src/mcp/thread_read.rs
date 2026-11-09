//! `bot_thread_read {bot, thread?, limit?}`: a bot's handset threads, as the
//! world holds them (`docs/npcd_discord.md` §5.5). With no thread, every
//! thread the bot is on — its id, name, kind, members, last seq, what the
//! bot has not read, and the Discord DM or channel it mirrors; with one,
//! named by id or by the name the bot calls it, its last `limit` messages,
//! oldest first, each with its seq, time, sender, origin, and whether it has
//! reached the bot's mind. For the bot's owner or an admin.

use std::collections::HashMap;
use std::sync::Arc;

use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::args::{optional, string};
use super::owned::owned_or_admin;
use super::{Answer, Tool};
use crate::accounts::public_id;
use crate::api::Authored;
use crate::discord::trace::render::utc;
use crate::sim::phone::{direct_id, Kind, Origin, Thread, Who};
use crate::world::Hosted;

/// How many messages a thread read shows when not asked.
pub const DEFAULT_LIMIT: usize = 20;

/// The most it shows.
pub const MAX_LIMIT: usize = 200;

/// For each thread a Discord DM or channel mirrors, which.
pub type Mapped = HashMap<String, String>;

pub struct ThreadRead;

impl Tool for ThreadRead {
    const NAME: &'static str = "bot_thread_read";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "annotations": {"readOnlyHint": true},
            "description": "Read a bot's handset threads as its world holds them. With no thread: \
                each thread it is on, with its id, name, kind, members, last seq, how many \
                messages the bot has not read, and the Discord DM or channel it mirrors. With a \
                thread (its id, or the name the bot calls it): its last messages, oldest first, \
                with seq, time, sender, where each came from, and whether it reached the bot's \
                mind — so you can tell a message that never arrived from one the bot did not \
                answer.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bot": {"type": "string", "description": "The bot's id, as bot_list gives it."},
                    "thread": {"type": "string", "description": "A thread's id (th3) or name; leave out to list them."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "How many of its last messages to show; 20 when left out."}
                },
                "required": ["bot"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let bot = match string(args, "bot") {
            Ok(b) => b,
            Err(e) => return Answer::error(e),
        };
        let npc = match owned_or_admin(s, headers, bot).await {
            Ok(npc) => npc,
            Err(refused) => return Answer::error(refused),
        };
        let Some((hosted, body)) = s.runtime.as_ref().and_then(|rt| rt.body_of(npc)) else {
            return Answer::ok(format!("Bot {bot} is in no world, so it has no threads."));
        };
        let Some(name) = hosted.read(|w| w.actor(&body).map(|a| a.name.clone())) else {
            return Answer::ok(format!("Bot {bot} is in no world, so it has no threads."));
        };
        let character = Who::new(&body, &name);
        let mapped = mapped(&hosted, npc, &body);
        let threads: Vec<Thread> = hosted.sim(|sim| {
            sim.threads
                .iter()
                .filter(|t| t.has(&body))
                .cloned()
                .collect()
        });
        let limit = args
            .get("limit")
            .and_then(Value::as_u64)
            .map_or(DEFAULT_LIMIT, |l| (l as usize).clamp(1, MAX_LIMIT));
        Answer::of(match optional(args, "thread") {
            None => Ok(list(bot, &character, &threads, &mapped)),
            Some(named) => find(&threads, &character.id, named)
                .map(|t| show(&character.id, t, mapped.get(&t.id).map(String::as_str), limit))
                .ok_or_else(|| format!("bot {bot} is on no thread `{named}`")),
        })
    }
}

/// Which of character `npc`'s threads in `hosted` a Discord DM or channel
/// mirrors; `body` is the character's.
fn mapped(hosted: &Hosted, npc: u64, body: &str) -> Mapped {
    let mut out: Mapped = hosted
        .servers(|s| s.channels_of(npc))
        .into_iter()
        .map(|(channel, c)| (c.thread, format!("Discord channel {channel}")))
        .collect();
    for (account, dm) in hosted.dms(|d| d.of(npc)) {
        out.insert(
            direct_id(&public_id(&account), body),
            format!("Discord DM channel {}", dm.channel_id),
        );
    }
    out
}

/// The thread of `threads` named `named` — its id, or what `character`
/// calls it.
fn find<'a>(threads: &'a [Thread], character: &str, named: &str) -> Option<&'a Thread> {
    let want = named.trim().to_lowercase();
    threads.iter().find(|t| t.id == want).or_else(|| {
        threads
            .iter()
            .find(|t| t.as_named_to(character).to_lowercase() == want)
    })
}

fn kind(t: &Thread) -> &'static str {
    match t.kind() {
        Kind::Direct => "direct",
        Kind::Group => "group",
    }
}

fn open(t: &Thread) -> &'static str {
    match t.open {
        true => "open",
        false => "closed",
    }
}

/// Every thread `character` (bot `bot`) is on, a line each.
pub fn list(bot: &str, character: &Who, threads: &[Thread], mapped: &Mapped) -> String {
    let name = &character.name;
    if threads.is_empty() {
        return format!("Bot {bot} ({name}) is on no thread.");
    }
    let lines: Vec<String> = threads
        .iter()
        .map(|t| {
            let members: Vec<&str> = t.members.iter().map(|m| m.name.as_str()).collect();
            format!(
                "  {} \"{}\" {}, {}; members {}; last seq {}; unread {}; {}",
                t.id,
                t.as_named_to(&character.id),
                kind(t),
                open(t),
                members.join(", "),
                t.last_seq(),
                t.unread_for(&character.id),
                mapped.get(&t.id).map_or("not on Discord", String::as_str),
            )
        })
        .collect();
    format!("Bot {bot} ({name})'s threads:\n{}", lines.join("\n"))
}

/// Thread `t`'s last `limit` messages, as the character `me` (by id) has
/// them.
pub fn show(me: &str, t: &Thread, mapped: Option<&str>, limit: usize) -> String {
    let from = t.messages.len().saturating_sub(limit);
    let told = t.member(me).map_or(0, |m| m.delivered);
    let name = |id: &str| t.member(id).map_or(id.to_string(), |m| m.name.clone());
    let lines: Vec<String> = t
        .messages
        .iter()
        .skip(from)
        .map(|m| {
            let origin = match m.origin {
                Some(Origin::Discord { message, channel }) => {
                    format!("discord {message} in {channel}")
                }
                None => "world".into(),
            };
            let reached = match (m.from == me, m.seq <= told) {
                (true, _) => "own",
                (false, true) => "delivered",
                (false, false) => "undelivered",
            };
            let to =
                m.to.as_deref()
                    .map_or_else(String::new, |to| format!(", to {}", name(to)));
            format!(
                "  #{} {} {} [{origin}, {reached}{to}]: {}",
                m.seq,
                utc(m.at_ms),
                name(&m.from),
                m.text
            )
        })
        .collect();
    let on = mapped.map_or_else(String::new, |m| format!("; {m}"));
    let head = format!(
        "Thread {} \"{}\" ({}, {}{on}), last {} of {}",
        t.id,
        t.as_named_to(me),
        kind(t),
        open(t),
        lines.len(),
        t.messages.len()
    );
    match lines.is_empty() {
        true => format!("{head}: nothing said yet."),
        false => format!("{head}:\n{}", lines.join("\n")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::phone::{Said, Threads};

    /// Noon UTC, 1 November 2026.
    const NOON: u64 = 1_793_534_400_000;

    fn zen() -> Who {
        Who::new("npc-1", "Zen")
    }

    /// Zen on a server channel with Kit, and in a DM with Wren; Zen's mind
    /// told of everything on the channel but Kit's last.
    fn threads() -> Vec<Thread> {
        let mut p = Threads::new();
        let general = p.open_standing("zenworld #general", &[zen()]);
        p.join(&general, &Who::new("discord:41", "Kit (@kit)"));
        let mut anyone = Said::words("anyone?");
        anyone.origin = Some(Origin::Discord {
            message: 901,
            channel: 610,
        });
        p.send_on(&general, "discord:41", anyone).unwrap();
        p.send_on(&general, "npc-1", Said::words("I'm here.")).unwrap();
        let dm = p.reach(&Who::new("u_wren", "Wren S"), &zen());
        p.send_on(&dm, "u_wren", Said::words("are you there?")).unwrap();
        p.mark_delivered_for("npc-1");
        let mut good = Said::words("good");
        good.to = Some("npc-1".into());
        p.send_on(&general, "discord:41", good).unwrap();
        let mut out: Vec<Thread> = p.iter().cloned().collect();
        for t in &mut out {
            for (i, m) in t.messages.iter_mut().enumerate() {
                m.at_ms = NOON + i as u64 * 1_000;
            }
        }
        out
    }

    fn mapped() -> Mapped {
        [
            ("th1".to_string(), "Discord channel 610".to_string()),
            (
                "dm:npc-1:u_wren".to_string(),
                "Discord DM channel 31".to_string(),
            ),
        ]
        .into_iter()
        .collect()
    }

    /// **The list is a line per thread**, exactly.
    #[test]
    fn the_threads_list_as_these_lines() {
        assert_eq!(
            list("1z", &zen(), &threads(), &mapped()),
            "Bot 1z (Zen)'s threads:\n\
             \x20 dm:npc-1:u_wren \"Wren S\" direct, open; members Wren S, Zen; last seq 1; \
             unread 1; Discord DM channel 31\n\
             \x20 th1 \"zenworld #general\" group, open; members Zen, Kit (@kit); last seq 3; \
             unread 1; Discord channel 610"
        );
        assert_eq!(
            list("1z", &zen(), &[], &mapped()),
            "Bot 1z (Zen) is on no thread."
        );
    }

    /// **A thread shows its last messages oldest first**, each with its seq,
    /// time, sender, origin and whether it reached the bot's mind.
    #[test]
    fn a_thread_shows_its_last_messages() {
        let threads = threads();
        let general = find(&threads, "npc-1", "ZENWORLD #general").unwrap();
        assert_eq!(
            show("npc-1", general, Some("Discord channel 610"), 20),
            "Thread th1 \"zenworld #general\" (group, open; Discord channel 610), last 3 of 3:\n\
             \x20 #1 2026-11-01 12:00:00 UTC Kit (@kit) [discord 901 in 610, delivered]: anyone?\n\
             \x20 #2 2026-11-01 12:00:01 UTC Zen [world, own]: I'm here.\n\
             \x20 #3 2026-11-01 12:00:02 UTC Kit (@kit) [world, undelivered, to Zen]: good"
        );
        assert_eq!(
            show("npc-1", general, None, 1),
            "Thread th1 \"zenworld #general\" (group, open), last 1 of 3:\n\
             \x20 #3 2026-11-01 12:00:02 UTC Kit (@kit) [world, undelivered, to Zen]: good"
        );
        assert_eq!(
            find(&threads, "npc-1", "dm:npc-1:u_wren").map(|t| t.id.as_str()),
            Some("dm:npc-1:u_wren")
        );
        assert!(find(&threads, "npc-1", "nowhere").is_none());
    }
}
