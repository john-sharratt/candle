//! Seeing what the bridge did (`docs/npcd_discord.md` §5.5).
//!
//! Every message heard and every post made is said in the log at INFO — a
//! failed post at WARN — by its ids and outcome, never its text, and kept in
//! its bot's ring of the last hundred events, in memory on the member
//! running the bot's shard, beside how the shard's connection stands. The
//! trace route and `bot_discord_read` answer them with the records the world
//! keeps.
//!
//! | Module | What it owns |
//! |---|---|
//! | [`note`] | one event, its ids and its outcome |
//! | [`ring`] | a bot's last hundred |
//! | [`link`] | how a shard's connection stands |
//! | [`report`] | the trace of one bot, as the route answers it |
//! | [`render`] | the trace as text, as `bot_discord_read` answers it |
//! | [`route`] | `GET /v1/npc/:nid/discord/trace` |

pub mod link;
pub mod note;
pub mod render;
pub mod report;
pub mod ring;
pub mod route;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use self::link::{read, Link, SharedLink};
use self::note::Note;
use self::ring::Ring;
use crate::api::Authored;

/// Every bot's trace on this member.
#[derive(Default)]
pub struct Trace {
    by: Mutex<HashMap<u64, Traced>>,
}

#[derive(Default)]
struct Traced {
    ring: Ring,
    /// The connection of the shard running here, while one does.
    link: Option<SharedLink>,
}

impl Trace {
    /// Keep `note` for character `npc`'s bot.
    pub fn keep(&self, npc: u64, note: Note) {
        self.by
            .lock()
            .expect("trace lock")
            .entry(npc)
            .or_default()
            .ring
            .push(note);
    }

    /// A fresh link for the shard opening for character `npc`'s bot.
    pub fn attach(&self, npc: u64) -> SharedLink {
        let link = Arc::new(Mutex::new(Link::default()));
        self.by
            .lock()
            .expect("trace lock")
            .entry(npc)
            .or_default()
            .link = Some(Arc::clone(&link));
        link
    }

    /// Character `npc`'s bot's shard has closed here.
    pub fn detach(&self, npc: u64) {
        if let Some(t) = self.by.lock().expect("trace lock").get_mut(&npc) {
            t.link = None;
        }
    }

    /// How the shard running here for character `npc`'s bot stands, if one
    /// does.
    pub fn link(&self, npc: u64) -> Option<Link> {
        let by = self.by.lock().expect("trace lock");
        by.get(&npc)?.link.as_ref().map(read)
    }

    /// Character `npc`'s bot's events, newest first.
    pub fn notes(&self, npc: u64) -> Vec<Note> {
        self.by
            .lock()
            .expect("trace lock")
            .get(&npc)
            .map(|t| t.ring.newest_first())
            .unwrap_or_default()
    }
}

/// Say `note` of character `npc`'s bot at INFO, and keep it.
pub fn note(a: &Authored, npc: u64, note: Note) {
    tracing::info!(npc, "discord: {}", note.line());
    a.bots.trace().keep(npc, note);
}

/// Say `note`, a failure, of character `npc`'s bot at WARN, and keep it.
pub fn failed(a: &Authored, npc: u64, note: Note) {
    tracing::warn!(npc, "discord: {}", note.line());
    a.bots.trace().keep(npc, note);
}
