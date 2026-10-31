//! A bot's posts (`docs/npcd_discord.md` §5.1): what a DM or a channel is
//! sent, cut to what Discord takes ([`split`]), and each post waited out
//! against a `429` and tried again a few times.
//!
//! A post Discord forbids (`403`: the bot may not write there) is said
//! apart from any other failure, so a channel can stop being offered; any
//! other failure leaves the post for the next beat. Every failure is said at
//! WARN and kept in the bot's trace (§5.5) with Discord's status and code;
//! every message posted, by whoever posted it, at INFO with the id Discord
//! gave it ([`posted`]).

pub mod split;

use std::time::Duration;

use tokio::time::sleep;

use crate::api::{now_ms, Authored};
use crate::discord::error::DiscordError;
use crate::discord::online::Online;
use crate::discord::rejected::Rejected;
use crate::discord::trace::note::{Direction, Note};
use crate::discord::trace::{failed, note};

/// How many times one post is tried against a `429` in a beat.
const RATE_TRIES: usize = 3;

/// The longest a post waits on one `429` before the next beat takes it.
const RATE_WAIT_MAX: Duration = Duration::from_secs(30);

/// Discord's status for a request the bot is not allowed to make.
const FORBIDDEN: u16 = 403;

/// What became of one post.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Post {
    /// Discord took it, as this message.
    Posted(u64),
    /// Discord forbade it: the bot may not write there.
    Forbidden,
    /// It was not posted this beat.
    Unposted,
}

/// Where a post goes, and what of the world's it carries: a message of a
/// thread, by its seq.
#[derive(Debug, Clone, Copy)]
pub struct Target<'a> {
    /// `dm` or `channel`: what the trace calls the post.
    pub kind: &'a str,
    pub channel: u64,
    pub thread: &'a str,
    pub seq: u64,
}

impl Target<'_> {
    fn note(&self, outcome: &str) -> Note {
        Note::new(now_ms(), Direction::Out, self.kind, outcome)
            .channel(self.channel)
            .thread(self.thread)
            .seq(self.seq)
    }
}

/// Post `text` to `to` as `bot`. A `429` is waited out and tried again, a
/// few times.
pub async fn post(
    a: &Authored,
    bot: &Online,
    to: Target<'_>,
    text: &str,
) -> Result<Post, Rejected> {
    for _ in 0..RATE_TRIES {
        match a.discord.create_message(&bot.token, to.channel, text).await {
            Ok(id) => return Ok(Post::Posted(id)),
            Err(DiscordError::RateLimited { retry_after_s }) => {
                failed(a, bot.npc, to.note("failed 429 (waiting it out)"));
                sleep(Duration::from_secs_f64(retry_after_s.max(0.0)).min(RATE_WAIT_MAX)).await;
            }
            Err(e) => {
                failed(a, bot.npc, to.note(&format!("failed {}", status_of(&e))));
                if matches!(e, DiscordError::Status { status, .. } if status == FORBIDDEN) {
                    return Ok(Post::Forbidden);
                }
                Rejected::or_logged(bot.npc, "a post", e)?;
                return Ok(Post::Unposted);
            }
        }
    }
    Ok(Post::Unposted)
}

/// Say that message `to.seq` was posted to `to`, in `parts` posts, the last
/// of them Discord's message `id`.
pub fn posted(a: &Authored, bot: &Online, to: Target<'_>, id: u64, parts: usize) {
    note(a, bot.npc, to.note("posted").message(id).parts(parts));
}

/// A failure's status and Discord code, as the trace writes it:
/// `403/50013`.
fn status_of(e: &DiscordError) -> String {
    match e {
        DiscordError::Status { status, code, .. } => format!("{status}/{code}"),
        DiscordError::TokenRejected => "401".into(),
        DiscordError::RateLimited { .. } => "429".into(),
        DiscordError::NotABot => "not_a_bot".into(),
        DiscordError::Transport(_) => "transport".into(),
        DiscordError::Paused { .. } => "paused".into(),
    }
}
