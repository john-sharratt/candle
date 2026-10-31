//! What a Discord call can answer instead of what was asked.
//!
//! Each is answered once and never retried here: a retry loop against a
//! rejected token spends the host's share of Discord's invalid-request limit
//! (10,000 per ten minutes per IP, `docs/npcd_discord.md` §2), which every bot
//! on the host shares. None of them carries the token.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub enum DiscordError {
    /// `401`: Discord does not know the token — mistyped, or reset since.
    TokenRejected,
    /// The token is a user's, not a bot's. A self-bot is against Discord's
    /// terms, and only a bot can be a character.
    NotABot,
    /// `429`: try again after this many seconds, as Discord's `retry_after`.
    RateLimited { retry_after_s: f64 },
    /// Any other status, with what Discord said about it.
    Status { status: u16, detail: String },
    /// Discord was not reached, or answered something that is not its API.
    Transport(String),
}

impl Display for DiscordError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenRejected => f.write_str("Discord rejected the token"),
            Self::NotABot => f.write_str("that token is a user's, not a bot's"),
            Self::RateLimited { retry_after_s } => {
                write!(f, "Discord is rate limiting; retry after {retry_after_s} s")
            }
            Self::Status { status, detail } => {
                write!(f, "Discord answered {status}: {detail}")
            }
            Self::Transport(detail) => write!(f, "Discord could not be reached: {detail}"),
        }
    }
}

impl Error for DiscordError {}
