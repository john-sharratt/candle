//! What a Discord call can answer instead of what was asked.
//!
//! Each is answered once and never retried here: a retry loop against a
//! rejected token spends the host's share of Discord's invalid-request limit
//! (10,000 per ten minutes per IP, `docs/npcd_discord.md` §2), which every bot
//! on the host shares. None of them carries the token.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Discord's JSON error code for a DM it will not open: the bot and the person
/// share no server (`docs/npcd_discord.md` §2).
pub const NO_SHARED_SERVER: u64 = 50278;

#[derive(Debug, Clone, PartialEq)]
pub enum DiscordError {
    /// `401`: Discord does not know the token — mistyped, or reset since.
    TokenRejected,
    /// The token is a user's, not a bot's. A self-bot is against Discord's
    /// terms, and only a bot can be a character.
    NotABot,
    /// `429`: try again after this many seconds, as Discord's `retry_after`.
    RateLimited { retry_after_s: f64 },
    /// Any other status, with Discord's JSON error `code` (`0` when it gave
    /// none) and what it said about it.
    Status {
        status: u16,
        code: u64,
        detail: String,
    },
    /// Discord was not reached, or answered something that is not its API.
    Transport(String),
    /// Not sent: the host is at its share of Discord's invalid-request limit
    /// ([`crate::discord::limit`]), and sends nothing for this many seconds.
    Paused { retry_after_s: f64 },
}

impl Display for DiscordError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenRejected => f.write_str("Discord rejected the token"),
            Self::NotABot => f.write_str("that token is a user's, not a bot's"),
            Self::RateLimited { retry_after_s } => {
                write!(f, "Discord is rate limiting; retry after {retry_after_s} s")
            }
            Self::Status { status, detail, .. } => {
                write!(f, "Discord answered {status}: {detail}")
            }
            Self::Transport(detail) => write!(f, "Discord could not be reached: {detail}"),
            Self::Paused { retry_after_s } => write!(
                f,
                "too many of this host's requests to Discord were refused; nothing is sent for \
                 {retry_after_s:.0} s"
            ),
        }
    }
}

impl Error for DiscordError {}

impl DiscordError {
    /// Whether this is Discord refusing to open a DM with somebody the bot
    /// shares no server with.
    pub fn no_shared_server(&self) -> bool {
        matches!(self, Self::Status { code, .. } if *code == NO_SHARED_SERVER)
    }
}
