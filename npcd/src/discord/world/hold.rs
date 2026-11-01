//! How long a world's mirror holds off after Discord refused it
//! (`docs/npcd_discord.md` §2, §5.1).
//!
//! Every `401`, `403` and `429` counts against the host's share of
//! Discord's invalid-request limit — 10,000 in ten minutes per IP, shared
//! by every bot on it, past which Cloudflare bans the IP. A mirror that asks
//! again on the next beat after a refusal spends two of those a second: a
//! listener without Manage Channels, asking every 500 ms to make
//! `#the-channel`, spent 722 in eight minutes. So a refusal holds the whole
//! world's mirror — its channels, its reconcile, its posts — until asking
//! again is worth it:
//!
//! - a rate limit (`429`), or the host's own pause, for as long as it asks;
//! - a refusal that only a person can lift — a permission the server has
//!   not given (`403`), a token Discord rejected (`401`) — for
//!   [`REFUSED_FOR`];
//! - anything else for [`FAILED_FOR`].

use std::time::Duration;

use crate::discord::error::DiscordError;

/// How long a refusal only a person can lift holds the mirror.
pub const REFUSED_FOR: Duration = Duration::from_secs(10 * 60);

/// How long any other failure holds the mirror.
pub const FAILED_FOR: Duration = Duration::from_secs(30);

/// Discord's status for a request the bot is not allowed to make.
const FORBIDDEN: u16 = 403;

/// How long the mirror holds off after `e`.
pub fn after(e: &DiscordError) -> Duration {
    match e {
        DiscordError::RateLimited { retry_after_s } | DiscordError::Paused { retry_after_s } => {
            Duration::from_secs_f64(retry_after_s.max(1.0))
        }
        DiscordError::Status { status, .. } if *status == FORBIDDEN => REFUSED_FOR,
        DiscordError::TokenRejected | DiscordError::NotABot => REFUSED_FOR,
        DiscordError::Status { .. } | DiscordError::Transport(_) => FAILED_FOR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A rate limit holds for what it asks, a refusal a person must lift
    /// for ten minutes, anything else for thirty seconds.**
    #[test]
    fn each_refusal_holds_the_mirror_as_long_as_it_asks() {
        assert_eq!(
            after(&DiscordError::RateLimited {
                retry_after_s: 39.5
            }),
            Duration::from_secs_f64(39.5)
        );
        assert_eq!(
            after(&DiscordError::RateLimited { retry_after_s: 0.2 }),
            Duration::from_secs(1),
            "never under a second"
        );
        assert_eq!(
            after(&DiscordError::Paused {
                retry_after_s: 120.0
            }),
            Duration::from_secs(120)
        );
        let forbidden = DiscordError::Status {
            status: 403,
            code: 50013,
            detail: "Missing Permissions".into(),
        };
        assert_eq!(after(&forbidden), REFUSED_FOR);
        assert_eq!(after(&DiscordError::TokenRejected), REFUSED_FOR);
        let down = DiscordError::Status {
            status: 502,
            code: 0,
            detail: "upstream".into(),
        };
        assert_eq!(after(&down), FAILED_FOR);
        assert_eq!(after(&DiscordError::Transport("reset".into())), FAILED_FOR);
    }
}
