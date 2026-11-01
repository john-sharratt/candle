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
//!
//! Each refusal in a row doubles the hold ([`backoff`]), to at most
//! [`HOLD_MAX`]; a beat that gets through unrefused starts it again, and so
//! does setting the world's server in the console — the person has just
//! changed what Discord was refusing. A bot's own DM is held the same way,
//! keyed `dm/<channel_id>` (`crate::discord::dm::outbound`).
//!
//! The holds start short: a permission granted a minute after the refusal
//! is used within a minute. Even at the first rung, a refusal every 30 s is
//! 20 invalid requests in Discord's ten minutes, against the 10,000 it bans
//! at.

use std::time::Duration;

use crate::discord::error::DiscordError;

/// How long a first refusal only a person can lift holds the mirror.
pub const REFUSED_FOR: Duration = Duration::from_secs(30);

/// How long any other first failure holds the mirror.
pub const FAILED_FOR: Duration = Duration::from_secs(5);

/// The longest a mirror is held, however many refusals in a row.
pub const HOLD_MAX: Duration = Duration::from_secs(15 * 60);

/// How long the `refusals`th refusal in a row holds the mirror, `wait` the
/// least its answer asks: doubled for each refusal before it, and never
/// more than [`HOLD_MAX`]. A mirror refused again as soon as it asks backs
/// off from seconds to a quarter of an hour rather than asking at one
/// fixed rate.
pub fn backoff(wait: Duration, refusals: u32) -> Duration {
    wait.saturating_mul(1u32 << refusals.saturating_sub(1).min(16))
        .min(HOLD_MAX)
}

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
    /// for thirty seconds, anything else for five.**
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

    /// **Each refusal in a row doubles the hold, to at most a quarter of an
    /// hour**: a missing permission is asked about at 30 s, 1, 2, 4 and
    /// 8 minutes, then every 15.
    #[test]
    fn refusals_in_a_row_double_the_hold_to_a_quarter_hour() {
        let s = Duration::from_secs;
        assert_eq!(backoff(REFUSED_FOR, 1), s(30));
        assert_eq!(backoff(REFUSED_FOR, 2), s(60));
        assert_eq!(backoff(REFUSED_FOR, 3), s(120));
        assert_eq!(backoff(REFUSED_FOR, 5), s(480));
        assert_eq!(backoff(REFUSED_FOR, 6), HOLD_MAX);
        assert_eq!(backoff(REFUSED_FOR, 200), HOLD_MAX, "no overflow");
        assert_eq!(backoff(FAILED_FOR, 1), s(5));
        assert_eq!(backoff(FAILED_FOR, 4), s(40));
        assert_eq!(backoff(s(7200), 1), HOLD_MAX, "capped from the first");
    }
}
