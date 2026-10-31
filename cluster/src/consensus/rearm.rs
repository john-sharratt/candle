//! How a lease accepted before a restart counts after it (§5.9).
//!
//! The vote records the wall-clock time at which this acceptor counts the
//! lease to end: its acceptance plus its `ttl`. On reopening, the lease is
//! armed for what is left of that, by the wall clock read at the reopen,
//! clamped to `[0, ttl]`:
//!
//! - **Some of the term left**: armed for exactly that. Every restart counts
//!   against the same recorded end, so restarts never extend a lease.
//! - **None left**: not armed — the key is free at once — and its slot is
//!   freed one `ttl` after the recorded end, as it would have been without the
//!   restart.
//! - **The wall clock reads earlier than the acceptance**: it went back, so
//!   the record says nothing about how long the lease has run. The lease is
//!   armed for its full `ttl` from the reopen, which errs on the side of the
//!   holder.
//! - **No end recorded** (`0`, a vote written before votes carried one): the
//!   same — the full `ttl` from the reopen. Dropping it instead would free, at
//!   this acceptor, a lease its holder may still count, and a reopened member
//!   and a reopened gateway together are a quorum of a group of two.
//!
//! **What this assumes.** The wall clock moves forward across a restart by no
//! more than the time that really passed, give or take [`super::HOLDER_MARGIN`]
//! (2 s), the margin by which a holder stops before its acceptors. A forward
//! step larger than that while the acceptor is down ends the lease early at
//! this one acceptor, and with a quorum of acceptors so stepped, another
//! proposer could take a lease its holder still counts. A backward step is
//! safe: it re-arms for at least what was left. Members and gateways keep
//! their clocks NTP-disciplined, and the gateway already measures each
//! member's offset at admission (§3.4).

use std::time::Duration;

use super::wall::WallMillis;

/// A reopened lease: how long it is armed for, and how long until its slot is
/// freed, both from the reopen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rearm {
    pub left: Duration,
    pub free_in: Duration,
}

/// The lease recorded to end at `ends`, of term `ttl`, reopened at `wall`.
pub fn rearm(ends: WallMillis, ttl: Duration, wall: WallMillis) -> Rearm {
    let ttl_ms = ttl.as_millis() as u64;
    let accepted = ends.0.saturating_sub(ttl_ms);
    if ends.0 == 0 || wall.0 < accepted {
        return Rearm {
            left: ttl,
            free_in: ttl + ttl,
        };
    }
    Rearm {
        left: Duration::from_millis(ends.0.saturating_sub(wall.0)),
        free_in: Duration::from_millis(ends.0.saturating_add(ttl_ms).saturating_sub(wall.0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: u64 = 60_000;
    const T: u64 = 1_793_000_000_000;

    fn ttl() -> Duration {
        Duration::from_millis(20 * MIN)
    }

    fn mins(m: u64) -> Duration {
        Duration::from_millis(m * MIN)
    }

    #[test]
    fn a_reopen_inside_the_term_arms_what_is_left() {
        let ends = WallMillis(T + 20 * MIN);
        assert_eq!(
            rearm(ends, ttl(), WallMillis(T + 5 * MIN)),
            Rearm {
                left: mins(15),
                free_in: mins(35)
            }
        );
        assert_eq!(
            rearm(ends, ttl(), WallMillis(T)),
            Rearm {
                left: mins(20),
                free_in: mins(40)
            },
            "at the acceptance itself, the whole term"
        );
    }

    #[test]
    fn a_reopen_past_the_term_arms_nothing_and_frees_on_time() {
        let ends = WallMillis(T + 20 * MIN);
        assert_eq!(
            rearm(ends, ttl(), WallMillis(T + 25 * MIN)),
            Rearm {
                left: Duration::ZERO,
                free_in: mins(15)
            }
        );
        assert_eq!(
            rearm(ends, ttl(), WallMillis(T + 60 * MIN)),
            Rearm {
                left: Duration::ZERO,
                free_in: Duration::ZERO
            }
        );
    }

    #[test]
    fn a_clock_gone_back_arms_the_full_term() {
        let ends = WallMillis(T + 20 * MIN);
        assert_eq!(
            rearm(ends, ttl(), WallMillis(T - MIN)),
            Rearm {
                left: mins(20),
                free_in: mins(40)
            }
        );
    }

    #[test]
    fn a_vote_with_no_recorded_end_arms_the_full_term() {
        assert_eq!(
            rearm(WallMillis(0), ttl(), WallMillis(T)),
            Rearm {
                left: mins(20),
                free_in: mins(40)
            }
        );
    }

    /// **Restarts never extend a lease** — the production fault of
    /// 2026-10-31, where every web deploy re-armed `upgrade/zend` for a full
    /// 20 minutes at the gateway's tie-break acceptor.
    #[test]
    fn repeated_reopens_count_against_one_end() {
        let ends = WallMillis(T + 20 * MIN);
        let at5 = rearm(ends, ttl(), WallMillis(T + 5 * MIN));
        let at10 = rearm(ends, ttl(), WallMillis(T + 10 * MIN));
        assert_eq!(T + 5 * MIN + at5.left.as_millis() as u64, T + 20 * MIN);
        assert_eq!(T + 10 * MIN + at10.left.as_millis() as u64, T + 20 * MIN);
    }
}
