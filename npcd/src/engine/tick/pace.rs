//! How alert a character is allowed to become, and the deadline that means it
//! is not scheduled at all.
//!
//! A pace caps how alert a character reads and never schedules a thought: a
//! character thinks when something reaches it. Each character has its own
//! floor, so one with continuous work is not relaxed to a slow heartbeat that
//! would stall it. The fast end is clamped, since thinking faster than the
//! world moves only re-reads a world that has not changed.

use super::*;

/// The slowest a wholly idle character thinks. Long, because a character with
/// nothing happening genuinely has nothing to think about, and the cost of a
/// wake is a decode.
pub const IDLE_HEARTBEAT: Duration = Duration::from_secs(120);

/// The fastest a character settles back to after something happened.
pub const ALERT_HEARTBEAT: Duration = Duration::from_secs(4);

/// A character that is not scheduled to think at all.
///
/// **The ordinary state of a quiet character**, and the whole of what "no idle
/// events" means: nothing is due, so nothing thinks, until the world puts
/// something in its inbox. Spelled as a deadline past any clock rather than as
/// an `Option`, so the one comparison in [`Scheduler::wait_due`] answers both
/// "is it time" and "is it scheduled".
pub(super) const NEVER: u64 = u64::MAX;

/// The slowest a character is allowed to become.
///
/// The relax curve is right for a character whose life is *reactive*: nothing
/// happening means nothing to think about, and settling to a two-minute
/// heartbeat is the whole reason a thousand of them are affordable.
///
/// It is wrong for a character whose work is continuous. A Maker with a
/// standing task always has something to do, so its quiet turns are not empty —
/// they carry the task forward. Letting it settle to two minutes would not save
/// anything; it would stall the work and read as a character that had forgotten
/// what it was doing.
///
/// So the floor is per character. Everything else about the curve — tighten on
/// a preempt, relax while quiet — is unchanged; this only says how far the
/// relaxing is allowed to go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pace {
    pub(super) slowest: Duration,
}

impl Pace {
    /// A character that thinks when something happens to it.
    pub const AMBIENT: Pace = Pace {
        slowest: IDLE_HEARTBEAT,
    };

    /// A character with work in front of it, which never truly goes quiet.
    pub const WORKING: Pace = Pace {
        slowest: ALERT_HEARTBEAT,
    };

    /// Thinks at least this often.
    ///
    /// Clamped to [`ALERT_HEARTBEAT`] at the fast end: below that a character
    /// thinks faster than the world moves, and spends decodes discovering that
    /// nothing has changed since the last one.
    pub fn at_most(gap: Duration) -> Pace {
        Pace {
            slowest: gap.max(ALERT_HEARTBEAT),
        }
    }

    pub fn slowest(self) -> Duration {
        self.slowest
    }
}

impl Default for Pace {
    fn default() -> Self {
        Pace::AMBIENT
    }
}
