//! When a timeline tombstone may shrink to its deleted-set entry
//! (`docs/replicated_substrate.md` §6.8).
//!
//! Two conditions, both required: a complete compaction has run since the
//! tombstone (the deleted set records which, as *swept*), and the tombstone is
//! older than the member's retention by its `hlc`, whose high bits read as
//! milliseconds. Expiry only decides when a tombstone's bytes are reclaimed —
//! the timeline stays dead either way — so members need not agree on the
//! retention, and nothing limits how long one may be away.

use std::time::Duration;

use super::identity::COUNTER_BITS;

/// How long a timeline tombstone is kept by default before it may shrink:
/// `--tombstone-retention`'s default.
pub const DEFAULT_TOMBSTONE_RETENTION: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// One compaction's expiry rule: its clock and the member's retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TombstoneExpiry {
    /// The wall clock the compaction runs at, in milliseconds since the epoch.
    pub now_ms: u64,
    pub retention: Duration,
}

impl TombstoneExpiry {
    /// Whether a tombstone written at `hlc` is older than the retention. A
    /// tombstone written before identities carries none, and has been
    /// standing longer than any retention.
    pub fn is_old(&self, hlc: u64) -> bool {
        let written_ms = hlc >> COUNTER_BITS;
        self.now_ms.saturating_sub(written_ms) > self.retention.as_millis() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_MS: u64 = 24 * 60 * 60 * 1000;

    fn hlc_at(ms: u64) -> u64 {
        ms << COUNTER_BITS
    }

    #[test]
    fn a_tombstone_is_old_past_the_retention_and_not_before() {
        let expiry = TombstoneExpiry {
            now_ms: 100 * DAY_MS,
            retention: Duration::from_secs(30 * 24 * 60 * 60),
        };
        assert!(!expiry.is_old(hlc_at(70 * DAY_MS)), "exactly 30 days");
        assert!(expiry.is_old(hlc_at(70 * DAY_MS - 1)));
        assert!(
            !expiry.is_old(hlc_at(99 * DAY_MS) | 0xFFFF),
            "the counter bits"
        );
        assert!(expiry.is_old(0), "an unstamped tombstone");
        assert!(
            !expiry.is_old(hlc_at(200 * DAY_MS)),
            "a clock behind the writer's"
        );
    }
}
