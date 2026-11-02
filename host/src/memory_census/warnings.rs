//! When the census warns (`docs/memory_census.md` §7): the machine's commit
//! against its limit, and how often a standing warning repeats.

use heap_census::os::MachineCommit;

/// The share of the commit limit at which the census warns.
pub const COMMIT_HIGH: f64 = 0.85;
/// The share at which it reports an error.
pub const COMMIT_CRITICAL: f64 = 0.95;

/// How often a warning that still holds is repeated.
pub const REPEAT_MS: u64 = 30 * 60 * 1000;

/// Where the machine's commit stands against its limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CommitLevel {
    Normal,
    High,
    Critical,
}

impl CommitLevel {
    pub fn of(c: &MachineCommit) -> CommitLevel {
        if c.limit_bytes == 0 {
            return CommitLevel::Normal;
        }
        let share = c.committed_bytes as f64 / c.limit_bytes as f64;
        if share >= COMMIT_CRITICAL {
            CommitLevel::Critical
        } else if share >= COMMIT_HIGH {
            CommitLevel::High
        } else {
            CommitLevel::Normal
        }
    }
}

/// Hours until the commit limit at `grew_bytes` over `over_ms`; `None` when it
/// is not growing.
pub fn hours_to_limit(c: &MachineCommit, grew_bytes: i64, over_ms: u64) -> Option<f64> {
    if grew_bytes <= 0 || over_ms == 0 {
        return None;
    }
    let per_hour = grew_bytes as f64 * 3_600_000.0 / over_ms as f64;
    Some(c.limit_bytes.saturating_sub(c.committed_bytes) as f64 / per_hour)
}

/// A warning that repeats at most every [`REPEAT_MS`] while it holds, and at
/// once when what it says changes — a commit level crossed either way, a
/// growth verdict that names another cause.
#[derive(Debug, Default)]
pub struct Repeat<L: Copy + Eq> {
    last: Option<(L, u64)>,
}

impl<L: Copy + Eq> Repeat<L> {
    pub fn new() -> Self {
        Repeat { last: None }
    }

    /// Whether `level` is to be said at `now_ms`. Saying it is recorded.
    pub fn due(&mut self, level: L, now_ms: u64) -> bool {
        let due = match self.last {
            None => true,
            Some((was, at)) => level != was || now_ms.saturating_sub(at) >= REPEAT_MS,
        };
        if due {
            self.last = Some((level, now_ms));
        }
        due
    }

    /// The condition has passed: the next time it holds is said at once.
    pub fn clear(&mut self) {
        self.last = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn commit(committed: u64, limit: u64) -> MachineCommit {
        MachineCommit {
            committed_bytes: committed,
            limit_bytes: limit,
        }
    }

    /// **BlackWorld at 20:40Z, 251.4 of 280.7 GB, warns**, at 89.6%.
    #[test]
    fn the_commit_level_follows_the_share_of_the_limit() {
        assert_eq!(
            CommitLevel::of(&commit(200 * GB, 280 * GB)),
            CommitLevel::Normal
        );
        assert_eq!(
            CommitLevel::of(&commit(251_355_602_944, 280_658_038_784)),
            CommitLevel::High
        );
        assert_eq!(
            CommitLevel::of(&commit(270 * GB, 280 * GB)),
            CommitLevel::Critical
        );
        assert_eq!(CommitLevel::of(&commit(1, 0)), CommitLevel::Normal);
    }

    /// **BlackWorld's 29.3 GB of headroom at 1.40 GB per 20.7 min is about
    /// seven hours.**
    #[test]
    fn hours_to_the_limit_follow_the_slope() {
        let h = hours_to_limit(
            &commit(251_355_602_944, 280_658_038_784),
            1_401_941_504,
            1_240_875,
        )
        .unwrap();
        assert!((7.1..7.4).contains(&h), "{h}");
        assert_eq!(hours_to_limit(&commit(1, 2), 0, 1), None);
    }

    #[test]
    fn a_standing_warning_repeats_only_every_half_hour_or_when_it_changes() {
        let mut r = Repeat::new();
        assert!(r.due(CommitLevel::High, 0));
        assert!(!r.due(CommitLevel::High, REPEAT_MS - 1));
        assert!(
            r.due(CommitLevel::Critical, REPEAT_MS - 1),
            "a change is said at once"
        );
        assert!(!r.due(CommitLevel::Critical, REPEAT_MS));
        assert!(r.due(CommitLevel::High, REPEAT_MS), "and so is easing");
        assert!(r.due(CommitLevel::High, 2 * REPEAT_MS));
        r.clear();
        assert!(r.due(CommitLevel::High, 2 * REPEAT_MS + 1));
    }
}
