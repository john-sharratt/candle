//! The working-set marks a dialogue's turns carry, and the locks they rebuild
//! (`docs/zend_working_set.md` §4.8).
//!
//! The turn that carries a tool round's results is tagged with what the round
//! did to the working set, in the order it happened: one [`lock_tag`] per call
//! the fast path served, one [`release_one_tag`] per lock whose promise ended
//! — its reply left the dialogue's projection, or the file it holds was
//! written — and [`RELEASE_TAG`] where every lock was released, after a call
//! whose effects are unknown. The history is then the record of every promise
//! made to the model, and a restart rebuilds the locks from it instead of
//! re-deciding them.
//!
//! **The marks are not gather scope.** A turn's tags also say which tag-scoped
//! gallery it belongs to, and "no tags" is what makes a turn ordinary dialogue.
//! Every reader of tags as scope goes through [`gather_tags`] or
//! [`is_dialogue`], so a dialogue turn that carries only marks stays dialogue.

use crate::projection::TimelineId;

/// Every mark starts with this.
const MARK_PREFIX: &str = "working_set:";

/// The prefix of a lock mark; the conversation's raw timeline id follows.
const LOCK_PREFIX: &str = "working_set:lock:";

/// The mark for "every lock was released here".
pub const RELEASE_TAG: &str = "working_set:release";

/// The prefix of a one-lock release mark; the conversation's raw timeline id
/// follows.
const RELEASE_ONE_PREFIX: &str = "working_set:release:";

/// The mark for a call served by pinning `timeline`.
pub fn lock_tag(timeline: TimelineId) -> String {
    format!("{LOCK_PREFIX}{}", timeline.raw())
}

/// The mark for "the lock on `timeline` alone was released here".
pub fn release_one_tag(timeline: TimelineId) -> String {
    format!("{RELEASE_ONE_PREFIX}{}", timeline.raw())
}

/// Whether `tag` is a working-set mark rather than a gather scope.
pub fn is_mark(tag: &str) -> bool {
    tag.starts_with(MARK_PREFIX)
}

/// `tags` without the working-set marks — the turn's gather scope.
pub fn gather_tags(tags: &[String]) -> impl Iterator<Item = &String> {
    tags.iter().filter(|t| !is_mark(t))
}

/// Whether a turn with `tags` is ordinary dialogue: it belongs to no gather
/// scope.
pub fn is_dialogue(tags: &[String]) -> bool {
    gather_tags(tags).next().is_none()
}

/// One mark, read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    Lock(TimelineId),
    Release,
    ReleaseOne(TimelineId),
}

/// The mark `tag` is, if it is one this build understands.
pub fn parse(tag: &str) -> Option<Mark> {
    if tag == RELEASE_TAG {
        return Some(Mark::Release);
    }
    if let Some(raw) = tag.strip_prefix(RELEASE_ONE_PREFIX) {
        return TimelineId::from_raw(raw.parse().ok()?).map(Mark::ReleaseOne);
    }
    let raw = tag.strip_prefix(LOCK_PREFIX)?.parse().ok()?;
    TimelineId::from_raw(raw).map(Mark::Lock)
}

/// The locks standing at the end of a conversation whose turns carry `turns`
/// (oldest first, each its tag list): the lock marks not released since, in
/// the order first served, each conversation once.
pub fn standing_locks<'a>(turns: impl IntoIterator<Item = &'a [String]>) -> Vec<TimelineId> {
    promised_locks(turns.into_iter().enumerate())
        .into_iter()
        .map(|(tl, _)| tl)
        .collect()
}

/// [`standing_locks`], each with the turn that carries its promise: the
/// latest turn whose marks lock it, as `turns` names it (oldest first, each
/// its key and tag list). A lock served again later is promised by the later
/// reply — the earlier one may leave the projection while it still stands.
pub fn promised_locks<'a, K: Copy>(
    turns: impl IntoIterator<Item = (K, &'a [String])>,
) -> Vec<(TimelineId, K)> {
    let mut locks: Vec<(TimelineId, K)> = Vec::new();
    for (turn, tags) in turns {
        for mark in tags.iter().filter_map(|t| parse(t)) {
            match mark {
                Mark::Release => locks.clear(),
                Mark::ReleaseOne(tl) => locks.retain(|(held, _)| *held != tl),
                Mark::Lock(tl) => match locks.iter_mut().find(|(held, _)| *held == tl) {
                    Some(held) => held.1 = turn,
                    None => locks.push((tl, turn)),
                },
            }
        }
    }
    locks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    fn tags(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// The marks are exact strings — they are persisted with the turn.
    #[test]
    fn marks_are_spelled_exactly() {
        assert_eq!(lock_tag(tl(42)), "working_set:lock:42");
        assert_eq!(RELEASE_TAG, "working_set:release");
        assert_eq!(parse("working_set:lock:42"), Some(Mark::Lock(tl(42))));
        assert_eq!(parse("working_set:release"), Some(Mark::Release));
        assert_eq!(parse("working_set:lock:x"), None);
        assert_eq!(parse("tool"), None);
    }

    /// A dialogue turn carrying only marks is still dialogue; a gallery turn
    /// with a mark keeps its scope.
    #[test]
    fn marks_are_not_gather_scope() {
        assert!(is_dialogue(&[]));
        assert!(is_dialogue(&tags(&[
            "working_set:release",
            "working_set:lock:3"
        ])));
        assert!(!is_dialogue(&tags(&["tool", "working_set:release"])));
        let turn = tags(&["working_set:lock:3", "code"]);
        let scope: Vec<&String> = gather_tags(&turn).collect();
        assert_eq!(scope, vec!["code"]);
    }

    /// Locks come back from the marks after the last release, and none from
    /// before it, in the order they were served.
    #[test]
    fn only_the_locks_after_the_last_release_stand() {
        let turns = [
            tags(&["working_set:lock:1"]),
            tags(&["working_set:release"]),
            tags(&["working_set:lock:2", "working_set:lock:3"]),
            tags(&[]),
            tags(&["working_set:lock:4", "working_set:lock:2"]),
        ];
        let locks = standing_locks(turns.iter().map(|t| t.as_slice()));
        assert_eq!(locks, vec![tl(2), tl(3), tl(4)]);
    }

    /// A one-lock release ends only that lock; the others stand, and a lock
    /// served again after its release stands again.
    #[test]
    fn a_one_lock_release_ends_only_its_own_lock() {
        assert_eq!(release_one_tag(tl(7)), "working_set:release:7");
        assert_eq!(
            parse("working_set:release:7"),
            Some(Mark::ReleaseOne(tl(7)))
        );
        let turns = [
            tags(&["working_set:lock:1", "working_set:lock:2"]),
            tags(&["working_set:release:1"]),
            tags(&["working_set:lock:3"]),
            tags(&["working_set:release:3", "working_set:lock:3"]),
        ];
        let locks = standing_locks(turns.iter().map(|t| t.as_slice()));
        assert_eq!(locks, vec![tl(2), tl(3)]);
    }

    /// Each standing lock comes with the turn whose reply promised it, the
    /// latest when it was served more than once.
    #[test]
    fn each_lock_is_promised_by_the_latest_turn_that_served_it() {
        let turns = [
            (10u32, tags(&["working_set:lock:1"])),
            (11, tags(&["working_set:lock:2"])),
            (12, tags(&[])),
            (13, tags(&["working_set:lock:1"])),
        ];
        let promised = promised_locks(turns.iter().map(|(i, t)| (*i, t.as_slice())));
        assert_eq!(promised, vec![(tl(1), 13), (tl(2), 11)]);
    }

    /// Reads served before a round's first write precede its release in the
    /// same turn's list, so they are released too — as they were live.
    #[test]
    fn a_release_later_in_the_same_turn_releases_the_reads_before_it() {
        let turns = [tags(&[
            "working_set:lock:1",
            "working_set:release",
            "working_set:lock:2",
        ])];
        let locks = standing_locks(turns.iter().map(|t| t.as_slice()));
        assert_eq!(locks, vec![tl(2)]);
    }
}
