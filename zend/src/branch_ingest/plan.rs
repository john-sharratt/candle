//! What one pass does to one layer, decided from keys alone
//! (`docs/zend_branch_ingest.md` §7.1–§7.2): which units to ingest, and which
//! conversations to tombstone.

use std::collections::{HashMap, HashSet};

use candle_conversation::projection::TimelineId;

/// A unit on some branch: its key, and the path or folder it is of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Live<'a> {
    pub key: &'a str,
    pub subject: &'a str,
}

/// A committed conversation of the layer: its key, and the path or folder
/// it is of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Committed {
    pub timeline: TimelineId,
    pub key: String,
    pub subject: String,
}

/// One pass over one layer.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Indices into the live units of those no conversation holds, in the
    /// order given.
    pub queued: Vec<usize>,
    /// Conversations to tombstone now: of a key no branch holds, whose path
    /// or folder has nothing queued to replace it — and any second
    /// conversation of a key another already holds.
    pub tombstone: Vec<TimelineId>,
}

/// Plan a pass: `live` is the corpus, `committed` what the layer holds.
///
/// A dead key whose path or folder has a queued key stays until that
/// replacement commits — its ingest tombstones it then — so a path is never
/// missing from the layer while it is being read again.
pub fn plan(live: &[Live<'_>], committed: &[Committed]) -> Plan {
    let live_keys: HashSet<&str> = live.iter().map(|l| l.key).collect();
    let mut holder: HashMap<&str, TimelineId> = HashMap::new();
    let mut tombstone = Vec::new();
    for c in committed {
        match holder.get(c.key.as_str()) {
            Some(_) => tombstone.push(c.timeline),
            None => {
                holder.insert(&c.key, c.timeline);
            }
        }
    }
    let queued: Vec<usize> = live
        .iter()
        .enumerate()
        .filter(|(_, l)| !holder.contains_key(l.key))
        .map(|(i, _)| i)
        .collect();
    let replacing: HashSet<&str> = queued.iter().map(|&i| live[i].subject).collect();
    for c in committed {
        let held_here = holder.get(c.key.as_str()) == Some(&c.timeline);
        if held_here
            && !live_keys.contains(c.key.as_str())
            && !replacing.contains(c.subject.as_str())
        {
            tombstone.push(c.timeline);
        }
    }
    Plan { queued, tombstone }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn committed(n: u64, key: &str, subject: &str) -> Committed {
        Committed {
            timeline: tl(n),
            key: key.to_string(),
            subject: subject.to_string(),
        }
    }

    fn live<'a>(key: &'a str, subject: &'a str) -> Live<'a> {
        Live { key, subject }
    }

    /// **What no conversation holds is queued; what a conversation holds is
    /// left alone** — however many branches carry it.
    #[test]
    fn what_is_not_held_is_queued() {
        let p = plan(
            &[live("a@1", "a"), live("b@1", "b"), live("a@2", "a")],
            &[committed(1, "a@1", "a")],
        );
        assert_eq!(p.queued, [1, 2]);
        assert!(p.tombstone.is_empty(), "a@1 is still on a branch");
    }

    /// **A path gone from every branch goes at once.**
    #[test]
    fn a_path_on_no_branch_is_tombstoned() {
        let p = plan(
            &[live("a@1", "a")],
            &[committed(1, "a@1", "a"), committed(2, "gone@1", "gone")],
        );
        assert!(p.queued.is_empty());
        assert_eq!(p.tombstone, [tl(2)]);
    }

    /// **A version no branch holds stays while its path's new version is
    /// queued** — its ingest retires it — and goes once nothing replaces it.
    #[test]
    fn a_replaced_version_waits_for_its_replacement() {
        let old = [committed(1, "a@1", "a")];
        let waiting = plan(&[live("a@2", "a")], &old);
        assert_eq!(waiting.queued, [0]);
        assert!(waiting.tombstone.is_empty());

        let replaced = plan(
            &[live("a@2", "a")],
            &[committed(1, "a@1", "a"), committed(2, "a@2", "a")],
        );
        assert!(replaced.queued.is_empty());
        assert_eq!(replaced.tombstone, [tl(1)]);
    }

    /// **Two conversations of one key are one too many**: the first stays.
    #[test]
    fn a_duplicate_of_a_held_key_is_tombstoned() {
        let p = plan(
            &[live("a@1", "a")],
            &[committed(1, "a@1", "a"), committed(2, "a@1", "a")],
        );
        assert!(p.queued.is_empty());
        assert_eq!(p.tombstone, [tl(2)]);
    }

    #[test]
    fn nothing_live_and_nothing_held_plans_nothing() {
        assert_eq!(plan(&[], &[]), Plan::default());
    }
}
