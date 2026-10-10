//! Drain a work list a bounded group at a time, each group's capture done under a
//! hold that is released before the group's slow half runs.
//!
//! The persistence thread reads device K/V that a KV compaction may relocate. A
//! compaction sweeps the holders it can reach — block tables, residences, projection
//! caches — and this thread's locals are not among them. A clone of a residence's
//! `SealedSequence`s held here across a pass keeps the slots it names alive after the
//! residence itself has moved on, so the next pass finds them live, plans them again,
//! and finds no holder to move onto the copy: every one of them is reported
//! `unwitnessed`, its claim wasted and its source never reclaimed, for as long as the
//! clone lives. Holding the clones across a redo-log write made that seconds.
//!
//! So a group is snapshotted and gathered while the hold keeps compaction out, every
//! snapshot is dropped before the hold goes back, and only the gathered host bytes
//! cross into the write that follows.

use std::collections::HashSet;
use std::hash::Hash;

/// What one drain did.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HeldDrain {
    /// Groups that were snapshotted and finished.
    pub groups: usize,
    /// Whether the hold was refused before the list ran dry; what is left is the
    /// next pass's.
    pub deferred: bool,
}

/// Drain `snapshot`'s work list `max_group` items at a time.
///
/// For each group: take `hold`, snapshot the list, keep at most `max_group` items not
/// already tried this drain, hand each to `gather` — which consumes the snapshot, so it
/// is dropped there, under the hold — then release the hold and pass what was gathered
/// to `finish`.
///
/// An item is tried once per drain whether its gather succeeds or not, so an item that
/// keeps failing cannot hold the rest of the list back and the drain always ends. A
/// refused hold ends it too: nothing waits on a compaction, and the next pass resumes.
pub(crate) fn drain_in_held_groups<H, K, S, G>(
    max_group: usize,
    mut hold: impl FnMut() -> Option<H>,
    mut snapshot: impl FnMut() -> Vec<(K, S)>,
    mut gather: impl FnMut(K, S) -> Option<G>,
    mut finish: impl FnMut(Vec<(K, G)>),
) -> HeldDrain
where
    K: Copy + Eq + Hash,
{
    assert!(
        max_group > 0,
        "a drain needs room for at least one item a group"
    );
    let mut drain = HeldDrain::default();
    let mut tried: HashSet<K> = HashSet::new();
    loop {
        let Some(held) = hold() else {
            drain.deferred = true;
            return drain;
        };
        // Everything past the group is dropped here, still under the hold.
        let mut group: Vec<(K, S)> = snapshot()
            .into_iter()
            .filter(|(key, _)| !tried.contains(key))
            .collect();
        let more = group.len() > max_group;
        group.truncate(max_group);
        if group.is_empty() {
            return drain;
        }
        let mut gathered = Vec::with_capacity(group.len());
        for (key, item) in group {
            tried.insert(key);
            if let Some(g) = gather(key, item) {
                gathered.push((key, g));
            }
        }
        drop(held);
        finish(gathered);
        drain.groups += 1;
        if !more {
            return drain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    type Log = Rc<RefCell<Vec<String>>>;

    /// A snapshot that records when it is dropped — a stand-in for a clone of a
    /// residence's sealed K/V, which keeps its slots alive until then.
    struct Snap {
        key: u32,
        log: Log,
    }

    impl Drop for Snap {
        fn drop(&mut self) {
            self.log.borrow_mut().push(format!("drop {}", self.key));
        }
    }

    /// A hold that records its release.
    struct Hold {
        log: Log,
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            self.log.borrow_mut().push("release".into());
        }
    }

    fn snaps(keys: &[u32], log: &Log) -> Vec<(u32, Snap)> {
        keys.iter()
            .map(|&key| {
                (
                    key,
                    Snap {
                        key,
                        log: Rc::clone(log),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn no_snapshot_outlives_its_hold() {
        let log: Log = Rc::default();
        let pending = Rc::new(RefCell::new(vec![1u32, 2, 3, 4, 5]));
        let drain = drain_in_held_groups(
            2,
            || {
                log.borrow_mut().push("hold".into());
                Some(Hold {
                    log: Rc::clone(&log),
                })
            },
            || snaps(&pending.borrow(), &log),
            |key, _snap| Some(key),
            |done| {
                log.borrow_mut().push(format!("finish {done:?}"));
                pending
                    .borrow_mut()
                    .retain(|k| !done.iter().any(|(d, _)| d == k));
            },
        );
        assert_eq!(
            drain,
            HeldDrain {
                groups: 3,
                deferred: false
            }
        );
        let log = log.borrow();
        assert_eq!(
            *log,
            [
                // Group 1: the three snapshots past the group go at once, the two in it
                // as they are gathered — all before the release.
                "hold",
                "drop 3",
                "drop 4",
                "drop 5",
                "drop 1",
                "drop 2",
                "release",
                "finish [(1, 1), (2, 2)]",
                "hold",
                "drop 5",
                "drop 3",
                "drop 4",
                "release",
                "finish [(3, 3), (4, 4)]",
                "hold",
                "drop 5",
                "release",
                "finish [(5, 5)]",
            ]
        );
    }

    #[test]
    fn a_refused_hold_snapshots_nothing_and_defers() {
        let snapshots = RefCell::new(0);
        let finished = RefCell::new(0);
        let drain = drain_in_held_groups(
            4,
            || None::<()>,
            || {
                *snapshots.borrow_mut() += 1;
                vec![(1u32, ())]
            },
            |_, ()| Some(()),
            |_| *finished.borrow_mut() += 1,
        );
        assert_eq!(
            drain,
            HeldDrain {
                groups: 0,
                deferred: true
            }
        );
        assert_eq!(*snapshots.borrow(), 0);
        assert_eq!(*finished.borrow(), 0);
    }

    #[test]
    fn items_that_keep_failing_cannot_hold_the_rest_back() {
        // 1 and 2 never gather and stay pending; 3 and 4 must still be reached, and the
        // drain must end rather than snapshot 1 and 2 forever.
        let pending = RefCell::new(vec![1u32, 2, 3, 4]);
        let mut finished = Vec::new();
        let drain = drain_in_held_groups(
            2,
            || Some(()),
            || pending.borrow().iter().map(|&k| (k, ())).collect(),
            |key, ()| (key > 2).then_some(key),
            |done| {
                pending
                    .borrow_mut()
                    .retain(|k| !done.iter().any(|(d, _)| d == k));
                finished.push(done);
            },
        );
        assert_eq!(
            drain,
            HeldDrain {
                groups: 2,
                deferred: false
            }
        );
        assert_eq!(finished, [vec![], vec![(3, 3), (4, 4)]]);
        assert_eq!(*pending.borrow(), [1, 2]);
    }

    #[test]
    fn a_hold_refused_mid_drain_keeps_what_was_finished() {
        let mut holds = 0;
        let pending = RefCell::new(vec![1u32, 2, 3]);
        let mut finished = Vec::new();
        let drain = drain_in_held_groups(
            1,
            || {
                holds += 1;
                (holds < 3).then_some(())
            },
            || pending.borrow().iter().map(|&k| (k, ())).collect(),
            |key, ()| Some(key),
            |done| {
                pending
                    .borrow_mut()
                    .retain(|k| !done.iter().any(|(d, _)| d == k));
                finished.extend(done);
            },
        );
        assert_eq!(
            drain,
            HeldDrain {
                groups: 2,
                deferred: true
            }
        );
        assert_eq!(finished, [(1, 1), (2, 2)]);
        assert_eq!(*pending.borrow(), [3]);
    }

    #[test]
    fn an_empty_list_finishes_nothing() {
        let mut finished = 0;
        let drain = drain_in_held_groups(
            3,
            || Some(()),
            Vec::<(u32, ())>::new,
            |_, ()| Some(()),
            |_| finished += 1,
        );
        assert_eq!(drain, HeldDrain::default());
        assert_eq!(finished, 0);
    }
}
