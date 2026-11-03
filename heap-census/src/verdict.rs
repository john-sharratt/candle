//! The verdict on growth (`docs/memory_census.md` §2): over a window, did the
//! process grow, and if so, is it our code holding more, the heap keeping freed
//! memory, or something outside the heap?
//!
//! | Private | Heap committed | Live | Verdict |
//! |---|---|---|---|
//! | grows | grows | flat | [`Verdict::Retention`] |
//! | grows | grows | grows | [`Verdict::Owner`] |
//! | grows | flat | flat | [`Verdict::OutsideHeap`] |
//!
//! "Grows" for live and committed means at least half the private growth: the
//! share that explains it. Growth under the threshold is [`Verdict::Steady`],
//! whatever moved inside it.

use std::cmp::Reverse;

use serde::Serialize;

use super::classes::{upto_bytes, CLASSES};
use super::reading::Reading;

/// One size class's growth over the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ClassGrowth {
    pub class: usize,
    /// The largest request the class holds.
    pub upto_bytes: u64,
    pub grew_bytes: i64,
}

/// What the window shows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// Private memory grew less than the threshold.
    Steady,
    /// The heap committed more while our code held no more: freed memory kept.
    Retention,
    /// Our code holds more. The classes that grew most, largest first, when the
    /// counting allocator is installed.
    Owner { classes: Vec<ClassGrowth> },
    /// The process grew, but neither its heap nor our code did: a native
    /// library's own heap, a mapping, or direct virtual allocation.
    OutsideHeap,
    /// The process grew and our code holds no more, on a platform whose heap
    /// cannot say what it committed: either freed memory the heap keeps, or
    /// memory outside it — the store gauges tell which.
    NotHeld,
}

/// The window's figures and what they mean.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Growth {
    pub over_ms: u64,
    pub private_grew: i64,
    pub committed_grew: Option<i64>,
    pub live_grew: Option<i64>,
    #[serde(flatten)]
    pub verdict: Verdict,
}

/// How many of the most-grown classes an owner verdict names.
pub const NAMED_CLASSES: usize = 3;

fn grew(old: Option<u64>, new: Option<u64>) -> Option<i64> {
    Some(new? as i64 - old? as i64)
}

/// The verdict from `old` to `new`: steady unless private memory grew by at
/// least `threshold` bytes.
pub fn judge(old: &Reading, new: &Reading, threshold: u64) -> Growth {
    let private_grew = new.private_bytes as i64 - old.private_bytes as i64;
    let committed_grew = grew(old.heap_committed(), new.heap_committed());
    let live_grew = grew(old.live_bytes(), new.live_bytes());
    let explains = |d: Option<i64>| d.is_some_and(|d| d.saturating_mul(2) >= private_grew);
    // Our code holding more is an owner only when the heap committed more to
    // hold it — or the platform cannot say what the heap committed. Live bytes
    // growing inside space the heap had already committed, while the process
    // grows elsewhere, is growth outside the heap.
    let heap_grew = committed_grew.is_none() || explains(committed_grew);
    let verdict = if private_grew < threshold as i64 {
        Verdict::Steady
    } else if explains(live_grew) && heap_grew {
        Verdict::Owner {
            classes: grown_classes(old, new),
        }
    } else if committed_grew.is_none() {
        Verdict::NotHeld
    } else if explains(committed_grew) {
        Verdict::Retention
    } else {
        Verdict::OutsideHeap
    };
    Growth {
        over_ms: new.at_ms.saturating_sub(old.at_ms),
        private_grew,
        committed_grew,
        live_grew,
        verdict,
    }
}

/// The classes that grew most from `old` to `new`, largest first, at most
/// [`NAMED_CLASSES`]; none without the counting allocator.
pub fn grown_classes(old: &Reading, new: &Reading) -> Vec<ClassGrowth> {
    let (Some(o), Some(n)) = (&old.classes, &new.classes) else {
        return Vec::new();
    };
    let mut grown: Vec<ClassGrowth> = (0..CLASSES)
        .map(|k| ClassGrowth {
            class: k,
            upto_bytes: upto_bytes(k),
            grew_bytes: n[k] as i64 - o[k] as i64,
        })
        .filter(|g| g.grew_bytes > 0)
        .collect();
    grown.sort_by_key(|g| Reverse(g.grew_bytes));
    grown.truncate(NAMED_CLASSES);
    grown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::os::OsHeap;

    const GIB: u64 = 1 << 30;

    fn reading(at_ms: u64, private: u64, committed: u64, live: u64) -> Reading {
        let mut classes = [0u64; CLASSES];
        classes[15] = live;
        Reading {
            at_ms,
            private_bytes: private,
            heap: Some(OsHeap {
                allocated_bytes: live,
                committed_bytes: committed,
                read_us: 0,
            }),
            classes: Some(classes),
            allocated_total: Some(0),
        }
    }

    #[test]
    fn growth_under_the_threshold_is_steady() {
        let g = judge(
            &reading(0, 100 * GIB, 50 * GIB, 40 * GIB),
            &reading(1, 101 * GIB, 51 * GIB, 41 * GIB),
            2 * GIB,
        );
        assert_eq!(g.verdict, Verdict::Steady);
        assert_eq!(g.private_grew, GIB as i64);
    }

    /// **Committed up, live flat: retention.** BlackWorld's suspected shape.
    #[test]
    fn committed_growing_over_flat_live_is_retention() {
        let g = judge(
            &reading(0, 225 * GIB, 60 * GIB, 30 * GIB),
            &reading(1, 229 * GIB, 64 * GIB, 30 * GIB),
            2 * GIB,
        );
        assert_eq!(g.verdict, Verdict::Retention);
        assert_eq!(
            (g.committed_grew, g.live_grew),
            (Some(4 * GIB as i64), Some(0))
        );
    }

    /// **Live up: an owner, named by the class that grew.**
    #[test]
    fn live_growing_is_an_owner_named_by_its_class() {
        let g = judge(
            &reading(0, 225 * GIB, 60 * GIB, 30 * GIB),
            &reading(1, 229 * GIB, 64 * GIB, 34 * GIB),
            2 * GIB,
        );
        assert_eq!(
            g.verdict,
            Verdict::Owner {
                classes: vec![ClassGrowth {
                    class: 15,
                    upto_bytes: 32_768,
                    grew_bytes: 4 * GIB as i64
                }]
            }
        );
    }

    /// **Live up inside a heap that committed nothing more, while the process
    /// grew: outside the heap**, not an owner — the growth is not ours.
    #[test]
    fn live_growing_in_space_already_committed_is_not_an_owner() {
        let g = judge(
            &reading(0, 225 * GIB, 60 * GIB, 30 * GIB),
            &reading(1, 229 * GIB, 60 * GIB, 32 * GIB),
            2 * GIB,
        );
        assert_eq!(g.verdict, Verdict::OutsideHeap);
    }

    /// **Without the heap's committed figure, growth our code does not hold is
    /// named not held** — retention and outside the heap cannot be told apart.
    #[test]
    fn growth_without_the_heaps_figure_is_not_held() {
        let mut old = reading(0, 225 * GIB, 0, 30 * GIB);
        let mut new = reading(1, 229 * GIB, 0, 30 * GIB);
        old.heap = None;
        new.heap = None;
        assert_eq!(judge(&old, &new, 2 * GIB).verdict, Verdict::NotHeld);
        new.classes.as_mut().unwrap()[15] = 34 * GIB;
        assert!(matches!(
            judge(&old, &new, 2 * GIB).verdict,
            Verdict::Owner { .. }
        ));
    }

    /// **Neither heap nor live up: outside the heap.**
    #[test]
    fn growth_the_heap_does_not_explain_is_outside_it() {
        let g = judge(
            &reading(0, 225 * GIB, 60 * GIB, 30 * GIB),
            &reading(1, 229 * GIB, 60 * GIB, 30 * GIB),
            2 * GIB,
        );
        assert_eq!(g.verdict, Verdict::OutsideHeap);
    }

    /// **The verdict serialises flat**, as the log and the report carry it.
    #[test]
    fn a_verdict_reads_as_json() {
        let g = judge(
            &reading(0, 0, 0, 0),
            &reading(60_000, 4 * GIB, 4 * GIB, 0),
            2 * GIB,
        );
        assert_eq!(
            serde_json::to_string(&g).unwrap(),
            r#"{"over_ms":60000,"private_grew":4294967296,"committed_grew":4294967296,"live_grew":0,"verdict":"retention"}"#
        );
    }
}
