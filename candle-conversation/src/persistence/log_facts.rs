//! What a log publishes for any thread to read without a call on its owner.
//!
//! A status read has no business in the owner's queue: it wants one word the
//! log already holds, and in the queue it waits for whatever the owner is doing.
//! A member's upgrade loop asks for the record types every second, and on
//! BlackWorld (2026-10-31) it stood still for fifteen minutes behind a liveness
//! audit, with the member's status still showing the step it last wrote.
//!
//! Only facts a stale read cannot hurt live here. `own_sendable_through` does
//! not: a lease's next holder waits for that `oseq`, so a stale, low answer
//! would let it serve before it had what the last holder wrote.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

#[derive(Debug)]
struct Cells {
    held_types: AtomicU32,
    out_of_space: AtomicBool,
    dead_ratio: AtomicU32,
}

/// See the module documentation. Cheap to clone; every clone reads the same
/// cells.
#[derive(Debug, Clone)]
pub struct LogFacts(Arc<Cells>);

impl LogFacts {
    /// A log holding the record types `held_types`, one bit per tag, with no
    /// write failed and no liveness scan yet.
    pub(super) fn new(held_types: u32) -> LogFacts {
        LogFacts(Arc::new(Cells {
            held_types: AtomicU32::new(held_types),
            out_of_space: AtomicBool::new(false),
            dead_ratio: AtomicU32::new(f32::NAN.to_bits()),
        }))
    }

    /// A record of the type with bit `mask` was appended.
    pub(super) fn note_held(&self, mask: u32) {
        // Every append lands here, and nearly every one is of a type already
        // held: the load keeps those off the cache line a reader wants.
        if self.0.held_types.load(Ordering::Relaxed) & mask != mask {
            self.0.held_types.fetch_or(mask, Ordering::Relaxed);
        }
    }

    /// Whether a record of the type with bit `mask` has been held since open.
    pub(super) fn holds(&self, mask: u32) -> bool {
        self.0.held_types.load(Ordering::Relaxed) & mask == mask
    }

    /// Every record type the substrate has held since it opened — the open
    /// walk's, and each appended since — by tag, ascending.
    pub fn held_record_tags(&self) -> Vec<u8> {
        let held = self.0.held_types.load(Ordering::Relaxed);
        (0..32u8).filter(|t| held & (1 << t) != 0).collect()
    }

    pub(super) fn set_out_of_space(&self, full: bool) {
        self.0.out_of_space.store(full, Ordering::Relaxed);
    }

    /// Whether a write has failed for lack of space since the last test write
    /// succeeded.
    pub fn out_of_space(&self) -> bool {
        self.0.out_of_space.load(Ordering::Relaxed)
    }

    pub(super) fn set_dead_ratio(&self, ratio: f32) {
        self.0.dead_ratio.store(ratio.to_bits(), Ordering::Relaxed);
    }

    /// The dead ratio the last liveness scan found; `None` before the first.
    pub fn dead_ratio(&self) -> Option<f32> {
        let ratio = f32::from_bits(self.0.dead_ratio.load(Ordering::Relaxed));
        (!ratio.is_nan()).then_some(ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Held types accumulate, and a clone reads the same cells.
    #[test]
    fn held_types_accumulate_and_read_by_tag() {
        let facts = LogFacts::new(0b101);
        let reader = facts.clone();
        assert_eq!(reader.held_record_tags(), [0, 2]);
        facts.note_held(1 << 31);
        facts.note_held(1 << 31);
        assert_eq!(reader.held_record_tags(), [0, 2, 31]);
        assert!(reader.holds(1 << 2));
        assert!(!reader.holds(1 << 1));
    }

    /// No scan yet reads as no ratio, not a ratio of nothing.
    #[test]
    fn a_dead_ratio_is_absent_until_a_scan_sets_it() {
        let facts = LogFacts::new(0);
        assert_eq!(facts.dead_ratio(), None);
        facts.set_dead_ratio(0.25);
        assert_eq!(facts.dead_ratio(), Some(0.25));
    }

    #[test]
    fn the_space_latch_sets_and_clears() {
        let facts = LogFacts::new(0);
        assert!(!facts.out_of_space());
        facts.set_out_of_space(true);
        assert!(facts.out_of_space());
        facts.set_out_of_space(false);
        assert!(!facts.out_of_space());
    }
}
