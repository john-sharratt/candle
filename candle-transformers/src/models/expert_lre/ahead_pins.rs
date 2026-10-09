//! The pad pins read-ahead holds.
//!
//! The device reads an expert ahead only from the slot image the host vetted
//! for it (`moe_bucketize.cu`, READ-AHEAD). A warm slot needs nothing — it never
//! changes. A pad slot does: the stager may evict and overwrite it as soon as no
//! invocation of its own row can read it, and a read-ahead reads it from inside
//! another row's invocation. So a pad-backed expert is listed only while the
//! host holds its pad slot pinned (`Residency::pin_pad`, which the stager never
//! evicts), and the pin outlives the listing until every invocation that could
//! have read the listing — any row's, begun before the list was rewritten — has
//! finished (`ReclaimClock::readers_key`).
//!
//! A row's listing is replaced in two steps around the ring write:
//! [`AheadPins::plan`] chooses which pad-backed candidates are listed — those
//! the old listing already pinned keep their pin, others take a new one while
//! fewer than the cap are held — and [`AheadPins::commit`], after the write and
//! with the readers' key read after it, schedules the old listing's pins the
//! new one dropped. [`AheadPins::due`] releases the scheduled pins whose key
//! has passed. Holding the pad's pins to a share of it keeps the stager room
//! for every cold demand row.

/// One expert a row's listing could name: its slot image and whether that
/// image is a pad slot (which needs a pin).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub(crate) expert: usize,
    pub(crate) image: u64,
    pub(crate) pad: bool,
}

/// A row's planned listing: what the ring is given, and the experts whose pad
/// slots the caller must pin before writing it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) listing: Vec<(usize, u64)>,
    pub(crate) pin: Vec<usize>,
}

pub(crate) struct AheadPins {
    /// Per row, the pad-backed experts its current listing holds pinned.
    listed: Vec<Vec<usize>>,
    /// The listing [`Self::plan`] chose and [`Self::commit`] installs.
    staged: Option<(usize, Vec<usize>)>,
    /// Pins the listings dropped: `(key, row, expert)`, released once `key`
    /// has passed.
    pending: Vec<(u64, usize, usize)>,
    /// Pins held, listed or pending.
    held: usize,
}

impl AheadPins {
    pub(crate) fn new(rows: usize) -> Self {
        Self {
            listed: vec![Vec::new(); rows],
            staged: None,
            pending: Vec::new(),
            held: 0,
        }
    }

    pub(crate) fn held(&self) -> usize {
        self.held
    }

    /// Whether `row`'s current listing holds `expert`'s pad slot pinned — a
    /// read-ahead claim of the expert copied from the pad.
    pub(crate) fn lists(&self, row: usize, expert: usize) -> bool {
        self.listed[row].contains(&expert)
    }

    /// Choose `row`'s listing from `candidates`, best first. A warm candidate
    /// is listed as it is; a pad-backed one keeps the pin the current listing
    /// holds for it, or takes a new one while fewer than `cap` are held, and is
    /// left out otherwise.
    pub(crate) fn plan(&mut self, row: usize, candidates: &[Candidate], cap: usize) -> Plan {
        let mut listing = Vec::with_capacity(candidates.len());
        let mut pinned = Vec::new();
        let mut pin = Vec::new();
        for c in candidates {
            if c.pad {
                if self.listed[row].contains(&c.expert) {
                    pinned.push(c.expert);
                } else if self.held < cap {
                    self.held += 1;
                    pinned.push(c.expert);
                    pin.push(c.expert);
                } else {
                    continue;
                }
            }
            listing.push((c.expert, c.image));
        }
        self.staged = Some((row, pinned));
        Plan { listing, pin }
    }

    /// The planned listing is in the ring, and `key` is the readers' key read
    /// after it: install it, and schedule the pins the old listing held and
    /// the new one dropped for release once `key` has passed.
    pub(crate) fn commit(&mut self, key: u64) {
        let Some((row, pinned)) = self.staged.take() else {
            return;
        };
        let old = std::mem::replace(&mut self.listed[row], pinned);
        for e in old {
            if !self.listed[row].contains(&e) {
                self.pending.push((key, row, e));
            }
        }
    }

    /// The dropped pins whose readers have finished (`passed(key)`), to be
    /// released now.
    pub(crate) fn due(&mut self, passed: impl Fn(u64) -> bool) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        self.pending.retain(|&(key, row, e)| {
            let done = passed(key);
            if done {
                out.push((row, e));
            }
            !done
        });
        self.held -= out.len();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warm(expert: usize) -> Candidate {
        Candidate {
            expert,
            image: 0x7000_0000 + expert as u64 * 0x1000,
            pad: false,
        }
    }

    fn pad(expert: usize) -> Candidate {
        Candidate {
            expert,
            image: 0x9000_0000 + expert as u64 * 0x1000,
            pad: true,
        }
    }

    /// Warm candidates list as they are; pad ones take a pin each while under
    /// the cap, and past it are left out, the order otherwise kept.
    #[test]
    fn pad_candidates_are_pinned_under_the_cap_and_dropped_past_it() {
        let mut p = AheadPins::new(3);
        let plan = p.plan(1, &[pad(4), warm(2), pad(9), pad(5)], 2);
        assert_eq!(
            plan,
            Plan {
                listing: vec![(4, 0x9000_4000), (2, 0x7000_2000), (9, 0x9000_9000)],
                pin: vec![4, 9],
            }
        );
        assert_eq!(p.held(), 2);
        p.commit(10);
        assert!(p.due(|_| true).is_empty(), "nothing dropped yet");
    }

    /// A relisting keeps the pins it lists again — no second pin — and
    /// schedules the ones it dropped for release once their readers' key has
    /// passed, never before.
    #[test]
    fn a_relisting_carries_kept_pins_and_releases_dropped_ones_after_their_readers() {
        let mut p = AheadPins::new(3);
        p.plan(2, &[pad(4), pad(9)], 8);
        p.commit(10);
        let plan = p.plan(2, &[pad(9), pad(6), warm(1)], 8);
        assert_eq!(plan.pin, vec![6], "9 is carried, only 6 is new");
        assert_eq!(p.held(), 3, "4 is still held until its readers finish");
        p.commit(17);
        assert!(p.due(|k| k < 17).is_empty(), "key 17 not passed");
        assert_eq!(p.due(|k| k <= 17), vec![(2, 4)]);
        assert_eq!(p.held(), 2);
        assert!(p.due(|_| true).is_empty(), "released once");
    }

    /// Rows are independent: relisting one drops nothing from another, and a
    /// row's dropped pins wait on its own key.
    #[test]
    fn rows_keep_their_own_pins() {
        let mut p = AheadPins::new(3);
        p.plan(0, &[pad(1)], 8);
        p.commit(5);
        p.plan(1, &[pad(1)], 8);
        p.commit(6);
        assert_eq!(p.held(), 2, "the same expert of two rows is two slots");
        p.plan(0, &[], 8);
        p.commit(7);
        assert_eq!(p.due(|k| k <= 7), vec![(0, 1)]);
        assert_eq!(p.held(), 1);
    }

    /// A row lists a pad expert from its commit until a relisting drops it,
    /// whether or not the dropped pin is still pending; a warm expert, needing
    /// no pin, is never held.
    #[test]
    fn a_pad_expert_is_listed_from_its_commit_until_dropped() {
        let mut p = AheadPins::new(2);
        p.plan(1, &[pad(4), warm(2)], 8);
        assert!(!p.lists(1, 4), "not before the commit");
        p.commit(3);
        assert!(p.lists(1, 4));
        assert!(!p.lists(1, 2), "warm needs no pin");
        assert!(!p.lists(0, 4), "another row");
        p.plan(1, &[warm(2)], 8);
        p.commit(4);
        assert!(!p.lists(1, 4), "dropped, its pin pending");
        assert_eq!(p.held(), 1);
    }

    /// Pins scheduled for release still count against the cap until released.
    #[test]
    fn a_pending_release_still_counts_against_the_cap() {
        let mut p = AheadPins::new(2);
        p.plan(0, &[pad(1), pad(2)], 2);
        p.commit(3);
        p.plan(0, &[], 2);
        p.commit(4);
        let plan = p.plan(1, &[pad(7)], 2);
        assert!(
            plan.listing.is_empty() && plan.pin.is_empty(),
            "two still held"
        );
        p.commit(5);
        assert_eq!(p.due(|k| k <= 4).len(), 2);
        assert_eq!(p.plan(1, &[pad(7)], 2).pin, vec![7]);
    }
}
