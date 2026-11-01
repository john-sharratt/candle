//! Holes in this member's own records that no member holds — and so are gone
//! (§7.8).
//!
//! A gap in a member's own origin is usually a record a purge dropped here
//! that another member still holds, and a catch-up fills it back. One no member
//! holds is never filled: every member's coverage of the origin stops at it, so
//! a member joining is held out of service until its coverage reaches what the
//! origin wrote, which it never does. WhiteWorld's zend waited on BlackWorld's
//! own `oseq` 13,240,133 on 2026-11-01, with the gap watch fetching it again
//! and again from the one member that lacked it too.
//!
//! The evidence is already in hand. Every `CATCHUP` carries its requester's
//! whole applied set, so the member answering learns, of each of its own gaps,
//! whether that requester holds it. A gap every other member has been seen to
//! lack is declared gone (`Store::gone`): it is applied here, leaves the
//! origin's gaps, and is listed gone to every member that asks — whose
//! coverage then moves past it. A requester seen to hold any of a gap takes the
//! evidence back, since that member can give it back. A member away while this
//! is decided is not counted: one that still held the record and comes back
//! later finds it gone here, which is the cost of not waiting for it for ever.

use std::collections::{BTreeMap, BTreeSet};

use applied_set::OriginApplied;

/// The most own gaps followed at once; the rest wait until these settle.
pub const TRACKED: usize = 1024;

/// The longest gap declared gone, in `oseq`s. A wider one is more than a lost
/// write, and is left for an operator.
pub const WIDEST: u64 = 4096;

/// Who has been seen to lack each of this member's own gaps.
#[derive(Debug, Default)]
pub struct OwnGaps {
    lacked_by: BTreeMap<(u64, u64), BTreeSet<u64>>,
}

impl OwnGaps {
    /// Requester `who` asked holding `theirs` of this member's origin, while
    /// this member holds `mine` of it and has written through `written`. Each
    /// gap of `mine` below `written`, no wider than [`WIDEST`], is noted as
    /// lacked by `who` when it holds none of it — and forgotten when it holds
    /// any, since it can give that back.
    pub fn saw(
        &mut self,
        who: u64,
        theirs: Option<&OriginApplied>,
        mine: &OriginApplied,
        written: u64,
    ) {
        let gaps: Vec<(u64, u64)> = mine
            .gaps()
            .into_iter()
            // `s..=e` holds `e - s + 1` oseqs: at most `WIDEST`.
            .filter(|(s, e)| *e < written && e - s < WIDEST)
            .collect();
        self.lacked_by.retain(|gap, _| gaps.contains(gap));
        for gap @ (s, e) in gaps {
            let holds = theirs.is_some_and(|t| (s..=e).any(|o| t.contains(o)));
            match holds {
                true => {
                    self.lacked_by.remove(&gap);
                }
                false => {
                    if self.lacked_by.len() >= TRACKED && !self.lacked_by.contains_key(&gap) {
                        continue;
                    }
                    self.lacked_by.entry(gap).or_default().insert(who);
                }
            }
        }
    }

    /// The gaps every one of `others` — the other members now — has been seen
    /// to lack, taken out of what is followed. None while there are no others:
    /// a member alone has nobody to ask.
    pub fn gone(&mut self, others: &BTreeSet<u64>) -> Vec<(u64, u64)> {
        if others.is_empty() {
            return Vec::new();
        }
        let due: Vec<(u64, u64)> = self
            .lacked_by
            .iter()
            .filter(|(_, lacked)| others.is_subset(lacked))
            .map(|(gap, _)| *gap)
            .collect();
        for gap in &due {
            self.lacked_by.remove(gap);
        }
        due
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Coverage through `through`, with `holes` missing above it, up to `top`.
    fn coverage(top: u64, holes: &[u64]) -> OriginApplied {
        let mut a = OriginApplied::default();
        for o in 1..=top {
            if !holes.contains(&o) {
                a.insert(o);
            }
        }
        a
    }

    fn members(nodes: &[u64]) -> BTreeSet<u64> {
        nodes.iter().copied().collect()
    }

    /// **A gap every other member lacks is gone; one any member holds is
    /// not.**
    #[test]
    fn a_gap_every_other_member_lacks_is_gone() {
        let mine = coverage(20, &[5]);
        let mut g = OwnGaps::default();
        g.saw(2, Some(&coverage(20, &[5])), &mine, 20);
        assert!(g.gone(&members(&[2, 3])).is_empty(), "3 not seen yet");
        g.saw(3, None, &mine, 20);
        assert_eq!(g.gone(&members(&[2, 3])), vec![(5, 5)]);
        assert!(g.gone(&members(&[2, 3])).is_empty(), "declared once");

        let mut held = OwnGaps::default();
        held.saw(2, Some(&coverage(20, &[5])), &mine, 20);
        held.saw(3, Some(&coverage(20, &[])), &mine, 20);
        assert!(
            held.gone(&members(&[2, 3])).is_empty(),
            "3 can give it back"
        );
    }

    /// **A requester that holds the gap takes the evidence back**, and so
    /// does the gap closing here.
    #[test]
    fn holding_it_or_closing_it_forgets_the_evidence() {
        let mut g = OwnGaps::default();
        g.saw(2, Some(&coverage(20, &[5])), &coverage(20, &[5]), 20);
        g.saw(2, Some(&coverage(20, &[])), &coverage(20, &[5]), 20);
        assert!(g.gone(&members(&[2])).is_empty());

        let mut closed = OwnGaps::default();
        closed.saw(2, Some(&coverage(20, &[5])), &coverage(20, &[5]), 20);
        closed.saw(3, None, &coverage(20, &[]), 20);
        assert!(closed.gone(&members(&[2])).is_empty(), "filled here since");
    }

    /// **Only what was written, and only a narrow gap**: a gap at or above
    /// what this member has written may be a write still on its way, and a
    /// gap wider than [`WIDEST`] is not a lost write.
    #[test]
    fn only_a_narrow_gap_below_what_was_written() {
        let mut g = OwnGaps::default();
        g.saw(2, None, &coverage(20, &[19]), 19);
        assert!(
            g.gone(&members(&[2])).is_empty(),
            "not below what was written"
        );

        let wide: Vec<u64> = (5..5 + WIDEST + 1).collect();
        let mut w = OwnGaps::default();
        w.saw(2, None, &coverage(WIDEST + 10, &wide), WIDEST + 10);
        assert!(w.gone(&members(&[2])).is_empty(), "too wide");
    }

    /// A member alone declares nothing gone.
    #[test]
    fn a_member_alone_declares_nothing() {
        let mut g = OwnGaps::default();
        g.saw(2, None, &coverage(20, &[5]), 20);
        assert!(g.gone(&BTreeSet::new()).is_empty());
    }
}
