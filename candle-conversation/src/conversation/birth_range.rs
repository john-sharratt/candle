//! Where a reserved range stands in a conversation's birth prefix
//! (`docs/placed_section_layout.md` §3.1, §3.1b).
//!
//! The birth prefix holds each range as a hole of its full size. A member of
//! the range is sealed behind what precedes that hole, across the part of the
//! range above it left empty, so it ends at the range's top.

use crate::projection::{CollectionId, PrefixItem, RangeFill};

/// A reserved range in the birth prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BirthRange {
    /// The collection the range is reserved for.
    pub collection: CollectionId,
    /// The hole's index in the birth prefix.
    pub at: usize,
    /// The positions the range takes on this conversation's engine.
    pub positions: usize,
    /// Which end of the range its members stand at.
    pub fill: RangeFill,
}

impl BirthRange {
    /// What a member is sealed behind when the members of its run up to and
    /// including it take `through` positions: everything the birth prefix holds
    /// before the range, then — in a range filled from the top — the part of
    /// the range above them left empty, so the run ends at the range's top. A
    /// range filled from the bottom starts its run straight after the prompt.
    pub(crate) fn member_prefix(&self, birth: &[PrefixItem], through: usize) -> Vec<PrefixItem> {
        let mut out = birth[..self.at.min(birth.len())].to_vec();
        if self.fill == RangeFill::Top {
            let hole = self.positions.saturating_sub(through);
            if hole > 0 {
                out.push(PrefixItem::Hole(hole));
            }
        }
        out
    }

    /// The range the members' run is sealed to end at the top of, when it is
    /// filled from the top.
    pub(crate) fn sealed_at_top(&self) -> Option<usize> {
        (self.fill == RangeFill::Top).then_some(self.positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::SectionId;

    /// **A member is sealed behind the prompt before its range and the part of
    /// the range above its run left empty.**
    #[test]
    fn a_member_ends_its_run_at_the_top_of_the_range() {
        let (a, b, c) = (SectionId::new(1), SectionId::new(2), SectionId::new(3));
        let birth = vec![
            PrefixItem::Section(a),
            PrefixItem::Hole(50),
            PrefixItem::Section(b),
            PrefixItem::Hole(1_000),
            PrefixItem::Section(c),
        ];
        let range = BirthRange {
            collection: CollectionId::for_test(9),
            at: 3,
            positions: 1_000,
            fill: RangeFill::Top,
        };
        assert_eq!(
            range.member_prefix(&birth, 300),
            vec![
                PrefixItem::Section(a),
                PrefixItem::Hole(50),
                PrefixItem::Section(b),
                PrefixItem::Hole(700),
            ]
        );
        // A run that fills the range leaves no hole above it.
        assert_eq!(
            range.member_prefix(&birth, 1_000),
            vec![
                PrefixItem::Section(a),
                PrefixItem::Hole(50),
                PrefixItem::Section(b),
            ]
        );
        // A range filled from the bottom starts its run after the prompt.
        let bottom = BirthRange {
            fill: RangeFill::Bottom,
            ..range
        };
        assert_eq!(
            bottom.member_prefix(&birth, 300),
            vec![
                PrefixItem::Section(a),
                PrefixItem::Hole(50),
                PrefixItem::Section(b),
            ]
        );
        assert_eq!(bottom.sealed_at_top(), None);
        assert_eq!(range.sealed_at_top(), Some(1_000));
    }
}
