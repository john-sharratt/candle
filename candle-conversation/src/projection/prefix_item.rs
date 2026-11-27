//! One place in a prompt a section is sealed behind: a sealed section, or a
//! hole of empty positions (`docs/placed_section_layout.md` §3.1, §3.1b).
//!
//! A section sealed after a reserved range is computed behind the range left
//! empty, so its prefix names the positions the range takes as well as the
//! sections before it. The same place in a prompt is a [`ProjectionSegment`]
//! when a projection emits it and a [`PrefixItem`] when a section is sealed
//! behind it.
//!
//! [`ProjectionSegment`]: super::ProjectionSegment

use super::ids::SectionId;

/// One place in a prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefixItem {
    /// A sealed section, borrowed from the substrate.
    Section(SectionId),
    /// Positions left empty: no token, no K/V, no index row.
    Hole(usize),
}

impl PrefixItem {
    /// The section this place names, if it is one.
    pub fn section(self) -> Option<SectionId> {
        match self {
            PrefixItem::Section(id) => Some(id),
            PrefixItem::Hole(_) => None,
        }
    }
}

/// A run of a prompt's tokens and the hole ahead of it: what a forward over a
/// prefix with holes runs, a run at a time, each past its hole.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenRun {
    /// Positions left empty before the run.
    pub hole: usize,
    pub tokens: Vec<u32>,
}

/// Whether `runs` hold no token, whatever holes they leave.
pub fn runs_are_empty(runs: &[TokenRun]) -> bool {
    runs.iter().all(|r| r.tokens.is_empty())
}

/// The sections `items` names, in order, holes left out.
pub fn prefix_sections(items: &[PrefixItem]) -> Vec<SectionId> {
    items.iter().filter_map(|i| i.section()).collect()
}

/// `ids` as a prefix of sections with no hole between them.
pub fn sections_prefix(ids: &[SectionId]) -> Vec<PrefixItem> {
    ids.iter().copied().map(PrefixItem::Section).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A prefix names its sections in order, holes left out.**
    #[test]
    fn a_prefix_names_its_sections_in_order() {
        let (a, b) = (SectionId::new(3), SectionId::new(5));
        let items = [
            PrefixItem::Section(a),
            PrefixItem::Hole(100),
            PrefixItem::Section(b),
        ];
        assert_eq!(prefix_sections(&items), vec![a, b]);
        assert_eq!(
            sections_prefix(&[a, b]),
            vec![PrefixItem::Section(a), PrefixItem::Section(b)]
        );
        assert_eq!(PrefixItem::Hole(7).section(), None);
    }
}
