//! One place in a reserved range: a section, or the hole one left.

use crate::files::key::SectionKey;

/// A section placed in a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub key: SectionKey,
    /// The positions it takes: its tokens.
    pub tokens: usize,
    /// A mandatory section (`files::mandatory`): in scope from the opening,
    /// it never leaves and never slides out.
    pub mandatory: bool,
    /// Consecutive steps its hit rate has stayed below the eviction bar.
    pub below: u32,
}

/// One place in a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    Section(Entry),
    /// Positions a section that left holds empty until the range is
    /// compacted.
    Hole(usize),
}

impl Place {
    /// The positions the place takes.
    pub fn positions(&self) -> usize {
        match self {
            Place::Section(e) => e.tokens,
            Place::Hole(n) => *n,
        }
    }

    /// The section placed here, if any.
    pub fn section(&self) -> Option<&Entry> {
        match self {
            Place::Section(e) => Some(e),
            Place::Hole(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_place_takes_its_sections_tokens_or_its_holes_positions() {
        let e = Entry {
            key: SectionKey(1),
            tokens: 40,
            mandatory: false,
            below: 0,
        };
        assert_eq!(Place::Section(e).positions(), 40);
        assert_eq!(Place::Hole(12).positions(), 12);
        assert_eq!(Place::Section(e).section(), Some(&e));
        assert_eq!(Place::Hole(12).section(), None);
    }
}
