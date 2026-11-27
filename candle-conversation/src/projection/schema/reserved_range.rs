//! A placed section's reserved range of positions
//! (`docs/placed_section_layout.md` §3.1, §3.7b).
//!
//! A collection with a range always takes the same positions whatever it
//! holds: what its members do not fill is a hole ahead of them, so everything
//! after the range is read at the same positions on every projection and in
//! the prompt the conversation was born with. The size is in the schema, and
//! a model whose native context is below [`SMALL_CONTEXT_BELOW`] takes the
//! smaller size it declares.

/// Native contexts below this take a range's small-context size: they cannot
/// spare the full one.
pub const SMALL_CONTEXT_BELOW: usize = 131_072;

/// A collection's reserved range, in positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReservedRange {
    /// Positions on a model whose native context is [`SMALL_CONTEXT_BELOW`] or
    /// more.
    pub tokens: usize,
    /// Positions on a model whose native context is smaller.
    pub small_context: usize,
    /// Which end of the range the members stand at.
    pub fill: RangeFill,
}

/// Which end of a range its members stand at (`docs/placed_section_layout.md`
/// §3.6).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RangeFill {
    /// At the top, nearest what follows the range, the hole ahead of them: a
    /// range ordered by relevance — the tools, the knowledge base. Each member
    /// is sealed ending at the top.
    #[default]
    Top,
    /// At the bottom, straight after what precedes the range, the hole after
    /// them: a range ordered by birth — the journal, whose sections are sealed
    /// as one chain from the range's start.
    Bottom,
}

impl ReservedRange {
    /// The positions this range takes on an engine trained to `native`
    /// positions. An engine that reports no window takes the full size.
    pub fn positions(&self, native: Option<usize>) -> usize {
        match native {
            Some(n) if n < SMALL_CONTEXT_BELOW => self.small_context,
            _ => self.tokens,
        }
    }

    /// The hole ahead of members that take `used` positions: the range less
    /// what they take, none when they take it all or more.
    pub fn hole(&self, native: Option<usize>, used: usize) -> usize {
        self.positions(native).saturating_sub(used)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JOURNAL: ReservedRange = ReservedRange {
        tokens: 8_192,
        small_context: 2_048,
        fill: RangeFill::Bottom,
    };

    /// **The native context picks the size.**
    #[test]
    fn the_native_context_picks_the_size() {
        assert_eq!(JOURNAL.positions(Some(262_144)), 8_192, "Flash-Next");
        assert_eq!(JOURNAL.positions(Some(131_072)), 8_192, "at the line");
        assert_eq!(JOURNAL.positions(Some(131_071)), 2_048, "below it");
        assert_eq!(JOURNAL.positions(Some(32_768)), 2_048);
        assert_eq!(JOURNAL.positions(None), 8_192, "no window reported");
    }

    /// **The hole is what the members leave.**
    #[test]
    fn the_hole_is_what_the_members_leave() {
        assert_eq!(JOURNAL.hole(Some(262_144), 0), 8_192);
        assert_eq!(JOURNAL.hole(Some(262_144), 1_000), 7_192);
        assert_eq!(JOURNAL.hole(Some(262_144), 8_192), 0);
        assert_eq!(JOURNAL.hole(Some(262_144), 9_000), 0);
        assert_eq!(JOURNAL.hole(Some(32_768), 1_000), 1_048);
    }
}
