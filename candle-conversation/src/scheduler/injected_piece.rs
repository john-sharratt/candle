//! The name a sealed piece goes by in the projection assembler's logs.
//!
//! Formatted only when a line is actually written: a projection injects dozens
//! of pieces per rebuild and names one only when something about it is wrong.

use std::fmt;

use crate::projection::{SectionId, TimelineId, TurnIndex};

/// A sealed piece injected onto a slot, by its kind and stored identity.
#[derive(Debug, Clone, Copy)]
pub(super) enum InjectedPiece {
    Section(SectionId),
    Turn(TimelineId, TurnIndex),
    TurnHalf(TimelineId, TurnIndex),
    /// Glue K/V this slot forwarded once and re-injects from its cache.
    CachedGlue {
        tokens: usize,
    },
    /// The in-flight user message, captured on an earlier apply.
    CachedUserMessage {
        tokens: usize,
    },
}

impl fmt::Display for InjectedPiece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Section(id) => write!(f, "section {}", id.raw()),
            Self::Turn(t, i) => write!(f, "turn {}:{}", t.raw(), i.0),
            Self::TurnHalf(t, i) => write!(f, "turn half {}:{}", t.raw(), i.0),
            Self::CachedGlue { tokens } => write!(f, "cached glue island[{tokens}]"),
            Self::CachedUserMessage { tokens } => write!(f, "cached user message[{tokens}]"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_piece_names_its_kind_and_identity() {
        let t = TimelineId::from_raw(7).unwrap();
        assert_eq!(
            InjectedPiece::Section(SectionId::new(3)).to_string(),
            "section 3"
        );
        assert_eq!(InjectedPiece::Turn(t, TurnIndex(2)).to_string(), "turn 7:2");
        assert_eq!(
            InjectedPiece::TurnHalf(t, TurnIndex(0)).to_string(),
            "turn half 7:0"
        );
        assert_eq!(
            InjectedPiece::CachedGlue { tokens: 12 }.to_string(),
            "cached glue island[12]"
        );
        assert_eq!(
            InjectedPiece::CachedUserMessage { tokens: 40 }.to_string(),
            "cached user message[40]"
        );
    }
}
