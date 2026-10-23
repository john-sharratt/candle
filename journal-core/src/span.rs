//! The stretch of turns a draft is about, copied out so the draft runs with no
//! lock held.
//!
//! Each turn carries the support a claim citing it would have
//! ([`SpanTurn::evidence`]), decided by the caller from where the turn came
//! from — a tool result, somebody else's words, the writer's own act. A turn
//! that is no evidence at all (scenery, a standing instruction, the writer's
//! own reflection, the journal's own question) has none, and cannot be cited.

use crate::entry::Kind;

/// One turn of a stretch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanTurn {
    /// The number a claim cites it by.
    pub id: u64,
    pub at_ms: u64,
    /// What a claim citing it rests on; `None` when it may not be cited.
    pub evidence: Option<Kind>,
    /// The turn as the writer is shown it, speaker included.
    pub text: String,
}

/// A stretch of turns not yet written about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    pub from_turn: u64,
    pub to_turn: u64,
    pub from_ms: u64,
    pub to_ms: u64,
    pub turns: Vec<SpanTurn>,
}

impl Span {
    /// The stretch made of `turns`, oldest first. `None` when there are none.
    pub fn of(turns: Vec<SpanTurn>) -> Option<Self> {
        let (first, last) = (turns.first()?, turns.last()?);
        Some(Self {
            from_turn: first.id,
            to_turn: last.id,
            from_ms: first.at_ms,
            to_ms: last.at_ms,
            turns,
        })
    }

    /// The ids a claim may cite.
    pub fn citable_ids(&self) -> Vec<u64> {
        self.turns
            .iter()
            .filter(|t| t.evidence.is_some())
            .map(|t| t.id)
            .collect()
    }

    pub fn turn(&self, id: u64) -> Option<&SpanTurn> {
        self.turns.iter().find(|t| t.id == id)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    pub fn turn(id: u64, at_ms: u64, evidence: Option<Kind>, text: &str) -> SpanTurn {
        SpanTurn {
            id,
            at_ms,
            evidence,
            text: text.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::turn;
    use super::*;

    #[test]
    fn a_span_runs_from_its_first_turn_to_its_last() {
        let s = Span::of(vec![
            turn(4, 40, Some(Kind::Heard), "a"),
            turn(9, 90, None, "b"),
        ])
        .unwrap();
        assert_eq!((s.from_turn, s.to_turn, s.from_ms, s.to_ms), (4, 9, 40, 90));
    }

    #[test]
    fn no_turns_is_no_span() {
        assert_eq!(Span::of(Vec::new()), None);
    }

    #[test]
    fn only_a_turn_with_evidence_may_be_cited() {
        let s = Span::of(vec![
            turn(1, 1, Some(Kind::Did), "a"),
            turn(2, 2, None, "scenery"),
            turn(3, 3, Some(Kind::Observed), "c"),
        ])
        .unwrap();
        assert_eq!(s.citable_ids(), vec![1, 3]);
        assert!(s.turn(2).is_some() && s.turn(7).is_none());
    }
}
