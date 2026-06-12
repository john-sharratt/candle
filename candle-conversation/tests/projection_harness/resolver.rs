//! [`HarnessResolver`]: minimal [`ContentResolver`] backed by BDP section scores.
//!
//! All turn-related methods return zero — the harness only exercises the
//! system-prompt collection selection path (tool sections in the `tools`
//! collection).  Section scores are set from the output of a
//! `BdpScanner::scan_sections` call in [`super::harness::Harness::scan`].

use std::collections::HashMap;

use candle_conversation::projection::{
    ContentResolver, DepthWeights, GroupId, PerDepthScores, ScoreFormula, SectionId, TurnIndex,
};

pub struct HarnessResolver {
    pub section_scores: HashMap<SectionId, PerDepthScores>,
}

impl HarnessResolver {
    pub fn new() -> Self {
        Self {
            section_scores: HashMap::new(),
        }
    }
}

impl ContentResolver for HarnessResolver {
    fn turn_count(&self, _group: GroupId) -> u32 {
        0
    }
    fn turn_token_count(&self, _group: GroupId, _index: TurnIndex) -> usize {
        0
    }
    fn turn_score(
        &self,
        _group: GroupId,
        _index: TurnIndex,
        _formula: ScoreFormula,
        _weights: &DepthWeights,
    ) -> f32 {
        0.0
    }

    fn section_token_count(&self, _section: SectionId) -> usize {
        // Nominal non-zero token count so the reconciler doesn't skip sections.
        100
    }

    fn section_score(
        &self,
        section: SectionId,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        self.section_scores
            .get(&section)
            .map(|s| {
                weights.combine(
                    s.syn.pick(formula),
                    s.sem.pick(formula),
                    s.prag.pick(formula),
                )
            })
            .unwrap_or(0.0)
    }
}
