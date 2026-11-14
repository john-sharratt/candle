//! How many questions a unit is asked to write: one for every `per_tokens` of
//! its own sealed content, held to `min..=max`.
//!
//! ```yaml
//! questions: { per_tokens: 250, min: 8, max: 64 }
//! ```
//!
//! A file that defines forty things has forty things to be asked about by
//! name, and a ten-line module has a few; one count for both left the first
//! with most of its names unasked and filled the second with rephrasings. The
//! unit's sealed tokens are its reading — what the ingest actually read of it,
//! a file the read-round cap cut short counted only as far as it was read —
//! and the same on every member, because a unit is calibrated only once it
//! holds every turn (`run::units_of`).

use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::spec::MAX_QUESTIONS;

/// A layer's question count, per unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestionCount {
    /// One question for each this many of the unit's tokens.
    pub per_tokens: usize,
    /// The fewest any unit is asked for.
    pub min: usize,
    /// The most any unit is asked for, at most [`MAX_QUESTIONS`].
    pub max: usize,
}

impl QuestionCount {
    /// Why this count cannot be used, or `None` when it can.
    pub fn invalid(&self) -> Option<String> {
        if self.per_tokens == 0 {
            return Some("`per_tokens` must be at least 1".into());
        }
        if self.min == 0 || self.min > self.max {
            return Some(format!(
                "`min` must be 1..=`max` ({}), not {}",
                self.max, self.min
            ));
        }
        if self.max > MAX_QUESTIONS {
            return Some(format!(
                "`max` must be at most {MAX_QUESTIONS}, not {}",
                self.max
            ));
        }
        None
    }

    /// The questions a unit of `tokens` sealed tokens is asked for.
    pub fn for_tokens(&self, tokens: usize) -> usize {
        tokens.div_ceil(self.per_tokens).clamp(self.min, self.max)
    }

    /// Feed the rule into a calibration marker's digest: a changed rule
    /// calibrates every unit again, while a unit's own count follows from the
    /// rule and its content key alone.
    pub fn digest_into(&self, h: &mut Sha256) {
        h.update(self.per_tokens.to_le_bytes());
        h.update(self.min.to_le_bytes());
        h.update(self.max.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULE: QuestionCount = QuestionCount {
        per_tokens: 250,
        min: 8,
        max: 64,
    };

    /// **One question per 250 tokens, rounded up, between 8 and 64** — the
    /// BlackWorld code-reading layer's p10, median, p90 and largest units
    /// (663, 2,815, 11,568 and 303,030 tokens, 2026-11-15).
    #[test]
    fn a_unit_is_asked_one_question_per_its_tokens_within_the_bounds() {
        assert_eq!(RULE.for_tokens(0), 8);
        assert_eq!(RULE.for_tokens(663), 8);
        assert_eq!(RULE.for_tokens(2_000), 8);
        assert_eq!(RULE.for_tokens(2_001), 9);
        assert_eq!(RULE.for_tokens(2_815), 12);
        assert_eq!(RULE.for_tokens(11_568), 47);
        assert_eq!(RULE.for_tokens(16_000), 64);
        assert_eq!(RULE.for_tokens(303_030), 64);
    }

    /// A rule that divides by zero, asks for none, inverts its bounds or
    /// passes the hard cap is refused, and says which.
    #[test]
    fn a_rule_out_of_bounds_is_refused() {
        assert_eq!(RULE.invalid(), None);
        let with = |per_tokens, min, max| QuestionCount {
            per_tokens,
            min,
            max,
        };
        assert_eq!(
            with(0, 8, 64).invalid().as_deref(),
            Some("`per_tokens` must be at least 1")
        );
        assert_eq!(
            with(250, 0, 64).invalid().as_deref(),
            Some("`min` must be 1..=`max` (64), not 0")
        );
        assert_eq!(
            with(250, 65, 64).invalid().as_deref(),
            Some("`min` must be 1..=`max` (64), not 65")
        );
        assert_eq!(
            with(250, 8, MAX_QUESTIONS + 1).invalid().as_deref(),
            Some("`max` must be at most 64, not 65")
        );
    }

    /// Every field moves the digest.
    #[test]
    fn each_field_moves_the_digest() {
        let digest = |c: QuestionCount| {
            let mut h = Sha256::new();
            c.digest_into(&mut h);
            h.finalize().to_vec()
        };
        let base = digest(RULE);
        assert_ne!(
            digest(QuestionCount {
                per_tokens: 251,
                ..RULE
            }),
            base
        );
        assert_ne!(digest(QuestionCount { min: 9, ..RULE }), base);
        assert_ne!(digest(QuestionCount { max: 63, ..RULE }), base);
        assert_eq!(digest(RULE), base);
    }
}
