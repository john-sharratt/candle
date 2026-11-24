//! Table 9.12's precision and cost by arm: the mean grounding of the answers
//! that cite anything (`score.rs`), the mean completion tokens, and the mean
//! tool calls of the answers whose calls were read back — beside recall, so
//! an arm that recalls more by saying more, or by inventing names, shows it.

use crate::arms::{Arm, ARMS};
use crate::run::Line;
use crate::stats::mean;

/// One arm's row.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundingRow {
    pub arm: Arm,
    /// Its answers that did not fail.
    pub answers: usize,
    /// Of those, the ones with a backticked span to ground.
    pub cited: usize,
    /// Their mean grounding.
    pub grounding: f64,
    pub completion_tokens: f64,
    /// The answers whose tool calls were read back.
    pub counted: usize,
    /// Their mean tool calls; NaN when none was.
    pub tool_calls: f64,
}

/// Every arm with an answer, in table order.
pub fn grounding_rows(lines: &[Line]) -> Vec<GroundingRow> {
    ARMS.iter()
        .filter_map(|&arm| {
            let ls: Vec<&Line> = lines
                .iter()
                .filter(|l| l.error.is_none() && l.arm == arm.id())
                .collect();
            if ls.is_empty() {
                return None;
            }
            let g: Vec<f64> = ls.iter().filter_map(|l| l.grounding).collect();
            let calls: Vec<f64> = ls
                .iter()
                .filter_map(|l| l.tool_calls.map(f64::from))
                .collect();
            Some(GroundingRow {
                arm,
                answers: ls.len(),
                cited: g.len(),
                grounding: mean(&g),
                completion_tokens: mean(
                    &ls.iter()
                        .map(|l| l.completion_tokens as f64)
                        .collect::<Vec<_>>(),
                ),
                counted: calls.len(),
                tool_calls: mean(&calls),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::Placement;

    fn line(arm: Arm, grounding: Option<f64>, tokens: u64, calls: Option<u32>) -> Line {
        Line {
            arm: arm.id().into(),
            run: 1,
            id: "q".into(),
            category: "direct".into(),
            names_file: false,
            recall: 0.0,
            missed: Vec::new(),
            grounding,
            ungrounded: Vec::new(),
            completion_tokens: tokens,
            think: None,
            placement: Placement::default(),
            secs: 0.0,
            answer: String::new(),
            error: None,
            retrieved: Vec::new(),
            retrieval_hit: None,
            tool_calls: calls,
        }
    }

    /// **Grounding is the mean over the answers that cite**, tokens over every
    /// answer, tool calls over the answers whose calls were read; a failed
    /// answer and an arm with none are left out.
    #[test]
    fn each_arm_is_its_answers_means() {
        let mut failed = line(Arm::C1, Some(0.0), 999, None);
        failed.error = Some("boom".into());
        let rows = grounding_rows(&[
            line(Arm::C1, Some(1.0), 100, None),
            line(Arm::C1, Some(0.5), 300, None),
            line(Arm::C1, None, 200, None),
            failed,
            line(Arm::T1, Some(1.0), 50, Some(4)),
            line(Arm::T1, None, 150, Some(1)),
        ]);
        assert_eq!(rows.len(), 2);
        let c1 = &rows[0];
        assert_eq!(
            (
                c1.arm,
                c1.answers,
                c1.cited,
                c1.grounding,
                c1.completion_tokens,
                c1.counted
            ),
            (Arm::C1, 3, 2, 0.75, 200.0, 0)
        );
        assert!(c1.tool_calls.is_nan());
        let t1 = &rows[1];
        assert_eq!(
            (t1.arm, t1.counted, t1.tool_calls, t1.completion_tokens),
            (Arm::T1, 2, 2.5, 100.0)
        );
    }
}
