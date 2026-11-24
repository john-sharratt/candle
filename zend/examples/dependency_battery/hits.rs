//! R1's retrieval hit rate: how often the files BM25 packed into its prompt
//! held an answer file — the baseline's counterpart of the memory's answer
//! file placed at the opening (§6.5). A question's hit is the mean over its
//! answered runs (its ranking is the same each run, so 0 or 1); the rate is
//! the mean over questions, and hit@1 the same for the top-ranked file alone.

use std::collections::BTreeMap;

use crate::arms::Arm;
use crate::run::Line;
use crate::stats::mean;

/// R1's row.
#[derive(Debug, Clone, PartialEq)]
pub struct HitRow {
    pub arm: Arm,
    pub questions: usize,
    pub answers: usize,
    /// The share of questions with an answer file among the packed files.
    pub hit: f64,
    /// The share whose top-ranked file is an answer file.
    pub hit_at_1: f64,
    /// The mean files packed.
    pub files: f64,
    /// The share of packed files cut at the budget.
    pub cut: f64,
}

/// `arm`'s hit rate over its answered lines that recorded a retrieval; `None`
/// when none did.
pub fn retrieval_hits(lines: &[Line], arm: Arm) -> Option<HitRow> {
    let ls: Vec<&Line> = lines
        .iter()
        .filter(|l| l.error.is_none() && l.arm == arm.id() && l.retrieval_hit.is_some())
        .collect();
    if ls.is_empty() {
        return None;
    }
    let mut by_q: BTreeMap<&str, Vec<&Line>> = BTreeMap::new();
    for l in &ls {
        by_q.entry(l.id.as_str()).or_default().push(l);
    }
    let per_q = |f: &dyn Fn(&Line) -> bool| -> f64 {
        mean(
            &by_q
                .values()
                .map(|runs| mean(&runs.iter().map(|l| f(l) as u8 as f64).collect::<Vec<_>>()))
                .collect::<Vec<_>>(),
        )
    };
    let packed: Vec<bool> = ls
        .iter()
        .flat_map(|l| l.retrieved.iter().map(|f| f.cut))
        .collect();
    Some(HitRow {
        arm,
        questions: by_q.len(),
        answers: ls.len(),
        hit: per_q(&|l| l.retrieval_hit == Some(true)),
        hit_at_1: per_q(&|l| l.retrieved.first().is_some_and(|f| f.answer_file)),
        files: mean(
            &ls.iter()
                .map(|l| l.retrieved.len() as f64)
                .collect::<Vec<_>>(),
        ),
        cut: mean(&packed.iter().map(|&c| c as u8 as f64).collect::<Vec<_>>()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::Placement;
    use crate::rag::context::RetrievedFile;

    fn file(rank: u32, answer_file: bool, cut: bool) -> RetrievedFile {
        RetrievedFile {
            path: format!("f{rank}.rs"),
            rank,
            score: 1.0,
            chars: 1,
            cut,
            answer_file,
        }
    }

    fn line(id: &str, run: u32, retrieved: Vec<RetrievedFile>) -> Line {
        Line {
            arm: "r1".into(),
            run,
            id: id.into(),
            category: "direct".into(),
            names_file: false,
            recall: 0.0,
            missed: Vec::new(),
            grounding: None,
            ungrounded: Vec::new(),
            completion_tokens: 0,
            think: None,
            placement: Placement::default(),
            secs: 0.0,
            answer: String::new(),
            error: None,
            retrieval_hit: Some(retrieved.iter().any(|f| f.answer_file)),
            retrieved,
            tool_calls: None,
        }
    }

    /// q1 hits at rank 1, q2 at rank 2, q3 not at all: hit 2/3, hit@1 1/3;
    /// q1's two runs count as one question.
    #[test]
    fn the_hit_rate_is_over_questions() {
        let lines = vec![
            line("q1", 1, vec![file(1, true, false), file(2, false, true)]),
            line("q1", 2, vec![file(1, true, false), file(2, false, true)]),
            line("q2", 1, vec![file(1, false, false), file(2, true, true)]),
            line("q3", 1, vec![file(1, false, true)]),
        ];
        let r = retrieval_hits(&lines, Arm::R1).unwrap();
        assert_eq!((r.questions, r.answers), (3, 4));
        assert_eq!(r.hit, 2.0 / 3.0);
        assert_eq!(r.hit_at_1, 1.0 / 3.0);
        assert_eq!(r.files, 7.0 / 4.0);
        assert_eq!(r.cut, 4.0 / 7.0);
        assert_eq!(retrieval_hits(&lines, Arm::C1), None);
    }
}
