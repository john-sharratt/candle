//! Retrieval scoring for the probe layer — the oracle that decides whether any
//! of this worked.
//!
//! # The held-out design
//!
//! Scoring a probe that is *in* the corpus measures almost nothing: its own
//! signature is resident, so it self-matches, and a pass proves only that the
//! substrate is self-consistent. That check is still worth running as a
//! **necessary** condition — a probe that cannot retrieve its own folder even
//! with its own signature present is broken beyond argument, and one that
//! retrieves somebody *else's* folder from that position is actively harmful —
//! but it cannot be the quality measure.
//!
//! The quality measure needs a query the corpus has never seen. That is free
//! here, and it falls out of generating wide: each register produces
//! [`super::CANDIDATES_PER_REGISTER`] candidates and keeps
//! [`super::PROBES_PER_REGISTER`], so the next admissible candidates —
//! well-formed, gate-passing, simply out of slots — are **held out**: never
//! ingested, never distilled, kept only as test queries. A held-out question
//! about a folder is exactly the thing a real user would type, and whether it
//! retrieves that folder is exactly the question the layer exists to answer.
//!
//! This is the shape the tool work taught. A battery scored 3/6 when the truth
//! was 0/6 because its questions were answerable without retrieving anything; an
//! oracle has to demand something the model cannot supply unaided. A held-out
//! question naming a folder that must be found satisfies that, and needs no
//! human judgement to score.
//!
//! # What is measured
//!
//! Per register, because the registers make different promises and one can carry
//! the others. The systemic register in particular is the one a specificity-only
//! design would have dropped, and the one whose failure mode (collision across
//! folders) is invisible in an aggregate number.

use std::collections::BTreeMap;

use super::Register;

/// One scored query.
#[derive(Debug, Clone)]
pub struct Trial {
    /// The question put to the scan.
    pub query: String,
    /// The directory this question was written about.
    pub expected_dir: String,
    /// Which probe budget it came from.
    pub register: Register,
    /// Whether the query's own signature was in the corpus when it was scored.
    /// A held-out trial (`false`) is the quality measure; a resident one is the
    /// self-match necessary condition.
    pub resident: bool,
    /// Directories the scan selected, best first.
    pub ranked: Vec<String>,
}

impl Trial {
    /// 1-based rank of the expected directory, or `None` if it was not selected.
    pub fn rank(&self) -> Option<usize> {
        self.ranked
            .iter()
            .position(|d| dirs_match(d, &self.expected_dir))
            .map(|i| i + 1)
    }

    /// Whether the top-ranked directory is the right one.
    pub fn hit_at_1(&self) -> bool {
        self.rank() == Some(1)
    }

    /// Whether the expected directory is anywhere in the selection.
    pub fn hit_at_k(&self, k: usize) -> bool {
        self.rank().is_some_and(|r| r <= k)
    }

    /// Reciprocal rank, zero when unselected.
    pub fn reciprocal_rank(&self) -> f64 {
        self.rank().map_or(0.0, |r| 1.0 / r as f64)
    }

    /// The directory that beat the expected one, when the trial missed at rank 1.
    ///
    /// This is what turns an aggregate miss into a diagnosis: a systemic probe
    /// losing to its own parent directory is a collision, while one losing to an
    /// unrelated folder is a promiscuous question.
    pub fn usurper(&self) -> Option<&str> {
        match self.ranked.first() {
            Some(top) if !dirs_match(top, &self.expected_dir) => Some(top.as_str()),
            _ => None,
        }
    }
}

/// Directory equality, tolerant of the trailing-slash difference between a
/// `DirUnit::dir` (`zend/src/`) and a projection tile's label.
fn dirs_match(a: &str, b: &str) -> bool {
    a.trim_end_matches('/') == b.trim_end_matches('/')
}

/// Aggregate outcome over a set of trials.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Score {
    pub trials: usize,
    pub hit_at_1: usize,
    pub hit_at_3: usize,
    pub mrr_sum: f64,
}

impl Score {
    pub fn add(&mut self, trial: &Trial) {
        self.trials += 1;
        if trial.hit_at_1() {
            self.hit_at_1 += 1;
        }
        if trial.hit_at_k(3) {
            self.hit_at_3 += 1;
        }
        self.mrr_sum += trial.reciprocal_rank();
    }

    pub fn hit_at_1_pct(&self) -> f64 {
        pct(self.hit_at_1, self.trials)
    }

    pub fn hit_at_3_pct(&self) -> f64 {
        pct(self.hit_at_3, self.trials)
    }

    pub fn mrr(&self) -> f64 {
        if self.trials == 0 {
            0.0
        } else {
            self.mrr_sum / self.trials as f64
        }
    }
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

/// The full scoreboard: overall and per register, held-out and resident kept
/// apart because they answer different questions.
#[derive(Debug, Clone, Default)]
pub struct Scoreboard {
    pub held_out: Score,
    pub resident: Score,
    pub by_register: BTreeMap<&'static str, Score>,
    /// Held-out misses, worst first, for the diagnosis table.
    pub misses: Vec<Trial>,
}

impl Scoreboard {
    pub fn tally(trials: &[Trial]) -> Self {
        let mut board = Scoreboard::default();
        for trial in trials {
            if trial.resident {
                board.resident.add(trial);
                continue;
            }
            board.held_out.add(trial);
            board
                .by_register
                .entry(trial.register.id())
                .or_default()
                .add(trial);
            if !trial.hit_at_1() {
                board.misses.push(trial.clone());
            }
        }
        // Unselected before merely mis-ranked: a probe nothing retrieved is a
        // different defect from one that lost a close call, and the first is
        // what needs looking at.
        board.misses.sort_by(|a, b| {
            a.reciprocal_rank()
                .partial_cmp(&b.reciprocal_rank())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        board
    }

    /// Directories that stole first place most often, with counts.
    ///
    /// A single directory at the top of this list is the promiscuous-attractor
    /// signature — historically the workspace root, whose listing names every
    /// crate and so matched every question.
    pub fn top_usurpers(&self, n: usize) -> Vec<(String, usize)> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for miss in &self.misses {
            if let Some(u) = miss.usurper() {
                *counts.entry(u.to_string()).or_insert(0) += 1;
            }
        }
        let mut out: Vec<(String, usize)> = counts.into_iter().collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.truncate(n);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(expected: &str, ranked: &[&str], register: Register, resident: bool) -> Trial {
        Trial {
            query: "q?".to_string(),
            expected_dir: expected.to_string(),
            register,
            resident,
            ranked: ranked.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn rank_is_one_based_and_absent_when_unselected() {
        let t = trial("a/", &["b/", "a/", "c/"], Register::Locational, false);
        assert_eq!(t.rank(), Some(2));
        assert!(!t.hit_at_1());
        assert!(t.hit_at_k(3));
        assert_eq!(t.reciprocal_rank(), 0.5);

        let missed = trial("a/", &["b/", "c/"], Register::Locational, false);
        assert_eq!(missed.rank(), None);
        assert_eq!(missed.reciprocal_rank(), 0.0);
    }

    /// A unit's `dir` carries a trailing slash and a projection tile's label may
    /// not. Without tolerating that, every trial reads as a miss and the whole
    /// harness reports zero — a failure that looks like a catastrophic result
    /// rather than a formatting bug.
    #[test]
    fn a_trailing_slash_difference_is_not_a_miss() {
        let t = trial("zend/src/", &["zend/src"], Register::Conceptual, false);
        assert!(t.hit_at_1(), "{:?}", t.ranked);
    }

    /// The two populations answer different questions and must never be pooled:
    /// resident trials self-match and would inflate the quality number.
    #[test]
    fn resident_and_held_out_trials_are_scored_separately() {
        let trials = vec![
            trial("a/", &["a/"], Register::Locational, true),
            trial("a/", &["a/"], Register::Locational, true),
            trial("a/", &["b/"], Register::Locational, false),
        ];
        let board = Scoreboard::tally(&trials);
        assert_eq!(board.resident.trials, 2);
        assert_eq!(board.resident.hit_at_1, 2);
        assert_eq!(board.held_out.trials, 1);
        assert_eq!(board.held_out.hit_at_1, 0);
    }

    /// Per-register scoring is what makes the systemic register's collapse
    /// visible; an aggregate would hide it behind three healthy registers.
    #[test]
    fn per_register_scores_isolate_a_failing_register() {
        let mut trials = Vec::new();
        for _ in 0..6 {
            trials.push(trial("a/", &["a/"], Register::Locational, false));
            trials.push(trial("a/", &["b/"], Register::Systemic, false));
        }
        let board = Scoreboard::tally(&trials);
        assert_eq!(board.by_register["locational"].hit_at_1_pct(), 100.0);
        assert_eq!(board.by_register["systemic"].hit_at_1_pct(), 0.0);
        assert_eq!(board.held_out.hit_at_1_pct(), 50.0);
    }

    /// The promiscuous-attractor signature: one folder taking first place across
    /// unrelated queries.
    #[test]
    fn a_single_folder_stealing_every_slot_is_surfaced() {
        let mut trials = Vec::new();
        for dir in ["a/", "b/", "c/", "d/"] {
            trials.push(trial(dir, &["."], Register::Systemic, false));
        }
        let board = Scoreboard::tally(&trials);
        assert_eq!(board.top_usurpers(3), vec![(".".to_string(), 4)]);
    }

    /// Misses are ordered so the ones nothing retrieved come first — a different
    /// defect from a near miss, and the one worth reading.
    #[test]
    fn misses_are_ordered_worst_first() {
        let trials = vec![
            trial("a/", &["x/", "a/"], Register::Locational, false),
            trial("b/", &["x/", "y/", "z/"], Register::Locational, false),
        ];
        let board = Scoreboard::tally(&trials);
        assert_eq!(board.misses[0].expected_dir, "b/", "unselected first");
        assert_eq!(board.misses[1].expected_dir, "a/");
    }

    #[test]
    fn an_empty_scoreboard_reports_zero_rather_than_dividing_by_zero() {
        let board = Scoreboard::tally(&[]);
        assert_eq!(board.held_out.hit_at_1_pct(), 0.0);
        assert_eq!(board.held_out.mrr(), 0.0);
        assert!(board.top_usurpers(5).is_empty());
    }
}
