//! What a reprojection's belief scan teaches a working set
//! (`docs/zend_working_set.md` §4.3).

use std::collections::HashMap;

use super::ids::TimelineId;
use super::project::ProjectionTarget;
use super::schema::Schema;
use crate::working_set::WorkingSetConfig;

/// The fresh score of the best-ranked file: one full-strength hit.
const FULL_HIT: f32 = 1000.0;

/// How many files of a group, per probe, earn a fresh score.
pub const RANKED: usize = 8;

/// The working set the projection target's layer declares, if any.
pub fn working_set_config(schema: &Schema, target: ProjectionTarget) -> Option<&WorkingSetConfig> {
    schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .and_then(|l| l.working_set.as_ref())
}

/// One working-set group's scan, per probe: each conversation's best
/// normalized exchange score.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawBest {
    pub tail: Vec<(TimelineId, f32)>,
    pub question: Vec<(TimelineId, f32)>,
}

/// The fresh score each conversation feeds its momentum: by RANK in its own
/// group's scan, never by magnitude, and only where the turn's question and
/// its trailing window AGREE. A file both probes place in their top
/// [`RANKED`] gets [`FULL_HIT`] divided by the worse of its two ranks; a file
/// only one of them places gets nothing. A group scanned with no question
/// probe ranks on its trailing window alone.
///
/// A rank needs neither a learned level nor the group's size, so the band
/// cannot inflate: fed the normalized scores, momentum took a one-off match
/// at 643,768 — six hundred full hits — and an object-detection demo sat in
/// the working set of a conversation about a paper. Agreement, because the
/// trailing window alone is whatever the turn last wrote: through tool rounds
/// it is JSON listings, and it crowned test files the question was never
/// about (measured 2026-10-24, ~70K of test files beside a paper).
pub fn ranked_fresh(groups: &[RawBest]) -> HashMap<TimelineId, f32> {
    let mut fresh: HashMap<TimelineId, f32> = HashMap::new();
    for group in groups {
        let tail = ranks(&group.tail);
        if group.question.is_empty() {
            for (timeline, rank) in tail {
                fresh.insert(timeline, FULL_HIT / rank as f32);
            }
            continue;
        }
        let question = ranks(&group.question);
        for (timeline, rank) in tail {
            if let Some(&other) = question.get(&timeline) {
                fresh.insert(timeline, FULL_HIT / rank.max(other) as f32);
            }
        }
    }
    fresh
}

/// Each conversation's 1-based rank in `probe` by raw vote, down to
/// [`RANKED`]; one that scored nothing has none.
fn ranks(probe: &[(TimelineId, f32)]) -> HashMap<TimelineId, usize> {
    let mut ranked: Vec<(TimelineId, f32)> =
        probe.iter().copied().filter(|(_, s)| *s > 0.0).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.raw().cmp(&b.0.raw())));
    ranked
        .into_iter()
        .take(RANKED)
        .enumerate()
        .map(|(i, (timeline, _))| (timeline, i + 1))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// The best file gets one full hit and each rank below a share of it; a
    /// file that scored nothing gets nothing, whatever the magnitudes.
    #[test]
    fn fresh_scores_follow_rank_not_magnitude() {
        let group = RawBest {
            tail: vec![
                (tl(1), 9.0),
                (tl(2), 643_768.0),
                (tl(3), 0.0),
                (tl(4), 10.0),
            ],
            question: Vec::new(),
        };
        let fresh = ranked_fresh(&[group]);
        assert_eq!(fresh.len(), 3);
        assert_eq!(fresh[&tl(2)], 1000.0);
        assert_eq!(fresh[&tl(4)], 500.0);
        assert_eq!(fresh[&tl(1)], 1000.0 / 3.0);
    }

    /// Past [`RANKED`] a file earns nothing.
    #[test]
    fn only_the_top_ranks_earn_a_fresh_score() {
        let tail: Vec<(TimelineId, f32)> = (1..=12).map(|n| (tl(n), n as f32)).collect();
        let fresh = ranked_fresh(&[RawBest {
            tail,
            question: Vec::new(),
        }]);
        assert_eq!(fresh.len(), RANKED);
        assert_eq!(fresh[&tl(12)], 1000.0);
        assert!(!fresh.contains_key(&tl(4)));
    }

    /// **Only where both probes agree**: a file scores by the worse of its two
    /// ranks, and one only the trailing window places gets nothing. Each group
    /// ranks its own files, and one scanned without a question ranks on its
    /// tail alone.
    #[test]
    fn both_probes_must_place_a_file_and_it_scores_by_the_worse_rank() {
        let files = RawBest {
            tail: vec![(tl(1), 5.0), (tl(2), 9.0), (tl(3), 4.0)],
            question: vec![(tl(1), 7.0), (tl(2), 3.0)],
        };
        let folders = RawBest {
            tail: vec![(tl(50), 0.5)],
            question: Vec::new(),
        };
        let fresh = ranked_fresh(&[files, folders]);
        assert_eq!(fresh[&tl(1)], 500.0, "tail rank 2, question rank 1");
        assert_eq!(fresh[&tl(2)], 500.0, "tail rank 1, question rank 2");
        assert!(!fresh.contains_key(&tl(3)), "the question never placed it");
        assert_eq!(fresh[&tl(50)], 1000.0, "no question probe: the tail alone");
    }
}
