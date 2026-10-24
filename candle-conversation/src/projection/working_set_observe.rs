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

/// One working-set group's scan, per probe: each conversation's best raw vote.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawBest {
    pub tail: Vec<(TimelineId, f32)>,
    pub question: Vec<(TimelineId, f32)>,
}

/// The fresh score each conversation feeds its momentum: by RANK in its own
/// group's scan, never by magnitude. The best file of a probe gets
/// [`FULL_HIT`], the `r`-th `FULL_HIT / r`, down to rank [`RANKED`]; a file
/// placed by both probes keeps the better; one that scored nothing gets
/// nothing.
///
/// A rank needs neither a learned level nor the group's size, so the band
/// cannot inflate: fed the normalized scores, momentum took a one-off match
/// at 643,768 — six hundred full hits — and an object-detection demo sat in
/// the working set of a conversation about a paper.
pub fn ranked_fresh(groups: &[RawBest]) -> HashMap<TimelineId, f32> {
    let mut fresh: HashMap<TimelineId, f32> = HashMap::new();
    for group in groups {
        for probe in [&group.tail, &group.question] {
            let mut ranked: Vec<(TimelineId, f32)> =
                probe.iter().copied().filter(|(_, s)| *s > 0.0).collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.raw().cmp(&b.0.raw())));
            for (rank, (timeline, _)) in ranked.into_iter().take(RANKED).enumerate() {
                let score = FULL_HIT / (rank + 1) as f32;
                let entry = fresh.entry(timeline).or_insert(0.0);
                *entry = entry.max(score);
            }
        }
    }
    fresh
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

    /// A file placed by both probes keeps its better rank; each group ranks its
    /// own files, so a folder and a file can both lead.
    #[test]
    fn each_probe_and_each_group_rank_on_their_own() {
        let files = RawBest {
            tail: vec![(tl(1), 5.0), (tl(2), 9.0)],
            question: vec![(tl(1), 7.0), (tl(2), 3.0)],
        };
        let folders = RawBest {
            tail: vec![(tl(50), 0.5)],
            question: Vec::new(),
        };
        let fresh = ranked_fresh(&[files, folders]);
        assert_eq!(fresh[&tl(1)], 1000.0, "the question ranks it first");
        assert_eq!(fresh[&tl(2)], 1000.0);
        assert_eq!(fresh[&tl(50)], 1000.0);
    }
}
