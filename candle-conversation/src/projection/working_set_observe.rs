//! What a reprojection's belief scan teaches a working set
//! (`docs/zend_working_set.md` §4.3).

use std::collections::HashMap;

use super::ids::{GroupId, TimelineId, TurnKey};
use super::project::ProjectionTarget;
use super::schema::{Recency, Schema};
use crate::working_set::WorkingSetConfig;

/// The working set the projection target's layer declares, if any.
pub fn working_set_config(schema: &Schema, target: ProjectionTarget) -> Option<&WorkingSetConfig> {
    schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .and_then(|l| l.working_set.as_ref())
}

/// Each conversation's best fresh score this scan, over the working-set groups
/// of `schema` only — the `fresh` a file's momentum is fed. A group that ages
/// its units (`docs/guidance_layer.md` §4) scales each of its scores by
/// `aged(recency, conversation)` first.
pub fn working_set_fresh(
    schema: &Schema,
    candidates: &[(GroupId, Vec<(TurnKey, f32)>)],
    aged: &dyn Fn(&Recency, TimelineId) -> f32,
) -> HashMap<TimelineId, f32> {
    let groups: HashMap<GroupId, Option<&Recency>> = schema
        .layers
        .iter()
        .flat_map(|l| l.groups.iter())
        .filter(|g| g.is_working_set())
        .map(|g| (g.id, g.recency.as_ref()))
        .collect();
    best_per_conversation(candidates, |group, timeline| {
        let recency = groups.get(&group)?;
        Some(recency.map_or(1.0, |r| aged(r, timeline)))
    })
}

/// The best score per conversation across `candidates`' groups, each scaled
/// by `weight(group, conversation)`; a group it answers `None` for does not
/// count.
fn best_per_conversation(
    candidates: &[(GroupId, Vec<(TurnKey, f32)>)],
    weight: impl Fn(GroupId, TimelineId) -> Option<f32>,
) -> HashMap<TimelineId, f32> {
    let mut best: HashMap<TimelineId, f32> = HashMap::new();
    for (group, scored) in candidates {
        for (key, score) in scored {
            let Some(w) = weight(*group, key.timeline) else {
                continue;
            };
            let entry = best.entry(key.timeline).or_insert(f32::MIN);
            *entry = entry.max(*score * w);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::ids::TurnIndex;

    fn key(timeline: u64, index: u32) -> TurnKey {
        TurnKey::new(TimelineId::from_raw(timeline).unwrap(), TurnIndex(index))
    }

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// A file's fresh score is its best exchange, and only the counted groups
    /// feed it.
    #[test]
    fn the_best_exchange_per_conversation_in_counted_groups() {
        let files = GroupId::new(1);
        let other = GroupId::new(2);
        let candidates = vec![
            (
                files,
                vec![(key(7, 0), 300.0), (key(7, 2), 900.0), (key(8, 0), 50.0)],
            ),
            (other, vec![(key(9, 0), 5_000.0)]),
        ];
        let best = best_per_conversation(&candidates, |g, _| (g == files).then_some(1.0));
        assert_eq!(best.len(), 2);
        assert_eq!(best[&tl(7)], 900.0);
        assert_eq!(best[&tl(8)], 50.0);
    }

    /// **An aged unit's score is scaled before its best is taken**, and a
    /// unit its group does not age keeps its score.
    #[test]
    fn an_aged_units_score_is_scaled() {
        let guidance = GroupId::new(3);
        let candidates = vec![(
            guidance,
            vec![(key(7, 0), 800.0), (key(7, 1), 600.0), (key(8, 0), 400.0)],
        )];
        let best = best_per_conversation(&candidates, |_, t| {
            Some(if t == tl(7) { 0.25 } else { 1.0 })
        });
        assert_eq!(best[&tl(7)], 200.0);
        assert_eq!(best[&tl(8)], 400.0);
    }
}
