//! What a reprojection's belief scan teaches a working set
//! (`docs/zend_working_set.md` §4.3).

use std::collections::HashMap;

use super::ids::{GroupId, TimelineId, TurnKey};
use super::project::ProjectionTarget;
use super::schema::Schema;
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
/// of `schema` only — the `fresh` a file's momentum is fed.
pub fn working_set_fresh(
    schema: &Schema,
    candidates: &[(GroupId, Vec<(TurnKey, f32)>)],
) -> HashMap<TimelineId, f32> {
    best_per_conversation(candidates, |group| {
        schema
            .layers
            .iter()
            .flat_map(|l| l.groups.iter())
            .any(|g| g.id == group && g.is_working_set())
    })
}

/// The best score per conversation across `candidates`' groups that `counts`.
fn best_per_conversation(
    candidates: &[(GroupId, Vec<(TurnKey, f32)>)],
    counts: impl Fn(GroupId) -> bool,
) -> HashMap<TimelineId, f32> {
    let mut best: HashMap<TimelineId, f32> = HashMap::new();
    for (group, scored) in candidates {
        if !counts(*group) {
            continue;
        }
        for (key, score) in scored {
            let entry = best.entry(key.timeline).or_insert(f32::MIN);
            *entry = entry.max(*score);
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
        let best = best_per_conversation(&candidates, |g| g == files);
        assert_eq!(best.len(), 2);
        assert_eq!(best[&TimelineId::from_raw(7).unwrap()], 900.0);
        assert_eq!(best[&TimelineId::from_raw(8).unwrap()], 50.0);
    }
}
