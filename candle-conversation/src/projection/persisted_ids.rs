//! The ids a schema's layers and groups are persisted under.
//!
//! Every turn record carries its timeline's layer and group as bare numbers
//! (`persistence::streams::TurnDecl`), and a reload registers the timeline
//! under those numbers again. Numbered by position, removing a layer from the
//! middle of a schema renumbers every layer and group after it, and their
//! persisted conversations come back registered to the wrong layer or to none.
//! A schema that will lose layers pins its numbers with `persisted_id` on every
//! layer and every group; one that never pins is numbered by position.

use std::collections::HashSet;

use super::error::ConstructionError;

/// The first id that is not a layer's or group's: the system prompt's
/// name-scoping id sits here (`LayerId::system_prompt`), and the engine's
/// reserved ids above it.
const RESERVED_FROM: u32 = u32::MAX / 2;

/// The id each of `declared` is persisted under, in order: its own
/// `persisted_id` when every one carries one, its 1-based position when none
/// does. `kind` names them in an error.
///
/// Some pinned and some not is refused rather than numbered: a positional id
/// could land on a pinned one. So is a zero, a repeat, or an id in the
/// reserved band.
pub(super) fn assign(declared: &[Option<u32>], kind: &str) -> Result<Vec<u32>, ConstructionError> {
    let pinned = declared.iter().filter(|d| d.is_some()).count();
    if pinned == 0 {
        return Ok((1..=declared.len() as u32).collect());
    }
    if pinned != declared.len() {
        return Err(ConstructionError::PersistedIds(format!(
            "{pinned} of {} {kind}s carry a persisted_id — every {kind} must carry one, or none",
            declared.len()
        )));
    }
    let ids: Vec<u32> = declared.iter().flatten().copied().collect();
    let mut seen = HashSet::new();
    for &id in &ids {
        if id == 0 || id >= RESERVED_FROM {
            return Err(ConstructionError::PersistedIds(format!(
                "{kind} persisted_id {id} is outside 1..{RESERVED_FROM}"
            )));
        }
        if !seen.insert(id) {
            return Err(ConstructionError::PersistedIds(format!(
                "two {kind}s share persisted_id {id}"
            )));
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Unpinned, the ids are positions; pinned, they are the pins** — so a
    /// pinned schema can lose a layer from the middle and every other id
    /// stands.
    #[test]
    fn unpinned_ids_are_positions_and_pinned_ids_are_the_pins() {
        assert_eq!(assign(&[None, None, None], "layer").unwrap(), vec![1, 2, 3]);
        assert_eq!(
            assign(&[Some(1), Some(2), Some(4), Some(10)], "layer").unwrap(),
            vec![1, 2, 4, 10]
        );
        assert!(assign(&[], "layer").unwrap().is_empty());
    }

    /// Some pinned and some not, a repeat, a zero and the reserved band are
    /// each refused.
    #[test]
    fn unusable_pins_are_refused() {
        for declared in [
            vec![Some(1), None],
            vec![Some(3), Some(3)],
            vec![Some(0)],
            vec![Some(RESERVED_FROM)],
        ] {
            let err = assign(&declared, "group").unwrap_err();
            assert!(
                matches!(err, ConstructionError::PersistedIds(_)),
                "{declared:?}: {err:?}"
            );
        }
    }
}
