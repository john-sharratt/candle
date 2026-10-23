//! Whether the dialogue has started losing sight of turns its journal has not
//! looked at.
//!
//! The dialogue's projection keeps its most recent turns and the older ones its
//! selection scores well, and leaves the rest out. A turn left out that no
//! entry covers is one the session can no longer see and has not written down,
//! so it asks for a draft now rather than at the next [`super::state::EVERY_TURNS`].
//! The projection event a round already records names every turn it selected,
//! so the signal costs nothing.

use candle_conversation::projection::TimelineId;
use candle_conversation::ProjectionEvent;

/// The turns of `timeline` that `event`'s projection selected.
pub fn selected_turns(event: &ProjectionEvent, timeline: TimelineId) -> Vec<u32> {
    event
        .selection
        .turns
        .iter()
        .filter(|t| t.selected && t.timeline == Some(timeline.raw()))
        .map(|t| t.index)
        .collect()
}

/// Whether a turn from `first_uncovered` up to (not including) the turn just
/// decoded, `newest`, was left out of a projection that selected `selected`.
pub fn left_out(selected: &[u32], first_uncovered: u32, newest: u32) -> bool {
    (first_uncovered..newest).any(|t| !selected.contains(&t))
}

#[cfg(test)]
mod tests {
    use candle_conversation::projection::{ProjectionSelection, SelectedTurn};
    use candle_conversation::summary_tree::TurnKind;

    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn turn(timeline: Option<u64>, index: u32, selected: bool) -> SelectedTurn {
        SelectedTurn {
            layer: "dialogue".into(),
            group: "primary_conversation".into(),
            index,
            role: "user".into(),
            tokens: 10,
            kind: TurnKind::Normal,
            reason: None,
            qualified: true,
            timeline,
            selected,
            score: 0.0,
        }
    }

    fn event(turns: Vec<SelectedTurn>) -> ProjectionEvent {
        ProjectionEvent {
            selection: ProjectionSelection {
                system: Vec::new(),
                turns,
            },
            ..ProjectionEvent::default()
        }
    }

    #[test]
    fn only_this_conversations_selected_turns_are_counted() {
        let e = event(vec![
            turn(Some(7), 3, true),
            turn(Some(9), 4, true),
            turn(Some(7), 5, false),
            turn(None, 6, true),
            turn(Some(7), 8, true),
        ]);
        assert_eq!(selected_turns(&e, tl(7)), [3, 8]);
    }

    #[test]
    fn an_uncovered_turn_left_out_is_pressure() {
        assert!(left_out(&[0, 1, 3], 0, 4), "turn 2 was left out");
    }

    #[test]
    fn a_covered_turn_left_out_is_no_pressure() {
        assert!(!left_out(&[3], 3, 4), "0..=2 are covered");
    }

    #[test]
    fn the_turn_just_decoded_is_not_counted() {
        assert!(!left_out(&[0, 1], 0, 2));
        assert!(!left_out(&[], 5, 5), "nothing uncovered before it");
    }
}
