//! Tests that a turn keeps its reasoning only when it belongs to the slot's own
//! request.
//!
//! Every round of the request the slot is continuing keeps its reasoning. A turn
//! of a finished request, or one borrowed from another timeline at the same index,
//! is windowed. With no slot identity nothing keeps its reasoning, which is the
//! safe direction. A face on the spine carries its unit's reason.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, ResolvedTurn, TurnId};

    use super::super::tests_fixtures::tests::tl;

    /// Names one face's unit; every turn costs ten tokens.
    pub(in super::super) struct FaceResolver {
        pub(in super::super) face: TurnKey,
        pub(in super::super) unit: TurnKey,
    }

    impl ContentResolver for FaceResolver {
        fn group_turns(&self, _group: GroupId) -> Vec<TurnKey> {
            Vec::new()
        }
        fn turn_token_count(&self, _turn: TurnKey) -> usize {
            10
        }
        fn trained_window(&self) -> Option<usize> {
            None
        }
        fn turn_score(&self, _turn: TurnKey) -> f32 {
            0.0
        }
        fn section_token_count(&self, _section: SectionId) -> usize {
            0
        }
        fn unit_of(&self, turn: TurnKey) -> Option<TurnKey> {
            (turn == self.face).then_some(self.unit)
        }
    }

    /// **A face's card carries its unit's reason.** A dialogue holds a seed
    /// link's opening by its face, so the spine names the face while the
    /// selection's origins name the unit; looked up under the face, every seed
    /// card was drawn with no reason at all.
    #[test]
    fn a_face_on_the_spine_carries_its_units_reason() {
        let b = crate::projection::Builder::from_yaml(
            r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: dialogue
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: conversation
        selection: { kind: conversation, recent: 4 }
"#,
        )
        .unwrap();
        let layer = b.id_for_layer("dialogue").unwrap();
        let group = b.id_for_group("conversation").unwrap();
        let face = TurnKey::new(tl(31), TurnIndex(0));
        let unit = TurnKey::new(tl(30), TurnIndex(0));
        let segments = [ProjectionSegment::Sealed(SealedKind::Turn(
            ResolvedTurn {
                id: TurnId {
                    layer_id: layer,
                    group_id: group,
                    index: face.index,
                },
                timeline: Some(face.timeline),
            },
            crate::Role::Assistant,
        ))];
        let origins = HashMap::from([(unit, SelectionOrigin::Seed)]);
        let pieces = materialize_conversation(
            &segments,
            &origins,
            &FaceResolver { face, unit },
            b.schema(),
            |_| String::new(),
        );
        let reasons: Vec<Option<SelectionOrigin>> = pieces
            .iter()
            .filter_map(|p| match p {
                MaterializedPiece::Turn { turn } => Some(turn.reason),
                MaterializedPiece::Glue { .. } => None,
            })
            .collect();
        assert_eq!(reasons, [Some(SelectionOrigin::Seed)]);
    }

    /// Every round of the request the slot is continuing keeps its reasoning.
    #[test]
    fn the_slots_request_rounds_keep_their_reasoning() {
        let slot = tl(7);
        let request = [TurnIndex(2), TurnIndex(3), TurnIndex(4)];
        for idx in request {
            assert!(keeps_reasoning(Some(slot), slot, &request, idx), "{idx:?}");
        }
    }

    #[test]
    fn a_turn_of_a_finished_request_is_windowed() {
        let slot = tl(7);
        let request = [TurnIndex(3), TurnIndex(4)];
        assert!(!keeps_reasoning(Some(slot), slot, &request, TurnIndex(1)));
    }

    /// **The regression this guards.** A projection that spans timelines —
    /// retrieval, or a forked conversation inheriting its ancestors — must not
    /// let a borrowed conversation's turn arrive with its reasoning. Read as the
    /// slot's own thought, an inherited "I already read that file" makes the
    /// next turn skip its own work.
    #[test]
    fn another_timelines_turn_is_windowed_at_the_same_index() {
        let slot = tl(7);
        let ancestor = tl(3);
        assert!(
            !keeps_reasoning(Some(slot), ancestor, &[TurnIndex(1)], TurnIndex(1)),
            "an index of the slot's request names no FOREIGN turn",
        );
    }

    /// With no slot identity there is no request being continued, so nothing
    /// keeps its reasoning — windowing is the safe direction.
    #[test]
    fn without_a_slot_target_nothing_keeps_its_reasoning() {
        assert!(!keeps_reasoning(None, tl(3), &[TurnIndex(1)], TurnIndex(1)));
    }
}
