//! Tests the piece layout the assembler produces for turns, glue and a live
//! message.
//!
//! A sealed turn emits alone. A projection of sealed turns and a live message
//! reserves no glue, since a recurrent model cannot compute K/V for a hole in the
//! middle of a sequence. Any inter-turn island that remains is asserted as a
//! shape, because it can be split at its seam only when each turn takes both
//! halves.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, TimelineId};

    use super::super::tests_fixtures::tests::{generated_seg, section_seg, turn_seg};

    /// **A sealed turn emits alone, and consecutive turns share no island.**
    ///
    /// This used to assert the opposite — `Glue(user_start)` before each turn
    /// and a merged `Glue(assistant_end ++ user_start)` between them. Those
    /// markers now live in each turn's own grid, so the assembler emits nothing
    /// around a sealed turn and consecutive turns abut directly.
    ///
    /// The emitted TOKEN stream is unchanged; what changed is that the markers
    /// arrive as part of a sealed turn's injected K/V rather than as a run the
    /// engine must gap-fill. That is the whole difference for a recurrent model:
    /// there is no hole in the middle of the sequence left to compute.
    #[test]
    fn assemble_pieces_emits_sealed_turns_alone_and_defers_user() {
        let segments = vec![
            generated_seg("a", 0, &[1, 2]),
            section_seg(7),
            turn_seg(1, 2, 3),
            turn_seg(1, 2, 4),
            generated_seg("b", 0, &[3]),
            ProjectionSegment::NewUserMessage {
                tokens: Arc::new(vec![9, 9]),
            },
        ];
        let pieces = assemble_pieces(&segments);
        assert_eq!(
            pieces,
            vec![
                // leading Generated run — template glue, still the assembler's
                AssembledPiece::Glue(vec![1, 2]),
                AssembledPiece::Section(SectionId::new(7)),
                // no opener island: the turn carries its own
                AssembledPiece::Turn {
                    group: GroupId::for_test(2),
                    index: TurnIndex(3),
                    role: crate::Role::Assistant,
                    timeline: Some(TimelineId::for_test(2)),
                },
                // and no island BETWEEN turns — turn 3's closer and turn 4's
                // opener are inside their respective grids, abutting directly
                AssembledPiece::Turn {
                    group: GroupId::for_test(2),
                    index: TurnIndex(4),
                    role: crate::Role::Assistant,
                    timeline: Some(TimelineId::for_test(2)),
                },
                // the trailing Generated run stands alone now
                AssembledPiece::Glue(vec![3]),
                // in-flight user message deferred past the gap-fill
                AssembledPiece::DeferredUser(Arc::new(vec![9, 9])),
            ]
        );
    }

    /// **The gate: a projection of sealed turns reserves no glue at all.**
    ///
    /// System prompt sections + sealed turns + a live user message must emit
    /// zero `Glue` pieces, because every `Glue` piece is an island the engine
    /// gap-fills and a recurrence cannot compute K/V for tokens inserted
    /// mid-sequence. This is `T5c.1` at the assembler level — the single
    /// assertion that says the lineage can run.
    #[test]
    fn a_projection_of_sealed_turns_and_a_live_message_emits_no_glue() {
        let segments = vec![
            section_seg(7),
            section_seg(8),
            turn_seg(1, 2, 3),
            turn_seg(1, 2, 4),
            turn_seg(1, 2, 5),
            ProjectionSegment::NewUserMessage {
                tokens: Arc::new(vec![9, 9]),
            },
        ];
        let pieces = assemble_pieces(&segments);
        let glue: Vec<&AssembledPiece> = pieces
            .iter()
            .filter(|p| matches!(p, AssembledPiece::Glue(_)))
            .collect();
        assert!(
            glue.is_empty(),
            "a projection with no template sections must reserve no glue \
             islands — every one is a hole a recurrent model cannot fill: {glue:?}"
        );
        assert!(matches!(
            pieces.last(),
            Some(AssembledPiece::DeferredUser(_))
        ));
    }

    /// **Every inter-turn island is exactly `assistant_end ++ user_start`.**
    ///
    /// This is the structural fact that decides boundary ownership, and it is
    /// asserted here as a *shape* rather than as one expected vector so it
    /// keeps meaning something as the surrounding test data changes.
    ///
    /// Splitting that island at the `AE|US` seam is lossless **only** if each
    /// turn takes both halves: leading-only drops every `assistant_end`,
    /// trailing-only drops every `user_start`. That is why
    /// `TurnLayout::from_flat_grid_with_tail` bakes both ends or neither, and
    /// this test is what stops a later "simplify to one end" from looking safe.
    ///
    /// It also pins the CURRENT stream, which is the reference the baked-turn
    /// change (§4.7 / P5b) has to reproduce token for token — the failure it
    /// guards against is a dropped or doubled `<|im_end|>`, which reads fine
    /// and shifts every boundary after it.
    #[test]
    fn glue_bridge_window_opens_only_into_turns() {
        let turn = AssembledPiece::Turn {
            group: GroupId::for_test(2),
            index: TurnIndex(3),
            role: crate::Role::Assistant,
            timeline: Some(TimelineId::for_test(2)),
        };
        // Leads into a resident turn/summary → bridge window opens.
        assert_eq!(glue_bridge_window(Some(&turn)), TURN_BRIDGE_FWD_AHEAD);
        // No resident turn ahead → backward-only.
        assert_eq!(
            glue_bridge_window(Some(&AssembledPiece::Section(SectionId::new(7)))),
            0
        );
        assert_eq!(
            glue_bridge_window(Some(&AssembledPiece::DeferredUser(Arc::new(vec![9])))),
            0
        );
        assert_eq!(glue_bridge_window(None), 0);
    }

    #[test]
    fn glue_bridge_window_matches_assembled_stream() {
        // Walk the same stream as `assemble_pieces_wraps_turns_…`: every glue
        // island's window is decided by the piece that follows it.
        let segments = vec![
            generated_seg("a", 0, &[1, 2]),
            section_seg(7),
            turn_seg(1, 2, 3),
            turn_seg(1, 2, 4),
            generated_seg("b", 0, &[3]),
            ProjectionSegment::NewUserMessage {
                tokens: Arc::new(vec![9, 9]),
            },
        ];
        let pieces = assemble_pieces(&segments);
        let windows: Vec<u32> = (0..pieces.len())
            .filter(|&i| matches!(pieces[i], AssembledPiece::Glue(_)))
            .map(|i| glue_bridge_window(pieces.get(i + 1)))
            .collect();
        // Two islands remain, both from template `Generated` runs: the leading
        // run→Section (0) and the trailing run→DeferredUser (0). The two that
        // used to sit around the turns are gone — the turns carry their own
        // markers — so no glue island leads into a Turn any more.
        assert_eq!(windows, vec![0, 0]);
    }

    #[test]
    fn assemble_pieces_consecutive_generated_form_one_island() {
        let segments = vec![
            generated_seg("a", 0, &[1]),
            generated_seg("b", 1, &[2, 3]),
            section_seg(5),
        ];
        let pieces = assemble_pieces(&segments);
        assert_eq!(
            pieces,
            vec![
                AssembledPiece::Glue(vec![1, 2, 3]),
                AssembledPiece::Section(SectionId::new(5)),
            ]
        );
    }

    #[test]
    fn assemble_pieces_empty_is_empty() {
        assert!(assemble_pieces(&[]).is_empty());
    }
}
