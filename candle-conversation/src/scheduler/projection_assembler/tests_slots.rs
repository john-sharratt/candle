//! Tests the island cache's retention, the rebuild comparison, and the geometry of
//! the slot record.
//!
//! An island survives exactly the retention horizon of capture passes untouched,
//! and a hit restarts its clock, so shapes that alternate never evict each other.
//! A rebuild is compared only when the slot held content of its own. The record's
//! ends and prompt end are checked against the sums of the widths that set them.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{ResolvedSection, TimelineId};

    use super::super::apply::leads_of;
    use super::super::tests_fixtures::tests::{generated_seg, section_seg, turn_seg};

    /// The island cache's retention contract: an entry survives exactly
    /// [`GLUE_ISLAND_RETAIN_GENERATIONS`] capture passes untouched, and a
    /// hit-refresh (the build path stamping the current generation) restarts
    /// its clock — so projection shapes that alternate within the horizon
    /// never evict each other, while dead shapes age out.
    #[test]
    fn island_cache_retention_and_refresh() {
        let mut state = SlotState::default();
        // A cached island carries its K/V and (for a gap-filling model) no index
        // page — see the capture site.
        let sealed = || CapturedSpan {
            kv: Arc::new(Vec::<SealedSequence>::new()),
            page: None,
        };

        // Captured at generation 1.
        let gen = state.begin_island_capture();
        assert_eq!(gen, 1);
        state.glue_islands.insert(0xA, (gen, sealed()));

        // Untouched entries survive the horizon's worth of passes...
        for _ in 0..GLUE_ISLAND_RETAIN_GENERATIONS {
            state.begin_island_capture();
        }
        assert!(
            state.glue_islands.contains_key(&0xA),
            "entry must survive the full retention horizon"
        );
        // ...and age out one pass later.
        state.begin_island_capture();
        assert!(
            !state.glue_islands.contains_key(&0xA),
            "entry past the horizon must be pruned"
        );

        // A hit-refresh restarts the clock: stamp the CURRENT generation (what
        // the build path's cache hit does), then run a full horizon again.
        let gen = state.begin_island_capture();
        state.glue_islands.insert(0xB, (gen, sealed()));
        for _ in 0..GLUE_ISLAND_RETAIN_GENERATIONS {
            let g = state.glue_generation;
            state.glue_islands.get_mut(&0xB).unwrap().0 = g; // hit refresh
            state.begin_island_capture();
        }
        assert!(
            state.glue_islands.contains_key(&0xB),
            "a hit-refreshed entry must never age out while its shape recurs"
        );
    }

    /// The rebuild check compares only a slot holding content of its own. The
    /// first case is a resumed passthrough conversation's, as measured: its
    /// slot was built holding the 2,671-token system prompt alone, and the
    /// first turn's projection brought 3,486 tokens of history into it.
    #[test]
    fn a_rebuild_is_compared_only_when_the_slot_held_content_of_its_own() {
        assert_eq!(
            rebuild_mismatch(2671, 2671, 6157),
            None,
            "a slot built system-only is populated"
        );
        assert_eq!(rebuild_mismatch(0, 0, 6157), None, "an empty slot");
        assert_eq!(rebuild_mismatch(0, 2671, 6157), None, "a reset slot");
        assert_eq!(
            rebuild_mismatch(6157, 2671, 6157),
            None,
            "a faithful rebuild"
        );
        assert_eq!(
            rebuild_mismatch(6157, 2671, 9643),
            Some((3486, true)),
            "a piece restored twice"
        );
        assert_eq!(
            rebuild_mismatch(6157, 2671, 6000),
            Some((157, false)),
            "content lost"
        );
    }

    #[test]
    fn slot_state_default_is_empty() {
        let s = SlotState::default();
        assert!(s.pending_user_part.is_none());
        assert!(s.working_set.sections.is_empty() && s.working_set.turns.is_empty());
    }

    #[test]
    fn working_set_extracts_sealed_sections_and_turns() {
        // Sealed sections/turns contribute their ids; Generated glue does not.
        let segments = vec![
            section_seg(1),
            turn_seg(0, 5, 2),
            generated_seg("glue", 0, &[9, 9]),
            section_seg(3),
            turn_seg(0, 5, 7),
        ];
        let ws = working_set_from_segments(&segments);
        assert_eq!(ws.sections, vec![SectionId::new(1), SectionId::new(3)]);
        assert_eq!(
            ws.turns,
            vec![
                TurnKey::new(TimelineId::for_test(5), TurnIndex(2)),
                TurnKey::new(TimelineId::for_test(5), TurnIndex(7)),
            ]
        );

        // A projection of only live-prefilled glue attends no sealed KV.
        let none = working_set_from_segments(&[generated_seg("g", 0, &[1])]);
        assert!(none.sections.is_empty() && none.turns.is_empty());
    }

    /// **A projection's working set is held one projection past it.** Each
    /// install moves the current set to `previous_working_set`, so what one
    /// projection dropped is still named until the slot's next projection;
    /// a fork starts with nothing held of its own.
    #[test]
    fn the_replaced_working_set_is_held_for_one_projection() {
        let turn = |i| TurnKey::new(TimelineId::for_test(5), TurnIndex(i));
        let mut s = SlotState::default();
        s.set_working_set(working_set_from_segments(&[
            turn_seg(0, 5, 1),
            turn_seg(0, 5, 2),
        ]));
        assert!(s.previous_working_set.turns.is_empty());

        s.set_working_set(working_set_from_segments(&[
            turn_seg(0, 5, 2),
            turn_seg(0, 5, 3),
        ]));
        assert_eq!(s.working_set.turns, vec![turn(2), turn(3)]);
        assert_eq!(s.previous_working_set.turns, vec![turn(1), turn(2)]);

        s.set_working_set(working_set_from_segments(&[turn_seg(0, 5, 4)]));
        assert_eq!(s.previous_working_set.turns, vec![turn(2), turn(3)]);
        assert!(
            !s.previous_working_set.turns.contains(&turn(1)),
            "dropped two projections ago: no longer held"
        );

        let fork = s.fork();
        assert_eq!(fork.working_set.turns, vec![turn(4)]);
        assert!(fork.previous_working_set.turns.is_empty());
    }

    /// **Records laid end to end learn where they now stand.** After a splice
    /// the pieces keep their own widths and contributions and nothing else:
    /// each one's ends are the sums of the widths before it, and its prefix
    /// hash is the fold of the contributions through it — what a walk placing
    /// them in this order would have recorded.
    #[test]
    fn restacked_records_end_where_their_widths_put_them() {
        let piece = |blocks: usize, tokens: u32, contrib: u64| {
            let mut p = PlacedPiece::ending_at(99);
            p.blocks = blocks;
            p.tokens = tokens;
            p.contrib = contrib;
            p
        };
        let mut records = vec![piece(2, 40, 11), piece(1, 7, 22), piece(3, 70, 33)];
        restack(&mut records);
        let ends: Vec<(usize, u32)> = records
            .iter()
            .map(|r| (r.blocks_after, r.pos_after))
            .collect();
        assert_eq!(ends, [(2, 40), (3, 47), (6, 117)]);
        let mut walked = DefaultHasher::new();
        for (r, c) in records.iter().zip([11u64, 22, 33]) {
            walked.write_u64(c);
            assert_eq!(r.prefix_h.finish(), walked.finish());
        }
    }

    /// **Each hole is the lead of the first block after it**: holes that stand
    /// together lead the same block, a piece's later blocks lead nothing from
    /// here, the holes after the last block are pending, and a piece that holds
    /// blocks and leaves positions is refused — no record says where its hole
    /// falls.
    #[test]
    fn each_hole_leads_the_first_block_after_it() {
        let piece = |blocks: usize, holes: u32| {
            let mut p = PlacedPiece::ending_at(99);
            p.blocks = blocks;
            p.holes = holes;
            p
        };
        let records = [
            piece(2, 0),
            piece(0, 100),
            piece(0, 50),
            piece(3, 0),
            piece(1, 0),
            piece(0, 7),
        ];
        let (leads, pending, blocks) = leads_of(&records).unwrap();
        assert_eq!(leads, [(0, 0), (2, 150), (5, 0)]);
        assert_eq!((pending, blocks), (7, 6));
        assert!(leads_of(&[piece(2, 3)]).is_err());
        let (leads, pending, blocks) = leads_of(&[]).unwrap();
        assert!(leads.is_empty());
        assert_eq!((pending, blocks), (0, 0));
    }

    /// The system prompt is the leading run of sections and the glue between
    /// them, and ends where the slot stood after the last of them — not at the
    /// first section, and not past a turn.
    #[test]
    fn the_prompt_ends_after_the_leading_sections_and_their_glue() {
        let placed_at = |ends: &[u32]| -> Vec<PlacedPiece> {
            ends.iter()
                .map(|&pos_after| PlacedPiece {
                    pos_after,
                    at_after: pos_after,
                    ..PlacedPiece::ending_at(0)
                })
                .collect()
        };
        let pieces = assemble_pieces(&[
            generated_seg("a", 0, &[1, 2]),
            section_seg(7),
            generated_seg("b", 0, &[3]),
            section_seg(8),
            turn_seg(1, 2, 3),
            section_seg(9),
            ProjectionSegment::NewUserMessage {
                tokens: Arc::new(vec![9, 9]),
            },
        ]);
        assert_eq!(
            prompt_end_of(&pieces, &placed_at(&[2, 900, 901, 4100, 4300, 4500])),
            4100
        );
        // A projection that opens on a turn holds no prompt.
        let opens_on_a_turn = assemble_pieces(&[turn_seg(1, 2, 3), section_seg(9)]);
        assert_eq!(prompt_end_of(&opens_on_a_turn, &placed_at(&[200, 400])), 0);
    }

    /// **The pinned sections' places are the ranges the model attends whole**:
    /// each marked section at the place its piece took, glue and unmarked
    /// sections left out, and adjacent pinned sections merged into one range.
    #[test]
    fn the_pinned_sections_become_position_ranges() {
        let placed = |ends: &[(u32, u32)]| -> Vec<PlacedPiece> {
            ends.iter()
                .map(|&(tokens, pos_after)| PlacedPiece {
                    tokens,
                    pos_after,
                    at_after: pos_after,
                    ..PlacedPiece::ending_at(0)
                })
                .collect()
        };
        let pinned = |id: u32, pin: Pin| {
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection {
                id: SectionId::new(id),
                pin,
            }))
        };
        let segments = vec![
            generated_seg("open", 0, &[1, 2]),
            pinned(7, Pin::Whole),
            pinned(8, Pin::No),
            pinned(9, Pin::Member),
            pinned(10, Pin::Member),
            generated_seg("glue", 0, &[3]),
            pinned(11, Pin::Whole),
            turn_seg(1, 2, 3),
        ];
        let pieces = assemble_pieces(&segments);
        // open 0..2, s7 2..302, s8 302..5302, s9 5302..5402, s10 5402..5502,
        // glue 5502..5503, s11 5503..5603, then the turn.
        let records = placed(&[
            (2, 2),
            (300, 302),
            (5000, 5302),
            (100, 5402),
            (100, 5502),
            (1, 5503),
            (100, 5603),
            (40, 5643),
        ]);
        assert_eq!(
            pin_ranges_of(&segments, &pieces, &records),
            vec![2..302, 5302..5502, 5503..5603]
        );
        // Nothing marked, nothing pinned.
        let plain = vec![section_seg(7), section_seg(8)];
        let pieces = assemble_pieces(&plain);
        assert!(pin_ranges_of(&plain, &pieces, &placed(&[(5, 5), (5, 10)])).is_empty());
    }
}
