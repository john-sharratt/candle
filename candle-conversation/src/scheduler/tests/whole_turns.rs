//! **A turn is placed whole, whatever its pages inside**
//! (`docs/hit_driven_projection.md` §8.1).
//!
//! A turn's pages start and end with the turn, but the page edge between its
//! two records is only where the seal cut one: a turn sealed before the cut
//! landed on the record edge has a page straddling it. The slot places whole
//! turns only, so such a turn goes in at its full width with every page it
//! holds. Measured 2026-11-18: a reading's turn with pages `[0, 4858, 4907,
//! 4908]` against a record edge of 4850 could not be placed a record at a time,
//! and every reprojection of the dialogue that carved it failed. These run the
//! real assembler on the CPU double.

use super::positional_ledger::toy_page;
use super::*;
use crate::index_pages;
use crate::persistence::content_hash::turn_stream_id;
use crate::projection::{
    GroupId, LayerId, ProjectionSegment, ResolvedTurn, SealedKind, TimelineId, TurnId, TurnIndex,
};
use crate::substrate::TurnPartWrite;
use crate::token_buffer::TokenBuffer;
use crate::turn_layout::TurnLayout;

const GROUP: u32 = 9;

/// The grid every turn here has: `[user_start 3][user 4][im_end 2][a_start 3]
/// [answer 4][im_end 2]`, 18 tokens; the user record is `[0, 8)`.
const TOTAL: usize = 18;
const EDGE: usize = 8;

fn layout() -> TurnLayout {
    TurnLayout::from_flat_grid_with_tail(
        3,
        7,
        12,
        TOTAL as u32,
        2,
        3,
        2,
        "q".into(),
        Some("a".into()),
        false,
    )
}

/// One sealed turn of [`TOTAL`] tokens with K/V on the session and toy pages
/// of `widths`, written the way a seal writes them.
fn sealed_turn(
    widths: &[usize],
) -> (
    Scheduler,
    RecurrentProbe,
    Conversation,
    TimelineId,
    TurnIndex,
) {
    let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
    let conversation = Conversation::new();
    let timeline = TimelineId::for_test(u64::from(GROUP));
    conversation.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(GROUP));
    let src = scheduler.session.create_sequence().expect("scratch slot");
    scheduler
        .session
        .ensure_capacity(&[src], TOTAL)
        .expect("capacity");
    for b in scheduler.session.backings() {
        b.set_len(src, TOTAL);
    }
    scheduler
        .session
        .advance_sequence(src, TOTAL)
        .expect("advance");
    let sealed = scheduler
        .session
        .snapshot_sequence_per_layer(src)
        .expect("snapshot");
    let blocks = sealed[0].chunks.len() as u64;
    let index = conversation
        .record_turn(
            timeline,
            Role::Assistant,
            TurnPartWrite {
                layout: layout(),
                token_ids: TokenBuffer::from(vec![1u32; TOTAL]),
                token_count: TOTAL,
                block_start: 0,
                block_end: blocks,
                sealed_gpu: Some(Arc::new(sealed)),
                ..Default::default()
            },
            |seqs| Ok(seqs.to_vec()),
        )
        .expect("record the turn");
    let pages: Vec<(usize, Vec<u8>)> = widths
        .iter()
        .enumerate()
        .map(|(i, &w)| (w, toy_page(w, &[i as u8])))
        .collect();
    conversation.enqueue_index_page(
        turn_stream_id(timeline.raw(), index.0),
        index_pages::encode(&pages),
    );
    (scheduler, probe, conversation, timeline, index)
}

fn turn(timeline: TimelineId, index: TurnIndex) -> ProjectionSegment {
    ProjectionSegment::Sealed(SealedKind::Turn(
        ResolvedTurn {
            id: TurnId {
                layer_id: LayerId::for_test(1),
                group_id: GroupId::for_test(GROUP),
                index,
            },
            timeline: Some(timeline),
        },
        Role::Assistant,
    ))
}

fn pages_pushed(probe: &RecurrentProbe, slot: SequenceId) -> usize {
    probe.pushed.lock().unwrap().get(&slot.0).map_or(0, |p| {
        p.iter().filter(|x| matches!(x, Pushed::Page(_))).count()
    })
}

/// A turn whose pages meet at its record edge goes in whole with every page.
#[test]
fn a_turn_goes_in_whole_with_every_page() {
    let (mut scheduler, probe, conversation, timeline, index) = sealed_turn(&[EDGE, 6, 4]);
    let slot = create_scratch_slot(&mut scheduler, &conversation);
    let complete = scheduler
        .apply_projection(slot, BlockCount(0), &[turn(timeline, index)])
        .expect("the turn places");
    assert!(complete, "every piece placed, its gap-fill run here");
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(TOTAL));
    assert_eq!(pages_pushed(&probe, slot), 3);
}

/// **A turn whose page straddles its record edge still goes in whole**: its
/// pages `[0, 5, 18]` miss the edge at 8, and the turn places at its full
/// width with both pages.
#[test]
fn a_turn_whose_page_straddles_its_record_edge_goes_in_whole() {
    let (mut scheduler, probe, conversation, timeline, index) = sealed_turn(&[5, 13]);
    let slot = create_scratch_slot(&mut scheduler, &conversation);
    let complete = scheduler
        .apply_projection(slot, BlockCount(0), &[turn(timeline, index)])
        .expect("the turn places whole");
    assert!(complete);
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(TOTAL));
    assert_eq!(pages_pushed(&probe, slot), 2);
}
