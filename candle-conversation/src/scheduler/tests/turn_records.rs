//! **A turn's records are placed alone, each with its own pages**
//! (`docs/hit_driven_projection.md` §8.1).
//!
//! A tool round is one turn's assistant record and the next turn's user
//! record, so the slot places records rather than whole turns. Each record's
//! K/V is a window of its turn's chunks and its rows are the pages inside that
//! window; a record whose edge falls inside a page has no rows of its own and
//! is refused. These run the real assembler on the CPU double.

use super::positional_ledger::toy_page;
use super::*;
use crate::index_pages;
use crate::persistence::content_hash::turn_stream_id;
use crate::projection::{
    GroupId, LayerId, ProjectionSegment, ResolvedTurn, SealedKind, TimelineId, TurnId, TurnIndex,
};
use crate::substrate::TurnPartWrite;
use crate::token_buffer::TokenBuffer;
use crate::turn_layout::{TurnLayout, TurnRecord};

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

fn record(timeline: TimelineId, index: TurnIndex, which: TurnRecord) -> ProjectionSegment {
    ProjectionSegment::Sealed(SealedKind::TurnHalf(
        ResolvedTurn {
            id: TurnId {
                layer_id: LayerId::for_test(1),
                group_id: GroupId::for_test(GROUP),
                index,
            },
            timeline: Some(timeline),
        },
        which,
    ))
}

fn pages_pushed(probe: &RecurrentProbe, slot: SequenceId) -> usize {
    probe.pushed.lock().unwrap().get(&slot.0).map_or(0, |p| {
        p.iter().filter(|x| matches!(x, Pushed::Page(_))).count()
    })
}

/// Each record goes in at its own width with only the pages inside it: the
/// user record's one page, then the assistant record's two.
#[test]
fn each_record_goes_in_with_the_pages_inside_it() {
    let (mut scheduler, probe, conversation, timeline, index) = sealed_turn(&[EDGE, 6, 4]);
    let slot = create_scratch_slot(&mut scheduler, &conversation);
    scheduler
        .apply_projection(
            slot,
            BlockCount(0),
            &[record(timeline, index, TurnRecord::User)],
        )
        .expect("the user record places");
    assert_eq!(scheduler.session.sequence_offset(slot.0), Some(EDGE));
    assert_eq!(pages_pushed(&probe, slot), 1);

    let other = create_scratch_slot(&mut scheduler, &conversation);
    scheduler
        .apply_projection(
            other,
            BlockCount(0),
            &[record(timeline, index, TurnRecord::Assistant)],
        )
        .expect("the assistant record places");
    assert_eq!(
        scheduler.session.sequence_offset(other.0),
        Some(TOTAL - EDGE)
    );
    assert_eq!(pages_pushed(&probe, other), 2);
}

/// Pages that straddle the edge between the records describe neither, so a
/// record placed from them is refused rather than given rows for the wrong
/// span.
#[test]
fn a_record_whose_edge_falls_inside_a_page_is_refused() {
    let (mut scheduler, _probe, conversation, timeline, index) = sealed_turn(&[5, 13]);
    let slot = create_scratch_slot(&mut scheduler, &conversation);
    let placed = scheduler.apply_projection(
        slot,
        BlockCount(0),
        &[record(timeline, index, TurnRecord::User)],
    );
    assert!(placed.is_err(), "the user record has no pages of its own");
}
