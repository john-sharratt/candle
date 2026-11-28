//! **A rebuild that splices is a build of the same pieces.**
//!
//! A projection may insert and evict anywhere, and a rebuild keeps every piece
//! the slot already holds, splicing the changes in where they fall
//! (`piece_splice`). Whatever it keeps or splices, the slot it leaves must be
//! the one a rebuild from nothing would have built: the same K/V chunks in the
//! same order, the same offset, the same per-position state at the same
//! positions, the same records, and the same ranges written back. These run
//! the real assembler on the CPU double, against a fresh slot built from the
//! same pieces.

use super::positional_ledger::toy_page;
use super::*;
use crate::index_pages;
use crate::persistence::content_hash::turn_stream_id;
use crate::projection::{
    GroupId, LayerId, ProjectionSegment, ResolvedTurn, SealedKind, TimelineId, TurnId, TurnIndex,
};
use crate::substrate::TurnPartWrite;
use crate::token_buffer::TokenBuffer;

const GROUP: u32 = 7;

/// A scheduler on the recurrent double, a conversation, and its sealed turns.
struct Fixture {
    scheduler: Scheduler,
    probe: RecurrentProbe,
    conversation: Conversation,
    timeline: TimelineId,
    turns: Vec<TurnIndex>,
}

/// One sealed turn per width, each with K/V of its own on the session and a
/// toy page covering it, written the way a seal writes it.
fn fixture(widths: &[usize]) -> Fixture {
    let (mut scheduler, _tx, probe) = make_test_scheduler_recurrent();
    let conversation = Conversation::new();
    let timeline = TimelineId::for_test(u64::from(GROUP));
    conversation.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(GROUP));
    let turns = widths
        .iter()
        .enumerate()
        .map(|(i, &w)| seal_turn(&mut scheduler, &conversation, timeline, w, i as u8))
        .collect();
    Fixture {
        scheduler,
        probe,
        conversation,
        timeline,
        turns,
    }
}

fn seal_turn(
    scheduler: &mut Scheduler,
    conversation: &Conversation,
    timeline: TimelineId,
    tokens: usize,
    label: u8,
) -> TurnIndex {
    let src = scheduler.session.create_sequence().expect("scratch slot");
    scheduler
        .session
        .ensure_capacity(&[src], tokens)
        .expect("capacity");
    for b in scheduler.session.backings() {
        b.set_len(src, tokens);
    }
    scheduler
        .session
        .advance_sequence(src, tokens)
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
                token_ids: TokenBuffer::from(vec![u32::from(label); tokens]),
                token_count: tokens,
                block_start: 0,
                block_end: blocks,
                sealed_gpu: Some(Arc::new(sealed)),
                ..Default::default()
            },
            |seqs| Ok(seqs.to_vec()),
        )
        .expect("record the turn");
    conversation.enqueue_index_page(
        turn_stream_id(timeline.raw(), index.0),
        index_pages::encode(&[(tokens, toy_page(tokens, &[label]))]),
    );
    index
}

impl Fixture {
    /// The segments placing turns `which`, in that order.
    fn segments(&self, which: &[usize]) -> Vec<ProjectionSegment> {
        which
            .iter()
            .map(|&t| {
                ProjectionSegment::Sealed(SealedKind::Turn(
                    ResolvedTurn {
                        id: TurnId {
                            layer_id: LayerId::for_test(1),
                            group_id: GroupId::for_test(GROUP),
                            index: self.turns[t],
                        },
                        timeline: Some(self.timeline),
                    },
                    Role::Assistant,
                ))
            })
            .collect()
    }

    /// The segments placing `layout`: a turn by its index, or a hole of the
    /// given positions.
    fn segments_with(&self, layout: &[Place]) -> Vec<ProjectionSegment> {
        layout
            .iter()
            .flat_map(|p| match *p {
                Place::Turn(t) => self.segments(&[t]),
                Place::Hole(positions) => vec![ProjectionSegment::Hole { positions }],
            })
            .collect()
    }

    fn slot(&mut self) -> SequenceId {
        create_scratch_slot(&mut self.scheduler, &self.conversation)
    }

    fn project(&mut self, slot: SequenceId, which: &[usize]) {
        let segments = self.segments(which);
        self.scheduler
            .apply_projection(slot, BlockCount(0), &segments)
            .expect("projection applies");
    }

    fn project_layout(&mut self, slot: SequenceId, layout: &[Place]) {
        let segments = self.segments_with(layout);
        self.scheduler
            .apply_projection(slot, BlockCount(0), &segments)
            .expect("projection applies");
    }

    /// Everything a rebuild leaves that a reader of the slot depends on.
    fn built(&self, slot: SequenceId) -> Built {
        let kv = self
            .scheduler
            .session
            .snapshot_sequence_per_layer(slot.0)
            .expect("snapshot")
            .iter()
            .map(|layer| {
                layer
                    .chunks
                    .iter()
                    .map(|c| {
                        let gids: Vec<i64> = c.gids.as_slice().iter().map(|g| g.raw()).collect();
                        (c.offset, c.token_count, gids)
                    })
                    .collect()
            })
            .collect();
        let records = self
            .scheduler
            .slot_projection_state
            .get(&slot)
            .map(|s| s.placed_pieces.iter().map(|p| p.ends()).collect())
            .unwrap_or_default();
        let read = self.conversation.read();
        Built {
            kv,
            offset: self.scheduler.session.sequence_offset(slot.0),
            positional: self
                .probe
                .ledger
                .lock()
                .unwrap()
                .get(&slot.0)
                .map(|l| l.canonical())
                .unwrap_or_default(),
            records,
            ranges: self
                .turns
                .iter()
                .map(|&t| read.turn_block_range(self.timeline, t))
                .collect(),
        }
    }

    /// Pages handed to the model by a push onto the tail, ever, for `slot`.
    fn pages_pushed(&self, slot: SequenceId) -> usize {
        self.probe
            .pushed
            .lock()
            .unwrap()
            .get(&slot.0)
            .map_or(0, |p| {
                p.iter().filter(|x| matches!(x, Pushed::Page(_))).count()
            })
    }
}

/// One place in a projection's layout.
#[derive(Debug, Clone, Copy)]
enum Place {
    Turn(usize),
    Hole(usize),
}

type ChunkView = (u16, u16, Vec<i64>);

#[derive(Debug, PartialEq)]
struct Built {
    kv: Vec<Vec<ChunkView>>,
    offset: Option<usize>,
    positional: Vec<super::positional_ledger::Placed>,
    records: Vec<(usize, u32, u64)>,
    ranges: Vec<Option<(u64, u64)>>,
}

/// Rebuild a slot holding `from` into `to`, and build `to` from nothing on
/// another; the two must be the same slot. Returns how many pages the rebuild
/// pushed onto the tail, as against splicing them in.
fn spliced_equals_built(widths: &[usize], from: &[usize], to: &[usize]) -> usize {
    let mut f = fixture(widths);
    let spliced = f.slot();
    f.project(spliced, from);
    let pushed_before = f.pages_pushed(spliced);
    f.project(spliced, to);
    let pushed = f.pages_pushed(spliced) - pushed_before;
    let after_splice = f.built(spliced);

    let fresh = f.slot();
    f.project(fresh, to);
    let after_build = f.built(fresh);
    assert_eq!(
        after_splice, after_build,
        "rebuilding {from:?} into {to:?} left a different slot from building {to:?}"
    );
    pushed
}

/// The flip that made the pauses: one piece near the front changes and every
/// turn behind it stays. Only the new piece's page is placed — by the splice,
/// so nothing is pushed onto the tail at all.
#[test]
fn a_change_near_the_front_is_a_build_of_the_new_pieces() {
    let pushed = spliced_equals_built(
        &[40, 7, 33, 64, 5, 70, 12],
        &[0, 1, 2, 3, 4, 5],
        &[0, 6, 2, 3, 4, 5],
    );
    assert_eq!(
        pushed, 0,
        "every kept turn stayed and the new one was spliced in"
    );
}

/// Replace, evict and insert in one rebuild, and a new last piece walked onto
/// the tail: still a build of the result, and only the tail's page is pushed.
#[test]
fn inserts_evictions_and_a_new_tail_are_a_build_of_the_result() {
    let pushed = spliced_equals_built(
        &[40, 7, 33, 64, 5, 70, 12, 9],
        &[0, 1, 2, 3, 4],
        &[0, 5, 2, 4, 6, 7],
    );
    assert_eq!(
        pushed, 2,
        "only 6 and 7, past the last kept turn, are walked"
    );
}

/// Order matters: the same turns reordered keep what they can and place the
/// rest, and the slot is still a build of the new order.
#[test]
fn a_reordered_projection_is_a_build_of_the_new_order() {
    spliced_equals_built(&[40, 7, 33, 64, 5], &[0, 1, 2, 3, 4], &[3, 0, 1, 2, 4]);
}

/// Rebuild a slot holding `from` into `to` and build `to` from nothing on
/// another, holes included; the two must be the same slot, at the same
/// position. Returns that position and the offset.
fn spliced_layout_equals_built(widths: &[usize], from: &[Place], to: &[Place]) -> (usize, usize) {
    let mut f = fixture(widths);
    let spliced = f.slot();
    f.project_layout(spliced, from);
    f.project_layout(spliced, to);
    let after_splice = f.built(spliced);
    let at_splice = f.scheduler.session.sequence_position(spliced.0);

    let fresh = f.slot();
    f.project_layout(fresh, to);
    let after_build = f.built(fresh);
    let at_build = f.scheduler.session.sequence_position(fresh.0);
    assert_eq!(
        after_splice, after_build,
        "rebuilding {from:?} into {to:?} left a different slot from building {to:?}"
    );
    assert_eq!(
        at_splice, at_build,
        "the two slots stand at different positions"
    );
    (
        at_build.expect("a position"),
        f.scheduler
            .session
            .sequence_offset(fresh.0)
            .expect("an offset"),
    )
}

/// **A hole moves the position and not the offset**, and a slot built across
/// one holds exactly its turns.
#[test]
fn a_hole_moves_the_position_and_not_the_offset() {
    use Place::{Hole, Turn};
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33],
        &[Turn(0), Hole(100), Turn(1)],
        &[Turn(0), Hole(100), Turn(1), Turn(2)],
    );
    assert_eq!(offset, 40 + 7 + 33);
    assert_eq!(at, 40 + 100 + 7 + 33);
}

/// **A hole that changed is where the slot is cut and walked**: the rebuild
/// is still a build of the result, at the result's position.
#[test]
fn a_changed_hole_is_a_build_of_the_result() {
    use Place::{Hole, Turn};
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33],
        &[Turn(0), Hole(100), Turn(1), Turn(2)],
        &[Turn(0), Hole(60), Turn(1), Turn(2)],
    );
    assert_eq!((at, offset), (40 + 60 + 7 + 33, 40 + 7 + 33));
}

/// **A change after a kept hole splices past it**: the rebuild is a build of
/// the result, and the hole stands where it stood.
#[test]
fn a_change_after_a_kept_hole_is_a_build_of_the_result() {
    use Place::{Hole, Turn};
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33, 64, 5],
        &[Turn(0), Hole(100), Turn(1), Turn(2), Turn(3)],
        &[Turn(0), Hole(100), Turn(1), Turn(4), Turn(3)],
    );
    assert_eq!((at, offset), (40 + 100 + 7 + 5 + 64, 40 + 7 + 5 + 64));
}

/// **A reorder right behind a kept hole keeps the hole**: the knowledge
/// base's range — its front a hole, its sections reordered at a step — cut at
/// the hole and walked is a build of the result, at the result's position. The
/// K/V holds a hole as the lead of the chunk after it, so a cut that drops
/// that chunk must leave the hole pending, or every position after it reads
/// 100 short of the index.
#[test]
fn a_reorder_right_behind_a_kept_hole_keeps_the_hole() {
    use Place::{Hole, Turn};
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33],
        &[Turn(0), Hole(100), Turn(1), Turn(2)],
        &[Turn(0), Hole(100), Turn(2), Turn(1)],
    );
    assert_eq!((at, offset), (40 + 100 + 33 + 7, 40 + 33 + 7));
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33],
        &[Hole(100), Turn(1), Turn(2)],
        &[Hole(100), Turn(2), Turn(1)],
    );
    assert_eq!((at, offset), (100 + 33 + 7, 33 + 7));
}

/// **A gap between pieces that the next layout fills is gone from the
/// slot** (`docs/placed_section_layout.md` §3.4): a locked section keeps its
/// place across the gap a section that left made, and a section that then
/// fills it is placed where the gap was. The hole was the lead of the chunk
/// after it; a cut that replaced that chunk and carried the lead onto what
/// replaced it read every later position the gap's width past the index —
/// 2,046 positions live on BlackWorld, the QSA select refused and the next
/// wave faulted.
#[test]
fn a_gap_the_next_layout_fills_is_gone() {
    use Place::{Hole, Turn};
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33, 5],
        &[Turn(0), Hole(100), Turn(1), Hole(50), Turn(2)],
        &[Turn(0), Hole(100), Turn(1), Turn(3), Turn(2)],
    );
    assert_eq!((at, offset), (40 + 100 + 7 + 5 + 33, 40 + 7 + 5 + 33));
    // The gap filled by the section that sat after it, moved up.
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33],
        &[Turn(0), Hole(100), Turn(1), Hole(50), Turn(2)],
        &[Turn(0), Hole(100), Turn(1), Turn(2)],
    );
    assert_eq!((at, offset), (40 + 100 + 7 + 33, 40 + 7 + 33));
}

/// **A gap that opens between kept pieces is in the slot**: the section
/// between two locked ones leaves, and its place stays a hole of its width.
#[test]
fn a_gap_that_opens_between_kept_pieces_is_held() {
    use Place::{Hole, Turn};
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33, 5],
        &[Turn(0), Hole(100), Turn(1), Turn(3), Turn(2)],
        &[Turn(0), Hole(100), Turn(1), Hole(5), Turn(2)],
    );
    assert_eq!((at, offset), (40 + 100 + 7 + 5 + 33, 40 + 7 + 33));
}

/// **A hole moves, and is resized, where it stands**: the gap between two
/// locked sections moves down a place, and then grows; every rebuild is a
/// build of its result.
#[test]
fn a_hole_moves_and_is_resized_where_it_stands() {
    use Place::{Hole, Turn};
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33, 5],
        &[Turn(0), Hole(100), Turn(1), Hole(20), Turn(3), Turn(2)],
        &[Turn(0), Hole(100), Turn(1), Turn(3), Hole(20), Turn(2)],
    );
    assert_eq!((at, offset), (40 + 100 + 7 + 5 + 20 + 33, 40 + 7 + 5 + 33));
    let (at, offset) = spliced_layout_equals_built(
        &[40, 7, 33, 5],
        &[Turn(0), Hole(100), Turn(1), Hole(20), Turn(3), Turn(2)],
        &[Turn(0), Hole(100), Turn(1), Hole(64), Turn(3), Turn(2)],
    );
    assert_eq!((at, offset), (40 + 100 + 7 + 64 + 5 + 33, 40 + 7 + 5 + 33));
}

/// Nothing in common and everything in common are the two ends: a rebuild
/// from nothing, and a rebuild that changes nothing.
#[test]
fn nothing_shared_and_everything_shared_are_builds_too() {
    spliced_equals_built(&[40, 7, 33, 64], &[0, 1], &[2, 3]);
    let pushed = spliced_equals_built(&[40, 7, 33], &[0, 1, 2], &[0, 1, 2]);
    assert_eq!(pushed, 0, "nothing changed, so nothing was placed");
}
