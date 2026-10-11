//! What a turn tombstone names (`docs/replicated_substrate.md` §6.5, §6.6).
//!
//! A turn is addressed by its index in the timeline, but an index is a
//! position, and a position can move: two writers' turn *N* are placed by
//! their stamps, so a turn written at *N* can end up at *N + 1* once another
//! writer's turn arrives below it. A tombstone that named the position would
//! then retire the wrong turn. So a turn tombstone names the turn as its
//! writer wrote it — the index the writer gave it and the writer's origin —
//! which no later arrival changes.

use crate::projection::TimelineId;

/// One retired turn: its timeline, the index its writer gave it, and the
/// writer. A turn with no identity, written before identities existed, has
/// writer zero, and its index is its position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TurnMark {
    pub timeline: TimelineId,
    pub turn_index: u32,
    pub writer: u64,
}
