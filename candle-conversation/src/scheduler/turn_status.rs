//! Whether a resident gallery turn's conversation still stands — read off the
//! substrates, for the memory report's census and the gallery sweep alike.

use serde::Serialize;

use crate::persistence::streams::{StreamDecl, StreamId};
use crate::projection::{GroupId, Substrate, TimelineId};

/// Whether a resident turn's conversation still stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    /// Its timeline and the turn itself stand.
    Live,
    /// Its timeline, or the turn alone, is tombstoned: no scan reads it again.
    Tombstoned,
    /// No substrate this engine reads declares its stream (or its substrate's
    /// lock was held when the census looked).
    Absent,
}

/// The turn on `sid` as `substrate` holds it: its standing and the group it
/// was written to. `None` when this substrate does not declare it.
pub fn turn_in(substrate: &Substrate, sid: StreamId) -> Option<(TurnStatus, Option<GroupId>)> {
    let Some(StreamDecl::Turn(t)) = substrate.stream_of(sid).and_then(|s| s.decl.as_ref()) else {
        return None;
    };
    let dead = TimelineId::from_raw(t.timeline_id).is_some_and(|tl| {
        substrate.is_tombstoned(tl) || substrate.is_turn_tombstoned(tl, t.turn_index)
    });
    let status = if dead {
        TurnStatus::Tombstoned
    } else {
        TurnStatus::Live
    };
    Some((status, GroupId::from_raw(t.group_id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::content_hash::turn_stream_id;
    use crate::projection::{Conversation, LayerId};
    use crate::substrate::TurnPartWrite;
    use crate::turn::Role;

    /// **A turn is live until its timeline or the turn itself is tombstoned,
    /// and absent from a substrate that never declared it.**
    #[test]
    fn a_turn_reads_live_then_tombstoned_and_absent_elsewhere() {
        let conv = Conversation::ephemeral();
        let tl = TimelineId::from_raw(71).unwrap();
        let (layer, group) = (LayerId::from_raw(1).unwrap(), GroupId::from_raw(4).unwrap());
        conv.register_timeline(tl, layer, group);
        for _ in 0..2 {
            conv.record_turn(
                tl,
                Role::User,
                TurnPartWrite {
                    token_count: 1,
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record_turn");
        }
        let (first, second) = (turn_stream_id(tl.raw(), 0), turn_stream_id(tl.raw(), 1));
        let status = |sid| turn_in(&conv.read(), sid).map(|(s, _)| s);
        assert_eq!(
            turn_in(&conv.read(), first),
            Some((TurnStatus::Live, Some(group)))
        );
        conv.tombstone_turn(tl, 1).unwrap();
        assert_eq!(status(first), Some(TurnStatus::Live));
        assert_eq!(
            status(second),
            Some(TurnStatus::Tombstoned),
            "the turn alone"
        );
        conv.tombstone_timeline(tl).unwrap();
        assert_eq!(
            status(first),
            Some(TurnStatus::Tombstoned),
            "the whole timeline"
        );
        assert_eq!(
            turn_in(&Conversation::ephemeral().read(), first),
            None,
            "a substrate that never declared it"
        );
    }
}
