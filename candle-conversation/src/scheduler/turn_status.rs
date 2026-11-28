//! Whether a resident gallery stream still stands — a turn's conversation, or
//! a knowledge-base section's declaration — read off the substrates, for the
//! memory report's census and the gallery sweep alike.

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

/// The resident stream `sid` as `substrate` holds it: its standing and, for a
/// turn, the group it was written to. `None` when this substrate does not
/// declare it.
///
/// **Every declared stream stands, not only a turn.** The gallery also holds
/// the knowledge base's sections, which are `PromptSection` streams; a stream
/// that is not a turn stands while its declaration does. A tombstoned section
/// has none, so it reads `None` here and the sweep gives it up as absent.
/// Reading only turns judged every section absent, so the sweep gave up the
/// whole knowledge base each time a tombstone epoch moved and the next scan
/// uploaded all of it again — 20,547 sections, ~36 s on the wave thread, every
/// ~40 s of a calibration sweep (2026-11-28).
pub fn turn_in(substrate: &Substrate, sid: StreamId) -> Option<(TurnStatus, Option<GroupId>)> {
    let StreamDecl::Turn(t) = substrate.stream_of(sid)?.decl.as_ref()? else {
        return Some((TurnStatus::Live, None));
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
    use crate::persistence::content_hash::{hash_bytes, turn_stream_id, ContentHash};
    use crate::persistence::streams::ContentAddress;
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

    /// **A knowledge-base section stands while it is declared**, so the
    /// gallery sweep keeps it; tombstoned, it is declared no more and reads
    /// absent.
    #[test]
    fn a_declared_section_reads_live_until_tombstoned() {
        let conv = Conversation::ephemeral();
        let address = ContentAddress {
            prefix_hash: ContentHash { lo: 9, hi: 0 },
            section_hash: hash_bytes(b"a file's section"),
        };
        let sid = conv
            .declare_section_stream(address, "shared/files/candle/a.rs@b#1-10")
            .expect("declare");
        assert_eq!(turn_in(&conv.read(), sid), Some((TurnStatus::Live, None)));
        conv.tombstone_section(sid, None).unwrap();
        assert_eq!(turn_in(&conv.read(), sid), None, "tombstoned");
    }
}
