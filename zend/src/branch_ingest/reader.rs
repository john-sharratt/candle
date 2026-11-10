//! Whose reading an ingest conversation is, read from its first turn.
//!
//! A reading replicates as it is written: its turns, then its content key once
//! the chain is done. Another member therefore holds every reading in flight
//! on its peers. A peer's reading begun and not yet keyed looks exactly like a
//! crash's debris, and one keyed whose later turns are still on their way
//! looks exactly like a chain that stopped part-way. A member that judged
//! either as its own to clear tombstoned the peer's live work, and the
//! tombstone replicated back. On 2026-11-10 WhiteWorld's boot sweep retired
//! ten of BlackWorld's in-flight readings, which then failed on BlackWorld
//! with their index pages gone. WhiteWorld's passes also re-queued files
//! BlackWorld had just committed, because their reply turns had not arrived.
//!
//! So each member settles only what it began itself. The first turn's record
//! names the member that wrote it (`VersionStamp::origin`) and when
//! (`VersionStamp::hlc`). A peer's reading is left to the peer for
//! [`PEER_GRACE`], which is time for its turns to arrive. Past that it is
//! settled wherever it is found, so a member that stopped for good leaves no
//! debris behind it.

use std::time::Duration;

use candle_conversation::persistence::identity::{now_ms, wall_ms, VersionStamp};
use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

/// How long a peer's reading is left to the peer: well past the minutes a
/// reading takes, and the time its turns take to replicate.
pub const PEER_GRACE: Duration = Duration::from_secs(30 * 60);

/// Who settles an ingest conversation: retires it if it never committed,
/// rebuilds it if its chain stopped part-way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Whose {
    /// This member: it began the reading, or the peer that did has not
    /// finished it within [`PEER_GRACE`].
    Ours,
    /// A peer that began it within [`PEER_GRACE`]: its turns may still be on
    /// their way.
    Peers,
    /// No turn yet, so nothing says who began it. It holds no turn, so it
    /// takes no part in any gather, and is left alone.
    Unknown,
}

/// Whose a conversation is, from its first turn's stamp, this member's node
/// and the wall clock now.
pub fn whose(first_turn: Option<VersionStamp>, node: u64, now_ms: u64) -> Whose {
    let Some(stamp) = first_turn else {
        return Whose::Unknown;
    };
    if stamp.origin == node {
        return Whose::Ours;
    }
    let age_ms = now_ms.saturating_sub(wall_ms(stamp.hlc));
    if u128::from(age_ms) >= PEER_GRACE.as_millis() {
        Whose::Ours
    } else {
        Whose::Peers
    }
}

/// [`whose`] for `timeline` on `engine`, now.
pub fn whose_reading(engine: &ConversationEngine, timeline: TimelineId) -> Whose {
    let conv = engine.conversation();
    let first = conv.read().first_turn_stamp(timeline.raw());
    whose(first, conv.node(), now_ms())
}

/// Whether this member settles `timeline` — retires it unkeyed, rebuilds it
/// broken. Never while a turn of it is a gap: those records are still on
/// their way from the member that wrote them, and a gap reads as an empty turn,
/// which a chain check takes for a turn that answered nothing. Under a lagging
/// replica that is no brief window: on 2026-11-10 WhiteWorld, minutes behind
/// BlackWorld's re-ingest, held readings hollow past [`PEER_GRACE`], rebuilt
/// them, and so added to the very traffic it was behind on. Otherwise
/// [`whose_reading`] decides.
pub fn settles_here(engine: &ConversationEngine, timeline: TimelineId) -> bool {
    !engine.conversation().read().holds_gap(timeline)
        && whose_reading(engine, timeline) == Whose::Ours
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::persistence::identity::COUNTER_BITS;

    const HERE: u64 = 0xa0ac_8817_8cb2_b57d;
    const PEER: u64 = 0x8939_59be_acad_c372;
    const NOW_MS: u64 = 1_794_313_193_436;

    fn stamp(origin: u64, at_ms: u64) -> VersionStamp {
        VersionStamp {
            gen: 1,
            epoch: 0,
            hlc: at_ms << COUNTER_BITS,
            origin,
        }
    }

    #[test]
    fn a_reading_this_member_began_is_ours_at_any_age() {
        assert_eq!(whose(Some(stamp(HERE, NOW_MS)), HERE, NOW_MS), Whose::Ours);
        assert_eq!(whose(Some(stamp(HERE, 0)), HERE, NOW_MS), Whose::Ours);
    }

    #[test]
    fn a_peers_reading_is_the_peers_inside_the_grace() {
        let a_minute_ago = NOW_MS - 60_000;
        assert_eq!(
            whose(Some(stamp(PEER, a_minute_ago)), HERE, NOW_MS),
            Whose::Peers
        );
        let just_inside = NOW_MS - (30 * 60_000 - 1);
        assert_eq!(
            whose(Some(stamp(PEER, just_inside)), HERE, NOW_MS),
            Whose::Peers
        );
    }

    #[test]
    fn a_peers_reading_past_the_grace_is_ours() {
        let at_the_grace = NOW_MS - 30 * 60_000;
        assert_eq!(
            whose(Some(stamp(PEER, at_the_grace)), HERE, NOW_MS),
            Whose::Ours
        );
    }

    #[test]
    fn a_stamp_from_before_identities_is_ours() {
        assert_eq!(
            whose(Some(VersionStamp::default()), HERE, NOW_MS),
            Whose::Ours
        );
    }

    #[test]
    fn a_reading_with_no_turn_is_unknown() {
        assert_eq!(whose(None, HERE, NOW_MS), Whose::Unknown);
    }

    #[test]
    fn a_peers_clock_ahead_of_this_one_is_still_the_peers() {
        let ahead = NOW_MS + 5_000;
        assert_eq!(whose(Some(stamp(PEER, ahead)), HERE, NOW_MS), Whose::Peers);
    }
}
