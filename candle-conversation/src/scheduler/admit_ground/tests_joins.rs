//! Tests of when a turn may join a forward and of which chunk a long turn
//! reaches deepest.
//!
//! A blocked section joins a full forward whatever its band, and a file's bulk
//! section does not. A full forward takes no join beside its members, and a
//! closed forward offers only joins even when it is narrower than a chunk. The
//! deepest chunk of a long turn ends at its last token, so the price of the
//! turn's deepest forward bounds every chunk it will run.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::{deepest_chunk_offset, joins_a_full_forward, room_left, CHUNK_SIZE};

    use crate::projection::DecodePriority::{High, Low, Normal, Urgent};

    /// A blocked turn's section joins whatever its band; a file's does not.
    #[test]
    fn a_short_or_live_section_joins_a_full_forward_and_bulk_does_not() {
        assert!(joins_a_full_forward(Urgent, 100_000));
        assert!(joins_a_full_forward(High, 100_000));
        assert!(joins_a_full_forward(Low, 60));
        assert!(joins_a_full_forward(Normal, 256));
        assert!(!joins_a_full_forward(Normal, 257));
        assert!(!joins_a_full_forward(Low, 6_968));
    }

    #[test]
    fn a_closed_forward_offers_only_joins_even_below_a_chunk() {
        // Nothing standing, the rate refused the first rows offered: the pass
        // closes at width zero and composes at one chunk.
        assert_eq!(room_left(CHUNK_SIZE, 0, false, 0), Some(CHUNK_SIZE));
        assert_eq!(
            room_left(CHUNK_SIZE, 0, true, 0),
            Some(0),
            "a closed pass re-offers the turn as a join, not with the rows it was refused"
        );
        // With a member already in that chunk, there is no room for a join.
        assert_eq!(room_left(CHUNK_SIZE, 0, true, 1), None);
        // Open, a wider forward still has its unspent rows.
        assert_eq!(
            room_left(4 * CHUNK_SIZE, CHUNK_SIZE, false, 1),
            Some(3 * CHUNK_SIZE)
        );
    }

    /// **A full forward takes no turn beside its members**, however many more
    /// chunks it is wide: filled in order, the join would advance by nothing
    /// while holding its store.
    #[test]
    fn a_full_forward_takes_no_join() {
        assert_eq!(room_left(64 * CHUNK_SIZE, 64 * CHUNK_SIZE, false, 2), None);
        assert_eq!(room_left(64 * CHUNK_SIZE, 64 * CHUNK_SIZE, true, 2), None);
        assert_eq!(
            room_left(64 * CHUNK_SIZE, 63 * CHUNK_SIZE, false, 2),
            Some(CHUNK_SIZE)
        );
    }

    #[test]
    fn the_deepest_chunk_ends_at_the_turns_last_token() {
        // A 128,000-token turn fed 8,192 rows at a time from a fresh sequence:
        // its deepest full-width chunk starts at 119,808 and ends at 128,000.
        assert_eq!(deepest_chunk_offset(0, 128_000, 8_192), 119_808);
        // The same turn appended behind 4,096 standing tokens.
        assert_eq!(deepest_chunk_offset(4_096, 128_000, 8_192), 123_904);
    }

    #[test]
    fn a_turn_no_wider_than_a_chunk_is_its_own_deepest() {
        assert_eq!(deepest_chunk_offset(0, 4_096, 8_192), 0);
        assert_eq!(deepest_chunk_offset(300, 8_192, 8_192), 300);
    }
}
