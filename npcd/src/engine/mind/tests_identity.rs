//! Tests the frame fingerprint and the day's conversation name that a restart
//! depends on.
//!
//! A restart rejoins a conversation only when its fingerprint matches, so the
//! same frame must fingerprint the same way in a process that never saw the
//! one that wrote it, and every difference that changes a character, its acts
//! or its prompt must change the answer. A fingerprint built from a pointer, an
//! ordering or a timestamp would fail by quietly never resuming. The day's
//! conversation name must also survive a restart, so a second timeline opened
//! mid-day still belongs to the same character.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// The fingerprint is what decides whether a conversation on disk is still
    /// this character's to continue, so the same frame has to produce the same
    /// answer in a process that never saw the one that wrote it. A hash of the
    /// text does; anything carrying a pointer, an ordering or a timestamp would
    /// not, and would fail by quietly never resuming.
    #[test]
    fn the_same_frame_fingerprints_the_same_in_any_process() {
        let frame = "You are Ada. You may move_to, say, ask.";
        assert_eq!(frame_fingerprint(frame), frame_fingerprint(frame));
        assert_eq!(
            frame_fingerprint(frame).len(),
            64,
            "not a sha256 hex digest"
        );
    }

    /// **Every difference has to count.** A character rejoining a conversation
    /// written under a prompt it no longer has reads its own past making calls
    /// in a shape it can no longer make. The cases below are the ones that
    /// actually change between builds: who the character is, and which acts
    /// exist.
    #[test]
    fn a_changed_frame_fingerprints_differently() {
        let base = "You are Ada. You may move_to, say, ask.";
        for changed in [
            "You are Bram. You may move_to, say, ask.",
            "You are Ada. You may move_to, say, ask, gesture.",
            "You are Ada. You may move_to, say, ask. ",
            "",
        ] {
            assert_ne!(
                frame_fingerprint(base),
                frame_fingerprint(changed),
                "{changed:?} fingerprinted the same as the frame it differs from"
            );
        }
    }

    /// A restart mid-day opens a second timeline, and it must carry the same
    /// day's name so both halves stay attributable to the character.
    #[test]
    fn a_days_conversation_id_is_stable_across_a_restart() {
        assert_eq!(conversation_id(4, 9, 1), conversation_id(4, 9, 1));
        assert_ne!(conversation_id(4, 9, 1), conversation_id(4, 10, 1));
        assert_ne!(
            conversation_id(4, 9, 1),
            conversation_id(4, 9, 2),
            "a mission handed in starts the next"
        );
    }
}
