//! Tests that a refusal for lack of room is told apart from a fault.
//!
//! A reprojection refused for want of a full span, a refused tier, or the KV
//! budget is skipped, and its turn decodes on. Any other error still fails the
//! turn. The test pins that the distinction is made, since a refusal that fails a
//! turn would end a conversation over a device that was only briefly full.

use super::*;

#[cfg(test)]
pub(super) mod ground_refusal_tests {
    use super::is_ground_refusal;
    use crate::error::ConversationError;

    /// The device having no room now — a full span, a refused tier, the KV
    /// budget — is told apart from a fault, so a reprojection it refused is
    /// skipped and its turn decodes on, where any fault still fails it.
    #[test]
    fn no_room_now_is_told_apart_from_a_fault() {
        let model = |s: &str| ConversationError::Model(candle::Error::Msg(s.to_string()));
        assert!(is_ground_refusal(&model(
            "slot arena: no region for recurrent state after 20 of 36 slots of 3859968 B — no \
             tier stands, so the span is genuinely full — the weight side is at its own floor"
        )));
        assert!(is_ground_refusal(&model(
            "wave too wide for the partition: short by 16777216 B"
        )));
        assert!(is_ground_refusal(&model(
            "kv-cache GPU VRAM budget exceeded"
        )));
        assert!(!is_ground_refusal(&model(
            "the index disagrees with the K/V it describes"
        )));
        assert!(!is_ground_refusal(&ConversationError::Channel(
            "closed".into()
        )));
    }
}
