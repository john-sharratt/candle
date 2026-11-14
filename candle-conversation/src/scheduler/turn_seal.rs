//! What the gap quantizes a slot's blocks under (`docs/kv_quantize_in_the_gap.md`
//! §3.1). The gap quantizes a turn's K/V while the turn still runs, so it must
//! use the policy the turn's own seal would: its conversation's — a level,
//! forced formats, or none at all for a lossless conversation. A section is
//! left native: it attends its own build first, and its seal quantizes it under
//! the section's policy.

use candle_nn::kv_cache::CompressionPolicy;
use candle_transformers::models::batched_inference::GapSeal;

use super::*;
use crate::persistence::thread::effective_turn_policy;
use crate::substrate::ConvCompression;

/// The seal a turn takes from its conversation's override over the session's
/// policy: that policy, or native where the conversation is lossless or the
/// session quantizes nothing.
pub(super) fn turn_seal(
    session: Option<&CompressionPolicy>,
    cc: Option<ConvCompression>,
) -> GapSeal {
    match effective_turn_policy(session, cc) {
        Some(p) => GapSeal::Policy(p),
        None => GapSeal::Native,
    }
}

impl Scheduler {
    /// Quantize `slot`'s turn under its conversation's policy. The
    /// conversation is the one the turn is filed under — a view's parent's —
    /// and a slot with none (a raw path) keeps the session's.
    pub(super) fn seal_turn_under_its_conversation(&mut self, slot: SequenceId) {
        let owner = self.turn_views.get(&slot).map_or(slot, |v| v.parent_id);
        self.seal_under_conversation_of(slot, owner);
    }

    /// Quantize `slot`'s blocks under the policy of the conversation `owner`
    /// is registered to — the slot itself, or the parent a view is carved
    /// from. Nothing for an owner with no conversation target.
    pub(super) fn seal_under_conversation_of(&mut self, slot: SequenceId, owner: SequenceId) {
        let (Some(conversation), Some(target)) = (
            self.slot_conversations.get(&owner),
            self.slot_targets.get(&owner),
        ) else {
            return;
        };
        let cc = conversation.read().timeline_compression(target.timeline);
        let seal = turn_seal(self.session.seal_policy().as_ref(), cc);
        self.set_seal(slot, seal);
    }

    /// Leave `slot`'s blocks native for the gap: a section ingest.
    pub(super) fn seal_section_natively(&mut self, slot: SequenceId) {
        self.set_seal(slot, GapSeal::Native);
    }

    fn set_seal(&mut self, slot: SequenceId, seal: GapSeal) {
        if let Err(e) = self.session.set_sequence_seal(slot.0, seal) {
            tracing::warn!("gap seal for slot {}: {e}", slot.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc() -> ConvCompression {
        ConvCompression {
            lossless: false,
            level: None,
            disable_k_override: false,
            force_k: None,
            force_v: None,
        }
    }

    #[test]
    fn a_lossless_conversation_stays_native() {
        let base = CompressionPolicy::new(5);
        let lossless = ConvCompression {
            lossless: true,
            ..cc()
        };
        assert!(matches!(
            turn_seal(Some(&base), Some(lossless)),
            GapSeal::Native
        ));
    }

    #[test]
    fn a_conversations_level_and_forced_formats_are_its_turns() {
        let base = CompressionPolicy::new(5);
        let own = ConvCompression {
            level: Some(9),
            force_v: Some(QuantFormat::Q8_0),
            ..cc()
        };
        let GapSeal::Policy(p) = turn_seal(Some(&base), Some(own)) else {
            panic!("a policy of its own")
        };
        assert_eq!(p.compression_level, 9);
        assert_eq!(p.override_v_quant, Some(QuantFormat::Q8_0));
    }

    #[test]
    fn no_override_is_the_session_policy_and_no_policy_is_native() {
        let base = CompressionPolicy::new(5);
        let GapSeal::Policy(p) = turn_seal(Some(&base), None) else {
            panic!("the session's policy")
        };
        assert_eq!(p.compression_level, 5);
        assert!(matches!(turn_seal(None, None), GapSeal::Native));
    }
}
