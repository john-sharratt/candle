//! A turn's own banned tokens, joined to the sampling it decodes under.
//!
//! **`TurnOptions::turn_banned_tokens` was read by nothing.** Documented as
//! merged with the conversation's list, it was dropped where the turn's
//! sampling was resolved, so no caller's ban ever reached the sampler — not a
//! conversation's `<think>`, not its call marker, not the braces that keep a
//! turn of plain words from writing a call. A person was answered with
//! `{"name": "message", "arguments": {...}}` under all three.

use crate::config::SamplingConfig;

/// `sampling` with `banned` added to what it already bans, each id once.
pub(super) fn with_turn_bans(mut sampling: SamplingConfig, banned: &[i32]) -> SamplingConfig {
    for &id in banned {
        if !sampling.banned_tokens.contains(&id) {
            sampling.banned_tokens.push(id);
        }
    }
    sampling
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A turn's bans reach the sampler beside the conversation's**, and an
    /// id banned by both is banned once.
    #[test]
    fn a_turns_bans_join_the_conversations() {
        let mut base = SamplingConfig::argmax();
        base.banned_tokens = vec![7, 9];
        let joined = with_turn_bans(base, &[9, 11, 13]);
        assert_eq!(joined.banned_tokens, [7, 9, 11, 13]);
    }

    /// No bans of its own leave the sampling as it was.
    #[test]
    fn no_turn_bans_change_nothing() {
        let mut base = SamplingConfig::argmax();
        base.banned_tokens = vec![7];
        assert_eq!(with_turn_bans(base, &[]).banned_tokens, [7]);
    }
}
