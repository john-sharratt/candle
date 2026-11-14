//! What the gap quantizes one sequence's complete blocks under
//! (`docs/kv_quantize_in_the_gap.md` §3.1).

use candle_nn::kv_cache::CompressionPolicy;

/// What the gap quantizes one sequence's complete blocks under. Each layer
/// takes the chosen policy through
/// [`BatchedInferenceSession::layer_seal_policies`](super::BatchedInferenceSession::layer_seal_policies),
/// so a capped layer keeps its cap.
#[derive(Debug, Clone, Copy)]
pub enum GapSeal {
    /// The session's seal policy.
    Session,
    /// The sequence's conversation's own policy — its level and forced
    /// formats.
    Policy(CompressionPolicy),
    /// Left native: a lossless conversation, which keeps its turns' native
    /// K/V, or a section, whose seal quantizes it under the section's own
    /// policy once its build has attended.
    Native,
}

impl GapSeal {
    /// The policy the gap quantizes under, given the session's, or `None`
    /// when the sequence is left native.
    pub fn policy(&self, session: &CompressionPolicy) -> Option<CompressionPolicy> {
        match self {
            GapSeal::Session => Some(*session),
            GapSeal::Policy(p) => Some(*p),
            GapSeal::Native => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seal_resolves_to_its_policy_or_none() {
        let session = CompressionPolicy::new(5);
        let own = CompressionPolicy::new(9);
        assert_eq!(
            GapSeal::Session
                .policy(&session)
                .map(|p| p.compression_level),
            Some(5)
        );
        assert_eq!(
            GapSeal::Policy(own)
                .policy(&session)
                .map(|p| p.compression_level),
            Some(9)
        );
        assert!(GapSeal::Native.policy(&session).is_none());
    }
}
