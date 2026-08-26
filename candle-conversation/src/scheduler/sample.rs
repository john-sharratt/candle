use super::*;

impl Scheduler {
    /// Sample a single sequence using the batched sampler.
    pub(super) fn sample_single(
        &self,
        logits: &Tensor,
        config: &SamplingConfig,
        state: &mut SequenceSamplingState,
    ) -> Result<u32, ConversationError> {
        let tokens = self
            .sampler
            .sample_batch(logits, &mut [state], &[config])
            .map_err(ConversationError::Model)?;

        tokens
            .into_iter()
            .next()
            .ok_or_else(|| ConversationError::Channel("no token sampled".into()))
    }

    /// Check if a token is the EOS token.
    pub(super) fn is_eos(&self, token: u32) -> bool {
        self.eos_tokens.contains(&token)
    }
}
