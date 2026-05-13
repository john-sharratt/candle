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

    /// Sample a batch of sequences using the batched sampler.
    pub(super) fn sample_batch_from_logits(
        &self,
        logits_vec: &[Tensor],
        states: &mut [&mut SequenceSamplingState],
        configs: &[&SamplingConfig],
    ) -> Result<Vec<u32>, ConversationError> {
        // Stack logits into a single tensor for batched sampling
        // Each logits tensor should be [1, vocab_size] or [seq_len, vocab_size]
        // We take the last position of each
        let last_logits: Vec<Tensor> = logits_vec
            .iter()
            .map(|logits| {
                match logits.dims().len() {
                    1 => Ok(logits.clone()),
                    2 => {
                        let seq_len = logits.dim(0)?;
                        logits.i(seq_len - 1)
                    }
                    3 => {
                        // [batch, seq_len, vocab]
                        let seq_len = logits.dim(1)?;
                        logits.i((.., seq_len - 1, ..))?.squeeze(0)
                    }
                    n => Err(candle::Error::Msg(format!("unexpected logits rank: {n}"))),
                }
            })
            .collect::<candle::Result<Vec<_>>>()
            .map_err(ConversationError::Model)?;

        // Stack into [batch_size, vocab_size]
        let stacked = Tensor::stack(&last_logits, 0).map_err(ConversationError::Model)?;

        self.sampler
            .sample_batch(&stacked, states, configs)
            .map_err(ConversationError::Model)
    }

    /// Check if a token is the EOS token.
    pub(super) fn is_eos(&self, token: u32) -> bool {
        self.eos_tokens.contains(&token)
    }
}
