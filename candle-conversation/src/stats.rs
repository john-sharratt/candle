use candle_transformers::models::batched_inference::SequenceStats;

/// Why a turn stopped generating.
///
/// What an OpenAI-compatible client reads as `finish_reason`: a reply the
/// response budget cut off is `length`, and a client told `stop` instead takes
/// the truncated reply for a finished one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FinishReason {
    /// The model ended the turn itself with an end-of-sequence token, or the
    /// turn was ended for it — a completed tool-call stencil, a health abort, a
    /// caller that went away.
    #[default]
    Stop,
    /// The turn spent its whole response budget without ending.
    Length,
}

impl FinishReason {
    /// Whether the token just committed ends the turn, and why: `Stop` for an
    /// end-of-sequence token, `Length` once `generated` tokens have used up a
    /// budget of `max_tokens`, `None` to keep decoding. An end-of-sequence
    /// token in the budget's last slot is still a `Stop` — the model finished.
    pub fn after_token(is_eos: bool, generated: usize, max_tokens: usize) -> Option<Self> {
        if is_eos {
            Some(Self::Stop)
        } else if generated >= max_tokens {
            Some(Self::Length)
        } else {
            None
        }
    }
}

/// Statistics for a completed turn.
pub struct TurnStats {
    /// Prefill wall time in milliseconds.
    pub prefill_ms: f64,

    /// Decode wall time in milliseconds.
    pub decode_ms: f64,

    /// Total wall time in milliseconds (prefill + decode + overhead).
    pub total_ms: f64,

    /// Number of tokens generated.
    pub tokens_generated: usize,

    /// Tokens per second during decode phase.
    pub tokens_per_second: f64,

    /// Number of tokens consumed by the prefill phase for this turn
    /// (the full formatted prefill — `no_think_prefix` + user message
    /// + `user_end` + `assistant_start` [+ `/think_block`]).  Combined
    /// with `chunk_size` this lets calibration consumers partition
    /// the turn's per-chunk sig entries into prefill vs decode chunks
    /// without a separate prefill-only capture.
    pub prefill_token_count: usize,

    /// Tokens the model attended to by the end of the turn — the projected
    /// context, this turn's prefill and every token it generated: the decoding
    /// view's length, read before the view is finalized. A count of tokens,
    /// unlike [`SequenceStats::active_tokens`], which adds up every layer's
    /// cache.
    pub context_tokens: usize,

    /// Why the turn stopped generating.
    pub finish: FinishReason,

    /// Represents all the stats for the sequence
    pub sequence: SequenceStats,
}

#[cfg(test)]
mod tests {
    use super::FinishReason;

    #[test]
    fn an_end_of_sequence_token_stops_the_turn() {
        assert_eq!(
            FinishReason::after_token(true, 17, 600),
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn spending_the_budget_without_ending_is_a_length_finish() {
        assert_eq!(
            FinishReason::after_token(false, 600, 600),
            Some(FinishReason::Length)
        );
    }

    #[test]
    fn an_end_of_sequence_in_the_budgets_last_slot_is_still_a_stop() {
        assert_eq!(
            FinishReason::after_token(true, 600, 600),
            Some(FinishReason::Stop)
        );
    }

    #[test]
    fn a_turn_under_budget_without_an_end_keeps_decoding() {
        assert_eq!(FinishReason::after_token(false, 599, 600), None);
    }

    #[test]
    fn a_turn_that_is_never_told_otherwise_stopped() {
        assert_eq!(FinishReason::default(), FinishReason::Stop);
    }
}
