use candle_transformers::models::batched_inference::SequenceStats;

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

    /// Represents all the stats for the sequence
    pub sequence: SequenceStats,
}
