//! Whether a token the stencil dropped re-opens the reasoning segment.
//!
//! A steered think span drops the model's own `</think>` (or an EOS sampled
//! mid-thought) and keeps the block open; the sampler flipped its segment flag
//! off on that close, so the decode loop re-arms it. But a drop is also a
//! delimiter the grammar writes in the model's place, or an EOS intercepted
//! inside a call's value — after the block has closed. Re-armed on those, the
//! call was treated as reasoning again, and the close budget forced a
//! `</think>` into the middle of its arguments.

/// Whether dropping `token` re-opens the segment: it was the segment's own
/// close (`close`, negative when the checkpoint has none), or the row was still
/// inside the segment when it was sampled.
pub(super) fn drop_rearms_segment(token: u32, close: i32, in_segment: bool) -> bool {
    in_segment || (close >= 0 && token == close as u32)
}

#[cfg(test)]
mod tests {
    use super::drop_rearms_segment;

    const THINK_CLOSE: i32 = 151668;
    const EOS: u32 = 151645;
    const BRACKET: u32 = 60;

    #[test]
    fn only_a_dropped_close_of_the_segment_reopens_it() {
        // The model's own `</think>`, suppressed: the sampler had flipped the
        // flag off, and the block goes on.
        assert!(drop_rearms_segment(THINK_CLOSE as u32, THINK_CLOSE, false));
        // An EOS dropped while still reasoning: the block goes on.
        assert!(drop_rearms_segment(EOS, THINK_CLOSE, true));
        // After the block closed: a dropped delimiter or an intercepted EOS
        // inside a call is not reasoning.
        assert!(!drop_rearms_segment(BRACKET, THINK_CLOSE, false));
        assert!(!drop_rearms_segment(EOS, THINK_CLOSE, false));
        // A checkpoint with no close token re-arms only from inside.
        assert!(!drop_rearms_segment(THINK_CLOSE as u32, -1, false));
    }
}
