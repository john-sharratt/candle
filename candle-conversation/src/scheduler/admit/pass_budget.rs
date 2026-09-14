//! How many tokens one prefill forward may carry.

/// Tokens one prefill forward may carry: the configured target, bounded by what
/// the model can run in one forward.
///
/// **Two numbers for two different reasons, and the narrower wins.** The target
/// is a throughput choice — a wave's fixed cost is paid per forward, so a
/// deployment that keeps many prefills queued raises it (npcd runs 8,192). The
/// model's cap is a correctness bound: the widest wave whose transient tier its
/// geometry prices inside the span, which on a routed model is several times
/// narrower than on a dense one because the expert chain carries
/// `experts_per_tok` rows per token. The target read without the cap sized
/// npcd's world ingest on the routed Qwen3.6-35B-A3B for a 3.3 GB tier its
/// partition did not have, and every turn in the wave failed.
///
/// Never zero: a budget that admits nothing makes no progress.
pub(crate) fn prefill_pass_budget(target: usize, model_cap: usize) -> usize {
    target.min(model_cap).max(1)
}

#[cfg(test)]
mod tests {
    use super::prefill_pass_budget;

    /// **The narrower of the target and the model's cap, and never zero.**
    #[test]
    fn the_pass_budget_is_the_narrower_of_target_and_cap() {
        assert_eq!(
            prefill_pass_budget(8192, 3072),
            3072,
            "a routed model's cap binds"
        );
        assert_eq!(
            prefill_pass_budget(2048, 8192),
            2048,
            "the target binds when it is the narrower"
        );
        assert_eq!(
            prefill_pass_budget(0, 0),
            1,
            "a zero budget would never make progress"
        );
    }
}
