//! How many tokens one prefill forward may carry.

/// Tokens one prefill forward may carry: the configured target, bounded by what
/// the model can run in one forward, with that bound never below `least`.
///
/// **Two numbers for two different reasons, and the narrower wins.** The target
/// is a throughput choice — a wave's fixed cost is paid per forward, so a
/// deployment that keeps many prefills queued raises it (npcd runs 8,192). The
/// model's cap is a correctness bound: the widest forward whose transient tier
/// the model's geometry prices inside the budget the fill published, and that
/// the KV side can still back. On a routed model it is several times narrower
/// than on a dense one, because the expert chain carries `experts_per_tok` rows
/// per token; the target read without it sizes a forward for a tier the
/// partition does not have, and every turn in that forward fails.
///
/// **`least` floors the model's cap, not the target.** A budget that reads zero
/// prices to a single row, and a forward that narrow makes no useful progress,
/// so the cap never falls below the least chunk and the placement is the judge
/// of that chunk. A target below `least` still wins: it is the deployment's own
/// choice.
///
/// Never zero: a budget that admits nothing makes no progress.
pub(crate) fn prefill_pass_budget(target: usize, model_cap: usize, least: usize) -> usize {
    target.min(model_cap.max(least)).max(1)
}

#[cfg(test)]
mod tests {
    use super::prefill_pass_budget;

    /// **The narrower of the target and the floored cap, and never zero.**
    #[test]
    fn the_pass_budget_is_the_narrower_of_target_and_floored_cap() {
        assert_eq!(
            prefill_pass_budget(8192, 3072, 128),
            3072,
            "a routed model's cap binds"
        );
        assert_eq!(
            prefill_pass_budget(2048, 8192, 128),
            2048,
            "the target binds when it is the narrower"
        );
        assert_eq!(
            prefill_pass_budget(8192, 1, 128),
            128,
            "a cap priced against an empty budget floors at the least chunk"
        );
        assert_eq!(
            prefill_pass_budget(64, 1, 128),
            64,
            "a target below the least chunk is still the deployment's choice"
        );
        assert_eq!(
            prefill_pass_budget(0, 0, 0),
            1,
            "a zero budget would never make progress"
        );
    }
}
