//! How many tokens one prefill forward may carry.

/// Tokens one prefill forward may carry: the configured target, bounded by what
/// the model can run in one forward.
///
/// **Two numbers for two different reasons, and the narrower wins.** The target
/// is a throughput choice — a wave's fixed cost is paid per forward, so a
/// deployment that keeps many prefills queued raises it (npcd runs 8,192). The
/// model's cap is a correctness bound: the widest forward the model runs — where
/// compute saturates, and what the KV side can still back, since the admit phase
/// claims every chunk a wave will write before it computes anything.
///
/// The transient tier is **not** this bound's to price. Each wave's tier is
/// priced against the budget the fill published, beside the head the wave
/// actually carries, by the rows handed to its prefills (`build_section_batch`,
/// `form_wave_group`). A pass budget read against that same live budget
/// collapsed to one token whenever it read zero.
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
