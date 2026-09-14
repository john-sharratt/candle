//! How many tokens one prefill forward may carry.

/// Tokens one prefill forward may carry: the configured target, bounded by what
/// the model can run in one forward, with that bound never below `least` — and
/// never above what the KV side can still back.
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
/// **`least` floors the model's cap — not the target, and not the KV side.** A
/// tier budget that reads zero prices to a single row, and a forward that narrow
/// makes no useful progress, so the cap never falls below the least chunk and the
/// placement is the judge of that chunk. A target below `least` still wins: it is
/// the deployment's own choice. `kv_cap` applies after the floor, because the
/// admit phase claims every chunk a forward will write before it computes
/// anything: a chunk wider than the free regions can back fails partway through
/// claiming, so the floor must never buy past it. `None` when the KV side cannot
/// say.
///
/// Never zero: a budget that admits nothing makes no progress.
pub(crate) fn prefill_pass_budget(
    target: usize,
    model_cap: usize,
    kv_cap: Option<usize>,
    least: usize,
) -> usize {
    target
        .min(model_cap.max(least))
        .min(kv_cap.unwrap_or(usize::MAX))
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::prefill_pass_budget;

    /// **The narrower of the target, the floored cap and the KV side, and never
    /// zero.**
    #[test]
    fn the_pass_budget_is_the_narrower_of_target_floored_cap_and_kv() {
        assert_eq!(
            prefill_pass_budget(8192, 3072, None, 128),
            3072,
            "a routed model's cap binds"
        );
        assert_eq!(
            prefill_pass_budget(2048, 8192, None, 128),
            2048,
            "the target binds when it is the narrower"
        );
        assert_eq!(
            prefill_pass_budget(8192, 1, None, 128),
            128,
            "a cap priced against an empty budget floors at the least chunk"
        );
        assert_eq!(
            prefill_pass_budget(64, 1, None, 128),
            64,
            "a target below the least chunk is still the deployment's choice"
        );
        assert_eq!(
            prefill_pass_budget(8192, 40, Some(40), 128),
            40,
            "the floor never buys past what the KV side can back"
        );
        assert_eq!(
            prefill_pass_budget(8192, 3072, Some(5000), 128),
            3072,
            "room on the KV side does not widen the cap"
        );
        assert_eq!(
            prefill_pass_budget(0, 0, None, 0),
            1,
            "a zero budget would never make progress"
        );
    }
}
