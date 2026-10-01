//! How many tokens one prefill forward may carry.

use candle_nn::kv_cache::CHUNK_SIZE;

/// Tokens the KV side's free ground can still back, from the bytes it has free
/// and what one 32-token block costs.
///
/// **Three units, and only one of them is tokens.** The KV side counts
/// *regions* (16 MiB each), admission prices *bytes*, and this budget is in
/// *tokens*; nothing converts between them implicitly. The chain is
/// bytes → whole blocks → tokens, and the block is the middle step because a
/// block is what the allocator actually claims — a quantized block is not
/// generally divisible by its element count, so tokens-per-byte is not a
/// meaningful quantity (see `per_block_kv_bytes`).
///
/// `None` when `per_block` is zero: a model whose geometry prices a block at
/// nothing has no KV to be bounded by, and dividing by it would be the same
/// unbounded answer stated as a very large number.
///
/// Rounds *down*, and the bound is what a perfect packing could back. Whole
/// regions are carved into arenas per size class, so real placement fits
/// somewhat less; this is a ceiling on a ceiling, and the claim path is still
/// the judge of an individual chunk.
pub(crate) fn kv_token_cap(free_bytes: usize, per_block: u64) -> Option<usize> {
    if per_block == 0 {
        return None;
    }
    let blocks = free_bytes as u64 / per_block;
    Some(
        blocks
            .saturating_mul(CHUNK_SIZE as u64)
            .min(usize::MAX as u64) as usize,
    )
}

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
    use super::{kv_token_cap, prefill_pass_budget};
    use candle_nn::kv_cache::CHUNK_SIZE;

    /// **The cap is in tokens, and it is reached from bytes through whole
    /// blocks.**
    ///
    /// The caller used to hand `prefill_pass_budget` a *region count* multiplied
    /// by `CHUNK_SIZE` — a region is 16 MiB, so that was neither bytes nor
    /// tokens, and it read as tokens. The numbers here are the real chain, with
    /// a block priced as Qwen3-30B-A3B's active formats do (48 layers × 4 KV
    /// heads × 128 head_dim × (64 + 64) B/block = 1,572,864 B).
    #[test]
    fn the_kv_cap_converts_bytes_through_whole_blocks_to_tokens() {
        let per_block = 1_572_864u64;
        // Exactly ten blocks' worth of ground backs ten blocks of tokens.
        assert_eq!(
            kv_token_cap(10 * per_block as usize, per_block),
            Some(10 * CHUNK_SIZE)
        );
        // A partial block at the end backs no tokens — the allocator claims
        // whole blocks, so rounding up would promise ground that is not there.
        assert_eq!(
            kv_token_cap(10 * per_block as usize + 1, per_block),
            Some(10 * CHUNK_SIZE)
        );
        assert_eq!(
            kv_token_cap(per_block as usize - 1, per_block),
            Some(0),
            "less than one block backs nothing"
        );
        // The old expression's scale, for the record, and it erred toward
        // throttling rather than over-promising: 400 free regions is 6.4 GiB,
        // which really backs 136,512 tokens at this price (4,266 whole blocks),
        // where `regions × CHUNK_SIZE` claimed 12,800 — tight by 10.7×. A cap
        // that reads an order of magnitude low is a prefill throttle nobody
        // asked for, which is why it never showed up as a failure.
        let free = 400 * 16 * 1024 * 1024;
        assert_eq!(kv_token_cap(free, per_block), Some(136_512));
        assert_eq!(
            kv_token_cap(free, 0),
            None,
            "a model that prices a block at nothing bounds nothing"
        );
    }

    /// A KV side that can back less than the floor still binds — the whole point
    /// of applying `kv_cap` after `least`.
    #[test]
    fn a_kv_cap_below_the_least_chunk_still_binds() {
        let per_block = 1_572_864u64;
        let cap = kv_token_cap(2 * per_block as usize, per_block).expect("priced");
        assert_eq!(cap, 64);
        assert_eq!(prefill_pass_budget(8192, 3072, Some(cap), 128), 64);
    }

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
