//! The batched-forward gate's configuration ladder, shared with the combined probe.
//!
//! The ladder runs the BF16 widths and the compression rungs in a fixed order, so
//! its table is one measurement with one variable changed. C5 is the operating
//! point that zend runs, and C10 is a probe placed just inside its breaking edge, so
//! a failure at C10 means the thresholds have drifted and the fix is the factor row
//! rather than a wider tolerance. The ×16 rung is measured only on a card with
//! 24 GiB or more, because below that its per-session state does not fit. A card
//! that skips it says so, so reduced coverage is never read as a pass.

use super::*;

/// Load the reference (oracle) model from the pinned split's first shard
/// path. Refuses a non-`qwen4exp` file and validates the frozen geometry —
/// a silent architecture change fails here, not in a kernel.
pub fn oracle_from_gguf_path(
    first_shard: &std::path::Path,
    device: &Device,
) -> Result<Qwen4ExpModel> {
    let model = load_oracle_model(first_shard, device)?;
    let cfg = &model.cfg;
    if cfg.num_layers != 48
        || cfg.attn_head_dim != 256
        || cfg.hidden_size != 2560
        || cfg.moe.n_experts != 512
        || cfg.hc.count != 4
    {
        candle::bail!(
            "quantized_qwen38_moe: checkpoint geometry diverged from the frozen schema \
             (layers {}, head_dim {}, hidden {}, experts {}, hc {})",
            cfg.num_layers,
            cfg.attn_head_dim,
            cfg.hidden_size,
            cfg.moe.n_experts,
            cfg.hc.count
        );
    }
    Ok(model)
}

/// **The gate's config ladder** — the BF16 widths and the C rungs, in the
/// order they are read.
///
/// **Public, and outside the test module, so the same ladder can be driven
/// from above.** `candle-conversation`'s combined probe runs these rows on the
/// model the engine loads, then appends the engine's own rows to the same
/// table; a copy there would drift. Shared too by the plain gate and the
/// speculative one, so the two tables are the same measurement with one
/// variable changed.
///
/// The DeepSeek gate's shape — cold ×1, the batched widths, then a warm ×1
/// to read steady state — followed by:
///
/// **The C-ladder.** Compressed KV. `QWEN4EXP_KV_FACTORS` is the row these
/// calibrate, and the two ends do different jobs: **C5 is the operating
/// point** (what zend runs) and **C10 is the probe** — tuned to pass just
/// under its breaking edge so the row can be placed with knowledge of how
/// much edge is left. A red C10 means the thresholds drifted past it, and
/// the fix is the factor row rather than a widened tolerance.
///
/// Two things are unusual here and are why this row could not be inherited
/// from the 3.6 sibling. Only **12 of 48** layers hold K/V, so a level's
/// error touches a quarter of the stack; and QSA caps the read at 2051
/// cells, so above that depth a block's error reaches the output only if
/// the indexer selects it. These rungs run at gate depth (~713 tokens),
/// where selection is the identity — the row they produce is a bound for
/// deeper contexts, not a measurement of them.
///
/// C5 rather than C4 for the middle rung: it is the level zend actually
/// runs (`session.rs` sets `compression_level(5)`), so the gate covers the
/// operating point directly instead of leaving it to be inferred between
/// two rungs either side of it. **C5 at ×8 as well as ×2**: ×8 is the row the
/// engine probe's clean baseline reproduces through the engine, so the two sit
/// side by side in the combined table. C10 at two widths rather than one: the
/// pair is what shows the top rung holding as the cohort grows, which is
/// exactly where a row tuned at a single width quietly stops covering the next
/// (the 3.5 sibling needed a retune for precisely that).
///
/// **The ×16 rung is gated on VRAM.** What bounds width here is per-session
/// state, not the checkpoint: 36 GDN layers carry 144 MiB of recurrent state a
/// decoding sequence and 288 MiB while it prefills (its state, timeline and
/// prefill scratch), so sixteen prefilling together want 4.5 GiB before any
/// K/V or the 3.3 GiB tier a sixteen-wide decode stands. On the 16 GB card the
/// dense trunk plus an expert zone on its floor leave 221 regions — measured —
/// against the 288 their state alone needs.
/// Twenty-four GiB is the smallest card in the fleet with that room; below it
/// the rung is skipped and says so, because a narrower run is still a green
/// run and silence would let reduced coverage read as a pass.
#[cfg(all(feature = "cuda", any(test, feature = "ruler-bench")))]
pub fn batched_forward_configs(device: &Device) -> Vec<TestConfig> {
    let vram_gib = quant_ladder::device_vram_gib(device).expect("the card's VRAM is queryable");
    let wide = vram_gib >= 24;
    if !wide {
        println!(
            "  - BF16 ×16 skipped: a {vram_gib} GiB card, under the 24 GiB gate — sixteen \
             sequences' recurrent state does not fit beside the trunk"
        );
    }
    let widths: &[usize] = if wide {
        &[1, 4, 8, 16, 1]
    } else {
        &[1, 4, 8, 1]
    };
    let story = |mode, num_contexts| TestConfig {
        mode,
        use_batched: true,
        num_contexts,
        num_repeats: 1,
        test_mode: Some(TestMode::StoryRewrite),
    };
    let mut configs: Vec<TestConfig> = widths
        .iter()
        .map(|&n| story(InferenceMode::BF16, n))
        .collect();
    configs.extend([
        story(InferenceMode::C0, 2),
        story(InferenceMode::C5, 2),
        story(InferenceMode::C5, 8),
        story(InferenceMode::C8, 2),
        story(InferenceMode::C10, 2),
        story(InferenceMode::C10, 8),
    ]);
    configs
}
