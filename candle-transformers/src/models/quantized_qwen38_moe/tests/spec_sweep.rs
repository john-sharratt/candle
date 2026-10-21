//! **Speculation swept by width, one width per process.**
//!
//! What `models::qwen38_flash_next_sweep` is fitted from: each card's draft
//! ceiling at each wave width, measured on that card. Every width here is its
//! own `#[test]`, so a sweep driver runs them one process at a time, widest
//! last, and stops once speculation stops paying.
//!
//! At a width, one load: budget 0, then 1, 2, … up to the head's ceiling, stopping
//! at the first budget slower than the one before it, then budget 0 again. The
//! closing baseline is the drift check — on a card that sheds clock as a load
//! goes on, the two baselines disagree and the width's figures say so, rather
//! than reading the decay as a budget's cost (`cold_speculative_point`'s note).
//! Every point is validated by the gate's own StoryRewrite verdict: speculation
//! is lossless, so a budget that goes red is a rewind bug, never a tradeoff.
//!
//! Each width ends with one `SPEC_SWEEP` line the driver reads: the width, each
//! budget's decode rate, the closing baseline, the best budget and its gain.
//! The line is the instrument's output; the ladder rows are fitted from it.

use super::*;
use crate::models::batch_test::utils::TestParams;
use crate::models::batched_inference::{InferenceMode, ManagedBatchedModel};
use crate::models::dialect::Dialect;
use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};

/// Decode tokens each session generates at a point: the gate's own length.
const SWEEP_TOKENS: usize = 256;

/// One width's sweep: decode rate by budget, the closing baseline, and the
/// winner.
fn sweep_width(width: usize) -> Result<()> {
    let pack = engine()?;
    let device = Device::new_cuda(0)?;
    let int8mode = Int8Mode::auto(&device);
    let tok = tokenizer_json()?;
    let model = account_model_load(&device, || {
        let gpu = Qwen4ExpGpu::load(&pack, &device)?;
        Qwen4ExpBatched::new(gpu)
    })?;
    // A drafter that is not there reports every budget as plain decode, and the
    // sweep would read as "speculation never pays".
    assert!(
        model.draft_budget(1) > 0,
        "this artifact carries no NextN head: every budget would be plain decode"
    );
    let point = |budget: usize| -> Result<f64> {
        let mut params = TestParams::new(SWEEP_TOKENS, &tok, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(3600)
            .with_speculative(budget)
            // A garbage detector, as at `cold_speculative_point`: a wide cohort
            // cannot match a fixed string at a useful rate, and a broken accept
            // rule matches it in none of the cohort.
            .with_majority_pass_threshold(25);
        let mut results = params.run_loaded_collect(
            vec![TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: width,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }],
            &model,
        )?;
        params.validate_and_print(&mut results)?;
        Ok(results[0].generate_tokens_per_sec)
    };

    let base = point(0)?;
    let mut rates = vec![base];
    for budget in 1..=QWEN38_MAX_DRAFT {
        let rate = point(budget)?;
        rates.push(rate);
        if rate < rates[budget - 1] {
            break;
        }
    }
    let base_after = point(0)?;
    let (best, best_rate) = rates
        .iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .expect("budget 0 was measured");
    // Gain against the mean of the two baselines, so drift across the load
    // cannot read as a budget's win or loss.
    let baseline = (base + base_after) / 2.0;
    let by_budget: Vec<String> = rates
        .iter()
        .enumerate()
        .map(|(b, r)| format!("b{b}={r:.1}"))
        .collect();
    println!(
        "SPEC_SWEEP width={width} {} b0_after={base_after:.1} drift={:.3} best={best} gain={:.3}",
        by_budget.join(" "),
        base_after / base - 1.0,
        best_rate / baseline - 1.0,
    );
    Ok(())
}

/// One `#[test]` per width, so each runs in its own process; a macro because
/// thirty-two hand-written copies is where a transposed width hides.
macro_rules! sweep {
    ($($name:ident = $width:expr),* $(,)?) => {
        $(
            #[test]
            #[ignore = "one width of the speculation sweep: loads this card's engine and \
                        needs the card alone. Run widths in ascending order, one process \
                        each, and stop when the gain stops paying."]
            fn $name() -> Result<()> {
                sweep_width($width)
            }
        )*
    };
}

sweep!(
    spec_sweep_w01 = 1,
    spec_sweep_w02 = 2,
    spec_sweep_w03 = 3,
    spec_sweep_w04 = 4,
    spec_sweep_w05 = 5,
    spec_sweep_w06 = 6,
    spec_sweep_w07 = 7,
    spec_sweep_w08 = 8,
    spec_sweep_w09 = 9,
    spec_sweep_w10 = 10,
    spec_sweep_w11 = 11,
    spec_sweep_w12 = 12,
    spec_sweep_w13 = 13,
    spec_sweep_w14 = 14,
    spec_sweep_w15 = 15,
    spec_sweep_w16 = 16,
    spec_sweep_w17 = 17,
    spec_sweep_w18 = 18,
    spec_sweep_w19 = 19,
    spec_sweep_w20 = 20,
    spec_sweep_w21 = 21,
    spec_sweep_w22 = 22,
    spec_sweep_w23 = 23,
    spec_sweep_w24 = 24,
    spec_sweep_w25 = 25,
    spec_sweep_w26 = 26,
    spec_sweep_w27 = 27,
    spec_sweep_w28 = 28,
    spec_sweep_w29 = 29,
    spec_sweep_w30 = 30,
    spec_sweep_w31 = 31,
    spec_sweep_w32 = 32,
);
