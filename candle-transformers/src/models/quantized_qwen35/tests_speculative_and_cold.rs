//! Measures whether speculative decoding stays lossless on the dense 9B and how
//! much it gains, and where a width ceiling comes from.
//!
//! Speculation is checked against plain decode, which must produce the same text
//! under every draft budget, and its throughput is read as the ratio between the
//! two. Each cold point runs in its own process, because a laptop card's boost
//! budget depletes within one load, and position in the run would otherwise read
//! as a width cliff. A gate asserts that the drafter is loaded, since a missing
//! head would silently degrade to plain decode and report a plausible number.

use super::*;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::models::batch_test::utils::{account_model_load, TestConfig, TestMode, TestParams};

    use crate::models::batched_inference::{InferenceMode, ManagedBatchedModel};

    use crate::models::dialect::Dialect;

    use crate::models::qwen35::mtp::MTP_MAX_DRAFT;

    use candle::quantized::Int8Mode;

    use std::path::PathBuf;

    use super::super::tests_pinned_throughput::tests::qwen35_9b;

    /// **Speculative decode on the dense 9B, measured against itself.**
    ///
    /// The lineage's smallest checkpoint that can speculate at all, and the
    /// size where a decode step is weight-bandwidth-bound rather than
    /// launch-bound — so the win here is the one a production session would
    /// see. The 0.8B has no gate of its own because it has no MTP head in any
    /// conversion: [`speculative_gate`] on it would run the same plain decode
    /// under three budget labels and pass unconditionally.
    ///
    /// The same StoryRewrite configs run once per draft budget: once through
    /// the plain decode loop, then through the lossless speculative driver
    /// with the checkpoint's own MTP head as the drafter. Two things come out
    /// of it.
    ///
    /// * **Correctness.** Speculation accepts only the model's own argmaxes, so
    ///   every run must produce the same text as the first — and the harness
    ///   validates all of them against the same fixture at the same 100%
    ///   threshold. A rewind that left the recurrent state one token ahead of
    ///   the KV would show up here as degraded output, which is exactly the
    ///   failure `truncate_sequence` used to refuse to risk.
    /// * **Throughput.** `t/s (single)` is the per-session decode rate; the
    ///   ratio between the tables is what speculation bought.
    ///
    /// A rewrite task is the drafter's best case *and its honest one*: the
    /// output overlaps the prompt heavily, which is the same shape as editing
    /// code — the workload this engine is for.
    ///
    /// Swept at width, like its siblings: the ladder gate runs this same
    /// checkpoint in BF16 at 4 and 20 contexts, so a speculative step that
    /// cannot is the speculative path's problem and not the card's.
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-9B GGUF (7.5 GB) and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::speculative_decode_9b \
                -- --ignored --nocapture --test-threads=1"]
    fn speculative_decode_9b() -> Result<()> {
        let device = Device::new_cuda(0)?;
        speculative_gate(
            "Qwen3.5-9B",
            Int8Mode::Off,
            &[1, 4],
            &tokenizer_json()?,
            MTP_MAX_DRAFT,
            &device,
            dense_loader(qwen35_9b_off()?),
        )
    }

    /// **Is a width ceiling about KV bytes, or about the session count?**
    ///
    /// Holding the width fixed while varying the generated length separates
    /// them: total KV scales with `width × tokens`, while anything dimensioned
    /// by session count — DeltaNet's per-sequence recurrent state, the transient
    /// tier, admission — does not move at all.
    ///
    /// * Fast when short, slow when long → **KV capacity**.
    /// * Flat across lengths → **session count**, and the search moves to what
    ///   scales with it.
    ///
    /// It measured flat (242.6 / 257.1 / 262.8 tok/s at 64 / 128 / 256 tokens,
    /// width 10), which is what pointed at DeltaNet: 30 recurrent layers each
    /// holding a `32 × 128 × 128` F32 state per sequence, with its conv tail and
    /// timeline, is ~80 MiB per decoding session and is untouched by KV
    /// compression.
    ///
    /// **Each length is its own `params.run` and therefore its own model load**,
    /// which is what makes this readable — see `cold_speculative_point` for why
    /// that matters on this card.
    ///
    /// Budget 0 only. The question is about the baseline, and drafting would
    /// just add rows to both arms.
    #[test]
    #[ignore = "targeted measurement: width 10, three generated lengths, one model load. \
                Run cold. cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::kv_or_width_9b -- --ignored --nocapture \
                --test-threads=1"]
    fn kv_or_width_9b() -> Result<()> {
        let load = dense_loader(qwen35_9b_off()?);
        for tokens in [64usize, 128, 256] {
            println!("\n=== Qwen3.5-9B: width 10, {tokens} generated tokens, plain decode ===\n");
            let params = TestParams::new(tokens, &tokenizer_json()?, Dialect::qwen35())
                .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
                .with_suppress_thinking(true)
                .with_timeout_secs(3600)
                .with_speculative(0)
                .with_majority_pass_threshold(50)
                .with_int8mode(Int8Mode::Off);
            params.run(
                vec![TestConfig {
                    mode: InferenceMode::BF16,
                    use_batched: true,
                    num_contexts: 10,
                    num_repeats: 1,
                    test_mode: Some(TestMode::StoryRewrite),
                }],
                &load,
            )?;
        }
        Ok(())
    }

    /// One width, one budget, one process — the only shape this machine can be
    /// measured in.
    ///
    /// A laptop card runs at full boost for the first tens of seconds of load
    /// and settles to roughly half that afterwards (SM clock swings 450–2310 MHz
    /// with the SW power cap asserting, at 67–77 °C — spent boost budget, not
    /// thermal shutdown). Any gate that runs several configs against one loaded
    /// model therefore measures *position in the run* as much as the variable it
    /// varies: the same config measured 247 tok/s as the first config of a load
    /// and 103 as the third, across four repetitions, with the KV region ledger
    /// byte-identical between them.
    ///
    /// Every earlier "width curve" here was that artefact — width rose with
    /// position, so boost depletion read as a cliff. One point per invocation is
    /// what breaks the confound; the caller is responsible for letting the card
    /// settle between them.
    pub(crate) fn cold_speculative_point<M>(
        label: &str,
        tokenizer: &str,
        mode: InferenceMode,
        width: usize,
        budget: usize,
        int8mode: Int8Mode,
        load: impl Fn() -> Result<M>,
    ) -> Result<()>
    where
        M: ManagedBatchedModel,
    {
        println!("\n=== {label} cold point: {mode:?}, width {width}, draft budget {budget} ===\n");
        let params = TestParams::new(256, tokenizer, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_timeout_secs(3600)
            .with_speculative(budget)
            // A throughput measurement, and a wide cohort cannot match a fixed
            // string at any useful rate — the lineage's stochastic floor grows
            // with width. Kept only as a garbage detector: a broken accept rule
            // reproduces a 1228-character fixture in none of the cohort, not a
            // quarter of it. Losslessness is gated at 100% in
            // `speculative_decode_9b`.
            .with_majority_pass_threshold(25)
            .with_int8mode(int8mode);
        // Checked before the run, not after: a budget the checkpoint cannot
        // honour degrades silently to plain decode, and this measurement sets a
        // production constant.
        let checked = move || {
            let m = load()?;
            assert_drafter(&m, budget)?;
            Ok(m)
        };
        params.run(
            vec![TestConfig {
                mode,
                use_batched: true,
                num_contexts: width,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }],
            checked,
        )
    }

    /// One `#[test]` per measurement point, so each runs in its own process with
    /// its own boost budget. A macro because twelve hand-written copies of the
    /// same two lines is where a transposed width or budget hides.
    macro_rules! cold_point {
        ($name:ident, $width:expr, $budget:expr) => {
            #[test]
            #[ignore = "cold measurement point — run singly, letting the card settle between \
                        points. See `cold_speculative_point`."]
            fn $name() -> Result<()> {
                cold_speculative_point(
                    "Qwen3.5-9B",
                    &tokenizer_json()?,
                    InferenceMode::BF16,
                    $width,
                    $budget,
                    Int8Mode::Off,
                    dense_loader(qwen35_9b_off()?),
                )
            }
        };
    }

    cold_point!(cold_w4_b0, 4, 0);

    cold_point!(cold_w10_b0, 10, 0);

    cold_point!(cold_w10_b1, 10, 1);

    cold_point!(cold_w10_b2, 10, 2);

    cold_point!(cold_w16_b0, 16, 0);

    cold_point!(cold_w16_b1, 16, 1);

    cold_point!(cold_w16_b2, 16, 2);

    cold_point!(cold_w20_b0, 20, 0);

    cold_point!(cold_w20_b1, 20, 1);

    cold_point!(cold_w20_b2, 20, 2);

    // **Width 32 needs a card this lineage's dev machines do not have.** Thirty
    // recurrent layers holding a `32 × 128 × 128` F32 state per sequence, with
    // its conv tail and timeline, is ~80 MiB per decoding session — twice that
    // while it prefills — and compression cannot touch it, so 32 sessions
    // prefilling together is ~5 GiB of DeltaNet state before any
    // KV, and admission refuses on the 16 GB and 24 GB boxes. They are kept, and
    // `#[ignore]`d like every other point here, because they are the instrument
    // for the 72 GB Blackwell: the answer they carry is whether the turn at 16
    // is the card or the design, and only a bigger card can say.
    cold_point!(cold_w32_b0, 32, 0);

    cold_point!(cold_w32_b1, 32, 1);

    cold_point!(cold_w32_b2, 32, 2);

    /// **Does throughput decay across configs within one model load?**
    ///
    /// Every sweep so far has been read as a width curve, and the widths were
    /// confounded with their position in the run. Width 10 at 256 tokens
    /// measured 118 tok/s as the third config of a load and 263 as the first —
    /// the same deterministic work, 2.2x apart — while a fresh load is fast
    /// every time.
    ///
    /// This runs ONE config three times over, so width, length, budget and
    /// fixture are all held constant and only position varies. A monotone decay
    /// means state is surviving between configs that should not: stranded
    /// arenas holding regions, recurrent or draft state, unreleased sequences.
    /// It would also mean every multi-config gate in this file under-reports its
    /// later rows, and that the "width cliff" was never about width.
    #[test]
    #[ignore = "targeted measurement: one config three times over, single model load. \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::decay_across_configs_9b -- --ignored --nocapture \
                --test-threads=1"]
    fn decay_across_configs_9b() -> Result<()> {
        let load = dense_loader(qwen35_9b_off()?);
        let params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_timeout_secs(3600)
            .with_speculative(0)
            .with_majority_pass_threshold(50)
            .with_int8mode(Int8Mode::Off);
        let one = TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 10,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        };
        params.run(vec![one.clone(), one.clone(), one], &load)
    }

    /// Run the speculative comparison for a dense checkpoint of the lineage.
    ///
    /// **256 generated tokens**, not the gates' 10: speculation is a property
    /// of the decode loop, and ten tokens is six driver steps — a number small
    /// enough that the prefill and the first block dominate it. The draft
    /// budget is swept so the table shows where the yield stops paying for the
    /// rows it adds.
    ///
    /// `tokenizer` and `max_draft` are the caller's, because they are the two
    /// things that are genuinely per-checkpoint: a lineage sibling has its own
    /// vocabulary, and its head's ceiling is its own.
    pub(crate) fn speculative_gate<M>(
        label: &str,
        int8mode: Int8Mode,
        widths: &[usize],
        tokenizer: &str,
        max_draft: usize,
        device: &Device,
        load: impl Fn() -> Result<M>,
    ) -> Result<()>
    where
        M: ManagedBatchedModel,
    {
        let configs: Vec<TestConfig> = widths
            .iter()
            .map(|&n| TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect();
        // **Loaded once for the whole sweep.** Every budget runs the same
        // weights, and on these checkpoints opening the artifact costs minutes
        // — reloading per budget spent most of the wall clock re-reading bytes
        // that never changed. It also makes the comparison sounder: the expert
        // cache's warmth and the arena's shape carry across the budgets instead
        // of being rebuilt differently for each.
        let model = account_model_load(device, load)?;

        // Up to the head's own ceiling, and no further: past it the drafter
        // clamps, so the extra rows would report the same configuration twice.
        for draft in 0..=max_draft {
            println!(
                "\n=== {label}: {} ===\n",
                if draft == 0 {
                    "plain decode (baseline)".to_string()
                } else {
                    format!("speculative decode, draft budget {draft}")
                }
            );
            let mut params = TestParams::new(256, tokenizer, Dialect::qwen35())
                .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
                .with_suppress_thinking(true)
                .with_timeout_secs(3600);
            params = params.with_speculative(draft).with_int8mode(int8mode);
            // Same guard the cold points carry, for the same reason and with
            // more at stake: this is the sweep the production bracket is read
            // from, so a checkpoint that has quietly lost its NextN head would
            // report every budget as 1.00× here and read as "speculation stopped
            // paying" rather than "the drafter is missing".
            assert_drafter(&model, draft)?;
            params.run_loaded(configs.clone(), &model)?;
        }
        Ok(())
    }

    /// Assert the model really carries a drafter, for a caller whose whole
    /// purpose is measuring speculation.
    ///
    /// A budget the model cannot honour is not an error — `speculative_draft`
    /// answers with no proposals and the step degrades to plain decode, which is
    /// exactly the behaviour a checkpoint without a head should have. That makes
    /// a pin or conversion regression invisible here in the worst way: the
    /// measurement still produces a number, the number is a plain-decode number,
    /// and it is the number the production bracket gets set from.
    pub(in super::super) fn assert_drafter<M: ManagedBatchedModel>(
        model: &M,
        budget: usize,
    ) -> Result<()> {
        if budget > 0 && model.draft_budget(1) == 0 {
            candle::bail!(
                "this checkpoint reports a zero draft budget even at width 1 — it carries no \
                 NextN head, so a budget-{budget} measurement would silently report plain decode"
            );
        }
        Ok(())
    }

    /// The 9B's pack in `Int8Mode::Off`, the form the speculative gates measure.
    pub(in super::super) fn qwen35_9b_off() -> Result<PathBuf> {
        qwen35_9b(&Device::new_cuda(0)?, Some(Int8Mode::Off))
    }

    /// The dense lineage's loader over a model pack, as [`speculative_gate`]
    /// takes it.
    pub(in super::super) fn dense_loader(pack: PathBuf) -> impl Fn() -> Result<HybridBatched> {
        move || {
            let device = Device::new_cuda(0)?;
            let m = from_pack(&pack, &device)?;
            // A gate that silently fell back to plain decode would still pass
            // — speculation is lossless, so the only symptom is the speedup
            // going away. Assert the drafter is really there.
            assert!(
                m.has_drafter(),
                "the pinned dense checkpoint has no MTP head — the pin has moved \
                 off the -MTP-GGUF repo, or its conversion dropped the NextN tensors"
            );
            println!("✓ Model loaded\n");
            Ok(m)
        }
    }
}
