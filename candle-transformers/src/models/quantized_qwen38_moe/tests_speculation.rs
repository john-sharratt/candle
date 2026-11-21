//! Speculative-decoding tests: the rung ladder, the K-format bisect, the drafter
//! comparison, and the divergence record.
//!
//! The ladder runs every compression rung at the widths the C-ladder is read at,
//! with the draft budget left adaptive so the table shows production behaviour. It
//! asserts that the drafter is present, so a silent fallback to plain decode cannot
//! pass. Speculation is lossless, so every budget must validate the same, and a
//! rung that fails at depth is a rewind defect. The K-format test pins each suspect
//! format as the only one, with a Q4_0 control that must hold. The top-rung test
//! asserts only that C8 stays exact, and records where C9 and C10 drift under plain
//! decode, which the 64-token gate cannot see.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::batch_test::utils::{account_model_load, TestConfig, TestMode};

    use candle::Tensor;

    use super::super::tests_support::tests::{engine, tokenizer, tokenizer_json};

    /// The speculation sweep, by width, one process per width.
    #[path = "../../tests/spec_sweep.rs"]
    pub(in super::super) mod spec_sweep;

    /// **The speculative ladder** — every compression rung, and the widths.
    ///
    /// Wider than [`batched_forward_configs`] on purpose. That one is a *calibration* gate:
    /// it samples C0/C5/C8/C10 because those four are what pin
    /// `QWEN4EXP_KV_FACTORS`, and running the levels between them would cost
    /// time to re-measure a row already bounded by its ends. This one asks a
    /// different question — what speculation is worth at each operating point —
    /// and the answer is not interpolable between rungs, because the accepted
    /// prefix depends on how much the compressed K/V has moved the target's own
    /// argmaxes. A level where acceptance collapses is a level where the
    /// drafter and the target have stopped agreeing, and only the rung itself
    /// shows that.
    ///
    /// The 32-context row is past the ladder's last bracket, so it decodes
    /// plain: it is here to show the width at which speculation stops, beside
    /// the rows that speculate.
    pub(in super::super) fn speculative_ladder() -> Vec<TestConfig> {
        use crate::models::batched_inference::InferenceMode;

        let mut configs: Vec<TestConfig> = [1usize, 4, 8, 16, 32]
            .into_iter()
            .map(|n| TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect();
        // Every rung, C0 through C10, at the width the C-ladder is read at.
        configs.extend(
            [
                InferenceMode::C0,
                InferenceMode::C1,
                InferenceMode::C2,
                InferenceMode::C3,
                InferenceMode::C4,
                InferenceMode::C5,
                InferenceMode::C6,
                InferenceMode::C7,
                InferenceMode::C8,
                InferenceMode::C9,
                InferenceMode::C10,
            ]
            .map(|mode| TestConfig {
                mode,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }),
        );
        // The operating point and the probe at width too, since a rung that
        // holds at 2 can still lose the cohort at 8 — which is exactly the
        // failure the 3.5 sibling's retune was for.
        configs.extend(
            [InferenceMode::C5, InferenceMode::C10].map(|mode| TestConfig {
                mode,
                use_batched: true,
                num_contexts: 8,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }),
        );
        // A trailing single context, the way the plain gate ends: the leading
        // ×1 is cold — expert cache empty, arenas unbuilt — so it prices a
        // first request rather than the steady state. This one runs against
        // everything the ladder above warmed, which is the number a
        // single-session client actually sees.
        configs.push(TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        });
        configs
    }

    /// How deep this checkpoint's head is swept. Its own ceiling, not the
    /// hybrid's: a one-block NextN head is trained to predict a single position
    /// and is used recursively past that, so each extra row is conditioned on
    /// the head's own previous guess and pays less than the one before it. Four
    /// is far enough to see where that stops paying.
    pub(in super::super) const QWEN38_MAX_DRAFT: usize = 4;

    /// **The gate.** The same iterating `test_parallel_batched_forwarding`
    /// every production model here has (`docs/qwen38_flash_next.md` §11): the
    /// BF16 config ladder at the widths the DeepSeek gate runs, per-session
    /// StoryRewrite validation at the 100% threshold, the performance table,
    /// and the expert-pipeline stats. The C-ladder rungs wait on this model's
    /// own `PRODUCTION_*` threshold derivation (§8 item 6) — KV compression
    /// rows are per-model and per-machine and nothing carries over.
    /// **Speculative decode, swept.** The batched gate's shape and its
    /// StoryRewrite validation, run once per draft budget so the table shows
    /// what each extra read-ahead row buys.
    ///
    /// Measured here rather than on a short probe because speculation is a
    /// property of the *decode loop*: the win is tokens per driver step, and a
    /// handful of tokens is a handful of steps, where the prefill and the first
    /// block dominate whatever the drafter does.
    ///
    /// **Budget 0 is the baseline in the same harness**, not a number carried
    /// over from another run — the comparison has to hold the KV rungs, the
    /// expert cache's warmth and the session count fixed, and the only way to
    /// be sure of that is to measure both here.
    ///
    /// Speculation is lossless: every budget must produce the same validity, so
    /// a rung that goes red at depth is a rewind bug, never a quality tradeoff.
    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_speculative_ladder \
                -- --ignored --nocapture --test-threads=1"]
    fn test_speculative_ladder() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::ManagedBatchedModel;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        println!("\n=== Qwen3.8-Flash-Next: speculative ladder (production budget) ===\n");
        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);

        // **No `with_speculative`.** The default is `DraftBudget::Adaptive`,
        // which asks the model at each config's width — production behaviour,
        // and the whole point of this table. Pinning a budget would measure a
        // configuration nothing runs, and would hide the one thing the wide
        // rows exist to show: where the ladder stops speculating.
        let params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(7200);

        params.run(speculative_ladder(), || {
            let gpu = Qwen4ExpGpu::load(&pack, &device)?;
            let m = Qwen4ExpBatched::new(gpu)?;
            // A ladder that silently fell back to plain decode would still pass
            // — speculation is lossless, so the only symptom is the speedup
            // going away. Assert the drafter is really there.
            assert!(
                m.draft_budget(1) > 0,
                "this artifact reports a zero draft budget at width 1 — it carries no \
                 NextN head, so every row here would report plain decode"
            );
            println!("✓ Model loaded\n");
            Ok(m)
        })
    }

    /// **Which K format breaks it — one format per run, through the gate that
    /// reproduces.**
    ///
    /// C6 adds exactly four K candidates C5 never had — `Q1_S`, `Q2_A`, `Q2_S`
    /// and `Q8_1` — and acceptance falls 97.6% → 2.9% across that boundary.
    /// What says this is a *discrete* defect rather than precision loss is that
    /// C6 through C10 are all equally bad (1.5–4.9%) while their compression
    /// ratios keep climbing, 5.61× to 7.53×. Graduated loss would make C10 far
    /// worse than C6; a single bad format that any of those levels can select
    /// makes them all the same.
    ///
    /// So each suspect is pinned as the ONLY K format, via
    /// `BatchedConfig::override_k_quant`, and run through the gate's own
    /// StoryRewrite — the harness that actually reproduces the 97.6% figure.
    /// A control from C5's set (`Q4_0`) runs first: it must hold, or the
    /// override itself is what breaks things and no other row means anything.
    ///
    /// Only the sealed prefill is compressed — the live decode tail stays F16 —
    /// so a bad format here corrupts the *history* the head attends over while
    /// the tokens around it stay clean. That is consistent with proposals that
    /// read fluently and are never the trunk's.
    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_which_k_format_breaks_drafting \
                -- --ignored --nocapture --test-threads=1"]
    fn test_which_k_format_breaks_drafting() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;
        use candle_nn::kv_cache::QuantFormat;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let model = account_model_load(&device, || {
            let gpu = Qwen4ExpGpu::load(&pack, &device)?;
            Qwen4ExpBatched::new(gpu)
        })?;

        // `Q4_0` is the control — it is in C5's set, which drafts at 97.6%.
        // The rest are exactly what C6 adds.
        let suspects = [
            ("Q4_0 (control, in C5)", QuantFormat::Q4_0),
            ("Q1_S", QuantFormat::Q1_S),
            ("Q2_A", QuantFormat::Q2_A),
            ("Q2_S", QuantFormat::Q2_S),
            ("Q8_1", QuantFormat::Q8_1),
        ];

        for (label, fmt) in suspects {
            println!("\n=== K pinned to {label} ===\n");
            let params = TestParams::new(256, &tok, Dialect::qwen35())
                .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
                .with_suppress_thinking(true)
                .with_int8mode(int8mode)
                .with_override_k_quant(Some(fmt))
                .with_timeout_secs(3600);
            // One config: the level that works, so the ONLY variable is which
            // format its keys actually land in.
            let configs = vec![TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }];
            let ok = params.run_loaded(configs, &model).is_ok();
            println!("  {label}: validity {}", if ok { "PASS" } else { "FAIL" });
        }
        Ok(())
    }

    /// **Why acceptance collapses between C5 and C6 — the drafter, side by
    /// side, at both levels.**
    ///
    /// Drafted-token acceptance is 97.6% at C5 and 2.9% at C6, then flat at
    /// 1.5–4.9% through C10 no matter how much lossier the level gets. That
    /// shape is not compression noise decorrelating a prediction — noise gives
    /// a gradient, and a level far lossier than C6 would be far worse than C6.
    /// Flat-and-near-zero across five rungs is something **breaking** at the
    /// C5→C6 boundary and staying broken.
    ///
    /// The trunk is not what breaks: C6, C7 and C8 all validate 100%, so the
    /// target produces the same text it always did. Only the head's proposals
    /// collapse.
    ///
    /// So this runs the same sequence at both levels, deep enough that the KV
    /// has actually been compressed, and reports:
    ///
    /// * **which proposal position first misses** — position 0 missing means
    ///   the head is reading bad state; only later positions missing means the
    ///   walk's own recurrence;
    /// * **the head layer's block count against the trunk's** — a head layer
    ///   standing at a different length from the trunk rotates its drafted
    ///   positions wrongly, which could only ever surface as acceptance quietly
    ///   collapsing: this symptom exactly.
    ///
    /// A short prompt cannot see any of it: chunks are 32 tokens, so nothing is
    /// compressed until they fill, and C5 and C6 are then the same run.
    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_drafter_c5_vs_c6 \
                -- --ignored --nocapture --test-threads=1"]
    fn test_drafter_c5_vs_c6() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, InferenceMode, ManagedBatchedModel};
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::IndexOp;
        use candle_nn::kv_cache::GapKind;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let gpu = Qwen4ExpGpu::load(&pack, &device)?;
        let model = Qwen4ExpBatched::new(gpu)?;
        let tok = tokenizer()?;
        let head_kv = model
            .engine()
            .cfg
            .mtp_kv_layer()
            .ok_or_else(|| candle::Error::msg("no draft head"))?;

        // Long enough that chunks fill and the policy actually compresses.
        let para = "The archivist walked the long gallery each morning, counting the \
                    lamps that had failed overnight and noting them in a ledger nobody \
                    else read. ";
        let prompt = para.repeat(24);
        let ids: Vec<u32> = tok
            .encode(prompt.as_str(), true)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("prompt: {} tokens", ids.len());

        const DRAFT: usize = 4;
        // Sampled at several depths, not one. A single sample cannot tell "the
        // drafter is broken here" from "the drafter degrades with depth", and
        // those want different explanations. BF16 is in the list as the arm
        // with no compression at all — if it tracks the compressed rungs, the
        // variable is depth; if it holds while they fall, it is compression.
        const CHECKPOINTS: [usize; 4] = [8, 32, 64, 96];

        for mode in [
            InferenceMode::BF16,
            InferenceMode::C0,
            InferenceMode::C5,
            InferenceMode::C6,
        ] {
            println!("\n=== {mode:?} ===");
            // **The formats, not only the level.** Setting `compression_level`
            // alone leaves the KV in the default float format, so nothing is
            // ever compressed and every rung behaves identically — a null
            // result that looks exactly like "the rungs are the same". The
            // harness builds a rung from all three fields; so must this.
            let config = BatchedConfig {
                k_format: mode.k_format(),
                v_format: mode.v_format(),
                compression_level: mode.compression_level(),
                ..Default::default()
            };
            let mut session = model.create_batched_session(config)?;
            let seq = session.create_sequence()?;
            let n_layers = ManagedBatchedModel::num_layers(&model);

            let prompt_t = Tensor::from_vec(ids.clone(), (1, ids.len()), &Device::Cpu)?;
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&prompt_t),
                &[],
                &[],
                0,
                n_layers,
                None,
            )?;
            session.advance_sequence(seq, ids.len())?;
            // **Quantize the prefilled history, as the scheduler's gaps do.**
            // Compression happens in the gap, not at write, so a run that only
            // decodes leaves every chunk float — and then C0, C5 and C6
            // produce byte-identical output, which reads as "the rungs do not
            // differ" rather than "nothing was compressed". Every complete
            // block is quantized before the decode below reads it.
            session.gap_settle(GapKind::Prefill)?;
            let mut next = step.logits_owned()?[0]
                .i(0)?
                .argmax(0)?
                .to_scalar::<u32>()?;

            // Decode past the point where chunks have filled and been sealed,
            // so the KV the head reads is genuinely compressed.
            let plain = |session: &mut _, t: u32| -> Result<u32> {
                let x = Tensor::from_vec(vec![t], (1, 1), &Device::Cpu)?;
                let s = model.forward_wave(
                    session,
                    &[seq],
                    std::slice::from_ref(&x),
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    n_layers,
                    None,
                )?;
                session.advance_sequence(seq, 1)?;
                s.logits_owned()?[0].i(0)?.argmax(0)?.to_scalar::<u32>()
            };
            let mut done = 0usize;
            for &mark in &CHECKPOINTS {
                while done < mark {
                    next = plain(&mut session, next)?;
                    done += 1;
                }
                let caches = session.sequence_caches(seq).expect("live slot");
                let trunk_blocks = caches.caches[0].k_cache().chunked_max_blocks();
                let head_blocks = caches.caches[head_kv].k_cache().chunked_max_blocks();
                let offset = session.sequence_offset(seq).unwrap_or(0);

                let proposals = model.speculative_draft(&mut session, &[seq], &[next], DRAFT)?;
                let drafted = proposals[0].clone();

                // Ground truth: what the trunk actually produces next. These
                // decodes advance the sequence, so the next checkpoint simply
                // continues from here.
                let mut truth: Vec<u32> = Vec::with_capacity(DRAFT);
                let mut cur = next;
                for _ in 0..DRAFT {
                    cur = plain(&mut session, cur)?;
                    truth.push(cur);
                }
                done += DRAFT;
                next = cur;
                let matched = drafted
                    .iter()
                    .zip(truth.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                println!(
                    "  +{mark:<3} offset {offset:<4} trunk {trunk_blocks} \
                     head {head_blocks} | accepted {matched}/{DRAFT}  drafted {drafted:?} \
                     truth {truth:?}"
                );
            }
            model.release_sequence(seq)?;
        }
        Ok(())
    }

    /// **C9/C10 drift from the reference without speculation, and this is the
    /// record of that.**
    ///
    /// Measured, at 256 generated tokens, width 2, every draft budget:
    ///
    /// | budget | C8 | C9 | C10 |
    /// |---|---|---|---|
    /// | 0 (plain decode) | exact | diverges @505/506 | diverges @505/440 |
    /// | 1 | exact | @505/506 | @505/443 |
    /// | 2 | exact | @505/506 | @505/440 |
    /// | 4 | exact | @505/506 | @505/440 |
    ///
    /// Same characters at every depth, and present with **no speculation at
    /// all**. So the top rungs are not a speculation defect: they lose the
    /// reference around 110–130 generated tokens on their own.
    ///
    /// **The standing gate cannot see this**, and that is the point worth
    /// keeping. `test_parallel_batched_forwarding` generates **64** tokens —
    /// roughly 250 characters — and the divergence starts past 440. Its C10 row
    /// passing is a true statement about the first 64 tokens and says nothing
    /// about the rung.
    ///
    /// C9 is also exactly where the **Q0 family** first becomes selectable for
    /// keys (`Q0`, a 1-byte zero block; `Q0_V`/`Q0_X`/`Q0_M2`/`Q0_M4`, whole-
    /// block summaries at 2–8 bytes per 32 elements); C8 and below have none.
    /// That remains a correlation, but now with a *plain-decode* defect — which
    /// makes it a `QWEN4EXP_KV_FACTORS` threshold question rather than anything
    /// to do with the draft head.
    ///
    /// This test exists so nobody re-attributes it to the drafter. It cost
    /// three wrong controls to establish: a substring filter that ran twelve
    /// models' gates, a run that aborted before it validated anything, and a
    /// gate whose generation was too short to reach the divergence at all.
    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_top_rung_divergence_vs_budget \
                -- --ignored --nocapture --test-threads=1"]
    fn test_top_rung_divergence_vs_budget() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let model = account_model_load(&device, || {
            let gpu = Qwen4ExpGpu::load(&pack, &device)?;
            Qwen4ExpBatched::new(gpu)
        })?;

        // The two rungs that drift, at the width they drift at, plus C8 as the
        // control that must stay exact at every budget.
        let configs: Vec<TestConfig> = [InferenceMode::C8, InferenceMode::C9, InferenceMode::C10]
            .into_iter()
            .map(|mode| TestConfig {
                mode,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect();

        let sound: Vec<TestConfig> = configs.iter().take(1).cloned().collect();
        let top: Vec<TestConfig> = configs.iter().skip(1).cloned().collect();

        let run = |set: &[TestConfig], budget: usize, label: &str| -> Result<bool> {
            println!("\n=== {label}, draft budget {budget} ===\n");
            let params = TestParams::new(256, &tok, Dialect::qwen35())
                .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
                .with_suppress_thinking(true)
                .with_int8mode(int8mode)
                .with_speculative(budget)
                .with_timeout_secs(3600);
            Ok(params.run_loaded(set.to_vec(), &model).is_ok())
        };

        // **C8 is the assertion.** A rung that plain decode holds must stay
        // exact at every depth — that is what lossless means, and it is the
        // only part of this test that can fail for speculation's sake.
        for budget in [0usize, 1, 2, 4] {
            assert!(
                run(&sound, budget, "C8 (sound rung)")?,
                "C8 diverged at draft budget {budget} — a rung plain decode holds exactly \
                 must survive speculation, so this is a rewind defect"
            );
        }

        // **C9/C10 are the record.** They drift from the reference around
        // 110-130 generated tokens under PLAIN DECODE, which is why the
        // standing gate — 64 tokens — has never seen it. Asserted the way it
        // measures rather than the way it ought to be, so that a future run
        // where speculation *does* make them worse is visible as a change.
        let mut top_ok: Vec<usize> = Vec::new();
        for budget in [0usize, 1, 2, 4] {
            if run(&top, budget, "C9/C10 (top rungs)")? {
                top_ok.push(budget);
            }
        }
        println!("\nbudgets at which C9/C10 held: {top_ok:?}  (expected: none)");
        assert!(
            top_ok.is_empty(),
            "C9/C10 held at budgets {top_ok:?} — they diverge under plain decode today, so \
             holding is a change worth looking at, not a pass"
        );
        Ok(())
    }

    /// **The draft-budget sweep.** Holds the width fixed and varies the depth,
    /// which is how the ladder's brackets were derived — distinct from
    /// [`test_speculative_ladder`], which holds the ladder and varies the rung.
    #[test]
    #[ignore = "loads this card's engine artifact and needs a GPU. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_speculative_decode \
                -- --ignored --nocapture --test-threads=1"]
    fn test_speculative_decode() -> Result<()> {
        use crate::models::quantized_qwen35::tests::speculative_gate;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let device = Device::new_cuda(0)?;
        speculative_gate(
            "Qwen3.8-Flash-Next",
            int8mode,
            &[1, 4],
            &tok,
            QWEN38_MAX_DRAFT,
            &device,
            || {
                let device = Device::new_cuda(0)?;
                let gpu = Qwen4ExpGpu::load(&pack, &device)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }
}
