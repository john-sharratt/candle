//! Runs the story-rewrite ladder on the dense 9B, plain and through a LoRA
//! adapter, to check the hybrid's output under every KV mode.
//!
//! The base gate and the adapter run share one ladder, so their rows can be
//! compared, and each sequence has its own protagonist, so a leak between
//! sessions shows as a wrong name. The adapter's thresholds were never derived
//! from the base calibration, so the adapted ladder is deliberately outside the
//! gate's name and runs only when named. The greedy probe builds its prompt in
//! one piece, so a model that will not follow the task can be told apart from a
//! harness that fed it something else.

use super::*;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::model_overrides::{self, Checkpoint};

    use crate::models::batch_test::test_helpers::hf_get;

    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};

    use crate::models::batched_inference::InferenceMode;

    use crate::models::dialect::Dialect;

    use candle::quantized::Int8Mode;

    use hf_hub::RepoType;

    use std::path::PathBuf;

    use super::super::tests_pinned_throughput::tests::qwen35_9b;

    /// The story-rewrite gate on the dense 9B: the hybrid sweep end to end —
    /// prefill, decode, the recurrent carry across both, and the paged KV of
    /// the eight layers that actually attend.
    ///
    /// Dense on purpose. It is the same `forward_wave`, the same layer
    /// dispatch and the same recurrent bookkeeping as the 35B, with the expert
    /// cache taken out of the picture — so a failure here is the hybrid, and a
    /// failure only on the 35B is the experts.
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-9B GGUF (7.5 GB) and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::test_parallel_batched_forwarding_9b \
                -- --ignored --nocapture --test-threads=1 \
                (serial: these load multi-gigabyte checkpoints and will exhaust \
                the card if cargo runs them concurrently)"]
    fn test_parallel_batched_forwarding_9b() -> Result<()> {
        println!("\n=== Qwen3.5-9B hybrid batched forwarding ===\n");
        let device = Device::new_cuda(0)?;

        // **Resolved here, then used twice**, so the table's `int8` column
        // cannot disagree with what the model loaded — the label is set
        // independently of the loader, and a gate that pins one and defaults the
        // other reports a numeric path it is not running.
        // `auto`, so the gate runs the twin production selects rather than a
        // pinned one. It used to pin `Performance` on the grounds that the two
        // twins are interchangeable — measured true *on this model* (every rung
        // valid either way, throughput inside the run-to-run band) and false in
        // general: the same swap takes Llama-3's ladder to C6 0/1, C7 0/1,
        // C8 9/10. A dial that is free on one model is not free on the next, so
        // the gate follows the default instead of asserting the two are alike.
        let int8mode = Int8Mode::auto(&device);
        let params = TestParams::new(10, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(1800);

        let configs = story_rewrite_ladder();

        let pack = qwen35_9b(&device, Some(int8mode))?;
        let load = || {
            let m = from_pack(&pack, &device)?;
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }

    /// Every row the dense 9B's story-rewrite gate runs: the float modes, the
    /// plain-quantized KV, and the full C0–C10 ladder.
    ///
    /// Shared by the base gate and the LoRA gate rather than written twice. The
    /// two exist to be compared — the adapted run is this same ladder over the
    /// same prompts with the adapter switched on — and a row present in one and
    /// missing from the other would make that comparison quietly untrue.
    pub(in super::super) fn story_rewrite_ladder() -> Vec<TestConfig> {
        vec![
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // Several sequences at once: the recurrent mixer runs per span and
            // a slicing error here shows up as one conversation leaking into
            // another, which the single-context rows cannot catch.
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // ── Quantized KV — the same ladder the 0.8B gate runs, on the
            // checkpoint whose projections are themselves quantized (Q6_K +
            // int8 activations), so threshold slack the BF16 0.8B leaves on
            // the table is not silently spent twice here.
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C0,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C1,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C2,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C3,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C4,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C6,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C7,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C8 runs wide (20 contexts) — deepest production-comfortable
            // rung; the width stresses isolation under compression and yields
            // a real aggregate-throughput figure.
            TestConfig {
                mode: InferenceMode::C8,
                use_batched: true,
                num_contexts: 20,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C10×10 is the top rung and the calibration target:
            // `QWEN35_9B_KV_FACTORS` is tuned so the whole range C0–C10
            // passes with C10 just under the breaking edge. A red C10 row
            // means the thresholds drifted past the edge — retighten the
            // factor row rather than widening tolerances.
            TestConfig {
                mode: InferenceMode::C10,
                use_batched: true,
                num_contexts: 10,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
        ]
    }

    /// The PEFT LoRA adapter this gate runs the 9B through.
    ///
    /// The repository's default is a **Bambara language adapter** — the same
    /// worked example `candle-conversation`'s 9B preset carries. Deliberately
    /// mundane: what this gate proves is that the *adapted path runs and still
    /// does the task*, and any well-formed PEFT adapter over the seven
    /// projections proves exactly that. It happens to be trained against
    /// `unsloth/Qwen3.5-9B`, the same base lineage the pinned GGUF converts.
    ///
    /// Overridable as `checkpoints.Qwen35_9B_LoRA` so a deployment can run the
    /// gate against the adapter it actually serves rather than the example.
    pub(in super::super) fn qwen35_9b_lora() -> Checkpoint {
        model_overrides::checkpoint(
            "Qwen35_9B_LoRA",
            Checkpoint::new(
                "uknowae/bambara-qwen3.5-9b",
                "main",
                "adapter_model.safetensors",
            ),
        )
    }

    /// Fetch both of a PEFT adapter's files and return the directory holding
    /// them.
    ///
    /// `adapter_config.json` is fixed by PEFT; the weights file comes from the
    /// checkpoint so an override can name a differently-packaged one. They share
    /// a revision directory in the HF cache, so fetching each and taking the
    /// common parent is the whole of it — asserted rather than assumed, because
    /// a loader pointed at a directory containing only one of the two fails in a
    /// way that reads as a corrupt adapter.
    pub(in super::super) fn pinned_adapter(ck: &Checkpoint) -> Result<PathBuf> {
        let cfg = hf_get(
            &ck.repo,
            RepoType::Model,
            &ck.revision,
            "adapter_config.json",
        )?;
        let weights = hf_get(&ck.repo, RepoType::Model, &ck.revision, &ck.filename)?;
        let dir = cfg
            .parent()
            .ok_or_else(|| candle::Error::Msg(format!("{cfg:?} has no parent directory")))?;
        if weights.parent() != Some(dir) {
            candle::bail!("adapter files landed in different directories: {cfg:?} and {weights:?}");
        }
        Ok(dir.to_path_buf())
    }

    /// **The 9B story-rewrite ladder, run through a LoRA adapter.**
    ///
    /// The same gate as [`test_parallel_batched_forwarding_9b`] — same prompts,
    /// same configs (both call [`story_rewrite_ladder`]), same checkpoint — with
    /// every sequence opted into the adapter. What it proves is that the
    /// adapted path is a *working inference path*, not merely one that loads:
    /// the model still performs the rewrite, across the float modes and the
    /// whole C0–C10 KV ladder.
    ///
    /// # What a failure here means that the base gate cannot say
    ///
    /// The base gate exercises none of the adapter code. This one exercises all
    /// seven adapted projections on every layer that carries them — q/k/v/o on
    /// the eight attention layers, gate/up/down on all thirty-two — plus the
    /// consequence that makes them possible: an adapted layer reports
    /// `Int8Mode::Off`, so its activations stay float through the norm and its
    /// attention context is not emitted as q8a1024. That switch is invisible in
    /// the output. If it ever stops happening, `project_qkv` and the MLP say so
    /// by name instead of adapting the wrong tensor, and this is the gate that
    /// makes them say it.
    ///
    /// The comparison against the base run is the point of sharing the ladder:
    /// the adapter is a fine-tune, so the *prose* should differ while the task
    /// is still done. A run that fails the rewrite outright is the adapter
    /// wired wrongly; a run whose rows match the base run token for token is an
    /// adapter that is not being applied at all.
    ///
    /// # This is not a gate, and the name says so
    ///
    /// **The adapted ladder's thresholds were never derived.** It shares
    /// [`story_rewrite_ladder`] with the base gate, and that ladder's rungs are
    /// calibrated against the *base* checkpoint — `QWEN35_9B_KV_FACTORS` was
    /// measured with no adapter in the path. An adapted decode spends error
    /// budget the calibration did not allow for, so the top rungs fail on
    /// numbers nobody chose for them.
    ///
    /// A red result here therefore means "the adapted C-ladder is uncalibrated",
    /// which was already known, and not "something broke". A check that is
    /// always red teaches you to skip the colour, which is worse than having no
    /// check — so it is deliberately **outside** the
    /// `test_parallel_batched_forwarding_9b` name the gate command filters on,
    /// and is run by naming it.
    ///
    /// What it is still good for, today: proving the adapter loads, reaches
    /// every projection it should, and changes the prose without breaking the
    /// task. The low rungs answer that. Making it a gate means deriving
    /// adapted KV factors, which is a measurement pass, not a threshold to
    /// nudge.
    #[test]
    #[ignore = "NOT A GATE — the adapted C-ladder thresholds were never derived, so the top \
                rungs fail by construction (see the doc comment). Downloads the pinned \
                Qwen3.5-9B GGUF plus its LoRA adapter and needs a GPU. Run deliberately with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::lora_ladder_9b_uncalibrated \
                -- --ignored --nocapture --test-threads=1"]
    fn lora_ladder_9b_uncalibrated() -> Result<()> {
        println!("\n=== Qwen3.5-9B hybrid batched forwarding — LoRA ===\n");
        let adapter = qwen35_9b_lora();
        let adapter_dir = pinned_adapter(&adapter)?;
        println!("adapter: {}@{}", adapter.repo, adapter.revision);
        println!("adapter: {adapter_dir:?}");
        let device = Device::new_cuda(0)?;

        // The same resolution the base gate does, for the same reason: the
        // table's `int8` column must name what the model actually loaded. The
        // adapted layers override it to `Off` per layer — that is the adapter's
        // doing and is deliberately not reflected here, because this is the
        // mode the *checkpoint* was loaded in.
        let int8mode = Int8Mode::auto(&device);
        let params = TestParams::new(10, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_int8mode(int8mode)
            .with_lora("rp")
            .with_timeout_secs(1800);

        let pack = qwen35_9b(&device, Some(int8mode))?;
        let load = || {
            let mut m = from_pack(&pack, &device)?;
            // BF16: the width this lineage's residual stream flows in, and so
            // the width the adapter's `A` matmul consumes. Converting per
            // projection instead would be a full-tensor pass on the hot path.
            m.load_adapter("rp", &adapter_dir, candle::DType::BF16)?;
            println!("✓ Model + adapter loaded\n");
            Ok(m)
        };
        params.run(story_rewrite_ladder(), load)
    }

    /// **Depth on the 9B**: the batched forward at 32K and 128K of KV.
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-9B GGUF (7.5 GB) and runs a 128K-token prompt. \
                Run with: cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen35::tests::long_context_9b \
                -- --ignored --nocapture --test-threads=1"]
    fn long_context_9b() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};

        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let pack = qwen35_9b(&device, Some(int8mode))?;
        long_context_gate(
            "Qwen3.5-9B (hybrid dense)",
            int8mode,
            &tokenizer_json()?,
            Dialect::qwen35(),
            // `qwen35.context_length` in the GGUF.
            262_144,
            &[
                (
                    32_768,
                    &[InferenceMode::BF16, InferenceMode::C5, InferenceMode::C10][..],
                ),
                (131_072, &[InferenceMode::BF16, InferenceMode::C10][..]),
            ],
            1,
            64,
            DepthTask::Coherence,
            &device,
            || from_pack(&pack, &device),
        )
    }

    /// **Output fidelity against depth on the 9B** (`batch_test::fidelity`):
    /// C0, C5 and C10 against BF16, teacher-forced on the needle answer, from
    /// 4K to 128K. The hybrid's full-attention layers read the whole prefix,
    /// so every stored token reaches every step.
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-9B GGUF (7.5 GB) and prefills to 128K under four \
                modes. Run with: cargo test --release --features cuda -p candle-transformers \
                --lib quantized_qwen35::tests::fidelity_vs_depth_9b \
                -- --ignored --nocapture --test-threads=1"]
    fn fidelity_vs_depth_9b() -> Result<()> {
        use crate::models::batch_test::fidelity::{assert_sound, ladder, print_table};
        use crate::models::batched_model::ensure_vram_governor;
        use tokenizers::Tokenizer;

        let device = Device::new_cuda(0)?;
        let pack = qwen35_9b(&device, Some(Int8Mode::auto(&device)))?;
        let tok = Tokenizer::from_bytes(tokenizer_json()?.as_bytes())
            .map_err(|e| candle::Error::Msg(format!("tokenizer: {e}")))?;
        ensure_vram_governor(&device);
        let model = from_pack(&pack, &device)?;
        let modes = [InferenceMode::C0, InferenceMode::C5, InferenceMode::C10];
        let depths = [4_096, 16_384, 65_536, 131_072 - 4_096];
        let rows = ladder(&model, &tok, InferenceMode::BF16, &modes, &depths)?;
        print_table("Qwen3.5-9B (hybrid, whole-prefix attention)", &rows);
        assert_sound("Qwen3.5-9B", &rows, InferenceMode::BF16);
        Ok(())
    }

    /// Greedily decode the story rewrite on the 9B, driving `forward_wave`
    /// directly instead of through the harness.
    ///
    /// The gate assembles its prompt in pieces — system prefilled per length
    /// group, user content padded, the turn suffix appended separately — and
    /// samples through a stack of policy. That is a lot of machinery between a
    /// wrong answer and its cause. This builds the same prompt in one piece,
    /// takes argmax, and prints what comes out, so "the model will not do the
    /// task" and "the harness fed it something else" stop looking alike.
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-9B GGUF (7.5 GB) and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::story_rewrite_greedy_9b -- --ignored --nocapture"]
    fn story_rewrite_greedy_9b() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use candle::{DType, Tensor};

        let tok_json = tokenizer_json()?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(tok_json.as_bytes())
            .map_err(|e| candle::Error::Msg(format!("tokenizer: {e}")))?;
        let params = TestParams::new(10, &tok_json, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true);
        let mut prompt = params.system_prompt_tokens(0);
        prompt.extend(params.user_prompt_tokens(0));
        println!("prompt {} tokens; tail:", prompt.len());
        println!(
            "  {:?}",
            tokenizer
                .decode(&prompt[prompt.len() - 24..], false)
                .unwrap_or_default()
        );

        let device = Device::new_cuda(0)?;
        let model = from_pack(&qwen35_9b(&device, Some(Int8Mode::Off))?, &device)?;
        // BF16, not F32: the paged decode kernel is compiled for the half
        // types only, and this probe decodes.
        // A greedy continuation of `tokens`, `n` tokens long.
        let greedy = |tokens: &[u32], n: usize| -> Result<String> {
            let mut session =
                model.create_batched_session(BatchedConfig::default().with_dtype(DType::BF16))?;
            let seq = session.create_sequence()?;
            let ids = Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &device)?;
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                &[ids],
                &[],
                &[],
                0,
                model.num_layers(),
                None,
            )?;
            let mut tok = step.logits.as_ref().expect("logits")[0]
                .flatten_all()?
                .argmax(0)?
                .to_scalar::<u32>()?;
            drop(step);
            let mut out = vec![tok];
            for _ in 0..n {
                let ids = Tensor::from_vec(vec![tok], (1, 1), &device)?;
                let step = model.forward_wave(
                    &mut session,
                    &[seq],
                    &[ids],
                    &[],
                    &[],
                    &[],
                    &[],
                    0,
                    model.num_layers(),
                    None,
                )?;
                tok = step.logits.as_ref().expect("logits")[0]
                    .flatten_all()?
                    .argmax(0)?
                    .to_scalar::<u32>()?;
                drop(step);
                out.push(tok);
            }
            Ok(tokenizer.decode(&out, true).unwrap_or_default())
        };

        // Is the checkpoint being read at all? A factual one-liner separates
        // "these weights are decoded wrong" from "this model will not follow
        // the instruction" — the two look identical in the gate's diff.
        let sanity_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
                           <|im_start|>user\nThe capital of France is<|im_end|>\n\
                           <|im_start|>assistant\n<think>\n\n</think>\n\n";
        let sanity: Vec<u32> = tokenizer
            .encode(sanity_text, false)
            .map_err(|e| candle::Error::Msg(format!("encode: {e}")))?
            .get_ids()
            .to_vec();
        println!("--- sanity ---\n{}\n---", greedy(&sanity, 12)?);

        let mut session =
            model.create_batched_session(BatchedConfig::default().with_dtype(DType::BF16))?;
        let seq = session.create_sequence()?;

        let ids = Tensor::from_vec(prompt.clone(), (1, prompt.len()), &device)?;
        let step = model.forward_wave(
            &mut session,
            &[],
            &[],
            &[seq],
            &[ids],
            &[],
            &[],
            0,
            model.num_layers(),
            None,
        )?;
        let mut tok = step.logits.as_ref().expect("logits")[0]
            .flatten_all()?
            .argmax(0)?
            .to_scalar::<u32>()?;
        drop(step);

        let mut out = vec![tok];
        for _ in 0..24 {
            let ids = Tensor::from_vec(vec![tok], (1, 1), &device)?;
            let step = model.forward_wave(
                &mut session,
                &[seq],
                &[ids],
                &[],
                &[],
                &[],
                &[],
                0,
                model.num_layers(),
                None,
            )?;
            tok = step.logits.as_ref().expect("logits")[0]
                .flatten_all()?
                .argmax(0)?
                .to_scalar::<u32>()?;
            drop(step);
            out.push(tok);
        }
        let text = tokenizer.decode(&out, true).unwrap_or_default();
        println!("--- greedy continuation ---\n{text}\n---");
        assert!(
            text.trim_start().starts_with("The Backyard"),
            "the model did not begin the rewrite; it produced {text:?}"
        );
        Ok(())
    }
}
