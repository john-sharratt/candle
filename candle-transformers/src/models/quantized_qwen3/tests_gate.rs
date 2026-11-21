//! Gates the Qwen3 model against its benchmark and fidelity rungs, and sets up the
//! ruler evaluation over the checkpoint.
//!
//! These tests are for real checkpoints and are not run by default. They cover the
//! batched forward at shallow and deeper contexts, the parallel batched forward, the
//! sequential and memory-mapped load benchmarks, output fidelity against depth, and
//! the ruler evaluation across compression modes. The context ladder stops at the
//! checkpoint's declared window, because reaching further would need position scaling
//! and the measurement would then describe a different model.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    use super::*;

    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};

    use crate::models::batched_inference::InferenceMode;

    use crate::models::batched_model::BatchedInference;

    use crate::models::dialect::Dialect;

    use crate::models::rope_schedule::RopePreset;

    use candle::quantized::gguf_file;

    /// A Qwen3 checkpoint wrapped for batched inference on the schedule its
    /// presets run: Qwen's progressive YaRN over the file's plain θ = 10⁶
    /// frequencies, trained to 32K, declared to 40,960.
    pub(in super::super) fn batched(
        model: ModelWeights,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<BatchedInference<ModelWeights>> {
        let inv = model
            .rope_inv_freq()
            .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
        let schedule =
            RopePreset::qwen3().gqa_schedule(inv, Some(1e6), DeclaredScaling::None, 40_960)?;
        BatchedInference::new_with_schedule(model, &schedule, max_seq_len, device)
    }

    #[test]
    #[ignore] // Run manually with: cargo test benchmark_large_model_sequential --release --features cuda -- --ignored --nocapture
    pub(in super::super) fn benchmark_large_model_sequential() -> Result<()> {
        // Benchmark large 7B-8B model with sequential loading
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        // Using Llama-3.2-3B (larger than 0.6B but not too huge for testing)
        let repo = api.model("bartowski/Llama-3.2-3B-Instruct-GGUF".to_string());

        let model_path = repo.get("Llama-3.2-3B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("\n=== Large Model Sequential Loading Benchmark ===");
        println!("Model: Llama-3.2-3B-Instruct-Q4_K_M");
        println!("Model path: {:?}\n", model_path);

        let device = Device::new_cuda(0).map_err(|e| {
            candle::Error::Msg(format!(
                "GPU required for this benchmark. CUDA error: {}",
                e
            ))
        })?;
        println!("Using device: {:?}\n", device);

        // Check file size
        let metadata = std::fs::metadata(&model_path)?;
        println!(
            "File size: {:.2} GB\n",
            metadata.len() as f64 / 1_000_000_000.0
        );

        // Warm up run
        println!("Warming up (loading once to populate OS cache)...");
        let mut file = std::fs::File::open(&model_path)?;
        let content = gguf_file::Content::read(&mut file)?;

        use crate::models::quantized_llama::ModelWeights as LlamaModelWeights;
        let _model = LlamaModelWeights::from_gguf_v3(content, &mut file, &device)?;
        println!("Warmup complete.\n");

        // Actual benchmark - run 3 times
        println!("Running 3 timed loads with File→RAM→GPU method...");
        let mut durations = Vec::new();
        for i in 0..3 {
            let mut file = std::fs::File::open(&model_path)?;
            let content = gguf_file::Content::read(&mut file)?;

            let start = std::time::Instant::now();
            let _model = LlamaModelWeights::from_gguf_v3(content, &mut file, &device)?;
            let duration = start.elapsed();

            println!("  Run {}: {:.3}s", i + 1, duration.as_secs_f64());
            durations.push(duration.as_secs_f64());
        }

        let avg = durations.iter().sum::<f64>() / durations.len() as f64;
        let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\n=== Sequential Loading Results (File→RAM→GPU) ===");
        println!("Average: {:.3}s", avg);
        println!("Min:     {:.3}s", min);
        println!("Max:     {:.3}s", max);

        Ok(())
    }

    #[test]
    #[ignore] // Run manually with: cargo test benchmark_large_model_mmap --release --features cuda -- --ignored --nocapture
    pub(in super::super) fn benchmark_large_model_mmap() -> Result<()> {
        // Benchmark large 7B-8B model with mmap loading
        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("Failed to initialize HF API: {}", e)))?;

        // Using Llama-3.2-3B (larger than 0.6B but not too huge for testing)
        let repo = api.model("bartowski/Llama-3.2-3B-Instruct-GGUF".to_string());

        let model_path = repo.get("Llama-3.2-3B-Instruct-Q4_K_M.gguf").map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download model: {}. This test requires internet access.",
                e
            ))
        })?;

        println!("\n=== Large Model mmap Loading Benchmark ===");
        println!("Model: Llama-3.2-3B-Instruct-Q4_K_M");
        println!("Model path: {:?}\n", model_path);

        let device = Device::new_cuda(0).map_err(|e| {
            candle::Error::Msg(format!(
                "GPU required for this benchmark. CUDA error: {}",
                e
            ))
        })?;
        println!("Using device: {:?}\n", device);

        // Check file size
        let metadata = std::fs::metadata(&model_path)?;
        println!(
            "File size: {:.2} GB\n",
            metadata.len() as f64 / 1_000_000_000.0
        );

        // Warm up run
        println!("Warming up (loading once to populate OS cache)...");
        use crate::models::quantized_llama::ModelWeights as LlamaModelWeights;
        let _model = LlamaModelWeights::from_gguf_by_path_v3(&model_path, &device)?;
        println!("Warmup complete.\n");

        // Actual benchmark - run 3 times
        println!("Running 3 timed loads with File→GPU direct (mmap) method...");
        let mut durations = Vec::new();
        for i in 0..3 {
            let start = std::time::Instant::now();
            let _model = LlamaModelWeights::from_gguf_by_path_v3(&model_path, &device)?;
            let duration = start.elapsed();

            println!("  Run {}: {:.3}s", i + 1, duration.as_secs_f64());
            durations.push(duration.as_secs_f64());
        }

        let avg = durations.iter().sum::<f64>() / durations.len() as f64;
        let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!("\n=== mmap Loading Results (File→GPU direct) ===");
        println!("Average: {:.3}s", avg);
        println!("Min:     {:.3}s", min);
        println!("Max:     {:.3}s", max);

        Ok(())
    }

    #[test]
    #[ignore] // Run with: cargo test --release --features cuda -- --ignored test_parallel_batched_forwarding
    #[cfg(feature = "cuda")]
    pub(in super::super) fn test_parallel_batched_forwarding() -> Result<()> {
        use crate::models::batch_test::test_helpers::plain_pack;

        println!("\n=== Setting up Test Parameters ===\n");

        let num_generate_tokens = 20;
        let dialect = Dialect::chat_ml();

        let device = Device::new_cuda(0)?;
        println!("Using device: {:?}\n", device);
        let int8mode = Int8Mode::Performance;
        println!("int8 mode = {int8mode:?}\n");

        println!("\n=== Loading Model ===\n");
        // The checkpoint's model pack, with the tokenizer inside it — loaded at
        // runtime so the test validates real token ranges.
        let (model_path, tokenizer_json) = plain_pack(
            ("unsloth/Qwen3-8B-GGUF", "main", "Qwen3-8B-Q6_K.gguf"),
            ("Qwen/Qwen3-8B", "main"),
            int8mode,
            &device,
        )?;
        println!("Model pack: {:?}", model_path);

        let params = TestParams::new(num_generate_tokens, &tokenizer_json, dialect)
            .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?
            .with_suppress_thinking(true)
            .with_print_outputs(false)
            .with_timeout_secs(1200); // 8 minutes for this test (Qwen3-8B is large)

        let configs = vec![
            // Sequential (non-batched) test using BF16 for better performance on tensor cores
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: false,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: None,
            },
            // Batched tests with F16 and BF16 - single repeat for speed
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                test_mode: None,
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 16,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 4,
                num_repeats: 1,
                test_mode: None,
            },
            #[cfg(feature = "huge-context")]
            TestConfig {
                mode: InferenceMode::Q4_0,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 32,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 4,
                num_repeats: 1,
                test_mode: None,
            },
            TestConfig {
                mode: InferenceMode::C0,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C1,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C2,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C3,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C4,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C6,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 10,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C7,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 32,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C8,
                use_batched: true,
                #[cfg(feature = "huge-context")]
                num_contexts: 128,
                #[cfg(not(feature = "huge-context"))]
                num_contexts: 10,
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
            TestConfig {
                mode: InferenceMode::C10,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
        ];

        let load_model = || {
            let model = ModelWeights::from_gguf_by_path_with_int8(&model_path, &device, int8mode)?;
            println!("✓ Model loaded\n");
            batched(model, 4096, &device)
        };

        params.with_int8mode(int8mode).run(configs, load_model)?;

        Ok(())
    }

    /// **Depth on the 8B**: the batched forward at 8K and 32K of KV. The
    /// classic-attention dense baseline the hybrid lineage is measured against.
    ///
    /// Shallower than the hybrid models' ladders because the checkpoint is:
    /// Qwen3's trained window is 40,960, and reaching 128K on it would mean
    /// scaling the positions, which changes what the row measures.
    #[test]
    #[ignore = "downloads the Qwen3-8B Q6_K GGUF and prefills 32K of its 40,960 window. \
                Run with: cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen3::tests::long_context_8b \
                -- --ignored --nocapture --test-threads=1"]
    fn long_context_8b() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};

        let api = crate::models::batch_test::test_helpers::api()
            .map_err(|e| candle::Error::Msg(format!("HF API: {e}")))?;
        let tok_repo = api.model("Qwen/Qwen3-8B".to_string());
        let tokenizer_path = tok_repo
            .get("tokenizer.json")
            .map_err(|e| candle::Error::Msg(format!("tokenizer.json: {e}")))?;
        let tokenizer_json = std::fs::read_to_string(&tokenizer_path)
            .map_err(|e| candle::Error::Msg(format!("read tokenizer.json: {e}")))?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            "unsloth/Qwen3-8B-GGUF".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        let model_path = repo
            .get("Qwen3-8B-Q6_K.gguf")
            .map_err(|e| candle::Error::Msg(format!("model: {e}")))?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        long_context_gate(
            "Qwen3-8B (dense, Q6_K)",
            int8mode,
            &tokenizer_json,
            Dialect::chat_ml(),
            // `qwen3.context_length` in the GGUF. Qwen3 reaches 131K only under
            // YaRN, which this fleet does not use.
            40_960,
            // One shallow rung; the depth ladder belongs to the Qwen3.5+
            // checkpoints, which hold 262K without scaling anything.
            &[(
                8_192,
                &[InferenceMode::BF16, InferenceMode::C5, InferenceMode::C10][..],
            )],
            1,
            64,
            DepthTask::Coherence,
            &device,
            || {
                let model =
                    ModelWeights::from_gguf_by_path_with_int8(&model_path, &device, int8mode)?;
                batched(model, 4096, &device)
            },
        )
    }

    /// **Output fidelity against depth on the 8B** (`batch_test::fidelity`):
    /// C0, C5 and C10 against BF16, teacher-forced on the needle answer, from
    /// 4K to 32K of its 40,960 window. Classic full attention: every layer reads
    /// every stored token.
    #[test]
    #[ignore = "downloads the Qwen3-8B Q6_K GGUF and prefills to 32K under four modes. \
                Run with: cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen3::tests::fidelity_vs_depth_8b \
                -- --ignored --nocapture --test-threads=1"]
    fn fidelity_vs_depth_8b() -> Result<()> {
        use crate::models::batch_test::fidelity::{assert_sound, ladder, print_table};
        use crate::models::batch_test::test_helpers::api;
        use crate::models::batched_model::ensure_vram_governor;
        use hf_hub::{Repo, RepoType};
        use tokenizers::Tokenizer;

        let api = api().map_err(|e| candle::Error::Msg(format!("HF API: {e}")))?;
        let tokenizer_path = api
            .model("Qwen/Qwen3-8B".to_string())
            .get("tokenizer.json")
            .map_err(|e| candle::Error::Msg(format!("tokenizer.json: {e}")))?;
        let tok = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| candle::Error::Msg(format!("tokenizer: {e}")))?;
        let model_path = api
            .repo(Repo::with_revision(
                "unsloth/Qwen3-8B-GGUF".to_string(),
                RepoType::Model,
                "main".to_string(),
            ))
            .get("Qwen3-8B-Q6_K.gguf")
            .map_err(|e| candle::Error::Msg(format!("model: {e}")))?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        ensure_vram_governor(&device);
        let model = batched(
            ModelWeights::from_gguf_by_path_with_int8(&model_path, &device, int8mode)?,
            4096,
            &device,
        )?;
        let modes = [InferenceMode::C0, InferenceMode::C5, InferenceMode::C10];
        let depths = [4_096, 8_192, 16_384, 32_768];
        let rows = ladder(&model, &tok, InferenceMode::BF16, &modes, &depths)?;
        print_table("Qwen3-8B (full attention)", &rows);
        assert_sound("Qwen3-8B", &rows, InferenceMode::BF16);
        Ok(())
    }

    #[test]
    #[ignore] // cargo test --release --features cuda --lib -p candle-transformers quantized_qwen3::tests::test_ruler_eval -- --ignored --nocapture
    pub(in super::super) fn test_ruler_eval() -> Result<()> {
        use crate::models::batch_test::ruler_gen::{
            run_ruler_benchmark, RulerBenchConfig, RulerDataSource, RulerTask, QWEN3_EOS_IDS,
        };
        use crate::models::batch_test::test_helpers::{
            download_hf_gguf, load_hf_tokenizer, open_gguf,
        };

        // Change GGUF_VARIANT to swap model weight quantisation (Q4_0 < Q4_K_M by ~240 MB).
        const GGUF_VARIANT: &str = "Q6_K"; // options: Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0

        let tokenizer = load_hf_tokenizer("Qwen/Qwen3-8B")?;
        let device =
            Device::new_cuda(0).map_err(|e| candle::Error::Msg(format!("CUDA device: {e}")))?;
        let gguf_filename = format!("Qwen3-8B-{GGUF_VARIANT}.gguf");
        let model_path = download_hf_gguf("unsloth/Qwen3-8B-GGUF", &gguf_filename, "main")?;
        println!("Model path: {model_path:?}");
        let (content, mut file) = open_gguf(&model_path)?;
        let weights = ModelWeights::from_gguf(content, &mut file, &device)?;
        println!("✓ Model loaded");
        let model = batched(weights, 98_304, &device)?;

        // Phase A: 4K context, high concurrency (budget = 24×4096 = 98304).
        // Target batches: Q8→24, C5→~64, C8→~115, C9/C10→higher.
        // After results lock in at 4K, bump to: 8K(12), 16K(6), 32K(3).
        let model_name = format!("Qwen3-8B-{GGUF_VARIANT}");
        let cfg = RulerBenchConfig {
            model_name: &model_name,
            eos_ids: QWEN3_EOS_IDS,
            token_budget: 98_304,
            max_gen_tokens: 50,
            modes: &[
                (Some(InferenceMode::Q8_0), "Q8_0"),
                (Some(InferenceMode::Q8_Q4), "Q8/Q4"),
                (Some(InferenceMode::Q4_0), "Q4_0"),
                (Some(InferenceMode::C5), "C5"),
                (Some(InferenceMode::C8), "C8"),
                (Some(InferenceMode::C9), "C9"),
                (Some(InferenceMode::C10), "C10"),
            ],
            // Per-length sample counts sized for ~6 min each (÷2 per 2× context).
            // Adjust after first run if actual timings differ.
            lengths_samples: &[(4_096, 8), (8_192, 4), (16_384, 2), (32_768, 1)],
            tasks: &[
                RulerTask::NiahSingle1,
                RulerTask::NiahMultiKey2,
                RulerTask::Vt,
                RulerTask::Cwe,
            ],
            data_source: RulerDataSource::Generated,
        };

        run_ruler_benchmark(&model, &tokenizer, &cfg)
    }
}
