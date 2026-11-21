//! Ignored profiling tests for the Qwen3-MoE decode path: a per-operation decode
//! profile, a batch-size sweep and a routing trace capture.
//!
//! `decode_profile` prefills a realistic context, times single-token decode
//! steps, and prints the per-operation breakdown. The sweep over one, four and
//! sixteen sessions shows whether the forward is bound by launches: a step time
//! that barely grows with the batch means the launches are amortised. The
//! routing capture writes the trace that the offline predictor evaluation
//! replays.

use super::*;

// Tests

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::needless_question_mark, clippy::type_complexity)]

    use super::*;

    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};

    use crate::models::batched_inference::InferenceMode;

    use crate::models::dialect::Dialect;

    use super::super::tests_gates::tests::{batched, instruct_2507_pack, INSTRUCT_2507_CONTEXT};

    /// Per-op decode profiler: prefill a realistic context, then time N single-token
    /// decode steps and dump the `snapshot_profiles` op breakdown so we can see where
    /// the per-token forward spends its time (attention kernel vs qkv/o_proj vs the
    /// MoE route, bucketize and expert GEMMs). Iterates in ~1 min so
    /// it drives forward-latency optimization without the daemon's substrate reload.
    ///
    /// Run with the profiler enabled:
    ///   cargo test -p candle-transformers --release --features cuda,profile --lib \
    ///     quantized_qwen3_moe::tests::decode_profile -- --ignored --nocapture
    #[test]
    #[ignore]
    fn decode_profile() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            println!("⚠ decode_profile requires --features cuda,profile");
            Ok(())
        }
        #[cfg(feature = "cuda")]
        {
            use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};

            let device = match Device::new_cuda(0) {
                Ok(d) => d,
                Err(_) => {
                    println!("skip: no CUDA device");
                    return Ok(());
                }
            };
            let pack = instruct_2507_pack(&device, None)?;
            let raw = ModelWeights::from_pack(&pack, &device, None)?;
            let model = batched(raw, INSTRUCT_2507_CONTEXT, 4096, &device)?;
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let mk = |t: &[u32]| -> Result<Tensor> { Tensor::new(t, &device)?.unsqueeze(0) };

            // Batch-size sweep to expose scaling: decode B sequences per step for
            // B in {1, 4, 16}. If the forward is launch-bound, ms/STEP barely grows
            // with B (launches amortize) → ms/token drops ~B× → near-linear tok/s.
            // Sub-linear tok/s ⇒ per-session (non-amortized) work dominates.
            let ctx: Vec<u32> = (0..3000u32).map(|i| (i % 2900) + 100).collect();
            let n_steps = 60usize;
            let nl = model.num_layers();
            let mut baseline_ms_step = 0.0f64;
            for (bi, &batch) in [1usize, 4, 16].iter().enumerate() {
                let seqs: Vec<usize> = (0..batch)
                    .map(|_| session.create_sequence())
                    .collect::<Result<_>>()?;
                for &sq in &seqs {
                    // Context prefill (q>1): prefill group.
                    let _ = model.forward_wave(
                        &mut session,
                        &[],
                        &[],
                        &[sq],
                        &[mk(&ctx)?],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?;
                    session.advance_sequence(sq, ctx.len())?;
                }
                let inputs =
                    |t: u32| -> Result<Vec<Tensor>> { (0..batch).map(|_| mk(&[t])).collect() };
                let mut tok = 42u32;
                for _ in 0..6 {
                    // Decode step (q=1): decode group — the path this benchmark times.
                    let _ = model.forward_wave(
                        &mut session,
                        &seqs,
                        &inputs(tok)?,
                        &[],
                        &[],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?;
                    for &sq in &seqs {
                        session.advance_sequence(sq, 1)?;
                    }
                    tok = (tok + 1) % 2900 + 100;
                }
                let _ = model.snapshot_profiles();
                let _ = crate::models::profile::pipeline_snapshot_and_reset();
                let t0 = std::time::Instant::now();
                for _ in 0..n_steps {
                    let _ = model.forward_wave(
                        &mut session,
                        &seqs,
                        &inputs(tok)?,
                        &[],
                        &[],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )?;
                    for &sq in &seqs {
                        session.advance_sequence(sq, 1)?;
                    }
                    tok = (tok + 1) % 2900 + 100;
                }
                let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let ms_step = total_ms / n_steps as f64;
                let ms_tok = ms_step / batch as f64;
                let tok_s = 1000.0 / ms_tok;
                if bi == 0 {
                    baseline_ms_step = ms_step;
                }
                let scale = if bi == 0 {
                    1.0
                } else {
                    (1000.0 / ms_tok) / (1000.0 / (baseline_ms_step)) // tok/s vs B=1 tok/s
                };
                println!(
                    "B={batch:2}: {ms_step:6.2} ms/step | {ms_tok:5.2} ms/token | {tok_s:7.0} tok/s | scale×{scale:.1} (ideal ×{batch})"
                );
                // Per-op breakdown only for B=1 (needs --features profile).
                if bi == 0 {
                    let mut snap = model.snapshot_profiles();
                    snap.merge(&crate::models::profile::pipeline_snapshot_and_reset());
                    let mut rows = snap.entries.clone();
                    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    for (name, ms, cnt) in rows.iter().take(8) {
                        println!("    {name:24} {:6.3} ms/tok  x{cnt}", ms / n_steps as f64);
                    }
                }
            }
            Ok(())
        }
    }

    /// Capture a real MoE routing trace for offline predictor evaluation.
    ///
    /// Runs a single BF16×1 StoryRewrite generation against Qwen3-30B-A3B with
    /// routing capture enabled, then bincode + gzip the trace to the checked-in
    /// fixture ([`routing_capture::FIXTURE_PATH`]).  The offline eval
    /// (`expert_lre::eval`) replays that fixture on CPU in milliseconds.
    ///
    /// Run with:
    ///   cargo test --release --features cuda --lib --package candle-transformers \
    ///     quantized_qwen3_moe::tests::capture_routing_trace -- --ignored --nocapture
    #[test]
    #[ignore]
    fn capture_routing_trace() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            println!("⚠ This test requires --features cuda");
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        {
            use crate::models::routing_capture;
            use std::io::Write;

            println!("\n=== Capturing Qwen3-30B-A3B MoE routing trace ===\n");

            let dialect = Dialect::chat_ml();
            use crate::models::batch_test::test_helpers::hf_get;
            let tokenizer_path = hf_get(
                "Qwen/Qwen3-30B-A3B-Instruct-2507",
                hf_hub::RepoType::Model,
                "main",
                "tokenizer.json",
            )
            .map_err(|e| candle::Error::Msg(format!("Failed to download tokenizer.json: {}", e)))?;
            let tokenizer_json = std::fs::read_to_string(&tokenizer_path)
                .map_err(|e| candle::Error::Msg(format!("Failed to read tokenizer.json: {}", e)))?;

            // Diverse prompts driven through a single model load.  Config 0
            // keeps the default StoryRewrite prompt; the rest override it.
            let prompts: Vec<String> = [
                "", // config 0 → story.md (StoryRewrite default)
                "Tell me about the candle repository and how it works",
                "How does AI implement inference in modern engines",
                "Output me a rust program example the is a basic API skeleton",
                "Write example code for a bubble sort of Int64 numbers",
                "Give me a list of jobs that are safe from AI automation",
                "Explain how a B-tree database index works and when to use one",
                "Write a Python function that merges two sorted linked lists",
                "What caused the fall of the Western Roman Empire?",
                "Summarize the key differences between TCP and UDP",
                "Write a haiku about a thunderstorm over the ocean",
                "Implement a least-recently-used cache in Rust with a fixed capacity",
                "Describe how photosynthesis converts sunlight into chemical energy",
                "Compare and contrast supervised and unsupervised machine learning",
                "Give me a step-by-step recipe for a classic margherita pizza",
                "Explain the difference between mutexes and semaphores in concurrency",
                "What are the tradeoffs between monolithic and microservice architectures?",
                "Write a SQL query to find the second highest salary in an employees table",
                "Explain how vaccines train the human immune system",
                "Describe the plot of a short mystery story set on a moving train",
                "List strategies for reducing tail latency in a distributed web service",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();

            // Stop each generation at EOS to avoid post-EOS degenerate routing.
            let eos_tokens: Vec<u32> = {
                let tok = tokenizers::Tokenizer::from_bytes(tokenizer_json.as_bytes()).unwrap();
                ["<|im_end|>", "<|endoftext|>"]
                    .iter()
                    .filter_map(|s| tok.token_to_id(s))
                    .collect()
            };

            let params = TestParams::new(256, &tokenizer_json, dialect)
                .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?
                .with_suppress_thinking(true)
                .with_print_outputs(false)
                .with_timeout_secs(3600)
                .with_per_config_prompts(prompts.clone())
                .with_stop_on_eos(eos_tokens);

            let device = Device::new_cuda(0)
                .map_err(|e| candle::Error::Msg(format!("CUDA required: {}", e)))?;
            let pack = instruct_2507_pack(&device, None)?;

            // One config per prompt: BF16, single context, no validation.
            let configs: Vec<TestConfig> = (0..prompts.len())
                .map(|_| TestConfig {
                    mode: InferenceMode::BF16,
                    use_batched: true,
                    num_contexts: 1,
                    num_repeats: 1,
                    test_mode: Some(TestMode::Skip),
                })
                .collect();

            let load_model = || {
                let model = ModelWeights::from_pack(&pack, &device, None)?;
                batched(model, INSTRUCT_2507_CONTEXT, 4096, &device)
            };

            // Capture every config (each is a distinct prompt, tagged by index).
            // Validation outcome is irrelevant here — we only want the routing
            // trace, so a harness validation error must not lose captured data.
            routing_capture::enable_all();
            let run_result = params.run(configs, load_model);
            routing_capture::disable();
            if let Err(e) = &run_result {
                println!("(harness reported: {e} — ignored; writing captured trace)");
            }

            let records = routing_capture::take();
            println!("Captured {} routing records", records.len());
            assert!(!records.is_empty(), "no routing records captured");

            // bincode → gzip → fixture.
            let raw = bincode::serialize(&records)
                .map_err(|e| candle::Error::Msg(format!("bincode serialize failed: {}", e)))?;
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder
                .write_all(&raw)
                .map_err(|e| candle::Error::Msg(format!("gzip write failed: {}", e)))?;
            let compressed = encoder
                .finish()
                .map_err(|e| candle::Error::Msg(format!("gzip finish failed: {}", e)))?;

            let path = std::path::Path::new(routing_capture::FIXTURE_PATH);
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| candle::Error::Msg(format!("mkdir fixtures failed: {}", e)))?;
            }
            std::fs::write(path, &compressed)
                .map_err(|e| candle::Error::Msg(format!("write fixture failed: {}", e)))?;

            println!(
                "Wrote {} records → {} ({} bytes raw, {} bytes gzip)",
                records.len(),
                routing_capture::FIXTURE_PATH,
                raw.len(),
                compressed.len(),
            );
            Ok(())
        }
    }

    // ── RULER long-context benchmark ──────────────────────────────────────────

    /// End-to-end RULER evaluation for Qwen3-30B-A3B.
    ///
    /// Generates RULER task data in-process (no external Python required),
    /// runs batched inference for four compression modes, and prints an
    /// accuracy table.
    ///
    /// Run with:
    ///   cargo test --release --features cuda,verbose --lib --package candle-transformers \
    ///     quantized_qwen3_moe::tests::test_ruler_eval -- --ignored --nocapture
    #[test]
    #[ignore]
    fn test_ruler_eval() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        println!("⚠  No CUDA — performance will be poor");

        use crate::models::batch_test::ruler_gen::{
            run_ruler_benchmark, RulerBenchConfig, RulerDataSource, RulerTask, QWEN3_EOS_IDS,
        };
        use crate::models::batch_test::test_helpers::load_hf_tokenizer;

        let tokenizer = load_hf_tokenizer("Qwen/Qwen3-30B-A3B-Instruct-2507")?;
        let device =
            Device::new_cuda(0).map_err(|e| candle::Error::Msg(format!("CUDA device: {e}")))?;
        let pack = instruct_2507_pack(&device, None)?;
        println!("Model pack: {pack:?}");
        let weights = ModelWeights::from_pack(&pack, &device, None)?;
        let model = batched(weights, INSTRUCT_2507_CONTEXT, 32_768, &device)?;
        println!("✓ Model loaded");

        let cfg = RulerBenchConfig {
            model_name: "Qwen3-30B-A3B-Q4_K_M",
            eos_ids: QWEN3_EOS_IDS,
            token_budget: 32_768,
            max_gen_tokens: 50,
            modes: &[
                (None, "F16 (lossless)"),
                (Some(InferenceMode::Q4_0), "Q4_0 (4.5 BPE)"),
                (Some(InferenceMode::C5), "C5  (PalQuant ~4.4 BPE)"),
                (Some(InferenceMode::Q3_0), "Q3_0 (3.5 BPE)"),
                (Some(InferenceMode::C8), "C8  (PalQuant ~3.3 BPE)"),
            ],
            lengths_samples: &[(4_096, 20), (8_192, 10), (16_384, 5), (32_768, 3)],
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
