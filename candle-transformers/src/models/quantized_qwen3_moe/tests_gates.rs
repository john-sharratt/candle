//! Model-level gate tests for Qwen3-MoE: the batched forward ladder, decode
//! reproducibility, the rollback the replay probe depends on, and long context.
//!
//! These load real checkpoints. The reproducibility test uses the same expert
//! cache as DeepSeek, so a non-reproducible result here would point to shared
//! infrastructure, while a clean result with the dense arm also clean points to
//! DeepSeek's own code. The replay test shows whether the rollback restores
//! identical state, which the probe needs in order to measure the model. The
//! long-context test runs at 32K and 128K of KV on the 2507 release.

use super::*;

// Tests

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::needless_question_mark, clippy::type_complexity)]

    use super::*;

    use crate::models::batch_test::utils::TestParams;

    use crate::models::batched_inference::InferenceMode;

    #[cfg(feature = "cuda")]
    use crate::models::batched_model::BatchedInference;

    use crate::models::dialect::Dialect;

    #[cfg(feature = "cuda")]
    use crate::models::rope_schedule::RopeSchedule;

    /// The 2507 refresh's declared window, native to its θ = 10⁷.
    #[cfg(feature = "cuda")]
    pub(in super::super) const INSTRUCT_2507_CONTEXT: usize = 262_144;

    /// A 30B-A3B checkpoint wrapped for batched inference on the frequencies
    /// its file states, to `context_length`. The tests here stay inside the
    /// trained window of whichever release they load, where every schedule is
    /// the file's own RoPE.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn batched(
        model: ModelWeights,
        context_length: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<BatchedInference<ModelWeights>> {
        let inv = model
            .rope_inv_freq()
            .ok_or_else(|| candle::Error::Msg("model has no inv_freq".into()))?;
        BatchedInference::new_with_schedule(
            model,
            &RopeSchedule::stated(inv, context_length)?,
            max_seq_len,
            device,
        )
    }

    /// The Qwen3-30B-A3B-Instruct-2507 Q4_K_M model pack at `mode` — `None` for
    /// the mode this card picks — built on first use.
    pub(in super::super) fn instruct_2507_pack(
        device: &Device,
        mode: Option<Int8Mode>,
    ) -> Result<std::path::PathBuf> {
        use crate::models::batch_test::test_helpers::gate_pack;
        use crate::models::model_pack::{PackFamily, PackRequest};
        gate_pack(
            &PackRequest::of(
                PackFamily::Routed,
                (
                    "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                    "",
                    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
                ),
                ("Qwen/Qwen3-30B-A3B-Instruct-2507", "main"),
                mode,
            ),
            device,
        )
    }

    /// **Does Qwen3-MoE decode reproduce itself, run to run?**
    ///
    /// DeepSeek-V4-Flash does not (`docs/archived/deepseek/deepseek_decode_reproducibility.md`).
    /// This test answers the question that result cannot: is the fault in
    /// shared infrastructure, or in DeepSeek-specific code?
    ///
    /// Qwen3-MoE routes through the **same `expert_lre` cache** as DeepSeek —
    /// the same residency, load and eviction machinery a depth bisect
    /// implicated (DeepSeek decode is bit-clean through the pinned layers 0–2
    /// and first fails at layer 3, exactly where experts become evictable).
    /// So: non-reproducible here ⇒ the fault is shared. Clean here, with the
    /// dense Llama arm also clean, ⇒ it is DeepSeek's own code, which is a far
    /// cheaper search.
    #[test]
    #[ignore]
    fn qwen3_moe_decode_is_reproducible() -> Result<()> {
        use crate::models::batch_test::utils::decode_reproducibility;

        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return Ok(());
        };
        // Loaded exactly as `test_parallel_batched_forwarding` does, so this
        // test differs from the established path in one variable only.
        let pack = match instruct_2507_pack(&device, Some(Int8Mode::Performance)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[skip] model unavailable: {e}");
                return Ok(());
            }
        };
        let weights = ModelWeights::from_pack(&pack, &device, None)?;
        let model = batched(weights, INSTRUCT_2507_CONTEXT, 4096, &device)?;

        // Fixed pseudo-token ids: the text is irrelevant, only that every pass
        // sees the identical input, so no tokenizer download is needed.
        let ids: Vec<u32> = (0..24u32).map(|i| (i * 37 + 11) % 2000 + 5).collect();
        let (agree, _) = decode_reproducibility(&model, &device, &ids, 32, 3, "qwen3-moe")?;
        assert!(
            agree,
            "Qwen3-MoE decode is not reproducible — it shares the expert_lre \
             cache with DeepSeek, so the fault is in SHARED infrastructure"
        );
        Ok(())
    }

    /// **The harness control for DeepSeek's replay probe.**
    ///
    /// `decode_replay_probe` replays a decode step through
    /// `truncate_sequence_to_tokens`, so it is only a measurement of the MODEL
    /// if that rollback restores byte-identical state. If it does not, the probe
    /// reports its own drift as non-determinism and every conclusion drawn from
    /// it is worthless. Running the identical code here — on a model whose token
    /// streams are reproducible — is what separates those two cases.
    ///
    /// Clean here ⇒ the harness is sound and DeepSeek's dirty depths are real.
    /// Dirty here ⇒ the probe is measuring itself, and DeepSeek's numbers must
    /// be thrown out rather than interpreted.
    #[test]
    #[ignore]
    fn qwen3_moe_decode_replay_is_bitwise_repeatable() -> Result<()> {
        use crate::models::batch_test::utils::{decode_replay_probe, prefill_replay_probe};

        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no CUDA device");
            return Ok(());
        };
        let pack = match instruct_2507_pack(&device, Some(Int8Mode::Performance)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[skip] model unavailable: {e}");
                return Ok(());
            }
        };
        let weights = ModelWeights::from_pack(&pack, &device, None)?;
        let model = batched(weights, INSTRUCT_2507_CONTEXT, 4096, &device)?;

        let ids: Vec<u32> = (0..24u32).map(|i| (i * 37 + 11) % 2000 + 5).collect();
        let dirty_prefill = prefill_replay_probe(&model, &device, &ids, 6, "qwen3-moe")?;
        let dirty = decode_replay_probe(&model, &device, &ids, 8, "qwen3-moe")?;
        assert_eq!(
            (dirty_prefill, dirty),
            (0, 0),
            "Qwen3-MoE is not bitwise repeatable: {dirty_prefill}/5 prefills and {dirty}/7 \
             decode replays diverged"
        );
        Ok(())
    }

    #[test]
    #[ignore] // Run with: cargo test --release --features cuda --lib --package candle-transformers quantized_qwen3_moe::tests::test_parallel_batched_forwarding -- --ignored --nocapture
    pub(in super::super) fn test_parallel_batched_forwarding() -> Result<()> {
        #[cfg(not(feature = "cuda"))]
        {
            println!("⚠ This test requires --features cuda");
            return Ok(());
        }

        println!("\n=== Qwen3-30B-A3B MoE Batched Forwarding Test ===\n");

        let num_generate_tokens = 10;
        let dialect = Dialect::chat_ml();

        // Download tokenizer (hf_get falls back from IPv6 to IPv4 if needed).
        use crate::models::batch_test::test_helpers::hf_get;
        let tokenizer_path = hf_get(
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )
        .map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to download tokenizer.json: {}. This test requires internet access.",
                e
            ))
        })?;
        let tokenizer_json = std::fs::read_to_string(&tokenizer_path).map_err(|e| {
            candle::Error::Msg(format!(
                "Failed to read tokenizer.json from {:?}: {}",
                tokenizer_path, e
            ))
        })?;

        let params = TestParams::new(num_generate_tokens, &tokenizer_json, dialect)
            .map_err(|e| candle::Error::Msg(format!("Failed to create TestParams: {}", e)))?
            .with_suppress_thinking(true)
            .with_print_outputs(false)
            .with_timeout_secs(1200); // 20 minutes

        println!("\n=== Loading Model ===\n");

        let device = Device::new_cuda(0).map_err(|e| {
            candle::Error::Msg(format!(
                "CUDA required for this test: {}. Use --features cuda",
                e
            ))
        })?;
        println!("Using device: {:?}\n", device);

        // One ladder, shared with the engine-driven table one crate up — see
        // `batched_forward_configs`.
        let configs = super::batched_forward_configs();

        // Inference numeric mode for the whole model — dense projections AND MoE experts (KO
        // twins): Performance, the same-width KO int8 the gate is calibrated at. One model load
        // per mode, which keeps the Markov expert predictor from being mixed across modes.
        let int8mode = Int8Mode::Performance;
        println!("int8 mode = {int8mode:?}\n");

        // The model pack, built on the first run and a read on every one after.
        let pack = instruct_2507_pack(&device, Some(int8mode))?;
        println!("Model pack: {:?}", pack);
        let load_model = || {
            let model = ModelWeights::from_pack(&pack, &device, None)?;
            println!("✓ Model loaded\n");
            batched(model, INSTRUCT_2507_CONTEXT, 4096, &device)
        };

        params.with_int8mode(int8mode).run(configs, load_model)?;

        Ok(())
    }

    /// **Depth on the 30B-A3B**: the batched forward at 32K and 128K of KV.
    ///
    /// The lineage's reference MoE and the checkpoint most of the published
    /// figures were measured on, so its depth curve is the one to compare the
    /// newer architectures against.
    #[test]
    #[ignore = "downloads the 30B-A3B Q4_K_M GGUF (~18 GB) and runs a 128K-token prompt. \
                Run with: cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen3_moe::tests::long_context_30b_a3b \
                -- --ignored --nocapture --test-threads=1"]
    fn long_context_30b_a3b() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batch_test::test_helpers::hf_get;

        let tokenizer_path = hf_get(
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            hf_hub::RepoType::Model,
            "main",
            "tokenizer.json",
        )
        .map_err(|e| candle::Error::Msg(format!("tokenizer.json: {e}")))?;
        let tokenizer_json = std::fs::read_to_string(&tokenizer_path)
            .map_err(|e| candle::Error::Msg(format!("read tokenizer.json: {e}")))?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let pack = instruct_2507_pack(&device, Some(int8mode))?;
        long_context_gate(
            "Qwen3-30B-A3B-Instruct-2507 (MoE, Q4_K_M)",
            int8mode,
            &tokenizer_json,
            Dialect::chat_ml(),
            // `qwen3moe.context_length` in the GGUF — the 2507 release is native
            // 262K, unlike the 32K original.
            INSTRUCT_2507_CONTEXT,
            // One shallow rung even though the window is wide: this is the
            // Qwen3-generation MoE, kept as the architectural comparison for
            // the Qwen3.5+ MoEs rather than as a depth subject of its own.
            &[(
                8_192,
                &[InferenceMode::BF16, InferenceMode::C5, InferenceMode::C10][..],
            )],
            1,
            64,
            DepthTask::Coherence,
            &device,
            || {
                let model = ModelWeights::from_pack(&pack, &device, None)?;
                batched(model, INSTRUCT_2507_CONTEXT, 4096, &device)
            },
        )
    }
}
