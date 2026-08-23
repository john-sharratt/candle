//! Qwen3.5 dense hybrids (0.8B, 9B) — the model file.
//!
//! The Qwen3.5 family is a hybrid Gated-DeltaNet + full-attention lineage
//! (3 recurrent layers per attention layer, gated GQA at `head_dim 256` with
//! partial rotary). All of the machinery — config parsing, the DeltaNet
//! mixer, the wave sweep, the loaders — is shared across the lineage and
//! lives in [`super::qwen35`]; this file pins what makes the *dense* models
//! themselves: their checkpoints, their tokenizer, and the gate tests that
//! hold the engine to exact outputs on them.
//!
//! The dense models are the hybrid engine's proving ground: the 0.8B is
//! unquantized BF16 (no quantization in the picture at all), and the 9B runs
//! the same sweep as the 35B with the expert cache out of the picture — so a
//! failure here is the hybrid, and a failure only on the MoE sibling is the
//! experts.

use std::path::Path;

use candle::{Device, Result};
use candle_nn::kv_cache::{QWEN35_0_8B_KV_FACTORS, QWEN35_9B_KV_FACTORS};

use super::qwen35::{load_hybrid_gguf, HybridBatched, Qwen35LoadOptions};

/// The tokenizer repo, pinned. The GGUF ships `tokenizer.ggml.tokens` but no
/// `tokenizer.json`, and the family shares one tokenizer across sizes — the
/// reference loader proves the two agree token for token.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.5-0.8B";
pub const TOKENIZER_REV: &str = "2fc06364715b967f1860aea9cf38778875588b17";

/// Pinned checkpoints (repo, revision, file). Revisions are pinned because an
/// upstream re-upload once silently invalidated a threshold tuning — a gate
/// must fail loudly on a checkpoint change, not drift.
pub const QWEN35_0_8B: (&str, &str, &str) = (
    "unsloth/Qwen3.5-0.8B-GGUF",
    "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
    "Qwen3.5-0.8B-BF16.gguf",
);
pub const QWEN35_9B: (&str, &str, &str) = (
    "unsloth/Qwen3.5-9B-GGUF",
    "3885219b6810b007914f3a7950a8d1b469d598a5",
    "Qwen3.5-9B-Q6_K.gguf",
);

/// Load a dense Qwen3.5 checkpoint (0.8B / 9B) and wrap it for the scheduler.
///
/// This is the dense models' concrete entry: it refuses a routed checkpoint
/// (a MoE file loaded here would stand up an expert cache the caller did not
/// plan VRAM for) and constructs the lineage's [`HybridBatched`] with the
/// checkpoint's own derived KV threshold factor row. The two pinned dense
/// models are told apart by width — the 0.8B is `hidden 1024`, the 9B
/// `hidden 4096` — and an unpinned dense sibling gets the row of the nearer
/// size, which is the closest calibration that exists for it.
pub fn from_gguf_path(
    file_path: &Path,
    device: &Device,
    options: Qwen35LoadOptions,
) -> Result<HybridBatched> {
    let model = load_hybrid_gguf(file_path, device, options)?;
    if model.cfg.moe.is_some() {
        candle::bail!(
            "quantized_qwen35: {file_path:?} is a routed (MoE) checkpoint — \
             load it through quantized_qwen35_moe instead"
        );
    }
    let kv_factors = if model.cfg.hidden_size >= 4096 {
        QWEN35_9B_KV_FACTORS
    } else {
        QWEN35_0_8B_KV_FACTORS
    };
    HybridBatched::new(model, kv_factors)
}

/// Shared by the sibling model files: the pinned family tokenizer as a JSON
/// string.
#[cfg(test)]
pub(crate) fn tokenizer_json() -> Result<String> {
    use crate::models::batch_test::test_helpers::hf_get;
    use hf_hub::RepoType;
    let p = hf_get(
        TOKENIZER_REPO,
        RepoType::Model,
        TOKENIZER_REV,
        "tokenizer.json",
    )?;
    std::fs::read_to_string(&p).map_err(|e| candle::Error::Msg(format!("read {p:?}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
    use crate::models::batched_inference::InferenceMode;
    use crate::models::dialect::Dialect;
    use candle::quantized::Int8Mode;
    use hf_hub::RepoType;

    fn pinned(spec: (&str, &str, &str)) -> Result<std::path::PathBuf> {
        hf_get(spec.0, RepoType::Model, spec.1, spec.2)
    }

    /// Prefill and decode rate for one model, measured on the engine directly.
    ///
    /// The gate's throughput figures include everything the harness does around
    /// a forward — tokenising, slicing the prompt into waves, sampling policy,
    /// validation — so a slow gate does not say whether the engine or the
    /// scaffolding is slow. This drives `forward_wave` itself: one wave for the
    /// whole prompt, then N single-token waves, with a device sync around each
    /// so the timings are of completed work rather than of launches.
    fn measure(model: &HybridBatched, prompt: &[u32], decode_steps: usize) -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use candle::{DType, Tensor};
        use std::time::Instant;

        let device = model.device().clone();
        let mut session =
            model.create_batched_session(BatchedConfig::default().with_dtype(DType::BF16))?;
        let seq = session.create_sequence()?;

        let ids = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &device)?;
        device.synchronize()?;
        let t0 = Instant::now();
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
        device.synchronize()?;
        let prefill = t0.elapsed();

        device.synchronize()?;
        let t1 = Instant::now();
        for _ in 0..decode_steps {
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
        }
        device.synchronize()?;
        let decode = t1.elapsed();

        println!(
            "  prefill {:>6} tok in {:>7.1} ms = {:>8.1} t/s   |   decode {:>3} tok in {:>7.1} ms = {:>6.1} t/s",
            prompt.len(),
            prefill.as_secs_f64() * 1e3,
            prompt.len() as f64 / prefill.as_secs_f64(),
            decode_steps,
            decode.as_secs_f64() * 1e3,
            decode_steps as f64 / decode.as_secs_f64(),
        );
        Ok(())
    }

    /// Per-layer prefill cost, split by layer kind.
    ///
    /// The wave is re-entrant over a layer range, so the prompt can be pushed
    /// through one layer at a time with a sync between each. That adds the
    /// per-wave phase overhead to every layer and so overstates the total, but
    /// it is the *distribution* this is for: which kind of layer the prefill
    /// time is actually in.
    fn profile_layers(model: &HybridBatched, prompt: &[u32]) -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, ManagedBatchedModel};
        use crate::models::delta_net::LayerKind;
        use candle::{DType, Tensor};
        use std::time::Instant;

        let device = model.device().clone();
        let mut session =
            model.create_batched_session(BatchedConfig::default().with_dtype(DType::BF16))?;
        let seq = session.create_sequence()?;
        let ids = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), &device)?;

        let kinds = model.model().cfg.layer_kinds.clone();
        let n = kinds.len();
        let mut residual: Option<Tensor> = None;
        let (mut attn_ms, mut delta_ms) = (0f64, 0f64);
        let (mut attn_n, mut delta_n) = (0usize, 0usize);
        for (li, kind) in kinds.iter().enumerate().take(n) {
            device.synchronize()?;
            let t = Instant::now();
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                std::slice::from_ref(&ids),
                &[],
                &[],
                li,
                li + 1,
                residual.take(),
            )?;
            residual = step.residual.clone();
            drop(step);
            device.synchronize()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            match kind {
                LayerKind::Attention => {
                    attn_ms += ms;
                    attn_n += 1;
                }
                LayerKind::DeltaNet => {
                    delta_ms += ms;
                    delta_n += 1;
                }
            }
        }
        println!(
            "    prefill: attention {attn_n:>2} layers {attn_ms:>8.1} ms ({:>6.1} ms each)   \
             deltanet {delta_n:>2} layers {delta_ms:>8.1} ms ({:>6.1} ms each)",
            attn_ms / attn_n.max(1) as f64,
            delta_ms / delta_n.max(1) as f64,
        );

        // The same split for a single-token step. Decode shares no kernel with
        // prefill on either side — a fused one-token recurrence instead of the
        // chunked scan, the paged decode kernel instead of the float fallback —
        // so its distribution has to be measured separately.
        let mut tok = 1000u32;
        let mut residual: Option<Tensor> = None;
        let (mut a_ms, mut d_ms) = (0f64, 0f64);
        for (li, kind) in kinds.iter().enumerate().take(n) {
            let ids = Tensor::from_vec(vec![tok], (1, 1), &device)?;
            device.synchronize()?;
            let t = Instant::now();
            let step = model.forward_wave(
                &mut session,
                &[seq],
                &[ids],
                &[],
                &[],
                &[],
                &[],
                li,
                li + 1,
                residual.take(),
            )?;
            residual = step.residual.clone();
            if let Some(l) = step.logits.as_ref() {
                tok = l[0].flatten_all()?.argmax(0)?.to_scalar::<u32>()?;
            }
            drop(step);
            device.synchronize()?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            match kind {
                LayerKind::Attention => a_ms += ms,
                LayerKind::DeltaNet => d_ms += ms,
            }
        }
        println!(
            "    decode : attention {attn_n:>2} layers {a_ms:>8.2} ms ({:>6.2} ms each)   \
             deltanet {delta_n:>2} layers {d_ms:>8.2} ms ({:>6.2} ms each)",
            a_ms / attn_n.max(1) as f64,
            d_ms / delta_n.max(1) as f64,
        );
        Ok(())
    }

    /// Engine throughput on the 0.8B and 9B, apart from the gate harness.
    #[test]
    #[ignore = "downloads the pinned 0.8B and 9B GGUFs and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::engine_throughput -- --ignored --nocapture"]
    fn engine_throughput() -> Result<()> {
        let tok_json = tokenizer_json()?;
        let params = TestParams::new(10, &tok_json, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true);
        let mut prompt = params.system_prompt_tokens(0);
        prompt.extend(params.user_prompt_tokens(0));
        let device = Device::new_cuda(0)?;

        for (label, spec) in [("0.8B", QWEN35_0_8B), ("9B", QWEN35_9B)] {
            let path = pinned(spec)?;
            let model = from_gguf_path(
                &path,
                &device,
                Qwen35LoadOptions {
                    int8mode: Some(Int8Mode::Off),
                    expert_pack_dir: None,
                },
            )?;
            println!("{label}:");
            measure(&model, &prompt, 32)?;
            profile_layers(&model, &prompt)?;
        }
        Ok(())
    }

    /// The story-rewrite gate on the 0.8B — the family's smallest hybrid, and
    /// the one whose weights are unquantized BF16.
    ///
    /// Same 3:1 schedule, same partial rotary, same `head_dim 256`, same paged
    /// KV and the same recurrent carry as its larger siblings; what it does
    /// not have is a quantized projection anywhere. That makes it the gate
    /// that answers "does the hybrid engine work", separately from "does this
    /// checkpoint's quantization survive the task".
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-0.8B GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::test_parallel_batched_forwarding_0_8b \
                -- --ignored --nocapture --test-threads=1 \
                (serial: these load multi-gigabyte checkpoints and will exhaust \
                the card if cargo runs them concurrently)"]
    fn test_parallel_batched_forwarding_0_8b() -> Result<()> {
        println!("\n=== Qwen3.5-0.8B hybrid batched forwarding ===\n");
        let params = TestParams::new(10, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_timeout_secs(1800);

        let model_path = pinned(QWEN35_0_8B)?;
        let device = Device::new_cuda(0)?;

        let configs = vec![
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // Several sequences at once: the recurrent mixer runs per span and
            // a slicing error here leaks one conversation into another, which
            // a single-context row cannot catch.
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 16,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // ── Quantized KV — the same ladder the Qwen3 mid-size gates run.
            // On this family only every fourth layer holds KV at all (the
            // DeltaNet layers' recurrent state is untouched by KV modes), so
            // this exercises the head_dim-256 sealing/palette read path that
            // the float rows above never reach.
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C0,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C1,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C2,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C3,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C4,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C6,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C7,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C8 runs wide — 32 concurrent contexts — the deepest rung that
            // still has to be *production-comfortable*: the width both
            // stresses session isolation under heavy compression and yields
            // a meaningful aggregate-throughput number for the mode.
            TestConfig {
                mode: InferenceMode::C8,
                use_batched: true,
                num_contexts: 32,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C10×10 is the top rung and the calibration target:
            // `QWEN35_0_8B_KV_FACTORS` is tuned so the WHOLE range C0–C10
            // passes with C10 sitting just under the breaking edge — ten
            // contexts so the edge is a measurement, not a coin toss. If this
            // row goes red, the thresholds have drifted past the edge:
            // retighten the factor row rather than widening tolerances.
            TestConfig {
                mode: InferenceMode::C10,
                use_batched: true,
                num_contexts: 10,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
        ];

        let load = || {
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    // `Off` here is a statement about the checkpoint, not an
                    // unexamined default: the 0.8B is unquantized BF16, so every
                    // projection is a float weight and there is no KO twin for
                    // an int8 mode to select.
                    int8mode: Some(Int8Mode::Off),
                    expert_pack_dir: None,
                },
            )?;
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }

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
        let model_path = pinned(QWEN35_9B)?;
        let device = Device::new_cuda(0)?;

        // **Resolved here, then used twice**, so the table's `int8` column
        // cannot disagree with what the model loaded — the label is set
        // independently of the loader, and a gate that pins one and defaults the
        // other reports a numeric path it is not running.
        // `Performance`, not `auto_sized`. Auto picks `Precision` on this card,
        // and the two differ only in the weight twin — the q8a128 activation is
        // the same — so the choice is a throughput/accuracy dial, not a
        // capability one, and the gate is the place to hold it steady rather
        // than let it move with the device.
        let int8mode = Int8Mode::Performance;
        let params = TestParams::new(10, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(1800);

        let configs = vec![
            TestConfig {
                mode: InferenceMode::F16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
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
                generate_max_len: 20,
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
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C0,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C1,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C2,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C3,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C4,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C6,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C7,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 40,
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
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
                generate_max_len: 40,
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
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
        ];

        let load = || {
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    int8mode: Some(int8mode),
                    expert_pack_dir: None,
                },
            )?;
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
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

        let model_path = pinned(QWEN35_9B)?;
        let device = Device::new_cuda(0)?;
        let model = from_gguf_path(
            &model_path,
            &device,
            Qwen35LoadOptions {
                int8mode: Some(Int8Mode::Off),
                expert_pack_dir: None,
            },
        )?;
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
