//! Qwen3.6-35B-A3B — the point-release routed hybrid's model file.
//!
//! Qwen3.6 is a point release of the Qwen3.5 architecture: its GGUFs carry
//! the same `qwen35moe` arch string, the same metadata keys and tensor
//! schema, and the same geometry as Qwen3.5-35B-A3B (40 layers at 3:1,
//! 16 Q / 2 KV heads at `head_dim 256`, DeltaNet 32 V / 16 QK at 128,
//! 256 experts top-8 plus a gated shared expert, hidden 2048). Everything
//! loads through the lineage machinery in [`super::qwen35`] unchanged; this
//! file pins the 3.6 checkpoint and tokenizer and holds its gate.

use std::path::Path;

use candle::{Device, Result};
use candle_nn::kv_cache::QWEN36_MOE_KV_FACTORS;

use super::qwen35::{load_hybrid_gguf, HybridBatched, Qwen35LoadOptions};

/// The 3.6 tokenizer, pinned. The point release ships its own tokenizer
/// repo; the vocabulary is the lineage's (the GGUF's `tokenizer.ggml.tokens`
/// and this file agree token for token, same as the 3.5 pairing).
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.6-35B-A3B";
pub const TOKENIZER_REV: &str = "995ad96eacd98c81ed38be0c5b274b04031597b0";

/// Pinned checkpoint (repo, revision, file) — revision-pinned so an upstream
/// re-upload fails the gate loudly instead of drifting under it.
pub const QWEN36_35B_A3B: (&str, &str, &str) = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "a483e9e6cbd595906af30beda3187c2663a1118c",
    "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
);

/// Load the routed Qwen3.6 checkpoint and wrap it for the scheduler.
///
/// The 3.6's concrete entry (the arch string is `qwen35moe`, so nothing
/// distinguishes the load itself): refuses a dense checkpoint, which loaded
/// through this entry would silently skip the expert cache the caller sized
/// VRAM around, and constructs the lineage's [`HybridBatched`] with the
/// 3.6's own derived KV threshold factor row.
pub fn from_gguf_path(
    file_path: &Path,
    device: &Device,
    options: Qwen35LoadOptions,
) -> Result<HybridBatched> {
    let model = load_hybrid_gguf(file_path, device, options)?;
    if model.cfg.moe.is_none() {
        candle::bail!(
            "quantized_qwen36_moe: {file_path:?} is a dense checkpoint — \
             load it through quantized_qwen35 instead"
        );
    }
    HybridBatched::new(model, QWEN36_MOE_KV_FACTORS)
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

    fn tokenizer_json() -> Result<String> {
        let p = hf_get(
            TOKENIZER_REPO,
            RepoType::Model,
            TOKENIZER_REV,
            "tokenizer.json",
        )?;
        std::fs::read_to_string(&p).map_err(|e| candle::Error::Msg(format!("read {p:?}: {e}")))
    }

    fn pinned() -> Result<std::path::PathBuf> {
        hf_get(
            QWEN36_35B_A3B.0,
            RepoType::Model,
            QWEN36_35B_A3B.1,
            QWEN36_35B_A3B.2,
        )
    }

    /// The story-rewrite gate on the 3.6-35B-A3B — the same shape as the
    /// Qwen3.5-35B gate, on the point release. Same expert-streaming posture:
    /// ~3B active parameters, so the 16 GB dev card runs it through the
    /// three-tier expert cache.
    #[test]
    #[ignore = "downloads the pinned Qwen3.6-35B-A3B GGUF (22 GB), repacks experts on first \
                run, and needs a GPU. Run with: cargo test --release --features cuda --lib \
                -p candle-transformers \
                quantized_qwen36_moe::tests::test_parallel_batched_forwarding_36_35b \
                -- --ignored --nocapture --test-threads=1 \
                (serial: these load multi-gigabyte checkpoints and will exhaust \
                the card if cargo runs them concurrently)"]
    fn test_parallel_batched_forwarding_36_35b() -> Result<()> {
        println!("\n=== Qwen3.6-35B-A3B hybrid MoE batched forwarding ===\n");
        let model_path = pinned()?;
        let device = Device::new_cuda(0)?;

        // One value for both the loader and the table's `int8` column, held at
        // `Performance` — same reasoning as the 3.5 gates.
        let int8mode = Int8Mode::Performance;
        let params = TestParams::new(10, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(3600);

        let configs = vec![
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                generate_max_len: 20,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // ── Quantized KV — the lineage ladder, with the streaming expert
            // cache in the picture: sealing runs while experts page, so a
            // threshold that only holds on the dense siblings breaks here.
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 1,
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
            // C8 runs wider than the single-context rungs — the deepest rung
            // that still has to be production-comfortable under expert
            // streaming.
            TestConfig {
                mode: InferenceMode::C8,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                generate_max_len: 40,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // C10×10 is the top rung and the calibration target:
            // `QWEN36_MOE_KV_FACTORS` is tuned so the whole range C0–C10
            // passes with C10 just under the breaking edge. A red C10 row
            // means the thresholds drifted past the edge — retighten the
            // factor row rather than widening tolerances.
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
            // Keep the pack beside the checkpoint: the gate reloads once per
            // invocation while iterating, and a persistent pack turns the
            // repack into a read.
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    int8mode: Some(int8mode),
                    expert_pack_dir: model_path.parent().map(|p| p.to_path_buf()),
                },
            )?;
            let cfg = &m.model().cfg;
            let moe = cfg.moe.expect("routed checkpoint");
            // The point release must still be the geometry this engine was
            // validated for — a silent architecture change fails here, not in
            // a kernel.
            assert_eq!(cfg.num_layers, 40);
            assert_eq!(cfg.attn_head_dim, 256);
            assert_eq!(moe.n_experts, 256);
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }

    /// **Speculative decode on the 3.6-35B — the production target.**
    ///
    /// The same sweep as the dense gates
    /// ([`crate::models::quantized_qwen35::tests::speculative_gate`]): the
    /// StoryRewrite fixture at 256 generated tokens, run once plain and once
    /// per draft budget, with the lossless driver and the weightless n-gram
    /// drafter. Every run validates against the same fixture at 100%, so a
    /// recurrent rewind that lost a token shows up as broken text rather than
    /// as a quiet quality slide.
    ///
    /// The MoE is what makes this the interesting one. A verify block widens
    /// the wave's routed-expert union, so on a streaming-expert config the
    /// extra rows cost expert DMA that a one-token decode would not have paid —
    /// which is exactly the trade the draft-budget sweep prices.
    #[test]
    #[ignore = "downloads the pinned Qwen3.6-35B-A3B GGUF (22 GB) and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen36_moe::tests::speculative_decode_36_35b \
                -- --ignored --nocapture --test-threads=1"]
    fn speculative_decode_36_35b() -> Result<()> {
        use crate::models::quantized_qwen35::tests::speculative_gate;

        let model_path = pinned()?;
        let int8mode = Int8Mode::Performance;
        speculative_gate("Qwen3.6-35B-A3B", int8mode, &[1, 4], move || {
            let device = Device::new_cuda(0)?;
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    int8mode: Some(int8mode),
                    expert_pack_dir: model_path.parent().map(|p| p.to_path_buf()),
                },
            )?;
            println!("✓ Model loaded\n");
            Ok(m)
        })
    }
}
