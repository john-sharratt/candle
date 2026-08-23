//! Qwen3.8-27B — the dense flagship hybrid's model file.
//!
//! Qwen3.8's GGUFs carry the lineage's `qwen35` arch string — same metadata
//! keys, same tensor schema — so everything loads through the machinery in
//! [`super::qwen35`] unchanged. What is new is the size: 64 layers at 3:1
//! (48 DeltaNet / 16 attention), 24 Q / 4 KV heads at `head_dim 256`,
//! DeltaNet 48 V / 16 QK at 128, hidden 5120, dense FFN 17408. All within
//! bounds the engine already enforces (`docs/qwen35_qwen38_models.md` §3).
//!
//! **Build-only on the 16 GB dev card** (§3's rule): dense means no expert
//! relief, and the Q4_K_M weighs 16.5 GB, so the gate below is authored and
//! kept compiling here but runs on the production workstation. Nothing about
//! the model is deferred except that GPU run.

use std::path::Path;

use candle::{Device, Result};
use candle_nn::kv_cache::QWEN38_KV_FACTORS;

use super::qwen35::{load_hybrid_gguf, HybridBatched, Qwen35LoadOptions};

/// The 3.8 tokenizer, pinned to the canonical base-repo revision.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.8-27B";
pub const TOKENIZER_REV: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";

/// Pinned checkpoint (repo, revision, file) — revision-pinned so an upstream
/// re-upload fails the gate loudly instead of drifting under it.
pub const QWEN38_27B: (&str, &str, &str) = (
    "unsloth/Qwen3.8-27B-GGUF",
    "4ca720788d1e01f1bff70c033e0d0028fd02e502",
    "Qwen3.8-27B-UD-Q4_K_M.gguf",
);

/// Load the dense Qwen3.8 checkpoint and wrap it for the scheduler.
///
/// The 27B's concrete entry (the arch string is `qwen35`): refuses a routed
/// checkpoint, which would stand up an expert cache the caller did not plan
/// VRAM for, and constructs the lineage's [`HybridBatched`] with the 27B's
/// KV threshold factor row (extrapolated until the workstation derives it —
/// see `QWEN38_KV_FACTORS`).
pub fn from_gguf_path(
    file_path: &Path,
    device: &Device,
    options: Qwen35LoadOptions,
) -> Result<HybridBatched> {
    let model = load_hybrid_gguf(file_path, device, options)?;
    if model.cfg.moe.is_some() {
        candle::bail!(
            "quantized_qwen38: {file_path:?} is a routed (MoE) checkpoint — \
             load it through quantized_qwen35_moe or quantized_qwen36_moe instead"
        );
    }
    HybridBatched::new(model, QWEN38_KV_FACTORS)
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
        hf_get(QWEN38_27B.0, RepoType::Model, QWEN38_27B.1, QWEN38_27B.2)
    }

    /// The story-rewrite gate on the 27B — the same shape as the dense 9B
    /// gate, at the flagship geometry (48 DN / 16 attention, hpg 6).
    ///
    /// **Runs on the production workstation, not the 16 GB dev card**: the
    /// checkpoint is dense (no expert streaming to shrink the resident set)
    /// and its Q4_K_M weighs 16.5 GB before KV.
    #[test]
    #[ignore = "downloads the pinned Qwen3.8-27B GGUF (16.5 GB) and needs a GPU with more \
                than 16 GB of VRAM (dense — no expert relief; this is a production-\
                workstation gate, per docs/qwen35_qwen38_models.md §3). Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen38::tests::test_parallel_batched_forwarding_27b \
                -- --ignored --nocapture --test-threads=1"]
    fn test_parallel_batched_forwarding_27b() -> Result<()> {
        println!("\n=== Qwen3.8-27B hybrid batched forwarding ===\n");
        let model_path = pinned()?;
        let device = Device::new_cuda(0)?;

        // One value for both the loader and the table's `int8` column, held at
        // `Performance` — same reasoning as the lineage's other gates.
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
            // ── Quantized KV — the lineage ladder. `QWEN38_KV_FACTORS` is
            // extrapolated from the measured lineage rows until this gate
            // derives it on the workstation; these rungs are the derivation
            // instrument when it does.
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
            // C10×10 is the top rung and the calibration target, matching
            // the rest of the lineage: the workstation derivation tunes
            // `QWEN38_KV_FACTORS` so the whole range C0–C10 passes with C10
            // just under the breaking edge (the current row is extrapolated
            // from the 9B until then). A red C10 row means the thresholds
            // drifted past the edge — retighten the factor row rather than
            // widening tolerances.
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
                    int8mode: Some(int8mode),
                    expert_pack_dir: None,
                },
            )?;
            let cfg = &m.model().cfg;
            // The flagship must still be the geometry this engine was audited
            // for (§3's constraint check) — a silent architecture change fails
            // here, not in a kernel.
            assert_eq!(cfg.num_layers, 64);
            assert_eq!(cfg.attn_head_dim, 256);
            assert_eq!(cfg.hidden_size, 5120);
            assert!(cfg.moe.is_none());
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }
}
