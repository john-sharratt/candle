//! The Qwen3.5 dense hybrid models, the 0.8B and the 9B: their pinned checkpoints,
//! their tokenizer, and the gate tests that hold them to exact output.
//!
//! The dense models share the lineage's machinery, so a failure here is the
//! hybrid itself, and a failure only on the MoE sibling is in the experts. Each
//! checkpoint is pinned by revision, because an upstream re-upload once
//! invalidated a KV calibration without anyone noticing. The MTP variants are
//! pinned so the draft head is present, and the 0.8B has none in any conversion.
//! Both gates run Q8_0 weights on the int8 matmul, the same arithmetic the
//! other siblings run.
//!
//! - Tests: `tests_pinned_throughput`, `tests_speculative_and_cold`,
//!   `tests_story_and_lora`.

use std::path::Path;

use candle::{Device, Result};
use candle_nn::kv_cache::{QWEN35_0_8B_KV_FACTORS, QWEN35_9B_KV_FACTORS};

use crate::models::draft_ladder::{QWEN35_0_8B_DRAFT, QWEN35_9B_DRAFT};

use super::qwen35::{load_hybrid_pack, HybridBatched};

/// The tokenizer repo, pinned. The GGUF ships `tokenizer.ggml.tokens` but no
/// `tokenizer.json`, and the family shares one tokenizer across sizes — the
/// reference loader proves the two agree token for token.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.5-0.8B";

pub const TOKENIZER_REV: &str = "2fc06364715b967f1860aea9cf38778875588b17";

/// Pinned checkpoints (repo, revision, file). Revisions are pinned because an
/// upstream re-upload once silently invalidated a threshold tuning — a gate
/// must fail loudly on a checkpoint change, not drift.
///
/// # Prefer the `-MTP-GGUF` repos wherever one exists
///
/// Qwen3.5/3.6 are trained with multi-token prediction and ship the head. The
/// **plain** GGUF conversion drops it; unsloth publishes a parallel
/// `…-MTP-GGUF` repo per model carrying the same quants under the same
/// filenames, plus `{arch}.nextn_predict_layers` and the `blk.N.nextn.*`
/// tensors. Pinning the plain one costs the drafter and shows up nowhere else
/// — the model loads and answers identically — which is exactly how an earlier
/// pass concluded the architecture had no MTP at all. Pin the MTP variant so
/// the head is *there*; whether it is loaded is a separate question the loader
/// answers.
///
/// The 0.8B has no MTP head in any conversion (none in the upstream config or
/// tensor index), so it keeps the plain repo.
///
/// **Q8_0, not the BF16 conversion.** The repo publishes both. BF16 was pinned
/// originally and made this the only model in the lineage on `Int8Mode::Off`:
/// float weights have no KO twin to select, so every projection took the FP path
/// while 9B/35B/27B took the int8 one — a numeric path no deployment of this
/// model would use, and the prime suspect for why `QWEN35_0_8B_KV_FACTORS` had
/// to sit ~3× tighter than every sibling's row.
///
/// One weight here is below the KO tile. `w_alpha`/`w_beta` are
/// `[n_v_heads, hidden]` = `[16, 1024]`, and `repack_ko` needs `nrows % 32 == 0`
/// (the q8a128 matmul tiles N in blocks of 32) — 16 linear-V heads does not
/// clear it, where the 9B and 35B have 32 and the 27B 48. That is a *per-tensor*
/// fact, not a model-level exclusion: `QMatMul::from_weights_with_mode` leaves a
/// non-tileable weight on the dequant path and gives every wide projection its
/// twin. It used to propagate the bail instead, which is what made a quantized
/// 0.8B unloadable in int8 over one small matrix.
///
/// (`Q8_1` is not a published GGUF file type — it exists as an internal block
/// format for activation quantization, not as a model conversion.)
pub const QWEN35_0_8B: (&str, &str, &str) = (
    "unsloth/Qwen3.5-0.8B-GGUF",
    "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0",
    "Qwen3.5-0.8B-Q8_0.gguf",
);

pub const QWEN35_9B: (&str, &str, &str) = (
    "unsloth/Qwen3.5-9B-MTP-GGUF",
    "9716a636ee4bddc3fed678220b7a33dd2a4160ae",
    "Qwen3.5-9B-Q6_K.gguf",
);

/// Load a dense Qwen3.5 model pack (0.8B / 9B) and wrap it for the scheduler.
///
/// This is the dense models' concrete entry: it refuses a routed checkpoint
/// (a MoE model loaded here would stand up an expert cache the caller did not
/// plan VRAM for) and constructs the lineage's [`HybridBatched`] with the
/// checkpoint's own derived KV threshold factor row. The two pinned dense
/// models are told apart by width — the 0.8B is `hidden 1024`, the 9B
/// `hidden 4096` — and an unpinned dense sibling gets the row of the nearer
/// size, which is the closest calibration that exists for it.
pub fn from_pack(path: &Path, device: &Device) -> Result<HybridBatched> {
    let model = load_hybrid_pack(path, device)?;
    if model.cfg.moe.is_some() {
        candle::bail!(
            "quantized_qwen35: {path:?} is a routed (MoE) model — \
             load it through quantized_qwen35_moe instead"
        );
    }
    // Both per-checkpoint rows are chosen by the same width test that already
    // tells the two dense siblings apart.
    let (kv_factors, draft) = if model.cfg.hidden_size >= 4096 {
        (QWEN35_9B_KV_FACTORS, QWEN35_9B_DRAFT)
    } else {
        (QWEN35_0_8B_KV_FACTORS, QWEN35_0_8B_DRAFT)
    };
    HybridBatched::new(model, kv_factors, draft)
}

#[cfg(test)]
mod tests_pinned_throughput;
#[cfg(test)]
mod tests_speculative_and_cold;
#[cfg(test)]
mod tests_story_and_lora;

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
pub(crate) mod tests {

    pub(crate) use super::tests_speculative_and_cold::tests::*;
}
