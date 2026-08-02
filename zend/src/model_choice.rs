//! VRAM-adaptive Qwen3-30B-A3B quant selection — the single place zend
//! decides which quant of the model it runs.
//!
//! Cards with ≥ 24 GiB of VRAM run Q6_K (~25 GB with expert LRU paging);
//! anything smaller runs Q4_K_M (~17 GB). Everything downstream — the
//! downloader's repo/filename coordinates and the session's model builder —
//! derives from [`qwen30`], so the decision cannot drift between call sites.

use std::sync::OnceLock;

use candle::quantized::get_total_vram_device0;
use candle_conversation::models::Model;

/// VRAM floor for the Q6_K quant. Q6 needs ~25 GB total footprint but the
/// expert LRU pages the overflow, so 24 GiB cards run it comfortably; the
/// 16 GB tier cannot and gets Q4_K_M.
const Q6_MIN_TOTAL_VRAM_BYTES: usize = 24 * 1024 * 1024 * 1024;

/// Pure threshold decision: which Qwen3-30B-A3B quant fits this much VRAM.
///
/// `None` (VRAM unknown — probe failed, no device) picks Q4_K_M: the smaller
/// quant is the one that runs everywhere.
fn choose_qwen30(total_vram_bytes: Option<usize>) -> Model {
    match total_vram_bytes {
        Some(total) if total >= Q6_MIN_TOTAL_VRAM_BYTES => Model::Qwen3_30B_A3B_Q6,
        _ => Model::Qwen3_30B_A3B_Q4,
    }
}

/// The Qwen3-30B-A3B quant this machine runs, decided once per process from
/// measured total VRAM and logged.
pub fn qwen30() -> Model {
    static CHOICE: OnceLock<Model> = OnceLock::new();
    CHOICE
        .get_or_init(|| {
            // Context-free probe: this runs before any CUDA `Device` exists
            // (the choice drives the *download*), so it must not require a
            // bound context the way `get_vram_info` does.
            let total = match get_total_vram_device0() {
                Ok(total) => Some(total),
                Err(e) => {
                    tracing::warn!("VRAM probe failed ({e}); defaulting to Q4_K_M");
                    None
                }
            };
            let model = choose_qwen30(total);
            let gib = total.map(|t| t as f64 / (1024.0 * 1024.0 * 1024.0));
            tracing::info!(
                "model choice: {} (total VRAM {})",
                model.clone().spec().model_filename,
                gib.map_or_else(|| "unknown".to_string(), |g| format!("{g:.1} GiB")),
            );
            model
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::{choose_qwen30, Q6_MIN_TOTAL_VRAM_BYTES};
    use candle_conversation::models::Model;

    #[test]
    fn unknown_vram_picks_q4() {
        assert!(matches!(choose_qwen30(None), Model::Qwen3_30B_A3B_Q4));
    }

    #[test]
    fn sixteen_gib_picks_q4() {
        let gib16 = 16 * 1024 * 1024 * 1024;
        assert!(matches!(
            choose_qwen30(Some(gib16)),
            Model::Qwen3_30B_A3B_Q4
        ));
    }

    #[test]
    fn just_below_threshold_picks_q4() {
        assert!(matches!(
            choose_qwen30(Some(Q6_MIN_TOTAL_VRAM_BYTES - 1)),
            Model::Qwen3_30B_A3B_Q4
        ));
    }

    #[test]
    fn exactly_24_gib_picks_q6() {
        assert!(matches!(
            choose_qwen30(Some(Q6_MIN_TOTAL_VRAM_BYTES)),
            Model::Qwen3_30B_A3B_Q6
        ));
    }

    #[test]
    fn large_vram_picks_q6() {
        let gib72 = 72 * 1024 * 1024 * 1024;
        assert!(matches!(
            choose_qwen30(Some(gib72)),
            Model::Qwen3_30B_A3B_Q6
        ));
    }
}
