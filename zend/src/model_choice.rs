//! The single place zend decides which model it runs.
//!
//! Everything downstream — the downloader's repo/filename/size coordinates and
//! the session's model builder — derives from [`model`], so the choice cannot
//! drift between call sites.
//!
//! On a card that can seat it, Zen Code runs **Qwen3.8-Flash-Next**: 48 layers
//! at 3:1, so 36 gated-DeltaNet layers carrying a recurrent state and 12
//! attention layers carrying paged K/V, 250 B total at ~13 B active, with a
//! native 262,144-token context. Below that it runs **Qwen3.6-35B-A3B**, the
//! same hybrid lineage at 40 layers — see the ladder below.
//!
//! Three quarters of either stack has no K/V to splice, which is what the
//! recurrent-state persistence work in `docs/deltanet_state_persistence.md`
//! exists to support — and this lineage adds two more carried classes on top,
//! a PLE window and a QSA index, neither of which is a delta-rule matrix and
//! neither of which can be rebuilt from restored K/V. Both ride the turn
//! record's model-opaque blob (`docs/qwen38_index_persistence.md`).
//!
//! # The ladder, and what it is actually keyed on
//!
//! Parameter count does not decide whether a card can run a model: a MoE's
//! resident footprint is its dense weights plus whatever expert working set
//! fits, and the three-tier expert cache pages the rest. Both rungs below run on
//! a small card in that sense, and this ladder is NOT a feasibility test.
//!
//! What does scale is the **dense resident footprint** — the weights that are
//! never paged. Flash-Next's measured weight zone is ~53.5 GB, so a card that
//! cannot seat it spends the run streaming the trunk instead of the experts, and
//! the model it was chosen for is the reason it is slow. Qwen3.6-35B-A3B is the
//! rung below: the same hybrid lineage, a dense trunk small enough to sit on a
//! consumer card, with the expert cache doing the paging it was designed for.
//!
//! So: above 60 GiB, Flash-Next. Otherwise the 35B hybrid. A bigger card still
//! buys speed rather than feasibility — the ladder picks which model is worth
//! that card, not which one it can technically load.
//!
//! **The engine GGUF is prepared locally, not downloaded.** Nothing publishes
//! it: it is the pinned Q8_0 split with the W4A16 release's experts imported to
//! Q4_KO and the MTP head folded in, merged into one file. `download.rs` reports
//! the build step rather than asking the hub for a name nobody published.

use candle_conversation::models::Model;
use std::sync::OnceLock;

/// Total VRAM above which the daemon runs Flash-Next rather than the 35B hybrid.
///
/// Sized from the measured weight zone (~53.5 GB of dense trunk) plus room for
/// the KV span and a wave's transients beside it. A card at or under this can
/// load Flash-Next, but would page its trunk to do so.
const FLASH_NEXT_MIN_VRAM_BYTES: usize = 60 * 1024 * 1024 * 1024;

/// The model zend runs, decided once per process.
///
/// **Once** matters: `download.rs` calls this to resolve which artifact to fetch
/// and `session.rs` calls it again to build the engine. Re-measuring between
/// those two could hand back different answers — a card whose free/total figures
/// shift, a device that failed to open the first time — and the daemon would
/// then download one model and load another, with the mismatch surfacing as a
/// missing-file error naming a model nobody selected.
pub fn model() -> Model {
    static CHOICE: OnceLock<Model> = OnceLock::new();
    CHOICE
        .get_or_init(|| {
            let vram = total_vram_bytes();
            let chosen = model_for_vram(vram);
            tracing::info!(
                vram_gib = vram as f64 / (1024.0 * 1024.0 * 1024.0),
                threshold_gib = FLASH_NEXT_MIN_VRAM_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
                model = ?chosen,
                "model selected from measured VRAM"
            );
            chosen
        })
        .clone()
}

/// The ladder itself, as a pure function of total VRAM so it is testable without
/// a GPU — the measurement is the only part that needs one.
///
/// A machine with no CUDA device reports 0 and takes the lower rung, which is
/// the right answer for CPU-only and for a probe that could not open the device:
/// the smaller model is the one that still works when the assumption is wrong.
fn model_for_vram(total_vram_bytes: usize) -> Model {
    if total_vram_bytes > FLASH_NEXT_MIN_VRAM_BYTES {
        Model::Qwen38_FlashNext_Q4KO
    } else {
        Model::Qwen36_35B_A3B_Q4
    }
}

/// Total VRAM on device 0, in bytes; `0` when there is no usable CUDA device.
///
/// Opening the device here is deliberate. This runs before the engine exists —
/// `download.rs` needs the answer to pick an artifact — so there is no device to
/// borrow, and the driver caches the context this creates for the load that
/// follows. A failure to open is not fatal: it reports 0 and the ladder takes
/// the conservative rung.
fn total_vram_bytes() -> usize {
    match candle::Device::new_cuda(0) {
        Ok(candle::Device::Cuda(d)) => d.mem_get_info().map(|(_free, total)| total).unwrap_or(0),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{model_for_vram, FLASH_NEXT_MIN_VRAM_BYTES};
    use candle_conversation::models::{DialectType, Model, ModelArch};

    const GIB: usize = 1024 * 1024 * 1024;

    /// The spec assertions below deliberately name their model rather than
    /// calling `model()`. `model()` now measures the card, so routing them
    /// through it would make every one of them assert against whatever hardware
    /// happened to run the suite — passing on this box, failing on a laptop, and
    /// testing the GPU rather than the spec either way.
    fn flash_next() -> Model {
        Model::Qwen38_FlashNext_Q4KO
    }

    // ── The ladder ────────────────────────────────────────────────────

    /// A card that can seat Flash-Next's ~53.5 GB dense trunk gets it.
    #[test]
    fn a_large_card_runs_flash_next() {
        assert!(matches!(
            model_for_vram(72 * GIB),
            Model::Qwen38_FlashNext_Q4KO
        ));
    }

    /// Everything below takes the hybrid rung — including cards that could
    /// technically load Flash-Next but would page its trunk to do it.
    #[test]
    fn a_smaller_card_runs_the_hybrid() {
        for vram in [24 * GIB, 16 * GIB, 8 * GIB] {
            assert!(
                matches!(model_for_vram(vram), Model::Qwen36_35B_A3B_Q4),
                "{vram} bytes should take the hybrid rung"
            );
        }
    }

    /// **No CUDA device reports 0, and 0 must not select the larger model.**
    /// A probe that cannot open the device is the case where the assumption is
    /// already wrong, so the ladder has to fail toward the model that still runs.
    #[test]
    fn an_unmeasurable_device_takes_the_conservative_rung() {
        assert!(matches!(model_for_vram(0), Model::Qwen36_35B_A3B_Q4));
    }

    /// The boundary is exclusive: exactly the threshold is not "more than" it.
    #[test]
    fn the_threshold_is_exclusive() {
        assert!(matches!(
            model_for_vram(FLASH_NEXT_MIN_VRAM_BYTES),
            Model::Qwen36_35B_A3B_Q4
        ));
        assert!(matches!(
            model_for_vram(FLASH_NEXT_MIN_VRAM_BYTES + 1),
            Model::Qwen38_FlashNext_Q4KO
        ));
    }

    /// **Both rungs must suppress thinking, by different mechanisms.**
    ///
    /// The two models do not share a dialect: Flash-Next is `Qwen35`, which has
    /// no `/no_think` switch and suppresses by prefilling an already-closed
    /// block; the hybrid is `ChatML`, which uses the switch. `thinking_suppression`
    /// returns the pair and exactly one half is live per family — so a rung whose
    /// dialect had neither would silently reason on every turn, which is the
    /// failure this ladder could otherwise reintroduce on the hardware nobody
    /// tests on.
    #[test]
    fn every_rung_has_a_working_suppression_mechanism() {
        for m in [Model::Qwen38_FlashNext_Q4KO, Model::Qwen36_35B_A3B_Q4] {
            let spec = m.clone().spec();
            let (switch, block) = spec.dialect.thinking_suppression(true);
            assert!(
                switch.is_empty() != block.is_empty(),
                "{m:?}: exactly one suppression half must be live (switch={switch:?}, block={block:?})"
            );
        }
    }

    /// The two rungs are deliberately different dialects, and the ladder must
    /// not quietly converge them — a change that made both ChatML would drop
    /// Flash-Next's structural suppression without failing anything else.
    #[test]
    fn the_rungs_keep_their_own_dialects() {
        assert!(matches!(
            Model::Qwen38_FlashNext_Q4KO.spec().chat_format,
            DialectType::Qwen35
        ));
        assert!(matches!(
            Model::Qwen36_35B_A3B_Q4.spec().chat_format,
            DialectType::ChatML
        ));
    }

    // ── Flash-Next's spec ─────────────────────────────────────────────

    #[test]
    fn flash_next_is_the_qwen4exp_arch() {
        assert!(matches!(flash_next().spec().arch, ModelArch::Qwen4Exp));
    }

    /// **The engine artifact is built here, so resolution must not go asking.**
    ///
    /// Nothing publishes the merged GGUF under its name. A spec that left this
    /// unset would send the daemon to the hub for a 404 on every start, and the
    /// operator would read "download failed" for a file that was never
    /// downloadable — instead of "run the prepare step".
    #[test]
    fn the_engine_artifact_is_prepared_not_downloaded() {
        assert!(flash_next().spec().prepared_from_source);
    }

    /// This lineage suppresses thinking by opening the assistant turn with an
    /// already-closed think block; Qwen3's `/no_think` marker does nothing on
    /// it. Reading a Qwen3 dialect here would leave reasoning blocks in every
    /// reply with nothing failing.
    #[test]
    fn the_dialect_is_the_flash_next_one() {
        assert!(matches!(
            flash_next().spec().chat_format,
            candle_conversation::models::DialectType::Qwen35
        ));
    }

    /// The context the checkpoint actually backs — measured flat from 32K to
    /// 128K, not a declared number. A conversation is bounded by this.
    #[test]
    fn the_native_context_is_not_truncated_by_the_spec() {
        assert_eq!(flash_next().spec().max_seq_len, 262_144);
    }

    /// **The residual stream is BF16, and the session has to be told.**
    ///
    /// Left unset, the session derives its activation width from the KV storage
    /// format, which is F16 for the quantized backing this engine always uses —
    /// a statement about the arena, not about the residual stream. This lineage
    /// publishes `"dtype": "bfloat16"`, and the two are not interchangeable:
    /// both are 16 bits, but F16 stops at 65504 where BF16 carries F32's
    /// exponent, so a block that legitimately produces 1e5 silently becomes
    /// `inf` and the next layer turns it into a NaN.
    #[test]
    fn the_activation_width_is_bf16() {
        assert_eq!(
            flash_next().spec().arch.native_activation_dtype(),
            Some(candle::DType::BF16),
        );
    }
}
