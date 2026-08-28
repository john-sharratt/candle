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

use crate::models::draft_ladder::QWEN38_27B_DRAFT;

use super::qwen35::{load_hybrid_gguf, HybridBatched, Qwen35LoadOptions};

/// The 3.8 tokenizer, pinned to the canonical base-repo revision.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.8-27B";
pub const TOKENIZER_REV: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";

/// The 27B at Q6_K — the quant the lineage's other **dense** gate (the 9B) runs.
///
/// Revision-pinned so an upstream re-upload fails the gate loudly instead of
/// drifting under it. ~22 GB.
pub const QWEN38_27B_Q6K: (&str, &str, &str) = (
    "bartowski/Qwen3.8-27B-GGUF",
    "f0eec4a4bb4975114a030d048952d83c0a53c034",
    "Qwen3.8-27B-Q6_K.gguf",
);

/// The 27B at Q4_K_M — the quant the routed gates run, and production's.
///
/// Revision-pinned like its sibling. It first went in as `"main"`, which is a
/// floating ref and quietly the opposite of what the policy above asks for: an
/// upstream re-upload would have moved the checkpoint under the gate with no
/// diff and no failure, which is precisely the drift the pinning exists to
/// catch. ~16.5 GB.
pub const QWEN38_27B_Q4KM: (&str, &str, &str) = (
    "ggml-org/Qwen3.8-27B-GGUF",
    "0669b98607d47046c7c2b3f801011d54a08cfccf",
    "Qwen3.8-27B-Q4_K_M.gguf",
);

/// Total VRAM at or above which the gate derives on [`QWEN38_27B_Q6K`].
///
/// The 27B is dense — no expert streaming to shrink the resident set — so the
/// card has to hold the whole checkpoint plus KV. Q6_K's ~22 GB clears a 32 GB
/// card with room for the ladder's widest cohort; below that the gate would
/// either refuse or spend its run swapping, and Q4_K_M's ~16.5 GB is what fits.
#[cfg(test)]
const Q6_MIN_TOTAL_VRAM_BYTES: usize = 32 * 1024 * 1024 * 1024;

/// The checkpoint this machine can derive the row on.
///
/// **A hardware capability test, not a configuration switch.** It reads what the
/// card actually is; there is no environment variable and no way to ask for the
/// other file. A 72 GB Blackwell derives on Q6_K and a 16 GB laptop derives on
/// Q4_K_M, and each is the strongest quant that machine can hold.
///
/// # Neither is a `UD-` file, and that is the point
///
/// Unsloth's Dynamic quants pick a type per tensor, and this model's
/// `UD-Q4_K_M` mixes in **IQ4_XS** — gguf dtype 23, which
/// `GgmlDType::from_gguf_file_code` does not implement (there is no IQ-family
/// arm at all). It downloads all 16.5 GB and then fails to load, on any card.
/// The 3.5 and 3.6 gates pin `UD-Q4_K_M` files that happen to contain no IQ
/// tensors, so the naming gives no warning whatever: two files with the same
/// suffix differ in whether this codebase can read them at all. Both files
/// named here are single-type, so neither can spring that surprise again.
///
/// # The two quants bracket production rather than match it
///
/// Q6_K sits *above* a Q4_K_M deployment and Q4_K_M sits *at* it, so a row
/// derived on the 32 GB-plus path errs loose and one derived below it errs
/// true. That asymmetry is worth stating in the row's own comment when the
/// derivation lands: a threshold measured with more weight precision than
/// production has is a threshold measured with headroom production will not get.
///
/// # It takes the device, and refuses to guess without one
///
/// `get_vram_info` needs a live CUDA context, so this must be called *after*
/// the device is built — hence the parameter, which makes that ordering a
/// compile-time requirement rather than a convention.
///
/// The first cut read the card before `Device::new_cuda` and fell back to
/// `unwrap_or(0)`, which reported **zero VRAM on a 72 GB card** and quietly
/// derived the row on the small quant. A failed measurement is not a small
/// card, and collapsing the two is how a gate silently measures the wrong
/// thing: the run looked entirely healthy and the only tell was the checkpoint
/// name. So the query's failure is an error now, not a default.
/// Returns the measured total alongside the choice, so a caller reporting the
/// decision quotes the number that *made* it rather than asking the driver a
/// second time — a second query can disagree with the first, and printing
/// "0.0 GiB → Q6_K" is worse than not printing anything.
#[cfg(test)]
fn checkpoint_for_this_card(
    device: &Device,
) -> Result<(usize, (&'static str, &'static str, &'static str))> {
    if !matches!(device, Device::Cuda(_)) {
        candle::bail!("qwen38 gate: CUDA device required to size the checkpoint");
    }
    let (_free, total) = candle::quantized::get_vram_info().map_err(|e| {
        candle::Error::Msg(format!(
            "qwen38 gate: could not read total VRAM ({e}), so the checkpoint cannot be \
             chosen. Refusing rather than defaulting — a failed query is not a small \
             card, and guessing here silently derives the threshold row against the \
             wrong quant."
        ))
    })?;
    let pick = if total >= Q6_MIN_TOTAL_VRAM_BYTES {
        QWEN38_27B_Q6K
    } else {
        QWEN38_27B_Q4KM
    };
    Ok((total, pick))
}

/// Load the dense Qwen3.8 checkpoint and wrap it for the scheduler.
///
/// The 27B's concrete entry (the arch string is `qwen35`): refuses a routed
/// checkpoint, which would stand up an expert cache the caller did not plan
/// VRAM for, and constructs the lineage's [`HybridBatched`] with the 27B's
/// KV threshold factor row (derived 2026-08-28 — see `QWEN38_KV_FACTORS`).
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
    HybridBatched::new(model, QWEN38_KV_FACTORS, QWEN38_27B_DRAFT)
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

    /// The checkpoint for this card, downloaded. Takes the device because the
    /// choice is a VRAM measurement — see [`checkpoint_for_this_card`].
    fn pinned(device: &Device) -> Result<std::path::PathBuf> {
        let (total, (repo, rev, file)) = checkpoint_for_this_card(device)?;
        let gib = total as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("  - Card: {gib:.1} GiB total → checkpoint {repo} @ {file}");
        hf_get(repo, RepoType::Model, rev, file)
    }

    /// The story-rewrite gate on the 27B — the same shape as the dense 9B
    /// gate, at the flagship geometry (48 DN / 16 attention, hpg 6).
    ///
    /// **Not the 16 GB dev card**: the checkpoint is dense — no expert
    /// streaming to shrink the resident set — and weighs 16.5–22 GB before KV.
    /// The checkpoint scales with the card (`checkpoint_for_this_card`), so the
    /// gate runs on anything that can hold Q4_K_M and derives on Q6_K above
    /// 32 GB.
    ///
    /// **Derived 2026-08-28** on a 72 GB Blackwell: C0–C10 all pass, C10 at
    /// 7.03× with both threshold axes bracketed (see `QWEN38_KV_FACTORS`). The
    /// row was extrapolated until then, and what had blocked the derivation was
    /// the pinned checkpoint rather than hardware — the old `UD-Q4_K_M` is an
    /// Unsloth Dynamic quant carrying IQ4_XS tensors this codebase cannot read.
    #[test]
    #[ignore = "downloads a pinned Qwen3.8-27B GGUF (16.5 GB at Q4_K_M, 22 GB at Q6_K) \
                and needs a GPU with more than 16 GB of VRAM (dense — no expert \
                relief). Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen38::tests::test_parallel_batched_forwarding_27b \
                -- --ignored --nocapture --test-threads=1"]
    fn test_parallel_batched_forwarding_27b() -> Result<()> {
        println!("\n=== Qwen3.8-27B hybrid batched forwarding ===\n");
        // Device first: the checkpoint is chosen from a VRAM measurement, and
        // `get_vram_info` needs a live CUDA context to answer.
        let device = Device::new_cuda(0)?;
        let model_path = pinned(&device)?;

        // One value for both the loader and the table's `int8` column, held at
        // `Performance` — same reasoning as the lineage's other gates.
        let int8mode = Int8Mode::Performance;
        let params = TestParams::new(10, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(3600);

        let mut configs = vec![
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 4,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            // ── Quantized KV — the lineage ladder, and the instrument
            // `QWEN38_KV_FACTORS` was derived with: each axis walked out on the
            // C10 rung until it broke, then backed off one step.
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
            // C10×10 is the top rung and the calibration target, matching the
            // rest of the lineage: `QWEN38_KV_FACTORS` is tuned so the whole
            // range C0–C10 passes with C10 just under the breaking edge (K
            // 1.3 ✓/1.4 ✗, V 2.3 ✓/2.4 ✗, derived 2026-08-28). A red C10 row
            // means the thresholds drifted past the edge — retighten the factor
            // row rather than widening tolerances.
            TestConfig {
                mode: InferenceMode::C10,
                use_batched: true,
                num_contexts: 10,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
        ];

        // **On a big card, keep going — but only on this rung.**
        //
        // The same widening the 3.6 gate carries, and for the same reason: what
        // bounds concurrency at C10 is per-session state, not the checkpoint.
        // DeltaNet holds `n_v_heads × head_dim × head_dim` in F32 per recurrent
        // layer per sequence, doubled for the live/backup ping-pong, and C10 KV
        // adds only a couple of MiB on top. Widening here measures the engine's
        // concurrency; widening a lower rung would measure KV bytes and so
        // measure the card instead.
        //
        // **This model is the dense flagship, so the widths are its own.** It
        // carries 48 DeltaNet layers against the 3.6's routed geometry, and its
        // weights are resident rather than expert-streamed — ~22 GB at Q6_K
        // before a single session exists. The 3.6's ×32/×64 are not
        // transferable on that footprint, so this steps ×20 and ×40: past what
        // the 16 GB card could hold, and inside a workstation's headroom with
        // the resident model already in place.
        // The gate is 40 GiB, above the 32 GiB that selects Q6_K: a card between
        // the two holds the resident checkpoint but not forty sessions of
        // recurrent state on top, so it runs the calibration rung and stops.
        // The two thresholds answer different questions and are deliberately
        // not the same number.
        //
        // **Say so when the wide rungs are skipped.** A narrower run is still a
        // green run, and silence would let reduced coverage read as a pass —
        // the one failure mode a gate must not have.
        let (_, total_vram) = device.mem_get_info().unwrap_or((0, 0));
        if total_vram >= 40 << 30 {
            configs.extend([20usize, 40].map(|n| TestConfig {
                mode: InferenceMode::C10,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }));
        } else {
            println!(
                "  - C10 widening skipped: {:.1} GiB total, under the 40 GiB gate — \
                 the ×20 and ×40 rungs will not run",
                total_vram as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }

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

    /// Speculative decode on the 27B — the lineage gate at the flagship
    /// geometry. See [`crate::models::quantized_qwen35::tests::speculative_gate`].
    ///
    /// **No repin was needed for this one.** The pinned `unsloth/Qwen3.8-27B-GGUF`
    /// already carries the NextN tensors (`blk.64.nextn.{enorm,hnorm,eh_proj,
    /// shared_head_norm}` against `block_count = 65`,
    /// `nextn_predict_layers = 1`), and unlike the 35B siblings the head is
    /// **dense** — `blk.64.ffn_{gate,up,down}`, no router and no experts — so it
    /// loads through the path the 9B already proved and needs nothing from the
    /// expert cache. That is also why there is no `-MTP-GGUF` repo to move to:
    /// none is published, and none is required.
    ///
    /// **Runs on the production workstation**, for the same reason as the gate
    /// above: dense at 16.5 GB leaves no room on a 16 GB card.
    #[test]
    #[ignore = "downloads the pinned Qwen3.8-27B GGUF (16.5 GB) and needs a GPU with more \
                than 16 GB of VRAM (dense — no expert relief; this is a production-\
                workstation gate, per docs/qwen35_qwen38_models.md §3). Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen38::tests::speculative_decode_27b \
                -- --ignored --nocapture --test-threads=1"]
    fn speculative_decode_27b() -> Result<()> {
        use crate::models::quantized_qwen35::tests::speculative_gate;

        // Device first — the checkpoint choice is a VRAM measurement.
        let probe = Device::new_cuda(0)?;
        let model_path = pinned(&probe)?;
        let int8mode = Int8Mode::Performance;
        speculative_gate("Qwen3.8-27B", int8mode, &[1, 4], move || {
            let device = Device::new_cuda(0)?;
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    int8mode: Some(int8mode),
                    expert_pack_dir: None,
                },
            )?;
            assert!(
                m.has_drafter(),
                "the pinned 27B declares an MTP head but none loaded — the pin has \
                 moved to a conversion that drops the NextN tensors"
            );
            println!("✓ Model loaded\n");
            Ok(m)
        })
    }
}
