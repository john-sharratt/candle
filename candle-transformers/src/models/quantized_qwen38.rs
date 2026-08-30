//! Qwen3.8-27B — the dense flagship hybrid's model file.
//!
//! Qwen3.8's GGUFs carry the lineage's `qwen35` arch string — same metadata
//! keys, same tensor schema — so everything loads through the machinery in
//! [`super::qwen35`] unchanged. What is new is the size: 64 layers at 3:1
//! (48 DeltaNet / 16 attention), 24 Q / 4 KV heads at `head_dim 256`,
//! DeltaNet 48 V / 16 QK at 128, hidden 5120, dense FFN 17408. All within
//! bounds the engine already enforces (`docs/qwen35_qwen38_models.md` §3).
//!
//! **It runs on the 16 GB dev card.** Dense means no expert relief and the
//! checkpoint outweighs the card at every rung, so the trunk's layers stream
//! through the weight zone the way a MoE model's experts do — hot VRAM slots
//! over pinned host RAM over a repacked `.pack` beside the GGUF, on the
//! deterministic held-prefix schedule in `docs/qwen38_layer_streaming.md` §9.5.
//! Which rung it streams is a VRAM measurement, not a fit test: see
//! [`QWEN38_27B_LADDER`].

use std::path::Path;

use candle::{Device, Result};
use candle_nn::kv_cache::QWEN38_KV_FACTORS;

use crate::models::draft_ladder::QWEN38_27B_DRAFT;

use super::qwen35::{load_hybrid_gguf, HybridBatched, Qwen35LoadOptions};

/// The 3.8 tokenizer, pinned to the canonical base-repo revision.
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.8-27B";
pub const TOKENIZER_REV: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";

/// The revision every rung of the quant ladder is pinned at.
///
/// One repository and one revision for the whole ladder, so a rung is chosen by
/// VRAM and by nothing else. The alternative — a rung per publisher — makes the
/// quant and the *conversion* vary together, and then a red C-ladder row after a
/// card change has two candidate causes instead of one.
const QWEN38_27B_REPO: &str = "bartowski/Qwen3.8-27B-GGUF";
const QWEN38_27B_REV: &str = "f0eec4a4bb4975114a030d048952d83c0a53c034";

/// The 27B at Q3_K_M — the rung for a card that cannot hold the model.
///
/// **The smallest weights are worth the most exactly where the model streams.**
/// Below the break-even width every forward moves `(N − H)·S` bytes across
/// PCIe (§9.5), so shrinking `S` shrinks the one term that decides decode. Q3_K_M
/// is 14.61 GB against Q4_K_M's 17.77 — 18% off every transfer, and a slot small
/// enough that the same zone holds more of them, which takes `H` up as well.
/// Both terms move the right way at once.
pub const QWEN38_27B_Q3KM: (&str, &str, &str) =
    (QWEN38_27B_REPO, QWEN38_27B_REV, "Qwen3.8-27B-Q3_K_M.gguf");

/// The 27B at Q4_K_M — the quant the routed gates run, and production's.
///
/// ~17.8 GB. Revision-pinned like every rung: this first went in as `"main"`,
/// which is a floating ref and quietly the opposite of what the policy asks for
/// — an upstream re-upload would have moved the checkpoint under the gate with
/// no diff and no failure, which is precisely the drift pinning exists to catch.
pub const QWEN38_27B_Q4KM: (&str, &str, &str) =
    (QWEN38_27B_REPO, QWEN38_27B_REV, "Qwen3.8-27B-Q4_K_M.gguf");

/// The 27B at Q6_K — the quant the lineage's other **dense** gate (the 9B) runs.
/// ~23.5 GB.
pub const QWEN38_27B_Q6K: (&str, &str, &str) =
    (QWEN38_27B_REPO, QWEN38_27B_REV, "Qwen3.8-27B-Q6_K.gguf");

/// The 27B at Q8_0 — near-lossless, for a card that can simply hold it. ~29.1 GB.
pub const QWEN38_27B_Q8_0: (&str, &str, &str) =
    (QWEN38_27B_REPO, QWEN38_27B_REV, "Qwen3.8-27B-Q8_0.gguf");

/// The NextN draft head — **the one file that cannot come from the ladder's repo.**
///
/// Only ggml-org converts the 27B's `mtp-` sidecar; the ladder's repo publishes
/// the trunk quants and nothing else. So the drafter is pinned separately, on its
/// own repository and revision, and stays on `Q4_0` at every rung — see
/// `pinned_mtp_head` for why that quant.
///
/// **A drafter from another conversion is safe in a way a trunk from another
/// conversion is not.** Speculation is lossless: the trunk verifies every
/// proposal, so a head converted by a different pipeline can only move the
/// acceptance rate, never the tokens that come out. That is exactly the property
/// that lets this one file break the ladder's one-publisher rule without
/// reopening the question the rule exists to close.
pub const QWEN38_27B_MTP: (&str, &str, &str) = (
    "ggml-org/Qwen3.8-27B-GGUF",
    "0669b98607d47046c7c2b3f801011d54a08cfccf",
    "mtp-Qwen3.8-27B-Q4_0.gguf",
);

/// The quant ladder, coarsest rung first.
///
/// `(minimum total VRAM, checkpoint)`. Read by [`checkpoint_for_this_card`],
/// which takes the **last** rung the card clears, so the order is load-bearing
/// and the test below pins it.
///
/// # The thresholds are about residency, not about fitting
///
/// A dense checkpoint no longer has to fit — layer streaming runs the 27B on a
/// 16 GB card at any of these quants (`docs/qwen38_layer_streaming.md`). What
/// the rung buys is **how much of the model stays resident**, and therefore how
/// many bytes cross PCIe on every forward. So the ladder is not "the largest
/// quant that fits" but "the largest quant whose residency the card can still
/// make good use of", and the two differ: a 16 GB card runs Q6_K perfectly well
/// and simply spends three times the bandwidth doing it.
///
/// Above 64 GB the model is resident whole at Q8_0 and the streaming machinery
/// degenerates to its no-eviction case (§7), which is the point at which quality
/// is the only axis left to spend on.
#[cfg(test)]
const QWEN38_27B_LADDER: &[(usize, (&str, &str, &str))] = &[
    (0, QWEN38_27B_Q3KM),
    (24 * 1024 * 1024 * 1024, QWEN38_27B_Q4KM),
    (32 * 1024 * 1024 * 1024, QWEN38_27B_Q6K),
    (64 * 1024 * 1024 * 1024, QWEN38_27B_Q8_0),
];

/// The checkpoint this machine can derive the row on.
///
/// **A hardware capability test, not a configuration switch.** It reads what the
/// card actually is; there is no environment variable and no way to ask for a
/// different file. A 72 GB Blackwell derives on Q8_0 and a 16 GB laptop derives
/// on Q3_K_M — see [`QWEN38_27B_LADDER`] for why the rungs are about residency
/// rather than about fitting.
///
/// # No rung is a `UD-` file, and that is the point
///
/// Unsloth's Dynamic quants pick a type per tensor, and this model's
/// `UD-Q4_K_M` mixes in **IQ4_XS** — gguf dtype 23, which
/// `GgmlDType::from_gguf_file_code` does not implement (there is no IQ-family
/// arm at all). It downloads all 16.5 GB and then fails to load, on any card.
/// The 3.5 and 3.6 gates pin `UD-Q4_K_M` files that happen to contain no IQ
/// tensors, so the naming gives no warning whatever: two files with the same
/// suffix differ in whether this codebase can read them at all. Every file named
/// here is a plain K-quant, so none can spring that surprise again.
///
/// # The ladder brackets production rather than matching it
///
/// `QWEN38_KV_FACTORS` was derived on Q4_K_M. Q6_K and Q8_0 sit *above* that and
/// Q3_K_M sits *below* it, so a row derived on a big card errs loose and one
/// derived on a small card errs true. That asymmetry belongs in the row's own
/// comment whenever the derivation is redone: a threshold measured with more
/// weight precision than production has is a threshold measured with headroom
/// production will not get — and one measured with less is a threshold that will
/// pass where production fails.
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
    // The last rung this card clears. The ladder is ordered coarsest-first, so
    Ok((total, rung_for(total)))
}

/// The rung a card of `total` bytes lands on.
///
/// The highest rung the card clears. The table ascends, so searching it from the
/// top and stopping at the first match *is* that rung — one expression, and
/// adding a rung is one line in [`QWEN38_27B_LADDER`] rather than another
/// `else if` here.
///
/// Split out from [`checkpoint_for_this_card`] so the selection can be pinned by
/// a test with no GPU: the driver query is the only part of that function this
/// one does not do, and a table asserted through a second copy of the arithmetic
/// asserts nothing about the first.
#[cfg(test)]
fn rung_for(total: usize) -> (&'static str, &'static str, &'static str) {
    QWEN38_27B_LADDER
        .iter()
        .rfind(|(min, _)| total >= *min)
        .map(|(_, c)| *c)
        .unwrap_or(QWEN38_27B_Q3KM)
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

    /// The filename [`rung_for`] lands on — the assertions below read better
    /// against a name than against a triple.
    fn rung(total: usize) -> &'static str {
        rung_for(total).2
    }

    #[test]
    fn the_quant_ladder_picks_by_card() {
        let gb = |n: usize| n * 1024 * 1024 * 1024;
        assert_eq!(rung(gb(16)), "Qwen3.8-27B-Q3_K_M.gguf", "16 GB");
        assert_eq!(rung(gb(24)), "Qwen3.8-27B-Q4_K_M.gguf", "24 GB");
        assert_eq!(rung(gb(32)), "Qwen3.8-27B-Q6_K.gguf", "32 GB");
        assert_eq!(rung(gb(64)), "Qwen3.8-27B-Q8_0.gguf", "64 GB");
        // The three dev machines, by name.
        assert_eq!(rung(gb(16)), "Qwen3.8-27B-Q3_K_M.gguf", "4090 Mobile");
        assert_eq!(rung(gb(24)), "Qwen3.8-27B-Q4_K_M.gguf", "3090");
        assert_eq!(rung(gb(72)), "Qwen3.8-27B-Q8_0.gguf", "PRO 5000 Blackwell");
        // Just below a threshold stays on the rung beneath it.
        assert_eq!(rung(gb(24) - 1), "Qwen3.8-27B-Q3_K_M.gguf");
        assert_eq!(rung(gb(32) - 1), "Qwen3.8-27B-Q4_K_M.gguf");
        assert_eq!(rung(gb(64) - 1), "Qwen3.8-27B-Q6_K.gguf");
    }

    /// `checkpoint_for_this_card` takes the **last** clearing rung, so a table
    /// out of order would silently answer with a coarser quant at every size
    /// above the misplaced entry.
    #[test]
    fn the_ladder_is_ordered_and_single_sourced() {
        let mins: Vec<usize> = QWEN38_27B_LADDER.iter().map(|(m, _)| *m).collect();
        assert!(
            mins.windows(2).all(|w| w[0] < w[1]),
            "ladder thresholds must ascend: {mins:?}"
        );
        assert_eq!(
            mins[0], 0,
            "the smallest rung must have no floor to fall to"
        );
        for (_, (repo, rev, _)) in QWEN38_27B_LADDER {
            assert_eq!(*repo, QWEN38_27B_REPO, "a rung from another publisher");
            assert_eq!(*rev, QWEN38_27B_REV, "a rung at another revision");
        }
        // The drafter is the deliberate exception, and the only one: it is the
        // sole file the ladder's publisher does not carry. Asserting the
        // difference keeps a later "tidy-up" from folding it back onto
        // `QWEN38_27B_REPO`, where the fetch 404s at the top of every gate row.
        assert_ne!(
            QWEN38_27B_MTP.0, QWEN38_27B_REPO,
            "the MTP sidecar is published by ggml-org, not by the ladder's repo"
        );
    }

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

    /// The NextN draft head that goes with the checkpoint this card runs.
    ///
    /// **The 27B keeps its head in a separate file**, unlike the 3.5/3.6, whose
    /// `-MTP-GGUF` repos embed the NextN tensors in the checkpoint. ggml-org
    /// ships `mtp-Qwen3.8-27B-{BF16,Q8_0,Q4_0}.gguf` alongside, and the main
    /// file declares no `nextn_predict_layers` at all — so a loader given only
    /// the checkpoint drafts nothing, decodes a token at a time, and reports
    /// nothing wrong. That is exactly what this gate did until the sidecar was
    /// wired: every row read `draft budget 0`.
    ///
    /// **Q4_0, and the quant matters more here than it looks.** The head is a
    /// full block at the trunk's geometry, and on a card where the model already
    /// streams, every megabyte it holds is a layer slot the zone does not have —
    /// which costs bandwidth on *every* forward. Q4_0 is 1.7 GB against Q8_0's
    /// 3.2, and the head only has to be right enough for its proposals to be
    /// accepted: speculation is lossless, so a lossier drafter costs acceptance
    /// rate and never correctness.
    ///
    /// **One head for the whole ladder.** It is pinned by [`QWEN38_27B_MTP`], on
    /// its own repository and revision, because only ggml-org converts the
    /// sidecar — and it does not vary with the rung, since the same losslessness
    /// argument that permits Q4_0 permits a Q4_0 head in front of a Q8_0 trunk.
    #[cfg(test)]
    fn pinned_mtp_head() -> Result<std::path::PathBuf> {
        let (repo, rev, file) = QWEN38_27B_MTP;
        hf_get(repo, RepoType::Model, rev, file)
    }

    /// The story-rewrite gate on the 27B — the same shape as the dense 9B
    /// gate, at the flagship geometry (48 DN / 16 attention, hpg 6).
    ///
    /// **It runs on every dev card**, including the 16 GB one, because the
    /// trunk's layers stream when they do not fit. The checkpoint scales with
    /// the card (`checkpoint_for_this_card`), so the row is derived on Q3_K_M at
    /// 16 GB, Q4_K_M at 24, Q6_K at 32 and Q8_0 at 64 — and the quant the row
    /// was measured on belongs in the row's comment, since the ladder brackets
    /// production rather than matching it.
    ///
    /// **Derived 2026-08-28** on a 72 GB Blackwell: C0–C10 all pass, C10 at
    /// 7.03× with both threshold axes bracketed (see `QWEN38_KV_FACTORS`). The
    /// row was extrapolated until then, and what had blocked the derivation was
    /// the pinned checkpoint rather than hardware — the old `UD-Q4_K_M` is an
    /// Unsloth Dynamic quant carrying IQ4_XS tensors this codebase cannot read.
    #[test]
    #[ignore = "downloads the Qwen3.8-27B GGUF this card's rung names (14.6 GB at \
                Q3_K_M through 29.1 GB at Q8_0 — see QWEN38_27B_LADDER) and builds a \
                layer pack beside it on first run. Runs on a 16 GB card: \
                the layers are weight-zone slot tenants and the ones that do not fit \
                stream (docs/qwen38_layer_streaming.md), so this is no longer a \
                production-workstation gate. Decode is bandwidth-bound there and slow \
                by construction — see §9.2. Run with: \
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

        // The drafter's own file — see `pinned_mtp_head`. Without it every row
        // of this table runs `draft budget 0`.
        let mtp_path = pinned_mtp_head()?;

        let load = || {
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    int8mode: Some(int8mode),
                    expert_pack_dir: None,
                    mtp_path: Some(mtp_path.clone()),
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
    #[ignore = "downloads the Qwen3.8-27B GGUF this card's rung names (14.6 GB at Q3_K_M \
                on the 16 GB card) and builds a layer pack beside it on first run, plus \
                the 1.7 GB MTP sidecar. Runs on a 16 GB card through layer streaming \
                (docs/qwen38_layer_streaming.md), where decode is bandwidth-bound: the \
                speedup a drafter buys is real but it is a multiple of a small number. \
                Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen38::tests::speculative_decode_27b \
                -- --ignored --nocapture --test-threads=1"]
    fn speculative_decode_27b() -> Result<()> {
        use crate::models::quantized_qwen35::tests::speculative_gate;

        // Device first — the checkpoint choice is a VRAM measurement.
        let probe = Device::new_cuda(0)?;
        let model_path = pinned(&probe)?;
        let mtp_path = pinned_mtp_head()?;
        let int8mode = Int8Mode::Performance;
        speculative_gate("Qwen3.8-27B", int8mode, &[1, 4], move || {
            let device = Device::new_cuda(0)?;
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    int8mode: Some(int8mode),
                    expert_pack_dir: None,
                    mtp_path: Some(mtp_path.clone()),
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
