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

use crate::models::draft_ladder::QWEN36_35B_A3B_DRAFT;

use super::qwen35::{load_hybrid_gguf, HybridBatched, Qwen35LoadOptions};

/// The 3.6 tokenizer, pinned. The point release ships its own tokenizer
/// repo; the vocabulary is the lineage's (the GGUF's `tokenizer.ggml.tokens`
/// and this file agree token for token, same as the 3.5 pairing).
pub const TOKENIZER_REPO: &str = "Qwen/Qwen3.6-35B-A3B";
pub const TOKENIZER_REV: &str = "995ad96eacd98c81ed38be0c5b274b04031597b0";

/// Pinned checkpoint (repo, revision, file) — revision-pinned so an upstream
/// re-upload fails the gate loudly instead of drifting under it.
///
/// **The `-MTP-` repo**, because the plain conversion drops the NextN tensors
/// and a model without them cannot speculate. Same quant as before
/// (`UD-Q4_K_M`), so only the head is new here — unlike the 3.5, whose repo
/// move also changed quantization. The head is a full routed block and the
/// expert cache carries its layer alongside the trunk's
/// (`expert_host_refs`).
pub const QWEN36_35B_A3B: (&str, &str, &str) = (
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
    "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
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
    HybridBatched::new(model, QWEN36_MOE_KV_FACTORS, QWEN36_35B_A3B_DRAFT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
    use crate::models::batched_inference::InferenceMode;
    use crate::models::dialect::Dialect;
    use crate::models::quantized_qwen35::tests::cold_speculative_point;
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
            // ── Quantized KV — the lineage ladder, with the streaming expert
            // cache in the picture: sealing runs while experts page, so a
            // threshold that only holds on the dense siblings breaks here.
            TestConfig {
                mode: InferenceMode::Q8_0,
                use_batched: true,
                num_contexts: 1,
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
            // C8 runs wider than the single-context rungs — the deepest rung
            // that still has to be production-comfortable under expert
            // streaming.
            TestConfig {
                mode: InferenceMode::C8,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
        ];

        // ── The top rung, at the widths worth seeing it at ───────────────────
        //
        // C10 is the calibration target: `QWEN36_MOE_KV_FACTORS` is tuned so the
        // whole range C0–C10 passes with C10 just under the breaking edge. A red
        // C10 row means the thresholds drifted past it — retighten the factor
        // row rather than widening tolerances.
        //
        // **That row has not been re-derived since this rung moved.** It was
        // calibrated when C10 ran at ×10; the widths below are ×8 and ×16, and
        // the 3.5 sibling needed a retune (K 1.5→1.2, V 2.0→2.5) for exactly
        // that widening, because ×16 lost a session under the old row. If a C10
        // row here goes red, that is the first thing to suspect, and the fix is
        // the factor row rather than this gate.
        //
        // Run at **8 and 16** rather than one middling width: the pair shows the
        // top rung holding as the cohort grows, which is where a threshold row
        // tuned at one width quietly stops covering the next.
        //
        // Both sit inside the draft ladder's bracket, so both decode
        // speculatively — but this gate generates ten tokens, which at budget 2
        // is about four drafted steps a session. That is enough for the accept
        // path to be *exercised* and nowhere near enough to measure what it
        // yields; the throughput answer comes from `cold_speculative_point` at
        // 256 tokens, one width per process. Read these rows as compression
        // correctness under speculation, not as a speed-up.
        configs.extend([8usize, 16].map(|n| TestConfig {
            mode: InferenceMode::C10,
            use_batched: true,
            num_contexts: n,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        }));

        // **On a big card, keep going — but only on this rung.**
        //
        // What bounds concurrency here is per-session state, not the
        // checkpoint: DeltaNet holds `n_v_heads × head_dim × head_dim` in F32
        // per recurrent layer per sequence, doubled for the live/backup
        // ping-pong — about 120 MiB a session on this geometry — and C10 KV adds
        // only a couple more. So 32 sessions want ~4 GiB of per-session state
        // and 64 want ~8 GiB, on top of a resident footprint that already fills
        // a 16 GiB card. Forty is the gate: comfortably past what this laptop
        // can hold, comfortably inside a workstation card.
        //
        // Only C10 is widened. The lower rungs carry uncompressed or lightly
        // compressed KV, where the same widths would be bounded by KV bytes
        // instead and would measure the card rather than the engine.
        let (_, total_vram) = device.mem_get_info().unwrap_or((0, 0));
        if total_vram >= 40 << 30 {
            configs.extend([32usize, 64].map(|n| TestConfig {
                mode: InferenceMode::C10,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }));
        }

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

    /// The 3.6's own cold points, one width and one budget per process.
    ///
    /// **The dense 9B's ladder cannot be assumed to carry over.** Both models run
    /// the same NextN head, but this one streams its experts: a verify block
    /// scores `k + 1` positions per sequence and each routes to its own top-8 of
    /// 256, so the wave's routed union widens with the block and the extra rows
    /// are paid for in PCIe traffic rather than in arithmetic. Where that stops
    /// being worth it depends on the resident ratio — how much of the expert set
    /// the card is holding — which is a property of this checkpoint on this
    /// machine, not of the lineage.
    ///
    /// Read the expert-pipeline table alongside the throughput: hit rate, DMA
    /// loads and late loads are what say whether a budget-2 wave is paying in
    /// bandwidth, and they are the half of the picture the dense measurement
    /// does not have.
    ///
    /// Run singly and cold — see `cold_speculative_point` for why a multi-config
    /// sweep on a laptop card measures its own boost budget.
    macro_rules! cold_point_36 {
        ($name:ident, $mode:expr, $width:expr, $budget:expr) => {
            #[test]
            #[ignore = "cold measurement point on the 3.6-35B — run singly, letting the card \
                        settle between points."]
            fn $name() -> Result<()> {
                let model_path = pinned()?;
                let int8mode = Int8Mode::Performance;
                cold_speculative_point(
                    "Qwen3.6-35B-A3B",
                    &tokenizer_json()?,
                    $mode,
                    $width,
                    $budget,
                    int8mode,
                    move || {
                        let device = Device::new_cuda(0)?;
                        from_gguf_path(
                            &model_path,
                            &device,
                            Qwen35LoadOptions {
                                int8mode: Some(int8mode),
                                expert_pack_dir: model_path.parent().map(|p| p.to_path_buf()),
                            },
                        )
                    },
                )
            }
        };
    }

    cold_point_36!(cold36_w4_b0, InferenceMode::BF16, 4, 0);
    cold_point_36!(cold36_w4_b2, InferenceMode::BF16, 4, 2);
    cold_point_36!(cold36_w10_b0, InferenceMode::BF16, 10, 0);
    cold_point_36!(cold36_w10_b2, InferenceMode::BF16, 10, 2);
    cold_point_36!(cold36_w16_b0, InferenceMode::BF16, 16, 0);
    cold_point_36!(cold36_w16_b2, InferenceMode::BF16, 16, 2);

    // **Does compression push the bracket out?** The BF16 points put the ceiling
    // near 16, and the suspicion is that what runs out is KV: a wave's rows are
    // cheap while the arena can serve them from free regions and expensive once
    // it cannot. C10 stores the same context in roughly a seventh of the bytes,
    // so if the ceiling is KV-bound it should move outward with compression —
    // and widths that could not be measured at all in BF16 (32 sessions did not
    // finish a 256-token run) should come into reach.
    //
    // If instead these track the BF16 numbers, the ceiling is about the wave's
    // row count rather than the memory behind it, and compression is orthogonal.
    cold_point_36!(cold36_c10_w20_b0, InferenceMode::C10, 20, 0);
    cold_point_36!(cold36_c10_w20_b2, InferenceMode::C10, 20, 2);
    cold_point_36!(cold36_c10_w32_b0, InferenceMode::C10, 32, 0);
    cold_point_36!(cold36_c10_w32_b2, InferenceMode::C10, 32, 2);

    /// **Speculative decode on the 3.6-35B — the production target.**
    ///
    /// The same sweep as the dense gates
    /// ([`crate::models::quantized_qwen35::tests::speculative_gate`]): the
    /// StoryRewrite fixture at 256 generated tokens, run once plain and once
    /// per draft budget, with the lossless driver and the checkpoint's own MTP
    /// head as the drafter. Every run validates against the same fixture at 100%, so a
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
        // **Widths 1 and 4, and this is the ceiling the fixture supports.** The
        // scheduler's draft budget is a step function of wave width
        // (`SPEC_MAX_WIDTH`), so this sweep is what sets it — but width 8 cannot
        // be a row here: at eight concurrent sessions the *baseline* run drops
        // to 6/8 against the expected string, two sessions diverging by a single
        // pronoun ~780 characters in. That is the lineage's known stochastic
        // floor — batched reductions reassociate with cohort width, and one
        // near-tied argmax eventually falls the other way — not a decode defect,
        // and it fails the 100% threshold before any speculative budget runs.
        // Widening the measurement needs a fixture that tolerates it, not a
        // looser threshold here: the threshold is what makes this gate a
        // losslessness check.
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
            // A gate that silently fell back to plain decode would still pass
            // — speculation is lossless, so the only symptom is the speedup
            // going away. Assert the drafter is really there.
            assert!(
                m.has_drafter(),
                "the pinned 3.6-35B has no MTP head — the pin has moved off the \
                 -MTP-GGUF repo, or its conversion dropped the NextN tensors"
            );
            println!("✓ Model loaded\n");
            Ok(m)
        })
    }
}
