//! Qwen3.5-35B-A3B — the routed (MoE) hybrid's model file.
//!
//! Everything structural — config parsing, the DeltaNet mixer, the wave
//! sweep, the loaders, the shared-expert MoE block — is lineage machinery in
//! [`super::qwen35`]; this file pins the 35B itself: its checkpoint, and the
//! gates that hold the engine to exact outputs on it.
//!
//! The 35B routes on **every** layer — DeltaNet and attention alike, 40
//! layers × 256 experts, top-8 plus a gated shared expert. Its 19.5 GB of
//! experts do not fit a 16 GB card and are not meant to: the three-tier
//! cache streams them VRAM ← pinned RAM ← the pack file, so what has to fit
//! is the dense weights plus a working set. A failure here that the dense 9B
//! does not show is the expert path — the loader's span measurement, the
//! `moe_layer_idx` numbering, or the elastic boundary.

use std::path::Path;

use candle::{Device, Result};
use candle_nn::kv_cache::QWEN35_MOE_KV_FACTORS;

use crate::models::draft_ladder::QWEN35_35B_A3B_DRAFT;

use super::qwen35::{load_hybrid_gguf, HybridBatched, Qwen35LoadOptions};

/// Pinned checkpoint (repo, revision, file) — revision-pinned so an upstream
/// re-upload fails the gate loudly instead of drifting under it.
///
/// **The `-MTP-` repo**, because the plain conversion drops the NextN tensors
/// and a model without them cannot speculate. On this checkpoint the head is a
/// full routed block — `blk.40`, its own router, 256 experts and shared expert
/// at the trunk's geometry — which is why the expert cache carries one layer
/// more than the trunk has (`expert_host_refs`).
///
/// **The quant moved with the repo**: `Q4_K_M` → `UD-Q4_K_M`, because the MTP
/// repo publishes only the UD variants. That is a different quantization of the
/// same weights, so the KV error-threshold factors below were derived against
/// the old file and the compression ladder is the gate that says whether they
/// still hold. Re-derive them if C10 starts flaking rather than loosening the
/// rung — the thresholds are model-specific by standing rule.
pub const QWEN35_35B_A3B: (&str, &str, &str) = (
    "unsloth/Qwen3.5-35B-A3B-MTP-GGUF",
    "63af8373893a7a73c6dfcb84cb63d815981da5e0",
    "Qwen3.5-35B-A3B-UD-Q4_K_M.gguf",
);

/// Load the routed Qwen3.5 checkpoint and wrap it for the scheduler.
///
/// The 35B's concrete entry: refuses a dense checkpoint (which loaded through
/// this entry would silently skip the expert cache the caller sized VRAM
/// around) and constructs the lineage's [`HybridBatched`] with the 35B's own
/// derived KV threshold factor row.
pub fn from_gguf_path(
    file_path: &Path,
    device: &Device,
    options: Qwen35LoadOptions,
) -> Result<HybridBatched> {
    let model = load_hybrid_gguf(file_path, device, options)?;
    if model.cfg.moe.is_none() {
        candle::bail!(
            "quantized_qwen35_moe: {file_path:?} is a dense checkpoint — \
             load it through quantized_qwen35 instead"
        );
    }
    HybridBatched::new(model, QWEN35_MOE_KV_FACTORS, QWEN35_35B_A3B_DRAFT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
    use crate::models::batched_inference::InferenceMode;
    use crate::models::dialect::Dialect;
    use crate::models::quantized_qwen35::tokenizer_json;
    use candle::quantized::Int8Mode;
    use hf_hub::RepoType;

    fn pinned() -> Result<std::path::PathBuf> {
        hf_get(
            QWEN35_35B_A3B.0,
            RepoType::Model,
            QWEN35_35B_A3B.1,
            QWEN35_35B_A3B.2,
        )
    }

    /// The story-rewrite gate on the 35B-A3B: the hybrid *plus* the streaming
    /// expert cache, on a checkpoint where **every** layer routes.
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-35B-A3B GGUF (22 GB), repacks experts on first \
                run, and needs a GPU. Run with: cargo test --release --features cuda --lib \
                -p candle-transformers \
                quantized_qwen35_moe::tests::test_parallel_batched_forwarding_35b \
                -- --ignored --nocapture --test-threads=1 \
                (serial: these load multi-gigabyte checkpoints and will exhaust \
                the card if cargo runs them concurrently)"]
    fn test_parallel_batched_forwarding_35b() -> Result<()> {
        println!("\n=== Qwen3.5-35B-A3B hybrid MoE batched forwarding ===\n");
        let model_path = pinned()?;
        let device = Device::new_cuda(0)?;

        // One value for both the loader and the table's `int8` column — the
        // label is set independently of the loader, and a gate that pins one
        // and defaults the other reports a numeric path it is not running.
        // `Performance` rather than `auto_sized`, so the gate holds the
        // throughput/accuracy dial steady instead of letting it move with the
        // device.
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
                num_contexts: 2,
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
        ];

        // ── The top rung, at the widths worth seeing it at ───────────────────
        //
        // C10 is the calibration target: `QWEN35_MOE_KV_FACTORS` is tuned so the
        // whole range C0–C10 passes with C10 just under the breaking edge. A red
        // C10 row means the thresholds drifted past it — retighten the factor
        // row rather than widening tolerances.
        //
        // Run at **8 and 16** rather than one middling width. Both sit inside
        // the draft ladder's bracket, so both speculate at full budget, and the
        // pair shows the compression and the speculation compounding as the
        // cohort grows — which is the number this engine is actually for.
        configs.extend([8usize, 16].map(|n| TestConfig {
            mode: InferenceMode::C10,
            use_batched: true,
            num_contexts: n,
            num_repeats: 1,
            generate_max_len: 40,
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
                generate_max_len: 40,
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
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }

    /// The shared expert against the F32 reference, on the real 35B.
    ///
    /// This is the only part of Qwen3.5's MoE that Qwen3-MoE does not already
    /// have (there is no `shexp` tensor anywhere in that model), so it is the
    /// only part that needs its own parity check — the routed half is the
    /// same `SparseMoeBlock` its gates already cover.
    ///
    /// The reference weights are the production ones dequantized, so the only
    /// variable under test is the quantized kernel against F32 matmuls.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-35B GGUF from the HF cache (22 GB) and needs a GPU"]
    fn shared_expert_matches_the_f32_reference_on_real_weights() -> Result<()> {
        use crate::models::quantized_matmul::QMatMul;
        use crate::models::quantized_mlp::QuantizedMlp;
        use crate::models::qwen35::config::Qwen35Config;
        use crate::models::qwen35::loader::detect_arch;
        use crate::models::qwen35::quantized_moe::shared_expert_contribution;
        use candle::quantized::cuda::DynamicActs;
        use candle::quantized::{gguf_file::Content, QTensor};
        use candle::{DType, Tensor};
        use std::io::{BufReader, Seek, SeekFrom};

        let path = pinned()?;
        let device = Device::new_cuda(0)?;
        let mut reader = BufReader::new(std::fs::File::open(&path)?);
        let content = Content::read(&mut reader)?;
        let arch = detect_arch(&content);
        let cfg = Qwen35Config::from_gguf_metadata(&arch, &content.metadata)?;
        let moe = cfg.moe.expect("the 35B declares experts");
        println!(
            "arch {arch}: {} layers, {} experts top-{}, shared ffn {}, hidden {}",
            cfg.num_layers,
            moe.n_experts,
            moe.n_experts_used,
            moe.shared_expert_ffn_size,
            cfg.hidden_size,
        );
        assert_eq!(moe.n_experts, 256);
        reader.seek(SeekFrom::Start(0))?;

        let mut raw =
            |name: &str| -> Result<QTensor> { content.tensor(&mut reader, name, &device) };
        let p = "blk.0";
        let shared = QuantizedMlp::from_weights(
            raw(&format!("{p}.ffn_gate_shexp.weight"))?,
            raw(&format!("{p}.ffn_up_shexp.weight"))?,
            raw(&format!("{p}.ffn_down_shexp.weight"))?,
            Int8Mode::Off,
        )?;
        let gate_w = raw(&format!("{p}.ffn_gate_inp_shexp.weight"))?;
        // The checkpoint stores the shared gate as a `[hidden]` vector; the
        // matmul wants `[1, hidden]`.
        let gate_f32 = gate_w.dequantize(&device)?.reshape((1, cfg.hidden_size))?;
        let shared_gate = QMatMul::from_qtensor_with_mode(
            QTensor::quantize(&gate_f32, candle::quantized::GgmlDType::F32)?,
            Int8Mode::Off,
        )?;

        let t = 6usize;
        let x = Tensor::randn(0f32, 1.0, (1, t, cfg.hidden_size), &device)?;
        let got = shared_expert_contribution(
            &shared,
            &shared_gate,
            &DynamicActs::Float(x.clone()),
            DType::F32,
        )?;

        // Reference: the SwiGLU through the MLP's plain FP path (a different
        // code path from `forward_dynamic`, over the same weights), gated by
        // a hand-written `sigmoid(w·x)` — exactly how
        // `MoeWeights::forward` combines them.
        let flat = x.reshape((t, cfg.hidden_size))?;
        let y = shared.forward(&x)?.reshape((t, cfg.hidden_size))?;
        let gate = candle_nn::ops::sigmoid(&flat.matmul(&gate_f32.t()?)?)?;
        let want = y.broadcast_mul(&gate)?;

        let got = got.reshape((t, cfg.hidden_size))?;
        let diff = got
            .sub(&want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let rel = diff / scale.max(1e-6);
        println!("shared-expert rel {rel:.4}");
        assert!(
            rel < 1e-4,
            "shared expert diverged from the reference: {rel}"
        );
        Ok(())
    }

    /// The expert strides read off the real 35B: every expert must land on
    /// its own bytes, contiguously, covering the tensor exactly.
    ///
    /// This is pure offset arithmetic over the mmap, so it needs no GPU and
    /// no cache — but getting it wrong would hand the cache another expert's
    /// weights, which is the kind of error that produces a *plausible* model.
    #[test]
    #[ignore = "reads the pinned Qwen3.5-35B GGUF header from the HF cache (22 GB)"]
    fn expert_refs_tile_the_merged_tensors() -> Result<()> {
        use crate::models::qwen35::config::Qwen35Config;
        use crate::models::qwen35::expert_loader::expert_host_refs;
        use crate::models::qwen35::loader::detect_arch;
        use candle::quantized::gguf_file::Content;
        use std::io::BufReader;

        let path = pinned()?;
        let mut reader = BufReader::new(std::fs::File::open(&path)?);
        let content = Content::read(&mut reader)?;
        let arch = detect_arch(&content);
        let cfg = Qwen35Config::from_gguf_metadata(&arch, &content.metadata)?;
        let refs = expert_host_refs(&content, &cfg)?;

        let moe = cfg.moe.expect("35B declares experts");
        // Every trunk layer routes, and so does the MTP draft head at
        // `blk.{num_layers}` — its experts are the LAST entry, which is what
        // puts them at the `moe_layer_idx` its router asks the cache for.
        assert_eq!(
            refs.len(),
            cfg.num_layers + cfg.num_mtp_layers,
            "every layer of the 35B routes, the draft head included"
        );
        assert_eq!(refs[0].len(), moe.n_experts);
        assert_eq!(
            refs.last().expect("at least one layer").len(),
            moe.n_experts,
            "the head carries a full expert set, not a partial one"
        );
        println!(
            "{} MoE layers x {} experts, slot {} + {} + {} bytes",
            refs.len(),
            refs[0].len(),
            refs[0][0].gate_len,
            refs[0][0].up_len,
            refs[0][0].down_len
        );

        for (li, layer) in refs.iter().enumerate() {
            // Uniform stride, ascending, no overlap and no gap.
            let (g0, u0, d0) = (
                layer[0].gate_offset,
                layer[0].up_offset,
                layer[0].down_offset,
            );
            let (gs, us, ds) = (layer[0].gate_len, layer[0].up_len, layer[0].down_len);
            assert!(gs > 0 && us > 0 && ds > 0, "layer {li} has a zero stride");
            for (e, r) in layer.iter().enumerate() {
                assert_eq!(r.gate_offset, g0 + e * gs, "layer {li} expert {e} gate");
                assert_eq!(r.up_offset, u0 + e * us, "layer {li} expert {e} up");
                assert_eq!(r.down_offset, d0 + e * ds, "layer {li} expert {e} down");
                assert_eq!(r.gate_len, gs);
                assert_eq!(r.gate_shape, layer[0].gate_shape);
                assert_eq!(r.gate_dtype, layer[0].gate_dtype);
            }
            // The three projections' expert blocks must not overlap each
            // other either — they are separate tensors in the file.
            let spans = [
                (g0, g0 + layer.len() * gs),
                (u0, u0 + layer.len() * us),
                (d0, d0 + layer.len() * ds),
            ];
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let (a, b) = (spans[i], spans[j]);
                    assert!(
                        a.1 <= b.0 || b.1 <= a.0,
                        "layer {li}: projection spans {a:?} and {b:?} overlap"
                    );
                }
            }
        }
        // The shapes are the per-expert 2-D projections, not the merged 3-D.
        assert_eq!(refs[0][0].gate_shape.len(), 2);
        assert_eq!(
            refs[0][0].gate_shape,
            vec![moe.expert_ffn_size, cfg.hidden_size],
            "gate is [expert_ffn, hidden] per expert"
        );
        Ok(())
    }

    /// Speculative decode on the 3.5-35B — the 3.6 gate's sibling, on the
    /// checkpoint where every layer routes. See
    /// [`crate::models::quantized_qwen35::tests::speculative_gate`].
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-35B-A3B GGUF (22 GB) and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35_moe::tests::speculative_decode_35b \
                -- --ignored --nocapture --test-threads=1"]
    fn speculative_decode_35b() -> Result<()> {
        use crate::models::quantized_qwen35::tests::speculative_gate;

        let model_path = pinned()?;
        let int8mode = Int8Mode::Performance;
        speculative_gate("Qwen3.5-35B-A3B", int8mode, &[1, 4], move || {
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
                "the pinned 35B has no MTP head — the pin has moved off the \
                 -MTP-GGUF repo, or its conversion dropped the NextN tensors"
            );
            println!("✓ Model loaded\n");
            Ok(m)
        })
    }
}
