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
//! The dense models are the hybrid engine's proving ground: the 0.8B is the
//! smallest checkpoint that exercises the whole hybrid, and the 9B runs the
//! same sweep as the 35B with the expert cache out of the picture — so a
//! failure here is the hybrid, and a failure only on the MoE sibling is the
//! experts.
//!
//! Both dense gates run **Q8_0 weights on the int8 matmul**. The 0.8B was
//! pinned to the BF16 conversion while its top compression rung depended on
//! BF16-width activations; once the rung was traced to that and the int8 path
//! carried it, Q8_0 became the pin — it is the same arithmetic the 9B and the
//! MoE siblings run, and roughly twice the throughput.

use std::path::Path;

use candle::{Device, Result};
use candle_nn::kv_cache::{QWEN35_0_8B_KV_FACTORS, QWEN35_9B_KV_FACTORS};

use crate::models::draft_ladder::{QWEN35_0_8B_DRAFT, QWEN35_9B_DRAFT};

use super::qwen35::{load_hybrid_gguf, HybridBatched, Qwen35LoadOptions};

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
    // Both per-checkpoint rows are chosen by the same width test that already
    // tells the two dense siblings apart.
    let (kv_factors, draft) = if model.cfg.hidden_size >= 4096 {
        (QWEN35_9B_KV_FACTORS, QWEN35_9B_DRAFT)
    } else {
        (QWEN35_0_8B_KV_FACTORS, QWEN35_0_8B_DRAFT)
    };
    HybridBatched::new(model, kv_factors, draft)
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
pub(crate) mod tests {
    use super::*;
    use crate::models::batch_test::test_helpers::hf_get;
    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};
    use crate::models::batched_inference::{InferenceMode, ManagedBatchedModel};
    use crate::models::dialect::Dialect;
    use crate::models::qwen35::mtp::MTP_MAX_DRAFT;
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

    /// **How the recurrent scan's cost scales with prompt length.**
    ///
    /// `delta_net_prefill_state` walks its span's chunks *serially* inside the
    /// block, and its grid — `(n_v_heads, DNP_DIM/DNP_TV, n_spans)` — carries no
    /// length dimension at all. So the serial depth grows as
    /// `ceil(len / DNP_CHUNK)` while the parallelism stays pinned at 128 blocks
    /// on a 110-SM card. Profiled at the gate's 134-token prompt that is
    /// invisible (3 chunks); this sweep is what makes it visible, and what sizes
    /// a blocked scan that would put sections on `grid.z`.
    ///
    /// Reports `dn:mix` (the scan's span) against `dn:ffn` (dense GEMMs, known
    /// compute-bound) so length-scaling is read against a control that should
    /// stay linear.
    /// Gated on `cuda` because it reads the scan's chunk width from the kernel's
    /// own constant, which lives in the cuda-only `candle-kernels` dependency —
    /// the alternative is a second copy of that number drifting out of sync.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "downloads the pinned 9B GGUF and needs a GPU. Run with: \
                cargo test --release --features cuda,profile --lib \
                -p candle-transformers quantized_qwen35::tests::prefill_scan_scaling \
                -- --ignored --nocapture"]
    fn prefill_scan_scaling() -> Result<()> {
        use crate::models::batched_inference::BatchedConfig;
        use crate::models::profile::pipeline_snapshot_and_reset;
        use candle::Tensor;
        use candle_kernels::delta_net::DELTA_NET_PREFILL_CHUNK;
        use std::time::Instant;

        let model_path = pinned(QWEN35_9B)?;
        let device = Device::new_cuda(0)?;
        let model = from_gguf_path(
            &model_path,
            &device,
            Qwen35LoadOptions {
                int8mode: Some(Int8Mode::auto(&device)),
                expert_pack_dir: None,
                mtp_path: None,
            },
        )?;

        // A real prompt cycled to length: the token ids must be in-vocabulary,
        // and repetition is fine here because the scan's cost is a function of
        // shape, not of content.
        let params = TestParams::new(4, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?;
        let mut seed = params.system_prompt_tokens(0);
        seed.extend(params.user_prompt_tokens(0));

        println!(
            "\n  {:>7} {:>7} {:>10} {:>12} {:>12} {:>10}",
            "tokens", "chunks", "wall ms", "dn:mix ms", "dn:ffn ms", "mix/chunk"
        );
        for &len in &[128usize, 512, 1024, 2048, 4096, 8192] {
            let ids: Vec<u32> = (0..len).map(|i| seed[i % seed.len()]).collect();
            let mut session = model.create_batched_session(BatchedConfig::default())?;
            let seq = session.create_sequence()?;
            let t = Tensor::from_vec(ids, (1, len), &device)?;

            let _ = pipeline_snapshot_and_reset();
            device.synchronize()?;
            let t0 = Instant::now();
            let step = model.forward_wave(
                &mut session,
                &[],
                &[],
                &[seq],
                &[t],
                &[],
                &[],
                0,
                model.num_layers(),
                None,
            )?;
            drop(step);
            device.synchronize()?;
            let wall = t0.elapsed().as_secs_f64() * 1e3;
            // `dn:mix` / `dn:ffn` are `gpu_span`s — CUDA event pairs that only
            // reach the accumulator when drained, so the snapshot is empty
            // without this.
            crate::models::profile::gpu_drain_blocking();
            let snap = pipeline_snapshot_and_reset();
            let get = |name: &str| -> f64 {
                snap.entries
                    .iter()
                    .find(|(n, _, _)| n == name)
                    .map(|(_, ms, _)| *ms)
                    .unwrap_or(0.0)
            };
            let (mix, ffn) = (get("dn:mix"), get("dn:ffn"));
            // The scan's serial depth at this length. Read from the kernel's own
            // constant, so retuning the chunk width cannot leave this column
            // quietly reporting the old one.
            let chunks = len.div_ceil(DELTA_NET_PREFILL_CHUNK);
            println!(
                "  {len:>7} {chunks:>7} {wall:>10.1} {mix:>12.1} {ffn:>12.1} {:>10.3}",
                mix / chunks as f64
            );
            // At the longest length, the whole span table: `dn:mix` and `dn:ffn`
            // are both linear here, so whatever makes the wall superlinear is a
            // third span — print it rather than infer it.
            if len == 8192 {
                let mut rows = snap.entries.clone();
                rows.sort_by(|a, b| b.1.total_cmp(&a.1));
                println!("    --- full span table at {len} tokens ---");
                for (name, ms, count) in rows.iter().take(12) {
                    println!("    {name:<24} {ms:>9.1} ms  (×{count})");
                }
            }
            session.free_sequence(seq)?;
            model.release_sequence(seq)?;
        }
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
                    mtp_path: None,
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
        let model_path = pinned(QWEN35_0_8B)?;
        let device = Device::new_cuda(0)?;

        // **Resolved here, then used twice**, so the table's `int8` column
        // cannot disagree with what the model loaded. This gate set only the
        // loader and left the label defaulting to `Off`, so it printed `off`
        // while running int8 — visible only as a doubled throughput that the
        // column said should not exist.
        let int8mode = Int8Mode::auto(&device);
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
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
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
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C0,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C1,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C2,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C3,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C4,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C5,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C6,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C7,
                use_batched: true,
                num_contexts: 2,
                num_repeats: 1,
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
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::C9,
                use_batched: true,
                num_contexts: 5,
                num_repeats: 1,
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
                test_mode: Some(TestMode::StoryRewrite),
            },
        ];

        let load = || {
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    // `auto`, like the 9B gate: the checkpoint is Q8_0, so the
                    // wide projections get KO twins and this rung runs the same
                    // numeric path as the rest of the lineage — and as a
                    // deployment would. The DeltaNet `w_alpha`/`w_beta` sit below
                    // the KO tile and stay dense on their own (see
                    // `QWEN35_0_8B`); that costs those two weights their twin,
                    // not the model its int8 path.
                    int8mode: Some(int8mode),
                    expert_pack_dir: None,
                    mtp_path: None,
                },
            )?;
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }

    /// **Speculative decode on the dense 9B, measured against itself.**
    ///
    /// The lineage's smallest checkpoint that can speculate at all, and the
    /// size where a decode step is weight-bandwidth-bound rather than
    /// launch-bound — so the win here is the one a production session would
    /// see. The 0.8B has no gate of its own because it has no MTP head in any
    /// conversion: [`speculative_gate`] on it would run the same plain decode
    /// under three budget labels and pass unconditionally.
    ///
    /// The same StoryRewrite configs run once per draft budget: once through
    /// the plain decode loop, then through the lossless speculative driver
    /// with the checkpoint's own MTP head as the drafter. Two things come out
    /// of it.
    ///
    /// * **Correctness.** Speculation accepts only the model's own argmaxes, so
    ///   every run must produce the same text as the first — and the harness
    ///   validates all of them against the same fixture at the same 100%
    ///   threshold. A rewind that left the recurrent state one token ahead of
    ///   the KV would show up here as degraded output, which is exactly the
    ///   failure `truncate_sequence` used to refuse to risk.
    /// * **Throughput.** `t/s (single)` is the per-session decode rate; the
    ///   ratio between the tables is what speculation bought.
    ///
    /// A rewrite task is the drafter's best case *and its honest one*: the
    /// output overlaps the prompt heavily, which is the same shape as editing
    /// code — the workload this engine is for.
    ///
    /// Swept at width, like its siblings: the ladder gate runs this same
    /// checkpoint in BF16 at 4 and 20 contexts, so a speculative step that
    /// cannot is the speculative path's problem and not the card's.
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-9B GGUF (7.5 GB) and needs a GPU. Run with: \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::speculative_decode_9b \
                -- --ignored --nocapture --test-threads=1"]
    fn speculative_decode_9b() -> Result<()> {
        speculative_gate(
            "Qwen3.5-9B",
            Int8Mode::Off,
            &[1, 4],
            dense_loader(pinned(QWEN35_9B)?, Int8Mode::Off),
        )
    }

    /// **Is a width ceiling about KV bytes, or about the session count?**
    ///
    /// Holding the width fixed while varying the generated length separates
    /// them: total KV scales with `width × tokens`, while anything dimensioned
    /// by session count — DeltaNet's per-sequence recurrent state, the transient
    /// tier, admission — does not move at all.
    ///
    /// * Fast when short, slow when long → **KV capacity**.
    /// * Flat across lengths → **session count**, and the search moves to what
    ///   scales with it.
    ///
    /// It measured flat (242.6 / 257.1 / 262.8 tok/s at 64 / 128 / 256 tokens,
    /// width 10), which is what pointed at DeltaNet: 30 recurrent layers each
    /// holding a `32 × 128 × 128` F32 state per sequence, doubled for the
    /// live/backup ping-pong, is 120 MiB per session and is untouched by KV
    /// compression.
    ///
    /// **Each length is its own `params.run` and therefore its own model load**,
    /// which is what makes this readable — see `cold_speculative_point` for why
    /// that matters on this card.
    ///
    /// Budget 0 only. The question is about the baseline, and drafting would
    /// just add rows to both arms.
    #[test]
    #[ignore = "targeted measurement: width 10, three generated lengths, one model load. \
                Run cold. cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::kv_or_width_9b -- --ignored --nocapture \
                --test-threads=1"]
    fn kv_or_width_9b() -> Result<()> {
        let load = dense_loader(pinned(QWEN35_9B)?, Int8Mode::Off);
        for tokens in [64usize, 128, 256] {
            println!("\n=== Qwen3.5-9B: width 10, {tokens} generated tokens, plain decode ===\n");
            let params = TestParams::new(tokens, &tokenizer_json()?, Dialect::qwen35())
                .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
                .with_suppress_thinking(true)
                .with_timeout_secs(3600)
                .with_speculative(0)
                .with_majority_pass_threshold(50)
                .with_int8mode(Int8Mode::Off);
            params.run(
                vec![TestConfig {
                    mode: InferenceMode::BF16,
                    use_batched: true,
                    num_contexts: 10,
                    num_repeats: 1,
                    test_mode: Some(TestMode::StoryRewrite),
                }],
                &load,
            )?;
        }
        Ok(())
    }

    /// One width, one budget, one process — the only shape this machine can be
    /// measured in.
    ///
    /// A laptop card runs at full boost for the first tens of seconds of load
    /// and settles to roughly half that afterwards (SM clock swings 450–2310 MHz
    /// with the SW power cap asserting, at 67–77 °C — spent boost budget, not
    /// thermal shutdown). Any gate that runs several configs against one loaded
    /// model therefore measures *position in the run* as much as the variable it
    /// varies: the same config measured 247 tok/s as the first config of a load
    /// and 103 as the third, across four repetitions, with the KV region ledger
    /// byte-identical between them.
    ///
    /// Every earlier "width curve" here was that artefact — width rose with
    /// position, so boost depletion read as a cliff. One point per invocation is
    /// what breaks the confound; the caller is responsible for letting the card
    /// settle between them.
    pub(crate) fn cold_speculative_point<M>(
        label: &str,
        tokenizer: &str,
        mode: InferenceMode,
        width: usize,
        budget: usize,
        int8mode: Int8Mode,
        load: impl Fn() -> Result<M>,
    ) -> Result<()>
    where
        M: ManagedBatchedModel,
    {
        println!("\n=== {label} cold point: {mode:?}, width {width}, draft budget {budget} ===\n");
        let params = TestParams::new(256, tokenizer, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_timeout_secs(3600)
            .with_speculative(budget)
            // A throughput measurement, and a wide cohort cannot match a fixed
            // string at any useful rate — the lineage's stochastic floor grows
            // with width. Kept only as a garbage detector: a broken accept rule
            // reproduces a 1228-character fixture in none of the cohort, not a
            // quarter of it. Losslessness is gated at 100% in
            // `speculative_decode_9b`.
            .with_majority_pass_threshold(25)
            .with_int8mode(int8mode);
        // Checked before the run, not after: a budget the checkpoint cannot
        // honour degrades silently to plain decode, and this measurement sets a
        // production constant.
        let checked = move || {
            let m = load()?;
            assert_drafter(&m, budget)?;
            Ok(m)
        };
        params.run(
            vec![TestConfig {
                mode,
                use_batched: true,
                num_contexts: width,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            }],
            checked,
        )
    }

    /// One `#[test]` per measurement point, so each runs in its own process with
    /// its own boost budget. A macro because twelve hand-written copies of the
    /// same two lines is where a transposed width or budget hides.
    macro_rules! cold_point {
        ($name:ident, $width:expr, $budget:expr) => {
            #[test]
            #[ignore = "cold measurement point — run singly, letting the card settle between \
                        points. See `cold_speculative_point`."]
            fn $name() -> Result<()> {
                cold_speculative_point(
                    "Qwen3.5-9B",
                    &tokenizer_json()?,
                    InferenceMode::BF16,
                    $width,
                    $budget,
                    Int8Mode::Off,
                    dense_loader(pinned(QWEN35_9B)?, Int8Mode::Off),
                )
            }
        };
    }

    cold_point!(cold_w4_b0, 4, 0);
    cold_point!(cold_w10_b0, 10, 0);
    cold_point!(cold_w10_b1, 10, 1);
    cold_point!(cold_w10_b2, 10, 2);
    cold_point!(cold_w16_b0, 16, 0);
    cold_point!(cold_w16_b1, 16, 1);
    cold_point!(cold_w16_b2, 16, 2);
    cold_point!(cold_w20_b0, 20, 0);
    cold_point!(cold_w20_b1, 20, 1);
    cold_point!(cold_w20_b2, 20, 2);
    // **Width 32 needs a card this lineage's dev machines do not have.** Thirty
    // recurrent layers holding a `32 × 128 × 128` F32 state per sequence, doubled
    // for the live/backup ping-pong, is ~120 MiB per session and compression
    // cannot touch it — so 32 sessions is ~3.8 GiB of DeltaNet state before any
    // KV, and admission refuses on the 16 GB and 24 GB boxes. They are kept, and
    // `#[ignore]`d like every other point here, because they are the instrument
    // for the 72 GB Blackwell: the answer they carry is whether the turn at 16
    // is the card or the design, and only a bigger card can say.
    cold_point!(cold_w32_b0, 32, 0);
    cold_point!(cold_w32_b1, 32, 1);
    cold_point!(cold_w32_b2, 32, 2);

    /// **Does throughput decay across configs within one model load?**
    ///
    /// Every sweep so far has been read as a width curve, and the widths were
    /// confounded with their position in the run. Width 10 at 256 tokens
    /// measured 118 tok/s as the third config of a load and 263 as the first —
    /// the same deterministic work, 2.2x apart — while a fresh load is fast
    /// every time.
    ///
    /// This runs ONE config three times over, so width, length, budget and
    /// fixture are all held constant and only position varies. A monotone decay
    /// means state is surviving between configs that should not: stranded
    /// arenas holding regions, recurrent or draft state, unreleased sequences.
    /// It would also mean every multi-config gate in this file under-reports its
    /// later rows, and that the "width cliff" was never about width.
    #[test]
    #[ignore = "targeted measurement: one config three times over, single model load. \
                cargo test --release --features cuda --lib -p candle-transformers \
                quantized_qwen35::tests::decay_across_configs_9b -- --ignored --nocapture \
                --test-threads=1"]
    fn decay_across_configs_9b() -> Result<()> {
        let load = dense_loader(pinned(QWEN35_9B)?, Int8Mode::Off);
        let params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_timeout_secs(3600)
            .with_speculative(0)
            .with_majority_pass_threshold(50)
            .with_int8mode(Int8Mode::Off);
        let one = TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 10,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        };
        params.run(vec![one.clone(), one.clone(), one], &load)
    }

    /// Run the speculative comparison for a dense checkpoint of the lineage.
    ///
    /// **256 generated tokens**, not the gates' 10: speculation is a property
    /// of the decode loop, and ten tokens is six driver steps — a number small
    /// enough that the prefill and the first block dominate it. The draft
    /// budget is swept so the table shows where the yield stops paying for the
    /// rows it adds.
    pub(crate) fn speculative_gate<M>(
        label: &str,
        int8mode: Int8Mode,
        widths: &[usize],
        load: impl Fn() -> Result<M>,
    ) -> Result<()>
    where
        M: ManagedBatchedModel,
    {
        let configs: Vec<TestConfig> = widths
            .iter()
            .map(|&n| TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: n,
                num_repeats: 1,
                test_mode: Some(TestMode::StoryRewrite),
            })
            .collect();
        // Up to the head's own ceiling, [`MTP_MAX_DRAFT`], and no further: past
        // it the drafter clamps, so the extra rows would report the same
        // configuration twice. The turnover the sweep used to look for is what
        // set that constant — the measurements are recorded there.
        for draft in 0..=MTP_MAX_DRAFT {
            println!(
                "\n=== {label}: {} ===\n",
                if draft == 0 {
                    "plain decode (baseline)".to_string()
                } else {
                    format!("speculative decode, draft budget {draft}")
                }
            );
            let mut params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
                .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
                .with_suppress_thinking(true)
                .with_timeout_secs(3600);
            params = params.with_speculative(draft).with_int8mode(int8mode);
            // Same guard the cold points carry, for the same reason and with
            // more at stake: this is the sweep the production bracket is read
            // from, so a checkpoint that has quietly lost its NextN head would
            // report every budget as 1.00× here and read as "speculation stopped
            // paying" rather than "the drafter is missing".
            let checked = || {
                let m = load()?;
                assert_drafter(&m, draft)?;
                Ok(m)
            };
            params.run(configs.clone(), checked)?;
        }
        Ok(())
    }

    /// Assert the model really carries a drafter, for a caller whose whole
    /// purpose is measuring speculation.
    ///
    /// A budget the model cannot honour is not an error — `speculative_draft`
    /// answers with no proposals and the step degrades to plain decode, which is
    /// exactly the behaviour a checkpoint without a head should have. That makes
    /// a pin or conversion regression invisible here in the worst way: the
    /// measurement still produces a number, the number is a plain-decode number,
    /// and it is the number the production bracket gets set from.
    fn assert_drafter<M: ManagedBatchedModel>(model: &M, budget: usize) -> Result<()> {
        if budget > 0 && model.draft_budget(1) == 0 {
            candle::bail!(
                "this checkpoint reports a zero draft budget even at width 1 — it carries no \
                 NextN head, so a budget-{budget} measurement would silently report plain decode"
            );
        }
        Ok(())
    }

    /// The dense lineage's loader, as [`speculative_gate`] takes it.
    fn dense_loader(
        model_path: std::path::PathBuf,
        int8mode: Int8Mode,
    ) -> impl Fn() -> Result<HybridBatched> {
        move || {
            let device = Device::new_cuda(0)?;
            let m = from_gguf_path(
                &model_path,
                &device,
                Qwen35LoadOptions {
                    int8mode: Some(int8mode),
                    expert_pack_dir: None,
                    mtp_path: None,
                },
            )?;
            // A gate that silently fell back to plain decode would still pass
            // — speculation is lossless, so the only symptom is the speedup
            // going away. Assert the drafter is really there.
            assert!(
                m.has_drafter(),
                "the pinned dense checkpoint has no MTP head — the pin has moved \
                 off the -MTP-GGUF repo, or its conversion dropped the NextN tensors"
            );
            println!("✓ Model loaded\n");
            Ok(m)
        }
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
        // `auto`, so the gate runs the twin production selects rather than a
        // pinned one. It used to pin `Performance` on the grounds that the two
        // twins are interchangeable — measured true *on this model* (every rung
        // valid either way, throughput inside the run-to-run band) and false in
        // general: the same swap takes Llama-3's ladder to C6 0/1, C7 0/1,
        // C8 9/10. A dial that is free on one model is not free on the next, so
        // the gate follows the default instead of asserting the two are alike.
        let int8mode = Int8Mode::auto(&device);
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
                test_mode: Some(TestMode::StoryRewrite),
            },
            TestConfig {
                mode: InferenceMode::BF16,
                use_batched: true,
                num_contexts: 1,
                num_repeats: 1,
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
                    mtp_path: None,
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
                mtp_path: None,
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
