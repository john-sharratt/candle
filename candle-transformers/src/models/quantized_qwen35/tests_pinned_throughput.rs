//! Measures prefill and decode throughput, and how the recurrent scan's cost
//! grows with prompt length, on the pinned dense checkpoints.
//!
//! The measurements drive the forward directly, with a device sync around each
//! step, so the figures are of completed work rather than of launches. Each
//! checkpoint is resolved through an override, so a machine can run the same
//! gates against another conversion, and a figure measured that way is a
//! comparison and not a re-derivation. The widest rungs are gated on the card's
//! total VRAM, because the cache for that many sessions would not fit a smaller
//! card. The widest row is appended last, since it is the one most able to
//! exhaust the device.

use super::*;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::model_overrides::{self, Checkpoint};

    use crate::models::batch_test::test_helpers::gate_pack;

    use crate::models::batch_test::utils::{TestConfig, TestMode, TestParams};

    use crate::models::batched_inference::{InferenceMode, ManagedBatchedModel};

    use crate::models::dialect::Dialect;

    use crate::models::model_pack::{PackFamily, PackRequest};

    use candle::quantized::Int8Mode;

    use std::path::PathBuf;

    /// A pinned dense checkpoint's model pack at `mode`, its checkpoint resolved
    /// through [`model_overrides`] under `key` in `models.override.yaml`'s
    /// `checkpoints:` — so a machine can run these gates against another
    /// conversion of the same model, a different quant of the pinned repo say,
    /// without editing the pin. With no override it is exactly `spec`.
    ///
    /// The gates' KV factor rows and draft ladders are derived against the
    /// pinned file, so a figure measured on an override is a comparison, not a
    /// re-derivation.
    pub(in super::super) fn pinned(
        key: &str,
        spec: (&str, &str, &str),
        device: &Device,
        mode: Option<Int8Mode>,
    ) -> Result<PathBuf> {
        let ck = model_overrides::checkpoint(key, Checkpoint::new(spec.0, spec.1, spec.2));
        gate_pack(
            &PackRequest::of(
                PackFamily::Qwen35,
                (&ck.repo, &ck.revision, &ck.filename),
                (TOKENIZER_REPO, TOKENIZER_REV),
                mode,
            ),
            device,
        )
    }

    /// [`QWEN35_9B`]'s pack, overridable as `Qwen35_9B`.
    pub(in super::super) fn qwen35_9b(device: &Device, mode: Option<Int8Mode>) -> Result<PathBuf> {
        pinned("Qwen35_9B", QWEN35_9B, device, mode)
    }

    /// [`QWEN35_0_8B`]'s pack, overridable as `Qwen35_0_8B`.
    pub(in super::super) fn qwen35_0_8b(
        device: &Device,
        mode: Option<Int8Mode>,
    ) -> Result<PathBuf> {
        pinned("Qwen35_0_8B", QWEN35_0_8B, device, mode)
    }

    /// Prefill and decode rate for one model, measured on the engine directly.
    ///
    /// The gate's throughput figures include everything the harness does around
    /// a forward — tokenising, slicing the prompt into waves, sampling policy,
    /// validation — so a slow gate does not say whether the engine or the
    /// scaffolding is slow. This drives `forward_wave` itself: one wave for the
    /// whole prompt, then N single-token waves, with a device sync around each
    /// so the timings are of completed work rather than of launches.
    pub(in super::super) fn measure(
        model: &HybridBatched,
        prompt: &[u32],
        decode_steps: usize,
    ) -> Result<()> {
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
    pub(in super::super) fn profile_layers(model: &HybridBatched, prompt: &[u32]) -> Result<()> {
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
        let mut residual = model.claim_residual(&session, prompt.len())?;
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
                Some(&mut residual),
            )?;
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
        let mut residual = model.claim_residual(&session, 1)?;
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
                Some(&mut residual),
            )?;
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

        let device = Device::new_cuda(0)?;
        let model = from_pack(&qwen35_9b(&device, Some(Int8Mode::auto(&device)))?, &device)?;

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

        let off = Some(Int8Mode::Off);
        for (label, path) in [
            ("0.8B", qwen35_0_8b(&device, off)?),
            ("9B", qwen35_9b(&device, off)?),
        ] {
            let model = from_pack(&path, &device)?;
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
        let device = Device::new_cuda(0)?;

        // **Resolved here, then used twice**, so the table's `int8` column
        // cannot disagree with what the model loaded. This gate set only the
        // loader and left the label defaulting to `Off`, so it printed `off`
        // while running int8 — visible only as a doubled throughput that the
        // column said should not exist.
        //
        // `auto`, like the 9B gate: the checkpoint is Q8_0, so the wide
        // projections get KO twins and this rung runs the same numeric path as
        // the rest of the lineage — and as a deployment would. The DeltaNet
        // `w_alpha`/`w_beta` sit below the KO tile and stay dense on their own
        // (see `QWEN35_0_8B`); that costs those two weights their twin, not the
        // model its int8 path.
        let int8mode = Int8Mode::auto(&device);
        let pack = qwen35_0_8b(&device, Some(int8mode))?;
        let params = TestParams::new(10, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_print_outputs(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(1800);

        let mut configs = vec![
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

        // **The wide C8 rung, on a card with room for it.** C8 is this sweep's
        // width row — see its ×32 entry above — and a few hundred concurrent
        // contexts is the aggregate-throughput measurement that width exists to
        // produce. It is gated on total VRAM rather than pinned, because the KV
        // of this many sessions does not fit a 16 GB or 24 GB card at this depth
        // and the row would report an OOM on two of the three dev machines.
        //
        // **Additive, not a replacement.** The ×32 row stays, so the two read as
        // a scaling curve and the number the sweep has always reported keeps its
        // meaning. C10×10 is untouched for the same reason in reverse: that row
        // is the calibration edge (`QWEN35_0_8B_KV_FACTORS` is tuned so C10 sits
        // just under it), and widening the rung a threshold is derived against
        // would silently move the thing being measured.
        //
        // Appended last deliberately: it is the widest row and the only one that
        // can exhaust the device, so every row that must pass is already
        // measured by the time it runs.
        const WIDE_C8_MIN_VRAM_BYTES: usize = 64 * 1024 * 1024 * 1024;
        let total_vram = match &device {
            Device::Cuda(d) => d.mem_get_info().map(|(_free, total)| total).unwrap_or(0),
            _ => 0,
        };
        let gib = |b: usize| b as f64 / (1024.0 * 1024.0 * 1024.0);
        if total_vram > WIDE_C8_MIN_VRAM_BYTES {
            println!(
                "C8 wide rung: {:.1} GiB of VRAM clears the {:.0} GiB gate — running ×64/×128/×256\n",
                gib(total_vram),
                gib(WIDE_C8_MIN_VRAM_BYTES),
            );
            // **Three widths, because one cannot tell scaling from a ceiling.**
            // A single wide row below ×32's says only that the number is lower,
            // not whether width is still buying throughput, has flattened, or has
            // gone past its useful end — and this rung has already been read
            // wrong once in exactly that way.
            //
            // What it read wrong, recorded because the shape of the mistake
            // recurs: aggregate decode used to fall away past ×64 (2,227 t/s at
            // ×64 against 741 at ×128, and a 2.8× spread between two identical
            // ×128 runs), and that was attributed to the memory ceiling — VRAM
            // did sample at 99.3% of the card, which made the story fit. It was
            // not the ceiling. `to_owned_tensor` cloned a view's whole parent
            // storage, so taking the `[slots, vocab]` logits block off the wave
            // arena one row at a time cost the entire block per row: O(slots²)
            // bytes and one full-block allocation per row, which is also what put
            // VRAM at 99.3% and made the figure irreproducible. At ×128 it was
            // 881 ms of a 1,152 ms forward. Fixing it took ×128 from 286/957 t/s
            // to 3,812/3,846 and left decode layer-bound (`fwd:layer` is now 78%
            // of the forward, was 19%). See invariant 2 in AGENTS.md.
            //
            // Two claims died with it. There is no memory ceiling here at these
            // widths — KV is ~7.8 GB at ×128, not the 70 GB the sampled figure
            // suggested — and there is no per-width warm-up: the leading and
            // trailing ×256 rows now agree, where four identical ×128 rows once
            // read 244/741/729/734 and the first looked 3× slow. Both symptoms
            // were the same quadratic copy, seen cold and seen wide.
            //
            // The leading row is kept anyway, and stays the widest. It costs one
            // row and it is what would show a per-width warm-up returning —
            // `fused_attn_partial_pool` in `int8_decode_kernel.cuh` really does
            // `cudaStreamSynchronize` + `cudaFree` + `cudaMalloc` when its
            // process-wide high-water buffer grows, and this fixture decodes only
            // 9 steps per row, so any such one-off lands almost entirely on the
            // first measurement at a new width. Read the trailing three as the
            // scaling curve and compare the two ×256 rows to confirm the rung is
            // warm.
            for contexts in [256, 64, 128, 256] {
                configs.push(TestConfig {
                    mode: InferenceMode::C8,
                    use_batched: true,
                    num_contexts: contexts,
                    num_repeats: 1,
                    test_mode: Some(TestMode::StoryRewrite),
                });
            }
        } else {
            println!(
                "C8 wide rung: skipped — {:.1} GiB of VRAM is under the {:.0} GiB gate\n",
                gib(total_vram),
                gib(WIDE_C8_MIN_VRAM_BYTES),
            );
        }

        let load = || {
            let m = from_pack(&pack, &device)?;
            println!("✓ Model loaded\n");
            Ok(m)
        };
        params.run(configs, load)
    }

    /// **Depth on the 0.8B**: the batched forward at 32K and 128K of KV.
    ///
    /// The smallest member of the lineage, and the one where the KV cache
    /// dominates soonest relative to the weights — at 128K it is the larger
    /// resident object by a wide margin, which is exactly the regime the
    /// compression ladder exists for.
    #[test]
    #[ignore = "downloads the pinned Qwen3.5-0.8B GGUF and runs a 128K-token prompt. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen35::tests::long_context_0_8b \
                -- --ignored --nocapture --test-threads=1"]
    fn long_context_0_8b() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};

        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let pack = qwen35_0_8b(&device, Some(int8mode))?;
        long_context_gate(
            "Qwen3.5-0.8B (hybrid dense)",
            int8mode,
            &tokenizer_json()?,
            Dialect::qwen35(),
            // `qwen35.context_length` in the GGUF.
            262_144,
            &[
                (
                    32_768,
                    &[InferenceMode::BF16, InferenceMode::C5, InferenceMode::C10][..],
                ),
                (131_072, &[InferenceMode::BF16, InferenceMode::C10][..]),
            ],
            1,
            64,
            DepthTask::Coherence,
            &device,
            || from_pack(&pack, &device),
        )
    }
}
