//! Profiling runs that measure how the engine's rates move with depth, width and
//! task.
//!
//! These runs report throughput by depth and by width, and fill the span tables
//! when built with profiling. Each one holds a variable fixed and varies another,
//! because the depth ladder and the width ladder differ in task and in whether
//! selection engages, so their rates cannot be compared directly. The story-rewrite
//! sweep holds the task fixed and varies only depth. The host-issue run times the
//! layer loop against the step's wall time with GPU spans off, so the step runs as
//! a release build runs it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::batch_test::utils::{TestConfig, TestMode};

    use super::super::tests_support::tests::{engine, tokenizer_json};

    /// **Depth**: the batched forward at 32K and 128K of KV.
    ///
    /// The ladder above reports this model at ~700 tokens, which is where its
    /// per-token cost is cleanest but says nothing about the property the
    /// engine is built for. This runs the same forward with a prompt deep
    /// enough to make the KV cache the dominant resident object, at three
    /// points on the compression ladder, so prefill rate, decode rate and
    /// compression can be read against depth rather than against width.
    #[test]
    #[ignore = "loads this card's engine artifact and runs a 128K-token prompt. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::test_long_context_scaling \
                -- --ignored --nocapture --test-threads=1"]
    fn test_long_context_scaling() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        long_context_gate(
            "Qwen3.8-Flash-Next (qwen4exp, 250B-A13B)",
            int8mode,
            &tok,
            Dialect::qwen35(),
            // `qwen4exp.context_length` in the merged engine GGUF.
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
            || {
                let gpu = Qwen4ExpGpu::load(&pack, &device)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// **Where does decode time go as the cache grows?**
    ///
    /// `test_long_context_scaling` measures two depths, which is enough to say
    /// that decode falls faster than it should (10.2 → 1.2 t/s across 4×) and
    /// not enough to say *which* part of the step is responsible. This walks a
    /// depth ladder at one compression rung so every span's cost can be read
    /// against depth as a curve: a term that is constant in depth stays flat, a
    /// term linear in depth doubles with each rung, and the one that does is
    /// the wall.
    ///
    /// BF16 only and a short generate, because the question is the *shape* of
    /// the decode step, not its absolute rate — and the prefill needed to reach
    /// 128K already dominates the run's wall clock.
    ///
    /// Run under `--features cuda,profile`, which is what fills the pipeline
    /// tables; without it the harness still reports throughput and the spans
    /// are inert.
    #[test]
    #[ignore = "profiling run: loads this card's engine artifact and prefills to 128K. \
                Run with: cargo test --release --features cuda,profile \
                -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_decode_vs_depth \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_decode_vs_depth() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let bf16 = &[InferenceMode::BF16][..];
        long_context_gate(
            "Qwen3.8-Flash-Next decode profile",
            int8mode,
            &tok,
            Dialect::qwen35(),
            262_144,
            &[
                (8_192, bf16),
                (16_384, bf16),
                (32_768, bf16),
                (65_536, bf16),
                (131_072, bf16),
            ],
            1,
            32,
            DepthTask::Coherence,
            &device,
            || {
                let gpu = Qwen4ExpGpu::load(&pack, &device)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// One session writing freely — the drafter guessing rather than copying —
    /// at a shallow depth and long enough for its draft depth to settle: the
    /// counterpart of [`profile_single_session_decode`]'s rewrite, and what the
    /// draft-depth rule (`draft_depth`) is priced against on text it predicts
    /// poorly.
    #[test]
    #[ignore = "profiling run: loads this card's engine artifact. Run with: cargo test \
                --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_single_session_essay \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_single_session_essay() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let bf16 = &[InferenceMode::BF16][..];
        long_context_gate(
            "Qwen3.8-Flash-Next single-session essay",
            int8mode,
            &tok,
            Dialect::qwen35(),
            262_144,
            &[(4_096, bf16), (4_096, bf16), (4_096, bf16)],
            1,
            256,
            DepthTask::Essay,
            &device,
            || {
                let gpu = Qwen4ExpGpu::load(&pack, &device)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// Eight sessions writing freely near zend's working shape: a 32K-token
    /// context each, C5 KV — the cohort a code-reading ingest decodes (zend's run
    /// at 41–66K; 8 × 64K fully resident does not fit the harness, which has no
    /// warm tier). Against the forward gate's eight sessions at a few thousand
    /// tokens, this is the row that says what depth costs an eight-wide
    /// speculative decode.
    #[test]
    #[ignore = "profiling run: loads this card's engine artifact and prefills 8 x 32K. Run \
                with: cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_eight_sessions_at_32k \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_eight_sessions_at_32k() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let c5 = &[InferenceMode::C5][..];
        // The same cohort at 8K in both KV formats first: what C5's compressed
        // reads cost against BF16 at a depth where both fit, then zend's format
        // at depth.
        let both = &[InferenceMode::BF16, InferenceMode::C5][..];
        long_context_gate(
            "Qwen3.8-Flash-Next eight sessions at 32K",
            int8mode,
            &tok,
            Dialect::qwen35(),
            262_144,
            &[(8_192, both), (32_768, c5)],
            8,
            256,
            DepthTask::Essay,
            &device,
            || {
                let gpu = Qwen4ExpGpu::load(&pack, &device)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// **Strata's single-session benchmark, on this engine.** The requests of
    /// Strata's published RTX 5090 run (`batch_test::strata_bench`): a synthetic
    /// Python module cut to 4,096 / 32,768 / 128,000 prompt tokens and a request
    /// to explain it, greedy, 256 tokens, one warm-up then three runs a length,
    /// reported as the median beside Strata's 179.4 / 175.7 / 165.0 t/s.
    #[test]
    #[ignore = "benchmark: loads this card's engine artifact and prefills to 128K. Run with: \
                cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::strata_bench_single_session \
                -- --ignored --nocapture --test-threads=1"]
    fn strata_bench_single_session() -> Result<()> {
        use crate::models::batch_test::strata_bench::strata_bench;
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        strata_bench(
            "Qwen3.8-Flash-Next",
            int8mode,
            &tokenizer_json()?,
            Dialect::qwen35(),
            InferenceMode::BF16,
            &device,
            || {
                let gpu = Qwen4ExpGpu::load(&pack, &device)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// **Output fidelity against depth on Flash-Next** (`batch_test::fidelity`):
    /// C0, C5 and C10 against BF16, teacher-forced on the needle answer, from
    /// 4K to 128K. Its full-attention layers select a bounded set through the
    /// checkpoint's QSA indexer, so the stored tokens reaching a step do not
    /// grow with depth past the indexer's budget.
    #[test]
    #[ignore = "loads this card's engine artifact and prefills to 128K under four modes. \
                Run with: cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::fidelity_vs_depth_flash_next \
                -- --ignored --nocapture --test-threads=1"]
    fn fidelity_vs_depth_flash_next() -> Result<()> {
        use crate::models::batch_test::fidelity::{assert_sound, ladder, print_table};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use tokenizers::Tokenizer;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let tok = Tokenizer::from_bytes(tokenizer_json()?.as_bytes())
            .map_err(|e| candle::Error::Msg(format!("tokenizer: {e}")))?;
        let model = Qwen4ExpBatched::new(Qwen4ExpGpu::load(&pack, &device)?)?;
        let modes = [InferenceMode::C0, InferenceMode::C5, InferenceMode::C10];
        let depths = [4_096, 16_384, 65_536, 131_072 - 4_096];
        let rows = ladder(&model, &tok, InferenceMode::BF16, &modes, &depths)?;
        print_table("Qwen3.8-Flash-Next (QSA-bounded attention)", &rows);
        assert_sound("Qwen3.8-Flash-Next", &rows, InferenceMode::BF16);
        Ok(())
    }

    /// The width ladder's task, run at the depth ladder's depths.
    ///
    /// [`profile_decode_vs_depth`] and [`test_parallel_batched_forwarding`]
    /// differ in THREE ways at once — depth, task, and whether QSA engages —
    /// so their decode rates cannot be read against each other, and the ~80 t/s
    /// of the width ladder against the ~29 t/s of the depth sweep is mostly the
    /// draft head accepting 4.85 tokens a step instead of 2.58 on a rewrite it
    /// can largely copy.
    ///
    /// This row holds the task fixed and varies only depth: the padding goes in
    /// FRONT of the same story the ladder rewrites, so acceptance stays in the
    /// ladder's regime while the prefix grows to 128K. It is the honest answer
    /// to "what does conversational-quality decode cost at depth", which
    /// neither of the other two tables gives on its own.
    #[test]
    #[ignore = "profiling run: loads this card's engine artifact and prefills to 128K. \
                Run with: cargo test --release --features cuda -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_story_rewrite_vs_depth \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_story_rewrite_vs_depth() -> Result<()> {
        use crate::models::batch_test::long_context::{long_context_gate, DepthTask};
        use crate::models::batched_inference::InferenceMode;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let tok = tokenizer_json()?;
        let bf16 = &[InferenceMode::BF16][..];
        long_context_gate(
            "Qwen3.8-Flash-Next story rewrite",
            int8mode,
            &tok,
            Dialect::qwen35(),
            262_144,
            &[
                (8_192, bf16),
                (16_384, bf16),
                (32_768, bf16),
                (65_536, bf16),
                (131_072, bf16),
            ],
            1,
            // The ladder's generation length, so the speculative loop runs the
            // same number of steps per row as the table this is comparable to.
            64,
            DepthTask::Rewrite,
            &device,
            || {
                let gpu = Qwen4ExpGpu::load(&pack, &device)?;
                Qwen4ExpBatched::new(gpu)
            },
        )
    }

    /// **Single-session decode, alone.** The gate's one-context row — same task,
    /// same 256 tokens, same dialect — run once cold and three times warm in one
    /// load, so the per-step cost of one conversation can be measured and
    /// profiled without the rest of the ladder in front of it.
    ///
    /// Under `--features cuda,profile` the harness fills its span tables; without
    /// it, the throughput table alone.
    #[test]
    #[ignore = "profiling run: loads this card's engine artifact. Run with: cargo test \
                --release --features cuda,profile -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_single_session_decode \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_single_session_decode() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::dialect::Dialect;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(3600);
        let one = || TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        };
        let configs = vec![one(), one(), one(), one()];
        params.run(configs, || {
            let gpu = Qwen4ExpGpu::load(&pack, &device)?;
            Qwen4ExpBatched::new(gpu)
        })
    }

    /// **The host side of a decode step, alone.** The same four runs as
    /// [`profile_single_session_decode`] with the GPU event spans sampled on
    /// no forward at all (`set_gpu_span_period(u32::MAX)`), so the step runs as
    /// the release build runs it — an event pair per span per layer is itself
    /// most of the profiled build's slowdown — while the host timers
    /// (`q4e:host:*`, `moe:hold_ring`, `spec:*`) still fill. Read
    /// `q4e:host:layers` per step against the step's wall time: a layer loop
    /// that takes the whole step is the step's bound, unless the hold on the
    /// summary ring (`moe:hold_ring`) is where it spends that time, which is
    /// the forward thread waiting for the GPU.
    #[test]
    #[ignore = "profiling run: loads this card's engine artifact. Run with: cargo test \
                --release --features cuda,profile -p candle-transformers --lib \
                quantized_qwen38_moe::tests::profile_single_session_decode_host_issue \
                -- --ignored --nocapture --test-threads=1"]
    fn profile_single_session_decode_host_issue() -> Result<()> {
        use crate::models::batch_test::utils::TestParams;
        use crate::models::dialect::Dialect;
        use crate::models::profile::set_gpu_span_period;
        use crate::models::qwen4exp::{Qwen4ExpBatched, Qwen4ExpGpu};
        use candle::quantized::Int8Mode;

        set_gpu_span_period(u32::MAX);
        let pack = engine()?;
        let device = Device::new_cuda(0)?;
        let int8mode = Int8Mode::auto(&device);
        let params = TestParams::new(256, &tokenizer_json()?, Dialect::qwen35())
            .map_err(|e| candle::Error::Msg(format!("TestParams: {e}")))?
            .with_suppress_thinking(true)
            .with_int8mode(int8mode)
            .with_timeout_secs(3600);
        let one = || TestConfig {
            mode: InferenceMode::BF16,
            use_batched: true,
            num_contexts: 1,
            num_repeats: 1,
            test_mode: Some(TestMode::StoryRewrite),
        };
        let configs = vec![one(), one(), one(), one()];
        params.run(configs, || {
            let gpu = Qwen4ExpGpu::load(&pack, &device)?;
            Qwen4ExpBatched::new(gpu)
        })
    }
}
