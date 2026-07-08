//! Performance benchmark for the fused batched sampling kernel.
//!
//! Measures throughput across different configurations:
//! - Vocab sizes: 256, 1024, 4096, 32000, 50257, 128256
//! - Batch sizes: 1, 4, 16, 64, 256
//! - Modes: argmax, top-k=50, top-k=5 + top-p=0.9
//! - Dtypes: f32, f16, bf16
//!
//! Run with:
//!   cargo run --features cuda --release -p candle-core --example batched_sampling_benchmark

#[cfg(feature = "cuda")]
mod bench {

    use candle_core::cuda_backend::cudarc;
    use candle_kernels::sampling::run_batched_sampling;
    use core::ffi::c_void;
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
    use std::sync::Arc;
    use std::time::Instant;

    // ============================================================================
    // Configuration
    // ============================================================================

    const WARMUP_ITERS: usize = 20;
    const BENCH_ITERS: usize = 200;

    const VOCAB_SIZES: &[i32] = &[256, 1024, 4096, 32000, 50257, 128256];
    const BATCH_SIZES: &[i32] = &[1, 4, 16, 64, 256];

    struct BenchMode {
        name: &'static str,
        temperature: f32,
        top_k: i32,
        top_p: f32,
    }

    const MODES: &[BenchMode] = &[
        BenchMode {
            name: "argmax",
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        },
        BenchMode {
            name: "topk50",
            temperature: 0.8,
            top_k: 50,
            top_p: 1.0,
        },
        BenchMode {
            name: "topk5_topp0.9",
            temperature: 0.8,
            top_k: 5,
            top_p: 0.9,
        },
        BenchMode {
            name: "topk256",
            temperature: 0.8,
            top_k: 256,
            top_p: 1.0,
        },
    ];

    struct DTypeDef {
        name: &'static str,
        code: i32,
        bytes_per_elem: usize,
    }

    const DTYPES: &[DTypeDef] = &[
        DTypeDef {
            name: "f32",
            code: 0,
            bytes_per_elem: 4,
        },
        DTypeDef {
            name: "f16",
            code: 1,
            bytes_per_elem: 2,
        },
        DTypeDef {
            name: "bf16",
            code: 2,
            bytes_per_elem: 2,
        },
    ];

    // ============================================================================
    // Helpers
    // ============================================================================

    /// Generate random-ish logits on the host, upload as properly aligned GPU memory.
    /// Uses u32 backing for all types to guarantee 4-byte alignment.
    fn make_logits_gpu(
        stream: &Arc<CudaStream>,
        batch_size: i32,
        vocab_size: i32,
        dtype: &DTypeDef,
    ) -> CudaSlice<u32> {
        let n = (batch_size as usize) * (vocab_size as usize);
        // Generate f32 logits with a realistic distribution
        let f32_logits: Vec<f32> = (0..n)
            .map(|i| {
                // Mix of small and large values; ensure one clear peak per row
                let row = i / vocab_size as usize;
                let col = i % vocab_size as usize;
                let base = ((col as f32 * 0.01).sin() * 5.0) + ((row as f32 * 0.7).cos() * 2.0);
                if col == (row * 7 + 13) % vocab_size as usize {
                    base + 20.0 // spike
                } else {
                    base
                }
            })
            .collect();

        match dtype.code {
            0 => {
                // f32: reinterpret as u32
                let u32_data: Vec<u32> = f32_logits.iter().map(|v| v.to_bits()).collect();
                stream.memcpy_stod(&u32_data).expect("upload f32 logits")
            }
            1 => {
                // f16: pack two f16 values into each u32
                let f16_data: Vec<u16> = f32_logits
                    .iter()
                    .map(|v| half::f16::from_f32(*v).to_bits())
                    .collect();
                // Pad to even length if needed
                let mut padded = f16_data;
                if padded.len() % 2 != 0 {
                    padded.push(0);
                }
                let u32_data: Vec<u32> = padded
                    .chunks(2)
                    .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
                    .collect();
                stream.memcpy_stod(&u32_data).expect("upload f16 logits")
            }
            2 => {
                // bf16: pack two bf16 values into each u32
                let bf16_data: Vec<u16> = f32_logits
                    .iter()
                    .map(|v| half::bf16::from_f32(*v).to_bits())
                    .collect();
                let mut padded = bf16_data;
                if padded.len() % 2 != 0 {
                    padded.push(0);
                }
                let u32_data: Vec<u32> = padded
                    .chunks(2)
                    .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
                    .collect();
                stream.memcpy_stod(&u32_data).expect("upload bf16 logits")
            }
            _ => panic!("unknown dtype code"),
        }
    }

    struct BenchResult {
        dtype: &'static str,
        mode: &'static str,
        batch_size: i32,
        vocab_size: i32,
        median_us: f64,
        min_us: f64,
        max_us: f64,
        throughput_gbs: f64,  // GB/s of logit data read
        samples_per_sec: f64, // batch_size * (1e6/median_us)
    }

    fn run_bench(
        stream: &Arc<CudaStream>,
        logits_gpu: &CudaSlice<u32>,
        batch_size: i32,
        vocab_size: i32,
        dtype: &DTypeDef,
        mode: &BenchMode,
    ) -> BenchResult {
        let mut output_gpu: CudaSlice<u32> = stream
            .alloc_zeros(batch_size as usize)
            .expect("alloc output");
        let mut rng_offsets_gpu: CudaSlice<u64> =
            stream.alloc_zeros(batch_size as usize).expect("alloc rng");

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let (logits_ptr, _g1) = logits_gpu.device_ptr(stream);
            let (output_ptr, _g2) = output_gpu.device_ptr_mut(stream);
            let (rng_ptr, _g3) = rng_offsets_gpu.device_ptr_mut(stream);
            unsafe {
                run_batched_sampling(
                    logits_ptr as *const c_void,
                    batch_size,
                    vocab_size,
                    dtype.code,
                    mode.temperature,
                    mode.top_k,
                    mode.top_p,
                    1.0,
                    0.0,
                    0.0, // repeat/freq/presence penalties
                    0.0,
                    1.75,
                    2,
                    0, // DRY params (disabled)
                    0.0,
                    -1,               // eos_boost, eos_token_id
                    0,                // eos_ramp_start
                    0,                // eos_ramp_len
                    0.0,              // eos_boost_max_multiplier
                    0.0,              // cross_turn_penalty
                    std::ptr::null(), // cross_turn_counts
                    std::ptr::null(), // current_lens
                    0.0,              // segment_close_boost (disabled)
                    -1,               // segment_close_token_id (disabled)
                    0,                // segment_close_ramp_start (disabled)
                    0,                // segment_close_ramp_len (disabled)
                    0.0,              // segment_close_max_multiplier (disabled)
                    std::ptr::null(), // segment_lens
                    0.0,              // segment_temp_boost (disabled)
                    std::ptr::null(), // suppress_tokens (disabled)
                    0,                // suppress_count (disabled)
                    std::ptr::null(), // suppress_penalties (disabled)
                    std::ptr::null(), // token_counts
                    std::ptr::null(), // banned_tokens
                    0,
                    0,                // num_banned, banned_per_seq
                    std::ptr::null(), // recent_tokens
                    std::ptr::null(), // recent_lens
                    0,                // max_recent_len
                    std::ptr::null(), // stencil
                    0,                // stencil_size
                    output_ptr as *mut u32,
                    42, // seed
                    rng_ptr as *mut u64,
                );
            }
            stream.synchronize().expect("sync");
        }

        // Benchmark
        let mut timings = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            // Reset rng offsets each iteration for consistency
            let (logits_ptr, _g1) = logits_gpu.device_ptr(stream);
            let (output_ptr, _g2) = output_gpu.device_ptr_mut(stream);
            let (rng_ptr, _g3) = rng_offsets_gpu.device_ptr_mut(stream);

            stream.synchronize().expect("pre-sync");
            let t0 = Instant::now();
            unsafe {
                run_batched_sampling(
                    logits_ptr as *const c_void,
                    batch_size,
                    vocab_size,
                    dtype.code,
                    mode.temperature,
                    mode.top_k,
                    mode.top_p,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.75,
                    2,
                    0,
                    0.0,
                    -1,
                    0,                // eos_ramp_start
                    0,                // eos_ramp_len
                    0.0,              // eos_boost_max_multiplier
                    0.0,              // cross_turn_penalty
                    std::ptr::null(), // cross_turn_counts
                    std::ptr::null(), // current_lens
                    0.0,              // segment_close_boost (disabled)
                    -1,               // segment_close_token_id (disabled)
                    0,                // segment_close_ramp_start (disabled)
                    0,                // segment_close_ramp_len (disabled)
                    0.0,              // segment_close_max_multiplier (disabled)
                    std::ptr::null(), // segment_lens
                    0.0,              // segment_temp_boost (disabled)
                    std::ptr::null(), // suppress_tokens (disabled)
                    0,                // suppress_count (disabled)
                    std::ptr::null(), // suppress_penalties (disabled)
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    output_ptr as *mut u32,
                    42,
                    rng_ptr as *mut u64,
                );
            }
            stream.synchronize().expect("sync");
            let elapsed = t0.elapsed();
            timings.push(elapsed.as_nanos() as f64 / 1000.0); // microseconds
        }

        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_us = timings[timings.len() / 2];
        let min_us = timings[0];
        let max_us = timings[timings.len() - 1];

        let data_bytes = batch_size as f64 * vocab_size as f64 * dtype.bytes_per_elem as f64;
        let throughput_gbs = data_bytes / (median_us * 1e-6) / 1e9;
        let samples_per_sec = batch_size as f64 * 1e6 / median_us;

        BenchResult {
            dtype: dtype.name,
            mode: mode.name,
            batch_size,
            vocab_size,
            median_us,
            min_us,
            max_us,
            throughput_gbs,
            samples_per_sec,
        }
    }

    // ============================================================================
    // Penalty benchmark - measures overhead of enabling penalties
    // ============================================================================

    fn run_bench_with_penalties(
        stream: &Arc<CudaStream>,
        batch_size: i32,
        vocab_size: i32,
    ) -> (BenchResult, BenchResult) {
        let dtype = &DTYPES[0]; // f32
        let mode = &BenchMode {
            name: "topk50+penalties",
            temperature: 0.8,
            top_k: 50,
            top_p: 1.0,
        };
        let mode_base = &BenchMode {
            name: "topk50_base",
            temperature: 0.8,
            top_k: 50,
            top_p: 1.0,
        };

        let logits_gpu = make_logits_gpu(stream, batch_size, vocab_size, dtype);

        // Without penalties
        let base = run_bench(
            stream,
            &logits_gpu,
            batch_size,
            vocab_size,
            dtype,
            mode_base,
        );

        // With penalties: need token_counts, recent_tokens, banned_tokens
        let n = batch_size as usize * vocab_size as usize;
        let token_counts: Vec<i32> = (0..n).map(|i| if i % 100 == 0 { 3 } else { 0 }).collect();
        let max_recent = 64;
        let recent_tokens: Vec<i32> = (0..batch_size as usize * max_recent)
            .map(|i| (i % vocab_size as usize) as i32)
            .collect();
        let recent_lens: Vec<i32> = vec![max_recent as i32; batch_size as usize];
        let banned_per_seq = 16;
        let banned_tokens: Vec<i32> = (0..batch_size as usize * banned_per_seq)
            .map(|i| (i * 7 % vocab_size as usize) as i32)
            .collect();

        let tc_gpu = stream.memcpy_stod(&token_counts).expect("upload tc");
        let recent_gpu = stream.memcpy_stod(&recent_tokens).expect("upload recent");
        let rl_gpu = stream.memcpy_stod(&recent_lens).expect("upload rl");
        let ban_gpu = stream.memcpy_stod(&banned_tokens).expect("upload ban");

        let mut output_gpu: CudaSlice<u32> =
            stream.alloc_zeros(batch_size as usize).expect("alloc");
        let mut rng_offsets_gpu: CudaSlice<u64> =
            stream.alloc_zeros(batch_size as usize).expect("alloc");

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let (lp, _g1) = logits_gpu.device_ptr(stream);
            let (op, _g2) = output_gpu.device_ptr_mut(stream);
            let (rp, _g3) = rng_offsets_gpu.device_ptr_mut(stream);
            let (tcp, _g4) = tc_gpu.device_ptr(stream);
            let (recp, _g5) = recent_gpu.device_ptr(stream);
            let (rlp, _g6) = rl_gpu.device_ptr(stream);
            let (banp, _g7) = ban_gpu.device_ptr(stream);
            unsafe {
                run_batched_sampling(
                    lp as *const c_void,
                    batch_size,
                    vocab_size,
                    dtype.code,
                    mode.temperature,
                    mode.top_k,
                    mode.top_p,
                    1.2,
                    0.5,
                    0.3,
                    0.0,
                    1.75,
                    2,
                    0,
                    0.0,
                    -1,
                    0,                // eos_ramp_start
                    0,                // eos_ramp_len
                    0.0,              // eos_boost_max_multiplier
                    0.0,              // cross_turn_penalty
                    std::ptr::null(), // cross_turn_counts
                    std::ptr::null(), // current_lens
                    0.0,              // segment_close_boost (disabled)
                    -1,               // segment_close_token_id (disabled)
                    0,                // segment_close_ramp_start (disabled)
                    0,                // segment_close_ramp_len (disabled)
                    0.0,              // segment_close_max_multiplier (disabled)
                    std::ptr::null(), // segment_lens
                    0.0,              // segment_temp_boost (disabled)
                    std::ptr::null(), // suppress_tokens (disabled)
                    0,                // suppress_count (disabled)
                    std::ptr::null(), // suppress_penalties (disabled)
                    tcp as *const i32,
                    banp as *const i32,
                    (banned_per_seq * batch_size as usize) as i32,
                    banned_per_seq as i32,
                    recp as *const i32,
                    rlp as *const i32,
                    max_recent as i32,
                    std::ptr::null(),
                    0,
                    op as *mut u32,
                    42,
                    rp as *mut u64,
                );
            }
            stream.synchronize().expect("sync");
        }

        // Bench
        let mut timings = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let (lp, _g1) = logits_gpu.device_ptr(stream);
            let (op, _g2) = output_gpu.device_ptr_mut(stream);
            let (rp, _g3) = rng_offsets_gpu.device_ptr_mut(stream);
            let (tcp, _g4) = tc_gpu.device_ptr(stream);
            let (recp, _g5) = recent_gpu.device_ptr(stream);
            let (rlp, _g6) = rl_gpu.device_ptr(stream);
            let (banp, _g7) = ban_gpu.device_ptr(stream);

            stream.synchronize().expect("pre-sync");
            let t0 = Instant::now();
            unsafe {
                run_batched_sampling(
                    lp as *const c_void,
                    batch_size,
                    vocab_size,
                    dtype.code,
                    mode.temperature,
                    mode.top_k,
                    mode.top_p,
                    1.2,
                    0.5,
                    0.3,
                    0.0,
                    1.75,
                    2,
                    0,
                    0.0,
                    -1,
                    0,                // eos_ramp_start
                    0,                // eos_ramp_len
                    0.0,              // eos_boost_max_multiplier
                    0.0,              // cross_turn_penalty
                    std::ptr::null(), // cross_turn_counts
                    std::ptr::null(), // current_lens
                    0.0,              // segment_close_boost (disabled)
                    -1,               // segment_close_token_id (disabled)
                    0,                // segment_close_ramp_start (disabled)
                    0,                // segment_close_ramp_len (disabled)
                    0.0,              // segment_close_max_multiplier (disabled)
                    std::ptr::null(), // segment_lens
                    0.0,              // segment_temp_boost (disabled)
                    std::ptr::null(), // suppress_tokens (disabled)
                    0,                // suppress_count (disabled)
                    std::ptr::null(), // suppress_penalties (disabled)
                    tcp as *const i32,
                    banp as *const i32,
                    (banned_per_seq * batch_size as usize) as i32,
                    banned_per_seq as i32,
                    recp as *const i32,
                    rlp as *const i32,
                    max_recent as i32,
                    std::ptr::null(),
                    0,
                    op as *mut u32,
                    42,
                    rp as *mut u64,
                );
            }
            stream.synchronize().expect("sync");
            timings.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }

        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_us = timings[timings.len() / 2];
        let min_us = timings[0];
        let max_us = timings[timings.len() - 1];
        let data_bytes = batch_size as f64 * vocab_size as f64 * dtype.bytes_per_elem as f64;
        let throughput_gbs = data_bytes / (median_us * 1e-6) / 1e9;
        let samples_per_sec = batch_size as f64 * 1e6 / median_us;

        let with_pen = BenchResult {
            dtype: "f32",
            mode: mode.name,
            batch_size,
            vocab_size,
            median_us,
            min_us,
            max_us,
            throughput_gbs,
            samples_per_sec,
        };

        (base, with_pen)
    }

    // ============================================================================
    // Main
    // ============================================================================

    pub fn main() {
        println!("╔══════════════════════════════════════════════════════════════════════╗");
        println!("║       Fused Batched Sampling Kernel — Performance Benchmark        ║");
        println!("╚══════════════════════════════════════════════════════════════════════╝");
        println!();

        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();

        // Print GPU info
        let dev_name = ctx.name().unwrap_or_else(|_| "Unknown".to_string());
        println!("GPU: {}", dev_name);
        println!("Warmup iterations: {}", WARMUP_ITERS);
        println!("Benchmark iterations: {}", BENCH_ITERS);
        println!();

        // ========================================================================
        // Part 1: Core throughput sweep
        // ========================================================================
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│  Part 1: Core Throughput Sweep (no penalties)                    │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
        println!(
            "{:<6} {:<16} {:<6} {:<8} {:>10} {:>10} {:>10} {:>10} {:>14}",
            "dtype", "mode", "batch", "vocab", "median_us", "min_us", "max_us", "GB/s", "samples/s"
        );
        println!("{}", "─".repeat(100));

        let mut all_results: Vec<BenchResult> = Vec::new();

        for dtype in DTYPES {
            for mode in MODES {
                for &batch_size in BATCH_SIZES {
                    for &vocab_size in VOCAB_SIZES {
                        let logits_gpu = make_logits_gpu(&stream, batch_size, vocab_size, dtype);
                        let result =
                            run_bench(&stream, &logits_gpu, batch_size, vocab_size, dtype, mode);
                        println!(
                            "{:<6} {:<16} {:<6} {:<8} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>14.0}",
                            result.dtype,
                            result.mode,
                            result.batch_size,
                            result.vocab_size,
                            result.median_us,
                            result.min_us,
                            result.max_us,
                            result.throughput_gbs,
                            result.samples_per_sec,
                        );
                        all_results.push(result);
                    }
                }
            }
        }

        // ========================================================================
        // Part 2: Penalty overhead
        // ========================================================================
        println!();
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│  Part 2: Penalty Overhead (f32, top-k=50, repeat+freq+presence) │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();
        println!(
            "{:<8} {:<6} {:>10} {:>10} {:>10} {:>8}",
            "vocab", "batch", "base_us", "pen_us", "delta_us", "overhead%"
        );
        println!("{}", "─".repeat(62));

        for &vocab_size in &[32000i32, 128256] {
            for &batch_size in &[1i32, 16, 64] {
                let (base, with_pen) = run_bench_with_penalties(&stream, batch_size, vocab_size);
                let delta = with_pen.median_us - base.median_us;
                let overhead_pct = (delta / base.median_us) * 100.0;
                println!(
                    "{:<8} {:<6} {:>10.1} {:>10.1} {:>10.1} {:>7.1}%",
                    vocab_size, batch_size, base.median_us, with_pen.median_us, delta, overhead_pct,
                );
            }
        }

        // ========================================================================
        // Part 3: Latency at realistic inference configs
        // ========================================================================
        println!();
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│  Part 3: Realistic Inference Latency (single-sequence decode)    │");
        println!("└──────────────────────────────────────────────────────────────────┘");
        println!();

        let realistic_configs: &[(&str, i32, i32, i32, f32, f32)] = &[
            // (model, vocab, batch, topk, temp, topp)
            ("GPT-2", 50257, 1, 50, 0.8, 1.0),
            ("LLaMA-2", 32000, 1, 40, 0.6, 0.9),
            ("LLaMA-3", 128256, 1, 50, 0.7, 0.9),
            ("Qwen-2.5", 152064, 1, 50, 0.8, 0.95),
            ("LLaMA-3 b4", 128256, 4, 50, 0.7, 0.9),
            ("LLaMA-3 b16", 128256, 16, 50, 0.7, 0.9),
            ("LLaMA-3 b64", 128256, 64, 50, 0.7, 0.9),
        ];

        println!(
            "{:<16} {:<8} {:<6} {:>10} {:>12} {:>14}",
            "model", "vocab", "batch", "median_us", "GB/s", "samples/s"
        );
        println!("{}", "─".repeat(74));

        let dtype = &DTYPES[0]; // f32
        for &(model, vocab_size, batch_size, top_k, temp, top_p) in realistic_configs {
            let mode = BenchMode {
                name: model,
                temperature: temp,
                top_k,
                top_p,
            };
            let logits_gpu = make_logits_gpu(&stream, batch_size, vocab_size, dtype);
            let result = run_bench(&stream, &logits_gpu, batch_size, vocab_size, dtype, &mode);
            println!(
                "{:<16} {:<8} {:<6} {:>10.1} {:>12.1} {:>14.0}",
                model,
                vocab_size,
                batch_size,
                result.median_us,
                result.throughput_gbs,
                result.samples_per_sec,
            );
        }

        // ========================================================================
        // Summary
        // ========================================================================
        println!();
        println!("┌──────────────────────────────────────────────────────────────────┐");
        println!("│  Summary                                                         │");
        println!("└──────────────────────────────────────────────────────────────────┘");

        // Find the result for the common LLaMA-3 config
        let llama3_results: Vec<&BenchResult> = all_results
            .iter()
            .filter(|r| r.vocab_size == 128256 && r.batch_size == 1 && r.dtype == "f32")
            .collect();
        if let Some(argmax) = llama3_results.iter().find(|r| r.mode == "argmax") {
            println!(
                "  LLaMA-3 (128K vocab) single-sequence argmax:  {:.1} μs",
                argmax.median_us
            );
        }
        if let Some(topk) = llama3_results.iter().find(|r| r.mode == "topk50") {
            println!(
                "  LLaMA-3 (128K vocab) single-sequence top-k=50: {:.1} μs",
                topk.median_us
            );
        }

        let peak_throughput = all_results
            .iter()
            .max_by(|a, b| a.throughput_gbs.partial_cmp(&b.throughput_gbs).unwrap());
        if let Some(peak) = peak_throughput {
            println!(
                "  Peak memory throughput: {:.1} GB/s ({} {} batch={} vocab={})",
                peak.throughput_gbs, peak.dtype, peak.mode, peak.batch_size, peak.vocab_size
            );
        }

        let peak_samples = all_results
            .iter()
            .max_by(|a, b| a.samples_per_sec.partial_cmp(&b.samples_per_sec).unwrap());
        if let Some(peak) = peak_samples {
            println!(
                "  Peak sample throughput: {:.0} samples/s ({} {} batch={} vocab={})",
                peak.samples_per_sec, peak.dtype, peak.mode, peak.batch_size, peak.vocab_size
            );
        }

        println!();
        println!("Done.");
    }
} // mod bench

#[cfg(feature = "cuda")]
fn main() {
    bench::main();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This benchmark requires the `cuda` feature. Run with:");
    eprintln!(
        "  cargo run --features cuda --release -p candle-core --example batched_sampling_benchmark"
    );
}
