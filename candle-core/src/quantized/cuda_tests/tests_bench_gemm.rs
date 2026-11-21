//! Throughput benches for the quantize kernels and the grouped and dense int8
//! matmuls.
//!
//! The benches run at the shapes the models use, the DeepSeek-V4-Flash and
//! Qwen3 expert layers among them, so a change to a kernel shows as a figure in
//! the units the hardware is measured in. Each bench allocates its buffers
//! before timing and times the launches alone, so the number is kernel time and
//! not allocation. The statistical Q4 bench reports a confidence interval rather
//! than a best-of, because the FP16 against int8 parity question turns on
//! sub-percent differences.

use super::*;

/// Throughput bench for the q8a128 quantize/dequant kernels — they sit on the hot
/// inference path so they must run at ~peak memory bandwidth. Pre-allocates all
/// buffers and calls the kernels directly (no per-iter alloc), timing both the
/// f32 and bf16 vectorization paths. Reports GB/s (effective 4090M peak ≈ 450).
#[test]
#[ignore = "GPU throughput bench; run with --ignored --nocapture"]
pub(super) fn q8a128_throughput_bench() -> Result<()> {
    use candle_kernels::simple::quantized::{run_dequantize_q8a128, run_quantize_q8a128};
    use std::ffi::c_void;
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let rows = 8192usize;
    let cols = 4096usize; // [tokens, hidden]
    let n = rows * cols;
    let qbytes = (n / 128).div_ceil(8) * 1152;
    let (rows_i, cols_i) = (rows as i32, cols as i32);
    let warm = 10usize;
    let iters = 100usize;

    let f32_in: Vec<f32> = (0..n).map(|i| ((i % 257) as f32 - 128.0) * 0.01).collect();
    let f16_in: Vec<half::f16> = f32_in.iter().map(|&x| half::f16::from_f32(x)).collect();
    let bf16_in: Vec<half::bf16> = f32_in.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let f32_dev = dev.memcpy_stod(&f32_in)?;
    let f16_dev = dev.memcpy_stod(&f16_in)?;
    let bf16_dev = dev.memcpy_stod(&bf16_in)?;
    let mut qbuf = unsafe { dev.alloc::<u8>(qbytes)? };
    let mut dout_f32 = unsafe { dev.alloc::<f32>(n)? };
    let mut dout_f16 = unsafe { dev.alloc::<half::f16>(n)? };
    let mut dout_bf16 = unsafe { dev.alloc::<half::bf16>(n)? };
    let stream = dev.cuda_stream();
    let (f32p, _g0) = f32_dev.device_ptr(&stream);
    let (f16p, _g0b) = f16_dev.device_ptr(&stream);
    let (bf16p, _g1) = bf16_dev.device_ptr(&stream);
    let (qp, _g2) = qbuf.device_ptr_mut(&stream);
    let (dfp, _g3) = dout_f32.device_ptr_mut(&stream);
    let (dhp, _g3b) = dout_f16.device_ptr_mut(&stream);
    let (dbp, _g4) = dout_bf16.device_ptr_mut(&stream);

    // (name, input ptr, dtype code [0=f16,1=bf16,2=f32], element bytes, dequant out ptr)
    let configs: [(&str, u64, i32, usize, u64); 3] = [
        ("f32", f32p, 2, 4, dfp),
        ("f16", f16p, 0, 2, dhp),
        ("bf16", bf16p, 1, 2, dbp),
    ];

    for &(name, inp, dtype, elem_bytes, dout) in &configs {
        for _ in 0..warm {
            unsafe {
                run_quantize_q8a128(
                    inp as *const c_void,
                    qp as *mut c_void,
                    rows_i,
                    cols_i,
                    dtype,
                    SumScale::Raw.as_code(),
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                run_quantize_q8a128(
                    inp as *const c_void,
                    qp as *mut c_void,
                    rows_i,
                    cols_i,
                    dtype,
                    SumScale::Raw.as_code(),
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        dev.synchronize()?;
        let qt = t0.elapsed().as_secs_f64() / iters as f64;
        let q_gbps = (n * elem_bytes + qbytes) as f64 / qt / 1e9;

        for _ in 0..warm {
            unsafe {
                run_dequantize_q8a128(
                    qp as *const c_void,
                    dout as *mut c_void,
                    rows_i,
                    cols_i,
                    dtype,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        dev.synchronize()?;
        let t1 = Instant::now();
        for _ in 0..iters {
            unsafe {
                run_dequantize_q8a128(
                    qp as *const c_void,
                    dout as *mut c_void,
                    rows_i,
                    cols_i,
                    dtype,
                    stream.cu_stream() as *mut c_void,
                );
            }
        }
        dev.synchronize()?;
        let dt = t1.elapsed().as_secs_f64() / iters as f64;
        let d_gbps = (qbytes + n * elem_bytes) as f64 / dt / 1e9;

        println!(
            "q8a128 [{rows}x{cols} {name}]  quantize {:.4} ms {:6.1} GB/s  |  dequant {:.4} ms {:6.1} GB/s",
            qt * 1e3, q_gbps, dt * 1e3, d_gbps
        );
    }
    Ok(())
}

/// GPU throughput for the KO weight quantize/dequant kernels (F32 ↔ lane-major per-128
/// KO). Pre-uploads the weight matrix and calls the kernels directly (no per-iter alloc).
/// GB/s is on the f32 side (the bandwidth-relevant traffic) + the on-disk compression vs f32.
/// Run with: cargo test -p candle-core --features cuda ko_quant_throughput_bench -- --nocapture
#[test]
#[ignore = "GPU throughput bench; run with --ignored --nocapture"]
pub(super) fn ko_quant_throughput_bench() -> Result<()> {
    use crate::quantized::ko_quant::ko_chunk_bytes;
    use candle_kernels::simple::quantized::{run_dequantize_ko, run_quantize_ko};
    let dev = CudaDevice::new(0)?;
    let (nrows, ncols) = (8192usize, 8192usize); // 67.1M weights
    let n = nrows * ncols;
    let f32_bytes = n * 4;
    let (nr, nc) = (nrows as i32, ncols as i32);
    let (warm, iters) = (5usize, 50usize);

    let w: Vec<f32> = (0..n).map(|i| ((i % 257) as f32 - 128.0) * 0.01).collect();
    let w_dev = dev.memcpy_stod(&w)?;
    let stream = dev.cuda_stream();
    let (wp, _g0) = w_dev.device_ptr(&stream);

    for dtype in [
        GgmlDType::Q4_KO,
        GgmlDType::Q5_KO,
        GgmlDType::Q6_KO,
        GgmlDType::Q8_KO,
    ] {
        let qtype = dtype_to_qtype(dtype)? as i32;
        let qbytes = (nrows / 8) * (ncols / 128) * ko_chunk_bytes(dtype);
        let mut qbuf = unsafe { dev.alloc::<u8>(qbytes)? };
        let mut dbuf = unsafe { dev.alloc::<f32>(n)? };
        let (qp, _g1) = qbuf.device_ptr_mut(&stream);
        let (dp, _g2) = dbuf.device_ptr_mut(&stream);

        for _ in 0..warm {
            unsafe {
                run_quantize_ko(
                    wp as *const f32,
                    qp as *mut c_void,
                    nr,
                    nc,
                    qtype,
                    stream.cu_stream() as *mut c_void,
                )
            };
        }
        dev.synchronize()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                run_quantize_ko(
                    wp as *const f32,
                    qp as *mut c_void,
                    nr,
                    nc,
                    qtype,
                    stream.cu_stream() as *mut c_void,
                )
            };
        }
        dev.synchronize()?;
        let qt = t0.elapsed().as_secs_f64() / iters as f64;
        let q_gbps = (f32_bytes + qbytes) as f64 / qt / 1e9;

        for _ in 0..warm {
            unsafe {
                run_dequantize_ko(
                    qp as *const c_void,
                    dp as *mut f32,
                    nr,
                    nc,
                    qtype,
                    stream.cu_stream() as *mut c_void,
                )
            };
        }
        dev.synchronize()?;
        let t1 = Instant::now();
        for _ in 0..iters {
            unsafe {
                run_dequantize_ko(
                    qp as *const c_void,
                    dp as *mut f32,
                    nr,
                    nc,
                    qtype,
                    stream.cu_stream() as *mut c_void,
                )
            };
        }
        dev.synchronize()?;
        let dt = t1.elapsed().as_secs_f64() / iters as f64;
        let d_gbps = (qbytes + f32_bytes) as f64 / dt / 1e9;

        println!(
            "KO [{nrows}x{ncols} {dtype:?}]  quantize {:.4} ms {:6.1} GB/s  |  dequant {:.4} ms {:6.1} GB/s  |  {:.2}x vs f32",
            qt * 1e3, q_gbps, dt * 1e3, d_gbps, f32_bytes as f64 / qbytes as f64,
        );
    }
    Ok(())
}

/// Single-layer MoE grouped-GEMM replay bench at the DeepSeek-V4-Flash shapes —
/// the fast iteration loop for routed-GEMM kernel work (256 experts, gate/up
/// `[2048, 7168]`, down `[7168, 2048]`, MXFP4_KO, top-6 routing). Weight bytes
/// are random (timing is value-independent); 32 distinct weights per projection
/// cycle through the expert pointer table so the working set exceeds L2 like
/// the real 256-expert layer. Two regimes: a cfg8 user-turn wave (~3.9k tokens
/// × 6 ≈ 23k gathered rows, ~91/expert) and a width-capped cfg20 slice (8192 ×
/// 6 ≈ 49k rows, ~192/expert). Prints ms/call + effective weight bandwidth per
/// tile mode. nsys baseline (2026-08-17): m8 ≈ 9.3 ms/call ≈ 15% of DRAM peak —
/// latency-bound on the depth-1 weight ring.
#[test]
#[ignore = "GPU perf bench; run with --ignored --nocapture"]
pub(super) fn moe_layer_gemm_bench() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let mut rng = rand::rng();
    let n_experts = 256usize;
    let n_weights = 32usize;
    // (label, nrows=N, ncols=K)
    let shapes = [("gate/up", 2048usize, 7168usize), ("down", 7168, 2048)];
    // decode = 64 sessions × top-6 ≈ 2 rows/expert: the small-activation band where
    // the grid-order choice could flip (the whole activation fits L2 trivially).
    let regimes = [
        ("decode", 64usize * 6),
        ("cfg8", 3900 * 6),
        ("cfg20", 8192 * 6),
    ];

    for &(label, nrows, ncols) in &shapes {
        // Random-byte KO weights: [K/128 chunks] × [N/8 row-groups] × 544B.
        let wbytes = (ncols / 128) * (nrows / 8) * crate::quantized::ko_quant::MXFP4_KO_CHUNK_BYTES;
        let mut ptrs_pool: Vec<u64> = Vec::new();
        let mut _keep = Vec::new();
        for _ in 0..n_weights {
            let bytes: Vec<u8> = (0..wbytes).map(|_| rng.random::<u8>()).collect();
            let slice = dev.memcpy_stod(&bytes)?;
            let p = {
                let (p, _g) = slice.device_ptr(&stream);
                p
            };
            ptrs_pool.push(p);
            _keep.push(slice);
        }
        let weight_ptrs: Vec<u64> = (0..n_experts).map(|e| ptrs_pool[e % n_weights]).collect();

        for &(regime, total_batch) in &regimes {
            let per = total_batch / n_experts;
            let mut expert_offsets: Vec<i32> = vec![0];
            for e in 0..n_experts {
                let extra = usize::from(e < total_batch % n_experts);
                expert_offsets.push(expert_offsets.last().unwrap() + (per + extra) as i32);
            }
            let act: Vec<f32> = (0..total_batch * ncols)
                .map(|_| rng.random_range(-1.0f32..1.0))
                .collect();
            let q8 = quantize_acts_q8a128_test(&dev, &act, total_batch, ncols)?;

            for n_sub in [2usize, 4, 8] {
                for row_fast in [true, false] {
                    let run = || -> Result<()> {
                        q8.with_device_ptr(&dev, |act_ptr| {
                            crate::quantized::cuda::grouped_matmul_gemx_q8a128_with_mode(
                                act_ptr,
                                &weight_ptrs,
                                GgmlDType::MXFP4_KO,
                                nrows,
                                ncols,
                                total_batch,
                                &expert_offsets,
                                &dev,
                                Backing::Owned,
                                n_sub,
                                row_fast,
                                SumScale::Raw,
                            )
                        })?;
                        Ok(())
                    };
                    for _ in 0..3 {
                        run()?;
                    }
                    dev.synchronize()?;
                    let iters = 20;
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        run()?;
                    }
                    dev.synchronize()?;
                    let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
                    // Minimum weight traffic: each expert's full matrix once.
                    let min_w_gb = (n_experts * wbytes) as f64 / 1e9;
                    let lay = if row_fast { "rowF" } else { "tokF" };
                    println!(
                        "{label:>7} {regime}: n_sub={n_sub} {lay}  {ms:8.3} ms/call  \
                         min-weight-BW {:6.1} GB/s  ({total_batch} rows, ~{per}/expert)",
                        min_w_gb / (ms / 1e3),
                    );
                }
            }
        }
    }
    Ok(())
}

/// Benchmark each KO byte-permuted twin against its K format on the int8 q8a128
/// dense matmul across token counts. Both read 80/112/160-byte compact blocks; KO's
/// only difference is the (regularized) in-block byte order, so this measures whether
/// the reorder alone moves the needle before any wide-load optimization.
#[test]
#[ignore]
pub(super) fn ko_vs_k_int8_bench() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K

    // _KO = _K + 1 bit, per-128, F16 scale: (K dtype for the FP16 reference, KO dtype, maxq,
    // crumb_bytes, hi_bytes). int8 weights re-quantized per-128 from f32 (data-independent timing).
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let cases: &[(GgmlDType, GgmlDType, i32, usize, usize)] = &[
        (GgmlDType::Q4_K, GgmlDType::Q4_KO, 15, 0, 0), // same-bit (no +1): precision hit, smallest
        (GgmlDType::Q4_K, GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q5_K, GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_K, GgmlDType::Q8_KO, 255, 0, 0),
    ];

    // Dense per-128: as M ramps the weight amortizes over more tokens. Small M is memory-bound
    // (FP16/int8 read similar bytes → parity); large M is compute-bound where the accumulate-
    // then-scale collapse matters. FP16 = dequant→bf16 MMA on the K fixture; int8 = per-128 KO.
    for &(kd, kod, maxq, crumb, hi) in cases {
        // int8 KO weights — per-128 re-quant from f32, once per case (same across M).
        let ob = if kod == GgmlDType::Q8_KO {
            requant_q8_ko_per128(&wf32, nrows, ncols)
        } else {
            requant_ko_per128(&wf32, nrows, ncols, maxq, crumb, hi)
        };
        let ko_slice = dev.memcpy_stod(&ob)?;
        let stream = dev.cuda_stream();
        let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
        println!("=== dense per-128 [{nrows}x{ncols}]: FP16 {kd:?} vs int8 {kod:?} ===");
        println!(" tokens   f16(ms)  i8KO(ms) f16/i8KO  i8KO-tok/s");
        for &m in &[1usize, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
            let act: Vec<f32> = (0..m * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let q8 = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;
            let (_k_ptrs, k_st, _a, _l, _o) =
                build_int8_ab_fixture(&dev, nrows, ncols, &[m], &act, kd)?;
            // FP16 path: bf16 activations → dequant→f16 tensor-core MMA on the K weights.
            let act_bf16: Vec<half::bf16> = act.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let act_storage =
                CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
            let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![m, ncols]));
            let wshape = crate::Shape::from((nrows, ncols));

            let time_i8 = |ptr: u64| -> Result<f64> {
                for _ in 0..20 {
                    let _ = dense_qmatmul(
                        DynamicTensor::Int8(&q8),
                        ptr,
                        kod,
                        nrows,
                        0,
                        crate::DType::F32,
                        &dev,
                    )?;
                }
                dev.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..100 {
                        let _ = dense_qmatmul(
                            DynamicTensor::Int8(&q8),
                            ptr,
                            kod,
                            nrows,
                            0,
                            crate::DType::F32,
                            &dev,
                        )?;
                    }
                    dev.synchronize()?;
                    best = best.min(t.elapsed().as_secs_f64() / 100.0);
                }
                Ok(best)
            };
            let time_fp = || -> Result<f64> {
                for _ in 0..20 {
                    let _ = k_st[0].fwd_via_gemx(&wshape, &act_storage, &act_layout)?;
                }
                dev.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = Instant::now();
                    for _ in 0..100 {
                        let _ = k_st[0].fwd_via_gemx(&wshape, &act_storage, &act_layout)?;
                    }
                    dev.synchronize()?;
                    best = best.min(t.elapsed().as_secs_f64() / 100.0);
                }
                Ok(best)
            };
            let t_fp = time_fp()?;
            let t_ko = time_i8(ko_ptr)?;
            println!(
                "{m:>7} {:>9.4} {:>9.4} {:>8.2} {:>10.0}",
                t_fp * 1e3,
                t_ko * 1e3,
                t_fp / t_ko,
                m as f64 / t_ko
            );
        }
    }
    Ok(())
}

/// Representative MoE-decode benchmark: 128 experts, top-8 routing (Qwen3-30B-A3B
/// config), distinct weights per expert (no cross-expert L2 reuse). `batch` tokens
/// each route to 8 random unique experts and the grouped int8 matmul processes the
/// resulting (expert → token-slice) tiles in ONE launch — the real production path.
/// At batch=1 → 8 experts (1 token each); at batch=128 → ~all 128 experts (~8 tokens
/// each). This is what actually predicts the model, vs the dense single-weight bench.
#[test]
#[ignore]
pub(super) fn grouped_moe_ko_vs_k_bench() -> Result<()> {
    use std::collections::HashSet;
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K
    let num_experts = 128usize;
    let top_k = 8usize;
    let mut rng = rand::rng();

    // _KO = _K + 1 bit, per-128: (K dtype for FP16 ref, KO dtype, maxq, crumb_bytes, hi_bytes).
    let cases: &[(GgmlDType, GgmlDType, i32, usize, usize)] = &[
        (GgmlDType::Q4_K, GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q5_K, GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_K, GgmlDType::Q8_KO, 255, 0, 0),
    ];

    let shape = crate::Shape::from((nrows, ncols));
    for &(kd, kod, maxq, crumb, hi) in cases {
        // Build a pool of `num_experts` DISTINCT weights (K-compact + KO-permuted).
        let mut k_ptrs = vec![0u64; num_experts];
        let mut ko_ptrs = vec![0u64; num_experts];
        let mut k_pool = Vec::with_capacity(num_experts); // QCudaStorage, kept alive
        let mut ko_pool = Vec::with_capacity(num_experts); // CudaSlice<u8>, kept alive
        for e in 0..num_experts {
            let w: Vec<f32> = (0..nrows * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, kd)?;
            xs.quantize(&CudaStorage::wrap_cuda_slice(
                dev.memcpy_stod(&w)?,
                dev.clone(),
            ))?;
            let xs = xs.repack_gemx(&shape)?;
            k_ptrs[e] = xs.data_ptr();
            // int8 KO weights — per-128 re-quant from this expert's f32 (the _K+1-bit remap).
            let ob = if kod == GgmlDType::Q8_KO {
                requant_q8_ko_per128(&w, nrows, ncols)
            } else {
                requant_ko_per128(&w, nrows, ncols, maxq, crumb, hi)
            };
            let ko_slice = dev.memcpy_stod(&ob)?;
            {
                let stream = dev.cuda_stream();
                let (p, _g) = ko_slice.device_ptr(&stream);
                ko_ptrs[e] = p; // raw address stays valid while ko_slice lives in the pool
            }
            k_pool.push(xs);
            ko_pool.push(ko_slice);
        }

        println!(
            "=== grouped MoE [{nrows}x{ncols}], 128 experts, top-8: FP16 {kd:?} → int8 {kod:?} ==="
        );
        println!("  batch  experts  slices  f16-K(ms) i8-KO(ms) f16K/i8KO  i8KO-tok/s");
        for &batch in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512] {
            // Route each token to `top_k` unique experts; group by expert.
            let mut per_expert: Vec<Vec<usize>> = vec![Vec::new(); num_experts];
            for t in 0..batch {
                let mut chosen = HashSet::new();
                while chosen.len() < top_k {
                    chosen.insert(rng.random_range(0..num_experts));
                }
                for &e in &chosen {
                    per_expert[e].push(t);
                }
            }
            let touched: Vec<usize> = (0..num_experts)
                .filter(|&e| !per_expert[e].is_empty())
                .collect();
            let total: usize = touched.iter().map(|&e| per_expert[e].len()).sum();
            let mut expert_offsets = vec![0i32];
            let mut stacked: Vec<usize> = Vec::with_capacity(total);
            for &e in &touched {
                stacked.extend_from_slice(&per_expert[e]);
                expert_offsets.push(expert_offsets.last().unwrap() + per_expert[e].len() as i32);
            }
            // `batch` distinct token activations; stack them in expert-slice order.
            let token_acts: Vec<Vec<f32>> = (0..batch)
                .map(|_| (0..ncols).map(|_| rng.random_range(-1.0..1.0)).collect())
                .collect();
            let mut act = Vec::with_capacity(total * ncols);
            for &t in &stacked {
                act.extend_from_slice(&token_acts[t]);
            }
            // q8a128 activations for the int8 path; bf16 activations for the FP16 path.
            let q8a128 = quantize_acts_q8a128_test(&dev, &act, total, ncols)?;
            let act_bf16: Vec<bf16> = act.iter().map(|&v| bf16::from_f32(v)).collect();
            let act_storage =
                CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
            let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![total, ncols]));
            let wk: Vec<u64> = touched.iter().map(|&e| k_ptrs[e]).collect();
            let wko: Vec<u64> = touched.iter().map(|&e| ko_ptrs[e]).collect();

            let time_i8 = |wptrs: &[u64], dtype: GgmlDType| -> Result<f64> {
                for _ in 0..20 {
                    let _ = grouped_qmatmul(
                        DynamicTensor::Int8(&q8a128),
                        wptrs,
                        dtype,
                        nrows,
                        &expert_offsets,
                        &dev,
                        Backing::Owned,
                    )?;
                }
                dev.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let _ = grouped_qmatmul(
                            DynamicTensor::Int8(&q8a128),
                            wptrs,
                            dtype,
                            nrows,
                            &expert_offsets,
                            &dev,
                            Backing::Owned,
                        )?;
                    }
                    dev.synchronize()?;
                    best = best.min(t0.elapsed().as_secs_f64() / 100.0);
                }
                Ok(best)
            };
            let time_fp = |wptrs: &[u64], dtype: GgmlDType| -> Result<f64> {
                for _ in 0..20 {
                    let _ = grouped_matmul_gemx(
                        wptrs,
                        dtype,
                        nrows,
                        ncols,
                        &act_storage,
                        &act_layout,
                        &expert_offsets,
                        &dev,
                    )?;
                }
                dev.synchronize()?;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    for _ in 0..100 {
                        let _ = grouped_matmul_gemx(
                            wptrs,
                            dtype,
                            nrows,
                            ncols,
                            &act_storage,
                            &act_layout,
                            &expert_offsets,
                            &dev,
                        )?;
                    }
                    dev.synchronize()?;
                    best = best.min(t0.elapsed().as_secs_f64() / 100.0);
                }
                Ok(best)
            };
            let t_fp = time_fp(&wk, kd)?; // FP16 weights=K (current path)
            let t_i8ko = time_i8(&wko, kod)?; // int8 weights=KO (the upgrade)
                                              // Projected end-to-end token rate from the i8-KO matmul: 48 layers, each
                                              // = this FFN matmul + a 0.5× attention pass → 48 * 1.5 * t per decode step.
            let tok_s = batch as f64 / (48.0 * 1.5 * t_i8ko);
            println!(
                "{batch:>7} {:>8} {:>7} {:>9.4} {:>9.4} {:>8.2} {:>10.0}",
                touched.len(),
                total,
                t_fp * 1e3,
                t_i8ko * 1e3,
                t_fp / t_i8ko,
                tok_s
            );
        }
        drop(k_pool);
        drop(ko_pool);
    }
    Ok(())
}

/// Single-config statistical bench: Q4_K, 128 experts, 4096 activation rows
/// (32 token-slices/expert — a balanced prefill batch). Where the sweep bench
/// reports best-of-5 (which is noisy at the sub-percent level we care about for
/// the FP16-vs-int8 parity question), this draws M independent samples — each the
/// mean of N back-to-back launches — and reduces them to mean ± 95% CI, median,
/// min and coefficient of variation. The three paths are interleaved within every
/// sample so thermal/clock drift hits them equally and cancels in the comparison.
/// The final line states whether f16-K and i8-KO differ at 95% confidence or are
/// statistically parity.
#[test]
#[ignore = "retired: builds KO weights in the sub-major byte-permute layout and times the \
            i8-K (non-KO) path the pairing guard now rejects. Superseded by the per-128 \
            grouped_moe_ko_vs_k_bench (the production 128-expert f16-K vs i8-KO comparison)."]
pub(super) fn q4k_grouped_stats_bench() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K
    let num_experts = 128usize;
    let tokens_per_expert = 32usize;
    let total = num_experts * tokens_per_expert; // 4096 activation rows
    let mut rng = rand::rng();

    type Perm = fn(&[u8], &mut [u8]);
    let cases: &[(GgmlDType, GgmlDType, usize, Perm)] = &[
        (GgmlDType::Q4_K, GgmlDType::Q4_KO, 80, |kb, ob| {
            for (s, &kb0) in [0usize, 24, 40, 64].iter().enumerate() {
                ob[s * 16..s * 16 + 4].copy_from_slice(&kb[kb0..kb0 + 4]);
                ob[s * 16 + 4..s * 16 + 8].copy_from_slice(&kb[kb0 + 8..kb0 + 12]);
                ob[s * 16 + 8..s * 16 + 12].copy_from_slice(&kb[kb0 + 4..kb0 + 8]);
                ob[s * 16 + 12..s * 16 + 16].copy_from_slice(&kb[kb0 + 12..kb0 + 16]);
            }
            ob[64..68].copy_from_slice(&kb[16..20]);
            ob[68..72].copy_from_slice(&kb[20..24]);
            ob[72..76].copy_from_slice(&kb[56..60]);
            ob[76..80].copy_from_slice(&kb[60..64]);
        }),
        (GgmlDType::Q5_K, GgmlDType::Q5_KO, 112, |kb, ob| {
            const QS: [usize; 16] = [
                8, 12, 16, 20, 24, 28, 32, 36, 56, 60, 64, 68, 72, 76, 80, 84,
            ];
            const QH: [usize; 4] = [4, 40, 52, 88];
            const DM: [usize; 4] = [0, 44, 48, 92];
            for s in 0..4 {
                ob[s * 16..s * 16 + 4].copy_from_slice(&kb[QS[s * 4]..QS[s * 4] + 4]);
                ob[s * 16 + 4..s * 16 + 8].copy_from_slice(&kb[QS[s * 4 + 2]..QS[s * 4 + 2] + 4]);
                ob[s * 16 + 8..s * 16 + 12].copy_from_slice(&kb[QS[s * 4 + 1]..QS[s * 4 + 1] + 4]);
                ob[s * 16 + 12..s * 16 + 16].copy_from_slice(&kb[QS[s * 4 + 3]..QS[s * 4 + 3] + 4]);
            }
            for i in 0..4 {
                ob[64 + i * 4..64 + i * 4 + 4].copy_from_slice(&kb[QH[i]..QH[i] + 4]);
            }
            for s in 0..4 {
                ob[80 + s * 4..80 + s * 4 + 4].copy_from_slice(&kb[DM[s]..DM[s] + 4]);
            }
        }),
        (GgmlDType::Q6_K, GgmlDType::Q6_KO, 112, |kb, ob| {
            ob.copy_from_slice(kb)
        }),
        (GgmlDType::Q8_K, GgmlDType::Q8_KO, 160, |kb, ob| {
            const QS: [usize; 16] = [
                0, 8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 120, 128, 136,
            ];
            const DM: [usize; 4] = [24, 112, 116, 144];
            for m in 0..16 {
                ob[m * 8..m * 8 + 8].copy_from_slice(&kb[QS[m]..QS[m] + 8]);
            }
            for s in 0..4 {
                ob[128 + s * 4..128 + s * 4 + 4].copy_from_slice(&kb[DM[s]..DM[s] + 4]);
            }
        }),
    ];
    let shape = crate::Shape::from((nrows, ncols));

    // Balanced routing: exactly `tokens_per_expert` slices per expert, fixed once so
    // routing variance is not a noise source. Shared across all formats.
    let mut expert_offsets = vec![0i32];
    for _ in 0..num_experts {
        expert_offsets.push(expert_offsets.last().unwrap() + tokens_per_expert as i32);
    }
    let act: Vec<f32> = (0..total * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8a128 = quantize_acts_q8a128_test(&dev, &act, total, ncols)?;
    let act_bf16: Vec<bf16> = act.iter().map(|&v| bf16::from_f32(v)).collect();
    let act_storage = CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
    let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![total, ncols]));

    let n_inner = 50usize; // launches averaged per sample
    let m_samples = 200usize; // independent samples → CI ∝ 1/sqrt(M)

    // mean, median, min, 95% CI half-width, CV% — sample stats (N-1 variance).
    fn stats(xs: &[f64]) -> (f64, f64, f64, f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = var.sqrt();
        let ci95 = 1.96 * std / n.sqrt();
        let cv = std / mean * 100.0;
        let mut sorted = xs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (mean, sorted[sorted.len() / 2], sorted[0], ci95, cv)
    }

    println!(
        "=== grouped MoE stats: {num_experts} experts × {tokens_per_expert} = {total} act rows, [{nrows}x{ncols}] ==="
    );
    println!("    {m_samples} samples × {n_inner} launches each, interleaved per sample\n");

    for &(kd, kod, stride, perm) in cases {
        // 128 distinct expert weights (K-compact + KO-permuted), kept alive in pools.
        let mut k_ptrs = vec![0u64; num_experts];
        let mut ko_ptrs = vec![0u64; num_experts];
        let mut k_pool = Vec::with_capacity(num_experts);
        let mut ko_pool = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let w: Vec<f32> = (0..nrows * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, kd)?;
            xs.quantize(&CudaStorage::wrap_cuda_slice(
                dev.memcpy_stod(&w)?,
                dev.clone(),
            ))?;
            let xs = xs.repack_gemx(&shape)?;
            k_ptrs[e] = xs.data_ptr();
            let kb = xs.data()?;
            // Q6_KO/Q8_KO: requantize F32 → per-32 affine (scale,min). Q4/Q5 keep the byte
            // permutation (already per-32 affine).
            let de = if kod == GgmlDType::Q8_KO {
                requant_q8_ko_affine(&w, nrows, ncols)
            } else if kod == GgmlDType::Q6_KO {
                requant_q6_ko_affine(&w, nrows, ncols)
            } else {
                ko_deinterleave(
                    &kb,
                    stride,
                    ko_quant_bytes(kod),
                    kod == GgmlDType::Q8_KO,
                    perm,
                )
            };
            // Sub-major k1024 layout: Q4 from the compact, Q5/Q6/Q8 from the per-format `de`
            // ([quants|scale] — requant for Q6/Q8, de-interleave for Q5).
            let ob = match kod {
                GgmlDType::Q4_KO => q4ko_submajor_from_compact(&kb, nrows, ncols),
                GgmlDType::Q5_KO => q5ko_submajor_from_de(&de, nrows, ncols),
                GgmlDType::Q6_KO => q6ko_submajor_from_de(&de, nrows, ncols),
                GgmlDType::Q8_KO => q8ko_submajor_from_de(&de, nrows, ncols),
                _ => unreachable!("not a KO format: {kod:?}"),
            };
            let ko_slice = dev.memcpy_stod(&ob)?;
            {
                let stream = dev.cuda_stream();
                let (p, _g) = ko_slice.device_ptr(&stream);
                ko_ptrs[e] = p;
            }
            k_pool.push(xs);
            ko_pool.push(ko_slice);
        }

        let wk = k_ptrs.clone();
        let wko = ko_ptrs.clone();

        // One-shot precision: KO int8 result vs the FP16 reference of the SAME weights.
        {
            let ref_f16 = grouped_matmul_gemx(
                &wk,
                kd,
                nrows,
                ncols,
                &act_storage,
                &act_layout,
                &expert_offsets,
                &dev,
            )?;
            let ko_i8 = grouped_qmatmul(
                DynamicTensor::Int8(&q8a128),
                &wko,
                kod,
                nrows,
                &expert_offsets,
                &dev,
                Backing::Owned,
            )?;
            let vref = read_bf16_tensor(&dev, &ref_f16)?;
            let vko = read_f32_tensor(&dev, &ko_i8)?;
            println!(
                "  precision  i8-{kod:?} vs f16-{kd:?}: rel_l2 = {:.5}",
                rel_l2(&vref, &vko)
            );
        }

        // Warmup every path past clock spin-up before sampling.
        for _ in 0..100 {
            let _ = grouped_matmul_gemx(
                &wk,
                kd,
                nrows,
                ncols,
                &act_storage,
                &act_layout,
                &expert_offsets,
                &dev,
            )?;
            let _ = grouped_qmatmul(
                DynamicTensor::Int8(&q8a128),
                &wk,
                kd,
                nrows,
                &expert_offsets,
                &dev,
                Backing::Owned,
            )?;
            let _ = grouped_qmatmul(
                DynamicTensor::Int8(&q8a128),
                &wko,
                kod,
                nrows,
                &expert_offsets,
                &dev,
                Backing::Owned,
            )?;
        }
        dev.synchronize()?;

        let mut s_fp = Vec::with_capacity(m_samples);
        let mut s_i8k = Vec::with_capacity(m_samples);
        let mut s_i8ko = Vec::with_capacity(m_samples);
        for _ in 0..m_samples {
            let t = Instant::now();
            for _ in 0..n_inner {
                let _ = grouped_matmul_gemx(
                    &wk,
                    kd,
                    nrows,
                    ncols,
                    &act_storage,
                    &act_layout,
                    &expert_offsets,
                    &dev,
                )?;
            }
            dev.synchronize()?;
            s_fp.push(t.elapsed().as_secs_f64() / n_inner as f64);

            let t = Instant::now();
            for _ in 0..n_inner {
                let _ = grouped_qmatmul(
                    DynamicTensor::Int8(&q8a128),
                    &wk,
                    kd,
                    nrows,
                    &expert_offsets,
                    &dev,
                    Backing::Owned,
                )?;
            }
            dev.synchronize()?;
            s_i8k.push(t.elapsed().as_secs_f64() / n_inner as f64);

            let t = Instant::now();
            for _ in 0..n_inner {
                let _ = grouped_qmatmul(
                    DynamicTensor::Int8(&q8a128),
                    &wko,
                    kod,
                    nrows,
                    &expert_offsets,
                    &dev,
                    Backing::Owned,
                )?;
            }
            dev.synchronize()?;
            s_i8ko.push(t.elapsed().as_secs_f64() / n_inner as f64);
        }

        let (fp_m, fp_md, fp_min, fp_ci, fp_cv) = stats(&s_fp);
        let (k_m, k_md, k_min, k_ci, k_cv) = stats(&s_i8k);
        let (ko_m, ko_md, ko_min, ko_ci, ko_cv) = stats(&s_i8ko);

        println!("--- FP16 {kd:?} → int8 {kod:?} ---");
        println!("  path     mean(ms)   ±95%CI   median(ms)   min(ms)   CV%");
        println!(
            "  f16-K   {:>8.4}  {:>7.4}   {:>8.4}  {:>8.4}  {:>5.2}",
            fp_m * 1e3,
            fp_ci * 1e3,
            fp_md * 1e3,
            fp_min * 1e3,
            fp_cv
        );
        println!(
            "  i8-K    {:>8.4}  {:>7.4}   {:>8.4}  {:>8.4}  {:>5.2}",
            k_m * 1e3,
            k_ci * 1e3,
            k_md * 1e3,
            k_min * 1e3,
            k_cv
        );
        println!(
            "  i8-KO   {:>8.4}  {:>7.4}   {:>8.4}  {:>8.4}  {:>5.2}",
            ko_m * 1e3,
            ko_ci * 1e3,
            ko_md * 1e3,
            ko_min * 1e3,
            ko_cv
        );
        let ratio = fp_m / ko_m;
        let disjoint = (fp_m - fp_ci) > (ko_m + ko_ci) || (ko_m - ko_ci) > (fp_m + fp_ci);
        let pct = (ratio - 1.0) * 100.0;
        println!(
            "  f16-K / i8-KO = {ratio:.4} ({pct:+.2}%)  →  {}\n",
            if disjoint {
                if ratio > 1.0 {
                    "SIGNIFICANT win for i8-KO (95% CI)"
                } else {
                    "SIGNIFICANT win for f16-K (95% CI)"
                }
            } else {
                "parity (CIs overlap at 95%)"
            }
        );

        drop(k_pool);
        drop(ko_pool);
    }
    Ok(())
}

/// Perf A/B — time the INT8 grouped kernel (`q8a128 × KO`) vs the legacy FP16 grouped
/// kernel (dequant K-quant) on the SAME source weight, **single expert**, across a
/// tokens-per-expert sweep. Single expert so the weight load amortizes across M: small
/// M is weight-bandwidth-bound, large M is compute-bound where the INT8 `m16n8k32`
/// should pull ahead. Min-over-batches filters transient GPU stalls.
#[test]
#[ignore = "GPU perf benchmark; run with --ignored --nocapture"]
pub(super) fn grouped_int8_vs_legacy_bench() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 768; // N = intermediate (gate/up)
    let ncols = 2048; // K = hidden
    let shape = crate::Shape::from((nrows, ncols));
    let mut rng = rand::rng();

    // (K dtype for the FP16 ref, KO twin for int8, maxq, crumb_bytes, hi_bytes).
    let cases: &[(GgmlDType, GgmlDType, i32, usize, usize)] = &[
        (GgmlDType::Q4_K, GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q5_K, GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_K, GgmlDType::Q8_KO, 255, 0, 0),
    ];
    for &(kd, kod, maxq, crumb, hi) in cases {
        // One expert: FP16-K weight (gemx-repacked) + its KO twin, from the same f32.
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, kd)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&w)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        let k_ptrs = vec![xs.data_ptr()];
        let ob = if kod == GgmlDType::Q8_KO {
            requant_q8_ko_per128(&w, nrows, ncols)
        } else {
            requant_ko_per128(&w, nrows, ncols, maxq, crumb, hi)
        };
        let ko_slice = dev.memcpy_stod(&ob)?;
        let stream = dev.cuda_stream();
        let (ko_ptr, _kg) = ko_slice.device_ptr(&stream);
        let ko_ptrs = vec![ko_ptr];

        println!(
            "\n=== Grouped expert matmul: INT8 {kod:?} vs legacy FP16 {kd:?} [{nrows}x{ncols}], single expert ==="
        );
        println!(
            "{:>7} {:>11} {:>12} {:>9} {:>14}",
            "tokens", "int8(ms)", "legacy(ms)", "speedup", "int8 GFLOP/s"
        );

        for &m in &[1usize, 8, 16, 32, 64, 128, 256, 512] {
            let act_data: Vec<f32> = (0..m * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let act_bf16: Vec<bf16> = act_data.iter().map(|&v| bf16::from_f32(v)).collect();
            let act_storage =
                CudaStorage::wrap_cuda_slice(dev.memcpy_stod(&act_bf16)?, dev.clone());
            let act_layout = crate::Layout::contiguous(crate::Shape::from(vec![m, ncols]));
            let expert_offsets = vec![0i32, m as i32];
            // Pre-quantize activations ONCE to q8a128, off the timed path.
            let q8a128 = quantize_acts_q8a128_test(&dev, &act_data, m, ncols)?;

            let time_call = |legacy: bool| -> Result<f64> {
                for _ in 0..20 {
                    if legacy {
                        grouped_matmul_gemx(
                            &k_ptrs,
                            kd,
                            nrows,
                            ncols,
                            &act_storage,
                            &act_layout,
                            &expert_offsets,
                            &dev,
                        )?;
                    } else {
                        grouped_qmatmul(
                            DynamicTensor::Int8(&q8a128),
                            &ko_ptrs,
                            kod,
                            nrows,
                            &expert_offsets,
                            &dev,
                            Backing::Owned,
                        )?;
                    }
                }
                dev.synchronize()?;
                let iters = 100usize;
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        if legacy {
                            grouped_matmul_gemx(
                                &k_ptrs,
                                kd,
                                nrows,
                                ncols,
                                &act_storage,
                                &act_layout,
                                &expert_offsets,
                                &dev,
                            )?;
                        } else {
                            grouped_qmatmul(
                                DynamicTensor::Int8(&q8a128),
                                &ko_ptrs,
                                kod,
                                nrows,
                                &expert_offsets,
                                &dev,
                                Backing::Owned,
                            )?;
                        }
                    }
                    dev.synchronize()?;
                    best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
                }
                Ok(best)
            };

            let t_int8 = time_call(false)?;
            let t_legacy = time_call(true)?;
            let flops = 2.0 * m as f64 * nrows as f64 * ncols as f64;
            println!(
                "{:>7} {:>11.4} {:>12.4} {:>9.2} {:>14.1}",
                m,
                t_int8 * 1e3,
                t_legacy * 1e3,
                t_legacy / t_int8,
                flops / t_int8 / 1e9,
            );
        }
    }
    Ok(())
}

/// Crossover benchmark for the q8a128 **dense** int8 tiling modes — mode-1 (`Bm=16`, `N_SUB=1`)
/// vs mode-2 (`Bm=32`, `N_SUB=2`). The mode choice trades **weight** re-reads (M-tiling) against
/// **activation** re-reads (N-tiling) — both are DRAM pressure — so the crossover lives on the full
/// `(M, N, K)` surface, not weight bytes alone (equal-byte `6144×2048` and `2048×6144` cross at
/// different `M`). This sweeps the three axes independently and densely.
///
/// Per `(N, K)` and `M` it times both modes (L2 flushed between launches, median of `ITERS`) and
/// prints, alongside the measured `m1/m2`, the **modeled** quantities the fit needs:
/// - `traf2/1` = modeled DRAM traffic ratio mode-2 / mode-1
///   (`ceil(M/Bm)·W + ceil(N/Nt)·A`, `W=N·K·0.5`, `A=M·K`),
/// - `blk1`,`blk2` = launched block counts (`ceil(M/Bm)·ceil(N/Nt)`) — the occupancy that gates
///   whether mode-2's lower traffic can actually be realized at small `M·N`.
///
/// Run:
/// ```text
/// cargo test -p candle-core --features cuda --release \
///   q8a128_dense_mode_crossover -- --ignored --nocapture
/// ```
#[test]
#[ignore = "benchmark: run with --ignored --nocapture to fit the mode-2 crossover surface"]
pub(super) fn q8a128_dense_mode_crossover() -> Result<()> {
    use crate::quantized::int8_matmul_mode::q8a128_dense_use_mode2;

    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let l2 = dev.l2_cache_size().unwrap_or(0);
    let sm = dev.multiprocessor_count().unwrap_or(0);
    let flush_buf = unsafe { dev.alloc::<u8>((l2.max(1 << 20)) * 2)? };
    // Agreement accounting: only count cells with a *clear* winner (|1 - m1/m2| >= 5%); near-wash
    // cells flip on noise and shouldn't score against the formula.
    let (mut agree, mut clear) = (0usize, 0usize);

    const WARMUP: usize = 3;
    const ITERS: usize = 30;

    // Independent axes: N (output / weight-reuse), K (contraction, shared with the activation),
    // M (activation rows / activation-reuse). N % 32 == 0, K % 128 == 0.
    let ns: &[usize] = &[2048, 4096, 8192];
    let ks: &[usize] = &[2048, 4096, 8192];
    let ms: &[usize] = &[
        1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 80, 96, 128, 160,
        192, 256, 384, 512,
    ];

    let median = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let ceil = |x: usize, b: usize| ((x + b - 1) / b) as f64;

    println!(
        "\n=== q8a128 dense mode crossover surface ===\nL2={} MiB | SMs={sm} | WARMUP={WARMUP} ITERS={ITERS} | median µs, L2 flushed between launches",
        l2 >> 20
    );
    let mut summary: Vec<(usize, usize, f64, Option<usize>)> = Vec::new();

    let mut rng = rand::rng();
    for &n in ns {
        for &k in ks {
            let wf32: Vec<f32> = (0..n * k).map(|_| rng.random_range(-0.1f32..0.1)).collect();
            let ob = requant_ko_per128(&wf32, n, k, 15, 0, 0);
            let kod = GgmlDType::Q4_KO;
            let wbytes = ob.len();
            let ko_slice = dev.memcpy_stod(&ob)?;
            let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
            let w_mib = wbytes as f64 / (1 << 20) as f64;

            println!("\n# N={n} K={k}  W={w_mib:.1}MiB  act={k}B/token");
            println!(
                "  {:>5} | {:>9} | {:>9} | {:>6} | {:>7} | {:>6} | {:>6} | {:>5} | {:>5} | ok",
                "M", "mode1us", "mode2us", "m1/m2", "traf2/1", "blk1", "blk2", "meas", "pred"
            );

            let mut ratios: Vec<(usize, f64)> = Vec::new();
            for &m in ms {
                let act: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
                let op = quantize_acts_q8a128_test(&dev, &act, m, k)?;
                let time = |mode2: bool| -> Result<f64> {
                    let tile = if mode2 {
                        DenseTile::Mode2
                    } else {
                        DenseTile::Mode1
                    };
                    for _ in 0..WARMUP {
                        let _ = q8a128_dense_matmul(
                            &op,
                            ko_ptr,
                            kod,
                            n,
                            wbytes,
                            tile,
                            crate::DType::F32,
                            &dev,
                        )?;
                    }
                    dev.synchronize()?;
                    let mut v = Vec::with_capacity(ITERS);
                    for _ in 0..ITERS {
                        cuda_flush_l2(&flush_buf, &dev);
                        let t0 = Instant::now();
                        let _ = q8a128_dense_matmul(
                            &op,
                            ko_ptr,
                            kod,
                            n,
                            wbytes,
                            tile,
                            crate::DType::F32,
                            &dev,
                        )?;
                        dev.synchronize()?;
                        v.push(t0.elapsed().as_secs_f64() * 1e6);
                    }
                    Ok(median(v))
                };
                let us1 = time(false)?;
                let us2 = time(true)?;

                // Modeled DRAM traffic (Q4 weight = 0.5 B/elt; q8a128 act ≈ 1 B/elt).
                let w = (n * k) as f64 * 0.5;
                let a = (m * k) as f64;
                let t1 = ceil(m, 16) * w + ceil(n, 32) * a;
                let t2 = ceil(m, 32) * w + ceil(n, 64) * a;
                let blk1 = (ceil(m, 16) * ceil(n, 32)) as usize;
                let blk2 = (ceil(m, 32) * ceil(n, 64)) as usize;

                let r = us1 / us2;
                ratios.push((m, r));

                // Derived-formula prediction vs the measurement. Only score cells with a clear
                // winner (>=5% gap); near-wash cells flip on noise.
                let meas2 = us2 < us1;
                let pred2 = q8a128_dense_use_mode2(m, n, k, sm);
                let is_clear = (r - 1.0).abs() >= 0.05;
                if is_clear {
                    clear += 1;
                    if meas2 == pred2 {
                        agree += 1;
                    }
                }
                let ok = if !is_clear {
                    "~"
                } else if meas2 == pred2 {
                    "Y"
                } else {
                    "X"
                };
                println!(
                    "  {:>5} | {:>9.2} | {:>9.2} | {:>6.2} | {:>7.3} | {:>6} | {:>6} | {:>5} | {:>5} | {}",
                    m,
                    us1,
                    us2,
                    r,
                    t2 / t1,
                    blk1,
                    blk2,
                    if meas2 { "mode2" } else { "mode1" },
                    if pred2 { "mode2" } else { "mode1" },
                    ok
                );
            }
            // Sustained crossover: first M where this and the next sample both favor mode-2.
            let crossover = (0..ratios.len().saturating_sub(1))
                .find(|&i| ratios[i].1 > 1.0 && ratios[i + 1].1 > 1.0)
                .map(|i| ratios[i].0);
            println!("  -> sustained crossover M = {crossover:?}");
            summary.push((n, k, w_mib, crossover));
        }
    }

    println!("\n=== crossover summary  (N, K, W_MiB) -> M* ===");
    for (n, k, w, m) in &summary {
        println!(
            "  N={n:>5} K={k:>5}  W={w:>5.1}MiB  M*={}",
            m.map_or("none".to_string(), |x| x.to_string())
        );
    }
    println!(
        "\n=== formula vs measurement: {agree}/{clear} clear-winner cells agree ({:.1}%) ===",
        100.0 * agree as f64 / clear.max(1) as f64
    );
    Ok(())
}
