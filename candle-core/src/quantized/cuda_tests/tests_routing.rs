//! Device routing kernels: the Sinkhorn normalisation, top-k softmax routing and
//! the look-ahead vote words.
//!
//! `moe_route` must select the same top-k indices as a softmax, sort and
//! renormalise chain, with indices exact and weights to f32 epsilon. Degenerate
//! logits, whether all `-inf`, NaN, or with fewer finite experts than k, must
//! never route to an out-of-range expert. The look-ahead vote word places each
//! routed pick above every margin pick, and a parallel sum carries the router
//! probability of each pick, so the two can be read apart.

use super::*;

/// The fused Sinkhorn kernel (`run_sinkhorn_f32`) must reproduce the scalar reference that mirrors
/// `hyper.rs::sinkhorn` (softmax over cols + eps → col-norm → (iters-1)×[row-norm, col-norm]) for a
/// batch of small `[hc, hc]` matrices — the one-launch replacement for the ~120 tiny host ops.
#[test]
pub(super) fn sinkhorn_kernel_matches_scalar_reference() -> Result<()> {
    use candle_kernels::simple::sinkhorn::run_sinkhorn_f32;
    use std::ffi::c_void;

    // Scalar reference — identical op order to the kernel and to `hyper.rs::sinkhorn`.
    fn reference(a: &[f32], hc: usize, iters: usize, eps: f32) -> Vec<f32> {
        let mut c = vec![0f32; hc * hc];
        for i in 0..hc {
            let mut m = f32::MIN;
            for j in 0..hc {
                m = m.max(a[i * hc + j]);
            }
            let mut s = 0f32;
            for j in 0..hc {
                let e = (a[i * hc + j] - m).exp();
                c[i * hc + j] = e;
                s += e;
            }
            for j in 0..hc {
                c[i * hc + j] = c[i * hc + j] / s + eps;
            }
        }
        for j in 0..hc {
            let mut s = eps;
            for i in 0..hc {
                s += c[i * hc + j];
            }
            for i in 0..hc {
                c[i * hc + j] /= s;
            }
        }
        for _ in 0..iters - 1 {
            for i in 0..hc {
                let mut s = eps;
                for j in 0..hc {
                    s += c[i * hc + j];
                }
                for j in 0..hc {
                    c[i * hc + j] /= s;
                }
            }
            for j in 0..hc {
                let mut s = eps;
                for i in 0..hc {
                    s += c[i * hc + j];
                }
                for i in 0..hc {
                    c[i * hc + j] /= s;
                }
            }
        }
        c
    }

    let dev = CudaDevice::new(0)?;
    let (n, hc, iters, eps) = (5usize, 4usize, 20usize, 1e-6f32);
    let mut s = 0x1234_5678u32;
    let inp: Vec<f32> = (0..n * hc * hc)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 4.0 - 2.0
        })
        .collect();

    let inp_dev = dev.memcpy_stod(&inp)?;
    let mut out_dev = unsafe { dev.alloc::<f32>(n * hc * hc)? };
    let stream = dev.cuda_stream();
    {
        let (ip, _g0) = inp_dev.device_ptr(&stream);
        let (op, _g1) = out_dev.device_ptr_mut(&stream);
        unsafe {
            run_sinkhorn_f32(
                ip as *const f32,
                op as *mut f32,
                n as i32,
                hc as i32,
                iters as i32,
                eps,
                stream.cu_stream() as *mut c_void,
            );
        }
    }
    dev.synchronize()?;
    let got: Vec<f32> = dev.memcpy_dtov(&out_dev.slice(..))?;

    for m in 0..n {
        let want = reference(&inp[m * hc * hc..(m + 1) * hc * hc], hc, iters, eps);
        for k in 0..hc * hc {
            let g = got[m * hc * hc + k];
            let w = want[k];
            assert!(
                (g - w).abs() < 1e-5,
                "matrix {m} elem {k}: kernel {g} vs reference {w}"
            );
        }
        // Doubly-stochastic sanity: rows and columns each sum to ~1.
        for i in 0..hc {
            let rs: f32 = (0..hc).map(|j| got[m * hc * hc + i * hc + j]).sum();
            assert!((rs - 1.0).abs() < 1e-3, "matrix {m} row {i} sum {rs}");
        }
    }
    Ok(())
}

/// `moe_route` fuses softmax + top-k select + (optional) renormalize into one kernel. Validate it
/// against the reference math it replaces (`softmax → sort(desc) → narrow(k) → renorm`): the top-k
/// **indices** must match exactly and the **weights** to f32 epsilon. Covers F32/BF16 logits, both
/// `norm_topk` modes, num_tokens=1 (decode) through batched, k=8 (Qwen3) and a small k, and a
/// non-128 expert count. Logits are `0.5 × distinct-int` so they are bit-exact in bf16 (no
/// rounding → no ties → deterministic top-k).
#[test]
pub(super) fn cuda_moe_route_matches_reference() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());

    fn check(
        device: &crate::Device,
        num_tokens: usize,
        n_experts: usize,
        k: usize,
        dt: crate::DType,
        norm: bool,
    ) -> Result<()> {
        // A tie-free injective hash per row: mod 251 covers e<251, mod 521 (prime
        // > 512) covers the wide-slot instantiation. Scale by 0.5 (a power of
        // two); the 521-modulus values reach 260 and are exact only in f32, so
        // the 512-expert rung runs F32-only below.
        let m = if n_experts > 251 { 521 } else { 251 };
        let logit = move |t: usize, e: usize| -> f32 { (((e * 131 + t * 17) % m) as f32) * 0.5 };
        let data: Vec<f32> = (0..num_tokens * n_experts)
            .map(|i| logit(i / n_experts, i % n_experts))
            .collect();

        let logits =
            crate::Tensor::from_vec(data, (num_tokens, n_experts), device)?.to_dtype(dt)?;
        let (w, idx) = moe_route(&logits, k, norm)?;
        let w = w.to_vec2::<f32>()?;
        let idx = idx.to_vec2::<u32>()?;

        for t in 0..num_tokens {
            let row: Vec<f32> = (0..n_experts).map(|e| logit(t, e)).collect();
            // Reference top-k: descending value, lowest index on tie (none here).
            let mut order: Vec<usize> = (0..n_experts).collect();
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
            let top = &order[..k];

            let gmax = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&l| (l - gmax).exp()).collect();
            let z_all: f32 = exps.iter().sum();
            let z_top: f32 = top.iter().map(|&e| exps[e]).sum();
            let denom = if norm { z_top } else { z_all };

            for p in 0..k {
                assert_eq!(
                    idx[t][p] as usize, top[p],
                    "index mismatch nt={num_tokens} n_exp={n_experts} k={k} dt={dt:?} norm={norm} t={t} p={p}"
                );
                let want = exps[top[p]] / denom;
                assert!(
                    (w[t][p] - want).abs() <= 1e-4 * (1.0 + want.abs()),
                    "weight mismatch nt={num_tokens} k={k} dt={dt:?} norm={norm} t={t} p={p}: got {} want {want}",
                    w[t][p]
                );
            }
        }
        Ok(())
    }

    for &dt in &[crate::DType::F32, crate::DType::BF16] {
        for &norm in &[true, false] {
            check(&device, 1, 128, 8, dt, norm)?; // decode: single token
            check(&device, 7, 128, 8, dt, norm)?; // batched, Qwen3 top-8
            check(&device, 5, 128, 2, dt, norm)?; // small k
            check(&device, 3, 64, 4, dt, norm)?; // non-128 expert count
        }
    }
    // The 16-slot instantiation (n_experts > 256): qwen4exp's 512 × top-10,
    // plus the boundary just past the narrow kernel. F32-only — see the logit
    // representability note above.
    for &norm in &[true, false] {
        check(&device, 1, 512, 10, crate::DType::F32, norm)?;
        check(&device, 7, 512, 10, crate::DType::F32, norm)?;
        check(&device, 3, 257, 4, crate::DType::F32, norm)?;
    }
    Ok(())
}

/// A router whose projection shared its launch reads its columns of the wider row in
/// place: routing the first 512 columns of a 1824-wide row is routing those columns
/// copied out, index for index and bit for bit.
#[test]
pub(super) fn cuda_moe_route_reads_a_strided_row() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (rows, width, experts, k) = (5usize, 1824usize, 512usize, 10usize);
    let data: Vec<f32> = (0..rows * width)
        .map(|i| (((i * 131 + 17) % 521) as f32) * 0.5)
        .collect();
    let wide = crate::Tensor::from_vec(data, (rows, width), &device)?;
    let view = wide.narrow(1, 0, experts)?;
    let packed = view.contiguous()?;
    for norm in [true, false] {
        let (w_view, i_view) = moe_route(&view, k, norm)?;
        let (w_packed, i_packed) = moe_route(&packed, k, norm)?;
        assert_eq!(
            i_view.to_vec2::<u32>()?,
            i_packed.to_vec2::<u32>()?,
            "norm={norm}"
        );
        assert_eq!(
            w_view.to_vec2::<f32>()?,
            w_packed.to_vec2::<f32>()?,
            "norm={norm}"
        );
    }
    Ok(())
}

/// Degenerate-logit guard: a token whose logits are all `-inf`/`NaN`, or that has
/// fewer finite experts than `k`, must NEVER route to an out-of-range expert. The
/// kernel seeds its argmax with `bi = n_experts` as a "not found" sentinel; if that
/// leaks to the output it indexes past the per-layer expert tables and panics the
/// expert-paging pipeline (bricking decode). Assert every emitted index is in range
/// and every weight is finite, and that a short-finite row keeps its real experts
/// with a zero-weight fallback filling the rest.
#[test]
pub(super) fn cuda_moe_route_never_emits_out_of_range_index() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let n_experts = 128usize;
    let k = 8usize;
    let ninf = f32::NEG_INFINITY;

    let rows = 5usize;
    let mut data = vec![0f32; rows * n_experts];
    // Row 0: all -inf — no finite candidate for any slot.
    for e in 0..n_experts {
        data[e] = ninf;
    }
    // Row 1: all NaN — NaN loses every `>` compare, so the sentinel would survive.
    for e in 0..n_experts {
        data[n_experts + e] = f32::NAN;
    }
    // Row 2: only 3 finite experts, rest -inf (fewer finite than k).
    for e in 0..n_experts {
        data[2 * n_experts + e] = ninf;
    }
    data[2 * n_experts + 10] = 3.0;
    data[2 * n_experts + 20] = 2.0;
    data[2 * n_experts + 30] = 1.0;
    // Row 3: ordinary distinct logits — normal routing.
    for e in 0..n_experts {
        data[3 * n_experts + e] = (e as f32) * 0.5;
    }
    // Row 4: 3 finite experts, rest NaN (not -inf). This is the case whose KEPT
    // slots' weights depend on the warp `gmax` reduction dropping NaN (CUDA
    // `fmaxf`) so `gmax` stays finite (3.0) — otherwise the kept weights would
    // be `exp(finite - NaN) = NaN` and the index-based clamp would NOT catch it.
    for e in 0..n_experts {
        data[4 * n_experts + e] = f32::NAN;
    }
    data[4 * n_experts + 5] = 3.0;
    data[4 * n_experts + 15] = 2.0;
    data[4 * n_experts + 25] = 1.0;

    let logits = crate::Tensor::from_vec(data, (rows, n_experts), &device)?;
    for &norm in &[true, false] {
        let (w, idx) = moe_route(&logits, k, norm)?;
        let idx = idx.to_vec2::<u32>()?;
        let w = w.to_vec2::<f32>()?;
        for t in 0..rows {
            for p in 0..k {
                assert!(
                    (idx[t][p] as usize) < n_experts,
                    "row {t} slot {p}: index {} must be < n_experts {n_experts} (norm={norm})",
                    idx[t][p]
                );
                assert!(
                    w[t][p].is_finite(),
                    "row {t} slot {p}: weight {} must be finite (norm={norm})",
                    w[t][p]
                );
            }
        }
        // Rows 2 & 4: the three finite experts are selected in descending order,
        // with real (non-zero) kept weights; the remaining slots are the
        // zero-weight fallback (not a phantom expert).
        for (base, e0, e1, e2) in [(2usize, 10u32, 20u32, 30u32), (4, 5, 15, 25)] {
            assert_eq!(idx[base][0], e0, "row{base} norm={norm}");
            assert_eq!(idx[base][1], e1, "row{base} norm={norm}");
            assert_eq!(idx[base][2], e2, "row{base} norm={norm}");
            assert!(
                w[base][0] > 0.0,
                "row{base} kept slot 0 weight (norm={norm})"
            );
            for p in 3..k {
                assert_eq!(
                    w[base][p], 0.0,
                    "row{base} slot {p} must be zero-weight fallback (norm={norm})"
                );
            }
        }
    }
    Ok(())
}

/// Perf probe: fused `moe_route` vs the op-chain it replaced (softmax -> sort -> narrow -> renorm),
/// bf16 logits [num_tokens, 128] top-8, across decode (small) -> prefill (large) token counts.
/// Both include their output allocation (the real per-call cost). Run with:
///   cargo test -p candle-core --features cuda --release bench_moe_route -- --ignored --nocapture
#[test]
#[ignore = "GPU perf probe; run with --ignored --nocapture"]
pub(super) fn bench_moe_route() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let n_experts = 128usize;
    let k = 8usize;
    let iters = 2000usize;

    println!(
        "\n=== moe_route vs op-chain [n_experts={n_experts} k={k}] bf16, {iters} calls/sync ===",
    );
    println!(
        "{:>10} {:>13} {:>13} {:>13} {:>13}",
        "num_tokens", "full us", "rawkern us", "alloc2 us", "opchain us"
    );

    let bench = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
        for _ in 0..50 {
            f()?;
        }
        dev.synchronize()?;
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                f()?;
            }
            dev.synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        Ok(best)
    };

    for &nt in &[1usize, 4, 16, 64, 256, 1024, 4096] {
        let data: Vec<f32> = (0..nt * n_experts)
            .map(|i| (((i * 131) % 251) as f32) * 0.05)
            .collect();
        let logits = crate::Tensor::from_vec(data, (nt, n_experts), &device)?
            .to_dtype(crate::DType::BF16)?;

        // Pre-grab the logits device pointer + pre-allocate output buffers ONCE for the
        // raw-kernel probe — isolates the FFI launch + kernel from alloc/wrap overhead.
        let (storage, layout) = logits.storage_and_layout();
        let (lo1, _lo2) = layout.contiguous_offsets().unwrap();
        let lcuda = match &*storage {
            crate::Storage::Cuda(c) => c,
            _ => unreachable!(),
        };
        let lslice = match &lcuda.slice {
            crate::cuda_backend::CudaStorageSlice::BF16(s) => s.slice(lo1..),
            _ => unreachable!(),
        };
        let stream = dev.cuda_stream();
        let (lptr, _lg) = lslice.device_ptr(&stream);
        let out_idx = unsafe { dev.alloc::<u32>(nt * k)? };
        let out_w = unsafe { dev.alloc::<f32>(nt * k)? };
        let (iptr, _ig) = out_idx.device_ptr(&stream);
        let (wptr, _wg) = out_w.device_ptr(&stream);

        let mut full = || -> Result<()> {
            let _ = moe_route(&logits, k, true)?;
            Ok(())
        };
        let mut rawkern = || -> Result<()> {
            unsafe {
                candle_kernels::simple::moe_scatter::run_moe_route(
                    2,
                    lptr as *const std::ffi::c_void,
                    iptr as *mut u32,
                    wptr as *mut f32,
                    nt as i32,
                    n_experts as i32,
                    n_experts as i32,
                    k as i32,
                    1,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
            Ok(())
        };
        let mut alloc2 = || -> Result<()> {
            let a = unsafe { dev.alloc::<u32>(nt * k)? };
            let b = unsafe { dev.alloc::<f32>(nt * k)? };
            std::hint::black_box((&a, &b));
            Ok(())
        };
        let mut opchain = || -> Result<()> {
            let m = logits.max_keepdim(crate::D::Minus1)?;
            let e = logits.broadcast_sub(&m)?.exp()?;
            let s = e.sum_keepdim(crate::D::Minus1)?;
            let probs = e.broadcast_div(&s)?.to_dtype(crate::DType::F32)?;
            let (sw, si) = probs.sort_last_dim(false)?;
            let tw = sw.narrow(1, 0, k)?;
            let _ti = si.narrow(1, 0, k)?.contiguous()?;
            let sums = tw.sum(1)?;
            let _ = tw.broadcast_div(&sums.unsqueeze(1)?)?;
            Ok(())
        };

        let f_us = bench(&mut full)? * 1e6;
        let r_us = bench(&mut rawkern)? * 1e6;
        let a_us = bench(&mut alloc2)? * 1e6;
        let o_us = bench(&mut opchain)? * 1e6;
        println!("{nt:>10} {f_us:>13.3} {r_us:>13.3} {a_us:>13.3} {o_us:>13.3}");
    }
    Ok(())
}

/// **The look-ahead's vote word puts every routed pick above every margin pick.**
/// Two tokens over eight experts, one hop, `k = 2`, `n = 4`. Token 0's logits
/// descend, so it routes e0, e1 and reaches e2, e3 in the margin; token 1 routes
/// e0, e2 and reaches e3, e1. Each word is `routed << 16 | margin`, raw:
/// e0 routed twice; e1 and e2 routed once and reached once; e3 reached twice;
/// the rest nothing.
#[test]
pub(super) fn cuda_moe_predict_votes_counts_routed_picks_above_margin_picks() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let logits: Vec<f32> = vec![
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, // token 0
        8.0, 1.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0, // token 1
    ];
    let rows = crate::Tensor::from_vec(logits, (2, 8), &device)?;
    let out = dev.alloc_zeros::<u32>(8)?;
    let mass = dev.alloc_zeros::<f32>(8)?;
    let stream = dev.cuda_stream();
    let (op, _g) = out.device_ptr(&stream);
    let (mp, _gm) = mass.device_ptr(&stream);
    moe_predict_votes(&rows, 0, 1, 8, 2, 4, op, mp)?;
    assert_eq!(
        dev.memcpy_dtov(&out)?,
        vec![2 << 16, (1 << 16) | 1, (1 << 16) | 1, 2, 0, 0, 0, 0]
    );
    // Margin width zero is the plain top-k count, one routed pick per token.
    let plain = dev.alloc_zeros::<u32>(8)?;
    let (pp, _g2) = plain.device_ptr(&stream);
    moe_predict_votes(&rows, 0, 1, 8, 2, 2, pp, mp)?;
    assert_eq!(
        dev.memcpy_dtov(&plain)?,
        vec![2 << 16, 1 << 16, 1 << 16, 0, 0, 0, 0, 0]
    );
    Ok(())
}

/// **Beside each vote word, the softmax probability its picks carried.** Two
/// tokens over eight experts, `k = 2`, `n = 4`, with logits of 0 or −∞ so every
/// probability is exact: token 0 spreads evenly over e0–e3 (0.25 each, routing
/// e0, e1 and reaching e2, e3), token 1 over e0 and e2 (0.5 each, routing both
/// and naming nothing past its two finite experts). Raw words and masses.
#[test]
pub(super) fn cuda_moe_predict_votes_sums_each_picks_router_probability() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let x = f32::NEG_INFINITY;
    let logits: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.0, x, x, x, x, // token 0
        0.0, x, 0.0, x, x, x, x, x, // token 1
    ];
    let rows = crate::Tensor::from_vec(logits, (2, 8), &device)?;
    let out = dev.alloc_zeros::<u32>(8)?;
    let mass = dev.alloc_zeros::<f32>(8)?;
    let stream = dev.cuda_stream();
    let (op, _g) = out.device_ptr(&stream);
    let (mp, _gm) = mass.device_ptr(&stream);
    moe_predict_votes(&rows, 0, 1, 8, 2, 4, op, mp)?;
    assert_eq!(
        dev.memcpy_dtov(&out)?,
        vec![2 << 16, 1 << 16, (1 << 16) | 1, 1, 0, 0, 0, 0]
    );
    assert_eq!(
        dev.memcpy_dtov(&mass)?,
        vec![0.75, 0.25, 0.75, 0.25, 0.0, 0.0, 0.0, 0.0]
    );
    Ok(())
}
