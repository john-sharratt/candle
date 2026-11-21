//! Launch-cost probes for the grouped expert path, and the scatter's contract
//! that it defines its target.
//!
//! The grouped expert probes time the host cost of a call as the expert count
//! grows, once with a launch per segment and once with a single grouped launch,
//! so the cost of dispatch against N is visible. The scatter test fills its
//! target with poison. A kernel that seeds its sum from the target and passes
//! only over zeroed memory fails, and a row that no contribution writes must
//! still be stored as an explicit zero. The VRAM total probe must report a sane
//! size before any device exists.

use super::*;

// GROUPED EXPERT LAUNCH-COST PROBE (does N experts cost ~N launches?)
// The MoE pipeline dispatches all routed experts via a SINGLE run_quantized_matmul
// call with num_segments = N_experts — but the dispatcher's segment loop issues
// one cudaLaunchKernel PER expert. This probe measures per-call host time as N
// grows, with M=1 token per expert (single-session decode). If time scales ~N,
// the path is launch/host-bound and a single-launch grouped GEMM would cut it.
//
// All N segments point at the SAME weight (timing is pointer-independent), so
// this isolates pure dispatch/launch cost. No per-iter sync — launches queue,
// matching the real async path.
//
//   cargo test -p candle-core --release --features cuda --lib \
//     quantized::test::expert_grouped_launch_cost -- --ignored --nocapture
#[test]
#[ignore = "GPU launch-cost probe; run with --ignored --nocapture"]
pub(super) fn expert_grouped_launch_cost() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let n = 768usize;
    let k = 2048usize;

    let mut rng = rand::rng();
    let wvals: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let shape = Shape::from((n, k));
    let mut xs = QCudaStorage::zeros(&dev, n * k, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&wvals)?,
        dev.clone(),
    ))?;
    let xs_repacked = xs.repack_gemx(&shape)?;
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let m = 1usize; // tokens per expert (single-session decode)
    let iters = 1000usize;

    println!(
        "\n=== Per-segment expert launch cost [n={n} k={k}] Q4_K, M={m}/expert, \
         {iters} calls/sync (N launches via run_quantized_matmul) ===",
    );
    println!("{:>9} {:>12} {:>12}", "N_experts", "us/call", "us/expert");

    for &n_exp in &[1usize, 2, 4, 8, 16, 32] {
        let total = n_exp * m;
        let yvals: Vec<bf16> = (0..total * k)
            .map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0)))
            .collect();
        let y = dev.memcpy_stod(&yvals)?;
        let dst = unsafe { dev.alloc::<bf16>(total * n)? };

        let stream = dev.cuda_stream();
        let (wptr, _wg) = xs_repacked.data.inner.device_ptr(&stream);
        let (yptr, _yg) = y.device_ptr(&stream);
        let (dptr, _dg) = dst.device_ptr(&stream);

        // One segment per expert, all pointing at the same weight.
        let segments: Vec<VxSegment> = (0..n_exp)
            .map(|_| VxSegment {
                weights: wptr as *const c_void,
                batch_count: m as i32,
            })
            .collect();

        let call = || unsafe {
            run_quantized_matmul(
                segments.as_ptr(),
                n_exp as i32,
                yptr as *const c_void,
                dptr as *mut c_void,
                k as i32,
                n as i32,
                k as i32,
                n as i32,
                qtype,
                YType::BF16 as i32,
                0, // weight_bytes=0 → L2-cached assumption (matches grouped_matmul_gemx)
                0, // tile_mode (tiling only; result-invariant)
                OutDType::BF16 as i32,
                SumScale::Raw.as_code(),
                stream.cu_stream() as *mut c_void,
            );
        };

        for _ in 0..100 {
            call();
        }
        dev.synchronize()?;

        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                call();
            }
            dev.synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        println!(
            "{:>9} {:>12.3} {:>12.3}",
            n_exp,
            best * 1e6,
            best * 1e6 / n_exp as f64,
        );
    }
    Ok(())
}

// GROUPED SINGLE-LAUNCH COST (run_grouped_quantized_matmul: N experts, 1 launch)
// A/B partner for expert_grouped_launch_cost: same N-expert / M=1 sweep, but
// all experts run in ONE grouped kernel launch instead of N. Compare us/call
// directly against the per-segment probe to measure the launch+occupancy win.
//
//   cargo test -p candle-core --release --features cuda --lib \
//     quantized::test::expert_grouped_single_launch_cost -- --ignored --nocapture
#[test]
#[ignore = "GPU launch-cost probe; run with --ignored --nocapture"]
pub(super) fn expert_grouped_single_launch_cost() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let n = 768usize;
    let k = 2048usize;

    let mut rng = rand::rng();
    let wvals: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let shape = Shape::from((n, k));
    let mut xs = QCudaStorage::zeros(&dev, n * k, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&wvals)?,
        dev.clone(),
    ))?;
    let xs_repacked = xs.repack_gemx(&shape)?;
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let m = 1usize;
    let iters = 1000usize;

    println!(
        "\n=== Grouped expert SINGLE-LAUNCH cost [n={n} k={k}] Q4_K, M={m}/expert, \
         {iters} calls/sync ===",
    );
    println!("{:>9} {:>12} {:>12}", "N_experts", "us/call", "us/expert");

    let stream = dev.cuda_stream();
    let (wptr0, _wg0) = xs_repacked.data.inner.device_ptr(&stream);

    for &n_exp in &[1usize, 2, 4, 8, 16, 32] {
        let total = n_exp * m;
        let yvals: Vec<bf16> = (0..total * k)
            .map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0)))
            .collect();
        let y = dev.memcpy_stod(&yvals)?;
        let dst = unsafe { dev.alloc::<bf16>(total * n)? };

        // One tile per expert (M=1 <= 16). All experts share the same weight ptr.
        let weight_ptrs: Vec<u64> = vec![wptr0 as u64; n_exp];
        let tile_expert: Vec<i32> = (0..n_exp as i32).collect();
        let tile_b_start: Vec<i32> = (0..n_exp as i32).map(|e| e * m as i32).collect();
        let tile_b_cnt: Vec<i32> = vec![m as i32; n_exp];
        let num_tiles = n_exp as i32;
        let (yptr, _yg) = y.device_ptr(&stream);
        let (dptr, _dg) = dst.device_ptr(&stream);

        // Honest end-to-end cost: PACK weight_ptrs + all 3 tile tables into ONE
        // buffer and upload with a SINGLE memcpy_stod per call (vs 4 separate
        // copies), then the single grouped launch. weight_ptrs (u64, 8-aligned)
        // first, then the three i32 tables — base + offsets feed the kernel.
        let off_te = n_exp * 8; // bytes: weight_ptrs = n_exp × u64
        let off_tbs = off_te + num_tiles as usize * 4;
        let off_tbc = off_tbs + num_tiles as usize * 4;
        let total_bytes = off_tbc + num_tiles as usize * 4;
        let once = || -> Result<()> {
            let mut packed: Vec<u8> = Vec::with_capacity(total_bytes);
            for &w in &weight_ptrs {
                packed.extend_from_slice(&w.to_le_bytes());
            }
            for &x in &tile_expert {
                packed.extend_from_slice(&x.to_le_bytes());
            }
            for &x in &tile_b_start {
                packed.extend_from_slice(&x.to_le_bytes());
            }
            for &x in &tile_b_cnt {
                packed.extend_from_slice(&x.to_le_bytes());
            }
            let dev_buf = dev.memcpy_stod(&packed)?;
            let (base, _g) = dev_buf.device_ptr(&stream);
            let status = unsafe {
                run_grouped_quantized_matmul(
                    base as *const c_void,
                    (base + off_te as u64) as *const c_void,
                    (base + off_tbs as u64) as *const c_void,
                    (base + off_tbc as u64) as *const c_void,
                    yptr as *const c_void,
                    dptr as *mut c_void,
                    k as i32,
                    n as i32,
                    k as i32,
                    n as i32,
                    num_tiles,
                    qtype,
                    YType::BF16 as i32,
                    2, // FP grouped kernels ignore the int8 tile mode
                    1, // row-fast grid order
                    SumScale::Raw.as_code(),
                    std::ptr::null(),
                    stream.cu_stream() as *mut c_void,
                )
            };
            assert_eq!(status, 0, "grouped matmul launcher rejected the call");
            Ok(())
        };

        for _ in 0..100 {
            once()?;
        }
        dev.synchronize()?;

        let mut best = f64::MAX;
        for _ in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                once()?;
            }
            dev.synchronize()?;
            best = best.min(t0.elapsed().as_secs_f64() / iters as f64);
        }
        println!(
            "{:>9} {:>12.3} {:>12.3}",
            n_exp,
            best * 1e6,
            best * 1e6 / n_exp as f64,
        );
    }
    Ok(())
}

/// The context-free total-VRAM probe must succeed and report a sane size.
/// (In-process test order may mean a context already exists on this thread;
/// the no-context property is exercised for real by zend's download path,
/// which calls this before any `Device` is created.)
#[test]
pub(super) fn cuda_total_vram_device0_probe() -> Result<()> {
    let total = get_total_vram_device0()?;
    assert!(
        total > 1024 * 1024 * 1024,
        "reported total VRAM {total} bytes is implausibly small"
    );
    Ok(())
}

/// The deterministic MoE scatter **defines** its target; it does not accumulate
/// into it.
///
/// This is the contract that lets every caller allocate the combine target with
/// `Tensor::empty` (hot-path invariant 6) instead of paying a memset over the
/// layer's largest transient — and, on the kernel side, lets it skip a full read
/// of that target to add a value that is always zero.
///
/// **The target is filled with poison, not zeros.** That is the whole test. A
/// kernel that seeds its reduction from `ys` — which this one did, back when the
/// callers ran it twice per layer for hits and then misses — passes any test
/// that pre-zeroes the target, and silently adds garbage to every expert output
/// the day a caller stops zeroing. Poison makes the two behaviours differ by the
/// poison value, and the assertion below is exact: the scatter's arithmetic is
/// one `w · v` product per contribution, and the expected value is computed the
/// same way in the same order, so there is nothing here to tolerance.
#[test]
pub(super) fn cuda_deterministic_scatter_defines_its_target_rather_than_accumulating() -> Result<()>
{
    use crate::cuda_backend::Backing;
    use crate::quantized::cuda::fused_deterministic_scatter;
    use crate::{DType, Device, Tensor};

    let dev = CudaDevice::new(0)?;
    let device = Device::Cuda(dev.clone());

    // Three tokens with deliberately UNEQUAL contribution counts (2, 3, 0), so
    // the test covers the variable-k prefix-sum path and, in token 2, the row
    // that no contribution writes — the case where "defines its target" and
    // "accumulates into a zeroed target" agree only if the kernel stores an
    // explicit zero rather than leaving the row alone.
    const HIDDEN: usize = 5;
    const NUM_TOKENS: usize = 3;
    let token_starts: Vec<i32> = vec![0, 2, 5, 5];
    // token-major row order; each entry names a row of `down_out` and a weight.
    let perm: Vec<u32> = vec![0, 1, 2, 3, 4];
    let rw_ids: Vec<u32> = vec![0, 1, 2, 3, 4];
    let weights: Vec<f32> = vec![0.5, 0.25, 2.0, -1.0, 0.125];

    let down_rows = perm.len();
    let down: Vec<f32> = (0..down_rows * HIDDEN).map(|i| (i as f32) + 1.0).collect();

    // What the kernel must produce, from the same products in the same order.
    let mut want = [0f32; NUM_TOKENS * HIDDEN];
    for t in 0..NUM_TOKENS {
        for col in 0..HIDDEN {
            let mut sum = 0f32;
            for idx in token_starts[t] as usize..token_starts[t + 1] as usize {
                sum += weights[rw_ids[idx] as usize] * down[perm[idx] as usize * HIDDEN + col];
            }
            want[t * HIDDEN + col] = sum;
        }
    }

    let down_out = Tensor::from_vec(down, (down_rows, HIDDEN), &device)?;
    let weights_flat = Tensor::from_vec(weights.clone(), (weights.len(),), &device)?;
    let perm_dev = dev.memcpy_stod_from(&perm, Backing::Owned)?;
    let rw_dev = dev.memcpy_stod_from(&rw_ids, Backing::Owned)?;
    let ts_dev = dev.memcpy_stod_from(&token_starts, Backing::Owned)?;

    // Poison. Any nonzero pattern works; this one is large enough that a stale
    // read could not be mistaken for rounding and negative on alternating
    // elements so a sign error shows too.
    let poison: Vec<f32> = (0..NUM_TOKENS * HIDDEN)
        .map(|i| if i % 2 == 0 { 1.0e6 } else { -7.5e5 })
        .collect();

    for dtype in [DType::F32, DType::BF16] {
        let ys =
            Tensor::from_vec(poison.clone(), (NUM_TOKENS, HIDDEN), &device)?.to_dtype(dtype)?;
        fused_deterministic_scatter(
            &ys,
            &down_out,
            &perm_dev,
            &weights_flat,
            &rw_dev,
            &ts_dev,
            NUM_TOKENS,
            &dev,
        )?;
        let got = ys.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            // BF16 storage rounds the store; F32 must be exact.
            let ok = if dtype == DType::F32 {
                g == w
            } else {
                (g - w).abs() <= 1e-2 * w.abs().max(1.0)
            };
            assert!(
                ok,
                "{dtype:?} element {i} (token {}, col {}): got {g}, want {w} — a nonzero gap of \
                 roughly the poison value means the kernel seeded its reduction from `ys`, and \
                 every caller allocating the target with `Tensor::empty` is now returning garbage",
                i / HIDDEN,
                i % HIDDEN,
            );
        }
    }
    Ok(())
}
