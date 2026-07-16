//! Standalone GPU harness + micro-benchmark for the `prov_sign_pack` kernel.
//!
//! Builds a synthetic R16-chunk-layout device buffer (Q at +64 within each
//! 128-byte dim group, exactly as the paged arena stores it), launches the
//! kernel, validates every packed sign bit against a CPU reference, then times
//! it over many iterations. The `n_warps` default (~61k) matches a wave-batched
//! launch (≈20 scopes × 48 layers × 4 blocks × 4 heads × 4 palettes).
//!
//! Run (CUDA):
//!   cargo run -p candle-nn --example prov_sign_pack_bench --features cuda --release -- [n_warps] [iters]
//! Profile:
//!   "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\ncu.bat" --set full \
//!     target\release\examples\prov_sign_pack_bench.exe 61440 1

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("prov_sign_pack_bench requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels;
    use candle::Device;
    use half::f16;
    use std::time::Instant;

    const CHUNK_SIZE: usize = 32;
    const SUB_HEAD_DIM: usize = 32; // head_dim/N_PALETTE = 128/4
    const F16_PER_CHUNK: usize = SUB_HEAD_DIM * 64; // 128 bytes/dim-group = 64 f16
    const CHUNK_BYTES: i64 = (SUB_HEAD_DIM * 128) as i64;

    let mut args = std::env::args().skip(1);
    let n_warps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(61_440);
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(200);

    let dev = Device::new_cuda(0)?;
    let Device::Cuda(cuda) = &dev else {
        return Err(candle::Error::Msg("no CUDA device".into()));
    };

    // Deterministic sign for (warp, dim, token): true = positive (bit set).
    let sgn = |w: usize, d: usize, t: usize| -> bool {
        let x = ((((w * 131 + d) * 131 + t) as u64).wrapping_mul(2654435761)) >> 13;
        x & 1 == 0
    };

    // Host buffer laid out as `n_warps` R16 chunks; only Q (f16 index d*64 + 32 + t)
    // is meaningful to the kernel. K half left zero.
    let mut host = vec![f16::from_f32(0.0); n_warps * F16_PER_CHUNK];
    for w in 0..n_warps {
        for d in 0..SUB_HEAD_DIM {
            for t in 0..CHUNK_SIZE {
                let v = if sgn(w, d, t) { 1.0 } else { -1.0 };
                host[w * F16_PER_CHUNK + d * 64 + 32 + t] = f16::from_f32(v);
            }
        }
    }

    let buf = cuda.memcpy_stod(&host)?;
    let stream = cuda.cuda_stream();
    let (base, _g) = buf.device_ptr(&stream);
    let q_ptrs: Vec<i64> = (0..n_warps)
        .map(|w| base as i64 + w as i64 * CHUNK_BYTES)
        .collect();
    let q_ptrs_gpu = cuda.memcpy_stod(&q_ptrs)?;
    let out_gpu = unsafe { cuda.alloc::<u32>(n_warps * CHUNK_SIZE)? };

    let launch = || -> candle::Result<()> {
        let (pp, _a) = q_ptrs_gpu.device_ptr(&stream);
        let (op, _b) = out_gpu.device_ptr(&stream);
        unsafe {
            kernels::simple::prov_sign_pack::run_prov_sign_pack(
                pp as *const i64,
                op as *mut std::ffi::c_void,
                n_warps as i32,
                SUB_HEAD_DIM as i32,
                stream.cu_stream() as *mut _,
            );
        }
        Ok(())
    };

    // ── Correctness ──────────────────────────────────────────────────────────
    launch()?;
    let out: Vec<u32> = cuda.memcpy_dtov(&out_gpu)?;
    let mut bad = 0usize;
    for w in 0..n_warps {
        for t in 0..CHUNK_SIZE {
            let mut want = 0u32;
            for d in 0..SUB_HEAD_DIM {
                if sgn(w, d, t) {
                    want |= 1u32 << d;
                }
            }
            if out[w * CHUNK_SIZE + t] != want {
                bad += 1;
            }
        }
    }
    if bad != 0 {
        return Err(candle::Error::Msg(format!(
            "prov_sign_pack: {bad} packed words mismatched the CPU reference"
        )));
    }
    println!("correctness OK: {n_warps} warps × {CHUNK_SIZE} tokens, all bits match");

    // ── Benchmark ────────────────────────────────────────────────────────────
    // Warm up, then time `iters` launches wall-clock around a single sync (ncu
    // reports the precise per-kernel numbers).
    for _ in 0..10 {
        launch()?;
    }
    stream.synchronize().map_err(candle::Error::wrap)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        launch()?;
    }
    stream.synchronize().map_err(candle::Error::wrap)?;
    let elapsed = t0.elapsed();
    let per = elapsed.as_secs_f64() * 1e6 / iters as f64;
    let bytes_in = (n_warps * SUB_HEAD_DIM * CHUNK_SIZE * 2) as f64; // Q f16 read
    let gbps = bytes_in / (per * 1e-6) / 1e9;
    println!(
        "bench: {n_warps} warps, {iters} iters → {per:.1} µs/launch  \
         ({:.1} GB/s Q-read, out {} KB)",
        gbps,
        n_warps * CHUNK_SIZE * 4 / 1024,
    );
    Ok(())
}
