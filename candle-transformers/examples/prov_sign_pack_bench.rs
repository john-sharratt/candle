//! Individual-kernel microbench + ncu target for the provenance sign(Q) bit-pack
//! (`prov_sign_pack_kernel`).
//!
//! The kernel reads R16 Q — co-located in the K arena chunk at `+64` within each
//! 128-byte dim group — signs it, and packs one word of sign bits per
//! (sub-band, token). This drives it over a synthetic arena laid out exactly as
//! `gather_r16_kv` writes one, so the access pattern under measurement is the
//! production one: for fixed `d`, the warp's 32 tokens are 2 B apart (64 B
//! contiguous), and consecutive `d` are 128 B apart.
//!
//! **Both production band widths are swept**, because they are not the same
//! kernel in cost terms: `head_dim` 128 gives 32-dim bands (the outgoing 30B),
//! and `head_dim` 256 gives 64-dim bands (the incoming hybrid) — twice the loop
//! and twice the input traffic per warp against an output that does not grow.
//! The word is `u64` so a 64-dim band packs whole; the sweep is what says what
//! that costs the 32-dim case, which pays a wider store for bits it does not use.
//!
//! Usage:
//!   cargo run -p candle-transformers --example prov_sign_pack_bench \
//!       --features cuda --release -- [n_warps] [iters]
//!
//! Profile just the kernel (occupancy, registers, achieved bandwidth):
//!   ncu --kernel-name "regex:prov_sign_pack_kernel" --launch-count 8 --set full \
//!       target/release/examples/prov_sign_pack_bench 12288 200

#[cfg(feature = "cuda")]
fn main() -> candle::Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::cuda_backend::kernels;
    use candle::Device;

    const CHUNK: usize = 32; // CHUNK_SIZE — tokens per warp
    const DIM_GROUP: usize = 128; // bytes per dim group (K in [0,64), Q in [64,128))

    let a: Vec<String> = std::env::args().collect();
    let parse = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    // Default sized like a real seal: 48 layers × 4 KV heads × 4 palettes × 16
    // R16 blocks = 12288 warps.
    let n_warps = parse(1, 12288);
    let iters = parse(2, 200);
    let warmup = 20;

    let device = Device::new_cuda(0)?;
    let Device::Cuda(dev) = &device else {
        candle::bail!("cuda device required")
    };
    let stream = dev.cuda_stream();

    println!(
        "  {:>5} {:>9} {:>11} {:>10} {:>11} {:>10} {:>10}",
        "sub", "n_warps", "in MiB", "out MiB", "kernel µs", "eff GB/s", "ns/warp"
    );

    for sub in [32usize, 64] {
        // One warp's source span: `sub` dim groups of 128 B. Only the Q half of
        // each group is read, which is the production layout, not a padding
        // artefact — K lives in the other half and this kernel never touches it.
        let warp_src_bytes = sub * DIM_GROUP;
        let total_src = n_warps * warp_src_bytes;
        let out_len = n_warps * CHUNK;

        // Fill with a deterministic f16 bit pattern. The sign distribution does
        // not change the work — every dim is read and tested regardless — so the
        // content only has to be legal f16, never NaN-heavy enough to matter.
        let mut host = vec![0u8; total_src];
        for (i, b) in host.iter_mut().enumerate() {
            *b = ((i * 37 + 11) & 0xFF) as u8;
        }

        let mut src = unsafe { dev.alloc::<u8>(total_src)? };
        dev.memcpy_htod(&host, &mut src.slice_mut(..))?;
        let out = unsafe { dev.alloc::<u64>(out_len)? };

        // Per-warp base addresses, exactly as `resolve_provenance_q_ptrs` hands
        // them over: one device pointer per (layer, block, head, palette).
        let base = {
            let (p, _g) = src.device_ptr(&stream);
            p as i64
        };
        let ptrs_host: Vec<i64> = (0..n_warps)
            .map(|w| base + (w * warp_src_bytes) as i64)
            .collect();
        let mut ptrs = unsafe { dev.alloc::<i64>(n_warps)? };
        dev.memcpy_htod(&ptrs_host, &mut ptrs.slice_mut(..))?;

        let ptrs_view = ptrs.slice(..);
        let out_view = out.slice(..);
        let launch = || {
            let (pp, _pg) = ptrs_view.device_ptr(&stream);
            let (op, _og) = out_view.device_ptr(&stream);
            unsafe {
                kernels::simple::prov_sign_pack::run_prov_sign_pack(
                    pp as *const i64,
                    op as *mut std::ffi::c_void,
                    n_warps as i32,
                    sub as i32,
                    stream.cu_stream() as *mut _,
                );
            }
        };

        for _ in 0..warmup {
            launch();
        }
        stream.synchronize().map_err(candle::Error::wrap)?;

        // One sync for the whole batch, not per iteration: a per-iteration sync
        // would measure WDDM launch latency rather than the kernel.
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch();
        }
        stream.synchronize().map_err(candle::Error::wrap)?;
        let per_iter_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // Bytes the kernel must actually move: the Q half of each dim group in,
        // one word per (warp, token) out. The K half is never read, so counting
        // the full 128 B group would overstate the traffic the kernel is
        // responsible for.
        let in_bytes = n_warps * sub * CHUNK * 2;
        let out_bytes = out_len * 8;
        let gbps = (in_bytes + out_bytes) as f64 / (per_iter_us * 1e-6) / 1e9;

        println!(
            "  {sub:>5} {n_warps:>9} {:>11.1} {:>10.2} {per_iter_us:>11.1} {gbps:>10.1} {:>10.2}",
            in_bytes as f64 / (1 << 20) as f64,
            out_bytes as f64 / (1 << 20) as f64,
            per_iter_us * 1e3 / n_warps as f64,
        );
    }

    println!(
        "\n  in = Q bytes read (sub × 32 tokens × 2 B per warp); out = one u64 per (warp, token).\n  \
         eff GB/s counts only those bytes — the K half of each 128 B dim group is never read."
    );
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("prov_sign_pack_bench requires --features cuda");
}
