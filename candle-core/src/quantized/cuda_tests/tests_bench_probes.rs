//! Profiler probes and benches for the int8 crossover and pinned-read bandwidth.
//!
//! Each probe launches one kernel a few times at a fixed shape without timing,
//! so Nsight Compute can profile a single named launch. The crossover scans
//! find the M at which the Q4_KO kernel turns from weight-bound to
//! activation-bound. The pinned-read bench measures which access shape reaches
//! the 128-byte request size a host read needs, and the decode bench reports
//! the grouped expert GEMM against the bytes of the distinct experts it reads,
//! so its figure can be set against the DRAM floor.

use super::*;

/// Finer-grained M scan of the Q4_KO int8 matmul. Reports the activation element count
/// (M·K), the weight count (N·K), their ratio (= M/N), and the i8KO time, so the M where
/// the kernel turns from weight-bound to activation-bound reads off the table.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
pub(super) fn q4_crossover_scan() -> Result<()> {
    use std::time::Instant;
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let ob = requant_ko_per128(&wf32, nrows, ncols, 15, 0, 0); // Q4_KO
    let ko_slice = dev.memcpy_stod(&ob)?;
    let stream = dev.cuda_stream();
    let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
    let weight_count = nrows * ncols;
    println!("=== Q4 crossover scan N={nrows} K={ncols} weight_count={weight_count} ===");
    println!("     M    act_count   act/wt   i8KO(ms)   i8KO-tok/s");
    for &m in &[
        16usize, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 1024,
    ] {
        let act: Vec<f32> = (0..m * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let q8 = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;
        let time_i8 = || -> Result<f64> {
            for _ in 0..20 {
                let _ = dense_qmatmul(
                    DynamicTensor::Int8(&q8),
                    ko_ptr,
                    GgmlDType::Q4_KO,
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
                        ko_ptr,
                        GgmlDType::Q4_KO,
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
        let t = time_i8()?;
        let act_count = m * ncols;
        let ratio = act_count as f64 / weight_count as f64;
        println!(
            "{m:>6} {act_count:>12} {ratio:>8.3} {:>10.4} {:>12.0}",
            t * 1e3,
            m as f64 / t
        );
    }
    Ok(())
}

/// ncu probe: launches each KO int8 DENSE kernel (`q{4,5,6,8}_ko_int8_f32_dense`) once at the
/// big-M prefill config (m=4096) so Nsight Compute can read each kernel's occupancy/resource
/// limits in isolation — this is the shape where KO regresses vs FP16 (~0.87×). No timing;
/// clean launches for `ncu -k regex:_ko_int8_f32_dense -c 4 --section Occupancy`.
#[test]
#[ignore]
pub(super) fn ko_dense_ncu_probe() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize; // N
    let ncols = 2048usize; // K
    let m = 4096usize; // big-M prefill — the regressing shape
    let mut rng = rand::rng();
    let wf32: Vec<f32> = (0..nrows * ncols)
        .map(|_| rng.random_range(-0.1f32..0.1))
        .collect();
    let act: Vec<f32> = (0..m * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let q8 = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;
    // (dtype, maxq, crumb_bytes, hi_bytes) — Q8 uses its own symmetric per-128 producer.
    for &(kod, maxq, crumb, hi) in &[
        (GgmlDType::Q4_KO, 15i32, 0usize, 0usize),
        (GgmlDType::Q5_KO, 31, 0, 128),
        (GgmlDType::Q6_KO, 63, 256, 0),
        (GgmlDType::Q8_KO, 255, 0, 0),
    ] {
        let ob = if kod == GgmlDType::Q8_KO {
            requant_q8_ko_per128(&wf32, nrows, ncols)
        } else {
            requant_ko_per128(&wf32, nrows, ncols, maxq, crumb, hi)
        };
        let ko_slice = dev.memcpy_stod(&ob)?;
        let stream = dev.cuda_stream();
        let (ko_ptr, _g) = ko_slice.device_ptr(&stream);
        for _ in 0..3 {
            let _ = dense_qmatmul(
                DynamicTensor::Int8(&q8),
                ko_ptr,
                kod,
                nrows,
                0,
                crate::DType::F32,
                &dev,
            )?;
        }
        dev.synchronize()?;
    }
    Ok(())
}

/// ncu probe: launches the Q5_K FP16 grouped kernel (`q5_k_f32_grouped`) and the
/// Q5_KO int8 grouped kernel (`q5_ko_int8_f32_grouped`) a handful of times at the
/// 4096-row config so Nsight Compute can profile each in isolation by kernel name.
/// No timing — just clean, repeatable launches for `ncu -k regex:... -c N`.
#[test]
#[ignore]
pub(super) fn q5_ncu_probe() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let nrows = 768usize;
    let ncols = 2048usize;
    let num_experts = 128usize;
    let tokens_per_expert = 32usize;
    let total = num_experts * tokens_per_expert;
    let stride = 112usize; // Q5_K compact block
    let mut rng = rand::rng();
    let kd = GgmlDType::Q5_K;
    let kod = GgmlDType::Q5_KO;
    let perm: fn(&[u8], &mut [u8]) = |kb: &[u8], ob: &mut [u8]| {
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
    };
    let shape = crate::Shape::from((nrows, ncols));
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

    // A few launches of each kernel; ncu filters by name and profiles -c N of them.
    for _ in 0..6 {
        let _ = grouped_matmul_gemx(
            &k_ptrs,
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
            &ko_ptrs,
            kod,
            nrows,
            &expert_offsets,
            &dev,
            Backing::Owned,
        )?;
    }
    dev.synchronize()?;
    drop(k_pool);
    drop(ko_pool);
    Ok(())
}

/// **The grouped expert GEMM at Qwen3.8-Flash-Next's decode shapes, against the bytes it must
/// read.** One projection (640 × 2560, Q4_KO) over 256 VRAM-resident experts — 222 MB, past
/// the 96 MiB L2 — routed `rows × top-10` from eight rotating expert sets, each bucketized once
/// outside the timing. The timed loop is the GEMM alone, so the figure is device time per
/// launch, and its GB/s is the distinct experts' weight bytes over that time: how far the
/// decode-width launch sits from the DRAM floor.
#[test]
#[ignore = "measurement: cargo test --release --features cuda -p candle-core --lib \
            quantized::cuda::test::bench_grouped_gemm_flash_next_decode -- \
            --exact --ignored --nocapture"]
pub(super) fn bench_grouped_gemm_flash_next_decode() -> Result<()> {
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::HashSet;
    use std::time::Instant;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (n_experts, nrows, ncols, k) = (256usize, 640usize, 2560usize, 10usize);
    let ko_dtype = GgmlDType::Q4_KO;
    let mut rng = StdRng::seed_from_u64(0xf1a5_0001);
    let shape = crate::Shape::from((nrows, ncols));
    let mut storages = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut q = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q4_K)?;
        q.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&w)?,
            dev.clone(),
        ))?;
        storages.push(q.repack_ko(&shape, ko_dtype)?);
    }
    let expert_bytes = storages[0].storage_size_in_bytes();
    let ptrs: Vec<u64> = storages.iter().map(|s| s.data_ptr()).collect();
    let table = dev.memcpy_stod(&ptrs)?;
    println!(
        "Flash-Next expert projection {nrows}×{ncols} {ko_dtype:?}: {expert_bytes} B, \
         {n_experts} experts ({:.0} MB)",
        (n_experts * expert_bytes) as f64 / 1e6
    );
    println!(
        "{:>5} {:>9} {:>8} {:>9} {:>9}",
        "rows", "experts", "n_sub", "µs", "GB/s"
    );
    const SETS: usize = 8;
    for rows in [1usize, 5, 8, 16, 48] {
        let a_ub = rows * k;
        let n_sub = grouped_int8_n_sub(a_ub / n_experts, &[ko_dtype, ko_dtype]);
        let tile_w = 16 * n_sub;
        let launch_tiles = a_ub.min(a_ub.div_ceil(tile_w) + n_experts);
        let act: Vec<f32> = (0..a_ub * ncols)
            .map(|i| ((i % 97) as f32 - 48.0) / 50.0)
            .collect();
        let op = quantize_acts_q8a128_test(&dev, &act, a_ub, ncols)?;
        let mut sets = Vec::with_capacity(SETS);
        let mut distinct = 0usize;
        for s in 0..SETS {
            // Each set routes inside its own 32-expert window, so consecutive launches read
            // disjoint weights and none is served from the L2 the previous one filled.
            let base = (s * 32) as u32;
            let ids: Vec<u32> = (0..rows)
                .flat_map(|_| {
                    let mut chosen = HashSet::new();
                    while chosen.len() < k {
                        chosen.insert(base + rng.random_range(0..32u32));
                    }
                    chosen.into_iter().collect::<Vec<_>>()
                })
                .collect();
            distinct += ids.iter().collect::<HashSet<_>>().len();
            let t = crate::Tensor::from_vec(ids, (rows, k), &device)?;
            let mut ws = MoeBucketizeWorkspace::new(&dev, rows, k)?;
            moe_bucketize(
                &t,
                n_experts,
                tile_w,
                &mut ws,
                None,
                &DecodeRows::prefix(rows),
            )?;
            sets.push(ws);
        }
        let launch = |ws: &MoeBucketizeWorkspace| {
            grouped_qmatmul_dev_q8a128(
                &op,
                &table,
                0,
                n_experts,
                ko_dtype,
                nrows,
                &ws.tile_expert,
                &ws.tile_b_start,
                &ws.tile_b_cnt,
                launch_tiles,
                n_sub,
                None,
                &dev,
            )
        };
        for ws in &sets {
            drop(launch(ws)?);
        }
        dev.synchronize()?;
        let iters = 400usize;
        let start = Instant::now();
        for i in 0..iters {
            drop(launch(&sets[i % SETS])?);
        }
        dev.synchronize()?;
        let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let mean_distinct = distinct as f64 / SETS as f64;
        println!(
            "{rows:>5} {mean_distinct:>9.1} {n_sub:>8} {us:>9.1} {:>9.0}",
            mean_distinct * expert_bytes as f64 / (us * 1e-6) / 1e9
        );
    }
    Ok(())
}

/// Probe kernels for [`bench_pinned_host_read_bandwidth`], compiled at test time.
///
/// `rd_*`: every thread reads 16 B per load, grid-stride, four independent loads
/// in flight per thread per iteration, XOR-reduced so nothing is dead code.
/// `rd_chunks_*`: the grouped GEMM's pattern — each warp reads one `chunk`-byte
/// span at a time, lane `l` taking 16-byte units `l, l + 32, …` of it — through
/// `cp.async.cg` into shared memory (as `load_warp_chunk_int8` does), with an L2
/// prefetch-size hint `hint` (0 none, 1 `.L2::128B`, 2 `.L2::256B`), either at the
/// chunk's own start or widened to whole 128-byte lines (`aligned`).
pub(super) const PINNED_READ_PROBE: &str = r#"
typedef unsigned long long u64;
__device__ __forceinline__ uint4 ld_def(const uint4* p) { return *p; }
__device__ __forceinline__ uint4 ld_nc(const uint4* p) { return __ldg(p); }
__device__ __forceinline__ uint4 ld_cg(const uint4* p) {
    uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ld_l2_128(const uint4* p) {
    uint4 v; asm volatile("ld.global.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ld_l2_256(const uint4* p) {
    uint4 v; asm volatile("ld.global.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ unsigned x4(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }

#define RD(NAME, LD) \
extern "C" __global__ void NAME(const uint4* __restrict__ src, u64 n16, unsigned* out) { \
    const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x; \
    const u64 s = (u64)gridDim.x * blockDim.x; \
    unsigned acc = 0; \
    for (u64 i = tid; i + 3 * s < n16; i += 4 * s) { \
        uint4 a = LD(src + i), b = LD(src + i + s), c = LD(src + i + 2 * s), d = LD(src + i + 3 * s); \
        acc ^= x4(a) ^ x4(b) ^ x4(c) ^ x4(d); \
    } \
    if (acc == 0x9e3779b9u) out[0] = acc; \
}
RD(rd_def, ld_def)
RD(rd_nc, ld_nc)
RD(rd_cg, ld_cg)
RD(rd_l2_128, ld_l2_128)
RD(rd_l2_256, ld_l2_256)

__device__ __forceinline__ void cp16(void* dst, const void* src, int hint) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    if (hint == 1) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" :: "r"(d), "l"(src));
    else if (hint == 2) asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" :: "r"(d), "l"(src));
    else asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src));
}

// blockDim = 128 (4 warps, as the GEMM); each warp owns a 2 KiB smem slot.
extern "C" __global__ void rd_chunks(const unsigned char* __restrict__ src, int chunk, u64 nchunks,
                                     int hint, int aligned, unsigned* out) {
    __shared__ __align__(16) unsigned char slot[4][2048];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const u64 warp = blockIdx.x * 4ull + w, nwarps = gridDim.x * 4ull;
    unsigned acc = 0;
    for (u64 c = warp; c < nchunks; c += nwarps) {
        u64 lo = c * (u64)chunk, hi = lo + chunk;
        if (aligned) { lo &= ~127ull; hi = (hi + 127) & ~127ull; }
        const int units = (int)((hi - lo) / 16);
        for (int i = lane; i < units; i += 32) cp16(&slot[w][i * 16], src + lo + i * 16, hint);
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncwarp();
        acc ^= reinterpret_cast<unsigned*>(slot[w])[lane];
        __syncwarp();
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

// Wide copy host → device, grid-stride, U loads in flight per thread before
// their stores.
#define CP(NAME, U) \
extern "C" __global__ void NAME(const uint4* __restrict__ src, uint4* __restrict__ dst, u64 n16) { \
    const u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x; \
    const u64 s = (u64)gridDim.x * blockDim.x; \
    for (u64 i = tid; i < n16; i += U * s) { \
        uint4 v[U]; \
        _Pragma("unroll") for (int u = 0; u < U; ++u) \
            if (i + u * s < n16) v[u] = ld_cg(src + i + u * s); \
        _Pragma("unroll") for (int u = 0; u < U; ++u) \
            if (i + u * s < n16) dst[i + u * s] = v[u]; \
    } \
}
CP(cp_wide_u1, 1)
CP(cp_wide_u4, 4)
CP(cp_wide_u8, 8)

// The grouped GEMM's own address walk over one 512 × 2048 Q6_KO projection per
// `expert` (819,200 B): block = (row tile 0..15, expert); per K block, warp w reads
// chunk `k · 64 + tile · 4 + w` (800 B) — `depth` K blocks in flight, each awaited
// in order with the block synchronised between K blocks, as the impl does.
extern "C" __global__ void rd_gemm_walk(const unsigned char* __restrict__ src, int depth,
                                        unsigned* out) {
    __shared__ __align__(16) unsigned char slot[4][8][800];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int tile = blockIdx.x & 15;
    const u64 base = (u64)(blockIdx.x >> 4) * 819200ull;
    unsigned acc = 0;
    auto issue = [&](int k) {
        const unsigned char* p = src + base + ((u64)k * 64 + tile * 4 + w) * 800ull;
        for (int i = lane; i < 50; i += 32) cp16(&slot[w][k & 7][i * 16], p + i * 16, 0);
        asm volatile("cp.async.commit_group;");
    };
    for (int k = 0; k < depth && k < 16; ++k) issue(k);
    for (int k = 0; k < 16; ++k) {
        if (depth >= 4) asm volatile("cp.async.wait_group 3;");
        else asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        acc ^= reinterpret_cast<unsigned*>(slot[w][k & 7])[lane];
        __syncthreads();
        if (k + depth < 16) issue(k + depth); else asm volatile("cp.async.commit_group;");
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}
"#;

/// **How fast can a kernel read pinned host memory on this box, and what moves
/// it?** (`docs/moe_live_dispatch_design.md` §0.10.) The grouped GEMM pulls
/// weights from pinned memory at ~5.3 GB/s against the copy engine's 12.75; the
/// literature (EMOGI, VLDB'21) puts the difference in PCIe request size: a warp's
/// access becomes one 128-byte read request only when it is merged and
/// 128-byte aligned. This measures the probe kernels above over a 256 MiB
/// `cuMemAllocHost` buffer.
#[test]
#[ignore = "measurement: cargo test --release --features cuda -p candle-core --lib \
            quantized::cuda::test::bench_pinned_host_read_bandwidth -- --exact --ignored --nocapture"]
pub(super) fn bench_pinned_host_read_bandwidth() -> Result<()> {
    use cudarc::driver::{sys, LaunchConfig, PushKernelArg};
    use std::time::Instant;

    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        PINNED_READ_PROBE,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            ..Default::default()
        },
    )
    .map_err(crate::Error::wrap)?;
    let module = dev.load_ptx("pinned_read_probe", &ptx)?;

    const BYTES: usize = 256 << 20;
    let mut host: *mut c_void = std::ptr::null_mut();
    let r = unsafe { sys::cuMemAllocHost_v2(&mut host, BYTES + 4096) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "pinned buffer");
    unsafe { std::ptr::write_bytes(host as *mut u8, 0x5a, BYTES + 4096) };
    let out = dev.alloc_zeros::<u32>(1)?;
    let (out_ptr, _go) = out.device_ptr(&stream);
    let base = host as u64;
    let gbps = |bytes: usize, us: f64| bytes as f64 / (us * 1e-6) / 1e9;

    // Times `f` (which launches one read of `bytes`): one warm-up, then 5.
    let time = |f: &dyn Fn() -> Result<()>, bytes: usize| -> Result<f64> {
        f()?;
        stream.synchronize().map_err(crate::Error::wrap)?;
        let start = Instant::now();
        for _ in 0..5 {
            f()?;
        }
        stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(gbps(bytes, start.elapsed().as_secs_f64() * 1e6 / 5.0))
    };

    println!("1. flat 16-byte loads, 256 threads/block, 4 in flight per thread (GB/s)");
    println!(
        "   {:>10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "load", "32 blk", "82", "164", "328", "656", "1312"
    );
    for name in ["rd_def", "rd_nc", "rd_cg", "rd_l2_128", "rd_l2_256"] {
        let f = module.load_function(name).map_err(crate::Error::wrap)?;
        let mut row = format!("   {name:>10}");
        for blocks in [32u32, 82, 164, 328, 656, 1312] {
            let n16 = (BYTES / 16) as u64;
            let g = time(
                &|| {
                    let mut b = stream.launch_builder(&f);
                    b.arg(&base).arg(&n16).arg(&out_ptr);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (blocks, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map(|_| ())
                    .map_err(crate::Error::wrap)
                },
                BYTES,
            )?;
            row += &format!(" {g:>7.2}");
        }
        println!("{row}");
    }

    println!("2. flat loads (rd_def, 656 blocks) from a misaligned base (GB/s)");
    {
        let f = module.load_function("rd_def").map_err(crate::Error::wrap)?;
        for off in [0u64, 16, 32, 64, 96] {
            let src = base + off;
            let n16 = (BYTES / 16) as u64;
            let g = time(
                &|| {
                    let mut b = stream.launch_builder(&f);
                    b.arg(&src).arg(&n16).arg(&out_ptr);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (656, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map(|_| ())
                    .map_err(crate::Error::wrap)
                },
                BYTES,
            )?;
            println!("   +{off:>3} B  {g:>7.2}");
        }
    }

    println!("3. the GEMM's pattern: one chunk per warp via cp.async, 128 threads/block (GB/s)");
    println!(
        "   {:>6} {:>8} {:>8} {:>7} {:>7} {:>7}",
        "chunk", "aligned", "hint", "16 blk", "82", "328"
    );
    let f = module
        .load_function("rd_chunks")
        .map_err(crate::Error::wrap)?;
    for chunk in [800i32, 896, 1024] {
        for aligned in [0i32, 1] {
            for hint in [0i32, 1, 2] {
                let mut row = format!(
                    "   {chunk:>6} {:>8} {:>8}",
                    if aligned == 1 { "yes" } else { "no" },
                    ["none", "L2:128B", "L2:256B"][hint as usize]
                );
                let nchunks = (BYTES / chunk as usize - 1) as u64;
                for blocks in [16u32, 82, 328] {
                    let g = time(
                        &|| {
                            let mut b = stream.launch_builder(&f);
                            b.arg(&base)
                                .arg(&chunk)
                                .arg(&nchunks)
                                .arg(&hint)
                                .arg(&aligned)
                                .arg(&out_ptr);
                            unsafe {
                                b.launch(LaunchConfig {
                                    grid_dim: (blocks, 1, 1),
                                    block_dim: (128, 1, 1),
                                    shared_mem_bytes: 0,
                                })
                            }
                            .map(|_| ())
                            .map_err(crate::Error::wrap)
                        },
                        nchunks as usize * chunk as usize,
                    )?;
                    row += &format!(" {g:>7.2}");
                }
                println!("{row}");
            }
        }
    }

    let dst = unsafe { dev.alloc::<u8>(BYTES)? };
    let (dst_ptr, _gd) = dst.device_ptr(&stream);

    println!("4. wide copy pinned → VRAM, 128 threads/block (GB/s)");
    println!("   {:>10} {:>7} {:>7} {:>7} {:>7} {:>7}  one expert (819,200 B), µs at W = 4 / 8 / 16 / 32",
             "kernel", "W=4", "8", "16", "32", "82");
    for name in ["cp_wide_u1", "cp_wide_u4", "cp_wide_u8"] {
        let f = module.load_function(name).map_err(crate::Error::wrap)?;
        let launch = |blocks: u32, n16: u64| -> Result<()> {
            let mut b = stream.launch_builder(&f);
            b.arg(&base).arg(&dst_ptr).arg(&n16);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map(|_| ())
            .map_err(crate::Error::wrap)
        };
        let mut row = format!("   {name:>10}");
        let big = (64usize << 20) / 16;
        for w in [4u32, 8, 16, 32, 82] {
            let g = time(&|| launch(w, big as u64), big * 16)?;
            row += &format!(" {g:>7.2}");
        }
        row += "  ";
        for w in [4u32, 8, 16, 32] {
            let g = time(&|| launch(w, 51_200), 819_200)?;
            row += &format!(" {:>6.1}", 819_200.0 / (g * 1e9) * 1e6);
        }
        println!("{row}");
    }

    println!("5. the GEMM's address walk, 64 experts × 16 row-tile blocks (GB/s)");
    {
        let f = module
            .load_function("rd_gemm_walk")
            .map_err(crate::Error::wrap)?;
        for depth in [1i32, 4] {
            let g = time(
                &|| {
                    let mut b = stream.launch_builder(&f);
                    b.arg(&base).arg(&depth).arg(&out_ptr);
                    unsafe {
                        b.launch(LaunchConfig {
                            grid_dim: (64 * 16, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })
                    }
                    .map(|_| ())
                    .map_err(crate::Error::wrap)
                },
                64 * 819_200,
            )?;
            println!("   {depth} K block(s) in flight per warp: {g:>7.2}");
        }
    }

    println!("copy engine, same buffer:");
    {
        let (dst_ptr, _gd) = dst.device_ptr(&stream);
        let g = time(
            &|| {
                let r = unsafe {
                    sys::cuMemcpyHtoDAsync_v2(
                        dst_ptr,
                        host as *const c_void,
                        BYTES,
                        stream.cu_stream(),
                    )
                };
                if r == sys::CUresult::CUDA_SUCCESS {
                    Ok(())
                } else {
                    crate::bail!("H2D: {r:?}")
                }
            },
            BYTES,
        )?;
        println!("   {g:>7.2}");
    }
    unsafe {
        sys::cuMemFreeHost(host);
    }
    Ok(())
}
