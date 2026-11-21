//! Measurements for the live MoE dispatch's kernel side, before it is built.
//!
//! These benches answer what the live dispatch depends on: whether a grouped
//! GEMM reading pinned host memory gives the same bits as the VRAM run, how many
//! blocks reading host memory at once reach the link's knee, and how a launch
//! whose workers spin on cold experts behaves while the forward thread loads a
//! module. The simulated launch uses mapped words published by a stager, with
//! the same copy and sync structure as the real launch, so its timing carries
//! over.

use super::*;

/// **Measurements for live dispatch revision 2's kernel side**
/// (`docs/moe_live_dispatch_design.md` §0.10): the unmodified grouped GEMM with
/// its weights read from pinned host memory instead of VRAM.
///
/// 1. Correctness — table entries holding the host addresses of `cuMemAllocHost`
///    copies of the experts give output bit-identical to the VRAM run (device
///    access at the host address under UVA, and `cp.async` from host memory).
/// 2. Link throughput against the number of blocks reading host memory at once
///    — the knee sizes the worker-block count `W`.
/// 3. Re-reads — one pinned expert across 1/2/4/8 token tiles: does L2 absorb the
///    repeated reads of a prefill-width expert?
/// 4. Interference — VRAM tiles alone, pinned tiles alone, and both in one launch.
///
/// Plus the copy engine's H2D rate for the same bytes, the baseline every
/// number here is compared with. Shapes are qwen36-35B's gate projection
/// (512 × 2048, Q6_KO).
#[test]
#[ignore = "measurement: cargo test --release --features cuda -p candle-core --lib \
            quantized::cuda::test::bench_grouped_gemm_weights_from_pinned_host -- \
            --exact --ignored --nocapture"]
pub(super) fn bench_grouped_gemm_weights_from_pinned_host() -> Result<()> {
    use cudarc::driver::{sys, CudaSlice};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::time::Instant;

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let (n_experts, nrows, ncols) = (64usize, 512usize, 2048usize);
    let ko_dtype = GgmlDType::Q6_KO;
    let mut rng = StdRng::seed_from_u64(0xbe4c_0001);
    let shape = crate::Shape::from((nrows, ncols));
    let mut storages = Vec::with_capacity(n_experts);
    for _ in 0..n_experts {
        let w: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut q = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q6_K)?;
        q.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&w)?,
            dev.clone(),
        ))?;
        storages.push(q.repack_ko(&shape, ko_dtype)?);
    }
    let expert_bytes = storages[0].storage_size_in_bytes();
    // 256-byte aligned host slots, as a slot image's projections are.
    let host_stride = expert_bytes.div_ceil(256) * 256;
    let mut host: *mut c_void = std::ptr::null_mut();
    let r = unsafe { sys::cuMemAllocHost_v2(&mut host, n_experts * host_stride) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "pinned host copy");
    let vram_ptrs: Vec<u64> = storages.iter().map(|s| s.data_ptr()).collect();
    let host_ptrs: Vec<u64> = (0..n_experts)
        .map(|e| host as u64 + (e * host_stride) as u64)
        .collect();
    for e in 0..n_experts {
        let r = copy_to_host(
            &dev,
            host_ptrs[e] as *mut c_void,
            vram_ptrs[e],
            expert_bytes,
        );
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "copy expert to host");
    }
    let vram_table = dev.memcpy_stod(&vram_ptrs)?;
    let pinned_table = dev.memcpy_stod(&host_ptrs)?;
    // Experts 0..32 in VRAM, 32..64 pinned — for the interference runs.
    let mixed_ptrs: Vec<u64> = (0..n_experts)
        .map(|e| if e < 32 { vram_ptrs[e] } else { host_ptrs[e] })
        .collect();
    let mixed_table = dev.memcpy_stod(&mixed_ptrs)?;
    let stream = dev.cuda_stream();

    // One routed launch: `ids` is `[n_tokens]` (top-1), rows sorted by
    // bucketize. Returns (output, mean µs per launch).
    let run = |table: &CudaSlice<u64>, ids: &[u32], iters: usize| -> Result<(Vec<f32>, f64)> {
        let n_tokens = ids.len();
        let t = crate::Tensor::from_vec(ids.to_vec(), (n_tokens, 1), &device)?;
        let mut ws = MoeBucketizeWorkspace::new(&dev, n_tokens, 1)?;
        moe_bucketize(
            &t,
            n_experts,
            32,
            &mut ws,
            None,
            &DecodeRows::prefix(n_tokens),
        )?;
        let a_ub = n_tokens;
        let launch_tiles = a_ub.min(a_ub.div_ceil(32) + n_experts);
        let act: Vec<f32> = (0..a_ub * ncols)
            .map(|i| ((i % 97) as f32 - 48.0) / 50.0)
            .collect();
        let op = quantize_acts_q8a128_test(&dev, &act, a_ub, ncols)?;
        let launch = || {
            grouped_qmatmul_dev_q8a128(
                &op,
                table,
                0,
                n_experts,
                ko_dtype,
                nrows,
                &ws.tile_expert,
                &ws.tile_b_start,
                &ws.tile_b_cnt,
                launch_tiles,
                2,
                None,
                &dev,
            )
        };
        let out = launch()?;
        let out = read_f32_tensor(&dev, &out)?;
        for _ in 0..2 {
            drop(launch()?);
        }
        stream.synchronize().map_err(crate::Error::wrap)?;
        let start = Instant::now();
        for _ in 0..iters {
            drop(launch()?);
        }
        stream.synchronize().map_err(crate::Error::wrap)?;
        Ok((out, start.elapsed().as_secs_f64() * 1e6 / iters as f64))
    };
    let gbps = |bytes: usize, us: f64| bytes as f64 / (us * 1e-6) / 1e9;
    println!("expert gate projection: {nrows}×{ncols} {ko_dtype:?}, {expert_bytes} B");

    // ── 1. correctness ──
    let ids: Vec<u32> = (0..8u32).map(|i| i * 7 % n_experts as u32).collect();
    let (v_out, _) = run(&vram_table, &ids, 1)?;
    let (p_out, _) = run(&pinned_table, &ids, 1)?;
    assert_eq!(
        v_out.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        p_out.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
        "weights read from pinned host memory must give the VRAM run's bits"
    );
    println!("1. pinned-host weights: bit-identical to VRAM ✓");

    // ── 2. link throughput vs concurrent host-reading blocks ──
    println!("2. distinct experts (16 row-tile blocks each), one token tile each:");
    println!(
        "   {:>7} {:>7} {:>11} {:>9} {:>11} {:>9}",
        "experts", "blocks", "vram µs", "vram GB/s", "pinned µs", "pin GB/s"
    );
    for m in [1usize, 2, 4, 8, 16, 32, 64] {
        let ids: Vec<u32> = (0..m as u32).collect();
        let (_, vus) = run(&vram_table, &ids, 20)?;
        let (_, pus) = run(&pinned_table, &ids, 20)?;
        let bytes = m * expert_bytes;
        println!(
            "   {:>7} {:>7} {:>11.1} {:>9.1} {:>11.1} {:>9.2}",
            m,
            m * nrows / 32,
            vus,
            gbps(bytes, vus),
            pus,
            gbps(bytes, pus)
        );
    }

    // ── 2b. the knee below 16 blocks: 32-row experts, one block each ──
    {
        let small_rows = 32usize;
        let small_shape = crate::Shape::from((small_rows, ncols));
        let mut small = Vec::new();
        for _ in 0..16 {
            let w: Vec<f32> = (0..small_rows * ncols)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let mut q = QCudaStorage::zeros(&dev, ncols * small_rows, GgmlDType::Q6_K)?;
            q.quantize(&CudaStorage::wrap_cuda_slice(
                dev.memcpy_stod(&w)?,
                dev.clone(),
            ))?;
            small.push(q.repack_ko(&small_shape, ko_dtype)?);
        }
        let sb = small[0].storage_size_in_bytes();
        let sstride = sb.div_ceil(256) * 256;
        let mut shost: *mut c_void = std::ptr::null_mut();
        let r = unsafe { sys::cuMemAllocHost_v2(&mut shost, 16 * sstride) };
        assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "pinned small copy");
        let mut sptrs = vec![0u64; n_experts];
        for (e, s) in small.iter().enumerate() {
            let hp = shost as u64 + (e * sstride) as u64;
            let r = copy_to_host(&dev, hp as *mut c_void, s.data_ptr(), sb);
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "copy small expert");
            sptrs[e] = hp;
        }
        // Unused experts point at expert 0; nothing routes to them.
        for p in sptrs.iter_mut().skip(16) {
            *p = shost as u64;
        }
        let stable = dev.memcpy_stod(&sptrs)?;
        println!("2b. 32-row experts, one block each (pinned):");
        println!(
            "   {:>7} {:>11} {:>9} {:>14}",
            "blocks", "pinned µs", "GB/s", "GB/s per block"
        );
        for m in [1usize, 2, 4, 8, 16] {
            let ids: Vec<u32> = (0..m as u32).collect();
            let t = crate::Tensor::from_vec(ids.clone(), (m, 1), &device)?;
            let mut ws = MoeBucketizeWorkspace::new(&dev, m, 1)?;
            moe_bucketize(&t, n_experts, 32, &mut ws, None, &DecodeRows::prefix(m))?;
            let act: Vec<f32> = (0..m * ncols)
                .map(|i| ((i % 97) as f32 - 48.0) / 50.0)
                .collect();
            let op = quantize_acts_q8a128_test(&dev, &act, m, ncols)?;
            let launch = || {
                grouped_qmatmul_dev_q8a128(
                    &op,
                    &stable,
                    0,
                    n_experts,
                    ko_dtype,
                    small_rows,
                    &ws.tile_expert,
                    &ws.tile_b_start,
                    &ws.tile_b_cnt,
                    m.min(m.div_ceil(32) + n_experts),
                    2,
                    None,
                    &dev,
                )
            };
            for _ in 0..3 {
                drop(launch()?);
            }
            stream.synchronize().map_err(crate::Error::wrap)?;
            let start = Instant::now();
            for _ in 0..50 {
                drop(launch()?);
            }
            stream.synchronize().map_err(crate::Error::wrap)?;
            let us = start.elapsed().as_secs_f64() * 1e6 / 50.0;
            let g = gbps(m * sb, us);
            println!("   {:>7} {:>11.1} {:>9.2} {:>14.2}", m, us, g, g / m as f64);
        }
        unsafe {
            sys::cuMemFreeHost(shost);
        }
    }

    // ── 3. re-reads: one expert, T token tiles ──
    println!("3. one expert, T token tiles of 32:");
    println!(
        "   {:>3} {:>11} {:>11} {:>14}",
        "T", "vram µs", "pinned µs", "pinned/T=1"
    );
    let mut p1 = 0.0;
    for tiles in [1usize, 2, 4, 8] {
        let ids = vec![0u32; tiles * 32];
        let (_, vus) = run(&vram_table, &ids, 20)?;
        let (_, pus) = run(&pinned_table, &ids, 20)?;
        if tiles == 1 {
            p1 = pus;
        }
        println!(
            "   {:>3} {:>11.1} {:>11.1} {:>14.2}",
            tiles,
            vus,
            pus,
            pus / p1
        );
    }

    // ── 4. interference: VRAM tiles beside pinned tiles in one launch ──
    // Prefill-like VRAM work: 32 VRAM experts × 4 token tiles; pinned work: 8
    // pinned experts × 1 token tile.
    let vram_ids: Vec<u32> = (0..32u32)
        .flat_map(|e| std::iter::repeat_n(e, 128))
        .collect();
    let pin_ids: Vec<u32> = (32..40u32)
        .flat_map(|e| std::iter::repeat_n(e, 32))
        .collect();
    let both: Vec<u32> = vram_ids.iter().chain(pin_ids.iter()).copied().collect();
    let (_, a_us) = run(&mixed_table, &vram_ids, 20)?;
    let (_, b_us) = run(&mixed_table, &pin_ids, 20)?;
    let (_, c_us) = run(&mixed_table, &both, 20)?;
    println!("4. interference (one launch):");
    println!("   VRAM tiles alone   {a_us:>9.1} µs");
    println!("   pinned tiles alone {b_us:>9.1} µs");
    println!(
        "   both together      {c_us:>9.1} µs   (sum {:.1}, max {:.1})",
        a_us + b_us,
        a_us.max(b_us)
    );
    // The same with the pinned tiles FIRST in tile order — the order revision 2
    // dispatches them in: experts 0..32 pinned, 32..64 in VRAM.
    let first_ptrs: Vec<u64> = (0..n_experts)
        .map(|e| if e < 32 { host_ptrs[e] } else { vram_ptrs[e] })
        .collect();
    let first_table = dev.memcpy_stod(&first_ptrs)?;
    let vram_ids2: Vec<u32> = (32..64u32)
        .flat_map(|e| std::iter::repeat_n(e, 128))
        .collect();
    let pin_ids2: Vec<u32> = (0..8u32).flat_map(|e| std::iter::repeat_n(e, 32)).collect();
    let both2: Vec<u32> = pin_ids2.iter().chain(vram_ids2.iter()).copied().collect();
    let (_, a2) = run(&first_table, &vram_ids2, 20)?;
    let (_, b2) = run(&first_table, &pin_ids2, 20)?;
    let (_, c2) = run(&first_table, &both2, 20)?;
    println!("   pinned tiles first in tile order:");
    println!("   VRAM alone {a2:>9.1} µs, pinned alone {b2:>9.1} µs, both {c2:>9.1} µs (sum {:.1}, max {:.1})", a2 + b2, a2.max(b2));

    // ── the platform's ceiling for SM reads of host memory ──
    // A plain reduction over a view of the pinned copy: coalesced loads, no
    // GEMM structure — what any kernel reading host memory can get here.
    {
        let floats = n_experts * host_stride / 4;
        let view = std::mem::ManuallyDrop::new(unsafe {
            stream.upgrade_device_ptr::<f32>(host as u64, floats)
        });
        let storage = CudaStorage::wrap_cuda_slice((*view).clone(), dev.clone());
        let t = crate::tensor::from_storage(
            crate::Storage::Cuda(storage),
            crate::Shape::from(floats),
            crate::op::BackpropOp::none(),
            false,
        );
        let _ = t.sum_all()?.to_scalar::<f32>()?;
        let start = Instant::now();
        for _ in 0..10 {
            drop(t.sum_all()?);
        }
        stream.synchronize().map_err(crate::Error::wrap)?;
        let us = start.elapsed().as_secs_f64() * 1e6 / 10.0;
        println!(
            "SM reads of host memory (sum over {} MiB): {:.2} GB/s",
            (floats * 4) >> 20,
            gbps(floats * 4, us)
        );
        std::mem::forget(t);
    }

    // ── baseline: the copy engine, same bytes ──
    println!("copy engine H2D from the same pinned memory:");
    let dst = unsafe { dev.alloc::<u8>(n_experts * host_stride)? };
    let (dst_ptr, _gd) = dst.device_ptr(&stream);
    for m in [1usize, 8, 64] {
        let bytes = m * host_stride;
        let start = Instant::now();
        for _ in 0..20 {
            let r = unsafe {
                sys::cuMemcpyHtoDAsync_v2(dst_ptr, host as *const c_void, bytes, stream.cu_stream())
            };
            assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "H2D");
        }
        stream.synchronize().map_err(crate::Error::wrap)?;
        let us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        println!(
            "   {:>3} experts {:>9.1} µs {:>7.2} GB/s",
            m,
            us,
            gbps(bytes, us)
        );
    }
    unsafe {
        sys::cuMemFreeHost(host);
    }
    Ok(())
}

/// The revision-2 expert launch, simulated: one kernel, `W` worker blocks first
/// and hit blocks after (`docs/moe_live_dispatch_design.md` §0.10).
///
/// A **worker** pulls miss items `(expert, row tile)` off a counter. It reads the
/// expert's live-table entry — a mapped word holding the host address of the
/// expert's bytes, 0 while cold — spinning on it gently until a host store
/// publishes it; copies the item's 32-row slice (16 K blocks × 3,200 B) with wide
/// loads into its VRAM scratch slot (the "mini loop"); one `__syncthreads`; then
/// runs the fake GEMM over the slot as a 32-row matrix. A **hit** block runs the
/// fake GEMM over the VRAM copy of its expert, one tile. The fake GEMM has the
/// real one's access structure — per K block, each warp `cp.async`s its 800 B
/// chunk, waits, the block syncs — repeated `reps` times (token tiles), and
/// XORs every word once so the result checks the bytes. Blocks stamp the
/// global timer so the launch reports when hit work and miss work finished.
pub(super) const MOE_SIM: &str = r#"
typedef unsigned long long u64;
__device__ __forceinline__ u64 now_ns() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u64 ld_acq_sys(const u64* p) {
    u64 v; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
    uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src)); }

// The fake GEMM over a [16 K blocks][row groups] KO matrix: warp w reads chunk
// k·rg_stride + rg_base + w (800 B) per K block.
__device__ unsigned fake_gemm(const unsigned char* wbase, int rg_stride, int rg_base, int reps,
                              unsigned char (*slot)[800], int lane, int warp) {
    unsigned acc = 0;
    float f = 1.f;
    for (int r = 0; r < reps; ++r) {
        for (int k = 0; k < 16; ++k) {
            const unsigned char* p = wbase + ((u64)k * rg_stride + rg_base + warp) * 800ull;
            for (int i = lane; i < 50; i += 32) cp16(&slot[warp][i * 16], p + i * 16);
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 0;");
            __syncwarp();
            const unsigned* s = reinterpret_cast<const unsigned*>(slot[warp]);
            for (int i = lane; i < 200; i += 32) {
                const unsigned v = s[i];
                if (r == 0) acc ^= v;
                f = fmaf(f, 1.0000001f, (float)(v & 7));   // a little ALU per word
            }
            __syncthreads();
        }
    }
    if (f == 12345.f) acc ^= 1u;
    return acc;
}

__device__ __forceinline__ unsigned block_xor(unsigned a, unsigned* red) {
    if (threadIdx.x == 0) *red = 0u;
    __syncthreads();
    atomicXor(red, a);
    __syncthreads();
    return *red;
}

extern "C" __global__ void moe_sim(
    const u64* entries,            // mapped: host address of expert e's bytes, 0 = cold
    const unsigned* abort_word,    // mapped
    const unsigned char* vram_w,   // VRAM copy: expert e at vram_w + e·819200
    const int* miss_items, int n_miss,
    const int* hit_items, int n_hit,
    int W, int reps,
    unsigned char* scratch,        // VRAM: W × 51,200
    int* counter,
    unsigned* out_miss, unsigned* out_hit,
    u64* clocks,                   // [start (min), hits done (max), misses done (max), wait ns (sum)]
    u64 spin_limit_ns)
{
    __shared__ __align__(16) unsigned char slot[4][800];
    __shared__ int item_s;
    __shared__ u64 src_s;
    __shared__ unsigned red;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) atomicMin(&clocks[0], now_ns());

    if ((int)blockIdx.x < W) {
        unsigned char* my = scratch + (u64)blockIdx.x * 51200ull;
        for (;;) {
            if (tid == 0) item_s = atomicAdd(counter, 1);
            __syncthreads();
            const int it = item_s;
            __syncthreads();
            if (it >= n_miss) break;
            const int e = miss_items[2 * it], t = miss_items[2 * it + 1];
            if (tid == 0) {
                u64 s = ld_acq_sys(&entries[e]);
                if (s == 0) {
                    const u64 t0 = now_ns();
                    unsigned ns = 32;
                    while ((s = ld_acq_sys(&entries[e])) == 0) {
                        if (*(const volatile unsigned*)abort_word != 0u || now_ns() - t0 > spin_limit_ns) __trap();
                        __nanosleep(ns);
                        ns = ns < 32768u ? ns * 2u : 32768u;
                    }
                    atomicAdd(&clocks[3], now_ns() - t0);
                }
                src_s = s;
            }
            __syncthreads();
            const unsigned char* src = reinterpret_cast<const unsigned char*>(src_s);
            // The mini loop: 16 pieces × 200 uint4, four loads in flight per thread.
            for (int base = tid; base < 3200; base += 128 * 4) {
                uint4 v[4];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int j = base + u * 128;
                    if (j < 3200) {
                        const int k = j / 200, o = j % 200;
                        v[u] = ld_cg4(src + ((u64)k * 64 + t * 4) * 800ull + o * 16);
                    }
                }
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int j = base + u * 128;
                    if (j < 3200) reinterpret_cast<uint4*>(my)[j] = v[u];
                }
            }
            __syncthreads();
            const unsigned a = block_xor(fake_gemm(my, 4, 0, reps, slot, lane, warp), &red);
            if (tid == 0) {
                out_miss[it] = a;
                atomicMax(&clocks[2], now_ns());
            }
        }
    } else {
        const int it = blockIdx.x - W;
        if (it < n_hit) {
            const int e = hit_items[2 * it], t = hit_items[2 * it + 1];
            const unsigned a = block_xor(fake_gemm(vram_w + (u64)e * 819200ull, 64, t * 4, reps,
                                                   slot, lane, warp), &red);
            if (tid == 0) {
                out_hit[it] = a;
                atomicMax(&clocks[1], now_ns());
            }
        }
    }
}
"#;

/// A kernel no other code ever launches, in its own module: loading it while
/// `moe_sim` spins is the lazy-load case of §0.1.
pub(super) const LATE_MODULE: &str = r#"
extern "C" __global__ void late_kernel(float* x) { x[threadIdx.x] += 1.f; }
"#;

/// **Revision 2's expert launch, end to end, before any of it is built**
/// (`docs/moe_live_dispatch_design.md` §0.10): the grid shape (workers first,
/// hits after), the mini-loop copy + one sync, warm misses, cold misses served
/// from a real NVMe file by reader threads that only read and store, and a
/// forward thread that blocks in a module load while workers spin.
#[test]
#[ignore = "measurement: cargo test --release --features cuda -p candle-core --lib \
            quantized::cuda::test::bench_moe_worker_launch_simulation -- --exact --ignored --nocapture"]
pub(super) fn bench_moe_worker_launch_simulation() -> Result<()> {
    use crate::direct_io::DirectFile;
    use cudarc::driver::{sys, CudaModule, LaunchConfig, PushKernelArg};
    use std::cell::{Cell, RefCell};
    use std::sync::Arc;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    const EXPERT: usize = 819_200; // 512 × 2048 Q6_KO
    const N_E: usize = 64;
    const TILES: usize = 16; // 32-row tiles per expert
    let dev = CudaDevice::new(0)?;
    let stream = dev.cuda_stream();
    let compile = |src: &str, name: &str| -> Result<Arc<CudaModule>> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            src,
            cudarc::nvrtc::CompileOptions {
                arch: Some("compute_86"),
                ..Default::default()
            },
        )
        .map_err(crate::Error::wrap)?;
        dev.load_ptx(name, &ptx)
    };
    let module = compile(MOE_SIM, "moe_sim")?;
    let sim = module
        .load_function("moe_sim")
        .map_err(crate::Error::wrap)?;
    // Compiled now, loaded only during the spin (experiment D).
    let late_ptx = cudarc::nvrtc::compile_ptx_with_opts(
        LATE_MODULE,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            ..Default::default()
        },
    )
    .map_err(crate::Error::wrap)?;

    // ── The experts: pinned warm copy, VRAM copy, and a file on disk ──
    let mut warm: *mut c_void = std::ptr::null_mut();
    let r = unsafe { sys::cuMemAllocHost_v2(&mut warm, N_E * EXPERT) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "warm tier");
    let words = unsafe { std::slice::from_raw_parts_mut(warm as *mut u32, N_E * EXPERT / 4) };
    let mut x = 0x2545_f491u32;
    for w in words.iter_mut() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *w = x;
    }
    let bytes = unsafe { std::slice::from_raw_parts(warm as *const u8, N_E * EXPERT) };
    let vram = dev.memcpy_stod(bytes)?;
    let (vram_ptr, _gv) = vram.device_ptr(&stream);
    let path = std::env::temp_dir().join(format!("moe_sim_pack_{}.bin", std::process::id()));
    std::fs::write(&path, bytes)?;
    let file = Arc::new(DirectFile::open(&path)?);
    let mut pad: *mut c_void = std::ptr::null_mut();
    let r = unsafe { sys::cuMemAllocHost_v2(&mut pad, 16 * EXPERT) };
    assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "pad");

    // Expected XOR of item (e, t): its 16 pieces of 3,200 B.
    let expect = |e: usize, t: usize| -> u32 {
        let mut a = 0u32;
        for k in 0..16 {
            let off = e * EXPERT + (k * 64 + t * 4) * 800;
            for w in &words[off / 4..off / 4 + 800] {
                a ^= *w;
            }
        }
        a
    };

    // ── Mapped words ──
    let mapped = |n: usize| -> (*mut u64, u64) {
        let mut raw: *mut c_void = std::ptr::null_mut();
        let mut d: sys::CUdeviceptr = 0;
        unsafe {
            assert_eq!(
                sys::cuMemHostAlloc(&mut raw, n * 8, sys::CU_MEMHOSTALLOC_DEVICEMAP),
                sys::CUresult::CUDA_SUCCESS
            );
            assert_eq!(
                sys::cuMemHostGetDevicePointer_v2(&mut d, raw, 0),
                sys::CUresult::CUDA_SUCCESS
            );
            std::ptr::write_bytes(raw as *mut u8, 0, n * 8);
        }
        (raw as *mut u64, d)
    };
    let (entries_h, entries_d) = mapped(N_E);
    let (abort_h, abort_d) = mapped(1);
    let _ = abort_h;
    let scratch = unsafe { dev.alloc::<u8>(32 * 51_200)? };
    let (scratch_ptr, _gs) = scratch.device_ptr(&stream);
    let out_miss = dev.alloc_zeros::<u32>(N_E * TILES)?;
    let out_hit = dev.alloc_zeros::<u32>(N_E * TILES)?;
    let (om, _g1) = out_miss.device_ptr(&stream);
    let (oh, _g2) = out_hit.device_ptr(&stream);
    let spin_limit: u64 = 1_500_000_000;

    struct Run {
        kernel_us: f64,
        hits_us: f64,
        misses_us: f64,
        wait_ms: f64,
    }
    // Launch `misses` (experts, all tiles) on workers and `hits` (experts, all
    // tiles) as hit blocks; `cold` experts start unpublished and `during` runs
    // on this thread right after the launch (the stager and the forward thread).
    let run = |misses: &[usize],
               hits: &[usize],
               cold: &[usize],
               w: i32,
               reps: i32,
               during: &dyn Fn() -> Result<()>|
     -> Result<Run> {
        let items = |es: &[usize]| -> Vec<i32> {
            es.iter()
                .flat_map(|&e| (0..TILES).flat_map(move |t| [e as i32, t as i32]))
                .collect()
        };
        let mi = items(misses);
        let hi = items(hits);
        let mi_d = dev.memcpy_stod(if mi.is_empty() {
            &[0i32, 0][..]
        } else {
            &mi[..]
        })?;
        let hi_d = dev.memcpy_stod(if hi.is_empty() {
            &[0i32, 0][..]
        } else {
            &hi[..]
        })?;
        let (mi_p, _a) = mi_d.device_ptr(&stream);
        let (hi_p, _b) = hi_d.device_ptr(&stream);
        let counter = dev.alloc_zeros::<i32>(1)?;
        let (cp, _c) = counter.device_ptr(&stream);
        let clocks = dev.memcpy_stod(&[u64::MAX, 0, 0, 0])?;
        let (ck, _d) = clocks.device_ptr(&stream);
        for e in 0..N_E {
            let v = if cold.contains(&e) {
                0
            } else {
                warm as u64 + (e * EXPERT) as u64
            };
            unsafe { std::ptr::write_volatile(entries_h.add(e), v) };
        }
        stream.synchronize().map_err(crate::Error::wrap)?;
        let n_miss = (mi.len() / 2) as i32;
        let n_hit = (hi.len() / 2) as i32;
        let w = if n_miss == 0 { 0 } else { w };
        let grid = (w + n_hit) as u32;
        let start = Instant::now();
        let mut b = stream.launch_builder(&sim);
        b.arg(&entries_d)
            .arg(&abort_d)
            .arg(&vram_ptr)
            .arg(&mi_p)
            .arg(&n_miss)
            .arg(&hi_p)
            .arg(&n_hit)
            .arg(&w)
            .arg(&reps)
            .arg(&scratch_ptr)
            .arg(&cp)
            .arg(&om)
            .arg(&oh)
            .arg(&ck)
            .arg(&spin_limit);
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(crate::Error::wrap)?;
        unsafe {
            let _ = sys::cuStreamQuery(stream.cu_stream());
        }
        during()?;
        stream.synchronize().map_err(crate::Error::wrap)?;
        let kernel_us = start.elapsed().as_secs_f64() * 1e6;
        // Verify every item.
        let got_m = dev.memcpy_dtov(&out_miss)?;
        for (i, pair) in mi.chunks(2).enumerate() {
            assert_eq!(
                got_m[i],
                expect(pair[0] as usize, pair[1] as usize),
                "miss item {i}"
            );
        }
        let got_h = dev.memcpy_dtov(&out_hit)?;
        for (i, pair) in hi.chunks(2).enumerate() {
            assert_eq!(
                got_h[i],
                expect(pair[0] as usize, pair[1] as usize),
                "hit item {i}"
            );
        }
        let c = dev.memcpy_dtov(&clocks)?;
        let span = |end: u64| {
            if end > c[0] {
                (end - c[0]) as f64 / 1e3
            } else {
                0.0
            }
        };
        Ok(Run {
            kernel_us,
            hits_us: span(c[1]),
            misses_us: span(c[2]),
            wait_ms: c[3] as f64 / 1e6,
        })
    };
    let nothing = || -> Result<()> { Ok(()) };
    // Warm the kernel once.
    run(&[0], &[32], &[], 4, 1, &nothing)?;

    println!("A. warm misses only (all 16 tiles each), one token tile — copy + fake GEMM");
    println!(
        "   {:>7} {:>5} {:>10} {:>9} {:>14}",
        "experts", "W", "misses µs", "GB/s", "µs per expert"
    );
    for m in [1usize, 2, 8, 32] {
        for w in [4i32, 8, 16] {
            let misses: Vec<usize> = (0..m).collect();
            let r = run(&misses, &[], &[], w, 1, &nothing)?;
            println!(
                "   {:>7} {:>5} {:>10.1} {:>9.2} {:>14.1}",
                m,
                w,
                r.misses_us,
                (m * EXPERT) as f64 / (r.misses_us * 1e-6) / 1e9,
                r.misses_us / m as f64
            );
        }
    }

    println!("B. overlap: 32 VRAM experts (hits) + 8 warm misses, 4 token tiles (reps), W = 8");
    let hits: Vec<usize> = (32..64).collect();
    let misses8: Vec<usize> = (0..8).collect();
    let h = run(&[], &hits, &[], 8, 4, &nothing)?;
    let m = run(&misses8, &[], &[], 8, 4, &nothing)?;
    let both = run(&misses8, &hits, &[], 8, 4, &nothing)?;
    println!("   hits alone   : hits done {:>8.1} µs", h.hits_us);
    println!("   misses alone : misses done {:>8.1} µs", m.misses_us);
    println!(
        "   together     : hits done {:>8.1} µs, misses done {:>8.1} µs, kernel {:>8.1} µs",
        both.hits_us, both.misses_us, both.kernel_us
    );

    // The stager: QD reader threads, each reads whole cold experts from the
    // file into pad slots and publishes them with a host store. No CUDA calls.
    let stage = |cold: Vec<usize>, qd: usize, delay: Duration| -> JoinHandle<Vec<f64>> {
        let file = Arc::clone(&file);
        let pad = pad as usize;
        let entries = entries_h as usize;
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            let t0 = Instant::now();
            let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let cold = Arc::new(cold);
            let done: Vec<JoinHandle<Vec<f64>>> = (0..qd)
                .map(|q| {
                    let (file, next, cold) =
                        (Arc::clone(&file), Arc::clone(&next), Arc::clone(&cold));
                    std::thread::spawn(move || {
                        let mut published = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= cold.len() {
                                break;
                            }
                            let e = cold[i];
                            let slot = pad + i * EXPERT;
                            let dest =
                                unsafe { std::slice::from_raw_parts_mut(slot as *mut u8, EXPERT) };
                            file.read_at_with_handle(q, (e * EXPERT) as u64, dest)
                                .expect("pack read");
                            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                            unsafe {
                                std::ptr::write_volatile((entries as *mut u64).add(e), slot as u64)
                            };
                            published.push(t0.elapsed().as_secs_f64() * 1e3);
                        }
                        published
                    })
                })
                .collect();
            let mut all: Vec<f64> = done
                .into_iter()
                .flat_map(|h| h.join().expect("reader"))
                .collect();
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            all
        })
    };

    println!("C. cold misses from NVMe: 32 hits + 8 warm misses + 8 cold misses (stager QD 8, no CUDA calls), 4 token tiles, W = 8");
    let warm_cold: Vec<usize> = (0..16).collect(); // 0..8 warm, 8..16 cold (cold last)
    let cold: Vec<usize> = (8..16).collect();
    for qd in [1usize, 4, 8] {
        let handle = RefCell::new(None);
        let r = run(&warm_cold, &hits, &cold, 8, 4, &|| {
            *handle.borrow_mut() = Some(stage(cold.clone(), qd, Duration::ZERO));
            Ok(())
        })?;
        let pubs = handle.borrow_mut().take().unwrap().join().expect("stager");
        println!(
            "   QD {qd}: published {:.2}–{:.2} ms; hits done {:>8.1} µs, misses done {:>8.1} µs, kernel {:>8.1} µs, worker spin {:.2} ms",
            pubs[0], pubs[pubs.len() - 1], r.hits_us, r.misses_us, r.kernel_us, r.wait_ms
        );
    }

    println!("D. the forward thread blocks while workers spin: queues work behind, then loads + launches a never-loaded module");
    {
        let blocked = Cell::new(0.0f64);
        let handle = RefCell::new(None);
        let buf = dev.alloc_zeros::<f32>(32)?;
        let r = run(&warm_cold, &hits, &cold, 8, 4, &|| {
            // The stager starts 20 ms late, so the workers are certainly spinning
            // when this thread enters the driver.
            *handle.borrow_mut() = Some(stage(cold.clone(), 8, Duration::from_millis(20)));
            // Work queued behind the spinning launch.
            for _ in 0..8 {
                drop(dev.memcpy_stod(&[1.0f32; 32])?);
            }
            let t = Instant::now();
            let late = dev.load_ptx("late_module", &late_ptx)?;
            let f = late
                .load_function("late_kernel")
                .map_err(crate::Error::wrap)?;
            let (bp, _g) = buf.device_ptr(&stream);
            let mut b = stream.launch_builder(&f);
            b.arg(&bp);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(crate::Error::wrap)?;
            blocked.set(t.elapsed().as_secs_f64() * 1e3);
            Ok(())
        })?;
        let pubs = handle.borrow_mut().take().unwrap().join().expect("stager");
        println!(
            "   forward thread in load+launch {:.2} ms; stager published {:.2}–{:.2} ms (after a 20 ms start delay); kernel {:.1} µs, worker spin {:.2} ms",
            blocked.get(), pubs[0] + 20.0, pubs[pubs.len() - 1] + 20.0, r.kernel_us, r.wait_ms
        );
        assert_eq!(
            dev.memcpy_dtov(&buf)?[0],
            1.0,
            "the late kernel ran after the spinning launch"
        );
    }

    println!("E. spinning workers do not slow the hits: 32 hits + 8 cold misses published after 30 ms, W = 8");
    {
        let handle = RefCell::new(None);
        let r = run(&cold, &hits, &cold, 8, 4, &|| {
            *handle.borrow_mut() = Some(stage(cold.clone(), 8, Duration::from_millis(30)));
            Ok(())
        })?;
        drop(handle.borrow_mut().take().unwrap().join());
        println!(
            "   hits done {:>8.1} µs (alone: {:>8.1} µs); misses done {:>8.1} µs; worker spin {:.2} ms",
            r.hits_us, h.hits_us, r.misses_us, r.wait_ms
        );
    }

    unsafe {
        sys::cuMemFreeHost(warm);
        sys::cuMemFreeHost(pad);
        sys::cuMemFreeHost(entries_h as *mut c_void);
        sys::cuMemFreeHost(abort_h as *mut c_void);
    }
    let _ = std::fs::remove_file(&path);
    Ok(())
}
