//! The warm lift in isolation (`docs/warm_lift.md`), measured piece by piece:
//! the host copy into the pinned ring at 1-8 threads, each scatter kernel alone
//! over runs of every size a lift carries, and a whole lift shaped like
//! production's. Each prints its rates; run one at a time, on a card nothing
//! else is using:
//!
//! `cargo test --release -p candle-nn --features cuda --lib warm_lift_bench -- --ignored --nocapture --test-threads=1`
#![cfg(feature = "cuda")]

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use candle::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use candle::cuda_backend::kernels::simple::kv_migrate::run_kv_migrate_copy;
use candle::cuda_backend::kernels::simple::warm_lift::{run_warm_lift_scatter, WarmLiftEntry};
use candle::quantized::pinned_staging::PinnedBuf;
use candle::{CudaDevice, DType, Device, Tensor};

use crate::kv_cache::chunked::gpu_test_lock::gpu_serial;
use crate::kv_cache::chunked::{lift_warm_to_gpu, ChunkedKvBacking, SealedSequence, WarmLiftRing};

/// The production ring slot.
const SLOT: usize = 8 << 20;
/// Pageable source far past any last-level cache, so every band is a DRAM read.
const SOURCE: usize = 512 << 20;
/// An odd stride through the source's bands: a lift's bands sit wherever their
/// arena slots are, not back to back.
const SCATTER_STRIDE: usize = 40_503;
/// Run lengths a lift carries: a sub-4-bit band, C5's, a BF16 head-chunk, and
/// merged runs of several.
const RUNS: [usize; 6] = [1024, 2176, 4096, 16_384, 65_536, 262_144];

fn cuda_of(device: &Device) -> &CudaDevice {
    let Device::Cuda(cuda) = device else {
        unreachable!("a CUDA bench");
    };
    cuda
}

fn stream_of(device: &Device) -> Arc<CudaStream> {
    (*cuda_of(device).cuda_stream()).clone()
}

fn gbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

/// Fill `dst` from `src` the way the copiers fill a slot: `band`-byte runs read
/// from scattered source offsets, packed in order, the slot cut into `threads`
/// contiguous shares each copied by a thread spawned for that slot. `round`
/// moves the source window so no round rereads the last one's lines.
fn fill_like_the_copiers(src: &[u8], dst: &mut [u8], band: usize, threads: usize, round: usize) {
    let n_src = src.len() / band;
    let per_slot = dst.len() / band;
    let per_thread = per_slot.div_ceil(threads);
    thread::scope(|s| {
        for (t, share) in dst.chunks_mut(per_thread * band).enumerate() {
            s.spawn(move || {
                for (k, out) in share.chunks_mut(band).enumerate() {
                    let j = round * per_slot + t * per_thread + k;
                    let at = (j * SCATTER_STRIDE) % n_src * band;
                    out.copy_from_slice(&src[at..at + band]);
                }
            });
        }
    });
}

/// **Does more than one host thread help the RAM → pinned copy?** Pageable
/// source into the mapped write-combined slot the ring uses, and into ordinary
/// pageable memory for contrast, at 1, 2, 4 and 8 threads.
#[test]
#[ignore = "bench: run alone with --nocapture"]
fn bench_host_copy_into_the_ring_by_threads() {
    let _gpu = gpu_serial();
    let _device = Device::new_cuda(0).unwrap();
    let src: Vec<u8> = (0..SOURCE).map(|i| (i * 31 % 251) as u8).collect();
    let mut wc = PinnedBuf::alloc_owned_mapped(SLOT).unwrap();
    let mut pageable = vec![0u8; SLOT];
    let rounds = 48;
    println!("host copy, {} MiB per point:", (rounds * SLOT) >> 20);
    println!(
        "{:>8} {:>8} {:>12} {:>12}",
        "band", "threads", "wc GB/s", "pageable GB/s"
    );
    for band in [1024usize, 4096, 16_384] {
        for threads in [1usize, 2, 4, 8] {
            let rate = |dst: &mut [u8]| {
                fill_like_the_copiers(&src, dst, band, threads, 0);
                let t = Instant::now();
                for r in 1..=rounds {
                    fill_like_the_copiers(&src, dst, band, threads, r);
                }
                gbps(rounds * SLOT, t.elapsed().as_secs_f64())
            };
            let w = rate(wc.as_mut_slice());
            let p = rate(&mut pageable);
            println!("{band:>8} {threads:>8} {w:>12.2} {p:>12.2}");
        }
    }
}

/// **The scatter kernel alone**: one slot of `run`-byte runs, read from mapped
/// pinned memory at its host address and written to scattered device
/// addresses, launched back to back. Beside it, the copy engine moving the
/// same slot in one transfer — the link's own rate.
#[test]
#[ignore = "bench: run alone with --nocapture"]
fn bench_warm_lift_scatter_alone() {
    let _gpu = gpu_serial();
    let device = Device::new_cuda(0).unwrap();
    let cuda = cuda_of(&device);
    let stream = stream_of(&device);
    let blocks = (cuda.multiprocessor_count().unwrap() * 6) as i32;
    let mut ring = WarmLiftRing::new().unwrap();
    // SAFETY: written by the kernels below before anything reads it.
    let dst: CudaSlice<u8> = unsafe { stream.alloc::<u8>(SLOT) }.unwrap();
    let base = dst.device_ptr(&stream).0;
    let launches = 32;

    let mut whole: CudaSlice<u8> = unsafe { stream.alloc::<u8>(SLOT) }.unwrap();
    let host = vec![7u8; SLOT];
    {
        let slot = ring.acquire().unwrap();
        slot.payload.copy_from_slice(&host);
        device.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..launches {
            cuda.memcpy_htod(&slot.payload[..], &mut whole).unwrap();
        }
        device.synchronize().unwrap();
        println!(
            "copy engine, one {} MiB transfer from the slot: {:.2} GB/s",
            SLOT >> 20,
            gbps(launches * SLOT, t.elapsed().as_secs_f64())
        );
    }

    println!(
        "{:>8} {:>8} {:>10} {:>10}",
        "run", "runs", "GB/s", "us/slot"
    );
    for run in RUNS {
        let n = (SLOT / run).min(ring.max_entries());
        let mut slot = ring.acquire().unwrap();
        // Destinations in a scattered order over the buffer, as arena slots are.
        let entries: Vec<WarmLiftEntry> = (0..n)
            .map(|i| WarmLiftEntry {
                src_off: (i * run) as u64,
                dst: base + ((i * SCATTER_STRIDE) % n * run) as u64,
                bytes: run as u64,
            })
            .collect();
        let table = slot.write_table(&entries).unwrap();
        let payload = slot.dev_payload as *const u8;
        let cu = stream.cu_stream() as *mut std::ffi::c_void;
        // SAFETY: the slot and its table are mapped pinned memory held by
        // `ring`; every destination lies inside `dst`.
        let launch = || unsafe { run_warm_lift_scatter(payload, table, n as i32, blocks, cu) };
        launch();
        device.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..launches {
            launch();
        }
        device.synchronize().unwrap();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{run:>8} {n:>8} {:>10.2} {:>10.1}",
            gbps(launches * n * run, secs),
            secs * 1e6 / launches as f64
        );
        slot.scattered(&stream).unwrap();
    }
}

/// **The migration kernel alone**: device → device runs of every size,
/// scattered over 64 MiB, the shape a hot → hot copy or a demotion's gather
/// takes.
#[test]
#[ignore = "bench: run alone with --nocapture"]
fn bench_kv_migrate_copy_alone() {
    let _gpu = gpu_serial();
    let device = Device::new_cuda(0).unwrap();
    let cuda = cuda_of(&device);
    let stream = stream_of(&device);
    let span = 64usize << 20;
    let src = Tensor::ones(span, DType::U8, &device).unwrap();
    let (src_storage, _) = src.storage_and_layout();
    let src_base = match &*src_storage {
        candle::Storage::Cuda(s) => s.as_cuda_slice::<u8>().unwrap().device_ptr(&stream).0,
        _ => unreachable!("a CUDA bench"),
    };
    // SAFETY: written by the kernel below before anything reads it.
    let dst: CudaSlice<u8> = unsafe { stream.alloc::<u8>(span) }.unwrap();
    let dst_base = dst.device_ptr(&stream).0;
    let launches = 32;
    println!(
        "{:>8} {:>8} {:>10} {:>10}",
        "run", "records", "GB/s", "us/launch"
    );
    for run in RUNS {
        let n = span / run;
        let srcs: Vec<i64> = (0..n)
            .map(|i| (src_base + (i * run) as u64) as i64)
            .collect();
        let dsts: Vec<i64> = (0..n)
            .map(|i| (dst_base + ((i * SCATTER_STRIDE) % n * run) as u64) as i64)
            .collect();
        let lens: Vec<i64> = vec![run as i64; n];
        let srcs = cuda.memcpy_stod(&srcs).unwrap();
        let dsts = cuda.memcpy_stod(&dsts).unwrap();
        let lens = cuda.memcpy_stod(&lens).unwrap();
        let (sp, _a) = srcs.device_ptr(&stream);
        let (dp, _b) = dsts.device_ptr(&stream);
        let (lp, _c) = lens.device_ptr(&stream);
        let cu = stream.cu_stream() as *mut std::ffi::c_void;
        // SAFETY: every record names a live run of `src` and of `dst`.
        let launch = || unsafe {
            run_kv_migrate_copy(
                sp as *const i64,
                dp as *const i64,
                lp as *const i64,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                n as i32,
                cu,
            )
        };
        launch();
        device.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..launches {
            launch();
        }
        device.synchronize().unwrap();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{run:>8} {n:>8} {:>10.2} {:>10.1}",
            gbps(launches * span, secs),
            secs * 1e6 / launches as f64
        );
    }
}

/// A hot sequence of `tokens` tokens, `n_kv_head` heads of `head_dim`.
fn seeded(
    backing: &ChunkedKvBacking,
    device: &Device,
    seed: usize,
    n_kv_head: usize,
    tokens: usize,
    head_dim: usize,
) -> SealedSequence {
    let slot = backing.alloc_sequence().unwrap();
    backing.ensure_for_offset(slot, 0, tokens).unwrap();
    let n = n_kv_head * tokens * head_dim;
    let data: Vec<f32> = (0..n)
        .map(|i| ((seed * 7919 + i) % 4093) as f32 * 0.25)
        .collect();
    let k = Tensor::from_vec(data, (1, n_kv_head, tokens, head_dim), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(device)
        .unwrap();
    let v = (&k + 1.0).unwrap();
    backing.write_contiguous(slot, 0, &k, &v).unwrap();
    backing.set_len(slot, tokens);
    let sealed = backing.record_turn(slot).unwrap();
    backing.free_sequence(slot).unwrap();
    sealed
}

/// **A whole lift, production-shaped**: twelve layers, two KV heads, items of
/// 256 tokens, lifted from the warm tier through the production ring, timed to
/// the moment the device has landed it, with the host's split beside it.
#[test]
#[ignore = "bench: run alone with --nocapture"]
fn bench_whole_lift() {
    let _gpu = gpu_serial();
    let device = Device::new_cuda(0).unwrap();
    let stream = stream_of(&device);
    let (layers, heads, head_dim, tokens) = (12usize, 2usize, 128usize, 256usize);
    let mut ring = WarmLiftRing::new().unwrap();
    println!(
        "{:>6} {:>8} {:>10} {:>10}  split",
        "items", "MiB", "ms", "GB/s"
    );
    for items in [4usize, 16, 64] {
        let backings: Vec<ChunkedKvBacking> = (0..layers)
            .map(|_| {
                ChunkedKvBacking::new(items, heads, head_dim, DType::BF16, &device, tokens).unwrap()
            })
            .collect();
        let mut pinned: Option<PinnedBuf> = None;
        let warm: Vec<Vec<SealedSequence>> = backings
            .iter()
            .enumerate()
            .map(|(l, backing)| {
                let hot: Vec<SealedSequence> = (0..items)
                    .map(|i| seeded(backing, &device, l * 1000 + i, heads, tokens, head_dim))
                    .collect();
                let refs: Vec<&SealedSequence> = hot.iter().collect();
                backing
                    .migrate_sealed_to_cpu_batch_async(&device, &stream, &mut pinned, &refs)
                    .unwrap()
            })
            .collect();
        let refs: Vec<Vec<&SealedSequence>> = warm.iter().map(|w| w.iter().collect()).collect();
        device.synchronize().unwrap();
        let t = Instant::now();
        let lift = lift_warm_to_gpu(&backings, &mut ring, &device, &refs).unwrap();
        device.synchronize().unwrap();
        let secs = t.elapsed().as_secs_f64();
        let bytes = lift.split.bytes as usize;
        println!(
            "{items:>6} {:>8} {:>10.2} {:>10.2}  {:?}",
            bytes >> 20,
            secs * 1e3,
            gbps(bytes, secs),
            lift.split
        );
    }
}
