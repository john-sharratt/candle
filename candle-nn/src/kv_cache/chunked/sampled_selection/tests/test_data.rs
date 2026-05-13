//! Reusable staging helpers for sampled-selection tests and benchmarks.
//!
//! `load_cpu_arena`  — loads the dump, stages R16+F16 arenas on CPU,
//!                     applies 1:CPU_SAMPLE_STRIDE sampling to keep runtime short.
//!
//! `load_gpu_arena`  — loads the dump, stages R16+F16 arenas on the CUDA device,
//!                     builds `PagedSelectionGpuInputs` over the full dataset.
//!                     Only compiled when `--features cuda` is set.

use super::*;
use crate::kv_cache::chunked::sampled_selection::KvSampler;
#[cfg(feature = "cuda")]
use crate::kv_cache::chunked::sampled_selection::PagedSelectionGpuInputs;
use crate::kv_cache::{ChunkedKvBacking, KvFormat};
use candle::{DType, Device};
use std::time::Instant;

/// Batch size in KV blocks for the benchmark inner loop.
pub(super) const BATCH_KV_BLOCKS: usize = 4_096;

/// Stride used when subsampling the dump for CPU benchmarks.
pub(super) const CPU_SAMPLE_STRIDE: usize = 50;

// ─── Result types ────────────────────────────────────────────────────────────

/// All data needed to run a CPU- or GPU-path benchmark or test.
///
/// `paged_inputs_all` is `Some` on the GPU path and `None` on the CPU path.
pub(super) struct StagedArena {
    pub backing: ChunkedKvBacking,
    pub n_chunks: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub batch_chunk_count: usize,
    pub sampler: KvSampler,
    #[cfg(feature = "cuda")]
    pub paged_inputs_all: Option<PagedSelectionGpuInputs>,
}

// ─── Loaders ─────────────────────────────────────────────────────────────────

/// Load and stage R16+F16 KV arenas on CPU from the dump file.
///
/// Applies 1:CPU_SAMPLE_STRIDE strided sampling so the benchmark completes
/// in a reasonable time on a CPU-only build.
///
/// Returns `None` (printing a SKIP message) if no dump file is found.
/// Records `io.load_dump` and `io.stage_kv_arenas` into `benchmark_result`.
pub(super) fn load_cpu_arena(
    benchmark_result: &mut SampledSelectionBenchmarkResult,
) -> Option<StagedArena> {
    let load_start = Instant::now();
    let (header, chunks) = 'load: {
        if let Some(path) = r16_dump_path() {
            match load_dump(&path) {
                Some((hdr, ch)) if !ch.is_empty() => {
                    println!("CPU arena: R16 dump → {} chunks", ch.len());
                    break 'load (hdr, ch);
                }
                Some(_) => {
                    println!(
                        "CPU arena: R16 dump has 0 chunks (header-only capture), \
                         falling back to float dump with Q=0"
                    );
                }
                None => {
                    println!("CPU arena: R16 dump failed to parse, trying float dump");
                }
            }
        }
        match dump_path().and_then(|p| load_dump(&p)) {
            Some((hdr, ch)) => {
                println!("CPU arena: float dump → {} chunks", ch.len());
                (hdr, ch)
            }
            None => {
                println!(
                    "SKIP: no usable dump found at {R16_DUMP_REL_PATH} or {DUMP_REL_PATH}"
                );
                return None;
            }
        }
    };
    benchmark_result.record_duration("benchmark.io.load_dump", load_start.elapsed(), 1);

    if chunks.is_empty() {
        println!("SKIP: dump has 0 chunks");
        return None;
    }

    let blocks_per_chunk = header.n_kv_head * header.head_dim;
    let available_blocks = chunks.len() * blocks_per_chunk;

    // Subsample 1:CPU_SAMPLE_STRIDE to keep runtime under ~10 s.
    let sampled_chunks: Vec<&_> = chunks.iter().step_by(CPU_SAMPLE_STRIDE).collect();
    let n_chunks = sampled_chunks.len();
    let effective_blocks = n_chunks * blocks_per_chunk;
    let batch_chunk_count = (BATCH_KV_BLOCKS / blocks_per_chunk).max(1);
    let total_batches = n_chunks.div_ceil(batch_chunk_count);

    println!(
        "CPU arena test data: {} / {} KV blocks, {} chunks (1:{} sample), \
         {} heads, head_dim={} ({} batches)",
        effective_blocks, available_blocks, n_chunks, CPU_SAMPLE_STRIDE,
        header.n_kv_head, header.head_dim, total_batches
    );

    let cpu_device = Device::Cpu;
    let stage_start = Instant::now();
    let backing = ChunkedKvBacking::new_with_format(
        n_chunks,
        header.n_kv_head,
        header.head_dim,
        KvFormat::Quantized(crate::kv_cache::QuantFormat::R16),
        KvFormat::Float(DType::F16),
        &cpu_device,
        header.chunk_size,
    )
    .expect("create CPU R16+F16 backing");

    for (slot, chunk) in sampled_chunks.iter().enumerate() {
        let zero_q;
        let q = match chunk.q.as_ref() {
            Some(q) => q.as_slice(),
            None => {
                zero_q = vec![0.0f32; chunk.k.len()];
                zero_q.as_slice()
            }
        };
        let k_bytes = pack_r16_blocks(&chunk.k, q);
        let v_bytes = pack_f16(&chunk.v);
        backing
            .write_raw_sealed_chunk(
                slot,
                0,
                &k_bytes,
                &v_bytes,
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            )
            .expect("write chunk to CPU R16+F16 backing");
    }
    benchmark_result.record_duration("benchmark.io.stage_kv_arenas", stage_start.elapsed(), 1);

    let candidates = candidate_formats();
    let sampler = KvSampler::new_for_level_cpu(5, &candidates);

    Some(StagedArena {
        backing,
        n_chunks,
        n_kv_head: header.n_kv_head,
        head_dim: header.head_dim,
        batch_chunk_count,
        sampler,
        #[cfg(feature = "cuda")]
        paged_inputs_all: None,
    })
}

/// Load and stage R16+F16 KV arenas on the CUDA device from the dump file.
///
/// Prefers the R16 dump; falls back to the float dump (with synthetic Q=0) when
/// the R16 file was captured without block data (n_chunks == 0 — a session that
/// recorded the token sequence but no KV writes).
///
/// Uses the full dataset (no sampling). Builds `PagedSelectionGpuInputs` for
/// all chunks so the benchmark can slice batch windows without re-uploading.
///
/// Returns `None` (printing a SKIP message) if no dump or CUDA device is available.
/// Records `io.load_dump` and `io.stage_kv_arenas` into `benchmark_result`.
#[cfg(feature = "cuda")]
pub(super) fn load_gpu_arena(
    benchmark_result: &mut SampledSelectionBenchmarkResult,
) -> Option<StagedArena> {
    // Try R16 dump first.  If it parses but has 0 chunks (header-only capture),
    // fall back to the float dump using synthetic zero Q arrays.
    let load_start = Instant::now();
    let (header, chunks) = 'load: {
        if let Some(path) = r16_dump_path() {
            match load_dump(&path) {
                Some((hdr, ch)) if !ch.is_empty() => {
                    println!("GPU arena: R16 dump → {} chunks", ch.len());
                    break 'load (hdr, ch);
                }
                Some(_) => {
                    println!(
                        "GPU arena: R16 dump has 0 chunks (header-only capture), \
                         falling back to float dump with Q=0"
                    );
                }
                None => {
                    println!("GPU arena: R16 dump failed to parse, trying float dump");
                }
            }
        }
        match dump_path().and_then(|p| load_dump(&p)) {
            Some((hdr, ch)) => {
                println!("GPU arena: float dump → {} chunks", ch.len());
                (hdr, ch)
            }
            None => {
                println!(
                    "SKIP: no usable dump found at {R16_DUMP_REL_PATH} or {DUMP_REL_PATH}"
                );
                return None;
            }
        }
    };
    benchmark_result.record_duration("benchmark.io.load_dump", load_start.elapsed(), 1);

    if chunks.is_empty() {
        println!("SKIP: dump has 0 chunks");
        return None;
    }

    let cuda_init_start = Instant::now();
    let dev = match Device::cuda_if_available(0) {
        Ok(Device::Cuda(dev)) => dev,
        _ => {
            println!("SKIP: CUDA device required for GPU arena staging");
            return None;
        }
    };
    benchmark_result.record_duration("benchmark.io.init_cuda_device", cuda_init_start.elapsed(), 1);

    let blocks_per_chunk = header.n_kv_head * header.head_dim;
    let n_chunks = chunks.len();
    let batch_chunk_count = (BATCH_KV_BLOCKS / blocks_per_chunk).max(1);
    let total_batches = n_chunks.div_ceil(batch_chunk_count);

    println!(
        "GPU arena test data: {} KV blocks, {} chunks, {} heads, head_dim={} ({} batches)",
        n_chunks * blocks_per_chunk,
        n_chunks,
        header.n_kv_head,
        header.head_dim,
        total_batches
    );

    let stage_start = Instant::now();
    let backing = ChunkedKvBacking::new_with_format(
        n_chunks,
        header.n_kv_head,
        header.head_dim,
        KvFormat::Quantized(crate::kv_cache::QuantFormat::R16),
        KvFormat::Float(DType::F16),
        &Device::Cuda(dev.clone()),
        header.chunk_size,
    )
    .expect("create GPU R16+F16 backing");

    for (slot, chunk) in chunks.iter().enumerate() {
        let zero_q;
        let q = match chunk.q.as_ref() {
            Some(q) => q.as_slice(),
            None => {
                zero_q = vec![0.0f32; chunk.k.len()];
                zero_q.as_slice()
            }
        };
        let k_bytes = pack_r16_blocks(&chunk.k, q);
        let v_bytes = pack_f16(&chunk.v);
        backing
            .write_raw_sealed_chunk(
                slot,
                0,
                &k_bytes,
                &v_bytes,
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            )
            .expect("load dump chunk into GPU R16+F16 backing");
    }

    let all_slots = (0..n_chunks).collect::<Vec<_>>();
    let paged_inputs_all =
        PagedSelectionGpuInputs::from_backing(&backing, &all_slots, None, &dev)
            .expect("bind staged paged KV backing");
    benchmark_result.record_duration("benchmark.io.stage_kv_arenas", stage_start.elapsed(), 1);

    let candidates = candidate_formats();
    let sampler_init_start = Instant::now();
    let sampler = KvSampler::new_for_level(5, &candidates, &dev)
        .expect("upload sampler constants to GPU");
    benchmark_result.record_duration("benchmark.io.init_sampler", sampler_init_start.elapsed(), 1);

    Some(StagedArena {
        backing,
        n_chunks,
        n_kv_head: header.n_kv_head,
        head_dim: header.head_dim,
        batch_chunk_count,
        sampler,
        paged_inputs_all: Some(paged_inputs_all),
    })
}
