use super::test_data::load_cpu_arena;
#[cfg(feature = "cuda")]
use super::test_data::load_gpu_arena;
use super::*;
#[cfg(feature = "cuda")]
use crate::kv_cache::chunked::sampled_selection::{
    sample_quantization_sweep_kv_paged, PagedSelectionGpuInputs,
};
use crate::kv_cache::chunked::sampled_selection::{KvSamplerInputs, PagedSelectionCpuInputs};
use std::time::Instant;

// ─── inner loops ─────────────────────────────────────────────────────────────

fn run_cpu_benchmark(
    backing: &crate::kv_cache::ChunkedKvBacking,
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    batch_chunk_count: usize,
    sampler: &crate::kv_cache::chunked::sampled_selection::KvSampler,
    benchmark_result: &mut SampledSelectionBenchmarkResult,
) {
    let all_slots: Vec<usize> = (0..n_chunks).collect();
    for batch_slots in all_slots.chunks(batch_chunk_count) {
        let batch_start = Instant::now();
        let cpu_inputs = PagedSelectionCpuInputs {
            backing,
            batch_slots,
        };
        let (k_results, v_results) = sampler
            .run_sweep(
                &KvSamplerInputs::Cpu(cpu_inputs),
                n_kv_head,
                head_dim,
                0,
                Some(benchmark_result),
            )
            .expect("CPU KV quantization sweep");
        assert_eq!(k_results.surface.n_batch, batch_slots.len());
        assert_eq!(v_results.surface.n_batch, batch_slots.len());
        benchmark_result.record_duration("benchmark.batch.total", batch_start.elapsed(), 1);
    }
}

#[cfg(feature = "cuda")]
fn run_gpu_benchmark(
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    batch_chunk_count: usize,
    sampler: &crate::kv_cache::chunked::sampled_selection::KvSampler,
    paged_inputs_all: &PagedSelectionGpuInputs,
    benchmark_result: &mut SampledSelectionBenchmarkResult,
) {
    let all_slots: Vec<usize> = (0..n_chunks).collect();
    for (batch_idx, batch_slots) in all_slots.chunks(batch_chunk_count).enumerate() {
        let batch_start = Instant::now();
        let bind_start = Instant::now();
        let paged_inputs = paged_inputs_all
            .select_chunks(batch_idx * batch_chunk_count, batch_slots.len(), None)
            .expect("bind batch window to paged KV arenas");
        benchmark_result.record_duration("benchmark.batch.bind_arenas", bind_start.elapsed(), 1);
        let (k_results, v_results) = sample_quantization_sweep_kv_paged(
            &paged_inputs,
            sampler,
            batch_slots.len(),
            n_kv_head,
            head_dim,
            0,
            Some(benchmark_result),
        )
        .expect("GPU KV quantization sweep");
        assert_eq!(k_results.surface.n_batch, batch_slots.len());
        assert_eq!(v_results.surface.n_batch, batch_slots.len());
        benchmark_result.record_duration("benchmark.batch.total", batch_start.elapsed(), 1);
    }
}

// ─── CPU benchmark ───────────────────────────────────────────────────────────

#[test]
#[ignore]
fn cpu_arena_benchmark_full_run() {
    let workflow_start = Instant::now();
    let mut benchmark_result = SampledSelectionBenchmarkResult::default();

    let data = match load_cpu_arena(&mut benchmark_result) {
        Some(d) => d,
        None => return,
    };

    run_cpu_benchmark(
        &data.backing,
        data.n_chunks,
        data.n_kv_head,
        data.head_dim,
        data.batch_chunk_count,
        &data.sampler,
        &mut benchmark_result,
    );

    benchmark_result.record_duration("benchmark.total", workflow_start.elapsed(), 1);
    println!(
        "{}",
        benchmark_result.report("CPU Arena Sampled-Selection Profile")
    );
}

// ─── GPU benchmark ───────────────────────────────────────────────────────────

#[test]
#[ignore]
fn gpu_kernel_benchmark_full_run() {
    #[cfg(feature = "cuda")]
    gpu_kernel_benchmark_full_run_impl();
    #[cfg(not(feature = "cuda"))]
    cpu_arena_benchmark_full_run();
}

#[cfg(feature = "cuda")]
fn gpu_kernel_benchmark_full_run_impl() {
    let workflow_start = Instant::now();
    let mut benchmark_result = SampledSelectionBenchmarkResult::default();

    let data = match load_gpu_arena(&mut benchmark_result) {
        Some(d) => d,
        None => return,
    };

    let paged_inputs_all = data
        .paged_inputs_all
        .as_ref()
        .expect("GPU arena must have paged_inputs_all");
    run_gpu_benchmark(
        data.n_chunks,
        data.n_kv_head,
        data.head_dim,
        data.batch_chunk_count,
        &data.sampler,
        paged_inputs_all,
        &mut benchmark_result,
    );

    benchmark_result.record_duration("benchmark.total", workflow_start.elapsed(), 1);
    println!(
        "{}",
        benchmark_result.report("GPU Sampled-Selection Full Workflow Profile")
    );
}
