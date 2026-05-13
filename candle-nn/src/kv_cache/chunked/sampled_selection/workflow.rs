use super::profile::sampled_profile_record_duration;
use super::{
    batch_select_and_summarize, sample_error_surface,
    CompressionSummary, ErrorSurface, SampleFormat, SampleSide, SampledSelectionBenchmarkResult,
};
#[cfg(feature = "cuda")]
use super::{
    sample_error_surface_gpu_paged, KvSampler, KvSamplerInputs, PagedSelectionGpuInputs,
};

#[derive(Debug, Clone)]
pub struct SampleQuantizationResult {
    /// Per-cell winner index: one entry per `(batch, head, dim)` triplet.
    /// Stored as `u8` because values only ever index into `candidates` (≤ 255).
    pub winners: Vec<u8>,
    pub summary: CompressionSummary,
}

#[derive(Debug, Clone)]
pub struct SampleQuantizationSweepResult {
    pub surface: ErrorSurface,
    pub levels: Vec<SampleQuantizationResult>,
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn sample_quantization_paged(
    gpu_inputs: &PagedSelectionGpuInputs,
    n_batch: usize,
    n_head: usize,
    head_dim: usize,
    sample_token: usize,
    candidates: &[SampleFormat],
    side: SampleSide,
    threshold: f32,
    benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> candle::Result<SampleQuantizationSweepResult> {
    sample_quantization_sweep_paged(
        gpu_inputs,
        n_batch,
        n_head,
        head_dim,
        sample_token,
        candidates,
        side,
        &[threshold],
        benchmark_result,
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn sample_quantization_sweep_paged(
    gpu_inputs: &PagedSelectionGpuInputs,
    n_batch: usize,
    n_head: usize,
    head_dim: usize,
    sample_token: usize,
    candidates: &[SampleFormat],
    side: SampleSide,
    thresholds: &[f32],
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> candle::Result<SampleQuantizationSweepResult> {
    let workflow_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
    let surface_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
    let ggml_candidates = candidates
        .iter()
        .copied()
        .map(SampleFormat::to_ggml_dtype)
        .collect::<Vec<_>>();

    let surface = sample_error_surface_gpu_paged(
        gpu_inputs.per_head_table_buf(),
        gpu_inputs.head_gids_buf(),
        &ggml_candidates,
        sample_token,
        side,
        n_batch,
        n_head,
        head_dim,
        gpu_inputs.arena_chunks(),
        gpu_inputs.dev(),
        benchmark_result.as_deref_mut(),
    )?;

    if let Some(start) = surface_start {
        let scope = match side {
            SampleSide::Key => "quantization.key.surface.total",
            SampleSide::Value => "quantization.value.surface.total",
        };
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            scope,
            start.elapsed(),
            1,
        );
    }

    let levels = batch_select_and_summarize(&surface, thresholds, candidates, benchmark_result.as_deref_mut())?
        .into_iter()
        .map(|(winners, summary)| SampleQuantizationResult { winners, summary })
        .collect::<Vec<_>>();

    if let Some(start) = workflow_start {
        let scope = match side {
            SampleSide::Key => "quantization.key.total",
            SampleSide::Value => "quantization.value.total",
        };
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            scope,
            start.elapsed(),
            1,
        );
    }

    Ok(SampleQuantizationSweepResult { surface, levels })
}

/// Fused KV sweep — samples K and V error surfaces in a single GPU kernel launch,
/// then selects winner indices on the GPU (no large surface download), and finally
/// summarises both K and V in parallel on the CPU.
///
/// `sampler` holds all constant GPU state (candidates + thresholds) uploaded once
/// at session initialisation; no H→D copies for those on each call.
/// Returns `(k_result, v_result)`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn sample_quantization_sweep_kv_paged(
    gpu_inputs: &PagedSelectionGpuInputs,
    sampler: &KvSampler,
    _n_batch: usize,
    n_head: usize,
    head_dim: usize,
    sample_token: usize,
    benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> candle::Result<(SampleQuantizationSweepResult, SampleQuantizationSweepResult)> {
    sampler.run_sweep(
        &KvSamplerInputs::Gpu(gpu_inputs),
        n_head,
        head_dim,
        sample_token,
        benchmark_result,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sample_quantization(
    values: &[f32],
    n_batch: usize,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    sample_token: usize,
    candidates: &[SampleFormat],
    side: SampleSide,
    threshold: f32,
    benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> candle::Result<SampleQuantizationSweepResult> {
    sample_quantization_sweep(
        values,
        n_batch,
        n_head,
        chunk_size,
        head_dim,
        sample_token,
        candidates,
        side,
        &[threshold],
        benchmark_result,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sample_quantization_sweep(
    values: &[f32],
    n_batch: usize,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    sample_token: usize,
    candidates: &[SampleFormat],
    side: SampleSide,
    thresholds: &[f32],
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> candle::Result<SampleQuantizationSweepResult> {
    let workflow_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
    let surface_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());

    let surface = sample_error_surface(
        values,
        n_batch,
        n_head,
        chunk_size,
        head_dim,
        sample_token,
        candidates,
        side,
        benchmark_result.as_deref_mut(),
    )?;

    if let Some(start) = surface_start {
        let scope = match side {
            SampleSide::Key => "quantization.key.surface.total",
            SampleSide::Value => "quantization.value.surface.total",
        };
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            scope,
            start.elapsed(),
            1,
        );
    }

    let levels = batch_select_and_summarize(&surface, thresholds, candidates, benchmark_result.as_deref_mut())?
        .into_iter()
        .map(|(winners, summary)| SampleQuantizationResult { winners, summary })
        .collect::<Vec<_>>();

    if let Some(start) = workflow_start {
        let scope = match side {
            SampleSide::Key => "quantization.key.total",
            SampleSide::Value => "quantization.value.total",
        };
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            scope,
            start.elapsed(),
            1,
        );
    }

    Ok(SampleQuantizationSweepResult { surface, levels })
}
