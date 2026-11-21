//! Launches the fused K and V error-surface kernel that scores each candidate
//! format over a paged chunk.
//!
//! One launch computes both surfaces, so K and V are scored in the same pass over
//! the paged data. The two candidate lists must be identical, since a single set
//! of candidate indices addresses both surfaces. V errors are weighted by the
//! query-key relevance of each position, so a V position that many queries attend
//! to has its error amplified and is held to a stricter format.

use super::*;

#[allow(clippy::too_many_arguments)]
pub fn sample_error_surface_gpu_paged(
    per_head_table_buf: &GpuBuf,
    head_gids_buf: &GpuBuf,
    candidates: &[candle::quantized::GgmlDType],
    sample_token: usize,
    side: SampleSide,
    num_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    arena_chunks: usize,
    dev: &candle::CudaDevice,
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> Result<ErrorSurface> {
    let kernel_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
    let (gpu, gpu_q_relevance) = sample_quant_errors_paged(
        per_head_table_buf.dev_ptr(),
        head_gids_buf.dev_ptr(),
        candidates,
        sample_token,
        matches!(side, SampleSide::Key),
        num_chunks,
        n_kv_head,
        head_dim,
        arena_chunks,
        dev,
    )?;
    if let Some(start) = kernel_start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            match side {
                SampleSide::Key => "quantization.key.surface.gpu.kernel",
                SampleSide::Value => "quantization.value.surface.gpu.kernel",
            },
            start.elapsed(),
            1,
        );
    }
    let download_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
    let data = dev.memcpy_dtov(&gpu)?;
    let q_relevance = dev.memcpy_dtov(&gpu_q_relevance)?;
    if let Some(start) = download_start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            match side {
                SampleSide::Key => "quantization.key.surface.gpu.download",
                SampleSide::Value => "quantization.value.surface.gpu.download",
            },
            start.elapsed(),
            1,
        );
    }
    Ok(ErrorSurface {
        n_batch: num_chunks,
        n_head: n_kv_head,
        n_dim: head_dim,
        n_quant: candidates.len(),
        chunk_size: SELECT_BLOCK,
        side,
        data,
        q_relevance: Some(q_relevance),
    })
}

/// Fused variant that computes K and V error surfaces in a single kernel launch.
///
/// Returns `(k_surface, v_surface)`. The K and V candidate lists must be
/// identical. V errors are weighted by Q·K relevance — highly attended V
/// positions have their errors amplified, making V compression stricter there.
#[allow(clippy::too_many_arguments)]
pub fn sample_error_surface_kv_paged(
    per_head_table_buf: &GpuBuf,
    head_gids_buf: &GpuBuf,
    candidates: &[candle::quantized::GgmlDType],
    sample_token: usize,
    num_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    arena_chunks: usize,
    dev: &candle::CudaDevice,
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> Result<(ErrorSurface, ErrorSurface)> {
    use candle::quantized::cuda::sample_quant_errors_kv_paged;

    let kernel_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
    let (k_gpu, v_gpu) = sample_quant_errors_kv_paged(
        per_head_table_buf.dev_ptr(),
        head_gids_buf.dev_ptr(),
        candidates,
        sample_token,
        num_chunks,
        n_kv_head,
        head_dim,
        arena_chunks,
        dev,
    )?;
    if let Some(start) = kernel_start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            "quantization.kv.surface.gpu.kernel",
            start.elapsed(),
            1,
        );
    }

    let download_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
    let k_data = dev.memcpy_dtov(&k_gpu)?;
    let v_data = dev.memcpy_dtov(&v_gpu)?;
    if let Some(start) = download_start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            "quantization.kv.surface.gpu.download",
            start.elapsed(),
            1,
        );
    }

    let k_surface = ErrorSurface {
        n_batch: num_chunks,
        n_head: n_kv_head,
        n_dim: head_dim,
        n_quant: candidates.len(),
        chunk_size: SELECT_BLOCK,
        side: SampleSide::Key,
        data: k_data,
        q_relevance: None,
    };
    let v_surface = ErrorSurface {
        n_batch: num_chunks,
        n_head: n_kv_head,
        n_dim: head_dim,
        n_quant: candidates.len(),
        chunk_size: SELECT_BLOCK,
        side: SampleSide::Value,
        data: v_data,
        q_relevance: None,
    };
    Ok((k_surface, v_surface))
}
