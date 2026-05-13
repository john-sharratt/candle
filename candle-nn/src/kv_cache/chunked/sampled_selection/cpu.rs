//! CPU sampling kernel — mirrors the GPU `sample_quant_errors_paged` kernel.
//!
//! For each 32-element block, evaluates all candidate formats using a single
//! natural quantizer scale (no outer scale search).  Error metric:
//!   `max_abs_error(block, recon) / per_head_amax`
//! — identical to the GPU kernel formula.

use super::ops::{compute_head_amax, normalised_error, safe_head_scale};
use super::params::SELECT_BLOCK;
use super::profile::sampled_profile_record_duration;
use super::{ErrorSurface, SampleFormat, SampleSide, SampledSelectionBenchmarkResult};
use candle::Result;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub fn sample_error_surface_cpu(
    values: &[f32],
    n_batch: usize,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    _sample_token: usize,
    candidates: &[SampleFormat],
    side: SampleSide,
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> Result<ErrorSurface> {
    let total_start = benchmark_result.as_ref().map(|_| Instant::now());
    if chunk_size != SELECT_BLOCK {
        candle::bail!("sample_error_surface_cpu expects chunk_size=32, got {chunk_size}");
    }
    let expected = n_batch
        .checked_mul(n_head)
        .and_then(|v| v.checked_mul(chunk_size))
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| candle::Error::Msg("shape overflow in sample_error_surface_cpu".into()))?;
    if values.len() != expected {
        candle::bail!(
            "sample_error_surface_cpu length mismatch: got {}, expected {}",
            values.len(),
            expected
        );
    }

    let n_q = candidates.len();
    let n_cells = n_batch * head_dim * n_q * n_head;
    let mut data = vec![0.0f32; n_cells];

    let compute_start = benchmark_result.as_ref().map(|_| Instant::now());
    for b in 0..n_batch {
        for h in 0..n_head {
            let bh = b * n_head + h;
            let head_start = bh * head_dim * chunk_size;
            let head_end = head_start + head_dim * chunk_size;
            let head_amax = compute_head_amax(&values[head_start..head_end]);
            let head_scale = safe_head_scale(head_amax);

            for d in 0..head_dim {
                let mut block = [0.0f32; SELECT_BLOCK];
                for (t, dst) in block.iter_mut().enumerate() {
                    let idx = (bh * head_dim + d) * chunk_size + t;
                    *dst = values[idx];
                }
                for (qidx, &fmt) in candidates.iter().enumerate() {
                    let flat = ((b * head_dim + d) * n_q + qidx) * n_head + h;
                    let recon = fmt.apply_quant(&block);
                    data[flat] = normalised_error(&block, &recon, head_scale);
                }
            }
        }
    }

    if let Some(start) = compute_start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            match side {
                SampleSide::Key => "quantization.key.surface.cpu.compute",
                SampleSide::Value => "quantization.value.surface.cpu.compute",
            },
            start.elapsed(),
            1,
        );
    }
    if let Some(start) = total_start {
        sampled_profile_record_duration(
            benchmark_result.as_deref_mut(),
            match side {
                SampleSide::Key => "quantization.key.surface.cpu.total",
                SampleSide::Value => "quantization.value.surface.cpu.total",
            },
            start.elapsed(),
            1,
        );
    }

    Ok(ErrorSurface {
        n_batch,
        n_head,
        n_dim: head_dim,
        n_quant: candidates.len(),
        chunk_size,
        side,
        data,
        q_relevance: None,
    })
}
