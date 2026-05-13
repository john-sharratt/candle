#![allow(dead_code)]

mod cpu; // CPU sampling kernel — mirrors GPU algorithm
#[cfg(feature = "cuda")]
mod gpu; // GPU sampling kernel
mod ops; // shared utilities: selection, palette reduction, round-trip quant
pub mod params;
mod profile;
mod sampler;
mod types;
mod workflow;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use self::cpu::sample_error_surface_cpu;
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub use self::gpu::{
    sample_error_surface_gpu_paged, sample_error_surface_kv_paged, GpuQualityAggregation,
    KvSamplerGpu, PagedSelectionGpuInputs, SelectionBackend,
};
#[allow(unused_imports)]
pub use self::ops::{
    batch_select_and_summarize, batch_summarize_from_winners, cpu_palette4_reduce,
    cpu_parallel_kernel_map, cpu_parallel_kernel_range, model_compression_from_surface,
    select_smallest_passing,
};
#[allow(unused_imports)]
pub use self::params::{
    k_threshold_scaled_rust, KvErrorThresholdFactors, DEFAULT_CALIBRATION_ARENA_CHUNKS,
    DEFAULT_REPORT_ARENA_CHUNKS, ERROR_MARGIN_ABS, LLAMA_KV_FACTORS,
    PRODUCTION_K_QREL_HIGH_THRESHOLDS, PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_LEVEL_TIER,
    PRODUCTION_V_QREL_HIGH_THRESHOLDS, PRODUCTION_V_QREL_LOW_THRESHOLDS, QWEN3_8B_KV_FACTORS,
    QWEN3_MOE_KV_FACTORS, SELECT_BLOCK,
};
#[allow(unused_imports)]
pub use self::profile::SampledSelectionBenchmarkResult;
#[allow(unused_imports)]
pub use self::sampler::{KvSampler, KvSamplerInputs, PagedSelectionCpuInputs};
#[allow(unused_imports)]
pub use self::types::{CompressionSummary, ErrorSurface, SampleFormat, SampleSide};
#[allow(unused_imports)]
pub use self::workflow::{
    sample_quantization, sample_quantization_sweep, SampleQuantizationResult,
    SampleQuantizationSweepResult,
};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub use self::workflow::{
    sample_quantization_paged, sample_quantization_sweep_kv_paged, sample_quantization_sweep_paged,
};

#[allow(clippy::too_many_arguments)]
pub fn sample_error_surface(
    values: &[f32],
    n_batch: usize,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    sample_token: usize,
    candidates: &[SampleFormat],
    side: SampleSide,
    mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
) -> candle::Result<ErrorSurface> {
    self::cpu::sample_error_surface_cpu(
        values,
        n_batch,
        n_head,
        chunk_size,
        head_dim,
        sample_token,
        candidates,
        side,
        benchmark_result.as_deref_mut(),
    )
}
