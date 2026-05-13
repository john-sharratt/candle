//! [`KvSampler`] — backend-agnostic KV quantization sampler.
//!
//! On CUDA builds the sampler pre-uploads constant state (candidates, thresholds)
//! to the device once on construction and reuses it on every call.  On CPU-only
//! builds the same struct stores the constants in host memory and drives the CPU
//! sampling path through arena-backed [`PagedSelectionCpuInputs`].

use super::params::SELECT_BLOCK;
use super::profile::sampled_profile_record_duration;
use super::workflow::{SampleQuantizationResult, SampleQuantizationSweepResult};
use super::{
    batch_select_and_summarize, sample_error_surface_cpu,
    CompressionSummary, ErrorSurface, SampleFormat, SampleSide, SampledSelectionBenchmarkResult,
};
use crate::kv_cache::ChunkedKvBacking;
use candle::Result;

#[cfg(feature = "cuda")]
use super::gpu::{KvSamplerGpu, PagedSelectionGpuInputs};

// ─── CPU inputs ──────────────────────────────────────────────────────────────

/// Paged KV inputs for CPU-side sampling.
///
/// Wraps a reference to a [`ChunkedKvBacking`] that stores data in Float arenas
/// plus the list of batch slot indices to process in this sweep.
pub struct PagedSelectionCpuInputs<'a> {
    pub backing: &'a ChunkedKvBacking,
    /// Logical batch slots (one per chunk window position) to process.
    pub batch_slots: &'a [usize],
}

// ─── Backend dispatch ────────────────────────────────────────────────────────

/// Dispatch target passed to [`KvSampler::run_sweep`].
pub enum KvSamplerInputs<'a> {
    Cpu(PagedSelectionCpuInputs<'a>),
    #[cfg(feature = "cuda")]
    Gpu(&'a PagedSelectionGpuInputs),
}

// ─── KvSampler ───────────────────────────────────────────────────────────────

/// Backend-agnostic KV quantization sampler.
///
/// Constant state (candidates, thresholds, and on CUDA: their device copies) is
/// uploaded once at construction and reused across every call to [`run_sweep`].
pub struct KvSampler {
    candidates: Vec<SampleFormat>,
    k_thresholds: Vec<f32>,
    v_thresholds: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<KvSamplerGpu>,
}

impl KvSampler {
    /// Construct for GPU use — uploads candidates and thresholds to device once.
    #[cfg(feature = "cuda")]
    pub fn new(
        candidates: &[SampleFormat],
        k_thresholds: &[f32],
        v_thresholds: &[f32],
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        let gpu = KvSamplerGpu::new(candidates, k_thresholds, v_thresholds, dev)?;
        Ok(Self {
            candidates: candidates.to_vec(),
            k_thresholds: k_thresholds.to_vec(),
            v_thresholds: v_thresholds.to_vec(),
            gpu: Some(gpu),
        })
    }

    /// Construct for CPU-only use — no GPU uploads.  Available on all feature combinations.
    pub fn new_cpu(candidates: &[SampleFormat], k_thresholds: &[f32], v_thresholds: &[f32]) -> Self {
        Self {
            candidates: candidates.to_vec(),
            k_thresholds: k_thresholds.to_vec(),
            v_thresholds: v_thresholds.to_vec(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    /// Construct for CPU use — no GPU uploads.
    #[cfg(not(feature = "cuda"))]
    pub fn new(candidates: &[SampleFormat], k_thresholds: &[f32], v_thresholds: &[f32]) -> Self {
        Self::new_cpu(candidates, k_thresholds, v_thresholds)
    }

    /// Construct for GPU use scoped to a single compression level.
    ///
    /// Selects the production K/V hi/lo thresholds for one level instead of
    /// sweeping all 11 levels.
    #[cfg(feature = "cuda")]
    pub fn new_for_level(
        level: u8,
        candidates: &[SampleFormat],
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        let idx = level.min(10) as usize;
        let k_thresholds = [
            super::params::PRODUCTION_K_QREL_HIGH_THRESHOLDS[idx],
            super::params::PRODUCTION_K_QREL_LOW_THRESHOLDS[idx],
        ];
        let v_thresholds = [
            super::params::PRODUCTION_V_QREL_HIGH_THRESHOLDS[idx],
            super::params::PRODUCTION_V_QREL_LOW_THRESHOLDS[idx],
        ];
        let gpu = KvSamplerGpu::new(candidates, &k_thresholds, &v_thresholds, dev)?;
        Ok(Self {
            candidates: candidates.to_vec(),
            k_thresholds: k_thresholds.to_vec(),
            v_thresholds: v_thresholds.to_vec(),
            gpu: Some(gpu),
        })
    }

    /// CPU-only version of [`new_for_level`].  Available on all feature combinations.
    pub fn new_for_level_cpu(level: u8, candidates: &[SampleFormat]) -> Self {
        let idx = level.min(10) as usize;
        let k_thresholds = [
            super::params::PRODUCTION_K_QREL_HIGH_THRESHOLDS[idx],
            super::params::PRODUCTION_K_QREL_LOW_THRESHOLDS[idx],
        ];
        let v_thresholds = [
            super::params::PRODUCTION_V_QREL_HIGH_THRESHOLDS[idx],
            super::params::PRODUCTION_V_QREL_LOW_THRESHOLDS[idx],
        ];
        Self {
            candidates: candidates.to_vec(),
            k_thresholds: k_thresholds.to_vec(),
            v_thresholds: v_thresholds.to_vec(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn candidates(&self) -> &[SampleFormat] {
        &self.candidates
    }

    pub fn k_thresholds(&self) -> &[f32] {
        &self.k_thresholds
    }

    pub fn v_thresholds(&self) -> &[f32] {
        &self.v_thresholds
    }


    /// Run a full KV quantization sweep and return per-threshold summaries.
    ///
    /// Dispatches to the GPU or CPU path according to `inputs`.
    pub fn run_sweep(
        &self,
        inputs: &KvSamplerInputs<'_>,
        n_head: usize,
        head_dim: usize,
        sample_token: usize,
        benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
    ) -> Result<(SampleQuantizationSweepResult, SampleQuantizationSweepResult)> {
        match inputs {
            KvSamplerInputs::Cpu(cpu_inputs) => {
                self.run_sweep_cpu(cpu_inputs, n_head, head_dim, sample_token, benchmark_result)
            }
            #[cfg(feature = "cuda")]
            KvSamplerInputs::Gpu(gpu_inputs) => self.run_sweep_gpu(
                gpu_inputs,
                gpu_inputs.n_chunks(),
                n_head,
                head_dim,
                sample_token,
                benchmark_result,
            ),
        }
    }

    // ── GPU path ─────────────────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    fn run_sweep_gpu(
        &self,
        gpu_inputs: &PagedSelectionGpuInputs,
        n_batch: usize,
        n_head: usize,
        head_dim: usize,
        sample_token: usize,
        mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
    ) -> Result<(SampleQuantizationSweepResult, SampleQuantizationSweepResult)> {
        let workflow_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());

        let gpu = self.gpu.as_ref().ok_or_else(|| {
            candle::Error::Msg(
                "KvSampler::run_sweep_gpu called without initialized CUDA state".into(),
            )
        })?;

        let (k_summaries, v_summaries) = gpu
            .sample_and_select(
                gpu_inputs.per_head_table_buf(),
                gpu_inputs.head_gids_buf(),
                sample_token,
                n_batch,
                n_head,
                head_dim,
                gpu_inputs.arena_chunks(),
                benchmark_result.as_deref_mut(),
            )?
            .wait(benchmark_result.as_deref_mut())?;

        if let Some(start) = workflow_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.kv.total",
                start.elapsed(),
                1,
            );
        }

        let n_quant = self.candidates.len();
        let k_pairs: Vec<(Vec<u8>, CompressionSummary)> =
            k_summaries.into_iter().map(|s| (vec![], s)).collect();
        let v_pairs: Vec<(Vec<u8>, CompressionSummary)> =
            v_summaries.into_iter().map(|s| (vec![], s)).collect();
        Ok((
            make_sweep_result(n_batch, n_head, head_dim, n_quant, SampleSide::Key, k_pairs),
            make_sweep_result(n_batch, n_head, head_dim, n_quant, SampleSide::Value, v_pairs),
        ))
    }

    // ── CPU path ─────────────────────────────────────────────────────────────

    fn run_sweep_cpu(
        &self,
        inputs: &PagedSelectionCpuInputs<'_>,
        n_head: usize,
        head_dim: usize,
        sample_token: usize,
        mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
    ) -> Result<(SampleQuantizationSweepResult, SampleQuantizationSweepResult)> {
        let n_batch = inputs.batch_slots.len();
        let workflow_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());

        // Assemble [n_batch][n_head][head_dim][chunk_size] f32 values for K and V
        // by reading each slot's Float arena data via the backing.
        let per_chunk = n_head * head_dim * SELECT_BLOCK;
        let mut k_values = vec![0.0f32; n_batch * per_chunk];
        let mut v_values = vec![0.0f32; n_batch * per_chunk];

        let read_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
        for (b, &slot) in inputs.batch_slots.iter().enumerate() {
            let (k_chunk, v_chunk) = inputs.backing.read_f32_sampler_chunk(slot, 0)?;
            let off = b * per_chunk;
            k_values[off..off + per_chunk].copy_from_slice(&k_chunk);
            v_values[off..off + per_chunk].copy_from_slice(&v_chunk);
        }
        if let Some(start) = read_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.kv.cpu.read_arenas",
                start.elapsed(),
                1,
            );
        }

        let sample_k_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
        let k_surface = sample_error_surface_cpu(
            &k_values,
            n_batch,
            n_head,
            SELECT_BLOCK,
            head_dim,
            sample_token,
            &self.candidates,
            SampleSide::Key,
            None,
        )?;
        if let Some(start) = sample_k_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.k.cpu.sample_surface",
                start.elapsed(),
                1,
            );
        }

        let sample_v_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
        let v_surface = sample_error_surface_cpu(
            &v_values,
            n_batch,
            n_head,
            SELECT_BLOCK,
            head_dim,
            sample_token,
            &self.candidates,
            SampleSide::Value,
            None,
        )?;
        if let Some(start) = sample_v_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.v.cpu.sample_surface",
                start.elapsed(),
                1,
            );
        }

        let select_k_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
        let k_levels = batch_select_and_summarize(&k_surface, &self.k_thresholds, &self.candidates, None)?;
        if let Some(start) = select_k_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.k.cpu.select_and_summarize",
                start.elapsed(),
                1,
            );
        }

        let select_v_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
        let v_levels = batch_select_and_summarize(&v_surface, &self.v_thresholds, &self.candidates, None)?;
        if let Some(start) = select_v_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.v.cpu.select_and_summarize",
                start.elapsed(),
                1,
            );
        }

        if let Some(start) = workflow_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.kv.total",
                start.elapsed(),
                1,
            );
        }

        let n_quant = self.candidates.len();
        Ok((
            make_sweep_result(n_batch, n_head, head_dim, n_quant, SampleSide::Key, k_levels),
            make_sweep_result(n_batch, n_head, head_dim, n_quant, SampleSide::Value, v_levels),
        ))
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn make_sweep_result(
    n_batch: usize,
    n_head: usize,
    head_dim: usize,
    n_quant: usize,
    side: SampleSide,
    pairs: Vec<(Vec<u8>, CompressionSummary)>,
) -> SampleQuantizationSweepResult {
    let levels = pairs
        .into_iter()
        .map(|(winners, summary)| SampleQuantizationResult { winners, summary })
        .collect();
    SampleQuantizationSweepResult {
        surface: ErrorSurface {
            n_batch,
            n_head,
            n_dim: head_dim,
            n_quant,
            chunk_size: SELECT_BLOCK,
            side,
            data: vec![],
            q_relevance: None,
        },
        levels,
    }
}

