use super::params::SELECT_BLOCK;
use super::profile::sampled_profile_record_duration;
use super::CompressionSummary;
use super::{ErrorSurface, SampleFormat, SampleSide, SampledSelectionBenchmarkResult};
use crate::kv_cache::chunked::backing::BackingInner;
use crate::kv_cache::QuantFormat;
use crate::kv_cache::{arena_gid_stride, ChunkedKvBacking, HeadGids, KvFormat, N_PALETTE};
use candle::cuda_backend::cudarc::driver::{CudaEvent, CudaSlice, CudaStream, DevicePtr};
use candle::quantized::{
    cuda::{
        reduce_head_format_stats, sample_quant_errors_kv_paged_staged, sample_quant_errors_paged,
        select_and_summarize_kv_winners_paged_staged,
        select_kv_format_palette4_paged_batched_raw_from_device_ptrs,
    },
    pinned_staging::{Generation, GpuBuf, PinnedBuf},
    GgmlDType,
};
use candle::Result;
use std::sync::Arc;

fn stage_bytes_as_gpu_buf(
    bytes: &[u8],
    generation: Option<&Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    if let Some(generation) = generation {
        let mut pinned = generation.alloc(bytes.len())?;
        pinned.as_mut_slice().copy_from_slice(bytes);
        generation.submit(pinned)
    } else {
        let gpu_u8 = dev.memcpy_stod(bytes)?;
        Ok(GpuBuf::from_raw_owned(gpu_u8, dev))
    }
}

fn stage_i64_slice(
    values: &[i64],
    generation: Option<&Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    stage_bytes_as_gpu_buf(bytes, generation, dev)
}

fn stage_i32_slice(
    values: &[i32],
    generation: Option<&Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    stage_bytes_as_gpu_buf(bytes, generation, dev)
}

fn stage_f32_slice(
    values: &[f32],
    generation: Option<&Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    stage_bytes_as_gpu_buf(bytes, generation, dev)
}

pub enum SelectionBackend<'a> {
    Cpu,
    Gpu(&'a PagedSelectionGpuInputs),
}

#[derive(Debug, Clone)]
pub struct GpuQualityAggregation {
    pub k_head_formats: Vec<SampleFormat>,
    pub v_head_formats: Vec<SampleFormat>,
    pub k_worst_case_formats: Vec<SampleFormat>,
    pub v_worst_case_formats: Vec<SampleFormat>,
    pub k_palette4_formats: Vec<SampleFormat>,
    pub v_palette4_formats: Vec<SampleFormat>,
}

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

/// Holds pre-staged GPU buffers for KV quantization sampling.
///
/// Candidates and thresholds are constant for a given compression mode and model
/// Cached device scratch buffers for [`KvSamplerGpu::sample_and_select`].
///
/// Allocated lazily on the first call and reused across calls that share the
/// same `n_cells = n_chunks × n_kv_head × head_dim`.  Wrapped in a `Mutex`
/// for interior mutability (the public API takes `&self`).
struct WinnersCache {
    /// `n_cells` value this cache was sized for.
    n_cells: usize,
    /// Device scratch for K winner indices: `[n_k_thresholds × n_cells]` bytes.
    k_winners: CudaSlice<u8>,
    /// Device scratch for V winner indices: `[n_v_thresholds × n_cells]` bytes.
    v_winners: CudaSlice<u8>,
    /// Combined K+V accumulator: `[(n_k_thresholds + n_v_thresholds) × 3]` f32.
    /// Zeroed via a tiny async H2D copy before each kernel launch.
    kv_sums: CudaSlice<f32>,
    /// Pre-allocated host zeros for resetting `kv_sums` — avoids per-call heap allocation.
    zeros: Vec<f32>,
    /// Ping/pong selector: `false` → use `host_kv_sums_a`, `true` → use `host_kv_sums_b`.
    /// Toggled on every call so that a `PendingKvSummaries` from call N reads from the
    /// buffer that call N+1 is *not* writing into, enabling overlap without data races.
    use_b: bool,
}

/// Uploaded-once constants for the KV sampler — reused on
/// on every call to [`KvSamplerGpu::sample_and_select`].  This eliminates three
/// small H→D copies (`candidates`, `k_thresholds`, `v_thresholds`) that would
/// otherwise occur on every token.
pub struct KvSamplerGpu {
    candidates_buf: GpuBuf,     // i32 QType codes, one per candidate
    candidates_bpe_buf: GpuBuf, // f32 bits-per-element, one per candidate
    k_thresholds_buf: GpuBuf,   // f32 K error thresholds
    v_thresholds_buf: GpuBuf,   // f32 V error thresholds
    candidates: Vec<SampleFormat>,
    n_k_thresholds: usize,
    n_v_thresholds: usize,
    dev: candle::CudaDevice,
    /// Dedicated secondary stream for async DtoH transfers.
    /// Kept separate from the compute stream so enqueuing DMA doesn't stall
    /// kernel launches on the next call.
    dtoh_stream: Arc<CudaStream>,
    /// Ping/pong pinned host buffers for the combined K+V sums download.
    /// One is in-flight (DMA queued) while the other is free for the next call.
    /// Size = `(n_k_thresholds + n_v_thresholds) × 3 × 4` bytes each.
    host_kv_sums_a: Arc<PinnedBuf>,
    host_kv_sums_b: Arc<PinnedBuf>,
    /// Lazily-allocated per-call scratch buffers.  Held in a `Mutex` so
    /// `sample_and_select` can mutate them without requiring `&mut self`.
    scratch: std::sync::Mutex<Option<WinnersCache>>,
}

impl KvSamplerGpu {
    /// Upload candidates and thresholds to the device once.
    pub fn new(
        candidates: &[SampleFormat],
        k_thresholds: &[f32],
        v_thresholds: &[f32],
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        use candle::quantized::cuda::ggml_to_select_qtype;
        let cand_codes: Vec<i32> = candidates
            .iter()
            .map(|f| ggml_to_select_qtype(f.to_ggml_dtype()))
            .collect::<candle::Result<Vec<_>>>()?;
        let candidates_buf = stage_i32_slice(&cand_codes, None, dev)?;
        let bpe_vals: Vec<f32> = candidates.iter().map(|f| f.bits_per_elem()).collect();
        let candidates_bpe_buf = stage_f32_slice(&bpe_vals, None, dev)?;
        let k_thresholds_buf = stage_f32_slice(k_thresholds, None, dev)?;
        let v_thresholds_buf = stage_f32_slice(v_thresholds, None, dev)?;

        // Dedicated stream for DtoH transfers — separate from the compute stream.
        let dtoh_stream = dev
            .cuda_context()
            .new_stream()
            .map_err(candle::Error::wrap)?;

        // Pre-allocate pinned host buffers (one ping, one pong).
        // Size is fixed: (n_k + n_v) × 3 f32s, known at construction time.
        let n_sums = (k_thresholds.len() + v_thresholds.len()) * 3;
        let n_bytes = n_sums * std::mem::size_of::<f32>();
        let host_kv_sums_a = Arc::new(PinnedBuf::alloc_owned(n_bytes)?);
        let host_kv_sums_b = Arc::new(PinnedBuf::alloc_owned(n_bytes)?);

        Ok(Self {
            candidates_buf,
            candidates_bpe_buf,
            k_thresholds_buf,
            v_thresholds_buf,
            candidates: candidates.to_vec(),
            n_k_thresholds: k_thresholds.len(),
            n_v_thresholds: v_thresholds.len(),
            dev: dev.clone(),
            dtoh_stream,
            host_kv_sums_a,
            host_kv_sums_b,
            scratch: std::sync::Mutex::new(None),
        })
    }

    pub fn candidates(&self) -> &[SampleFormat] {
        &self.candidates
    }

    pub fn n_k_thresholds(&self) -> usize {
        self.n_k_thresholds
    }

    pub fn n_v_thresholds(&self) -> usize {
        self.n_v_thresholds
    }

    /// Sample KV quantization errors on the GPU, select winners on the GPU, summarize
    /// on the GPU, and kick off an **asynchronous** DtoH copy of the compact
    /// `(n_k_thresholds + n_v_thresholds) × 3` sums into pre-allocated pinned memory.
    ///
    /// Returns a [`PendingKvSummaries`] handle immediately — no CPU stall occurs.
    /// Call [`PendingKvSummaries::wait`] to synchronize the DtoH and decode the sums
    /// into `CompressionSummary` values.
    ///
    /// No winner data is ever downloaded — bandwidth cost is
    /// `(n_k_thresholds + n_v_thresholds) × 3 × 4` bytes instead of
    /// `(n_k_thresholds + n_v_thresholds) × n_cells` bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_and_select(
        &self,
        per_head_table_buf: &GpuBuf,
        head_gids_buf: &GpuBuf,
        sample_token: usize,
        num_chunks: usize,
        n_kv_head: usize,
        head_dim: usize,
        arena_chunks: usize,
        mut benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
    ) -> Result<PendingKvSummaries> {
        let kernel_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
        let (k_gpu, v_gpu) = sample_quant_errors_kv_paged_staged(
            per_head_table_buf.dev_ptr(),
            head_gids_buf.dev_ptr(),
            self.candidates_buf.dev_ptr(),
            self.candidates.len(),
            sample_token,
            num_chunks,
            n_kv_head,
            head_dim,
            arena_chunks,
            &self.dev,
        )?;
        if let Some(start) = kernel_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.kv.surface.gpu.kernel",
                start.elapsed(),
                1,
            );
        }

        let select_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());
        let pal_overhead = (head_dim * 2 + 4 * 8) as f32;

        // ── Pre-allocated scratch: lazily initialised, reused across calls. ──
        let n_cells = num_chunks * n_kv_head * head_dim;
        let n_sums = (self.n_k_thresholds + self.n_v_thresholds) * 3;

        // Determine which pinned host buffer to write into this call (ping/pong).
        // The use_b flag lives inside the scratch mutex so it's updated atomically
        // with the kernel launch — no separate lock needed.
        let event = {
            let mut scratch_guard = self
                .scratch
                .lock()
                .map_err(|_| candle::Error::Msg("KvSamplerGpu: scratch mutex poisoned".into()))?;

            // Allocate (or reallocate if n_cells changed) without dropping old buffers
            // until after the new ones are ready.
            let needs_realloc = scratch_guard
                .as_ref()
                .map_or(true, |c| c.n_cells != n_cells);
            if needs_realloc {
                let k_winners = unsafe { self.dev.alloc::<u8>(self.n_k_thresholds * n_cells)? };
                let v_winners = unsafe { self.dev.alloc::<u8>(self.n_v_thresholds * n_cells)? };
                let kv_sums = unsafe { self.dev.alloc::<f32>(n_sums)? };
                let zeros = vec![0.0f32; n_sums];
                *scratch_guard = Some(WinnersCache {
                    n_cells,
                    k_winners,
                    v_winners,
                    kv_sums,
                    zeros,
                    use_b: false,
                });
            }

            let cache = scratch_guard.as_mut().unwrap();

            // Toggle the ping/pong selector BEFORE selecting the buffer.
            cache.use_b = !cache.use_b;
            let current_buf: &Arc<PinnedBuf> = if cache.use_b {
                &self.host_kv_sums_b
            } else {
                &self.host_kv_sums_a
            };

            // Zero the combined accumulator via a tiny async H2D copy that lands
            // on the stream before the summarize kernels.  `cache.zeros` is
            // pre-allocated so there is no per-call heap allocation here.
            self.dev.memcpy_htod(&cache.zeros[..], &mut cache.kv_sums)?;

            // Get a &mut [f32] into the pinned host buffer.
            //
            // Safety: we hold the scratch mutex which serialises all calls to
            // `sample_and_select`.  The DMA is enqueued on `dtoh_stream` and writes
            // to this buffer BEFORE event.synchronize() returns in the caller.
            // Meanwhile, the *other* buffer (A or B, whichever is not selected) is
            // the one a concurrent `PendingKvSummaries::wait` might be reading from —
            // that is a different allocation, so there is no aliasing.
            let pinned_f32: &mut [f32] = unsafe {
                let buf_ptr = Arc::as_ptr(current_buf) as *mut PinnedBuf;
                let byte_slice = (*buf_ptr).as_mut_slice();
                std::slice::from_raw_parts_mut(byte_slice.as_mut_ptr() as *mut f32, n_sums)
            };

            // Run select + summarize kernels, enqueue async DtoH, return CudaEvent.
            select_and_summarize_kv_winners_paged_staged(
                &k_gpu,
                &v_gpu,
                self.k_thresholds_buf.dev_ptr(),
                self.n_k_thresholds,
                self.v_thresholds_buf.dev_ptr(),
                self.n_v_thresholds,
                self.candidates_bpe_buf.dev_ptr(),
                num_chunks,
                n_kv_head,
                head_dim,
                self.candidates.len(),
                32, // chunk_size (SELECT_BLOCK)
                pal_overhead,
                &mut cache.k_winners,
                &mut cache.v_winners,
                &mut cache.kv_sums,
                &self.dev,
                &self.dtoh_stream,
                pinned_f32,
            )?
            // scratch_guard drops here; the DMA is in-flight on dtoh_stream but the
            // mutex release is safe because the caller's event.synchronize() ensures
            // completion before the pinned data is read.
        };

        // Capture the Arc so PendingKvSummaries can read from the right buffer
        // after event.synchronize(), independent of the scratch mutex.
        let host_buf = {
            let sg = self
                .scratch
                .lock()
                .map_err(|_| candle::Error::Msg("KvSamplerGpu: scratch mutex poisoned".into()))?;
            let use_b = sg.as_ref().map(|c| c.use_b).unwrap_or(false);
            if use_b {
                Arc::clone(&self.host_kv_sums_b)
            } else {
                Arc::clone(&self.host_kv_sums_a)
            }
        };

        if let Some(start) = select_start {
            sampled_profile_record_duration(
                benchmark_result.as_deref_mut(),
                "quantization.kv.surface.gpu.select_summarize",
                start.elapsed(),
                1,
            );
        }

        let total_elems = (num_chunks * n_kv_head * head_dim * 32) as f64;
        Ok(PendingKvSummaries {
            event,
            host_kv_sums: host_buf,
            n_k_thresholds: self.n_k_thresholds,
            n_v_thresholds: self.n_v_thresholds,
            total_elems,
        })
    }
}

/// Handle returned by [`KvSamplerGpu::sample_and_select`].
///
/// The GPU kernels and the DtoH DMA are already in-flight when this is
/// constructed.  Call [`wait`](Self::wait) to synchronise the DMA and decode
/// the sums into [`CompressionSummary`] values.
///
/// This type is `Send` so it can be handed off to another thread if needed.
pub struct PendingKvSummaries {
    /// Event recorded on the DtoH stream after the async DMA.
    event: CudaEvent,
    /// Pinned host buffer that the DMA is writing into (one of the two ping/pong bufs).
    host_kv_sums: Arc<PinnedBuf>,
    n_k_thresholds: usize,
    n_v_thresholds: usize,
    total_elems: f64,
}

impl PendingKvSummaries {
    /// Block until the DtoH DMA is complete, then decode and return the summaries.
    pub fn wait(
        self,
        benchmark_result: Option<&mut SampledSelectionBenchmarkResult>,
    ) -> Result<(Vec<CompressionSummary>, Vec<CompressionSummary>)> {
        let wait_start = benchmark_result.as_ref().map(|_| std::time::Instant::now());

        // Block CPU until the async DtoH DMA completes.
        self.event.synchronize().map_err(candle::Error::wrap)?;

        if let Some(start) = wait_start {
            sampled_profile_record_duration(
                benchmark_result,
                "quantization.kv.surface.gpu.dtoh_wait",
                start.elapsed(),
                1,
            );
        }

        let n_k = self.n_k_thresholds;
        let n_v = self.n_v_thresholds;
        let n = (n_k + n_v) * 3;
        let total_elems = self.total_elems;

        // Reinterpret the pinned u8 bytes as f32 (native endian; same pointer, no copy).
        //
        // Safety: the DMA has completed (event.synchronize() above), `host_kv_sums`
        // holds the only live Arc reference to this buffer at this point (the WinnersCache's
        // use_b already toggled to the other buffer for the next call), and f32 has no
        // validity requirements beyond alignment (which cuMemHostAlloc guarantees ≥ 128 B).
        let sums: &[f32] = unsafe {
            std::slice::from_raw_parts(self.host_kv_sums.as_slice().as_ptr() as *const f32, n)
        };

        let to_summaries = |slice: &[f32], n_t: usize| -> Vec<CompressionSummary> {
            (0..n_t)
                .map(|t| {
                    let ideal_bits = slice[t * 3] as f64;
                    let head_bits = slice[t * 3 + 1] as f64;
                    let pal4_bits = slice[t * 3 + 2] as f64;
                    let ideal_bpe = if total_elems > 0.0 {
                        ideal_bits / total_elems
                    } else {
                        16.0
                    };
                    let head_bpe = if total_elems > 0.0 {
                        head_bits / total_elems
                    } else {
                        16.0
                    };
                    let palette4_bpe = if total_elems > 0.0 {
                        pal4_bits / total_elems
                    } else {
                        16.0
                    };
                    CompressionSummary {
                        ideal_bpe,
                        head_bpe,
                        palette4_bpe,
                        ideal_cr: 16.0 / ideal_bpe.max(1e-9),
                        head_cr: 16.0 / head_bpe.max(1e-9),
                        palette4_cr: 16.0 / palette4_bpe.max(1e-9),
                    }
                })
                .collect()
        };

        Ok((
            to_summaries(&sums[..n_k * 3], n_k),
            to_summaries(&sums[n_k * 3..], n_v),
        ))
    }
}

pub struct PagedSelectionGpuInputs {
    // Live per-block GIDs keep the arena chunk allocations resident.
    chunk_gids_keepalive: Vec<HeadGids>,
    // Keeps the per-head table tensor (and thus its device memory) alive.
    _per_head_table_tensor: Option<candle::Tensor>,
    per_head_table_buf: GpuBuf,
    head_gids: Vec<i64>,
    head_gids_buf: GpuBuf,
    blocks_per_chunk: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    dev: candle::CudaDevice,
}

impl PagedSelectionGpuInputs {
    pub fn from_f32_chunks(
        k_chunks: &[&[f32]],
        v_chunks: &[&[f32]],
        blocks_per_chunk: usize,
        n_kv_head: usize,
        _arena_chunks: usize,
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<(ChunkedKvBacking, Self)> {
        if k_chunks.len() != v_chunks.len() {
            candle::bail!("paged selection input length mismatch");
        }
        if blocks_per_chunk == 0 {
            candle::bail!("blocks_per_chunk must be > 0");
        }
        if n_kv_head == 0 || blocks_per_chunk % n_kv_head != 0 {
            candle::bail!(
                "blocks_per_chunk ({blocks_per_chunk}) must be divisible by n_kv_head ({n_kv_head})"
            );
        }
        let head_dim = blocks_per_chunk / n_kv_head;
        if head_dim % N_PALETTE != 0 {
            candle::bail!("head_dim ({head_dim}) must be divisible by N_PALETTE ({N_PALETTE})");
        }
        let chunk_size = SELECT_BLOCK;
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_band = sub_head_dim * chunk_size;

        let device = candle::Device::Cuda(dev.clone());
        let n_chunks = k_chunks.len();
        let backing = ChunkedKvBacking::new_with_format(
            n_chunks,
            n_kv_head,
            head_dim,
            KvFormat::Float(candle::DType::F32),
            KvFormat::Float(candle::DType::F32),
            &device,
            chunk_size,
        )?;

        for (slot, (k_chunk, v_chunk)) in k_chunks.iter().zip(v_chunks.iter()).enumerate() {
            // Rearrange from head-major layout ([H][D][T] per chunk) to palette-major
            // ([h0p0, h0p1, h0p2, h0p3, h1p0, ...]) as expected by write_raw_sealed_chunk.
            let mut k_bytes = Vec::with_capacity(n_kv_head * N_PALETTE * elems_per_band * 4);
            let mut v_bytes = Vec::with_capacity(n_kv_head * N_PALETTE * elems_per_band * 4);
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    let start = (h * head_dim + p * sub_head_dim) * chunk_size;
                    k_bytes.extend(
                        k_chunk[start..start + elems_per_band]
                            .iter()
                            .flat_map(|&f| f.to_le_bytes()),
                    );
                    v_bytes.extend(
                        v_chunk[start..start + elems_per_band]
                            .iter()
                            .flat_map(|&f| f.to_le_bytes()),
                    );
                }
            }
            backing.write_raw_sealed_chunk(
                slot,
                0,
                &k_bytes,
                &v_bytes,
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            )?;
        }

        let all_slots: Vec<usize> = (0..n_chunks).collect();
        let inputs = Self::from_backing_owned(&backing, &all_slots, generation, dev)?;
        Ok((backing, inputs))
    }

    /// Build a paged-selection input set with K stored as R16 (raw F16 + captured
    /// Q values), matching what production stores during prefill/decode.
    ///
    /// Use this instead of [`from_f32_chunks`] when the caller has Q-projection
    /// values per chunk (e.g., loaded from a v4 KV dump): the GPU selection
    /// kernel reads `block_r16->q[]` to compute per-block Q-relevance, drives
    /// the IQR threshold scaling, and selects formats per-(head, palette).
    /// Without R16+Q the kernel sees Q=0 everywhere, falls back to the
    /// geometric-mean threshold, and the per-block format distribution does
    /// not match what production produces.
    ///
    /// `k_chunks`, `v_chunks`, and `q_chunks` are head-major `[H][D][T]` flat
    /// f32 arrays of length `n_kv_head * head_dim * chunk_size`, the same
    /// layout used by `dump_sequence_*_chunks` and the v3/v4 dump readers.
    /// V is kept as F32 (matches production where V is always float during
    /// the selection phase).
    pub fn from_f32_chunks_with_q(
        k_chunks: &[&[f32]],
        v_chunks: &[&[f32]],
        q_chunks: &[&[f32]],
        blocks_per_chunk: usize,
        n_kv_head: usize,
        _arena_chunks: usize,
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<(ChunkedKvBacking, Self)> {
        if k_chunks.len() != v_chunks.len() || k_chunks.len() != q_chunks.len() {
            candle::bail!("paged selection input length mismatch (k/v/q must match)");
        }
        if blocks_per_chunk == 0 {
            candle::bail!("blocks_per_chunk must be > 0");
        }
        if n_kv_head == 0 || blocks_per_chunk % n_kv_head != 0 {
            candle::bail!(
                "blocks_per_chunk ({blocks_per_chunk}) must be divisible by n_kv_head ({n_kv_head})"
            );
        }
        let head_dim = blocks_per_chunk / n_kv_head;
        if head_dim % N_PALETTE != 0 {
            candle::bail!("head_dim ({head_dim}) must be divisible by N_PALETTE ({N_PALETTE})");
        }
        let chunk_size = SELECT_BLOCK;
        let sub_head_dim = head_dim / N_PALETTE;
        let elems_per_band = sub_head_dim * chunk_size;

        let device = candle::Device::Cuda(dev.clone());
        let n_chunks = k_chunks.len();
        // K arena = R16 (raw F16 + Q-capture). V arena stays F32 (matches the
        // production prefill/decode layout where V is always float during
        // selection).
        let backing = ChunkedKvBacking::new_with_format(
            n_chunks,
            n_kv_head,
            head_dim,
            KvFormat::Quantized(QuantFormat::R16),
            KvFormat::Float(candle::DType::F32),
            &device,
            chunk_size,
        )?;

        // R16 byte layout per (head, palette) sub-band:
        //   sub_head_dim blocks × 128 bytes/block
        //   block[d] = { F16 d[CHUNK_SIZE]   // K values for tokens 0..32 at dim d
        //              , F16-as-u16 q[CHUNK_SIZE] }  // Q values for tokens 0..32 at dim d
        //
        // Source layout: `dump_sequence_*_chunks` and the v3/v4 dump readers
        // produce `[H][P][T][D']` flat f32 — palette-segmented, then
        // **token-major** within each (h, p) sub-band (matches the F32 arena
        // shape `(1, CHUNK_SIZE, sub_head_dim)`).  R16's dim-major block
        // layout requires transposing each sub-band on the way in.
        let r16_bytes_per_band = sub_head_dim * 128;
        for (slot, ((k_chunk, v_chunk), q_chunk)) in k_chunks
            .iter()
            .zip(v_chunks.iter())
            .zip(q_chunks.iter())
            .enumerate()
        {
            let mut k_bytes = Vec::with_capacity(n_kv_head * N_PALETTE * r16_bytes_per_band);
            let mut v_bytes = Vec::with_capacity(n_kv_head * N_PALETTE * elems_per_band * 4);
            for h in 0..n_kv_head {
                for p in 0..N_PALETTE {
                    let start = (h * head_dim + p * sub_head_dim) * chunk_size;
                    // Pack 32 R16 blocks (one per dim) for this (h, p) sub-band.
                    // Source is [T][D'] token-major; iterate d outer to gather all
                    // 32 tokens for that dim into block[d].d[] / block[d].q[].
                    for d in 0..sub_head_dim {
                        for t in 0..chunk_size {
                            let src = start + t * sub_head_dim + d;
                            let k_h = half::f16::from_f32(k_chunk[src]);
                            k_bytes.extend_from_slice(&k_h.to_le_bytes());
                        }
                        for t in 0..chunk_size {
                            let src = start + t * sub_head_dim + d;
                            let q_h = half::f16::from_f32(q_chunk[src]);
                            k_bytes.extend_from_slice(&q_h.to_le_bytes());
                        }
                    }
                    // V is F32 — same token-major layout as `from_f32_chunks`.
                    v_bytes.extend(
                        v_chunk[start..start + elems_per_band]
                            .iter()
                            .flat_map(|&f| f.to_le_bytes()),
                    );
                }
            }
            backing.write_raw_sealed_chunk(
                slot,
                0,
                &k_bytes,
                &v_bytes,
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
                std::sync::Arc::new(Vec::new()),
            )?;
        }

        let all_slots: Vec<usize> = (0..n_chunks).collect();
        let inputs = Self::from_backing_owned(&backing, &all_slots, generation, dev)?;
        Ok((backing, inputs))
    }

    fn from_backing_owned(
        backing: &ChunkedKvBacking,
        batch_indices: &[usize],
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        let mut chunk_gids_keepalive = Vec::with_capacity(batch_indices.len());
        for &batch_idx in batch_indices {
            let sealed = backing.live_chunks_as_sealed(batch_idx, &[]).ok_or_else(|| {
                candle::Error::Msg(format!("missing live chunks for batch slot {batch_idx}"))
            })?;
            let chunk = sealed.first().ok_or_else(|| {
                candle::Error::Msg(format!(
                    "no paged chunk recorded for batch slot {batch_idx}"
                ))
            })?;
            chunk_gids_keepalive.push(chunk.gids.clone());
        }

        Self::from_head_gids(std::sync::Arc::clone(&backing.inner), &chunk_gids_keepalive, generation, dev)
    }

    pub fn from_backing(
        backing: &ChunkedKvBacking,
        batch_indices: &[usize],
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        Self::from_backing_owned(backing, batch_indices, generation, dev)
    }

    pub fn from_head_gids(
        backing: Arc<BackingInner>,
        chunk_gids_keepalive: &[HeadGids],
        generation: Option<&Generation>,
        dev: &candle::CudaDevice,
    ) -> Result<Self> {
        let per_head_table = backing.per_head_table_sync()?;
        let per_head_table_buf = {
            let (storage, layout) = per_head_table.storage_and_layout();
            match &*storage {
                candle::Storage::Cuda(cuda) => {
                    let slice = cuda.as_cuda_slice::<i64>()?;
                    let stream = dev.cuda_stream();
                    let (ptr, _guard) = slice.device_ptr(&stream);
                    let len = layout.shape().elem_count() * std::mem::size_of::<i64>();
                    GpuBuf::from_borrowed(ptr, len)
                }
                _ => candle::bail!("paged selection backing must be on CUDA"),
            }
        };

        let n_kv_head = backing.n_kv_head;
        let blocks_per_chunk = n_kv_head * backing.head_dim;
        let chunk_gids_keepalive = chunk_gids_keepalive.to_vec();
        let head_gids: Vec<i64> = chunk_gids_keepalive
            .iter()
            .flat_map(|gids| {
                (0..n_kv_head).flat_map(move |head_idx| {
                    [gids.k_gid(head_idx).raw(), gids.v_gid(head_idx).raw()]
                })
            })
            .collect();
        let head_gids_buf = stage_i64_slice(&head_gids, generation, dev)?;

        Ok(Self {
            chunk_gids_keepalive,
            _per_head_table_tensor: Some(per_head_table),
            per_head_table_buf,
            head_gids,
            head_gids_buf,
            blocks_per_chunk,
            n_kv_head,
            arena_chunks: arena_gid_stride(),
            dev: dev.clone(),
        })
    }

    pub fn select_chunks(
        &self,
        start_chunk: usize,
        chunk_count: usize,
        generation: Option<&Generation>,
    ) -> Result<Self> {
        let end_chunk = start_chunk + chunk_count;
        let gids_per_chunk = self.n_kv_head * 2;
        if end_chunk > self.chunk_gids_keepalive.len()
            || end_chunk * gids_per_chunk > self.head_gids.len()
        {
            candle::bail!("paged input chunk range out of bounds");
        }

        let head_gids =
            self.head_gids[start_chunk * gids_per_chunk..end_chunk * gids_per_chunk].to_vec();
        let chunk_gids_keepalive = self.chunk_gids_keepalive[start_chunk..end_chunk].to_vec();
        let head_gids_buf = stage_i64_slice(&head_gids, generation, &self.dev)?;

        Ok(Self {
            chunk_gids_keepalive,
            _per_head_table_tensor: self._per_head_table_tensor.clone(),
            per_head_table_buf: self.per_head_table_buf.clone(),
            head_gids,
            head_gids_buf,
            blocks_per_chunk: self.blocks_per_chunk,
            n_kv_head: self.n_kv_head,
            arena_chunks: self.arena_chunks,
            dev: self.dev.clone(),
        })
    }

    pub fn per_head_table_buf(&self) -> &GpuBuf {
        &self.per_head_table_buf
    }

    pub fn head_gids_buf(&self) -> &GpuBuf {
        &self.head_gids_buf
    }

    pub fn head_gids(&self) -> &[i64] {
        &self.head_gids
    }

    pub fn arena_chunks(&self) -> usize {
        self.arena_chunks
    }

    pub fn dev(&self) -> &candle::CudaDevice {
        &self.dev
    }

    #[inline]
    fn blocks_per_head(&self) -> usize {
        // 32-element blocks per head per chunk: one block per head-dimension
        // (32 tokens × 1 dim). Equal to head_dim (= blocks_per_chunk / n_kv_head).
        self.blocks_per_chunk / self.n_kv_head.max(1)
    }

    pub fn n_chunks(&self) -> usize {
        self.chunk_gids_keepalive.len()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn select_block_formats(
        &self,
        k_candidates: &[SampleFormat],
        v_candidates: &[SampleFormat],
        k_threshold_hi: f32,
        k_threshold_lo: f32,
        v_threshold_hi: f32,
        v_threshold_lo: f32,
    ) -> Result<(
        Vec<SampleFormat>,
        Vec<SampleFormat>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    )> {
        let blocks_per_head = self.blocks_per_head();
        let k_ggml: Vec<GgmlDType> = k_candidates
            .iter()
            .copied()
            .map(|f| f.to_ggml_dtype())
            .collect();
        let v_ggml: Vec<GgmlDType> = v_candidates
            .iter()
            .copied()
            .map(|f| f.to_ggml_dtype())
            .collect();

        let pht_ptr = self.per_head_table_buf.dev_ptr();
        let gids_ptr = self.head_gids_buf.dev_ptr();
        let n_chunks = self.chunk_gids_keepalive.len();
        let total_heads = n_chunks * self.n_kv_head;

        let (
            k_pal_tags_gpu,
            v_pal_tags_gpu,
            k_pal_scale_gpu,
            v_pal_scale_gpu,
            k_pal_map_gpu,
            v_pal_map_gpu,
            k_head_amax_gpu,
            v_head_amax_gpu,
            _k_eff_gpu,
            _v_eff_gpu,
            _k_htags_gpu,
            _v_htags_gpu,
            _q_rel_gpu,
        ) = unsafe {
            select_kv_format_palette4_paged_batched_raw_from_device_ptrs(
                pht_ptr,
                gids_ptr,
                n_chunks,
                &k_ggml,
                &v_ggml,
                k_threshold_hi,
                k_threshold_lo,
                v_threshold_hi,
                v_threshold_lo,
                blocks_per_head,
                self.n_kv_head,
                self.arena_chunks,
                &self.dev,
                &self.dev.cuda_bg_stream(),
            )?
        };

        let k_pal_tags_cpu: Vec<i32> = self.dev.memcpy_dtov(&k_pal_tags_gpu)?;
        let v_pal_tags_cpu: Vec<i32> = self.dev.memcpy_dtov(&v_pal_tags_gpu)?;
        let k_pal_scale_cpu: Vec<f32> = self.dev.memcpy_dtov(&k_pal_scale_gpu)?;
        let v_pal_scale_cpu: Vec<f32> = self.dev.memcpy_dtov(&v_pal_scale_gpu)?;
        let k_map_cpu: Vec<i32> = self.dev.memcpy_dtov(&k_pal_map_gpu)?;
        let v_map_cpu: Vec<i32> = self.dev.memcpy_dtov(&v_pal_map_gpu)?;
        let k_head_amax: Vec<f32> = self.dev.memcpy_dtov(&k_head_amax_gpu)?;
        let v_head_amax: Vec<f32> = self.dev.memcpy_dtov(&v_head_amax_gpu)?;

        let total_blocks = total_heads * blocks_per_head;

        let mut k_blk_tags = Vec::with_capacity(total_blocks);
        let mut v_blk_tags = Vec::with_capacity(total_blocks);
        let mut k_blk_scale = Vec::with_capacity(total_blocks);
        let mut v_blk_scale = Vec::with_capacity(total_blocks);

        for bi in 0..total_blocks {
            let head = bi / blocks_per_head;
            let k_slot = (k_map_cpu[bi].clamp(0, 3)) as usize;
            let v_slot = (v_map_cpu[bi].clamp(0, 3)) as usize;
            k_blk_tags.push(SampleFormat::try_from_cuda_tag(
                k_pal_tags_cpu[head * 4 + k_slot],
            )?);
            v_blk_tags.push(SampleFormat::try_from_cuda_tag(
                v_pal_tags_cpu[head * 4 + v_slot],
            )?);
            k_blk_scale.push(k_pal_scale_cpu[head * 4 + k_slot]);
            v_blk_scale.push(v_pal_scale_cpu[head * 4 + v_slot]);
        }

        Ok((
            k_blk_tags,
            v_blk_tags,
            k_blk_scale,
            v_blk_scale,
            k_head_amax,
            v_head_amax,
        ))
    }

    pub fn aggregate_quality_metric_formats_gpu(
        &self,
        k_formats: &[SampleFormat],
        v_formats: &[SampleFormat],
        generation: Option<&Generation>,
    ) -> Result<GpuQualityAggregation> {
        let (k_head_formats, v_head_formats, k_worst_case_formats, v_worst_case_formats) =
            self.reduce_per_head_formats_gpu(k_formats, v_formats, generation)?;
        // `k_formats` / `v_formats` are already palette-expanded per-block tags
        // produced by the fused GPU kernel (`select_kv_format_palette4_paged`)
        // via `select_block_formats`.  Re-running `cpu_palette4_reduce` on top
        // would shuffle blocks across palette slots using a coarser CPU greedy
        // that doesn't see block data — so we just pass the kernel's tags
        // through unchanged.
        let k_palette4_formats = k_formats.to_vec();
        let v_palette4_formats = v_formats.to_vec();
        Ok(GpuQualityAggregation {
            k_head_formats,
            v_head_formats,
            k_worst_case_formats,
            v_worst_case_formats,
            k_palette4_formats,
            v_palette4_formats,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn reduce_per_head_formats_gpu(
        &self,
        k_formats: &[SampleFormat],
        v_formats: &[SampleFormat],
        generation: Option<&Generation>,
    ) -> Result<(
        Vec<SampleFormat>,
        Vec<SampleFormat>,
        Vec<SampleFormat>,
        Vec<SampleFormat>,
    )> {
        let blocks_per_head = self.blocks_per_head();
        let total_blocks = self.chunk_gids_keepalive.len() * self.n_kv_head * blocks_per_head;
        if k_formats.len() != total_blocks || v_formats.len() != total_blocks {
            candle::bail!("reduce_per_head_formats_gpu length mismatch");
        }
        let k_codes: Vec<i32> = k_formats
            .iter()
            .copied()
            .map(SampleFormat::to_cuda_tag)
            .collect();
        let v_codes: Vec<i32> = v_formats
            .iter()
            .copied()
            .map(SampleFormat::to_cuda_tag)
            .collect();
        let k_buf = stage_i32_slice(&k_codes, generation, &self.dev)?;
        let v_buf = stage_i32_slice(&v_codes, generation, &self.dev)?;
        let (k_head_gpu, v_head_gpu, k_eff_gpu, v_eff_gpu) = reduce_head_format_stats(
            k_buf.dev_ptr(),
            v_buf.dev_ptr(),
            blocks_per_head,
            self.n_kv_head,
            self.chunk_gids_keepalive.len(),
            &self.dev,
        )?;
        Ok((
            self.dev
                .memcpy_dtov(&k_head_gpu)?
                .into_iter()
                .map(SampleFormat::try_from_cuda_tag)
                .collect::<Result<Vec<_>>>()?,
            self.dev
                .memcpy_dtov(&v_head_gpu)?
                .into_iter()
                .map(SampleFormat::try_from_cuda_tag)
                .collect::<Result<Vec<_>>>()?,
            self.dev
                .memcpy_dtov(&k_eff_gpu)?
                .into_iter()
                .map(SampleFormat::try_from_cuda_tag)
                .collect::<Result<Vec<_>>>()?,
            self.dev
                .memcpy_dtov(&v_eff_gpu)?
                .into_iter()
                .map(SampleFormat::try_from_cuda_tag)
                .collect::<Result<Vec<_>>>()?,
        ))
    }

    /// Fused per-head palette-4 selection: single kernel pass replacing `select_block_formats` + palette reduction.
    ///
    /// Returns `(k_palette4_rows, v_palette4_rows, k_pal_scale_rows, v_pal_scale_rows,
    ///           k_assignments, v_assignments, k_head_amax, v_head_amax)`.
    #[allow(clippy::type_complexity)]
    pub fn select_palette4_formats_fused(
        &self,
        k_candidates: &[SampleFormat],
        v_candidates: &[SampleFormat],
        k_threshold_hi: f32,
        k_threshold_lo: f32,
        v_threshold_hi: f32,
        v_threshold_lo: f32,
        _generation: Option<&Generation>,
    ) -> Result<(
        Vec<[SampleFormat; 4]>,
        Vec<[SampleFormat; 4]>,
        Vec<[f32; 4]>,
        Vec<[f32; 4]>,
        Vec<u8>,
        Vec<u8>,
        Vec<f32>,
        Vec<f32>,
    )> {
        let blocks_per_head = self.blocks_per_head();
        let n_chunks = self.chunk_gids_keepalive.len();
        let k_ggml: Vec<candle::quantized::GgmlDType> = k_candidates
            .iter()
            .copied()
            .map(|f| f.to_ggml_dtype())
            .collect();
        let v_ggml: Vec<candle::quantized::GgmlDType> = v_candidates
            .iter()
            .copied()
            .map(|f| f.to_ggml_dtype())
            .collect();

        let pht_ptr = self.per_head_table_buf.dev_ptr();
        let gids_ptr = self.head_gids_buf.dev_ptr();

        let (
            k_pal_tags_gpu,
            v_pal_tags_gpu,
            k_pal_scale_gpu,
            v_pal_scale_gpu,
            k_pal_map_gpu,
            v_pal_map_gpu,
            k_head_amax_gpu,
            v_head_amax_gpu,
            _k_eff_gpu,
            _v_eff_gpu,
            _k_htags_gpu,
            _v_htags_gpu,
            _q_rel_gpu,
        ) = unsafe {
            select_kv_format_palette4_paged_batched_raw_from_device_ptrs(
                pht_ptr,
                gids_ptr,
                n_chunks,
                &k_ggml,
                &v_ggml,
                k_threshold_hi,
                k_threshold_lo,
                v_threshold_hi,
                v_threshold_lo,
                blocks_per_head,
                self.n_kv_head,
                self.arena_chunks,
                &self.dev,
                &self.dev.cuda_bg_stream(),
            )?
        };

        let k_pal_tags_cpu: Vec<i32> = self.dev.memcpy_dtov(&k_pal_tags_gpu)?;
        let v_pal_tags_cpu: Vec<i32> = self.dev.memcpy_dtov(&v_pal_tags_gpu)?;
        let k_pal_scale_cpu: Vec<f32> = self.dev.memcpy_dtov(&k_pal_scale_gpu)?;
        let v_pal_scale_cpu: Vec<f32> = self.dev.memcpy_dtov(&v_pal_scale_gpu)?;
        let k_map_cpu: Vec<i32> = self.dev.memcpy_dtov(&k_pal_map_gpu)?;
        let v_map_cpu: Vec<i32> = self.dev.memcpy_dtov(&v_pal_map_gpu)?;
        let k_head_amax: Vec<f32> = self.dev.memcpy_dtov(&k_head_amax_gpu)?;
        let v_head_amax: Vec<f32> = self.dev.memcpy_dtov(&v_head_amax_gpu)?;

        let to_palette = |flat: &[i32]| -> Result<Vec<[SampleFormat; 4]>> {
            flat.chunks_exact(4)
                .map(|c| {
                    Ok([
                        SampleFormat::try_from_cuda_tag(c[0])?,
                        SampleFormat::try_from_cuda_tag(c[1])?,
                        SampleFormat::try_from_cuda_tag(c[2])?,
                        SampleFormat::try_from_cuda_tag(c[3])?,
                    ])
                })
                .collect()
        };
        let to_scale_palette = |flat: &[f32]| -> Vec<[f32; 4]> {
            flat.chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect()
        };

        let k_palette4_rows = to_palette(&k_pal_tags_cpu)?;
        let v_palette4_rows = to_palette(&v_pal_tags_cpu)?;
        let k_pal_scale_rows = to_scale_palette(&k_pal_scale_cpu);
        let v_pal_scale_rows = to_scale_palette(&v_pal_scale_cpu);
        let k_assignments: Vec<u8> = k_map_cpu.iter().map(|&v| v.clamp(0, 3) as u8).collect();
        let v_assignments: Vec<u8> = v_map_cpu.iter().map(|&v| v.clamp(0, 3) as u8).collect();

        Ok((
            k_palette4_rows,
            v_palette4_rows,
            k_pal_scale_rows,
            v_pal_scale_rows,
            k_assignments,
            v_assignments,
            k_head_amax,
            v_head_amax,
        ))
    }
}
