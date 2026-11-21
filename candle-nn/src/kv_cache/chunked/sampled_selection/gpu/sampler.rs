//! Samples the quantization error for every candidate threshold on the GPU, selects
//! winners, and returns the per-threshold summary sums asynchronously.
//!
//! The candidates and thresholds are uploaded once, when the sampler is built. A
//! call returns a pending handle as soon as the kernels and the device-to-host copy
//! are queued, so the caller does not stall. Only the compact sums come back to the
//! host, never the per-cell winners, so the copy is a few words per threshold. The
//! sums are held in two pinned buffers used in turn, so one call's pending result
//! is read while the next call writes the other buffer. Each pending handle waits
//! on its own event before it reads, and the scratch lock serialises launches.

use super::*;

/// Holds pre-staged GPU buffers for KV quantization sampling.
///
/// Candidates and thresholds are constant for a given compression mode and model
/// Cached device scratch buffers for [`KvSamplerGpu::sample_and_select`].
///
/// Allocated lazily on the first call and reused across calls that share the
/// same `n_cells = n_chunks × n_kv_head × head_dim`.  Wrapped in a `Mutex`
/// for interior mutability (the public API takes `&self`).
pub(super) struct WinnersCache {
    /// `n_cells` value this cache was sized for.
    pub(super) n_cells: usize,
    /// Device scratch for K winner indices: `[n_k_thresholds × n_cells]` bytes.
    pub(super) k_winners: CudaSlice<u8>,
    /// Device scratch for V winner indices: `[n_v_thresholds × n_cells]` bytes.
    pub(super) v_winners: CudaSlice<u8>,
    /// Combined K+V accumulator: `[(n_k_thresholds + n_v_thresholds) × 3]` f32.
    /// Zeroed via a tiny async H2D copy before each kernel launch.
    pub(super) kv_sums: CudaSlice<f32>,
    /// Pre-allocated host zeros for resetting `kv_sums` — avoids per-call heap allocation.
    pub(super) zeros: Vec<f32>,
    /// Ping/pong selector: `false` → use `host_kv_sums_a`, `true` → use `host_kv_sums_b`.
    /// Toggled on every call so that a `PendingKvSummaries` from call N reads from the
    /// buffer that call N+1 is *not* writing into, enabling overlap without data races.
    pub(super) use_b: bool,
}

/// Uploaded-once constants for the KV sampler — reused on
/// on every call to [`KvSamplerGpu::sample_and_select`].  This eliminates three
/// small H→D copies (`candidates`, `k_thresholds`, `v_thresholds`) that would
/// otherwise occur on every token.
pub struct KvSamplerGpu {
    pub(super) candidates_buf: GpuBuf, // i32 QType codes, one per candidate
    pub(super) candidates_bpe_buf: GpuBuf, // f32 bits-per-element, one per candidate
    pub(super) k_thresholds_buf: GpuBuf, // f32 K error thresholds
    pub(super) v_thresholds_buf: GpuBuf, // f32 V error thresholds
    pub(super) candidates: Vec<SampleFormat>,
    pub(super) n_k_thresholds: usize,
    pub(super) n_v_thresholds: usize,
    pub(super) dev: candle::CudaDevice,
    /// Dedicated secondary stream for async DtoH transfers.
    /// Kept separate from the compute stream so enqueuing DMA doesn't stall
    /// kernel launches on the next call.
    pub(super) dtoh_stream: Arc<CudaStream>,
    /// Ping/pong pinned host buffers for the combined K+V sums download.
    /// One is in-flight (DMA queued) while the other is free for the next call.
    /// Size = `(n_k_thresholds + n_v_thresholds) × 3 × 4` bytes each.
    pub(super) host_kv_sums_a: Arc<PinnedBuf>,
    pub(super) host_kv_sums_b: Arc<PinnedBuf>,
    /// Lazily-allocated per-call scratch buffers.  Held in a `Mutex` so
    /// `sample_and_select` can mutate them without requiring `&mut self`.
    pub(super) scratch: std::sync::Mutex<Option<WinnersCache>>,
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
        // No generation is open here — these are uploaded once when the
        // selector is built — so they own their bytes rather than bumping.
        let candidates_buf = stage_i32_slice(&cand_codes, None, None, dev)?;
        let bpe_vals: Vec<f32> = candidates.iter().map(|f| f.bits_per_elem()).collect();
        let candidates_bpe_buf = stage_f32_slice(&bpe_vals, None, None, dev)?;
        let k_thresholds_buf = stage_f32_slice(k_thresholds, None, None, dev)?;
        let v_thresholds_buf = stage_f32_slice(v_thresholds, None, None, dev)?;

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
            let needs_realloc = scratch_guard.as_ref().is_none_or(|c| c.n_cells != n_cells);
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
    pub(super) event: CudaEvent,
    /// Pinned host buffer that the DMA is writing into (one of the two ping/pong bufs).
    pub(super) host_kv_sums: Arc<PinnedBuf>,
    pub(super) n_k_thresholds: usize,
    pub(super) n_v_thresholds: usize,
    pub(super) total_elems: f64,
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
