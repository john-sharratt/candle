//! Measures quantization error for sampled KV blocks and reduces it to the
//! per-threshold totals that the format choice reads.
//!
//! The sampling kernels score every candidate format against K and V for each
//! chunk and head, giving error surfaces, and for V the errors are weighted by
//! their query-key relevance. Winners for each threshold are selected on the
//! device and summarised there, so the host receives a small fixed-size summary
//! instead of the full winner array. The staged variants take thresholds and
//! candidates already on the device, and the asynchronous form copies on a
//! dedicated stream and returns an event the caller waits on before it reads the
//! result.

use super::*;

/// Run the new paged-batched sampled-error kernel.
///
/// Output logical layout:
/// [batch_item][head_dim][quant_index][head]
#[allow(clippy::too_many_arguments)]
pub fn sample_quant_errors_paged(
    per_head_table_dev_ptr: u64,
    head_gids_dev_ptr: u64,
    candidates: &[GgmlDType],
    sample_token: usize,
    side_is_k: bool,
    num_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
    if sample_token >= 32 {
        crate::bail!(
            "sample_quant_errors_paged: sample_token {} out of range",
            sample_token
        );
    }
    let total = num_chunks
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(candidates.len()))
        .and_then(|v| v.checked_mul(n_kv_head))
        .ok_or_else(|| crate::Error::Msg("sample_quant_errors_paged: size overflow".into()))?;
    if total == 0 {
        return Ok((dev.alloc_zeros::<f32>(0)?, dev.alloc_zeros::<f32>(0)?));
    }

    let cand_codes: Vec<i32> = candidates
        .iter()
        .map(|d| ggml_to_select_qtype(*d))
        .collect::<Result<Vec<_>>>()?;
    let cand_gpu = dev.memcpy_stod(&cand_codes)?;
    let rel_total = num_chunks
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(n_kv_head))
        .ok_or_else(|| {
            crate::Error::Msg("sample_quant_errors_paged: relevance size overflow".into())
        })?;
    let mut error_out = unsafe { dev.alloc::<f32>(total)? };
    let mut q_relevance_out = unsafe { dev.alloc::<f32>(rel_total)? };

    {
        let stream = dev.cuda_stream();
        let pht_ptr = per_head_table_dev_ptr;
        let (cand_ptr, _cand_guard) = cand_gpu.device_ptr(&stream);
        let (out_ptr, _out_guard) = error_out.device_ptr_mut(&stream);
        let (qrel_ptr, _qrel_guard) = q_relevance_out.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb("run_sample_quant_errors_paged", file!(), line!());
            run_sample_quant_errors_paged(
                pht_ptr as *const i64,
                head_gids_dev_ptr as *const i64,
                cand_ptr as *const i32,
                cand_codes.len() as i32,
                out_ptr as *mut f32,
                qrel_ptr as *mut f32,
                sample_token as i32,
                if side_is_k { 1 } else { 0 },
                num_chunks as i32,
                n_kv_head as i32,
                head_dim as i32,
                arena_chunks as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }

    Ok((error_out, q_relevance_out))
}

/// Fused KV sampled-error computation — processes K and V in a single kernel launch.
///
/// Returns `(k_errors, v_errors, q_relevance)` where:
/// - `k_errors`: `[num_chunks × head_dim × num_candidates × n_kv_head]` K quantization errors
/// - `v_errors`: same shape, V quantization errors weighted by Q·K relevance
/// - `q_relevance`: `[num_chunks × head_dim × n_kv_head]` per-block attention relevance scores
///
/// K and V must share the same candidate list.
pub fn sample_quant_errors_kv_paged(
    per_head_table_dev_ptr: u64,
    head_gids_dev_ptr: u64,
    candidates: &[GgmlDType],
    sample_token: usize,
    num_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
    if sample_token >= 32 {
        crate::bail!(
            "sample_quant_errors_kv_paged: sample_token {} out of range",
            sample_token
        );
    }
    let total = num_chunks
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(candidates.len()))
        .and_then(|v| v.checked_mul(n_kv_head))
        .ok_or_else(|| crate::Error::Msg("sample_quant_errors_kv_paged: size overflow".into()))?;
    if total == 0 {
        return Ok((dev.alloc_zeros::<f32>(0)?, dev.alloc_zeros::<f32>(0)?));
    }

    let cand_codes: Vec<i32> = candidates
        .iter()
        .map(|d| ggml_to_select_qtype(*d))
        .collect::<Result<Vec<_>>>()?;
    let cand_gpu = dev.memcpy_stod(&cand_codes)?;
    let mut k_error_out = unsafe { dev.alloc::<f32>(total)? };
    let mut v_error_out = unsafe { dev.alloc::<f32>(total)? };

    {
        let stream = dev.cuda_stream();
        let pht_ptr = per_head_table_dev_ptr;
        let (cand_ptr, _cand_guard) = cand_gpu.device_ptr(&stream);
        let (k_out_ptr, _k_out_guard) = k_error_out.device_ptr_mut(&stream);
        let (v_out_ptr, _v_out_guard) = v_error_out.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb(
                "run_sample_quant_errors_kv_paged (uniform)",
                file!(),
                line!(),
            );
            run_sample_quant_errors_kv_paged(
                pht_ptr as *const i64,
                head_gids_dev_ptr as *const i64,
                cand_ptr as *const i32,
                cand_codes.len() as i32,
                k_out_ptr as *mut f32,
                v_out_ptr as *mut f32,
                sample_token as i32,
                num_chunks as i32,
                n_kv_head as i32,
                head_dim as i32,
                arena_chunks as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }

    Ok((k_error_out, v_error_out))
}

/// Like `sample_quant_errors_kv_paged` but with candidates already staged on the device.
///
/// Use this when the candidate list is constant across many calls — upload once via
/// `dev.memcpy_stod(&cand_codes)` and pass the raw device pointer here to avoid
/// re-uploading on every invocation.
pub fn sample_quant_errors_kv_paged_staged(
    per_head_table_dev_ptr: u64,
    head_gids_dev_ptr: u64,
    candidates_dev_ptr: u64, // pre-staged &[i32] QType codes on device
    n_candidates: usize,
    sample_token: usize,
    num_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
    if sample_token >= 32 {
        crate::bail!(
            "sample_quant_errors_kv_paged_staged: sample_token {} out of range",
            sample_token
        );
    }
    let total = num_chunks
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(n_candidates))
        .and_then(|v| v.checked_mul(n_kv_head))
        .ok_or_else(|| {
            crate::Error::Msg("sample_quant_errors_kv_paged_staged: size overflow".into())
        })?;
    if total == 0 {
        return Ok((dev.alloc_zeros::<f32>(0)?, dev.alloc_zeros::<f32>(0)?));
    }

    let mut k_error_out = unsafe { dev.alloc::<f32>(total)? };
    let mut v_error_out = unsafe { dev.alloc::<f32>(total)? };

    {
        let stream = dev.cuda_stream();
        let (k_out_ptr, _k_out_guard) = k_error_out.device_ptr_mut(&stream);
        let (v_out_ptr, _v_out_guard) = v_error_out.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb(
                "run_sample_quant_errors_kv_paged (raw)",
                file!(),
                line!(),
            );
            run_sample_quant_errors_kv_paged(
                per_head_table_dev_ptr as *const i64,
                head_gids_dev_ptr as *const i64,
                candidates_dev_ptr as *const i32,
                n_candidates as i32,
                k_out_ptr as *mut f32,
                v_out_ptr as *mut f32,
                sample_token as i32,
                num_chunks as i32,
                n_kv_head as i32,
                head_dim as i32,
                arena_chunks as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }

    Ok((k_error_out, v_error_out))
}

/// Select winner candidate indices from pre-computed K/V error surfaces.
///
/// Error surfaces are expected on device (output of `sample_quant_errors_kv_paged`).
/// For each (chunk, head, dim) cell and each threshold, writes the index of the
/// first candidate whose error ≤ threshold into the output arrays.
///
/// Winner layout: `[n_thresholds × n_cells]` as `u8`, where
/// `cell = (chunk * n_kv_head + head) * head_dim + dim`.
/// This matches the layout consumed by `batch_summarize_from_winners`.
///
/// Returns `(k_winners, v_winners)` already downloaded to CPU.
pub fn select_kv_winners_paged(
    k_errors: &CudaSlice<f32>,
    v_errors: &CudaSlice<f32>,
    k_thresholds: &[f32],
    v_thresholds: &[f32],
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_quant: usize,
    dev: &CudaDevice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let n_cells = n_chunks
        .checked_mul(n_kv_head)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| crate::Error::Msg("select_kv_winners_paged: n_cells overflow".into()))?;
    if n_cells == 0 || n_quant == 0 {
        let last = n_quant.saturating_sub(1) as u8;
        return Ok((
            vec![last; k_thresholds.len() * n_cells],
            vec![last; v_thresholds.len() * n_cells],
        ));
    }
    let k_thr_gpu = dev.memcpy_stod(k_thresholds)?;
    let v_thr_gpu = dev.memcpy_stod(v_thresholds)?;
    let mut k_winners = unsafe { dev.alloc::<u8>(k_thresholds.len() * n_cells)? };
    let mut v_winners = unsafe { dev.alloc::<u8>(v_thresholds.len() * n_cells)? };

    {
        let stream = dev.cuda_stream();
        let (k_err_ptr, _ke_guard) = k_errors.device_ptr(&stream);
        let (v_err_ptr, _ve_guard) = v_errors.device_ptr(&stream);
        let (k_thr_ptr, _kt_guard) = k_thr_gpu.device_ptr(&stream);
        let (v_thr_ptr, _vt_guard) = v_thr_gpu.device_ptr(&stream);
        let (k_win_ptr, _kw_guard) = k_winners.device_ptr_mut(&stream);
        let (v_win_ptr, _vw_guard) = v_winners.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb(
                "run_select_winners_kv_paged (per-head-amax)",
                file!(),
                line!(),
            );
            run_select_winners_kv_paged(
                k_err_ptr as *const f32,
                v_err_ptr as *const f32,
                k_thr_ptr as *const f32,
                v_thr_ptr as *const f32,
                k_win_ptr as *mut u8,
                v_win_ptr as *mut u8,
                k_thresholds.len() as i32,
                v_thresholds.len() as i32,
                n_cells as i32,
                n_quant as i32,
                n_kv_head as i32,
                head_dim as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }

    Ok((dev.memcpy_dtov(&k_winners)?, dev.memcpy_dtov(&v_winners)?))
}

/// Like `select_kv_winners_paged` but with threshold arrays already staged on the device.
///
/// Use this when K and V thresholds are constant across many calls — upload once and
/// pass the raw device pointers here to avoid re-uploading on every invocation.
pub fn select_kv_winners_paged_staged(
    k_errors: &CudaSlice<f32>,
    v_errors: &CudaSlice<f32>,
    k_thresholds_dev_ptr: u64, // pre-staged &[f32] on device
    n_k_thresholds: usize,
    v_thresholds_dev_ptr: u64, // pre-staged &[f32] on device
    n_v_thresholds: usize,
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_quant: usize,
    dev: &CudaDevice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let n_cells = n_chunks
        .checked_mul(n_kv_head)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| {
            crate::Error::Msg("select_kv_winners_paged_staged: n_cells overflow".into())
        })?;
    if n_cells == 0 || n_quant == 0 {
        let last = n_quant.saturating_sub(1) as u8;
        return Ok((
            vec![last; n_k_thresholds * n_cells],
            vec![last; n_v_thresholds * n_cells],
        ));
    }
    let mut k_winners = unsafe { dev.alloc::<u8>(n_k_thresholds * n_cells)? };
    let mut v_winners = unsafe { dev.alloc::<u8>(n_v_thresholds * n_cells)? };

    {
        let stream = dev.cuda_stream();
        let (k_err_ptr, _ke_guard) = k_errors.device_ptr(&stream);
        let (v_err_ptr, _ve_guard) = v_errors.device_ptr(&stream);
        let (k_win_ptr, _kw_guard) = k_winners.device_ptr_mut(&stream);
        let (v_win_ptr, _vw_guard) = v_winners.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb(
                "run_select_winners_kv_paged (uniform-threshold)",
                file!(),
                line!(),
            );
            run_select_winners_kv_paged(
                k_err_ptr as *const f32,
                v_err_ptr as *const f32,
                k_thresholds_dev_ptr as *const f32,
                v_thresholds_dev_ptr as *const f32,
                k_win_ptr as *mut u8,
                v_win_ptr as *mut u8,
                n_k_thresholds as i32,
                n_v_thresholds as i32,
                n_cells as i32,
                n_quant as i32,
                n_kv_head as i32,
                head_dim as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }

    Ok((dev.memcpy_dtov(&k_winners)?, dev.memcpy_dtov(&v_winners)?))
}

/// Fused winner selection + GPU summarization — avoids downloading the full winner array.
///
/// Runs `select_winners_kv_paged` to produce K and V winner buffers on device, then
/// immediately runs `summarize_winners_side_paged` on each buffer, producing compact
/// `[n_thresholds × 3]` f32 partial-sum arrays (ideal_bits, head_bits, pal4_bits).
///
/// Download cost: `(n_k_thresholds + n_v_thresholds) × 3 × 4` bytes ≈ 168 bytes
/// instead of `(n_k_thresholds + n_v_thresholds) × n_cells` bytes (~3.6 MB).
///
/// # Arguments
/// * `k_errors` / `v_errors`  — K/V error surfaces from `sample_quant_errors_kv_paged_staged`
/// * `k_thresholds_dev_ptr` / `v_thresholds_dev_ptr` — pre-staged threshold arrays on device
/// * `candidates_bpe_dev_ptr` — pre-staged `[n_quant]` f32 bits-per-element on device
/// * `n_chunks`, `n_kv_head`, `head_dim`, `n_quant`, `chunk_size` — problem dimensions
/// * `pal_overhead` — palette metadata bits per head `= head_dim * 2 + 4 * 8`
///
/// # Returns
/// `(k_sums, v_sums)` where each `Vec<f32>` has length `n_thresholds * 3`:
///   `sums[t * 3 + 0]` = ideal_bits,  `[t * 3 + 1]` = head_bits,  `[t * 3 + 2]` = pal4_bits
/// Like [`select_and_summarize_kv_winners_paged_staged`] but accepts pre-allocated
/// scratch buffers to eliminate per-call device allocation overhead, and performs
/// an **asynchronous** DtoH copy on a dedicated `dtoh_stream` into a caller-supplied
/// pinned host buffer.
///
/// * `k_winners_scratch` — device buffer of at least `n_k_thresholds × n_cells` bytes.
/// * `v_winners_scratch` — device buffer of at least `n_v_thresholds × n_cells` bytes.
/// * `kv_sums` — device buffer of exactly `(n_k_thresholds + n_v_thresholds) × 3`
///   `f32` values. **Must be zeroed by the caller** (the kernel uses
///   `atomicAdd`). The K sums occupy the first `n_k_thresholds × 3`
///   elements and the V sums the remainder.
/// * `dtoh_stream` — secondary stream used exclusively for the DtoH transfer.
///   The function records an event on the compute stream, makes
///   `dtoh_stream` GPU-wait on it, then enqueues the DMA.
/// * `pinned_dst` — pre-allocated pinned host slice of length
///   `(n_k_thresholds + n_v_thresholds) × 3`.  The DMA is enqueued
///   but **not waited on** — the caller must `synchronize()` the
///   returned `CudaEvent` before reading the data.
///
/// Returns a `CudaEvent` recorded on `dtoh_stream` after the DMA.  Call
/// `event.synchronize()` to block until the data is available in `pinned_dst`.
pub fn select_and_summarize_kv_winners_paged_staged(
    k_errors: &CudaSlice<f32>,
    v_errors: &CudaSlice<f32>,
    k_thresholds_dev_ptr: u64,
    n_k_thresholds: usize,
    v_thresholds_dev_ptr: u64,
    n_v_thresholds: usize,
    candidates_bpe_dev_ptr: u64,
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_quant: usize,
    chunk_size: usize,
    pal_overhead: f32,
    k_winners_scratch: &mut CudaSlice<u8>,
    v_winners_scratch: &mut CudaSlice<u8>,
    kv_sums: &mut CudaSlice<f32>,
    dev: &CudaDevice,
    dtoh_stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    pinned_dst: &mut [f32],
) -> Result<cudarc::driver::CudaEvent> {
    // A cross-stream fence and a readback: eager, behind everything this
    // thread has recorded, so the compute-done event covers it.
    let _eager = dev.pause_capture()?;
    let n_sums = (n_k_thresholds + n_v_thresholds) * 3;
    let n_cells = n_chunks
        .checked_mul(n_kv_head)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| crate::Error::Msg("select_and_summarize: n_cells overflow".into()))?;
    let n_bh = n_chunks
        .checked_mul(n_kv_head)
        .ok_or_else(|| crate::Error::Msg("select_and_summarize: n_bh overflow".into()))?;

    if n_cells == 0 || n_quant == 0 {
        // Fill pinned destination on the CPU (no kernel launch needed) and return
        // an event that fires as soon as the GPU processes the (empty) dtoh_stream.
        let last = (n_quant.saturating_sub(1) as f32) * chunk_size as f32 * (n_cells as f32);
        pinned_dst[..n_sums].fill(last);
        return dtoh_stream.record_event(None).map_err(crate::Error::wrap);
    }

    {
        let stream = dev.cuda_stream();
        let (k_err_ptr, _ke) = k_errors.device_ptr(&stream);
        let (v_err_ptr, _ve) = v_errors.device_ptr(&stream);
        let (k_win_ptr, _kw) = k_winners_scratch.device_ptr_mut(&stream);
        let (v_win_ptr, _vw) = v_winners_scratch.device_ptr_mut(&stream);
        let (kv_sums_ptr, _ks) = kv_sums.device_ptr_mut(&stream);
        // V sums start immediately after the K sums in the combined buffer.
        let v_sums_ptr = kv_sums_ptr + (n_k_thresholds * 3 * std::mem::size_of::<f32>()) as u64;

        unsafe {
            // 1. Select winners into on-device scratch buffers (never downloaded).
            crate::set_kernel_breadcrumb(
                "run_select_winners_kv_paged (fused-sweep)",
                file!(),
                line!(),
            );
            run_select_winners_kv_paged(
                k_err_ptr as *const f32,
                v_err_ptr as *const f32,
                k_thresholds_dev_ptr as *const f32,
                v_thresholds_dev_ptr as *const f32,
                k_win_ptr as *mut u8,
                v_win_ptr as *mut u8,
                n_k_thresholds as i32,
                n_v_thresholds as i32,
                n_cells as i32,
                n_quant as i32,
                n_kv_head as i32,
                head_dim as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );

            // 2. Summarize K winners → kv_sums[0 .. n_k_thresholds*3].
            crate::set_kernel_breadcrumb("run_summarize_winners_side_paged (K)", file!(), line!());
            run_summarize_winners_side_paged(
                k_win_ptr as *const u8,
                candidates_bpe_dev_ptr as *const f32,
                kv_sums_ptr as *mut f32,
                n_k_thresholds as i32,
                n_cells as i32,
                n_bh as i32,
                head_dim as i32,
                n_quant as i32,
                chunk_size as i32,
                pal_overhead,
                stream.cu_stream() as *mut std::ffi::c_void,
            );

            // 3. Summarize V winners → kv_sums[n_k_thresholds*3 ..].
            crate::set_kernel_breadcrumb("run_summarize_winners_side_paged (V)", file!(), line!());
            run_summarize_winners_side_paged(
                v_win_ptr as *const u8,
                candidates_bpe_dev_ptr as *const f32,
                v_sums_ptr as *mut f32,
                n_v_thresholds as i32,
                n_cells as i32,
                n_bh as i32,
                head_dim as i32,
                n_quant as i32,
                chunk_size as i32,
                pal_overhead,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        // Record when all three kernels are done on the compute stream, then let
        // the D2H stream GPU-wait for that event before issuing the async DMA.
        // This serialises compute → DMA on the GPU without blocking the CPU.
        let e_compute_done = stream.record_event(None).map_err(crate::Error::wrap)?;
        dtoh_stream
            .wait(&e_compute_done)
            .map_err(crate::Error::wrap)?;
    } // All device_ptr guards released here.

    // Enqueue async DtoH into the pre-allocated pinned host slice.
    // `stream.memcpy_dtoh` does NOT synchronise the CPU — it merely enqueues the DMA.
    let kv_slice = kv_sums.slice(0..n_sums);
    dtoh_stream
        .memcpy_dtoh(&kv_slice, pinned_dst)
        .map_err(crate::Error::wrap)?;

    // Record an event AFTER the DMA so callers can `event.synchronize()` when
    // they need the data, rather than stalling immediately.
    dtoh_stream.record_event(None).map_err(crate::Error::wrap)
}
