//! Fused QKV-projection + paged-decode-attention dispatch wrapper.
//!
//! This module exposes a Rust API that mirrors the design's full fused kernel
//! signature: it takes the per-step activation tensor + a Q4-packed W_qkv
//! weight bundle + slot machinery, and dispatches to
//! `fused_attn_v1_dispatch`.
//!
//! ## Status
//!
//! The FFI link is in place but the underlying kernel returns
//! `cudaErrorNotSupported` for all shapes until the lane mappings,
//! Q-side RoPE, full Phase 4 attention loop, and output writeback are
//! validated against a reference (see [STATUS.md](../../docs/fused_attn_v1_status.md)).
//!
//! Until the kernel is verified, `fused_qkv_attn` returns an error, and
//! callers should fall back to the standard separate-projection path
//! (`q_proj` × `k_proj` × `v_proj` followed by `paged_decode_attn`).
//!
//! ## Future model integration
//!
//! When the kernel is verified, model code (e.g. `LlamaAttention::forward`)
//! can switch to this path by:
//!   1. Concatenating `q_proj.weight`, `k_proj.weight`, `v_proj.weight`
//!      into a single `[d_model, total_output_dim]` tensor on first use.
//!   2. Quantizing that tensor to Q4_0 packed format with the layout
//!      expected by the kernel (see `build_load_queue` in `attn_fused_v1.cuh`).
//!   3. Calling `fused_qkv_attn(activations, w_qkv_q4, ...)` instead of
//!      the separate Linear ops + `paged_decode_attn`.
//!
//! That work is out of scope for the kernel scaffold; model integration is
//! tracked as Iteration 4b in the status doc.

#![cfg(feature = "cuda")]

use candle::{DType, Device, Result, Tensor};

/// Dispatch a fused QKV-projection + paged-decode-attention call.
///
/// Inputs:
/// - `activations`: `[num_active_slots, d_model]` FP16 or BF16
/// - `w_qkv_q4`: Q4_0-packed weights `[d_model, q_out + k_out + v_out]`
///   in the layout the kernel expects (K-chunk-major Q4_0 blocks).
/// - `w_qkv_scales`: optional separate scale tensor (currently embedded
///   per-block inside `w_qkv_q4` via Q4_0 format; pass `None`).
/// - `headers_ptr`: raw GPU virtual address of the SlotHeader array.
/// - `rope_cs`: RoPE cos/sin table, same as `paged_decode_attn`.
/// - `softmax_scale`: standard 1/sqrt(head_dim) factor.
/// - `sliding_window_size`: only honoured when the compiled kernel has the
///   sliding-window flag set (Mistral-style local attention).
///
/// Returns the attention output `[num_active_slots, n_q_head, head_dim]`
/// in the same dtype as `activations`.
#[allow(clippy::too_many_arguments)]
pub fn fused_qkv_attn(
    activations: &Tensor,
    w_qkv_q4: &Tensor,
    w_qkv_scales: Option<&Tensor>,
    headers_ptr: u64,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    d_model: usize,
    softmax_scale: f32,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    sliding_window_size: i32,
    use_qk_norm: bool,
    rope_partial: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_kernels::fused_attn_v1::fused_attn_v1_dispatch;

    let dev = activations.device();
    let stream = match dev {
        Device::Cuda(d) => d.cuda_stream(),
        _ => candle::bail!("fused_qkv_attn: only CUDA device supported"),
    };

    let dtype = activations.dtype();
    let q_dtype_tag: i32 = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        _ => candle::bail!("fused_qkv_attn: unsupported dtype {:?}", dtype),
    };
    let o_dtype_tag = q_dtype_tag;

    let num_active_slots = activations.dim(0)?;

    // Compute compute-cap for sm_version dispatch. We use the env-injected
    // CUDA_COMPUTE_CAP from candle-kernels build.rs as a heuristic since
    // device-side cudaGetDeviceProperties would require an extra FFI call.
    let sm_version: i32 = option_env!("CUDA_COMPUTE_CAP")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(89);

    // Acquire raw device pointers via the standard candle path.
    let dev_cu = match dev {
        Device::Cuda(d) => d,
        _ => unreachable!(),
    };

    let act_storage = activations.storage_and_layout().0;
    let act_ptr = match (&*act_storage, dtype) {
        (candle::Storage::Cuda(c), DType::F16) => {
            c.as_cuda_slice::<half::f16>()
                ?
                .device_ptr(&stream)
                .0 as *const std::ffi::c_void
        }
        (candle::Storage::Cuda(c), _) => {
            c.as_cuda_slice::<half::bf16>()
                ?
                .device_ptr(&stream)
                .0 as *const std::ffi::c_void
        }
        _ => candle::bail!("fused_qkv_attn: activations not on CUDA"),
    };

    let w_storage = w_qkv_q4.storage_and_layout().0;
    let w_ptr = match &*w_storage {
        candle::Storage::Cuda(c) => {
            c.as_cuda_slice::<u8>()?.device_ptr(&stream).0 as *const u8
        }
        _ => candle::bail!("fused_qkv_attn: w_qkv_q4 not on CUDA"),
    };

    // Optional scales pointer (currently unused — Q4_0 carries scales inline).
    let w_scales_ptr: *const std::ffi::c_void = match w_qkv_scales {
        Some(t) => {
            let s = t.storage_and_layout().0;
            match &*s {
                candle::Storage::Cuda(c) => c
                    .as_cuda_slice::<f32>()
                    ?
                    .device_ptr(&stream)
                    .0 as *const std::ffi::c_void,
                _ => candle::bail!("fused_qkv_attn: w_qkv_scales not on CUDA"),
            }
        }
        None => std::ptr::null(),
    };

    let rcs_storage = rope_cs.storage_and_layout().0;
    let rcs_ptr = match &*rcs_storage {
        candle::Storage::Cuda(c) => c
            .as_cuda_slice::<f32>()
            ?
            .device_ptr(&stream)
            .0 as *const f32,
        _ => candle::bail!("fused_qkv_attn: rope_cs not on CUDA"),
    };

    // Allocate output.
    let out_elems = num_active_slots * n_q_head * head_dim;
    let out_alloc = match dtype {
        DType::F16 => unsafe { dev_cu.alloc::<half::f16>(out_elems)? }
            .device_ptr(&stream)
            .0 as *mut std::ffi::c_void,
        DType::BF16 => unsafe { dev_cu.alloc::<half::bf16>(out_elems)? }
            .device_ptr(&stream)
            .0 as *mut std::ffi::c_void,
        _ => unreachable!(),
    };

    let raw_stream = stream.cu_stream() as *mut std::ffi::c_void;

    let rc = unsafe {
        fused_attn_v1_dispatch(
            head_dim as i32,
            n_q_head as i32,
            n_kv_head as i32,
            d_model as i32,
            rope_interleaved as i32,
            rope_partial as i32,
            use_qk_norm as i32,
            if sliding_window_size > 0 { 1 } else { 0 },
            sm_version,
            q_dtype_tag,
            o_dtype_tag,
            act_ptr,
            w_ptr,
            w_scales_ptr,
            headers_ptr as *const u8,
            out_alloc,
            num_active_slots as i32,
            softmax_scale,
            rcs_ptr,
            sliding_window_size,
            raw_stream,
        )
    };

    if rc != 0 {
        // 801 = cudaErrorNotSupported — caller should fall back to the
        // separate-projection path (q_proj × k_proj × v_proj + paged_decode).
        candle::bail!(
            "fused_qkv_attn: fused_attn_v1_dispatch returned cudaError {}",
            rc
        );
    }

    // Wrap the output allocation back into a Tensor of the right dtype/shape.
    // This is intentionally minimal — production code would need the proper
    // CudaStorage construction matching candle's internal conventions.
    candle::bail!(
        "fused_qkv_attn: kernel dispatched but Tensor wrapping not yet implemented \
         (kernel currently returns NotSupported anyway; remove this bail when \
         lane mapping + Phase 4 attention loop are validated)."
    )
}
