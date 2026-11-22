//! Runs the MoE router, the token gather and the deterministic scatter as fused
//! launches over routed activations.
//!
//! The router takes the top-k experts of each token with their softmax weights in
//! one kernel, and with `norm_topk` renormalises those weights, which cancels the
//! full softmax denominator. The gather copies each expert's tokens into one
//! stacked operand, and the q8a128 form copies the pre-quantized tiles directly.
//! The scatter combines expert outputs per token in a fixed order, one block per
//! token with no atomic adds, so the result does not depend on scheduling. It
//! defines every element of its destination, which is therefore allocated
//! uninitialised. A look-ahead vote counts which experts later layers' routers
//! would pick, from the projection that feeds them.

use super::*;

// FUSED MoE GATHER / WEIGHTED-SCATTER-ADD
// These wrap the CUDA kernels in candle-kernels so that compute.rs can call
// them from Tensor-level code without manually extracting device pointers.

/// Map Candle DType → MoeScatterDType enum value for the CUDA dispatcher.
pub(super) fn dtype_to_moe_scatter_dtype(dtype: crate::DType) -> Result<i32> {
    use candle_kernels::simple::moe_scatter::MoeScatterDType;
    match dtype {
        crate::DType::F32 => Ok(MoeScatterDType::F32 as i32),
        crate::DType::F16 => Ok(MoeScatterDType::F16 as i32),
        crate::DType::BF16 => Ok(MoeScatterDType::BF16 as i32),
        other => crate::bail!("moe_scatter: unsupported dtype {other:?}"),
    }
}

/// Fused gather: `out[i] = xs[token_ids[i]]` for 2-D tensors.
///
/// Single kernel launch replaces `Tensor::new(ids) + xs.index_select`.
/// Returns a new tensor `[total_rows, hidden_dim]`.
///
/// * `xs` — input activations `[num_tokens, hidden_dim]`
/// * `ids_dev` — pre-uploaded GPU u32 index buffer
/// * `total_rows` — number of rows to gather
/// * `device` — CUDA device
pub fn fused_moe_gather(
    xs: &crate::Tensor,
    ids_dev: &CudaSlice<u32>,
    total_rows: usize,
    device: &CudaDevice,
) -> Result<crate::Tensor> {
    use crate::cuda_backend::CudaStorageSlice;

    let (_, hidden_dim) = xs.dims2()?;

    let xs_storage = xs.storage_and_layout().0;
    let xs_cuda = match &*xs_storage {
        crate::Storage::Cuda(s) => s,
        _ => crate::bail!("fused_moe_gather: expected CUDA input"),
    };

    let dtype = xs.dtype();
    let moe_dtype = dtype_to_moe_scatter_dtype(dtype)?;

    // Allocate output
    let stream = device.cuda_stream();

    macro_rules! dispatch_gather {
        ($src_slice:expr, $elem_ty:ty) => {{
            let out_slice: CudaSlice<$elem_ty> =
                unsafe { device.alloc::<$elem_ty>(total_rows * hidden_dim)? };
            {
                let (src_ptr, _sg) = $src_slice.device_ptr(&stream);
                let (out_ptr, _og) = out_slice.device_ptr(&stream);
                let (ids_ptr, _ig) = ids_dev.device_ptr(&stream);
                unsafe {
                    candle_kernels::simple::moe_scatter::run_moe_gather(
                        moe_dtype,
                        out_ptr as *mut std::ffi::c_void,
                        src_ptr as *const std::ffi::c_void,
                        ids_ptr as *const u32,
                        total_rows,
                        hidden_dim,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            CudaStorage::wrap_cuda_slice(out_slice, device.clone())
        }};
    }

    let out_storage = match &xs_cuda.slice {
        CudaStorageSlice::BF16(s) => dispatch_gather!(s, half::bf16),
        CudaStorageSlice::F16(s) => dispatch_gather!(s, half::f16),
        CudaStorageSlice::F32(s) => dispatch_gather!(s, f32),
        _ => crate::bail!("fused_moe_gather: unsupported dtype {dtype:?}"),
    };

    let out_shape: Shape = vec![total_rows, hidden_dim].into();
    Ok(crate::tensor::from_storage(
        crate::Storage::Cuda(out_storage),
        out_shape,
        crate::op::BackpropOp::none(),
        false,
    ))
}

/// Fused MoE router: softmax + top-k select + (optional) renormalize over `logits`
/// `[num_tokens, n_experts]` in **one** kernel launch, replacing the
/// `softmax → sort(desc) → narrow(k) → renorm → flatten` op chain (≈6 launches over a tiny
/// tensor). Returns `(weights, indices)`, each `[num_tokens, k]`: f32 routing weights and u32
/// expert ids in descending-logit order. With `norm_topk` the selected top-k softmax weights are
/// renormalized (which cancels the global softmax denominator exactly, so the kernel never
/// computes the 128-wide softmax); without it they are the plain full-softmax values.
pub fn moe_route<'w>(
    logits: &LiveTensor<'w>,
    k: usize,
    norm_topk: bool,
) -> Result<(LiveTensor<'w>, LiveTensor<'w>)> {
    use crate::cuda_backend::CudaStorageSlice;

    let (num_tokens, n_experts) = logits.dims2()?;
    if k == 0 || k > 16 {
        crate::bail!("moe_route: k={k} must be in 1..=16 (kernel top-k register bound)");
    }
    if k > n_experts {
        crate::bail!("moe_route: k={k} exceeds n_experts={n_experts}");
    }
    if n_experts > 512 {
        crate::bail!(
            "moe_route: n_experts={n_experts} exceeds 512 (the widest slot \
             instantiation — 16 experts per lane)"
        );
    }
    let device = match logits.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("moe_route: expected a CUDA tensor"),
    };
    let dtype = logits.dtype();
    let moe_dtype = dtype_to_moe_scatter_dtype(dtype)?;

    let (storage, layout) = logits.storage_and_layout();
    // Rows of unit-stride columns at any pitch: a router whose projection shared its
    // launch reads its columns of the wider output row in place.
    let row_stride = layout.stride()[0];
    if layout.stride()[1] != 1 || (num_tokens > 1 && row_stride < n_experts) {
        return Err(crate::Error::RequiresContiguous { op: "moe_route" }.bt());
    }
    let row_stride = row_stride.max(n_experts);
    let o1 = layout.start_offset();
    let o2 = o1 + (num_tokens.saturating_sub(1)) * row_stride + n_experts;
    let cuda = match &*storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("moe_route: expected CUDA storage"),
    };

    let stream = device.cuda_stream();
    // The routing pair is read by the bucketize kernel inside the same FFN
    // scope as the logits it was derived from, so it belongs beside them.
    let inherit = cuda.backing;
    let (out_idx, idx_backing) =
        unsafe { alloc_inheriting::<u32>(&device, num_tokens * k, inherit)? };
    let (out_w, w_backing) = unsafe { alloc_inheriting::<f32>(&device, num_tokens * k, inherit)? };

    macro_rules! launch {
        ($s:expr) => {{
            let v = $s.slice(o1..o2);
            let (lp, _lg) = v.device_ptr(&stream);
            let (ip, _ig) = out_idx.device_ptr(&stream);
            let (wp, _wg) = out_w.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::moe_scatter::run_moe_route(
                    moe_dtype,
                    lp as *const std::ffi::c_void,
                    ip as *mut u32,
                    wp as *mut f32,
                    num_tokens as i32,
                    n_experts as i32,
                    row_stride as i32,
                    k as i32,
                    norm_topk as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }};
    }
    match &cuda.slice {
        CudaStorageSlice::F32(s) => launch!(s),
        CudaStorageSlice::F16(s) => launch!(s),
        CudaStorageSlice::BF16(s) => launch!(s),
        _ => crate::bail!("moe_route: unsupported logits dtype {dtype:?}"),
    }

    let shape: Shape = vec![num_tokens, k].into();
    let weights = crate::tensor::from_storage(
        crate::Storage::Cuda(CudaStorage::wrap_cuda_slice_backed(
            out_w,
            device.clone(),
            w_backing,
        )),
        shape.clone(),
        crate::op::BackpropOp::none(),
        false,
    );
    let indices = crate::tensor::from_storage(
        crate::Storage::Cuda(CudaStorage::wrap_cuda_slice_backed(
            out_idx,
            device.clone(),
            idx_backing,
        )),
        shape,
        crate::op::BackpropOp::none(),
        false,
    );
    Ok((weights, indices))
}

/// The most tokens one [`moe_predict_votes`] launch counts: each half of a vote
/// word is a 16-bit count.
pub const MOE_PREDICT_VOTES_MAX_TOKENS: usize = u16::MAX as usize;

/// Router look-ahead votes: `rows` is a `[num_tokens, width]` f32 projection
/// holding, from column `first_col`, `hops` later layers' router logits of
/// `n_experts` columns each. For every hop, each token names its top `n`, and
/// each expert's word is `(tokens naming it in their top k) << 16 | (tokens
/// naming it only in ranks k+1 … n)` — written to `out`, `[hops, n_experts]`
/// u32 at a device address (mapped host memory) — and its `mass`, the summed
/// softmax probability of those picks, `[hops, n_experts]` f32 at another,
/// behind a system fence. One launch, ordered on the launch stream like the
/// projection that produced `rows`.
#[allow(clippy::too_many_arguments)]
pub fn moe_predict_votes(
    rows: &LiveTensor<'_>,
    first_col: usize,
    hops: usize,
    n_experts: usize,
    k: usize,
    n: usize,
    out: u64,
    mass: u64,
) -> Result<()> {
    use crate::cuda_backend::CudaStorageSlice;

    let (num_tokens, width) = rows.dims2()?;
    if hops == 0 || num_tokens == 0 {
        return Ok(());
    }
    if n_experts > 512 || k == 0 || n < k || n > n_experts {
        crate::bail!("moe_predict_votes: n_experts={n_experts}, k={k}, n={n} out of range");
    }
    // Each half of a vote word counts tokens.
    if num_tokens > MOE_PREDICT_VOTES_MAX_TOKENS {
        crate::bail!("moe_predict_votes: {num_tokens} tokens overflow a 16-bit vote count");
    }
    if first_col + hops * n_experts > width {
        crate::bail!(
            "moe_predict_votes: {hops} hops of {n_experts} from column {first_col} exceed the \
             row's {width}"
        );
    }
    let device = match rows.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("moe_predict_votes: expected a CUDA tensor"),
    };
    let (storage, layout) = rows.storage_and_layout();
    let row_stride = layout.stride()[0];
    if layout.stride()[1] != 1 || (num_tokens > 1 && row_stride < width) {
        return Err(crate::Error::RequiresContiguous {
            op: "moe_predict_votes",
        }
        .bt());
    }
    let row_stride = row_stride.max(width);
    let o1 = layout.start_offset();
    let o2 = o1 + (num_tokens - 1) * row_stride + width;
    let crate::Storage::Cuda(cuda) = &*storage else {
        crate::bail!("moe_predict_votes: expected CUDA storage");
    };
    let CudaStorageSlice::F32(s) = &cuda.slice else {
        crate::bail!("moe_predict_votes: the projection must be f32");
    };
    let stream = device.cuda_stream();
    let v = s.slice(o1..o2);
    let (lp, _lg) = v.device_ptr(&stream);
    // SAFETY: `lp` covers every row read; `out` and `mass` are `hops ×
    // n_experts` u32 and f32 the caller owns for the launch's lifetime.
    unsafe {
        candle_kernels::simple::moe_scatter::run_moe_predict_votes(
            lp as *const f32,
            out as *mut u32,
            mass as *mut f32,
            num_tokens as i32,
            n_experts as i32,
            row_stride as i32,
            first_col as i32,
            hops as i32,
            k as i32,
            n as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

/// B3: gather pre-quantized q8a128 activations by token id into a stacked q8a128 operand the
/// experts consume directly — no gather-then-quantize. Tile by tile: output tile `(r, t)` is
/// source tile `(ids[r], t)`, quants and scale, so any `hidden` that is a multiple of 128 works,
/// including widths whose rows straddle super-blocks (2560 is 20 tiles). Mirrors
/// [`fused_moe_gather`] for the int8 path; pairs with a once-quantized FFN input.
pub fn fused_moe_gather_q8a128<'w>(
    xs_q8: &Q8a128Operand<'_>,
    ids_dev: &CudaSlice<u32>,
    total_rows: usize,
    device: &CudaDevice,
    origin: Backing,
) -> Result<Q8a128Operand<'w>> {
    let hidden = xs_q8.cols;
    if !hidden.is_multiple_of(128) {
        crate::bail!("fused_moe_gather_q8a128: hidden={hidden} must be a multiple of 128");
    }
    let tiles_per_row = hidden / 128;
    let out_bytes = q8a1024_byte_len(total_rows, hidden);
    let (out_ptr_planned, owned, out_backing) = resolve_u8_out(origin, device, out_bytes)?;
    {
        let stream = device.cuda_stream();
        let out_ptr = out_ptr_planned;
        let (ids_ptr, _ig) = ids_dev.device_ptr(&stream);
        xs_q8.with_device_ptr(device, |src_ptr| {
            unsafe {
                candle_kernels::simple::moe_scatter::run_moe_gather(
                    3, // q8a128 tile gather
                    out_ptr as *mut std::ffi::c_void,
                    src_ptr as *const std::ffi::c_void,
                    ids_ptr as *const u32,
                    total_rows,
                    tiles_per_row,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
            Ok(())
        })?;
    }
    // The gather copies blocks verbatim, header and all, so its rows store `Σx`
    // the way its source's did. Labelled with any other convention every expert
    // GEMM reading them would misread each block's sum.
    q8a128_from_out(
        owned,
        out_ptr_planned,
        out_backing,
        out_bytes,
        total_rows,
        hidden,
        xs_q8.sum_scale,
        device,
    )
}

/// Deterministic MoE scatter: sequential per-token reduce, no atomicAdd.
///
/// `down_out` is in expert-major order. `perm[i]` maps token-major index `i` to the
/// corresponding row in `down_out`, so no CPU-side reorder pass is required.
///
/// `token_starts` is a prefix-sum array of length `num_tokens + 1` on the GPU,
/// where `token_starts[t]` is the start index in the token-major arrays for token t.
/// Variable k per token is supported via these offsets.
///
/// `ys` is **DEFINED**, not accumulated into: one block per token with the
/// column loop striding the whole row, so every element is stored exactly once.
/// Allocate it uninitialised (hot-path invariant 6) — a memset would write the
/// exact bytes this kernel is about to stamp. A caller that may skip this launch
/// (nothing routed) owes its target a zero of its own.
pub fn fused_deterministic_scatter(
    ys: &LiveTensor<'_>,
    down_out: &LiveTensor<'_>,
    perm: &CudaSlice<u32>,
    weights_flat: &LiveTensor<'_>,
    reordered_weight_ids: &CudaSlice<u32>,
    token_starts: &CudaSlice<i32>,
    num_tokens: usize,
    device: &CudaDevice,
) -> Result<()> {
    use crate::cuda_backend::CudaStorageSlice;

    if num_tokens == 0 {
        return Ok(());
    }
    let (_, hidden_dim) = down_out.dims2()?;
    // The kernel is selected by where it WRITES. `down_out` is the grouped int8
    // GEMM's F32 output and is read as such — validated, not converted, so no
    // full-tensor pass stands between the GEMM and this accumulation.
    let moe_dtype = dtype_to_moe_scatter_dtype(ys.dtype())?;
    if down_out.dtype() != crate::DType::F32 {
        crate::bail!(
            "fused_deterministic_scatter: down_out must be F32 (the grouped GEMM's \
             emit type), got {:?} — convert the PRODUCER, never the tensor here",
            down_out.dtype(),
        );
    }

    if !ys.layout().is_contiguous() || ys.layout().start_offset() != 0 {
        crate::bail!("fused_deterministic_scatter: ys must be contiguous with zero offset");
    }
    if !down_out.layout().is_contiguous() || down_out.layout().start_offset() != 0 {
        crate::bail!("fused_deterministic_scatter: down_out must be contiguous with zero offset");
    }

    let ys_storage_guard = ys.storage_and_layout();
    let ys_cuda = match &*ys_storage_guard.0 {
        crate::Storage::Cuda(s) => s,
        _ => crate::bail!("fused_deterministic_scatter: expected CUDA ys"),
    };

    let src_storage = down_out.storage_and_layout().0;
    let src_cuda = match &*src_storage {
        crate::Storage::Cuda(s) => s,
        _ => crate::bail!("fused_deterministic_scatter: expected CUDA down_out"),
    };

    let wf_storage = weights_flat.storage_and_layout().0;
    let wf_cuda = match &*wf_storage {
        crate::Storage::Cuda(s) => s,
        _ => crate::bail!("fused_deterministic_scatter: expected CUDA weights_flat"),
    };
    let wf_f32 = match &wf_cuda.slice {
        CudaStorageSlice::F32(s) => s,
        _ => crate::bail!("fused_deterministic_scatter: weights_flat must be F32"),
    };

    let stream = device.cuda_stream();

    macro_rules! dispatch_var_k_scatter {
        ($ys_slice:expr, $src_slice:expr) => {{
            let (ys_ptr, _yg) = $ys_slice.device_ptr(&stream);
            let (src_ptr, _sg) = $src_slice.device_ptr(&stream);
            let (perm_ptr, _pg) = perm.device_ptr(&stream);
            let (wf_ptr, _wg) = wf_f32.device_ptr(&stream);
            let (wid_ptr, _widg) = reordered_weight_ids.device_ptr(&stream);
            let (ts_ptr, _tsg) = token_starts.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::moe_scatter::run_deterministic_scatter(
                    moe_dtype,
                    ys_ptr as *mut std::ffi::c_void,
                    src_ptr as *const std::ffi::c_void,
                    perm_ptr as *const u32,
                    wf_ptr as *const f32,
                    wid_ptr as *const u32,
                    ts_ptr as *const i32,
                    num_tokens as i32,
                    hidden_dim as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }};
    }

    let CudaStorageSlice::F32(src_s) = &src_cuda.slice else {
        crate::bail!("fused_deterministic_scatter: down_out storage is not F32");
    };
    match &ys_cuda.slice {
        CudaStorageSlice::BF16(ys_s) => dispatch_var_k_scatter!(ys_s, src_s),
        CudaStorageSlice::F16(ys_s) => dispatch_var_k_scatter!(ys_s, src_s),
        CudaStorageSlice::F32(ys_s) => dispatch_var_k_scatter!(ys_s, src_s),
        _ => crate::bail!(
            "fused_deterministic_scatter: unsupported ys dtype {:?}",
            ys.dtype()
        ),
    }

    Ok(())
}
