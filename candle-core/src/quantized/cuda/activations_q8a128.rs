//! Quantizes activations into the q8a128 operand the int8 matmul reads, and the
//! fused producers that write that operand directly.
//!
//! The inference's numeric mode is decided here: any int8 mode turns an
//! activation into q8a128 blocks, while `Int8Mode::Off` keeps the float tensor
//! for the dequantized-weight path. A producer that already holds the activation,
//! a fused RMSNorm or SwiGLU, emits the same blocks in one kernel, so a
//! projection pays no separate quantize launch or float round-trip. Each operand
//! is carved from the arena its activation lives in, and its per-128 sum
//! convention travels with the blocks so the matmul reads them as they were
//! written.

use super::*;

/// Convert activations `[M × K]` (or `[B × M × K]`) to the matmul operand selected by `mode`,
/// the single knob that drives the whole inference's numeric mode: any int8 mode → quantize to
/// q8a128 (the int8 tensor-core path, paired with KO weights), [`Int8Mode::Off`] → keep the float
/// tensor (the dequant-weight float path). The q8a128 activation is identical for
/// [`Int8Mode::Performance`] and [`Int8Mode::Precision`] — only the weight twin differs — so this
/// branches solely on [`Int8Mode::is_int8`]. Paired with `QMatMul::repack_for_optimization` on the
/// weight side; the matmul's KO⇔int8 guard keeps the two consistent.
///
/// `sum_scale` says how the per-128 `Σx` is stored in the blocks this produces,
/// and rides on the operand so the matmul reads them the way they were written.
/// [`SumScale::Raw`] is what every language model here passes and is
/// bit-identical to the unparameterised path; a model whose activations can
/// overflow f16 on a block sum passes [`SumScale::ByAmax`] — see [`SumScale`].
pub fn to_dynamic<'w>(
    xs: &crate::LiveTensor<'w>,
    mode: Int8Mode,
    device: &CudaDevice,
    sum_scale: SumScale,
) -> Result<DynamicActs<'w>> {
    use crate::cuda_backend::CudaStorageSlice;
    if !mode.is_int8() {
        return Ok(DynamicActs::Float(xs.clone()));
    }
    let (rows, cols) = match xs.dims() {
        &[m, k] => (m, k),
        &[b, m, k] => (b * m, k),
        s => crate::bail!("to_dynamic(int8): expected 2D/3D activations, got {s:?}"),
    };
    let dtype_code: i32 = match xs.dtype() {
        crate::DType::F16 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F32 => 2,
        d => crate::bail!("to_dynamic(int8): unsupported activation dtype {d:?}"),
    };
    let (storage, layout) = xs.storage_and_layout();
    let (o1, o2) = layout
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::RequiresContiguous { op: "to_dynamic" }.bt())?;
    let cuda = match &*storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("to_dynamic(int8): activations must be a CUDA tensor"),
    };
    let stream = device.cuda_stream();
    let op = match &cuda.slice {
        CudaStorageSlice::F16(s) => {
            let v = s.slice(o1..o2);
            let (ptr, _g) = v.device_ptr(&stream);
            quantize_acts_q8a128(ptr, dtype_code, rows, cols, device, cuda.backing, sum_scale)?
        }
        CudaStorageSlice::BF16(s) => {
            let v = s.slice(o1..o2);
            let (ptr, _g) = v.device_ptr(&stream);
            quantize_acts_q8a128(ptr, dtype_code, rows, cols, device, cuda.backing, sum_scale)?
        }
        CudaStorageSlice::F32(s) => {
            let v = s.slice(o1..o2);
            let (ptr, _g) = v.device_ptr(&stream);
            quantize_acts_q8a128(ptr, dtype_code, rows, cols, device, cuda.backing, sum_scale)?
        }
        _ => crate::bail!("to_dynamic(int8): activation slice dtype must be F16/BF16/F32"),
    };
    // Preserve the activation's leading dims (everything but K) so the int8 matmul rebuilds
    // the output rank ([B,M,K]→[B,M,N]) exactly like the float path.
    let lead: Vec<usize> = xs.dims()[..xs.rank() - 1].to_vec();
    Ok(DynamicActs::Int8(op.with_lead(lead)))
}

/// Bytes a `[rows, cols]` activation occupies in the q8a1024 flat-grouped layout:
/// 8 × 128-element tiles per 1152-byte super-block (see blocks.cuh).
///
/// The single definition of this size. `cols` is only required to be a multiple of
/// 128, so the tile count is rounded up to whole super-blocks across the whole
/// buffer — dividing `cols` by 1024 instead truncates for every K that is not a
/// multiple of 1024, which includes the MoE intermediate widths (Qwen3-30B-A3B's
/// is 768, where that form yields zero).
pub(crate) fn q8a1024_byte_len(rows: usize, cols: usize) -> usize {
    (rows * (cols / 128)).div_ceil(8) * 1152
}

/// Quantize activations `[rows, cols]` (dtype 0=F16,1=BF16,2=F32 at `act_ptr`) →
/// q8a1024 flat-grouped blocks (8 × 128-tiles per 1152-byte super-block; qs
/// de-interleaved from the per-32 ds — see blocks.cuh). The matmul mode is not chosen here; it is
/// derived later by the grid-fill formula `q8a128_dense_tile` at dispatch.
pub fn quantize_acts_q8a128<'w>(
    act_ptr: u64,
    dtype: i32,
    rows: usize,
    cols: usize,
    device: &CudaDevice,
    // The activation's own backing, so the operand is carved where the
    // activation lives. Unplumbed, this allocated from the pool unconditionally
    // — and it is on the int8 matmul's hot path, so *every* projection in a
    // quantized model did. The forbidden-allocation detector named it directly:
    // `quantize_acts_q8a128 <- to_dynamic <- forward_via_int8`, at the top of
    // the report by both call count and bytes.
    origin: Backing,
    // How the per-128 `Σx` is stored. One runtime argument the kernel selects a
    // coefficient from — not a template parameter, so this adds no instantiation
    // and the produced bytes are the only thing that differs.
    sum_scale: SumScale,
) -> Result<Q8a128Operand<'w>> {
    let bytes = q8a1024_byte_len(rows, cols);
    // The same three-value resolve `rms_norm_q8a128` below uses, which is the
    // fused form of this and was plumbed while this was not.
    let (out_ptr_planned, owned, out_backing) = resolve_u8_out(origin, device, bytes)?;
    // SAFETY: `out_ptr_planned` names `bytes` of either the carved arena range or
    // the owned fallback, and the kernel writes exactly that; `act_ptr` is the
    // caller's activation, read as `dtype` for `rows * cols`.
    unsafe {
        run_quantize_q8a128(
            act_ptr as *const std::ffi::c_void,
            out_ptr_planned as *mut std::ffi::c_void,
            rows as i32,
            cols as i32,
            dtype,
            sum_scale.as_code(),
            device.cuda_stream().cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(q8a128_from_out(
        owned,
        out_ptr_planned,
        out_backing,
        bytes,
        rows,
        cols,
        device,
    )?
    .with_sum_scale(sum_scale))
}

/// A q8a128 operand for `[rows, cols]` written by a caller's own fused producer.
///
/// The buffer is carved where `beside` lives — the open wave's span when it is
/// wave-scoped, the pool otherwise — exactly as [`quantize_acts_q8a128`] carves
/// its own, so a producer outside this crate (a model's fused norm or activation
/// kernel) emits the int8 operand with no standalone quantize launch and no
/// allocation the arena did not plan. `fill` receives the device address and
/// must launch the kernel that writes all `q8a1024_byte_len(rows, cols)` bytes
/// in the q8a1024 layout, with the Σx convention `sum_scale` names.
pub fn produce_q8a128<'w>(
    beside: &LiveTensor<'w>,
    rows: usize,
    cols: usize,
    sum_scale: SumScale,
    fill: impl FnOnce(u64) -> Result<()>,
) -> Result<Q8a128Operand<'w>> {
    if !cols.is_multiple_of(128) {
        crate::bail!("produce_q8a128: cols={cols} must be a multiple of 128");
    }
    let (storage, _) = beside.storage_and_layout();
    let Storage::Cuda(cuda) = &*storage else {
        crate::bail!("produce_q8a128: the producer's tensor must be on CUDA");
    };
    let device = cuda.device().clone();
    let origin = cuda.backing;
    drop(storage);
    let bytes = q8a1024_byte_len(rows, cols);
    let (ptr, owned, backing) = resolve_u8_out(origin, &device, bytes)?;
    fill(ptr)?;
    Ok(q8a128_from_out(owned, ptr, backing, bytes, rows, cols, &device)?.with_sum_scale(sum_scale))
}

/// Fused RMSNorm → q8a128: normalize each row of `xs` `[.. × K]` by `alpha` `[K]` and emit the
/// q8a128 activation operand directly, in ONE kernel — the producer epilogue for B1/B3/B5. This
/// replaces the unfused `rms_norm` (FP store) + [`quantize_acts_q8a128`] (re-read) pair, removing
/// the standalone quantize launch + FP round-trip that dominates M=1 decode. The result tracks
/// that two-call path within float margin (the quant grid is per-128; the normalized values are
/// rounded through the input dtype before quantization, mirroring the FP store). `K` must be a
/// multiple of 128 and ≤ 8192 (the row is cached in shared memory). Leading dims are preserved on
/// the operand so the int8 matmul rebuilds the output rank like the float path.
pub fn rms_norm_q8a128<'w>(
    xs: &crate::Tensor,
    alpha: &crate::Tensor,
    eps: f32,
    device: &CudaDevice,
    origin: Backing,
    // The Σx convention this fused producer writes, carried onto the operand so
    // the matmul reads the header the way it was written — see [`SumScale`].
    sum_scale: SumScale,
) -> Result<Q8a128Operand<'w>> {
    use crate::cuda_backend::CudaStorageSlice;
    let (rows, cols) = match xs.dims() {
        &[m, k] => (m, k),
        &[b, m, k] => (b * m, k),
        s => crate::bail!("rms_norm_q8a128: expected 2D/3D activations, got {s:?}"),
    };
    if cols % 128 != 0 {
        crate::bail!("rms_norm_q8a128: K={cols} must be a multiple of 128");
    }
    if cols > 8192 {
        crate::bail!("rms_norm_q8a128: K={cols} exceeds the cached-row limit (8192)");
    }
    if alpha.dims() != [cols] {
        crate::bail!(
            "rms_norm_q8a128: alpha must be [{cols}], got {:?}",
            alpha.dims()
        );
    }
    if xs.dtype() != alpha.dtype() {
        crate::bail!(
            "rms_norm_q8a128: xs/alpha dtype mismatch {:?} vs {:?}",
            xs.dtype(),
            alpha.dtype()
        );
    }
    // FloatDType code for the reduce dispatcher: 0=f32, 2=f16, 3=bf16.
    let dtype_code: i32 = match xs.dtype() {
        crate::DType::F32 => 0,
        crate::DType::F16 => 2,
        crate::DType::BF16 => 3,
        d => crate::bail!("rms_norm_q8a128: unsupported dtype {d:?}"),
    };

    let bytes = q8a1024_byte_len(rows, cols);
    let out_bytes = bytes;
    let (out_ptr_planned, owned, out_backing) = resolve_u8_out(origin, device, out_bytes)?;

    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let (xo1, xo2) = xs_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "rms_norm_q8a128(xs)",
        }
        .bt()
    })?;
    let xs_cuda = match &*xs_storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("rms_norm_q8a128: xs must be a CUDA tensor"),
    };
    let (a_storage, a_layout) = alpha.storage_and_layout();
    let (ao1, ao2) = a_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "rms_norm_q8a128(alpha)",
        }
        .bt()
    })?;
    let a_cuda = match &*a_storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("rms_norm_q8a128: alpha must be a CUDA tensor"),
    };

    let stream = device.cuda_stream();
    {
        let out_ptr = out_ptr_planned;
        // The src/alpha slices and their device-ptr guards must outlive the launch, so the kernel
        // call happens inside the match arm where all are still alive.
        macro_rules! launch {
            ($xsl:expr, $asl:expr) => {{
                let xv = $xsl.slice(xo1..xo2);
                let av = $asl.slice(ao1..ao2);
                let (sp, _gx) = xv.device_ptr(&stream);
                let (ap, _ga) = av.device_ptr(&stream);
                unsafe {
                    run_rmsnorm_q8a128_op(
                        dtype_code,
                        sp as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        ap as *const std::ffi::c_void,
                        rows as i32,
                        cols as i32,
                        eps,
                        sum_scale.as_code(),
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }};
        }
        match (&xs_cuda.slice, &a_cuda.slice) {
            (CudaStorageSlice::F16(x), CudaStorageSlice::F16(a)) => launch!(x, a),
            (CudaStorageSlice::BF16(x), CudaStorageSlice::BF16(a)) => launch!(x, a),
            (CudaStorageSlice::F32(x), CudaStorageSlice::F32(a)) => launch!(x, a),
            _ => crate::bail!("rms_norm_q8a128: xs/alpha must be F16/BF16/F32 and match"),
        }
    }

    let lead: Vec<usize> = xs.dims()[..xs.rank() - 1].to_vec();
    q8a128_from_out(
        owned,
        out_ptr_planned,
        out_backing,
        bytes,
        rows,
        cols,
        device,
    )
    .map(|o| o.with_lead(lead).with_sum_scale(sum_scale))
}

/// Fused SwiGLU → q8a128: compute `silu(gate) · up` element-wise and emit the q8a128 activation
/// operand directly, in ONE kernel — the producer epilogue for B4 (feeds the down projection).
/// Replaces the unfused `silu_mul` (FP store) + [`quantize_acts_q8a128`] (re-read). `gate`/`up`
/// must share shape `[.. × K]` and dtype; `K` (and the total element count) must be a multiple of
/// 128. Bit-identical to the two-call path: silu uses the same fast-exp path, the product is
/// rounded through the input dtype as the eager store rounds it, and the tile goes through the one
/// q8a128 emitter (`quantized_mlp`'s test asserts it at F32 and BF16, under both `Σx`
/// conventions). Leading dims are preserved.
pub fn silu_mul_q8a128<'w>(
    gate: &LiveTensor<'w>,
    up: &LiveTensor<'w>,
    device: &CudaDevice,
    origin: Backing,
    // The Σx convention this fused producer writes — see [`SumScale`]. **This is
    // the site the normalised form exists for**: the SwiGLU intermediate is
    // where the per-128 sums run away from f16's range.
    sum_scale: SumScale,
) -> Result<Q8a128Operand<'w>> {
    use crate::cuda_backend::CudaStorageSlice;
    if gate.dims() != up.dims() {
        crate::bail!(
            "silu_mul_q8a128: gate/up shape mismatch {:?} vs {:?}",
            gate.dims(),
            up.dims()
        );
    }
    if gate.dtype() != up.dtype() {
        crate::bail!(
            "silu_mul_q8a128: gate/up dtype mismatch {:?} vs {:?}",
            gate.dtype(),
            up.dtype()
        );
    }
    let (rows, cols) = match gate.dims() {
        &[m, k] => (m, k),
        &[b, m, k] => (b * m, k),
        s => crate::bail!("silu_mul_q8a128: expected 2D/3D activations, got {s:?}"),
    };
    if (rows * cols) % 128 != 0 {
        crate::bail!(
            "silu_mul_q8a128: rows*cols ({}) must be a multiple of 128",
            rows * cols
        );
    }
    // FusedSiluMul dtype code: 0=f32, 1=f16, 2=bf16.
    let dtype_code: i32 = match gate.dtype() {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        d => crate::bail!("silu_mul_q8a128: unsupported dtype {d:?}"),
    };

    let bytes = q8a1024_byte_len(rows, cols);
    let out_bytes = bytes;
    let (out_ptr_planned, owned, out_backing) = resolve_u8_out(origin, device, out_bytes)?;

    // Rows read in place: a dense operand, or a row-strided view such as one half
    // of a fused gate|up projection. Both operands must step rows identically.
    let (g_storage, g_layout) = gate.storage_and_layout();
    let (go1, g_stride) = silu_mul_rows(g_layout, "gate")?;
    let (u_storage, u_layout) = up.storage_and_layout();
    let (uo1, u_stride) = silu_mul_rows(u_layout, "up")?;
    // A single row has no second row to step to, whatever stride its view records.
    let (g_stride, u_stride) = if rows == 1 {
        (cols, cols)
    } else {
        (g_stride, u_stride)
    };
    if g_stride != u_stride {
        crate::bail!("silu_mul_q8a128: gate rows are {g_stride} apart but up rows {u_stride}");
    }
    if g_stride != cols && !cols.is_multiple_of(128) {
        crate::bail!(
            "silu_mul_q8a128: strided rows need cols ({cols}) to be a multiple of 128, so no \
             tile straddles two rows"
        );
    }
    // Each operand's extent, from its first element to the last one read.
    let span = (rows - 1) * g_stride + cols;
    let (go2, uo2) = (go1 + span, uo1 + span);
    let g_cuda = match &*g_storage {
        Storage::Cuda(c) => c,
        _ => crate::bail!("silu_mul_q8a128: gate must be a CUDA tensor"),
    };
    let u_cuda = match &*u_storage {
        Storage::Cuda(c) => c,
        _ => crate::bail!("silu_mul_q8a128: up must be a CUDA tensor"),
    };

    let stream = device.cuda_stream();
    {
        let out_ptr = out_ptr_planned;
        macro_rules! launch {
            ($gsl:expr, $usl:expr) => {{
                let gv = $gsl.slice(go1..go2);
                let uv = $usl.slice(uo1..uo2);
                let (gp, _gg) = gv.device_ptr(&stream);
                let (upp, _gu) = uv.device_ptr(&stream);
                unsafe {
                    run_silu_mul_q8a128_op(
                        dtype_code,
                        gp as *const std::ffi::c_void,
                        upp as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        rows as i32,
                        cols as i32,
                        sum_scale.as_code(),
                        g_stride as i32,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }};
        }
        match (&g_cuda.slice, &u_cuda.slice) {
            (CudaStorageSlice::F16(g), CudaStorageSlice::F16(u)) => launch!(g, u),
            (CudaStorageSlice::BF16(g), CudaStorageSlice::BF16(u)) => launch!(g, u),
            (CudaStorageSlice::F32(g), CudaStorageSlice::F32(u)) => launch!(g, u),
            _ => crate::bail!("silu_mul_q8a128: gate/up must be F16/BF16/F32 and match"),
        }
    }

    let lead: Vec<usize> = gate.dims()[..gate.rank() - 1].to_vec();
    q8a128_from_out(
        owned,
        out_ptr_planned,
        out_backing,
        bytes,
        rows,
        cols,
        device,
    )
    .map(|o| o.with_lead(lead).with_sum_scale(sum_scale))
}

/// `(start offset, row stride)` of a [`silu_mul_q8a128`] operand: unit column
/// stride, and every row one stride after the last — a dense tensor, or a row
/// range of a wider one, at rank 2 or 3 (a rank-3 batch must continue the same
/// row grid, so its stride is `rows × row stride`).
pub(super) fn silu_mul_rows(layout: &Layout, what: &str) -> Result<(usize, usize)> {
    let dims = layout.dims();
    let stride = layout.stride();
    let rank = dims.len();
    if !(rank == 2 || rank == 3) || stride[rank - 1] != 1 {
        crate::bail!(
            "silu_mul_q8a128: {what} is {dims:?} stride {stride:?}; rows must be unit-stride at \
             rank 2 or 3"
        );
    }
    let row_stride = stride[rank - 2];
    if rank == 3 && dims[0] > 1 && stride[0] != dims[1] * row_stride {
        crate::bail!(
            "silu_mul_q8a128: {what} is {dims:?} stride {stride:?}; its batches do not continue \
             one row grid"
        );
    }
    Ok((layout.start_offset(), row_stride))
}
