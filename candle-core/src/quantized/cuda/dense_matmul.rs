//! Runs the dense quantized matrix products over one activation, in the int8
//! q8a128 path or the dequantized float path, and picks each launch's shape.
//!
//! A projection's output and its staging buffer come from the activation's arena,
//! because the output is the next layer's input and must stay in the generation
//! it came from. The vector kernel is instantiated for batches of one to eight
//! rows; a larger batch runs as slices of one staging buffer, and batches past
//! eight go to the MMQ kernel except for the formats MMQ cannot serve. Decode-width
//! projections with narrow outputs run narrow or split over K so they fill the
//! device, and every launch shape produces the same bits. The caller names the
//! output dtype, so the int8 path converts on the store rather than in a separate
//! launch.

use super::*;

pub(super) fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (y_ptr, _y_guard) = y.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_dequantize_mul_mat_vec(
                data_ptr as *const std::ffi::c_void,
                y_ptr as *const f32,
                dst_ptr as *mut f32,
                ncols as i32,
                nrows as i32,
                qtype as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// `inherit` is the **activation's** arena. The weight is a model parameter and
/// names none, so a quantized projection's output belongs wherever the value
/// flowing through the layer came from — and so does the q8_1 staging buffer,
/// which dies at the end of this call.
///
/// `b_size` may exceed the mmvq kernel's 8-batch template ceiling: the batch is
/// then run as `ceil(b_size / 8)` launches over ≤8-row slices of ONE staging
/// buffer into ONE output. Both are batch-major — the q8_1 staging is
/// `[b][ncols_padded]` blocks, `dst` is `[b][nrows]` — so a slice is a pointer
/// offset, not a copy, and the rows of an N-row call are bit-identical to the
/// same ≤8-row calls run separately. `fwd` routes batches past 8 to MMQ, so the
/// chunking's caller is the format MMQ cannot take — Q8_K forces the vec path
/// at every batch size, and used to fail outright past 8.
#[allow(clippy::too_many_arguments)]
pub(super) fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
    inherit: Backing,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 {
        crate::bail!("mmvq needs at least one batch row")
    }
    // Start by quantizing y — the WHOLE batch in one launch, whatever the
    // per-launch ceiling below slices it into.
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let row_q8_bytes = ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let y_size_in_bytes = b_size * row_q8_bytes;
    let qtype = dtype_to_qtype(dtype)?;
    let (dst, dst_backing) = unsafe { alloc_inheriting::<f32>(dev, nrows * b_size, inherit)? };
    // The quantized activation is staged through the device's reused scratch
    // ([`CudaDevice::with_staging`]): it lives from the quantize to the kernel's
    // read and no further, so one buffer serves every call on the stream.
    dev.with_staging(y_size_in_bytes, |y_q8_1| {
        quantize_q8_1(y, y_q8_1, ncols, b_size, dev)?;
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (y_q8_1_ptr, _y_guard) = y_q8_1.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        // The kernel is templated over its batch count, instantiated 1..=8 —
        // each thread carries one accumulator per batch, so the ceiling is
        // register pressure, not convention.
        let mut off = 0usize;
        while off < b_size {
            let chunk = (b_size - off).min(8);
            unsafe {
                run_mul_mat_vec_q8_1(
                    (data_ptr as *const u8) as *const std::ffi::c_void,
                    (y_q8_1_ptr as *const u8).add(off * row_q8_bytes) as *const std::ffi::c_void,
                    (dst_ptr as *mut f32).add(off * nrows),
                    ncols as i32,
                    nrows as i32,
                    ncols_padded as i32,
                    nrows as i32,
                    chunk as i32,
                    qtype as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
            off += chunk;
        }
        Ok(())
    })?;
    Ok(CudaStorage::wrap_cuda_slice_backed(
        dst,
        dev.clone(),
        dst_backing,
    ))
}

/// As [`mul_mat_vec_via_q8_1`], including what `inherit` means.
#[allow(clippy::too_many_arguments)]
pub(super) fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
    inherit: Backing,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let qtype = dtype_to_qtype(dtype)?;
    let (dst, dst_backing) = unsafe { alloc_inheriting::<f32>(dev, x_rows * y_cols, inherit)? };
    // Staged as in [`mul_mat_vec_via_q8_1`].
    dev.with_staging(y_size_in_bytes, |y_q8_1| {
        quantize_q8_1(y, y_q8_1, k, y_cols, dev)?;
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (y_q8_1_ptr, _y_guard) = y_q8_1.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_mul_mat(
                data_ptr as *const std::ffi::c_void,
                y_q8_1_ptr as *const std::ffi::c_void,
                dst_ptr as *mut f32,
                x_cols as i32,
                x_rows as i32,
                y_cols as i32,
                k_padded as i32,
                x_rows as i32,
                qtype as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    })?;
    Ok(CudaStorage::wrap_cuda_slice_backed(
        dst,
        dev.clone(),
        dst_backing,
    ))
}

/// Float dense quantized matmul shared by `QCudaStorage::matmul_gemx` and the `Float`
/// arm of [`dense_qmatmul`]: quantized weight `[nrows(N) x ncols(K)]` (qtype, `weight_len`
/// bytes at `weight_ptr`) times float activations `rhs` -> same-dtype output `[.. x N]`.
/// An unsupported activation dtype is converted to BF16 and the output converted back.
#[allow(clippy::too_many_arguments)]
pub(super) fn dense_qmatmul_float(
    weight_ptr: u64,
    qtype: i32,
    weight_len: usize,
    nrows: usize,
    ncols: usize,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
    device: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    use crate::cuda_backend::CudaStorageSlice;

    let (batch_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => crate::bail!(
            "unexpected rhs shape in quantized_matmul {:?}",
            rhs_l.shape()
        ),
    };
    if ncols != k {
        crate::bail!(
            "mismatch on matmul dim N={nrows} K={ncols} vs rhs {:?}",
            rhs_l.shape()
        )
    }

    let input_dtype = rhs.dtype();
    let needs_conversion = !matches!(
        input_dtype,
        crate::DType::F16 | crate::DType::BF16 | crate::DType::F32
    );
    let rhs_converted: Option<CudaStorage> = if needs_conversion {
        Some(rhs.to_dtype(rhs_l, crate::DType::BF16)?)
    } else {
        None
    };
    let rhs_storage = rhs_converted.as_ref().unwrap_or(rhs);
    let rhs_layout_owned;
    let rhs_layout = if rhs_converted.is_some() {
        rhs_layout_owned = crate::Layout::contiguous(rhs_l.shape());
        &rhs_layout_owned
    } else {
        rhs_l
    };

    let ytype = match &rhs_storage.slice {
        CudaStorageSlice::F16(_) => YType::F16,
        CudaStorageSlice::BF16(_) => YType::BF16,
        CudaStorageSlice::F32(_) => YType::F32,
        _ => unreachable!("should have been converted to BF16"),
    };

    enum OutputSlice {
        F16(CudaSlice<f16>),
        BF16(CudaSlice<bf16>),
        F32(CudaSlice<f32>),
    }
    // **The output inherits the activation's provenance**, exactly as an
    // elementwise op does (`Map2::map` takes `s1.backing`). It used to be a bare
    // `device.alloc`, which made every dense quantized matmul output `Owned` —
    // and since a projection's output is the next layer's input, that stranded
    // the entire activation chain off the reservation. Measured on the 3.6-35B:
    // the DeltaNet mixer's `qkv` had no wave ticket, so its span tables, its
    // `empty_beside` temporaries and everything downstream allocated from the
    // driver too. One dropped backing, dozens of reported sites.
    let origin = rhs_storage.backing;
    let (dst_slice, out_backing) = match ytype {
        YType::F16 => {
            let (s, b) = unsafe { alloc_inheriting::<f16>(device, nrows * batch_size, origin)? };
            (OutputSlice::F16(s), b)
        }
        YType::BF16 => {
            let (s, b) = unsafe { alloc_inheriting::<bf16>(device, nrows * batch_size, origin)? };
            (OutputSlice::BF16(s), b)
        }
        YType::F32 => {
            let (s, b) = unsafe { alloc_inheriting::<f32>(device, nrows * batch_size, origin)? };
            (OutputSlice::F32(s), b)
        }
        YType::Q8A128 => {
            crate::bail!("YType::Q8A128 is the int8 matmul INPUT format, not an FP output dtype")
        }
    };

    {
        let stream = device.cuda_stream();
        let segment = VxSegment {
            weights: weight_ptr as *const std::ffi::c_void,
            batch_count: batch_size as i32,
        };
        macro_rules! run_matmul {
            ($y_ptr:expr, $dst_ptr:expr) => {{
                let status = unsafe {
                    run_quantized_matmul(
                        &segment as *const VxSegment,
                        1,
                        $y_ptr as *const std::ffi::c_void,
                        $dst_ptr as *mut std::ffi::c_void,
                        ncols as i32,
                        nrows as i32,
                        k as i32,
                        nrows as i32,
                        qtype,
                        ytype as i32,
                        weight_len,
                        0, // tile_mode: FP path ignores it
                        // out_dtype: the FP kernels store at the activation dtype, so this
                        // only has to be a valid code — it is not consulted.
                        OutDType::F32 as i32,
                        // sum_norm: likewise not consulted. This is the FLOAT
                        // activation path; there is no q8a128 header to interpret.
                        SumScale::Raw.as_code(),
                        stream.cu_stream() as *mut std::ffi::c_void,
                    )
                };
                check_matmul_status(status, "dense_qmatmul_float")?;
            }};
        }
        match (&rhs_storage.slice, &dst_slice) {
            (CudaStorageSlice::F16(y_data), OutputSlice::F16(dst)) => {
                let y_view = match rhs_layout.contiguous_offsets() {
                    Some((o1, o2)) => y_data.slice(o1..o2),
                    None => {
                        return Err(crate::Error::RequiresContiguous {
                            op: "quantized_matmul",
                        }
                        .bt())?
                    }
                };
                let (y_ptr, _y_guard) = y_view.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                run_matmul!(y_ptr, dst_ptr);
            }
            (CudaStorageSlice::BF16(y_data), OutputSlice::BF16(dst)) => {
                let y_view = match rhs_layout.contiguous_offsets() {
                    Some((o1, o2)) => y_data.slice(o1..o2),
                    None => {
                        return Err(crate::Error::RequiresContiguous {
                            op: "quantized_matmul",
                        }
                        .bt())?
                    }
                };
                let (y_ptr, _y_guard) = y_view.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                run_matmul!(y_ptr, dst_ptr);
            }
            (CudaStorageSlice::F32(y_data), OutputSlice::F32(dst)) => {
                let y_view = match rhs_layout.contiguous_offsets() {
                    Some((o1, o2)) => y_data.slice(o1..o2),
                    None => {
                        return Err(crate::Error::RequiresContiguous {
                            op: "quantized_matmul",
                        }
                        .bt())?
                    }
                };
                let (y_ptr, _y_guard) = y_view.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                run_matmul!(y_ptr, dst_ptr);
            }
            _ => unreachable!("ytype and rhs_storage slice should match"),
        }
    }

    // `wrap_cuda_slice` hardcodes `Backing::Owned`, which would throw away the
    // provenance the allocation above just established.
    let out_storage = CudaStorage {
        slice: match dst_slice {
            OutputSlice::F16(dst) => CudaStorageSlice::F16(dst),
            OutputSlice::BF16(dst) => CudaStorageSlice::BF16(dst),
            OutputSlice::F32(dst) => CudaStorageSlice::F32(dst),
        },
        device: device.clone(),
        backing: out_backing,
        anchor: None,
    };
    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);
    let out_shape: crate::Shape = out_shape.into();
    if needs_conversion {
        let out_layout = crate::Layout::contiguous(&out_shape);
        let converted_out = out_storage.to_dtype(&out_layout, input_dtype)?;
        Ok((converted_out, out_shape))
    } else {
        Ok((out_storage, out_shape))
    }
}

/// Dense (non-MoE) quantized matmul over a [`DynamicTensor`] activation: a single weight
/// `[nrows(N) × ncols(K)]` (KO format for the `Int8` arm, FP GEMX K/128-repack for the
/// `Float` arm) × the activation `[.. × K]` → `[.., N]` (leading dims preserved on both arms).
/// `Int8` runs the q8a128 tensor-core path (its tiling chosen by `q8a128_dense_tile` at
/// dispatch), output F32; `Float` runs the dequant-weight float path
/// ([`dense_qmatmul_float`]), output matching the activation dtype. The caller stays
/// agnostic to which numeric mode runs. `weight_len` is the quantized-weight byte length
/// used by the float path; the int8 path ignores it. See docs/archived/q8_matmul_pipeline.md.
/// KO format code for the segmented qkv kernel's `fmt` field.
///
/// **This table and the switch in `qkv_segmented_f32.cu` are one mapping split across two
/// languages**, and a dtype present here but absent there is the worst of the two failure
/// modes: the kernel's `default` arm falls through and leaves that segment's output columns
/// unwritten, which no caller can distinguish from a completed matmul. `QKV_FMT_COUNT` on the
/// kernel side bounds the index and the launcher refuses one out of range, so the pairing is
/// checked rather than assumed.
///
/// Codes 4 and 5 are appended rather than sorted by width so the original four stay stable.
pub(super) fn ko_fmt_code(dtype: GgmlDType) -> Result<i32> {
    Ok(match dtype {
        GgmlDType::Q4_KO => 0,
        GgmlDType::Q5_KO => 1,
        GgmlDType::Q6_KO => 2,
        GgmlDType::Q8_KO => 3,
        GgmlDType::Q2_KO => 4,
        GgmlDType::Q3_KO => 5,
        other => {
            crate::bail!("qkv_segmented_matmul: unsupported segment dtype {other:?} (KO only)")
        }
    })
}

/// The segments one segmented launch carries (`qkv_segmented_f32.cu` passes them by value as
/// `s0`, `s1`, `s2`): a launch over more would compute the rest with the third's weights.
pub(crate) const QKV_MAX_SEGS: usize = 3;

/// Fused qkv int8 dense matmul: a SINGLE launch multiplies the shared q8a128 activation `op`
/// (`[lead.., K]`) by up to three KO weights of possibly-different formats, writing the
/// **concatenated** `[lead.., ΣN]` output at `out_dtype`. Per thread-block the global N-tile resolves to one
/// segment (boundaries align to the 32-row tile → no divergence) and runs that segment's format on
/// the shared activation. Float-identical to running the segments as separate `q8a128_dense_matmul`
/// calls, but one launch with full GPU occupancy (the tiny k/v GEMVs no longer starve).
///
/// `segments`: `(weight device ptr, KO dtype, N)` for each of q, k, v — N must be a multiple of 32.
pub(crate) fn qkv_segmented_matmul<'w>(
    op: &Q8a128Operand<'w>,
    segments: &[(u64, GgmlDType, usize)],
    out_dtype: crate::DType,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    if !op.cols.is_multiple_of(128) {
        crate::bail!(
            "qkv_segmented_matmul: K={} must be a multiple of 128",
            op.cols
        );
    }
    if segments.is_empty() || segments.len() > QKV_MAX_SEGS {
        crate::bail!(
            "qkv_segmented_matmul: {} segments — the kernel takes 1 to at most {QKV_MAX_SEGS}",
            segments.len()
        );
    }
    let k = op.cols;
    let m = op.rows;

    // Build the device-side segment table (24-byte qkv_seg_t each): weights(8) fmt(4)
    // n_tile_start(4) n_size(4) dst_col_off(4).
    let mut bytes: Vec<u8> = Vec::with_capacity(segments.len() * 24);
    let mut tile_start: i32 = 0;
    let mut col_off: i32 = 0;
    let mut n_total: usize = 0;
    for &(wptr, dtype, n) in segments {
        if n % 32 != 0 {
            crate::bail!("qkv_segmented_matmul: segment N={n} must be a multiple of 32");
        }
        let fmt = ko_fmt_code(dtype)?;
        bytes.extend_from_slice(&wptr.to_le_bytes());
        bytes.extend_from_slice(&fmt.to_le_bytes());
        bytes.extend_from_slice(&tile_start.to_le_bytes());
        bytes.extend_from_slice(&(n as i32).to_le_bytes());
        bytes.extend_from_slice(&col_off.to_le_bytes());
        tile_start += n.div_ceil(32) as i32;
        col_off += n as i32;
        n_total += n;
    }
    let total_n_tiles = tile_start;

    let mode2 = q8a128_dense_use_mode2(m, n_total, k, cached_sm_count(device)) as i32;
    let out_code = out_dtype_code(out_dtype)?;
    // Inherits the fused ln1 activation's arena, so q/k/v land in the attention
    // generation alongside the norm that produced their operand.
    let (dst_ptr, owned_dst, out_backing) =
        resolve_out(out_dtype, op.backing(), device, m * n_total)?;
    op.with_device_ptr(device, |act_ptr| {
        // `bytes` is the HOST segment table (≤3 × 24B); the launcher copies it into by-value kernel
        // params, so there is no per-call device upload.
        let status = unsafe {
            run_qkv_segmented_matmul(
                bytes.as_ptr() as *const std::ffi::c_void,
                segments.len() as i32,
                act_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                k as i32,
                total_n_tiles,
                m as i32,
                n_total as i32,
                mode2,
                out_code,
                // One shared operand feeds all three segments, so its convention
                // is the launch's.
                op.sum_scale.as_code(),
                device.cuda_stream().cu_stream() as *mut std::ffi::c_void,
            )
        };
        check_matmul_status(status, "qkv_segmented_matmul")
    })?;

    let mut out_dims = op.lead.clone();
    out_dims.push(n_total);
    let out_shape: Shape = out_dims.into();
    tensor_from_owned_out(owned_dst, dst_ptr, out_backing, out_shape, device)
}

/// Several KO weights that read the same q8a128 operand, as ONE launch writing their outputs
/// side by side: `[lead.., ΣNᵢ]`, weight `i`'s columns after those before it. At decode width
/// the combined width runs narrow or splits K ([`q8a128_dense_plan`] — the segmented narrow or
/// split-K launch); otherwise it runs the unsplit segmented kernel. Every plan computes each
/// column as its weight's own launch does, bit for bit: the narrow and split sums are
/// tile-ordered whatever their geometry, and the unsplit kernel walks each tile row exactly as
/// the single-weight kernel does. The narrow and split forms need one format across the
/// segments and at most [`SPLITK_MAX_SEGS`] of them; the unsplit form takes up to three of any
/// KO formats.
pub(crate) fn dense_qmatmul_stacked<'w>(
    op: &Q8a128Operand<'w>,
    segments: &[(u64, GgmlDType, usize)],
    out_dtype: crate::DType,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    let n_total: usize = segments.iter().map(|&(_, _, n)| n).sum();
    let dtype = segments
        .first()
        .map(|&(_, d, _)| d)
        .ok_or_else(|| crate::Error::Msg("dense_qmatmul_stacked: no weights".into()))?;
    let one_format = segments.iter().all(|&(_, d, _)| d == dtype);
    let plan = q8a128_dense_plan(
        op.rows,
        n_total,
        op.cols,
        cached_sm_count(device),
        one_format && dtype != GgmlDType::MXFP4_KO && segments.len() <= SPLITK_MAX_SEGS,
        dtype,
    );
    let segs = || -> Vec<(u64, usize)> { segments.iter().map(|&(p, _, n)| (p, n)).collect() };
    match plan {
        DensePlan::Narrow { warps } => {
            q8a128_dense_matmul_narrow_segmented(op, &segs(), dtype, warps, out_dtype, device)
        }
        DensePlan::SplitK(splits) => {
            q8a128_dense_matmul_split_k_segmented(op, &segs(), dtype, splits, out_dtype, device)
        }
        DensePlan::Unsplit => qkv_segmented_matmul(op, segments, out_dtype, device),
    }
}

/// The q8a128 int8 dense launch with an **explicit** tiling ([`DenseTile`]). Production reaches
/// this from [`dense_qmatmul`] with the tiling chosen by [`q8a128_dense_tile`]; the benches and
/// the tiling tests call it directly to run each one at a fixed `(M, N, K)` — every tiling gives
/// the same bits. Result is the `[lead.., N]` output (rank rebuilt from `op.lead`) stored at
/// `out_dtype` — the MMA accumulates in F32 registers and converts on the store, so a narrow
/// `out_dtype` is bit-identical to the F32 kernel followed by a cast, minus the cast.
#[allow(clippy::too_many_arguments)]
pub(crate) fn q8a128_dense_matmul<'w>(
    op: &Q8a128Operand<'w>,
    weight_ptr: u64,
    weight_dtype: GgmlDType,
    nrows: usize,
    weight_len: usize,
    tile: DenseTile,
    out_dtype: crate::DType,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    if !nrows.is_multiple_of(32) {
        crate::bail!("q8a128_dense_matmul: nrows={nrows} must be a multiple of 32");
    }
    if !op.cols.is_multiple_of(128) {
        crate::bail!(
            "q8a128_dense_matmul: K={} must be a multiple of 128",
            op.cols
        );
    }
    let qtype = dtype_to_qtype(weight_dtype)? as i32;
    let (ncols, total_batch) = (op.cols, op.rows);
    let out_code = out_dtype_code(out_dtype)?;
    // Inherits the activation's arena: on the decode path this is o_proj over a
    // wave-backed attention context, so the projection lands in the same
    // generation rather than in the pool.
    let (dst_ptr, owned_dst, out_backing) =
        resolve_out(out_dtype, op.backing(), device, nrows * total_batch)?;
    op.with_device_ptr(device, |act_ptr| {
        let segment = VxSegment {
            weights: weight_ptr as *const std::ffi::c_void,
            batch_count: total_batch as i32,
        };
        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1, // num_segments = 1 (non-MoE)
                act_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,       // ncols_x = K
                nrows as i32,       // nrows_x = N
                total_batch as i32, // nrows_y = M
                nrows as i32,       // nrows_dst = N
                qtype,
                YType::Q8A128 as i32, // int8 activation; tiling forced via `tile`
                weight_len,           // weight_bytes (FP path only; int8 ignores)
                tile.code(),          // mode-1 / mode-2 / mode-4
                out_code,             // store width
                // The operand's own Σx convention — the bytes say how to read them.
                op.sum_scale.as_code(),
                device.cuda_stream().cu_stream() as *mut std::ffi::c_void,
            )
        };
        check_matmul_status(status, "q8a128_dense_matmul")
    })?;
    // Rebuild the output rank from the operand's leading dims ([lead.., N]) so the int8 arm matches
    // the float arm on 3D+ activations (data is row-major [M, N]).
    let mut out_dims = op.lead.clone();
    out_dims.push(nrows);
    let out_shape: Shape = out_dims.into();
    tensor_from_owned_out(owned_dst, dst_ptr, out_backing, out_shape, device)
}

/// `out_dtype` is the width the result is stored at. The int8 arm passes it to the
/// kernel, which converts the F32 accumulator on the way out — the caller gets the
/// dtype it asked for with no cast launch. The FP arm has no such choice: those
/// kernels store at the activation dtype, so `out_dtype` must equal the input's.
pub fn dense_qmatmul<'w>(
    input: DynamicTensor<'_, 'w>,
    weight_ptr: u64,
    weight_dtype: GgmlDType,
    nrows: usize,
    weight_len: usize,
    out_dtype: crate::DType,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    dense_qmatmul_with_plan(
        input,
        weight_ptr,
        weight_dtype,
        nrows,
        weight_len,
        out_dtype,
        None,
        device,
    )
}

/// [`dense_qmatmul`] with the int8 path's launch chosen by the caller: `Some(plan)` runs that
/// plan, `None` takes [`q8a128_dense_plan`]'s answer. Every plan produces the same bits. The
/// forced form is what the KO projection bench sweeps; production calls [`dense_qmatmul`].
#[allow(clippy::too_many_arguments)]
pub fn dense_qmatmul_with_plan<'w>(
    input: DynamicTensor<'_, 'w>,
    weight_ptr: u64,
    weight_dtype: GgmlDType,
    nrows: usize,
    weight_len: usize,
    out_dtype: crate::DType,
    plan: Option<DensePlan>,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    ensure_qmatmul_pairing(&input, weight_dtype)?;
    let qtype = dtype_to_qtype(weight_dtype)? as i32;
    match input {
        DynamicTensor::Int8(op) => {
            // A decode-width projection with a narrow N cannot fill the card unsplit — 13 blocks
            // for the 416-wide hyper-connection `down` on 110 SMs. Up to eight rows it runs
            // narrow, wider it splits K; everything the rule declines (every prefill shape among
            // them) takes the tilings below unchanged. MXFP4 does neither: its per-sub fold
            // accumulates straight into the running sum, which per-tile folds cannot reproduce
            // bit for bit.
            let sm = cached_sm_count(device);
            let plan = plan.unwrap_or_else(|| {
                q8a128_dense_plan(
                    op.rows,
                    nrows,
                    op.cols,
                    sm,
                    weight_dtype != GgmlDType::MXFP4_KO,
                    weight_dtype,
                )
            });
            match plan {
                DensePlan::Narrow { warps } => {
                    return q8a128_dense_matmul_narrow_segmented(
                        op,
                        &[(weight_ptr, nrows)],
                        weight_dtype,
                        warps,
                        out_dtype,
                        device,
                    );
                }
                DensePlan::SplitK(splits) => {
                    return q8a128_dense_matmul_split_k_segmented(
                        op,
                        &[(weight_ptr, nrows)],
                        weight_dtype,
                        splits,
                        out_dtype,
                        device,
                    );
                }
                DensePlan::Unsplit => {}
            }
            // Tiling choice (mode-1 Bm=16, mode-2 Bm=32, mode-4 Bm=64 × 128 rows): a grid-fill
            // decision driven by M and N (block count vs SM count) plus mode-2's [17,32] trap —
            // not weight bytes. The bench reaches the same launch with a forced tiling via
            // `q8a128_dense_matmul`.
            let tile = q8a128_dense_tile(op.rows, nrows, op.cols, sm);
            q8a128_dense_matmul(
                op,
                weight_ptr,
                weight_dtype,
                nrows,
                weight_len,
                tile,
                out_dtype,
                device,
            )
        }
        DynamicTensor::Float(t) => {
            let (storage, layout) = t.storage_and_layout();
            let cuda = match &*storage {
                crate::Storage::Cuda(c) => c,
                _ => crate::bail!("dense_qmatmul: Float input must be a CUDA tensor"),
            };
            let ncols = *layout.shape().dims().last().unwrap();
            let (out_storage, out_shape) = dense_qmatmul_float(
                weight_ptr, qtype, weight_len, nrows, ncols, cuda, layout, device,
            )?;
            // Checked against what the path actually produced rather than re-deriving
            // it from the activation: the FP kernels store at the activation dtype,
            // but an exotic activation is promoted first, so the input dtype is not
            // always the answer.
            if out_storage.dtype() != out_dtype {
                crate::bail!(
                    "dense_qmatmul: the FP path produced {:?}, but {out_dtype:?} was requested",
                    out_storage.dtype()
                );
            }
            Ok(crate::tensor::from_storage(
                crate::Storage::Cuda(out_storage),
                out_shape,
                crate::op::BackpropOp::none(),
                false,
            ))
        }
    }
}
