//! Dense quantized matmuls on the device: every batch width, and the regressions
//! that once broke decode.
//!
//! The tensor-core matmul must give each activation row exactly the output it
//! gets alone, at every batch from 1 to 64, so a remainder tile that drops or
//! misplaces a row fails bit for bit. The regression tests cover the fused gate
//! and up weights at large output width, and the cast of a contiguous view with
//! a non-zero offset, which once made the up projection equal the gate. The
//! K-quant tests check the repacked matmul against a CPU dequantization, and the
//! GGUF-loaded tests run against a model file named at run time.

use super::*;

/// `bytes` from device address `src` into host memory at `dst`, on `dev`'s
/// stream behind everything issued there, and waited for — the readback the
/// raw-pointer tests below make, as a driver status.
pub(super) fn copy_to_host(
    dev: &CudaDevice,
    dst: *mut c_void,
    src: u64,
    bytes: usize,
) -> cudarc::driver::sys::CUresult {
    use cudarc::driver::sys;
    let stream = dev.cuda_stream();
    // SAFETY: the caller's `dst` holds `bytes`, and `src` names `bytes` of live
    // device memory.
    let r = unsafe { sys::cuMemcpyDtoHAsync_v2(dst, src, bytes, stream.cu_stream()) };
    if r != sys::CUresult::CUDA_SUCCESS {
        return r;
    }
    // SAFETY: the device's live stream.
    unsafe { sys::cuStreamSynchronize(stream.cu_stream()) }
}

/// Guards the large-N (2*ffn) gemx matmul at batch=1: a fused ffn_gate+ffn_up
/// weight has distinct gate (rows 0..N/2) and up (rows N/2..N) halves, and the
/// kernel must not collapse the up half onto the gate half. This path was the
/// prime suspect while chasing a decode `up == gate` corruption at N=16384 — the
/// kernel was exonerated here (the real bug was an aliased `to_dtype_mut`, see
/// `cuda_to_dtype_mut_respects_start_offset`); this test pins the kernel so a
/// future regression in the large-N batch=1 gemx path is caught directly.
#[test]
pub(super) fn cuda_mm_gemx_large_n_batch1_no_row_aliasing() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    // **N is the dimension under test; K is only the dot-product length.** The
    // bug this pins is row indexing at large N, so `nrows` stays at the size that
    // failed. `ncols` was 3072, which made the fixture 50M elements — and
    // `QTensor::quantize` on CUDA runs on the *host* (it downloads, quantizes,
    // uploads), so that was two 100 MB round trips through a debug-build
    // quantizer for a test that never looks at K. At 512 the separation between
    // the halves is ~92 against a 0.5 threshold, so nothing is being run close to
    // the edge.
    let ncols = 512usize; // K (hidden)
    let half_n = 8192usize; // ffn
    let nrows = 2 * half_n; // N = 2*ffn (the failing size)
    let device = crate::Device::Cuda(dev.clone());

    // Build the gate (rows 0..ffn) and up (rows ffn..2ffn) weights SEPARATELY,
    // quantize each, then fuse via concat_rows_cuda — exactly how the MLP builds
    // its fused ffn_gate+ffn_up weight. Distinct patterns so the halves differ.
    //
    // Generated on the device from a row and a column ramp, rather than by a host
    // loop. The fixture is 2 × 8192 × 3072 = 50M elements, and a `sin` each on
    // the host in a debug build is most of a ten-second test — for data whose
    // only requirement is that the two halves differ and quantize sensibly. The
    // broadcast below is the same function of (row, col) the loop computed.
    let rows = crate::Tensor::arange(0f32, half_n as f32, &device)?.reshape((half_n, 1))?;
    let cols = crate::Tensor::arange(0f32, ncols as f32, &device)?.reshape((1, ncols))?;
    let gate_t = rows
        .affine(7.0, 0.0)?
        .broadcast_add(&cols.affine(11.0, 0.0)?)?
        .affine(0.001, 0.0)?
        .sin()?;
    let up_t = rows
        .affine(13.0, 0.0)?
        .broadcast_add(&cols.affine(5.0, 0.0)?)?
        .affine(0.001, 0.0)?
        .cos()?;
    let wg = crate::quantized::QTensor::quantize(&gate_t, GgmlDType::Q4_K)?;
    let wu = crate::quantized::QTensor::quantize(&up_t, GgmlDType::Q4_K)?;
    let fused = crate::quantized::QTensor::concat_rows_cuda(&[&wg, &wu])?;
    let fused_shape = fused.shape().clone();
    let xs_repacked = match fused.storage() {
        crate::quantized::QStorage::Cuda(s) => s.repack_gemx(&fused_shape)?,
        _ => unreachable!(),
    };
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;

    // Reference (batch>=2 is known-good) up value, filled on the y_cols=2 pass.
    let mut ref_up: Option<f32> = None;
    for &y_cols in &[2usize, 1, 4] {
        let y_data: Vec<f16> = vec![f16::from_f32(1.0); ncols * y_cols];
        let y = dev.memcpy_stod(&y_data)?;
        let dst = unsafe { dev.alloc::<f16>(nrows * y_cols)? };
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _g) = xs_repacked.data.inner.device_ptr(&stream);
            let segment = VxSegment {
                weights: data_ptr as *const std::ffi::c_void,
                batch_count: y_cols as i32,
            };
            let (y_ptr, _gy) = y.device_ptr(&stream);
            let (dst_ptr, _gd) = dst.device_ptr(&stream);
            let status = unsafe {
                run_quantized_matmul(
                    &segment as *const VxSegment,
                    1,
                    y_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    ncols as i32,
                    nrows as i32,
                    ncols as i32,
                    nrows as i32,
                    qtype,
                    YType::F16 as i32,
                    xs_repacked.data.len,
                    0,
                    OutDType::F16 as i32,
                    SumScale::Raw.as_code(),
                    stream.cu_stream() as *mut c_void,
                )
            };
            assert_eq!(status, 0, "matmul launcher rejected the call");
        }
        dev.synchronize()?;
        let res = dev.memcpy_dtov(&dst.slice(..))?;
        // batch 0 occupies dst[0..nrows].
        let gate0 = f16::to_f32(res[0]);
        let up0 = f16::to_f32(res[nrows / 2]);
        println!("y_cols={y_cols}: gate[0]={gate0:.3}  up[N/2]={up0:.3}");
        if y_cols == 2 {
            ref_up = Some(up0);
        }
        // The up row and gate row use different weight patterns, so their sums differ.
        assert!(
            (up0 - gate0).abs() > 0.5,
            "y_cols={y_cols}: UP ROW ALIASED TO GATE: up={up0} ~= gate={gate0}"
        );
        // And the up value must match the known-good (batch>=2) reference.
        if let Some(r) = ref_up {
            assert!(
                (up0 - r).abs() < 0.5,
                "y_cols={y_cols}: up row wrong: {up0} vs reference {r}"
            );
        }
    }
    Ok(())
}

/// Every tensor-core matmul batch from 1 to 64 — each remainder 1-15 under the
/// tc16 kernels (batch 1-31) and the tc32 kernels (batch 32-64), plus the
/// remainder-free batches 16, 32, 48, 64 — must give each activation row exactly
/// the output that row gets alone. An MMA output row reads only its own A row, so
/// a row's result does not depend on which tile (tc32, tc16 or the remainder
/// tile) carries it, and the comparison is bit-exact. The remainder tile reads
/// its row count at runtime: a wrong count either drops real rows (they keep the
/// sentinel) or stores padding rows past the batch (the row after the batch loses
/// its sentinel), and both fail here.
#[test]
pub(super) fn cuda_mm_gemx_every_batch_matches_rows_alone() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let ncols = 256usize; // K
    let nrows = 64usize; // N: two 32-row tiles
    let max_batch = 64usize;
    let device = crate::Device::Cuda(dev.clone());

    let rows = crate::Tensor::arange(0f32, nrows as f32, &device)?.reshape((nrows, 1))?;
    let cols = crate::Tensor::arange(0f32, ncols as f32, &device)?.reshape((1, ncols))?;
    let w = rows
        .affine(0.37, 0.0)?
        .broadcast_add(&cols.affine(0.11, 0.0)?)?
        .sin()?;
    let wq = crate::quantized::QTensor::quantize(&w, GgmlDType::Q8_0)?;
    let w_shape = wq.shape().clone();
    let w_repacked = match wq.storage() {
        crate::quantized::QStorage::Cuda(s) => s.repack_gemx(&w_shape)?,
        _ => unreachable!(),
    };
    let qtype = dtype_to_qtype(GgmlDType::Q8_0)? as i32;

    // Activation row b is a distinct ramp, so a row stored in the wrong slot
    // is caught as well as a row left unwritten.
    let y_data: Vec<f16> = (0..max_batch * ncols)
        .map(|i| {
            let (b, k) = (i / ncols, i % ncols);
            f16::from_f32((((b * 7 + k * 3) % 61) as f32 - 30.0) / 16.0)
        })
        .collect();
    let y = dev.memcpy_stod(&y_data)?;
    let sentinel = f16::from_bits(0x7E01);

    // Runs one matmul of `batch` rows starting at activation row `first`, into a
    // destination one row longer than the batch and pre-filled with the sentinel.
    let run = |first: usize, batch: usize| -> Result<Vec<f16>> {
        let dst = dev.memcpy_stod(&vec![sentinel; (batch + 1) * nrows])?;
        {
            let stream = dev.cuda_stream();
            let (w_ptr, _gw) = w_repacked.data.inner.device_ptr(&stream);
            let segment = VxSegment {
                weights: w_ptr as *const c_void,
                batch_count: batch as i32,
            };
            let (y_ptr, _gy) = y.device_ptr(&stream);
            let (dst_ptr, _gd) = dst.device_ptr(&stream);
            let y_first =
                (y_ptr as usize + first * ncols * std::mem::size_of::<f16>()) as *const c_void;
            let status = unsafe {
                run_quantized_matmul(
                    &segment as *const VxSegment,
                    1,
                    y_first,
                    dst_ptr as *mut c_void,
                    ncols as i32,
                    nrows as i32,
                    ncols as i32,
                    nrows as i32,
                    qtype,
                    YType::F16 as i32,
                    w_repacked.data.len,
                    0,
                    OutDType::F16 as i32,
                    SumScale::Raw.as_code(),
                    stream.cu_stream() as *mut c_void,
                )
            };
            assert_eq!(status, 0, "matmul launcher rejected batch {batch}");
        }
        dev.synchronize()?;
        Ok(dev.memcpy_dtov(&dst.slice(..))?)
    };

    let alone: Vec<Vec<u16>> = (0..max_batch)
        .map(|b| {
            let out = run(b, 1)?;
            Ok(out[..nrows].iter().map(|v| v.to_bits()).collect())
        })
        .collect::<Result<_>>()?;
    for row in &alone {
        assert!(
            row.iter().all(|&bits| bits != sentinel.to_bits()),
            "batch 1 left an output unwritten"
        );
    }

    for batch in 1..=max_batch {
        let out = run(0, batch)?;
        for b in 0..batch {
            let got: Vec<u16> = out[b * nrows..(b + 1) * nrows]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(
                got, alone[b],
                "batch {batch}: row {b} differs from the row alone"
            );
        }
        assert!(
            out[batch * nrows..]
                .iter()
                .all(|v| v.to_bits() == sentinel.to_bits()),
            "batch {batch}: wrote past the last row"
        );
    }
    Ok(())
}

/// Regression: `to_dtype_mut` on a *contiguous view with a non-zero start offset*
/// (e.g. the second half of a last-dim `narrow`) must cast the offset slice, not
/// the buffer start. The fused ffn_gate+ffn_up MLP hit this: at decode M=1 the
/// `up = gu.narrow(last, ffn, ffn)` view is contiguous (offset ffn); the in-place
/// cast ignored the offset and produced `up == gate`, corrupting decode.
#[test]
pub(super) fn cuda_to_dtype_mut_respects_start_offset() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());
    let n = 256usize;
    // [1, 1, 2n]: first half = 1.0, second half = 2.0.
    let data: Vec<f32> = (0..2 * n).map(|i| if i < n { 1.0 } else { 2.0 }).collect();
    let gu = crate::Tensor::from_vec(data, (1, 1, 2 * n), &device)?.to_dtype(crate::DType::F16)?;

    let mut up = gu.narrow(2, n, n)?; // contiguous, start_offset = n
    assert!(up.is_contiguous());
    up.to_dtype_mut(crate::DType::BF16)?;
    let up_v = up
        .flatten_all()?
        .to_dtype(crate::DType::F32)?
        .to_vec1::<f32>()?;
    assert!(
        up_v.iter().all(|&x| (x - 2.0).abs() < 1e-3),
        "up half aliased to gate: got {:?}",
        &up_v[..4]
    );

    let mut gate = gu.narrow(2, 0, n)?; // contiguous, start_offset = 0
    gate.to_dtype_mut(crate::DType::BF16)?;
    let gate_v = gate
        .flatten_all()?
        .to_dtype(crate::DType::F32)?
        .to_vec1::<f32>()?;
    assert!(
        gate_v.iter().all(|&x| (x - 1.0).abs() < 1e-3),
        "gate half wrong: got {:?}",
        &gate_v[..4]
    );
    Ok(())
}

#[test]
pub(super) fn cuda_quantize_q8_1() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let el = 256;
    let el_padded = pad(el, MATRIX_ROW_PADDING);
    let y_size_in_bytes = el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
    let y = dev.memcpy_stod(&vs)?;
    quantize_q8_1(&y.slice(..), &mut y_q8_1, el, 1, &dev)?;
    Ok(())
}

#[test]
pub(super) fn cuda_mmv_q8_1() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let ncols = 256;
    let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
    let y = dev.memcpy_stod(&vs)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
    let cuda_storage = mul_mat_vec_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_0,
        /* ncols */ ncols,
        /* nrows */ 1,
        /* b_size */ 1,
        &dev,
        // No wave here, so nothing to inherit: the result is an ordinary owned
        // allocation, which is what these tests read back from.
        Backing::Owned,
    )?;
    let vs = cuda_storage.as_cuda_slice::<f32>()?;
    let vs = dev.memcpy_dtov(&vs.slice(..))?;
    assert_eq!(vs.len(), 1);
    // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
    // Q8 means 1/256 precision.
    assert_eq!(vs[0], 5561664.5);

    let cuda_storage = dequantize_mul_mat_vec(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_0,
        /* ncols */ ncols,
        /* nrows */ 1,
        &dev,
    )?;
    let vs = cuda_storage.as_cuda_slice::<f32>()?;
    let vs = dev.memcpy_dtov(&vs.slice(..))?;
    assert_eq!(vs.len(), 1);
    assert_eq!(vs[0], 5561851.0);
    Ok(())
}

/// A batch past the mmvq templates' 8-row ceiling runs as ≤8-row launches over
/// slices of one staging buffer into one output — and the slicing is pure
/// pointer arithmetic, so the result must be BIT-identical to running the
/// slices as separate calls. Identity is the assertion because the chunks run
/// the same kernels on the same bytes; only the offsets can be wrong, and a
/// wrong offset moves whole rows, which tolerance would still catch but
/// identity catches without an argument about thresholds.
///
/// 12 rows on purpose: `sessions × (draft + 1)` of a four-session verify wave
/// at draft budget 2 — the production count that used to fall off the vec path
/// onto the MMQ tile.
#[test]
pub(super) fn cuda_mmv_q8_1_batches_past_the_template_ceiling() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let (ncols, nrows, b_size) = (512usize, 96usize, 12usize);
    // Distinct values row to row, so a chunk reading another's rows cannot
    // coincidentally agree.
    let vs: Vec<f32> = (0..ncols * b_size)
        .map(|v| ((v * 37 + 11) % 199) as f32 / 16.0 - 6.0)
        .collect();
    let y = dev.memcpy_stod(&vs)?;
    let w: Vec<f32> = (0..ncols * nrows)
        .map(|v| ((v * 13 + 5) % 251) as f32 / 32.0 - 4.0)
        .collect();
    let w_dev = dev.memcpy_stod(&w)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q6_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(w_dev, dev.clone()))?;

    let run = |view: &CudaView<f32>, b: usize| -> Result<Vec<f32>> {
        let out = mul_mat_vec_via_q8_1(
            &xs.data,
            view,
            GgmlDType::Q6_K,
            ncols,
            nrows,
            b,
            &dev,
            Backing::Owned,
        )?;
        let out = out.as_cuda_slice::<f32>()?;
        Ok(dev.memcpy_dtov(&out.slice(..))?)
    };

    let whole = run(&y.slice(..), b_size)?;
    assert_eq!(whole.len(), b_size * nrows);
    // Compare against the exact ≤8-row slices the implementation takes — 8 then
    // 4 — and no other split: the kernel's warp count varies with its batch
    // size, so calls at different batch sizes reduce in different orders and
    // only the same-sized launch is bit-comparable.
    let first = run(&y.slice(..8 * ncols), 8)?;
    let second = run(&y.slice(8 * ncols..), b_size - 8)?;
    assert_eq!(&whole[..8 * nrows], &first[..], "rows 0..8 differ");
    assert_eq!(&whole[8 * nrows..], &second[..], "rows 8..12 differ");
    Ok(())
}

#[test]
pub(super) fn cuda_mm_q8_1() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let ncols = 256;
    let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
    let y = dev.memcpy_stod(&vs)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
    let cuda_storage = mul_mat_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_0,
        /* x_rows */ 4,
        /* x_cols */ ncols,
        /* y_rows */ ncols,
        /* y_cols */ 4,
        &dev,
        Backing::Owned,
    )?;
    let vs = cuda_storage.as_cuda_slice::<f32>()?;
    let vs = dev.memcpy_dtov(&vs.slice(..))?;

    /*
       x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
       x @ x.t() / 16
    tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
            [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
            [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
            [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
            */
    assert_eq!(vs.len(), 16);
    assert_eq!(vs[0], 347604.0);
    assert_eq!(vs[1], 888153.06);
    assert_eq!(vs[4], 869780.7);
    assert_eq!(vs[5], 2483145.0);
    assert_eq!(vs[11], 9407368.0);
    assert_eq!(vs[14], 9470856.0);
    assert_eq!(vs[15], 13138824.0);
    Ok(())
}

// The following test used to fail under compute-sanitizer until #2526.
#[test]
pub(super) fn cuda_mm_q8_1_pad() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    let (x_rows, ncols, y_cols) = (4, 16, 2048);
    let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
    let y = dev.memcpy_stod(&vs)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
    let cuda_storage = mul_mat_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_0,
        /* x_rows */ x_rows,
        /* x_cols */ ncols,
        /* y_rows */ ncols,
        /* y_cols */ y_cols,
        &dev,
        Backing::Owned,
    )?;
    let vs = cuda_storage.as_cuda_slice::<f32>()?;
    let _vs = dev.memcpy_dtov(&vs.slice(..))?;
    Ok(())
}

#[test]
pub(super) fn cuda_mm_q4_k() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    // Use dimensions that match typical LLM layers (e.g., 3072 hidden dim)
    let ncols = 256; // K dimension - must be multiple of QK_K=256 for K-quants
    let x_rows = 4;
    let y_cols = 1;

    // Create input data
    let vs: Vec<f32> = (0..ncols * x_rows)
        .map(|v| v as f32 / (ncols as f32))
        .collect();
    let y_data: Vec<f32> = (0..ncols * y_cols)
        .map(|v| v as f32 / (ncols as f32))
        .collect();

    let y = dev.memcpy_stod(&y_data)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&vs)?,
        dev.clone(),
    ))?;

    let cuda_storage = mul_mat_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_K,
        /* x_rows */ x_rows,
        /* x_cols */ ncols,
        /* y_rows */ ncols,
        /* y_cols */ y_cols,
        &dev,
        Backing::Owned,
    )?;
    let result = cuda_storage.as_cuda_slice::<f32>()?;
    let result = dev.memcpy_dtov(&result.slice(..))?;

    // Compute expected result using CPU
    // For a simple check, just verify we get non-garbage values
    assert_eq!(result.len(), x_rows * y_cols);
    println!("Q4_K matmul result (y_cols=1): {:?}", result);

    // Check that results are reasonable (not NaN, not extreme values)
    for &v in &result {
        assert!(!v.is_nan(), "Result contains NaN");
        assert!(!v.is_infinite(), "Result contains Inf");
    }

    Ok(())
}

#[test]
pub(super) fn cuda_mm_q4_k_large() -> Result<()> {
    let dev = CudaDevice::new(0)?;
    // Test with larger dimensions that use MMQ path
    let ncols = 3072; // Hidden dim - should be multiple of 256
    let x_rows = 3072; // Output dim
    let y_cols = 128; // Sequence length - forces MMQ path

    // Create input data
    let weight_data: Vec<f32> = (0..ncols * x_rows)
        .map(|v| ((v % 256) as f32 - 128.0) / 256.0)
        .collect();
    let y_data: Vec<f32> = (0..ncols * y_cols)
        .map(|v| ((v % 256) as f32 - 128.0) / 256.0)
        .collect();

    let y = dev.memcpy_stod(&y_data)?;
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&weight_data)?,
        dev.clone(),
    ))?;

    let cuda_storage = mul_mat_via_q8_1(
        &xs.data,
        &y.slice(..),
        /* dtype */ GgmlDType::Q4_K,
        /* x_rows */ x_rows,
        /* x_cols */ ncols,
        /* y_rows */ ncols,
        /* y_cols */ y_cols,
        &dev,
        Backing::Owned,
    )?;
    let result = cuda_storage.as_cuda_slice::<f32>()?;
    let result = dev.memcpy_dtov(&result.slice(..))?;

    assert_eq!(result.len(), x_rows * y_cols);
    println!(
        "Q4_K matmul result (large): first 10 = {:?}",
        &result[..10.min(result.len())]
    );

    // Check that results are reasonable
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut zero_count = 0;
    for &v in &result {
        if v.is_nan() {
            nan_count += 1;
        }
        if v.is_infinite() {
            inf_count += 1;
        }
        if v == 0.0 {
            zero_count += 1;
        }
    }

    println!(
        "NaN count: {}, Inf count: {}, Zero count: {}",
        nan_count, inf_count, zero_count
    );
    assert_eq!(nan_count, 0, "Result contains NaN values");
    assert_eq!(inf_count, 0, "Result contains Inf values");

    Ok(())
}

/// Test Q4_K matmul with GEMX-repacked weights (K/128 with embedded scales)
/// This tests the FULL path used by the model: repack + run_quantized_matmul
#[test]
pub(super) fn cuda_mm_q4_k_repacked() -> Result<()> {
    use crate::Shape;
    use half::f16;

    let dev = CudaDevice::new(0)?;
    let ncols = 256; // K dimension - must be multiple of QK_K=256 for K-quants
    let x_rows = 4; // Only 4 rows = 1 phase
    let y_cols = 1;

    // Create VARYING weight data - need variance for quantization to work!
    // GGML returns (0,0) if all values in a 32-element block are equal
    let weights: Vec<f32> = (0..ncols * x_rows)
        .map(|v| {
            let row = v / ncols;
            let col = v % ncols;
            // Vary within each row: col/ncols gives 0..1 range
            // Scale by (row+1) to make rows distinguishable
            (row as f32 + 1.0) * (col as f32 / ncols as f32)
        })
        .collect();
    println!("Input weights: row 0 first 8: {:?}", &weights[..8]);
    println!(
        "Input weights: row 0 last 8: {:?}",
        &weights[ncols - 8..ncols]
    );
    println!(
        "Input weights: row 1 first 8: {:?}",
        &weights[ncols..ncols + 8]
    );

    // First quantize on CPU to verify
    let cpu_storage = crate::CpuStorage::F32(weights.clone());
    let cpu_src = crate::Storage::Cpu(cpu_storage);
    let mut cpu_q = crate::Device::Cpu.qzeros(ncols * x_rows, GgmlDType::Q4_K)?;
    cpu_q.quantize(&cpu_src)?;
    let cpu_data = cpu_q.data()?;
    println!("CPU quantized size: {} bytes", cpu_data.len());

    // Parse first block_q4_K on CPU
    let dm_bytes = &cpu_data[0..4];
    let d = f16::from_ne_bytes([dm_bytes[0], dm_bytes[1]]);
    let dmin = f16::from_ne_bytes([dm_bytes[2], dm_bytes[3]]);
    println!(
        "CPU Block 0 dm: d={}, dmin={}",
        f16::to_f32(d),
        f16::to_f32(dmin)
    );

    // scales are at offset 4 (after dm)
    let scales_bytes = &cpu_data[4..16];
    println!("CPU Block 0 scales[0..12]: {:?}", scales_bytes);

    // Create Y data (F16 for GEMX path) - all ones
    let y_data: Vec<f16> = (0..ncols * y_cols).map(|_| f16::from_f32(1.0)).collect();
    let y = dev.memcpy_stod(&y_data)?;

    // Create and quantize weights on GPU
    let shape = Shape::from((x_rows, ncols));
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&weights)?,
        dev.clone(),
    ))?;

    // Read back quantized data to verify structure
    let q_bytes = dev.memcpy_dtov(&xs.data.inner.slice(..xs.data.len))?;
    println!("GPU quantized size: {} bytes", q_bytes.len());

    // Parse first block_q4_K header (dm is at offset 0, 4 bytes = 2 half values)
    let dm_bytes = &q_bytes[0..4];
    let d = f16::from_ne_bytes([dm_bytes[0], dm_bytes[1]]);
    let dmin = f16::from_ne_bytes([dm_bytes[2], dm_bytes[3]]);
    println!(
        "GPU Block 0 dm: d={}, dmin={}",
        f16::to_f32(d),
        f16::to_f32(dmin)
    );

    // Second block (offset 144)
    let dm_bytes = &q_bytes[144..148];
    let d = f16::from_ne_bytes([dm_bytes[0], dm_bytes[1]]);
    let dmin = f16::from_ne_bytes([dm_bytes[2], dm_bytes[3]]);
    println!(
        "GPU Block 1 dm: d={}, dmin={}",
        f16::to_f32(d),
        f16::to_f32(dmin)
    );

    // Repack to K/128 format with embedded scales
    let xs_repacked = xs.repack_gemx(&shape)?;
    println!(
        "Repacked Q4_K: {} bytes -> {} bytes",
        xs.data.len, xs_repacked.data.len
    );

    // Compute expected result using CPU dequantization
    let mut dequant = vec![0.0f32; ncols * x_rows];
    {
        // Dequantize CPU data
        let cpu_blocks = unsafe {
            std::slice::from_raw_parts(
                cpu_data.as_ptr() as *const crate::quantized::k_quants::BlockQ4_K,
                x_rows,
            )
        };
        crate::quantized::k_quants::GgmlType::to_float(cpu_blocks, &mut dequant);
    }
    println!("Dequant row 0 first 8: {:?}", &dequant[..8]);
    println!("Dequant row 1 first 8: {:?}", &dequant[ncols..ncols + 8]);

    // Expected result: sum of dequant[row, :] for each row
    let expected: Vec<f32> = (0..x_rows)
        .map(|row| dequant[row * ncols..(row + 1) * ncols].iter().sum())
        .collect();
    println!("Expected sums: {:?}", expected);

    // Run matmul with repacked weights (K/128 format has embedded scales)
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let dst = unsafe { dev.alloc::<f16>(x_rows * y_cols)? };

    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = xs_repacked.data.inner.device_ptr(&stream);
        let segment = VxSegment {
            weights: data_ptr as *const std::ffi::c_void,
            batch_count: y_cols as i32,
        };
        let (y_ptr, _y_guard) = y.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1,
                y_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,
                x_rows as i32,
                ncols as i32,
                x_rows as i32,
                qtype,
                YType::F16 as i32,
                xs_repacked.data.len,
                0, // tile_mode (tiling only; result-invariant)
                OutDType::F16 as i32,
                SumScale::Raw.as_code(),
                stream.cu_stream() as *mut c_void,
            )
        };
        assert_eq!(status, 0, "matmul launcher rejected the call");
    }

    let result = dev.memcpy_dtov(&dst.slice(..))?;
    println!("Q4_K repacked matmul result:");
    for (i, &v) in result.iter().enumerate() {
        let got = f16::to_f32(v);
        let exp = expected[i];
        let err = (got - exp).abs() / exp.abs().max(1.0) * 100.0;
        println!("  Row {}: {} (expected {}, err {:.1}%)", i, got, exp, err);
    }

    // Verify results - check for zeros which indicate bugs
    let mut zero_count = 0;
    for (i, &v) in result.iter().enumerate() {
        assert!(!f16::is_nan(v), "Result {} is NaN", i);
        assert!(!f16::is_infinite(v), "Result {} is Inf", i);
        if v == f16::from_f32(0.0) {
            zero_count += 1;
        }
    }
    println!("Zero count: {} out of {}", zero_count, result.len());

    // For a non-trivial matmul with non-zero inputs, we shouldn't have zeros
    assert!(zero_count == 0, "Got {} zero results", zero_count);

    Ok(())
}

#[test]
pub(super) fn cuda_mm_q4_k_repacked_model_size() -> Result<()> {
    // Test with model-sized matrices (e.g., 3072x3072 hidden layer)
    use crate::Shape;
    use half::f16;

    let dev = CudaDevice::new(0)?;
    let ncols = 3072; // K dimension (hidden dim)
    let x_rows = 3072; // N dimension (output dim)
    let y_cols = 1; // Batch size 1 for decode

    // Create weight data with variance
    let weights: Vec<f32> = (0..ncols * x_rows)
        .map(|v| {
            let row = v / ncols;
            let col = v % ncols;
            // Sinusoidal pattern to ensure variance in all sub-blocks
            let r = row as f32 / x_rows as f32;
            let c = col as f32 / ncols as f32;
            ((r * 7.0 + c * 11.0).sin() + 1.0) * 0.5 // Range [0, 1]
        })
        .collect();

    // Create Y data (all ones for simple sum)
    let y_data: Vec<f16> = (0..ncols * y_cols).map(|_| f16::from_f32(1.0)).collect();
    let y = dev.memcpy_stod(&y_data)?;

    // Create and quantize weights on GPU
    let shape = Shape::from((x_rows, ncols));
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&weights)?,
        dev.clone(),
    ))?;

    // Repack to GEMX format (K/128 with embedded scales)
    let xs_repacked = xs.repack_gemx(&shape)?;
    println!(
        "Repacked Q4_K model-size: {} bytes -> {} bytes",
        xs.data.len, xs_repacked.data.len
    );

    // Run matmul (K/128 format has embedded scales)
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let dst = unsafe { dev.alloc::<f16>(x_rows * y_cols)? };

    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = xs_repacked.data.inner.device_ptr(&stream);
        let segment = VxSegment {
            weights: data_ptr as *const std::ffi::c_void,
            batch_count: y_cols as i32,
        };
        let (y_ptr, _y_guard) = y.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1,
                y_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,
                x_rows as i32,
                ncols as i32,
                x_rows as i32,
                qtype,
                YType::F16 as i32,
                xs_repacked.data.len,
                0, // tile_mode (tiling only; result-invariant)
                OutDType::F16 as i32,
                SumScale::Raw.as_code(),
                stream.cu_stream() as *mut c_void,
            )
        };
        assert_eq!(status, 0, "matmul launcher rejected the call");
    }

    let result = dev.memcpy_dtov(&dst.slice(..))?;

    // Check for NaN/Inf/zeros
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut zero_count = 0;
    for &v in &result {
        if f16::is_nan(v) {
            nan_count += 1;
        }
        if f16::is_infinite(v) {
            inf_count += 1;
        }
        if v == f16::from_f32(0.0) {
            zero_count += 1;
        }
    }

    println!("Model-size Q4_K result: {} rows", result.len());
    println!(
        "  First 10: {:?}",
        result
            .iter()
            .take(10)
            .map(|v| f16::to_f32(*v))
            .collect::<Vec<_>>()
    );
    println!(
        "  Last 10: {:?}",
        result
            .iter()
            .rev()
            .take(10)
            .map(|v| f16::to_f32(*v))
            .collect::<Vec<_>>()
    );
    println!(
        "  NaN: {}, Inf: {}, Zero: {}",
        nan_count, inf_count, zero_count
    );

    assert_eq!(nan_count, 0, "Got NaN values");
    assert_eq!(inf_count, 0, "Got Inf values");
    // For all-ones Y with sinusoidal weights, expect non-zero sums
    assert!(
        zero_count < result.len() / 100,
        "Too many zeros: {}",
        zero_count
    );

    Ok(())
}

/// Test Q6_K matmul with GEMX-repacked weights (K/128 with embedded scales)
/// Q6_K is a 6-bit K-quant format, similar to Q4_K but higher precision.
#[test]
pub(super) fn cuda_mm_q6_k_repacked() -> Result<()> {
    use crate::Shape;
    use half::f16;

    let dev = CudaDevice::new(0)?;
    // Test with MULTIPLE blocks per row (like real tensors)
    // Match GGUF 512x2048 shape exactly
    let ncols = 2048; // 8 blocks per row (same as GGUF 512x2048)
    let x_rows = 512;
    let y_cols = 1;

    // Create VARYING weight data for Q6_K - simple gradient pattern like original
    // Scale down to avoid f16 overflow (max ~65504)
    let weights: Vec<f32> = (0..ncols * x_rows)
        .map(|v| {
            let _row = v / ncols;
            let col = v % ncols;
            // All rows have same pattern - just (col / ncols), sum will be ~ncols/2
            col as f32 / ncols as f32
        })
        .collect();
    println!("Q6_K Input weights: row 0 first 8: {:?}", &weights[..8]);
    println!(
        "Q6_K Input weights: row 0 last 8: {:?}",
        &weights[ncols - 8..ncols]
    );

    // Create Y data (all ones for simple sum)
    let y_data: Vec<f16> = (0..ncols * y_cols).map(|_| f16::from_f32(1.0)).collect();
    let y = dev.memcpy_stod(&y_data)?;

    // Create and quantize weights on GPU using Q6_K
    let shape = Shape::from((x_rows, ncols));
    let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q6_K)?;
    xs.quantize(&CudaStorage::wrap_cuda_slice(
        dev.memcpy_stod(&weights)?,
        dev.clone(),
    ))?;

    println!("Q6_K quantized size: {} bytes", xs.data.len);

    // Repack to K/128 format with embedded scales
    let xs_repacked = xs.repack_gemx(&shape)?;
    println!(
        "Q6_K repacked: {} bytes -> {} bytes",
        xs.data.len, xs_repacked.data.len
    );

    // Compute expected via dequant
    let dequant_storage = xs.dequantize(ncols * x_rows)?;
    let dequant_f32 = dequant_storage.as_cuda_slice::<f32>()?;
    let dequant_vec = dev.memcpy_dtov(&dequant_f32.slice(..))?;
    println!("Dequant row 0 first 8: {:?}", &dequant_vec[..8]);
    println!(
        "Dequant row 0 el 256-264 (block boundary): {:?}",
        &dequant_vec[256..264]
    );
    println!("Dequant row 0 last 8: {:?}", &dequant_vec[ncols - 8..ncols]);

    let mut expected = vec![0.0f32; x_rows];
    for row in 0..x_rows {
        for col in 0..ncols {
            expected[row] += dequant_vec[row * ncols + col] * 1.0;
        }
    }
    println!("Q6_K expected sums: {:?}", expected);

    // Run matmul
    let qtype = dtype_to_qtype(GgmlDType::Q6_K)? as i32;
    let dst = unsafe { dev.alloc::<f16>(x_rows * y_cols)? };

    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = xs_repacked.data.inner.device_ptr(&stream);
        let segment = VxSegment {
            weights: data_ptr as *const std::ffi::c_void,
            batch_count: y_cols as i32,
        };
        let (y_ptr, _y_guard) = y.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1,
                y_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,
                x_rows as i32,
                ncols as i32,
                x_rows as i32,
                qtype,
                YType::F16 as i32,
                xs_repacked.data.len,
                0, // tile_mode (tiling only; result-invariant)
                OutDType::F16 as i32,
                SumScale::Raw.as_code(),
                stream.cu_stream() as *mut c_void,
            )
        };
        assert_eq!(status, 0, "matmul launcher rejected the call");
    }

    let result = dev.memcpy_dtov(&dst.slice(..))?;
    println!("Q6_K repacked matmul result ({} rows):", x_rows);
    let mut max_err = 0.0f32;
    let mut total_err = 0.0f32;
    for (i, &v) in result.iter().enumerate() {
        let got = f16::to_f32(v);
        let exp = expected[i];
        let err = (got - exp).abs() / exp.abs().max(1.0) * 100.0;
        max_err = max_err.max(err);
        total_err += err;
        if i < 4 || i >= x_rows - 4 {
            println!("  Row {}: {} (expected {}, err {:.1}%)", i, got, exp, err);
        } else if i == 4 {
            println!("  ... (rows 4-{} omitted) ...", x_rows - 5);
        }
    }
    let avg_err = total_err / x_rows as f32;

    println!(
        "Q6_K max error: {:.1}%, avg error: {:.2}%",
        max_err, avg_err
    );

    // Q6_K should be accurate
    assert!(max_err < 5.0, "Q6_K error too high: {:.1}%", max_err);

    Ok(())
}

/// Test Q6_K with GGUF-loaded data - compare repacked vs dequant path
#[test]
pub(super) fn cuda_mm_q6_k_gguf_debug() -> Result<()> {
    use crate::Tensor;
    use half::f16;

    // Skip if no model available
    let model_path = match std::env::var("GGUF_MODEL_PATH") {
        Ok(p) => p,
        Err(_) => {
            println!("Skipping - set GGUF_MODEL_PATH to test");
            return Ok(());
        }
    };

    let dev = CudaDevice::new(0)?;
    let device = crate::Device::Cuda(dev.clone());

    // Load model - read file into memory
    let file_bytes = std::fs::read(&model_path)?;
    let mut cursor = std::io::Cursor::new(&file_bytes);
    let content = crate::quantized::gguf_file::Content::read(&mut cursor)?;
    let ct = content;

    // Find a Q6_K tensor
    let q6k_tensor_name = ct
        .tensor_infos
        .iter()
        .find(|(name, info)| {
            info.ggml_dtype == crate::quantized::GgmlDType::Q6_K && name.contains("attn_v")
        })
        .map(|(name, _)| name.clone());

    let tensor_name = match q6k_tensor_name {
        Some(name) => name,
        None => {
            println!("No Q6_K tensor found in model");
            return Ok(());
        }
    };

    println!("Testing Q6_K tensor: {}", tensor_name);

    let tensor_info = ct.tensor_infos.get(&tensor_name).unwrap();
    let qtensor = tensor_info.read_from_mmap(&file_bytes, ct.tensor_data_offset, &device)?;
    let (nrows, ncols) = qtensor.shape().dims2()?;
    println!("Shape: {}x{}", nrows, ncols);

    // Create Y as all ones, shape (ncols,) for vec matmul
    let y_f16_data: Vec<f16> = vec![f16::from_f32(1.0); ncols];
    let y_f16 = Tensor::from_vec(y_f16_data, (1, ncols), &device)?;

    // Method 1: Dequantize to f16 then matmul (using QTensor directly, not QMatMul)
    let w_f16 = qtensor.dequantize_f16(&device)?;
    let result_dequant = y_f16.matmul(&w_f16.t()?)?;
    let result_dequant_vec: Vec<f16> = result_dequant.flatten_all()?.to_vec1()?;

    // Expected - just print what we get
    println!("\nResults via dequant f16 (first 8 rows):");
    for (i, &r) in result_dequant_vec.iter().enumerate().take(8.min(nrows)) {
        println!("Row {}: {:.4}", i, f16::to_f32(r));
    }

    Ok(())
}

/// Test Q4_K matmul with GGUF-loaded weights vs dequantized matmul
/// This verifies that GGUF-loaded data produces correct results
///
/// Run with: GGUF_MODEL_PATH=/path/to/model.gguf cargo test cuda_mm_q4_k_gguf_vs_dequant --ignored
#[test]
#[ignore] // Requires GGUF_MODEL_PATH environment variable
pub(super) fn cuda_mm_q4_k_gguf_vs_dequant() -> Result<()> {
    use crate::quantized::gguf_file;
    #[allow(unused_imports)]
    use half::f16;

    let model_path = std::env::var("GGUF_MODEL_PATH")
        .map_err(|_| crate::Error::Msg("Set GGUF_MODEL_PATH to a Q4_K GGUF model".into()))?;

    let dev = CudaDevice::new(0)?;

    println!("Loading model from: {}", model_path);

    // Open and parse GGUF file
    use memmap2::MmapOptions;
    let file = std::fs::File::open(&model_path)?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| crate::Error::Msg(format!("Failed to mmap file: {}", e)))?
    };
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    let ct = gguf_file::Content::read(&mut cursor)?;

    // Test both Q4_K and Q6_K tensors
    let test_tensors = [
        ("blk.0.attn_q.weight", "Q4_K (attention_wq)"),
        ("blk.0.attn_k.weight", "Q6_K (attention_wk)"),
        ("blk.0.attn_v.weight", "Q6_K (attention_wv)"),
        ("blk.0.attn_output.weight", "attention_wo"),
        ("blk.0.ffn_gate.weight", "feed_forward_w1"),
        ("blk.0.ffn_down.weight", "feed_forward_w2"),
        ("blk.0.ffn_up.weight", "feed_forward_w3"),
    ];

    let device = crate::Device::Cuda(dev.clone());

    for (tensor_name, label) in test_tensors {
        let tensor_info = match ct.tensor_infos.get(tensor_name) {
            Some(info) => info,
            None => {
                println!("Tensor {} not found, skipping", tensor_name);
                continue;
            }
        };

        println!("\n=== Testing {} ({:?}) ===", label, tensor_info.ggml_dtype);
        println!("Shape: {:?}", tensor_info.shape);

        let qtensor = tensor_info.read_from_mmap(&mmap, ct.tensor_data_offset, &device)?;
        let (nrows, ncols) = qtensor.shape().dims2()?;

        // Create random input
        let input_data: Vec<half::bf16> = (0..ncols)
            .map(|i| half::bf16::from_f32((i as f32 / ncols as f32) * 2.0 - 1.0))
            .collect();
        let input = dev.memcpy_stod(&input_data)?;

        // Method 1: Reference via dequantize
        let dequant = qtensor.dequantize(&device)?;
        let dequant_f32 = dequant.to_vec2::<f32>()?;
        let mut expected = vec![0.0f32; nrows];
        for row in 0..nrows {
            for col in 0..ncols {
                expected[row] += dequant_f32[row][col] * half::bf16::to_f32(input_data[col]);
            }
        }

        // Method 2: Quantized matmul with K/128 embedded scales
        // Repack
        let repacked = match &qtensor.storage {
            crate::quantized::QStorage::Cuda(s) => s.repack_gemx(qtensor.shape())?,
            _ => unreachable!(),
        };

        // For Q6K, verify repacking correctness by comparing first block
        if qtensor.dtype() == crate::quantized::GgmlDType::Q6_K {
            let original_data = match &qtensor.storage {
                crate::quantized::QStorage::Cuda(s) => {
                    let data_slice = s.data.inner.slice(..210.min(s.data.len));
                    dev.memcpy_dtov(&data_slice)?
                }
                _ => unreachable!(),
            };
            let repacked_data = {
                let data_slice = repacked.data.inner.slice(..192.min(repacked.data.len));
                dev.memcpy_dtov(&data_slice)?
            };
            println!(
                "Q6K block 0 original ql[0:8]: {:02x?}",
                &original_data[0..8]
            );
            println!(
                "Q6K block 0 repacked ql[0:8]: {:02x?}",
                &repacked_data[0..8]
            );
            println!(
                "Q6K block 0 original qh[0:8]: {:02x?}",
                &original_data[128..136]
            );
            println!(
                "Q6K block 0 repacked qh[0:8]: {:02x?}",
                &repacked_data[128..136]
            );

            // Verify ql match (bytes 0-127)
            let ql_match = original_data[..128] == repacked_data[..128];
            // Verify qh match (bytes 128-191)
            let qh_match = original_data[128..192] == repacked_data[128..192];
            println!("Q6K repack: ql match={}, qh match={}", ql_match, qh_match);
        }

        // Allocate output
        let qtype = dtype_to_qtype(qtensor.dtype())? as i32;
        let dst = unsafe { dev.alloc::<half::bf16>(nrows)? };

        // Run matmul (K/128 format has embedded scales)
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _data_guard) = repacked.data.inner.device_ptr(&stream);
            let segment = VxSegment {
                weights: data_ptr as *const std::ffi::c_void,
                batch_count: 1,
            };
            let (y_ptr, _y_guard) = input.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            let status = unsafe {
                run_quantized_matmul(
                    &segment as *const VxSegment,
                    1,
                    y_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    ncols as i32,
                    nrows as i32,
                    ncols as i32,
                    nrows as i32,
                    qtype,
                    YType::BF16 as i32,
                    repacked.data.len,
                    0, // tile_mode (tiling only; result-invariant)
                    OutDType::BF16 as i32,
                    SumScale::Raw.as_code(),
                    stream.cu_stream() as *mut c_void,
                )
            };
            assert_eq!(status, 0, "matmul launcher rejected the call");
        }

        let result = dev.memcpy_dtov(&dst.slice(..))?;
        let result_f32: Vec<f32> = result.iter().map(|v| half::bf16::to_f32(*v)).collect();

        // Compare
        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f32;
        for i in 0..nrows {
            let err = (result_f32[i] - expected[i]).abs() / expected[i].abs().max(1.0);
            max_err = max_err.max(err);
            sum_err += err;
        }
        let avg_err = sum_err / nrows as f32;

        println!(
            "Result first 5: {:?}",
            &result_f32[..5.min(result_f32.len())]
        );
        println!("Expected first 5: {:?}", &expected[..5.min(expected.len())]);
        println!(
            "Error: max={:.2}%, avg={:.2}%",
            max_err * 100.0,
            avg_err * 100.0
        );

        assert!(
            avg_err < 0.05,
            "{} average error too high: {:.2}%",
            label,
            avg_err * 100.0
        );
    }

    Ok(())
}

/// Test fused QKV (3 matrices concatenated) matmul
/// This verifies that concat_rows_cuda + extract_scales + repack works correctly
#[test]
#[ignore] // Requires GGUF_MODEL_PATH environment variable
pub(super) fn cuda_mm_q4_k_fused_qkv() -> Result<()> {
    use crate::quantized::gguf_file;
    #[allow(unused_imports)]
    use half::f16;

    let model_path = std::env::var("GGUF_MODEL_PATH")
        .map_err(|_| crate::Error::Msg("Set GGUF_MODEL_PATH to a Q4_K GGUF model".into()))?;

    let dev = CudaDevice::new(0)?;

    println!("Loading model from: {}", model_path);

    // Open and parse GGUF file
    use memmap2::MmapOptions;
    let file = std::fs::File::open(&model_path)?;
    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| crate::Error::Msg(format!("Failed to mmap file: {}", e)))?
    };
    let mut cursor = std::io::Cursor::new(&mmap[..]);
    let ct = gguf_file::Content::read(&mut cursor)?;

    // Load Q, K, V matrices from first layer
    let device = crate::Device::Cuda(dev.clone());

    let wq = ct
        .tensor_infos
        .get("blk.0.attn_q.weight")
        .ok_or_else(|| crate::Error::Msg("attn_q not found".into()))?
        .read_from_mmap(&mmap, ct.tensor_data_offset, &device)?;
    let wk = ct
        .tensor_infos
        .get("blk.0.attn_k.weight")
        .ok_or_else(|| crate::Error::Msg("attn_k not found".into()))?
        .read_from_mmap(&mmap, ct.tensor_data_offset, &device)?;
    let wv = ct
        .tensor_infos
        .get("blk.0.attn_v.weight")
        .ok_or_else(|| crate::Error::Msg("attn_v not found".into()))?
        .read_from_mmap(&mmap, ct.tensor_data_offset, &device)?;

    let (nq, ncols) = wq.shape().dims2()?;
    let (nk, _) = wk.shape().dims2()?;
    let (nv, _) = wv.shape().dims2()?;
    println!(
        "Q: {}x{}, K: {}x{}, V: {}x{}",
        nq, ncols, nk, ncols, nv, ncols
    );

    // Create random input
    let input_data: Vec<half::bf16> = (0..ncols)
        .map(|i| half::bf16::from_f32((i as f32 / ncols as f32) * 2.0 - 1.0))
        .collect();
    let input = dev.memcpy_stod(&input_data)?;

    // Method 1: Separate matmuls (reference)
    println!("\n=== Method 1: Separate matmuls ===");
    let dequant_q = wq.dequantize(&device)?.to_vec2::<f32>()?;
    let dequant_k = wk.dequantize(&device)?.to_vec2::<f32>()?;
    let dequant_v = wv.dequantize(&device)?.to_vec2::<f32>()?;

    let mut expected_q = vec![0.0f32; nq];
    let mut expected_k = vec![0.0f32; nk];
    let mut expected_v = vec![0.0f32; nv];

    for row in 0..nq {
        for col in 0..ncols {
            expected_q[row] += dequant_q[row][col] * half::bf16::to_f32(input_data[col]);
        }
    }
    for row in 0..nk {
        for col in 0..ncols {
            expected_k[row] += dequant_k[row][col] * half::bf16::to_f32(input_data[col]);
        }
    }
    for row in 0..nv {
        for col in 0..ncols {
            expected_v[row] += dequant_v[row][col] * half::bf16::to_f32(input_data[col]);
        }
    }
    println!("Expected Q first 5: {:?}", &expected_q[..5]);
    println!("Expected K first 5: {:?}", &expected_k[..5]);
    println!("Expected V first 5: {:?}", &expected_v[..5]);

    // Method 2: Fused QKV
    println!("\n=== Method 2: Fused QKV ===");
    let fused = crate::quantized::QTensor::concat_rows_cuda(&[&wq, &wk, &wv])?;
    let (nfused, ncols2) = fused.shape().dims2()?;
    println!("Fused shape: {}x{}", nfused, ncols2);
    assert_eq!(nfused, nq + nk + nv);
    assert_eq!(ncols2, ncols);

    // Repack to K/128 format with embedded scales
    let repacked = match &fused.storage {
        crate::quantized::QStorage::Cuda(s) => s.repack_gemx(fused.shape())?,
        _ => unreachable!(),
    };

    // Allocate output
    let qtype = dtype_to_qtype(GgmlDType::Q4_K)? as i32;
    let dst = unsafe { dev.alloc::<half::bf16>(nfused)? };

    // Run fused matmul (K/128 format has embedded scales)
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = repacked.data.inner.device_ptr(&stream);
        let segment = VxSegment {
            weights: data_ptr as *const std::ffi::c_void,
            batch_count: 1,
        };
        let (y_ptr, _y_guard) = input.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1,
                y_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,
                nfused as i32,
                ncols as i32,
                nfused as i32,
                qtype,
                YType::BF16 as i32,
                repacked.data.len,
                0, // tile_mode (tiling only; result-invariant)
                OutDType::BF16 as i32,
                SumScale::Raw.as_code(),
                stream.cu_stream() as *mut c_void,
            )
        };
        assert_eq!(status, 0, "matmul launcher rejected the call");
    }

    let result = dev.memcpy_dtov(&dst.slice(..))?;
    let result_f32: Vec<f32> = result.iter().map(|v| half::bf16::to_f32(*v)).collect();

    // Split result into Q, K, V
    let result_q = &result_f32[0..nq];
    let result_k = &result_f32[nq..nq + nk];
    let result_v = &result_f32[nq + nk..nq + nk + nv];

    println!("Result Q first 5: {:?}", &result_q[..5]);
    println!("Result K first 5: {:?}", &result_k[..5]);
    println!("Result V first 5: {:?}", &result_v[..5]);

    // Compare
    let mut max_err_q = 0.0f32;
    let mut sum_err_q = 0.0f32;
    for i in 0..nq {
        let err = (result_q[i] - expected_q[i]).abs() / expected_q[i].abs().max(1.0);
        max_err_q = max_err_q.max(err);
        sum_err_q += err;
    }

    let mut max_err_k = 0.0f32;
    let mut sum_err_k = 0.0f32;
    for i in 0..nk {
        let err = (result_k[i] - expected_k[i]).abs() / expected_k[i].abs().max(1.0);
        max_err_k = max_err_k.max(err);
        sum_err_k += err;
    }

    let mut max_err_v = 0.0f32;
    let mut sum_err_v = 0.0f32;
    for i in 0..nv {
        let err = (result_v[i] - expected_v[i]).abs() / expected_v[i].abs().max(1.0);
        max_err_v = max_err_v.max(err);
        sum_err_v += err;
    }

    println!(
        "\nQ error: max={:.2}%, avg={:.2}%",
        max_err_q * 100.0,
        (sum_err_q / nq as f32) * 100.0
    );
    println!(
        "K error: max={:.2}%, avg={:.2}%",
        max_err_k * 100.0,
        (sum_err_k / nk as f32) * 100.0
    );
    println!(
        "V error: max={:.2}%, avg={:.2}%",
        max_err_v * 100.0,
        (sum_err_v / nv as f32) * 100.0
    );

    // Should have reasonable accuracy
    let avg_err_q = sum_err_q / nq as f32;
    let avg_err_k = sum_err_k / nk as f32;
    let avg_err_v = sum_err_v / nv as f32;

    assert!(
        avg_err_q < 0.05,
        "Q average error too high: {:.2}%",
        avg_err_q * 100.0
    );
    assert!(
        avg_err_k < 0.05,
        "K average error too high: {:.2}%",
        avg_err_k * 100.0
    );
    assert!(
        avg_err_v < 0.05,
        "V average error too high: {:.2}%",
        avg_err_v * 100.0
    );

    Ok(())
}

/// Test that grouped_matmul_gemx produces the same results as matmul_gemx
/// for a single expert (1-expert grouped = regular matmul).
#[test]
pub(super) fn grouped_matmul_matches_direct() -> Result<()> {
    use crate::Shape;
    use half::bf16;

    let dev = CudaDevice::new(0)?;
    // Dimensions matching Qwen3-30B expert gate_proj: [768, 2048]
    let nrows = 768; // N = intermediate_dim (output)
    let ncols = 2048; // K = hidden_dim (input)
    let num_experts = 3;
    let expert_batches = &[5, 3, 7]; // tokens per expert

    // Create varying weight data for each expert
    let mut rng = rand::rng();
    let mut expert_storages = Vec::new();
    let mut weight_ptrs = Vec::new();
    let shape = Shape::from((nrows, ncols));

    for _ in 0..num_experts {
        let weights: Vec<f32> = (0..nrows * ncols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q4_K)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&weights)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        weight_ptrs.push(xs.data_ptr());
        expert_storages.push(xs);
    }

    // Create BF16 activations (total_batch = sum of expert_batches)
    let total_batch: usize = expert_batches.iter().sum();
    let act_data: Vec<f32> = (0..total_batch * ncols)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let act_bf16: Vec<bf16> = act_data.iter().map(|&v| bf16::from_f32(v)).collect();
    let act_gpu = dev.memcpy_stod(&act_bf16)?;
    let act_storage = CudaStorage::wrap_cuda_slice(act_gpu.clone(), dev.clone());
    let act_layout = crate::Layout::contiguous(Shape::from(vec![total_batch, ncols]));

    // Build expert_offsets prefix sum
    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }

    // ── Reference: per-expert matmul_gemx ──
    let mut ref_results: Vec<Vec<bf16>> = Vec::new();
    for e in 0..num_experts {
        let start = expert_offsets[e] as usize;
        let end = expert_offsets[e + 1] as usize;
        let batch = end - start;

        // Slice activations for this expert
        let expert_act_bf16 = &act_bf16[start * ncols..end * ncols];
        let expert_act_gpu = dev.memcpy_stod(expert_act_bf16)?;
        let expert_act_storage = CudaStorage::wrap_cuda_slice(expert_act_gpu, dev.clone());
        let expert_act_layout = crate::Layout::contiguous(Shape::from(vec![batch, ncols]));

        let (ref_storage, _) =
            expert_storages[e].matmul_gemx(&shape, &expert_act_storage, &expert_act_layout)?;
        let ref_bf16 = ref_storage.as_cuda_slice::<bf16>()?;
        let ref_result: Vec<bf16> = dev.memcpy_dtov(&ref_bf16.slice(..))?;
        ref_results.push(ref_result);
    }

    // ── Test: grouped_matmul_gemx ──
    let grouped_result = grouped_matmul_gemx(
        &weight_ptrs,
        GgmlDType::Q4_K,
        nrows,
        ncols,
        &act_storage,
        &act_layout,
        &expert_offsets,
        &dev,
    )?;

    // Read grouped result
    let grouped_storage = grouped_result.storage_and_layout().0;
    let grouped_cuda = match &*grouped_storage {
        crate::Storage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };
    let grouped_bf16 = grouped_cuda.as_cuda_slice::<bf16>()?;
    let grouped_vals: Vec<bf16> = dev.memcpy_dtov(&grouped_bf16.slice(..))?;

    // Compare per-expert
    let mut total_mismatches = 0;
    for e in 0..num_experts {
        let start = expert_offsets[e] as usize * nrows;
        let end = expert_offsets[e + 1] as usize * nrows;
        let ref_vals = &ref_results[e];
        let grp_vals = &grouped_vals[start..end];

        assert_eq!(
            ref_vals.len(),
            grp_vals.len(),
            "expert {} length mismatch",
            e
        );

        let mut err_count = 0;
        for i in 0..ref_vals.len() {
            let r = bf16::to_f32(ref_vals[i]);
            let g = bf16::to_f32(grp_vals[i]);
            let err = (r - g).abs();
            let rel = err / r.abs().max(1e-6);
            if rel > 0.01 {
                if err_count < 3 {
                    println!(
                        "  expert[{}] MISMATCH at [{}]: ref={:.6} grouped={:.6} rel={:.4}",
                        e, i, r, g, rel
                    );
                }
                err_count += 1;
            }
        }
        println!(
            "Expert[{}]: batch={}, mismatches={}/{}",
            e,
            expert_batches[e],
            err_count,
            ref_vals.len()
        );
        total_mismatches += err_count;
    }

    assert_eq!(
        total_mismatches, 0,
        "grouped_matmul_gemx produced different results for multi-expert case"
    );
    Ok(())
}

/// Test grouped_matmul_gemx with down-projection dimensions (nrows > ncols).
///
/// **What is at model scale here is the expert *count* and the ragged batches**,
/// not the per-expert matrix. 109 experts with batches varying 1–23 is what
/// exercises the grouped kernel's offset table and its pointer indirection; the
/// weights themselves only have to be a legal Q4_K shape with `nrows > ncols`.
/// They were `[2048, 768]`, making the fixture 171M host-generated randoms pushed
/// through a quantizer that — on CUDA — runs on the *host*, downloading and
/// re-uploading every expert: ten seconds to assert the output holds no NaN.
///
/// What is left after shrinking them is ~20ms of fixed per-quantize overhead,
/// once per expert, and shrinking further does not touch it — `[512, 256]` and
/// `[384, 256]` measure the same. So the floor here *is* the expert count, which
/// is the one thing this test may not trade away.
#[test]
pub(super) fn grouped_matmul_matches_direct_down() -> Result<()> {
    use crate::Shape;
    use half::bf16;

    let dev = CudaDevice::new(0)?;
    // Down projection: `nrows > ncols`, which is the shape relation under test.
    // Only `ncols` is constrained by the format — it is the row length, so it
    // carries Q4_K's 256-element block — and 256 is its smallest legal value.
    let nrows = 384;
    let ncols = 256;
    let num_experts = 109;
    // Vary batch sizes 1-23 to match real workload
    let expert_batches: Vec<usize> = (0..num_experts).map(|i| 1 + (i * 7 + 3) % 23).collect();
    println!("Testing: {} experts", num_experts);

    let mut expert_storages = Vec::new();
    let mut weight_ptrs = Vec::new();
    let shape = Shape::from((nrows, ncols));

    // Deterministic, and distinct per expert. This was `rng.random_range` per
    // element, which is both a large share of the fixture's cost in a debug build
    // and the reason a failure here could not be reproduced — a regression test
    // whose input changes every run tells you that it broke without telling you
    // what on.
    let element = |expert: usize, i: usize| -> f32 {
        let h = (expert as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add((i as u32).wrapping_mul(40_503));
        (h >> 20) as f32 / 2048.0 - 1.0
    };

    for expert in 0..num_experts {
        let weights: Vec<f32> = (0..nrows * ncols).map(|i| element(expert, i)).collect();
        let mut xs = QCudaStorage::zeros(&dev, ncols * nrows, GgmlDType::Q4_K)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(
            dev.memcpy_stod(&weights)?,
            dev.clone(),
        ))?;
        let xs = xs.repack_gemx(&shape)?;
        weight_ptrs.push(xs.data_ptr());
        expert_storages.push(xs);
    }

    let total_batch: usize = expert_batches.iter().sum();
    let act_bf16: Vec<bf16> = (0..total_batch * ncols)
        .map(|i| bf16::from_f32(element(num_experts + 1, i)))
        .collect();
    let act_gpu = dev.memcpy_stod(&act_bf16)?;
    let act_storage = CudaStorage::wrap_cuda_slice(act_gpu.clone(), dev.clone());
    let act_layout = crate::Layout::contiguous(Shape::from(vec![total_batch, ncols]));

    let mut expert_offsets: Vec<i32> = vec![0];
    for &b in &expert_batches {
        expert_offsets.push(expert_offsets.last().unwrap() + b as i32);
    }

    // At 109 experts, full per-expert reference comparison is slow.
    // Just verify no NaN/Inf in grouped output.

    // Test: grouped_matmul_gemx
    let grouped_result = grouped_matmul_gemx(
        &weight_ptrs,
        GgmlDType::Q4_K,
        nrows,
        ncols,
        &act_storage,
        &act_layout,
        &expert_offsets,
        &dev,
    )?;

    let grouped_storage = grouped_result.storage_and_layout().0;
    let grouped_cuda = match &*grouped_storage {
        crate::Storage::Cuda(s) => s,
        _ => panic!("expected CUDA storage"),
    };
    let grouped_bf16 = grouped_cuda.as_cuda_slice::<bf16>()?;
    let grouped_vals: Vec<bf16> = dev.memcpy_dtov(&grouped_bf16.slice(..))?;

    // Check grouped for NaN
    let nan_count = grouped_vals.iter().filter(|v| bf16::is_nan(**v)).count();
    let inf_count = grouped_vals
        .iter()
        .filter(|v| bf16::is_infinite(**v))
        .count();
    println!(
        "Grouped: {} elements, {} NaN, {} Inf",
        grouped_vals.len(),
        nan_count,
        inf_count
    );
    assert_eq!(nan_count, 0, "Grouped matmul produced NaN at model scale");
    assert_eq!(inf_count, 0, "Grouped matmul produced Inf at model scale");
    Ok(())
}
