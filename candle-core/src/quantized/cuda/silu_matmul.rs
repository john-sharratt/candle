//! The fused-activation int8 dense matmul: `silu(proj[:, 0..K]) · Wᵀ`, with the
//! activation quantized to q8a128 by the kernel's own tile loader.
//!
//! The matmul a hyper-connection's `up` projection runs. Its activation is the SiLU of the
//! gate columns of the `down` projection's output; as a separate producer that was one more
//! launch per pre-mix — at decode width a ~9 µs host issue for ~1.5 µs of GPU work, about a
//! fifth of the whole pre-mix (`gr_hyper_bench ko`). Here the GEMM's loader reads the F32
//! rows through their stride, applies `fast_exp::silu` and quantizes each 128-wide tile
//! straight into shared memory with the one q8a128 quantization every producer uses, so the
//! operand is the bytes `gr_silu_q8` would have written and the result is that pair's, bit
//! for bit.

use std::ffi::c_void;

use candle_kernels::quantized::run_dense_int8_silu_q8ko_f32;
use cudarc::driver::DevicePtr;

use super::super::int8_matmul_mode::q8a128_dense_use_mode2;
use super::super::{GgmlDType, SumScale};
use super::{
    cached_sm_count, check_matmul_status, resolve_out, tensor_from_owned_out, CudaDevice, Result,
};
use crate::cuda_backend::CudaStorageSlice;
use crate::{DType, LiveTensor, Shape, Storage};

/// `[M, N]` F32 `= silu(proj[:, 0..cols]) · Wᵀ` for a Q8_KO weight `[N, cols]` at
/// `weight_ptr`. `proj` is `[M, width]` F32 with unit column stride; its rows may be wider
/// than `cols` (the gate columns of a stacked projection), and are read in place.
pub(crate) fn q8a128_dense_matmul_silu<'w>(
    proj: &LiveTensor<'w>,
    cols: usize,
    weight_ptr: u64,
    weight_dtype: GgmlDType,
    nrows: usize,
    sum_scale: SumScale,
    device: &CudaDevice,
) -> Result<LiveTensor<'w>> {
    if weight_dtype != GgmlDType::Q8_KO {
        crate::bail!(
            "fused-silu matmul: built for the Q8_KO hyper-connection weight, got {weight_dtype:?}"
        );
    }
    let (m, width) = proj.dims2()?;
    if !cols.is_multiple_of(128) || cols > width || !nrows.is_multiple_of(32) {
        crate::bail!(
            "fused-silu matmul: K={cols} must be whole 128-tiles within a {width}-wide row, and \
             N={nrows} whole 32-row tiles"
        );
    }
    let (storage, layout) = proj.storage_and_layout();
    let stride = layout.stride();
    // A single row has no second row to step to, whatever stride its view records,
    // so its stride is the tile-aligned `cols` rather than a width that need not be.
    let row_stride = if m == 1 { cols } else { stride[0] };
    if stride[1] != 1 || !row_stride.is_multiple_of(4) || !layout.start_offset().is_multiple_of(4) {
        crate::bail!(
            "fused-silu matmul: proj {:?} with stride {stride:?} at offset {} — the loader \
             reads unit-stride 16-byte-aligned rows",
            proj.dims(),
            layout.start_offset()
        );
    }
    let Storage::Cuda(cs) = &*storage else {
        crate::bail!("fused-silu matmul: proj must be on CUDA");
    };
    let CudaStorageSlice::F32(slice) = &cs.slice else {
        crate::bail!("fused-silu matmul: proj must be F32");
    };
    let (dst_ptr, owned, backing) = resolve_out(DType::F32, cs.backing, device, m * nrows)?;
    let mode2 = q8a128_dense_use_mode2(m, nrows, cols, cached_sm_count(device));
    let stream = device.cuda_stream();
    let view = slice.slice(layout.start_offset()..);
    let (proj_ptr, _guard) = view.device_ptr(&stream);
    let status = unsafe {
        run_dense_int8_silu_q8ko_f32(
            weight_ptr as *const c_void,
            proj_ptr as *const f32,
            row_stride as i32,
            dst_ptr as *mut f32,
            cols as i32,
            nrows as i32,
            m as i32,
            sum_scale.as_code(),
            i32::from(mode2),
            stream.cu_stream() as *mut c_void,
        )
    };
    check_matmul_status(status, "q8a128 fused-silu matmul")?;
    drop(_guard);
    drop(storage);
    tensor_from_owned_out(owned, dst_ptr, backing, Shape::from((m, nrows)), device)
}
