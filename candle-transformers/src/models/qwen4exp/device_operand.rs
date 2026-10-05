//! An F32 tensor resolved to the device pointer a fused kernel launch takes.
//!
//! Shared by the Gated Residual's kernels (`hyper::cuda_fused`) and the PLE
//! block's (`ple_fused`), which hand their operands to the launchers the same
//! way.
//!
//! # Operands arrive as views, and that decides the vector width
//!
//! A tensor sliced out of a wave's own buffer is dense with a storage start
//! thousands of elements in. The pointer handed over is advanced by that offset,
//! and the offset is also what decides whether a `float4` load is legal: a dense
//! tensor starting at element 1 is 4-byte aligned however well-behaved its row
//! width is. [`Operand::vec_ok`] carries that per operand, and a launch
//! vectorises only when every operand agrees.

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaStream, DevicePtr};
use candle::{DType, LiveTensor, Result};

use crate::models::operand_guard::expect_dtype;

/// A dense F32 operand resolved to a device pointer.
pub struct Operand {
    pub ptr: u64,
    /// Whether this operand's base is 16-byte aligned, i.e. its start offset
    /// is a whole number of `float4`s. Allocations are far better aligned than
    /// that; a view's offset is what can break it.
    pub vec_ok: bool,
}

/// Resolve `t` and hand the pointer to `f`, holding the storage across it.
///
/// Contiguity is required (the kernels index `row · d + j` with no stride
/// metadata) but a nonzero start offset is not: it is added to the pointer
/// here and folded into `vec_ok`.
pub fn with_operand<R>(
    t: &LiveTensor<'_>,
    what: &str,
    f: impl FnOnce(Operand, &CudaStream) -> R,
) -> Result<R> {
    if !t.is_contiguous() {
        candle::bail!(
            "{what}: kernel operand has layout {:?} stride {:?}, which is not dense — these \
             kernels index by row and carry no stride argument",
            t.dims(),
            t.stride()
        );
    }
    with_ptr(t, what, f)
}

/// Resolve `t` at its start offset with no layout check, for an operand whose
/// kernel takes its stride as an argument. The caller validates the layout.
pub fn with_ptr<R>(
    t: &LiveTensor<'_>,
    what: &str,
    f: impl FnOnce(Operand, &CudaStream) -> R,
) -> Result<R> {
    expect_dtype(t, DType::F32, what)?;
    let (storage, layout) = t.storage_and_layout();
    let candle::Storage::Cuda(cs) = &*storage else {
        candle::bail!("{what}: expected CUDA storage");
    };
    let stream = cs.device().cuda_stream();
    let start = layout.start_offset();
    let slice = cs.as_cuda_slice::<f32>()?.slice(start..);
    let (ptr, _guard) = slice.device_ptr(&stream);
    Ok(f(
        Operand {
            ptr,
            vec_ok: start % 4 == 0,
        },
        &stream,
    ))
}
