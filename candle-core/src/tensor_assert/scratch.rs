//! The dequant staging buffer for quantized asserts.
//!
//! A quantized tensor's values only exist once dequantized, and the dequant
//! kernels write to memory — so unlike the dense path, a quantized assert needs
//! somewhere to put them. That somewhere is one grow-to-fit buffer per device,
//! retained for the life of the process, so the cost is a single allocation
//! sized to the largest tensor ever asserted rather than one per call.
//!
//! Growth is logged, because growth is the one moment this module allocates and
//! a growth observed inside a wave means a quantized assert has been placed
//! somewhere it does not belong.

use crate::cuda_backend::{CudaDevice, DeviceId};
use crate::quantized::cuda::dtype_to_qtype;
use crate::quantized::GgmlDType;
use crate::{Result, Shape};
use candle_kernels::simple::quantized::{run_dequantize_block, run_dequantize_ko};
use cudarc::driver::{CudaSlice, DevicePtr};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// `out_dtype` code for F32 in `run_dequantize_block`'s unified ordering.
const DEQUANT_OUT_F32: i32 = 0;

/// `QTYPE_Q8A128V` from `quantized/block_compact.cuh`, where a `static_assert`
/// pins it. The int8 activation operand the MoE's grouped GEMMs consume, and
/// the one quantized value on this path that is not a weight — which is why it
/// needs a way to be examined that does not go through `QTensor`.
pub const QTYPE_Q8A128V: i32 = 36;

/// Dequantize `elem_count` elements of a raw quantized activation buffer into
/// `dst` as F32.
pub fn dequantize_flat_into(src: u64, dst: u64, elem_count: usize, qtype: i32) {
    unsafe {
        crate::set_kernel_breadcrumb("run_dequantize_block", file!(), line!());
        run_dequantize_block(
            src as *const std::ffi::c_void,
            dst as *mut std::ffi::c_void,
            elem_count as i32,
            qtype,
            DEQUANT_OUT_F32,
        );
    }
}

struct Scratch {
    buf: CudaSlice<f32>,
}

type Registry = Mutex<HashMap<DeviceId, Scratch>>;

fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Run `f` with a device pointer to at least `elems` f32 of scratch.
pub fn with_f32_scratch<R>(
    dev: &CudaDevice,
    elems: usize,
    f: impl FnOnce(u64) -> Result<R>,
) -> Result<R> {
    let mut reg = registry()
        .lock()
        .map_err(|_| crate::Error::Msg("tensor_assert: scratch registry poisoned".to_string()))?;
    let scratch = match reg.entry(dev.id()) {
        std::collections::hash_map::Entry::Occupied(o) => {
            let s = o.into_mut();
            if s.buf.len() < elems {
                tracing::info!(
                    target: "candle_core::tensor_assert",
                    from = s.buf.len(), to = elems,
                    "tensor_assert: growing the quantized-assert scratch"
                );
                s.buf = alloc(dev, elems)?;
            }
            s
        }
        std::collections::hash_map::Entry::Vacant(v) => v.insert(Scratch {
            buf: alloc(dev, elems)?,
        }),
    };
    let stream = dev.cuda_stream();
    let (ptr, _g) = scratch.buf.device_ptr(&stream);
    f(ptr)
}

fn alloc(dev: &CudaDevice, elems: usize) -> Result<CudaSlice<f32>> {
    unsafe { dev.alloc::<f32>(elems) }
        .map_err(|e| crate::Error::Msg(format!("tensor_assert: scratch alloc {elems} f32: {e}")))
}

/// Dequantize a whole quantized tensor into `dst` as F32.
///
/// Two families, and the split is not cosmetic: KO formats are lane-major over
/// `(nrows, ncols)` chunks and have their own shape-aware entry point, while
/// every other format is a flat array of blocks that `run_dequantize_block`
/// walks by element count. Reading a KO tensor through the flat path would
/// misinterpret the layout and report values the matmul never sees.
pub fn dequantize_into(src: u64, dst: u64, shape: &Shape, dtype: GgmlDType) -> Result<()> {
    let qtype = dtype_to_qtype(dtype)? as i32;
    let elem_count = shape.elem_count();
    let dims = shape.dims();
    let ncols = *dims.last().unwrap_or(&elem_count);
    let nrows = elem_count.checked_div(ncols).unwrap_or(0);

    if is_ko(dtype) {
        if !nrows.is_multiple_of(8) || !ncols.is_multiple_of(128) {
            crate::bail!(
                "{dtype:?} dequant needs nrows % 8 == 0 and ncols % 128 == 0, got \
                 [{nrows}, {ncols}]"
            );
        }
        unsafe {
            crate::set_kernel_breadcrumb("run_dequantize_ko", file!(), line!());
            run_dequantize_ko(
                src as *const std::ffi::c_void,
                dst as *mut f32,
                nrows as i32,
                ncols as i32,
                qtype,
            );
        }
        return Ok(());
    }

    unsafe {
        crate::set_kernel_breadcrumb("run_dequantize_block", file!(), line!());
        run_dequantize_block(
            src as *const std::ffi::c_void,
            dst as *mut std::ffi::c_void,
            elem_count as i32,
            qtype,
            DEQUANT_OUT_F32,
        );
    }
    Ok(())
}

/// Whether `dtype` is one of the lane-major KO weight formats, which dequantize
/// through `run_dequantize_ko` rather than the flat block path.
///
/// Exhaustive on purpose: a new KO format must state which family it belongs to
/// here rather than silently inheriting the flat path and reading garbage.
fn is_ko(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q2_KO
            | GgmlDType::Q4_KO
            | GgmlDType::Q5_KO
            | GgmlDType::Q6_KO
            | GgmlDType::Q8_KO
    )
}

#[cfg(test)]
mod tests {
    use super::is_ko;
    use crate::quantized::GgmlDType;

    #[test]
    fn the_ko_family_is_exactly_the_lane_major_weight_formats() {
        for dt in [
            GgmlDType::Q2_KO,
            GgmlDType::Q4_KO,
            GgmlDType::Q5_KO,
            GgmlDType::Q6_KO,
            GgmlDType::Q8_KO,
        ] {
            assert!(is_ko(dt), "{dt:?} must take the KO dequant path");
        }
        for dt in [
            GgmlDType::Q4_K,
            GgmlDType::Q6_K,
            GgmlDType::Q8_0,
            GgmlDType::Q4_KS,
            GgmlDType::MXFP4,
        ] {
            assert!(!is_ko(dt), "{dt:?} must take the flat block dequant path");
        }
    }
}
