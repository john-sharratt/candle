//! A gated-shared-expert MoE layer's output, added to the residual in one launch.
//!
//! `x += narrow(routed + shared · sigmoid(gate))` — the gate's sigmoid, the
//! broadcast multiply, the add of the routed sum, the narrowing to the stream's
//! type and the residual add, which ran as four launches over `[n, d]` buffers
//! per MoE layer per wave. Bit-identical to them (`simple/moe_shared_residual.cu`
//! states the rounding it mirrors); the test below holds it to that against the
//! eager chain, which is [`SharedExpertParts::gated`] and `Tensor::add_mut`.

use std::ffi::c_void;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaStream, DevicePtr};
use candle::{DType, LiveTensor, Result, Storage, Tensor};
use candle_kernels::simple::moe_scatter::MoeScatterDType;
use candle_kernels::simple::moe_shared_residual::{
    run_moe_shared_residual, MOE_SHARED_RESIDUAL_LAUNCHED,
};
use half::{bf16, f16};

use super::quantized_moe::SharedExpertParts;

/// The kernel's dtype code for `dtype`.
fn dtype_code(dtype: DType, what: &str) -> Result<i32> {
    Ok(match dtype {
        DType::F32 => MoeScatterDType::F32,
        DType::F16 => MoeScatterDType::F16,
        DType::BF16 => MoeScatterDType::BF16,
        other => candle::bail!("moe shared residual: {what} is {other:?}, not a float type"),
    } as i32)
}

/// Resolve `t` at its start offset and hand the device pointer to `f`, holding
/// the storage across it. Layout is the caller's to validate.
fn with_ptr<R>(t: &LiveTensor<'_>, what: &str, f: impl FnOnce(u64, &CudaStream) -> R) -> Result<R> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(cs) = &*storage else {
        candle::bail!("moe shared residual: {what} is not on a CUDA device");
    };
    let stream = cs.device().cuda_stream();
    let start = layout.start_offset();
    macro_rules! resolve {
        ($t:ty) => {{
            let slice = cs.as_cuda_slice::<$t>()?.slice(start..);
            let (ptr, _guard) = slice.device_ptr(&stream);
            Ok(f(ptr, &stream))
        }};
    }
    match t.dtype() {
        DType::F32 => resolve!(f32),
        DType::F16 => resolve!(f16),
        DType::BF16 => resolve!(bf16),
        other => candle::bail!("moe shared residual: {what} is {other:?}, not a float type"),
    }
}

/// `x += narrow(routed + shared.y · sigmoid(shared.gate))`, in place.
///
/// `x` is the residual stream, `[.., d]`, dense at any start offset. `routed` and
/// `shared.y` are the layer's two halves at the FFN's working type, dense, one
/// `d`-row per row of `x`; `shared.gate` is `[n, 1]` read through its row stride.
/// `x` is taken `&mut` for `Tensor::add_mut`'s reason: the caller holds the
/// residual it is updating and nothing reads it expecting the old value.
pub fn add_moe_residual(
    x: &mut Tensor,
    routed: &LiveTensor<'_>,
    shared: &SharedExpertParts<'_>,
) -> Result<()> {
    let d = x.dim(x.rank() - 1)?;
    let n = x.elem_count() / d;
    let parts_dtype = routed.dtype();
    for (t, what) in [(routed, "routed"), (&shared.y, "shared expert output")] {
        if !t.is_contiguous() || t.elem_count() != n * d || t.dtype() != parts_dtype {
            candle::bail!(
                "moe shared residual: {what} is {:?} {:?} stride {:?}, expected dense {parts_dtype:?} \
                 with {} elements for the residual's [{n}, {d}]",
                t.dtype(),
                t.dims(),
                t.stride(),
                n * d
            );
        }
    }
    if !x.is_contiguous() {
        candle::bail!(
            "moe shared residual: the residual is {:?} stride {:?}, not dense",
            x.dims(),
            x.stride()
        );
    }
    let gate = &shared.gate;
    if gate.dims() != [n, 1] || gate.dtype() != parts_dtype {
        candle::bail!(
            "moe shared residual: gate is {:?} {:?}, expected {parts_dtype:?} [{n}, 1]",
            gate.dtype(),
            gate.dims()
        );
    }
    let gate_stride = gate.stride()[0];
    let codes = (
        dtype_code(parts_dtype, "the FFN output")?,
        dtype_code(x.dtype(), "the residual")?,
    );
    let status = with_ptr(x, "residual", |xp, stream| {
        with_ptr(routed, "routed", |rp, _| {
            with_ptr(&shared.y, "shared expert output", |sp, _| {
                with_ptr(gate, "gate", |gp, _| {
                    candle::set_kernel_breadcrumb("run_moe_shared_residual", file!(), line!());
                    unsafe {
                        run_moe_shared_residual(
                            codes.0,
                            codes.1,
                            xp as *mut c_void,
                            rp as *const c_void,
                            sp as *const c_void,
                            gp as *const c_void,
                            n as i32,
                            d as i32,
                            gate_stride as i32,
                            stream.cu_stream() as *mut c_void,
                        )
                    }
                })
            })
        })
    })????;
    if status != MOE_SHARED_RESIDUAL_LAUNCHED {
        candle::bail!(
            "moe shared residual: no kernel for {parts_dtype:?} parts into a {:?} residual",
            x.dtype()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gpu_test_lock::gpu_serial;
    use candle::Device;

    fn lcg(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5) * scale
            })
            .collect();
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    /// The residual's raw bits, for an exact comparison at any width.
    fn bits(t: &Tensor) -> Vec<u32> {
        let flat = t.flatten_all().unwrap();
        match t.dtype() {
            DType::F32 => flat
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            DType::BF16 => flat
                .to_vec1::<bf16>()
                .unwrap()
                .iter()
                .map(|v| u32::from(v.to_bits()))
                .collect(),
            DType::F16 => flat
                .to_vec1::<f16>()
                .unwrap()
                .iter()
                .map(|v| u32::from(v.to_bits()))
                .collect(),
            other => panic!("{other:?}"),
        }
    }

    /// The fused update is the four launches it replaces, bit for bit: the
    /// gate's sigmoid, the broadcast multiply, the add, the narrowing and the
    /// residual add — at every (parts, residual) pair a layer produces, over a
    /// row width that is not a whole number of blocks and a gate read as the
    /// first column of a padded projection.
    #[test]
    fn the_fused_residual_is_the_eager_chain_bit_for_bit() {
        let _gpu = gpu_serial();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let pairs = [
            (DType::F32, DType::F32),
            (DType::BF16, DType::BF16),
            (DType::BF16, DType::F16),
        ];
        for (parts, stream) in pairs {
            for (n, d) in [(1usize, 2048usize), (5, 2048), (40, 300)] {
                let at = |t: Tensor, dt| t.to_dtype(dt).unwrap();
                let x = at(lcg(&[n, d], 60 + n as u64, 4.0, &dev), stream);
                let routed = at(lcg(&[n, d], 61, 1.0, &dev), parts);
                let y = at(lcg(&[n, d], 62, 1.0, &dev), parts);
                let gate_full = at(lcg(&[n, 16], 63, 8.0, &dev), parts);
                let shared = SharedExpertParts {
                    y,
                    gate: gate_full.narrow(1, 0, 1).unwrap(),
                };

                let mut got = x.copy().unwrap();
                add_moe_residual(&mut got, &routed, &shared).unwrap();

                let mut want = x.copy().unwrap();
                let h = (&routed + &shared.gated().unwrap()).unwrap();
                want.add_mut(&h.to_dtype(stream).unwrap()).unwrap();

                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{parts:?} parts into {stream:?}, {n}×{d}"
                );
            }
        }
    }

    /// A pair with no instantiation is refused, never launched with a guess at
    /// the types — the residual is left exactly as it was.
    #[test]
    fn an_uninstantiated_dtype_pair_is_refused() {
        let _gpu = gpu_serial();
        let Ok(dev) = Device::new_cuda(0) else { return };
        let x = lcg(&[2, 256], 70, 1.0, &dev);
        let parts = |s| lcg(&[2, 256], s, 1.0, &dev).to_dtype(DType::BF16).unwrap();
        let gate = lcg(&[2, 16], 73, 1.0, &dev).to_dtype(DType::BF16).unwrap();
        let shared = SharedExpertParts {
            y: parts(72),
            gate: gate.narrow(1, 0, 1).unwrap(),
        };
        let mut got = x.copy().unwrap();
        assert!(add_moe_residual(&mut got, &parts(71), &shared).is_err());
        assert_eq!(bits(&got), bits(&x));
    }
}
