//! The device half of the Gated Residual: three kernels replacing the eager
//! chain in the parent module.
//!
//! The parent's eager path is the reference — it is what runs on the CPU
//! oracle, and the parity tests assert these against it. That is the same
//! arrangement `latent_moe/hyper.rs` uses for mHC, and the reason the eager
//! code is not deleted: it is not a fallback, it is the definition.
//!
//! Every operand here is F32. That is **validated, not converted** (hot-path
//! invariant 1b): the residual stream is F32 the whole way round the loop, so
//! a cast at this boundary would be a full-tensor pass per sub-block that also
//! hid a producer changing type.
//!
//! # Operands arrive as views, and that decides the vector width
//!
//! The wide residual is sliced out of the wave's own buffer, so these launches
//! routinely see a dense tensor whose storage starts thousands of elements in.
//! The launch path threads that offset through (it slices, it does not take
//! the storage base), but the offset is also what decides whether a `float4`
//! load is legal: a dense tensor starting at element 1 is 4-byte aligned
//! however well-behaved the row width is. [`Operand::vec_ok`] carries that
//! per-operand, and a launch vectorises only when **every** operand agrees.

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaStream, DevicePtr};
use candle::{DType, Result, Tensor};
use candle_kernels::simple::gr_hyper::{run_gr_combine, run_gr_mix, run_gr_norm, GR_MAX_HC};

use crate::models::operand_guard::expect_dtype;

/// A dense F32 operand resolved to a device pointer.
struct Operand {
    ptr: u64,
    /// Whether this operand's base is 16-byte aligned, i.e. its start offset
    /// is a whole number of `float4`s. Allocations are far better aligned than
    /// that; a view's offset is what can break it.
    vec_ok: bool,
}

/// Resolve `t` and hand the pointer to `f`, holding the storage across it.
///
/// Contiguity is required (the kernels index `row · d + j` with no stride
/// metadata) but a nonzero start offset is not: it is added to the pointer
/// here and folded into `vec_ok`.
fn with_operand<R>(t: &Tensor, what: &str, f: impl FnOnce(Operand, &CudaStream) -> R) -> Result<R> {
    expect_dtype(t, DType::F32, what)?;
    if !t.is_contiguous() {
        candle::bail!(
            "{what}: kernel operand has layout {:?} stride {:?}, which is not dense — these \
             kernels index by row and carry no stride argument",
            t.dims(),
            t.stride()
        );
    }
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

/// `xn = grouped_rms(x) ⊙ gain`, one launch over `[n, hc, d]`.
pub fn norm(x: &Tensor, gain: &Tensor, eps: f64) -> Result<Tensor> {
    let (n, hc, d) = x.dims3()?;
    if gain.elem_count() != hc * d {
        candle::bail!(
            "gr norm: gain has {} elements for hc·d = {}",
            gain.elem_count(),
            hc * d
        );
    }
    // Fully overwritten by the kernel (hot-path invariant 6).
    let xn = Tensor::empty((n, hc, d), DType::F32, x.device())?;
    with_operand(x, "gr norm: residual stream", |xo, stream| {
        with_operand(gain, "gr norm: gain", |go, _| {
            with_operand(&xn, "gr norm: out", |oo, _| {
                let vec_ok = xo.vec_ok && go.vec_ok && oo.vec_ok;
                candle::set_kernel_breadcrumb("run_gr_norm", file!(), line!());
                unsafe {
                    run_gr_norm(
                        xo.ptr as *const f32,
                        go.ptr as *const f32,
                        oo.ptr as *mut f32,
                        n as i32,
                        hc as i32,
                        d as i32,
                        eps as f32,
                        i32::from(vec_ok),
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            })
        })
    })???;
    Ok(xn)
}

/// `mixed[t,j] = mean_s( xn[t,s,j] · sigmoid(gate_raw[t,s,j]) )`, one launch.
pub fn mix(xn: &Tensor, gate_raw: &Tensor, hc: usize, d: usize) -> Result<Tensor> {
    let n = xn.elem_count() / (hc * d);
    if gate_raw.elem_count() != xn.elem_count() {
        candle::bail!(
            "gr mix: gate has {} elements against xn's {}",
            gate_raw.elem_count(),
            xn.elem_count()
        );
    }
    let mixed = Tensor::empty((n, d), DType::F32, xn.device())?;
    with_operand(xn, "gr mix: xn", |xo, stream| {
        with_operand(gate_raw, "gr mix: gate", |go, _| {
            with_operand(&mixed, "gr mix: out", |oo, _| {
                let vec_ok = xo.vec_ok && go.vec_ok && oo.vec_ok;
                candle::set_kernel_breadcrumb("run_gr_mix", file!(), line!());
                unsafe {
                    run_gr_mix(
                        xo.ptr as *const f32,
                        go.ptr as *const f32,
                        oo.ptr as *mut f32,
                        n as i32,
                        hc as i32,
                        d as i32,
                        i32::from(vec_ok),
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            })
        })
    })???;
    Ok(mixed)
}

/// `out = res + block_out · 2·sigmoid(inject/hc)`, one launch.
///
/// One read and one write of the wide buffer, where the eager chain took four
/// passes (sigmoid, scale, broadcast-multiply, add).
pub fn combine(res: &Tensor, block_out: &Tensor, inject: &Tensor) -> Result<Tensor> {
    let (n, hc, d) = res.dims3()?;
    // The kernel holds its per-stream weights in a fixed register array and
    // returns without launching above that width. `out` below is allocated
    // uninitialised, so an unlaunched combine would hand back garbage as the
    // new residual and carry it through every remaining layer with nothing
    // raised. The launcher cannot report it — refuse it here.
    if hc > GR_MAX_HC {
        candle::bail!("gr combine: hc={hc} exceeds GR_MAX_HC={GR_MAX_HC}");
    }
    if inject.elem_count() != n * hc {
        candle::bail!(
            "gr combine: inject has {} elements for n·hc = {}",
            inject.elem_count(),
            n * hc
        );
    }
    if block_out.elem_count() != n * d {
        candle::bail!(
            "gr combine: block output has {} elements for n·d = {}",
            block_out.elem_count(),
            n * d
        );
    }
    // Fully overwritten by the kernel (hot-path invariant 6).
    let out = Tensor::empty((n, hc, d), DType::F32, res.device())?;
    with_operand(res, "gr combine: residual", |ro, stream| {
        with_operand(block_out, "gr combine: block output", |bo, _| {
            with_operand(inject, "gr combine: inject", |io, _| {
                with_operand(&out, "gr combine: out", |oo, _| {
                    // `inject` is [n, hc] and read scalar-wise, so its own
                    // alignment does not gate the vector path; the three wide
                    // operands do.
                    let vec_ok = ro.vec_ok && bo.vec_ok && oo.vec_ok;
                    candle::set_kernel_breadcrumb("run_gr_combine", file!(), line!());
                    unsafe {
                        run_gr_combine(
                            ro.ptr as *const f32,
                            bo.ptr as *const f32,
                            io.ptr as *const f32,
                            oo.ptr as *mut f32,
                            n as i32,
                            hc as i32,
                            d as i32,
                            i32::from(vec_ok),
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                })
            })
        })
    })????;
    Ok(out)
}
