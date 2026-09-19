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
use candle::{DType, LiveTensor, Result, Tensor};
use candle_kernels::simple::gr_hyper::{
    gr_hc_supported, run_gr_combine, run_gr_mix, run_gr_norm, GR_MAX_HC,
};

use candle_nn::kv_cache::WaveGeneration;

use candle::wave_provenance::WaveTicket;

use crate::models::operand_guard::expect_dtype;
use crate::models::wave_buffers::{wave_empty, wave_empty_ticketed};

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
fn with_operand<R>(
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
fn with_ptr<R>(
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

/// `xn = grouped_rms(x) ⊙ gain`, one launch over `[n, hc, d]`.
///
/// **The seed of its phase.** `x` is the residual stream, which crosses layers
/// and so lives on the pool with no arena to inherit; `xn` is carved from
/// `wave` directly, and every op downstream of it — the low-rank gate GEMMs,
/// the mix, the block that reads the mix — inherits that arena from here.
pub fn norm<'w>(
    x: &LiveTensor<'_>,
    gain: &Tensor,
    eps: f64,
    wave: Option<&'w WaveGeneration>,
) -> Result<LiveTensor<'w>> {
    let (n, hc, d) = x.dims3()?;
    // Fully overwritten by the kernel (hot-path invariant 6).
    let xn = wave_empty((n, hc, d), DType::F32, x.device(), wave)?;
    norm_into(x, gain, eps, &xn)?;
    Ok(xn)
}

/// [`norm`] seeded from a wave **ticket** rather than the guard, for a result
/// that must be `'static`-typed while it sits on the span: the head's, whose
/// logits leave the forward with the guard handed back beside them.
pub fn norm_rooted(
    x: &LiveTensor<'_>,
    gain: &Tensor,
    eps: f64,
    root: Option<WaveTicket>,
) -> Result<Tensor> {
    let (n, hc, d) = x.dims3()?;
    let xn = wave_empty_ticketed((n, hc, d), DType::F32, x.device(), root)?;
    norm_into(x, gain, eps, &xn)?;
    Ok(xn)
}

/// The launch both seeds share: `xn` is written in full.
fn norm_into(x: &LiveTensor<'_>, gain: &Tensor, eps: f64, xn: &LiveTensor<'_>) -> Result<()> {
    let (n, hc, d) = x.dims3()?;
    if gain.elem_count() != hc * d {
        candle::bail!(
            "gr norm: gain has {} elements for hc·d = {}",
            gain.elem_count(),
            hc * d
        );
    }
    with_operand(x, "gr norm: residual stream", |xo, stream| {
        with_operand(gain, "gr norm: gain", |go, _| {
            with_operand(xn, "gr norm: out", |oo, _| {
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
    Ok(())
}

/// `mixed[t,j] = mean_s( xn[t,s,j] · sigmoid(gate_raw[t,s,j]) )`, one launch,
/// in `xn`'s arena.
pub fn mix<'w>(
    xn: &LiveTensor<'w>,
    gate_raw: &LiveTensor<'_>,
    hc: usize,
    d: usize,
) -> Result<LiveTensor<'w>> {
    // As for the combine: an uninstantiated stream count launches nothing and
    // leaves `mixed` unwritten.
    if !gr_hc_supported(hc) {
        candle::bail!("gr mix: hc={hc} is not a power of two up to GR_MAX_HC={GR_MAX_HC}");
    }
    let n = xn.elem_count() / (hc * d);
    if gate_raw.elem_count() != xn.elem_count() {
        candle::bail!(
            "gr mix: gate has {} elements against xn's {}",
            gate_raw.elem_count(),
            xn.elem_count()
        );
    }
    let mixed = xn.empty_beside((n, d), DType::F32)?;
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

/// `res += block_out · 2·sigmoid(inject/hc)`, in place, one launch.
///
/// One read and one write of the wide buffer, where the eager chain took four
/// passes (sigmoid, scale, broadcast-multiply, add) and a fresh residual. `res`
/// is taken `&mut` for the reason `Tensor::add_mut` is: the caller states it
/// holds the residual it is updating. A row-range view of the wave's residual
/// is a valid `res` — its start offset is threaded to the kernel — which is how
/// a wave's groups each combine their own rows with no concatenation between.
pub fn combine(
    res: &mut Tensor,
    block_out: &LiveTensor<'_>,
    inject: &LiveTensor<'_>,
) -> Result<()> {
    let (n, hc, d) = res.dims3()?;
    // The launcher returns without launching for a stream count it has no
    // instantiation for, which would leave the residual silently un-updated
    // for every remaining layer. It cannot report that — refuse it here.
    if !gr_hc_supported(hc) {
        candle::bail!("gr combine: hc={hc} is not a power of two up to GR_MAX_HC={GR_MAX_HC}");
    }
    // The inject is read through its row stride: it is the tail columns of the
    // pre-mix's stacked down-projection, a `[n, hc]` view of a `[n, low_rank +
    // hc]` buffer, and compacting it first would be a launch and an allocation
    // per call for 16 bytes a row.
    if inject.dims() != [n, hc] || inject.stride()[1] != 1 || inject.stride()[0] < hc {
        candle::bail!(
            "gr combine: inject is {:?} stride {:?}, expected [{n}, {hc}] with unit column \
             stride",
            inject.dims(),
            inject.stride()
        );
    }
    let inject_stride = inject.stride()[0];
    if block_out.elem_count() != n * d {
        candle::bail!(
            "gr combine: block output has {} elements for n·d = {}",
            block_out.elem_count(),
            n * d
        );
    }
    with_operand(res, "gr combine: residual", |ro, stream| {
        with_operand(block_out, "gr combine: block output", |bo, _| {
            with_ptr(inject, "gr combine: inject", |io, _| {
                // `inject` is [n, hc] and read scalar-wise, so its own
                // alignment does not gate the vector path; the two wide
                // operands do.
                let vec_ok = ro.vec_ok && bo.vec_ok;
                candle::set_kernel_breadcrumb("run_gr_combine", file!(), line!());
                unsafe {
                    run_gr_combine(
                        ro.ptr as *mut f32,
                        bo.ptr as *const f32,
                        io.ptr as *const f32,
                        n as i32,
                        hc as i32,
                        d as i32,
                        inject_stride as i32,
                        i32::from(vec_ok),
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            })
        })
    })???;
    Ok(())
}
