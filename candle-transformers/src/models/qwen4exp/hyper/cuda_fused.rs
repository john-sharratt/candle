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
//! Operands are handed over as [`crate::models::qwen4exp::device_operand`]
//! resolves them, offset and alignment included.

use std::ffi::c_void;

use candle::quantized::cuda::{produce_q8a128, Q8a128Operand};
use candle::quantized::SumScale;
use candle::wave_provenance::WaveTicket;
use candle::{DType, LiveTensor, Result, Tensor};
use candle_kernels::simple::gr_hyper::{
    gr_hc_supported, run_gr_combine, run_gr_combine_gated, run_gr_mix, run_gr_mix_q8, run_gr_norm,
    run_gr_norm_q8, run_gr_silu_q8, GR_MAX_HC, GR_Q8_LAUNCHED,
};

use crate::models::qwen4exp::device_operand::{with_operand, with_ptr};
use crate::models::wave_buffers::wave_empty_ticketed;

/// `xn = grouped_rms(x) ⊙ gain`, one launch over `[n, hc, d]`.
pub fn norm(x: &Tensor, gain: &Tensor, eps: f64, root: Option<WaveTicket>) -> Result<Tensor> {
    let (n, hc, d) = x.dims3()?;
    if gain.elem_count() != hc * d {
        candle::bail!(
            "gr norm: gain has {} elements for hc·d = {}",
            gain.elem_count(),
            hc * d
        );
    }
    // Fully overwritten by the kernel (hot-path invariant 6).
    // **The seed of the Gated Residual's provenance.** The residual itself is
    // pool-backed by design — it outlives every phase reset — so nothing in this
    // chain has an operand to inherit an arena from. Rooting the norm's output
    // on the open phase gives the rest of `hc_mix` a ticketed operand, and the
    // eager ops after it (the low-rank GEMMs, the silu, the collapse) inherit it
    // the ordinary way. With no phase open this is a pool allocation.
    let xn = wave_empty_ticketed((n, hc, d), DType::F32, x.device(), root)?;
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
                        stream.cu_stream() as *mut c_void,
                    );
                }
            })
        })
    })???;
    Ok(xn)
}

/// [`norm`], also returning `xn` as the q8a128 operand the down projection
/// reads — one launch where the int8 path took the norm and then a quantize that
/// re-read the whole wide buffer. Bit-identical to that pair: each warp quantizes
/// the values it is storing, with `quantize_q8a128_kernel`'s mapping and
/// arithmetic (`gr_hyper.cu`, `gr_norm_q8`).
///
/// Refuses a stream width that is not a multiple of 128 or an unaligned operand
/// rather than running a shuffle over a partial warp. The released width is 2560.
pub fn norm_q8(
    x: &Tensor,
    gain: &Tensor,
    eps: f64,
    root: Option<WaveTicket>,
    sum_scale: SumScale,
) -> Result<(Tensor, Q8a128Operand<'static>)> {
    let (n, hc, d) = x.dims3()?;
    if gain.elem_count() != hc * d {
        candle::bail!(
            "gr norm q8: gain has {} elements for hc·d = {}",
            gain.elem_count(),
            hc * d
        );
    }
    if !d.is_multiple_of(128) {
        candle::bail!("gr norm q8: stream width {d} is not a multiple of 128");
    }
    // Fully overwritten by the kernel (hot-path invariant 6), rooted as `norm`'s is.
    let xn = wave_empty_ticketed((n, hc, d), DType::F32, x.device(), root)?;
    let q8 = produce_q8a128(&xn, n, hc * d, sum_scale, |q8_ptr| -> Result<()> {
        with_operand(x, "gr norm q8: residual stream", |xo, stream| {
            with_operand(gain, "gr norm q8: gain", |go, _| {
                with_operand(&xn, "gr norm q8: out", |oo, _| {
                    if !(xo.vec_ok && go.vec_ok && oo.vec_ok) {
                        candle::bail!("gr norm q8: an operand is not 16-byte aligned");
                    }
                    candle::set_kernel_breadcrumb("run_gr_norm_q8", file!(), line!());
                    let status = unsafe {
                        run_gr_norm_q8(
                            xo.ptr as *const f32,
                            go.ptr as *const f32,
                            oo.ptr as *mut f32,
                            q8_ptr as *mut c_void,
                            n as i32,
                            hc as i32,
                            d as i32,
                            eps as f32,
                            sum_scale.as_code(),
                            stream.cu_stream() as *mut c_void,
                        )
                    };
                    q8_launched(status, "gr norm q8")
                })
            })
        })????;
        Ok(())
    })?;
    Ok((xn, q8))
}

/// A q8a128-emitting launcher's status as a result. A refusal wrote nothing, and
/// the operand it was to fill is read by the next GEMM — so it is an error here,
/// never a quiet skip.
fn q8_launched(status: i32, what: &str) -> Result<()> {
    if status != GR_Q8_LAUNCHED {
        candle::bail!("{what}: the launcher refused its shape (status {status}) and wrote nothing");
    }
    Ok(())
}

/// The q8a128 operand of `silu(proj[:, 0..cols])` for the up projection, read
/// through `proj`'s row stride — one launch in place of a strided `silu` into a
/// fresh dense buffer and a quantize that re-reads it, and bit-identical to that
/// pair (`gr_hyper.cu`, `gr_silu_q8`). The producer for waves past the fused `up`
/// loader's width (see [`super::HC_FUSED_SILU_MAX_ROWS`]).
pub fn silu_q8<'w>(
    proj: &LiveTensor<'w>,
    cols: usize,
    sum_scale: SumScale,
) -> Result<Q8a128Operand<'w>> {
    let (n, width) = proj.dims2()?;
    let row_stride = proj.stride()[0];
    if proj.stride()[1] != 1 || cols > width || !cols.is_multiple_of(128) {
        candle::bail!(
            "gr silu q8: {cols} gate columns of a {:?} operand with stride {:?} — need unit \
             column stride and a multiple of 128 within the row",
            proj.dims(),
            proj.stride()
        );
    }
    if !row_stride.is_multiple_of(4) {
        candle::bail!("gr silu q8: row stride {row_stride} breaks 16-byte row alignment");
    }
    produce_q8a128(proj, n, cols, sum_scale, |q8_ptr| {
        with_ptr(proj, "gr silu q8: projection", |po, stream| {
            if !po.vec_ok {
                candle::bail!("gr silu q8: the projection is not 16-byte aligned");
            }
            candle::set_kernel_breadcrumb("run_gr_silu_q8", file!(), line!());
            let status = unsafe {
                run_gr_silu_q8(
                    po.ptr as *const f32,
                    q8_ptr as *mut c_void,
                    n as i32,
                    cols as i32,
                    row_stride as i32,
                    sum_scale.as_code(),
                    stream.cu_stream() as *mut c_void,
                )
            };
            q8_launched(status, "gr silu q8")
        })?
    })
}

/// `mixed[t,j] = mean_s( xn[t,s,j] · sigmoid(gate_raw[t,s,j]) )`, one launch.
pub fn mix(
    xn: &Tensor,
    gate_raw: &Tensor,
    hc: usize,
    d: usize,
    root: Option<WaveTicket>,
) -> Result<Tensor> {
    Ok(mix_impl(xn, gate_raw, hc, d, root, None)?.0)
}

/// [`mix`], also returning `mixed` as the q8a128 operand the block's own
/// projections read — the collapse quantizing what it stores, in place of a
/// standalone quantize that re-read it (`gr_hyper.cu`, `gr_mix<.., EMIT_Q8>`).
/// Bit-identical to that pair. Refuses a width that is not a multiple of 128 or
/// an unaligned operand.
pub fn mix_q8(
    xn: &Tensor,
    gate_raw: &Tensor,
    hc: usize,
    d: usize,
    root: Option<WaveTicket>,
    sum_scale: SumScale,
) -> Result<(Tensor, Q8a128Operand<'static>)> {
    if !d.is_multiple_of(128) {
        candle::bail!("gr mix q8: width {d} is not a multiple of 128");
    }
    let (mixed, q8) = mix_impl(xn, gate_raw, hc, d, root, Some(sum_scale))?;
    Ok((mixed, q8.expect("requested")))
}

fn mix_impl(
    xn: &Tensor,
    gate_raw: &Tensor,
    hc: usize,
    d: usize,
    root: Option<WaveTicket>,
    q8: Option<SumScale>,
) -> Result<(Tensor, Option<Q8a128Operand<'static>>)> {
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
    // The block input, consumed inside the phase that produced it.
    let mixed = wave_empty_ticketed((n, d), DType::F32, xn.device(), root)?;
    // The launch, with or without the operand's address.
    let launch = |q8_ptr: Option<u64>| -> Result<()> {
        with_operand(xn, "gr mix: xn", |xo, stream| {
            with_operand(gate_raw, "gr mix: gate", |go, _| {
                with_operand(&mixed, "gr mix: out", |oo, _| {
                    let vec_ok = xo.vec_ok && go.vec_ok && oo.vec_ok;
                    let stream = stream.cu_stream() as *mut c_void;
                    match (q8_ptr, q8) {
                        (Some(ptr), Some(ss)) => {
                            if !vec_ok {
                                candle::bail!("gr mix q8: an operand is not 16-byte aligned");
                            }
                            candle::set_kernel_breadcrumb("run_gr_mix_q8", file!(), line!());
                            let status = unsafe {
                                run_gr_mix_q8(
                                    xo.ptr as *const f32,
                                    go.ptr as *const f32,
                                    oo.ptr as *mut f32,
                                    ptr as *mut c_void,
                                    n as i32,
                                    hc as i32,
                                    d as i32,
                                    ss.as_code(),
                                    stream,
                                )
                            };
                            q8_launched(status, "gr mix q8")?;
                        }
                        _ => {
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
                                    stream,
                                );
                            }
                        }
                    }
                    Ok(())
                })
            })
        })????;
        Ok(())
    };
    let op = match q8 {
        Some(ss) => Some(produce_q8a128(&mixed, n, d, ss, |ptr| launch(Some(ptr)))?),
        None => {
            launch(None)?;
            None
        }
    };
    Ok((mixed, op))
}

/// `res += block_out · 2·sigmoid(inject/hc)`, in place, one launch.
///
/// One read and one write of the wide buffer, where the eager chain took four
/// passes (sigmoid, scale, broadcast-multiply, add) and a second residual to
/// write them into. `res` is taken `&mut` for the reason `Tensor::add_mut` is:
/// the caller states it holds the residual it is updating, and nothing else
/// reads it expecting the old value. A row-range view of the wave's residual is
/// a valid `res` — its start offset is threaded to the kernel — which is how a
/// wave's groups each combine their own rows with no concatenation between.
///
/// `block_out` and `inject` are borrowed at the caller's wave lifetime: they are
/// the block's output and the pre-mix's projection, still on that phase's arena
/// span, and this kernel only reads them.
pub fn combine(
    res: &mut Tensor,
    block_out: &LiveTensor<'_>,
    inject: &LiveTensor<'_>,
) -> Result<()> {
    combine_impl(res, block_out, None, inject)
}

/// The MoE's shared-expert gate, as [`combine_gated`] reads it: the shared
/// expert's output and its raw (pre-sigmoid) per-token gate.
pub struct SharedGate<'a, 'w> {
    /// `[n, d]`.
    pub shared: &'a LiveTensor<'w>,
    /// `[n, 1]`, read through its row stride — the first column of the gate
    /// projection's padded output, never compacted.
    pub gate: &'a LiveTensor<'w>,
}

/// [`combine`] with the MoE's block output assembled in the same pass:
/// `res += (routed + shared · sigmoid(gate)) · 2·sigmoid(inject/hc)`.
///
/// Three launches per MoE layer — the gate's sigmoid, the broadcast multiply,
/// the add — and the `[n, d]` buffers between them become arithmetic in the
/// combine that was already reading the block output. Bit-identical to them: the
/// same shared sigmoid, and the product rounded before the add (`gr_hyper.cu`).
pub fn combine_gated(
    res: &mut Tensor,
    routed: &LiveTensor<'_>,
    shared: SharedGate<'_, '_>,
    inject: &LiveTensor<'_>,
) -> Result<()> {
    combine_impl(res, routed, Some(shared), inject)
}

fn combine_impl(
    res: &mut Tensor,
    block_out: &LiveTensor<'_>,
    gated: Option<SharedGate<'_, '_>>,
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
    let Some(SharedGate { shared, gate }) = gated else {
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
                            stream.cu_stream() as *mut c_void,
                        );
                    }
                })
            })
        })???;
        return Ok(());
    };
    if shared.elem_count() != n * d {
        candle::bail!(
            "gr combine: shared expert output has {} elements for n·d = {}",
            shared.elem_count(),
            n * d
        );
    }
    // The gate is the first column of the gate projection's padded output: one
    // scalar per row, `gate_stride` apart.
    if gate.dims() != [n, 1] {
        candle::bail!("gr combine: gate is {:?}, expected [{n}, 1]", gate.dims());
    }
    let gate_stride = gate.stride()[0];
    with_operand(res, "gr combine: residual", |ro, stream| {
        with_operand(block_out, "gr combine: routed", |bo, _| {
            with_operand(shared, "gr combine: shared", |so, _| {
                with_ptr(gate, "gr combine: gate", |go, _| {
                    with_ptr(inject, "gr combine: inject", |io, _| {
                        // The three wide operands gate the vector path; the gate
                        // and inject are read scalar-wise.
                        let vec_ok = ro.vec_ok && bo.vec_ok && so.vec_ok;
                        candle::set_kernel_breadcrumb("run_gr_combine_gated", file!(), line!());
                        unsafe {
                            run_gr_combine_gated(
                                ro.ptr as *mut f32,
                                bo.ptr as *const f32,
                                so.ptr as *const f32,
                                go.ptr as *const f32,
                                io.ptr as *const f32,
                                n as i32,
                                hc as i32,
                                d as i32,
                                inject_stride as i32,
                                gate_stride as i32,
                                i32::from(vec_ok),
                                stream.cu_stream() as *mut c_void,
                            );
                        }
                    })
                })
            })
        })
    })?????;
    Ok(())
}
