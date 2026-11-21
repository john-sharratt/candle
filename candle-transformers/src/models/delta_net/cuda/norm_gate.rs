//! The mixer's output epilogue: the per-head RMS norm, the gain and the z-gate,
//! over the whole wave in one launch.
//!
//! The one launch replaces about six operations and three full-width
//! intermediates. The gate is a kernel instantiation chosen by the `zgate`
//! argument, SiLU for the Qwen3.5 lineage and sigmoid for qwen4exp. The q8a128
//! variant emits the operand an out-projection reads, the bytes a standalone
//! quantise would write, so no F32 output is stored. That variant needs a head
//! width of 128.

use super::*;

/// The mixer epilogue over the whole wave in one launch: per `(token, V head)`
/// row, `out = (o / sqrt(mean(o²) + eps)) ⊙ gain ⊙ zgate(z)` — the per-head
/// RMS norm and the z-gate that were ~6 ops and three full-width
/// intermediates. The gate is a kernel template instantiation selected by
/// `zgate` (SiLU for the Qwen3.5 lineage, sigmoid for qwen4exp).
pub fn delta_net_norm_gate<'w>(
    o: &LiveTensor<'w>,
    z: &LiveTensor<'w>,
    gain: &Tensor,
    d: usize,
    eps: f32,
    zgate: ZGate,
) -> Result<LiveTensor<'w>> {
    let (t, cols) = norm_gate_shape(o, z, gain, d)?;
    let out = o.empty_beside((t, cols), DType::F32)?;
    norm_gate_launch(o, z, gain, d, eps, zgate, f32_ptr(&out, "out")?, 0, 0)?;
    Ok(out)
}

/// [`delta_net_norm_gate`]'s output as the q8a128 operand under `sum_scale`
/// alone — the bytes a standalone quantize of that output would write, emitted
/// by the epilogue from the values it computes — for an out-projection that
/// runs int8 and reads nothing else. No F32 output is stored. One tile per row,
/// so `d` must be 128.
pub fn delta_net_norm_gate_q8<'w>(
    o: &LiveTensor<'w>,
    z: &LiveTensor<'w>,
    gain: &Tensor,
    d: usize,
    eps: f32,
    zgate: ZGate,
    sum_scale: SumScale,
) -> Result<Q8a128Operand<'w>> {
    let (t, cols) = norm_gate_shape(o, z, gain, d)?;
    if d != 128 {
        candle::bail!("delta_net cuda: the norm-gate emits its operand one 128-wide tile per row, got d = {d}");
    }
    // The operand is placed beside `o`, the way the F32 output would have been.
    produce_q8a128(o, t, cols, sum_scale, |q8| {
        norm_gate_launch(o, z, gain, d, eps, zgate, 0, q8, sum_scale.as_code())
    })
}

/// `(T, cols)` of a norm-gate over `o`, after checking `z` and `gain` against it.
pub(super) fn norm_gate_shape(
    o: &LiveTensor<'_>,
    z: &LiveTensor<'_>,
    gain: &Tensor,
    d: usize,
) -> Result<(usize, usize)> {
    let (t, cols) = o.dims2()?;
    if z.dims2()? != (t, cols) {
        candle::bail!(
            "delta_net cuda: z must match o, got {:?} vs {:?}",
            z.dims(),
            o.dims()
        );
    }
    if gain.dims1()? != d || !cols.is_multiple_of(d) || d > MAX_HEAD_DIM {
        candle::bail!(
            "delta_net cuda: gain [{d}] must divide {cols} rows-wise (cap {MAX_HEAD_DIM})"
        );
    }
    Ok((t, cols))
}

/// One norm-gate launch writing the F32 output at `out_p`, the q8a128 operand
/// at `q8`, or both; a `0` address writes nothing there.
#[allow(clippy::too_many_arguments)]
pub(super) fn norm_gate_launch(
    o: &LiveTensor<'_>,
    z: &LiveTensor<'_>,
    gain: &Tensor,
    d: usize,
    eps: f32,
    zgate: ZGate,
    out_p: u64,
    q8: u64,
    sum_norm: i32,
) -> Result<()> {
    let (t, cols) = o.dims2()?;
    let dev = match o.device() {
        Device::Cuda(dv) => dv.clone(),
        _ => candle::bail!("delta_net cuda: o must live on a CUDA device"),
    };
    let o_p = f32_ptr(o, "o")?;
    let z_p = f32_ptr(z, "z")?;
    let g_p = f32_ptr(gain, "gain")?;
    let rows = t * (cols / d);
    let stream = dev.cuda_stream();
    // SAFETY: `o`, `z` and `gain` are live F32 buffers of the checked shapes,
    // and `out_p` / `q8` are either 0 or sized by the caller for `rows` rows
    // of `d` (one q8a128 tile per row); the launch is on the device's stream.
    let status = unsafe {
        run_delta_net_norm_gate_f32(
            o_p as *const f32,
            z_p as *const f32,
            g_p as *const f32,
            out_p as *mut f32,
            rows as i32,
            d as i32,
            eps,
            matches!(zgate, ZGate::Sigmoid) as i32,
            q8 as *mut u8,
            sum_norm,
            stream.cu_stream() as *mut c_void,
        )
    };
    if status != 0 {
        candle::bail!("delta_net cuda: the norm-gate refused rows={rows} d={d} and wrote nothing");
    }
    Ok(())
}
