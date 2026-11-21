//! Runs one Gated DeltaNet token step for every decode sequence in a wave, in a
//! single launch.
//!
//! States are read through a pointer table and never written. Each token's
//! generators go to row 0 of its sequence's timeline, to be folded in later. Each
//! sequence's output lands in its own row of the wave, so no concatenation is
//! needed. The batched conv does the same for the decode rows. `check_qk_channels`
//! requires the Q and K channels to cover whole 256-channel blocks, so the norm
//! reduction never straddles a block.

use super::*;

/// One fused gated-delta-rule token step for EVERY decode sequence in the
/// wave — one launch however many sessions it carries. States are read through
/// the pointer table and never written: each token's generators go to row 0 of
/// its sequence's timeline, for the fold. Each sequence's output lands in its
/// own row of the wave's `o`. Gates are computed in-kernel from the raw
/// projections; q/k come from the conv output with the tiled GQA mapping.
pub fn delta_net_decode_batch(
    fused: &DeltaNetFused<'_, '_>,
    table: &DeltaNetLayerTable,
) -> Result<()> {
    let (t, conv_dim) = fused.conved.dims2()?;
    let (ta, h_v) = fused.alpha.dims2()?;
    let (to, cols) = fused.o.dims2()?;
    if ta != t || to != t || !cols.is_multiple_of(h_v) {
        candle::bail!(
            "delta_net cuda: decode batch geometry mismatch (conved {t}, alpha {ta}, o {to})"
        );
    }
    let d = cols / h_v;
    // The timeline's fields are laid out for the fused kernels' head width.
    if d != DELTA_NET_PREFILL_DIM {
        candle::bail!(
            "delta_net cuda: the decode step writes a timeline at d == \
             {DELTA_NET_PREFILL_DIM}, got {d}"
        );
    }
    if !conv_dim.is_multiple_of(d) || conv_dim <= h_v * d {
        candle::bail!("delta_net cuda: conv row {conv_dim} is not [Q|K|V] at d = {d}");
    }
    let qk_heads = conv_dim / d - h_v;
    if !qk_heads.is_multiple_of(2) {
        candle::bail!("delta_net cuda: conv row {conv_dim} does not split into Q and K");
    }
    let h_k = qk_heads / 2;
    let (table_rows, n_decode) = table.ptrs.dims2()?;
    if table_rows != TABLE_ROWS || table.rows.dims1()? != n_decode {
        candle::bail!("delta_net cuda: malformed decode table");
    }
    let dev = match fused.conved.device() {
        Device::Cuda(dv) => dv.clone(),
        _ => candle::bail!("delta_net cuda: conved must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    // Row 1 holds the state pointers, row 2 the timelines.
    let states_p = typed_ptr(&table.ptrs.narrow(0, 1, 1)?, DType::I64, "state table")?;
    let timelines_p = typed_ptr(&table.ptrs.narrow(0, 2, 1)?, DType::I64, "timeline table")?;
    let rows_p = typed_ptr(&table.rows, DType::U32, "row table")?;
    let conved_p = f32_ptr(fused.conved, "conved")?;
    let alpha_p = f32_ptr(fused.alpha, "alpha")?;
    let blin_p = f32_ptr(fused.blin, "beta_lin")?;
    let dt_p = f32_ptr(fused.dt_bias, "dt_bias")?;
    let a_p = f32_ptr(fused.a, "a")?;
    let o_p = f32_ptr(fused.o, "o")?;
    let status = unsafe {
        run_delta_net_decode_step_f32(
            states_p as *const i64,
            timelines_p as *const i64,
            conved_p as *const f32,
            rows_p as *const u32,
            alpha_p as *const f32,
            blin_p as *const f32,
            dt_p as *const f32,
            a_p as *const f32,
            o_p as *mut f32,
            n_decode as i32,
            h_v as i32,
            h_k as i32,
            d as i32,
            d as i32,
            conv_dim as i32,
            fused.q_scale,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    if status != 0 {
        candle::bail!(
            "delta_net cuda: the decode step refused {n_decode} rows over h_v {h_v} / h_k {h_k} \
             at d {d} and wrote nothing"
        );
    }
    Ok(())
}

/// The batched decode conv: one launch over every decode sequence in the
/// wave, reading each sequence's raw QKV row of `qkv` and its conv tail, and
/// writing its post-epilogue row of `conved` and its raw input into row 0 of
/// its timeline. The tail is read and never written.
pub fn delta_net_conv_decode(
    qkv: &LiveTensor<'_>,
    kernel: &Tensor,
    table: &DeltaNetLayerTable,
    conved: &LiveTensor<'_>,
    qk_channels: usize,
    eps: f32,
) -> Result<()> {
    let (t, channels) = qkv.dims2()?;
    let (kc, kwidth) = kernel.dims2()?;
    if kc != channels {
        candle::bail!("delta_net cuda: kernel channels {kc} != x channels {channels}");
    }
    if conved.dims2()? != (t, channels) {
        candle::bail!(
            "delta_net cuda: conved must match qkv, got {:?}",
            conved.dims()
        );
    }
    check_qk_channels(qk_channels, channels)?;
    let (table_rows, n_decode) = table.ptrs.dims2()?;
    if table_rows != TABLE_ROWS || table.rows.dims1()? != n_decode {
        candle::bail!("delta_net cuda: malformed decode table");
    }
    let dev = match qkv.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: qkv must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    // Row 0 holds the tail pointers, row 2 the timelines.
    let tails_p = typed_ptr(&table.ptrs.narrow(0, 0, 1)?, DType::I64, "tail table")?;
    let timelines_p = typed_ptr(&table.ptrs.narrow(0, 2, 1)?, DType::I64, "timeline table")?;
    let rows_p = typed_ptr(&table.rows, DType::U32, "row table")?;
    let x_p = f32_ptr(qkv, "qkv")?;
    let k_p = f32_ptr(kernel, "kernel")?;
    let y_p = f32_ptr(conved, "conved")?;
    let status = unsafe {
        run_delta_net_conv_decode_f32(
            x_p as *const f32,
            k_p as *const f32,
            tails_p as *const i64,
            timelines_p as *const i64,
            rows_p as *const u32,
            y_p as *mut f32,
            n_decode as i32,
            channels as i32,
            kwidth as i32,
            qk_channels as i32,
            eps,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    if status != 0 {
        candle::bail!(
            "delta_net cuda: the decode conv refused {n_decode} rows over {channels} channels \
             ({qk_channels} Q|K), window {kwidth}, and wrote nothing"
        );
    }
    Ok(())
}

/// Validate the epilogue geometry both conv wrappers share: `qk_channels`
/// must cover whole 256-channel blocks so the in-kernel norm reduction never
/// straddles a block (`qk_channels = 2·h_k·128 = h_k·256` always does).
pub(super) fn check_qk_channels(qk_channels: usize, channels: usize) -> Result<()> {
    if qk_channels > channels || !qk_channels.is_multiple_of(256) {
        candle::bail!(
            "delta_net cuda: qk_channels {qk_channels} must be a multiple of 256 \
             within {channels} conv channels"
        );
    }
    Ok(())
}
