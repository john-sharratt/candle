//! The prefill kernels for the Gated DeltaNet mixer: a token-parallel conv, a
//! chunked scan, and a short-span kernel for verify blocks.
//!
//! The chunked scan replaces about sixty-five operations per chunk with two
//! launches. It uses a triangular solve rather than an explicit inverse, which is
//! both cheaper and more accurate. The solve works on row-major buffers that
//! cuBLAS reads as column-major, which is why its side and fill flags look
//! transposed. The short-span kernel runs the conv and the scan for every span up
//! to a fixed number of rows in one launch. It writes a timeline instead of an
//! advanced state, so a speculative rewind can fold just the rows it keeps.

use super::*;

/// Solve `(I + A) X = B` for every head, where `a [H, c, c]` is **strictly**
/// lower triangular (its diagonal is ignored — the identity supplies it) and
/// `rhs [H, c, n]` holds the right-hand sides. Returns `X [H, c, n]`.
///
/// # Why a solve and not an inverse
///
/// The chunked scan needs `T·(βv)` and `T·(βk⊙e^G)` where `T = (I + A)⁻¹`.
/// Forming `T` and multiplying is the obvious reading, and it is what the
/// reference [`super::mix::unit_lower_inverse`] does — by a recursive 2×2
/// block inversion that costs `log₂ c` levels of a handful of launches each,
/// about eighty host-issued ops per chunk, all of them serially dependent. Both
/// right-hand sides share a left side, so concatenating them into one solve
/// replaces the whole of that with a single `trsm`, which is also the more
/// accurate answer: substitution is backward-stable and never forms an inverse.
///
/// # Row-major against a column-major library
///
/// cuBLAS reads column-major, and every tensor here is row-major, so the
/// buffers are *reinterpreted* rather than transposed — a row-major `[c, c]`
/// with leading dimension `c` is exactly a column-major `Aᵀ`, and `Aᵀ` of a
/// lower triangle is an upper one. Transposing the equation to match,
/// `A X = B` becomes `Xᵀ Aᵀ = Bᵀ`, which is `trsm`'s right-side form over the
/// reinterpreted buffers. Hence `SIDE_RIGHT` and `FILL_MODE_UPPER` on data that
/// is neither: they describe the column-major view, not the tensor.
///
/// `trsm` overwrites its right-hand side with the solution, so `rhs` must be a
/// buffer the caller owns; the result aliases it.
pub fn solve_unit_lower<'w>(a: &LiveTensor<'w>, rhs: LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    let (h, c, c2) = a.dims3()?;
    if c != c2 {
        candle::bail!("delta_net cuda: solve matrix must be square, got [{h}, {c}, {c2}]");
    }
    let (hr, cr, n) = rhs.dims3()?;
    if hr != h || cr != c {
        candle::bail!(
            "delta_net cuda: rhs [{hr}, {cr}, {n}] does not match matrix [{h}, {c}, {c}]"
        );
    }
    let dev = match a.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: solve_unit_lower requires a CUDA tensor"),
    };
    if h == 0 || c == 0 || n == 0 {
        return Ok(rhs);
    }
    let stream = dev.cuda_stream();
    let a_p = f32_ptr(a, "solve matrix")?;
    let b_p = f32_ptr(&rhs, "solve rhs")?;

    // Two `[h]` address arrays, written on the device. `I64` because that is
    // the eight-byte dtype available; the bytes are pointers.
    let ptrs = rhs.empty_beside((2, h), DType::I64)?;
    let ptr_base = typed_ptr(&ptrs, DType::I64, "solve pointer table")?;
    let a_ptrs = ptr_base as *mut *const f32;
    // SAFETY: `ptrs` is `2 × h` eight-byte slots; the second row starts at `h`.
    let b_ptrs = unsafe { (ptr_base as *mut *mut f32).add(h) };
    unsafe {
        run_delta_net_batch_ptrs(
            a_p as *const f32,
            (c * c) as i64,
            b_p as *mut f32,
            (c * n) as i64,
            a_ptrs,
            b_ptrs,
            h as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }

    let alpha = 1f32;
    let status = unsafe {
        cublas::cublasStrsmBatched(
            *dev.cublas()?.handle(),
            cublas::cublasSideMode_t::CUBLAS_SIDE_RIGHT,
            cublas::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER,
            cublas::cublasOperation_t::CUBLAS_OP_N,
            cublas::cublasDiagType_t::CUBLAS_DIAG_UNIT,
            n as i32,
            c as i32,
            &alpha as *const f32,
            a_ptrs as *const *const f32,
            c as i32,
            b_ptrs as *const *mut f32,
            n as i32,
            h as i32,
        )
    };
    if status != cublas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        candle::bail!("delta_net cuda: cublasStrsmBatched failed: {status:?}");
    }
    Ok(rhs)
}

/// Token-parallel causal conv over every prefill span of a wave, with the SiLU
/// and Q|K-norm epilogue, written directly into each span's rows of the
/// whole-wave `conved` buffer — the operand buffer the scan kernels read q/k/v
/// from through strides.
///
/// `qkv [T_wave, channels]` is **token-major** — the projection's own rows, so
/// nothing is transposed to channel-major. Each span's advanced tail (raw
/// values) goes to its tail-out, a **different** allocation from its entering
/// tail: blocks computing the first `kwidth−1` outputs read the entering tail
/// concurrently, so writing over it would race. The tail-out is the sequence's
/// prefill scratch, so the advance lands where `commit_wave` installs it and
/// nothing is copied.
// Eight: the kernel's own operands. Boxing them into a struct would put a
// shape-checked argument list behind a constructor that checks nothing.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_conv_prefill(
    qkv: &LiveTensor<'_>,
    kernel: &Tensor,
    table: &DeltaNetSpanTable,
    qk_channels: usize,
    eps: f32,
    conved: &LiveTensor<'_>,
) -> Result<()> {
    let (t_wave, channels) = qkv.dims2()?;
    let (kc, kwidth) = kernel.dims2()?;
    if kc != channels {
        candle::bail!("delta_net cuda: kernel channels {kc} != x channels {channels}");
    }
    if !(2..=DELTA_NET_PREFILL_CONV_MAX).contains(&kwidth) {
        candle::bail!(
            "delta_net cuda: the prefill conv takes windows of 2..={DELTA_NET_PREFILL_CONV_MAX}, \
             got {kwidth}"
        );
    }
    let (tc, cc) = conved.dims2()?;
    if cc != channels || tc != t_wave {
        candle::bail!(
            "delta_net cuda: conv output {:?} does not match its operand {:?}",
            conved.dims(),
            qkv.dims()
        );
    }
    check_qk_channels(qk_channels, channels)?;
    let dev = match qkv.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: x must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    check_table_rows(table, LONG_TABLE_ROWS, "the prefill conv")?;
    let x_p = f32_ptr(qkv, "qkv")?;
    let k_p = f32_ptr(kernel, "kernel")?;
    let y_p = f32_ptr(conved, "conved")?;
    let ptrs_p = typed_ptr(&table.ptrs, DType::I64, "span ptrs")?;
    let spans_p = typed_ptr(&table.spans, DType::U32, "span extents")?;
    let status = unsafe {
        run_delta_net_conv_prefill_f32(
            x_p as *const f32,
            k_p as *const f32,
            y_p as *mut f32,
            ptrs_p as *const i64,
            spans_p as *const u32,
            table.n as i32,
            table.max_len as i32,
            channels as i32,
            kwidth as i32,
            qk_channels as i32,
            eps,
            stream.cu_stream() as *mut c_void,
        )
    };
    if status != 0 {
        candle::bail!(
            "delta_net cuda: the prefill conv refused {} spans of up to {} rows, {channels} \
             channels, window {kwidth} and wrote nothing",
            table.n,
            table.max_len
        );
    }
    Ok(())
}

/// The fused prefill scan for every prefill span of a wave: the chunked
/// parallel form of the gated delta rule in two launches, replacing
/// `delta_chunked`'s ~65 ops per chunk.
///
/// Each span's entering state `[h_v, 128, 128]` is read and its advanced state
/// written to its state-out — the prefill scratch — in the stored orientation
/// (no `s_fla` transpose exists anywhere on this path); everything else is
/// read from — and written into — the wave tensors in `fused`, through
/// strides. The intra-chunk kernel (parallel over chunks × heads) hands
/// `u`/`w`/`kq` to the sequential state walk through wave-backed transients;
/// the state kernel keeps its S tile in shared memory across the whole sequence.
pub fn delta_net_prefill_scan(
    fused: &DeltaNetFused<'_, '_>,
    table: &DeltaNetSpanTable,
) -> Result<()> {
    // Resolved over the WHOLE wave, not a span: the kernels rebase to their own
    // span from the table, so the pointers handed to them are the packed
    // buffers' bases.
    let t_wave = fused.conved.dim(0)?;
    let p = fused.resolve_wave()?;
    if p.d != DELTA_NET_PREFILL_DIM {
        candle::bail!(
            "delta_net cuda: the prefill scan is compiled for d == {DELTA_NET_PREFILL_DIM}, \
             got {}",
            p.d
        );
    }
    let dev = match fused.conved.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: the wave must live on a CUDA device"),
    };
    // Transients: fully kernel-written, allocated in the wave's arena by
    // operand provenance (pool fallback when full). `kq` rows are valid for
    // `s ≤ t` only; the state kernel never reads past that, so the rest stays
    // uninitialised.
    //
    // ONE allocation for the whole wave rather than one per span: a head's rows
    // are `t_wave` apart and each span writes the slice at its own `start`, so
    // the spans never overlap and no span reads another's. Sized by the wave and
    // not by the spans' total, so the row index is the same number the packed
    // buffers use and no second mapping exists to disagree.
    let (h_v, d) = (p.h_v, p.d);
    let u = fused.conved.empty_beside((h_v, t_wave, d), DType::F32)?;
    let w = fused.conved.empty_beside((h_v, t_wave, d), DType::F32)?;
    let kq = fused
        .conved
        .empty_beside((h_v, t_wave, DELTA_NET_PREFILL_CHUNK), DType::F32)?;
    let g_cs = fused.conved.empty_beside((h_v, t_wave), DType::F32)?;
    {
        let stream = dev.cuda_stream();
        aligned16(p.qk_p, "the conv output")?;
        let u_p = aligned16(f32_ptr(&u, "u")?, "u")?;
        let w_p = aligned16(f32_ptr(&w, "w")?, "w")?;
        let kq_p = aligned16(f32_ptr(&kq, "kq")?, "kq")?;
        let gcs_p = f32_ptr(&g_cs, "g_cs")?;
        check_table_rows(table, LONG_TABLE_ROWS, "the prefill scan")?;
        let ptrs_p = typed_ptr(&table.ptrs, DType::I64, "span ptrs")?;
        let spans_p = typed_ptr(&table.spans, DType::U32, "span extents")?;
        let raw = stream.cu_stream() as *mut core::ffi::c_void;
        unsafe {
            run_delta_net_prefill_intra_f32(
                p.qk_p as *const f32,
                p.v_p as *const f32,
                p.alpha_p as *const f32,
                p.blin_p as *const f32,
                p.dt_p as *const f32,
                p.a_p as *const f32,
                u_p as *mut f32,
                w_p as *mut f32,
                kq_p as *mut f32,
                gcs_p as *mut f32,
                spans_p as *const u32,
                table.n as i32,
                table.max_len as i32,
                t_wave as i32,
                p.h_v as i32,
                p.h_k as i32,
                p.conv_dim as i32,
                fused.q_scale,
                raw,
            );
            run_delta_net_prefill_state_f32(
                p.qk_p as *const f32,
                u_p as *const f32,
                w_p as *const f32,
                kq_p as *const f32,
                gcs_p as *const f32,
                p.o_p as *mut f32,
                ptrs_p as *const i64,
                spans_p as *const u32,
                table.n as i32,
                t_wave as i32,
                p.h_v as i32,
                p.h_k as i32,
                p.conv_dim as i32,
                fused.q_scale,
                raw,
            );
        }
    }
    Ok(())
}

/// The conv, the intra-chunk solve and the state pass's reads of the fused
/// scan for the wave's short spans — each at most [`DELTA_NET_SHORT_SPAN_ROWS`]
/// rows — in ONE launch, where [`delta_net_conv_prefill`] +
/// [`delta_net_prefill_scan`] take three chained through global memory.
///
/// The same bits as that pair in `fused.o`'s span rows. It writes each span's
/// timeline (G, k, v_new and the raw conv rows) and leaves the span's state and
/// conv tail as they were — the fold applies the rows the step keeps. It reads
/// the raw `qkv` rows and convolves them itself, so the span rows of
/// `fused.conved` are neither read nor written — that buffer supplies the
/// wave's geometry here, and holds only the decode rows the batched decode conv
/// writes beside this.
///
/// Refused with an error, nothing written, when a span is longer than
/// [`DELTA_NET_SHORT_SPAN_ROWS`] or the conv is not
/// [`DELTA_NET_SHORT_SPAN_CONV`] wide — the shapes the kernel serves.
pub fn delta_net_short_span_scan(
    fused: &DeltaNetFused<'_, '_>,
    qkv: &LiveTensor<'_>,
    kernel: &Tensor,
    table: &DeltaNetSpanTable,
    eps: f32,
) -> Result<()> {
    check_table_rows(table, TABLE_ROWS, "the short-span scan")?;
    if table.max_len > DELTA_NET_SHORT_SPAN_ROWS {
        candle::bail!(
            "delta_net cuda: the short-span scan takes spans of at most \
             {DELTA_NET_SHORT_SPAN_ROWS} rows, got one of {}",
            table.max_len
        );
    }
    let p = fused.resolve_wave()?;
    if p.d != DELTA_NET_PREFILL_DIM {
        candle::bail!(
            "delta_net cuda: the short-span scan is compiled for d == \
             {DELTA_NET_PREFILL_DIM}, got {}",
            p.d
        );
    }
    let t_wave = fused.conved.dim(0)?;
    if qkv.dims2()? != (t_wave, p.conv_dim) {
        candle::bail!(
            "delta_net cuda: qkv {:?} does not match the wave's conv rows [{t_wave}, {}]",
            qkv.dims(),
            p.conv_dim
        );
    }
    if kernel.dims2()? != (p.conv_dim, DELTA_NET_SHORT_SPAN_CONV) {
        candle::bail!(
            "delta_net cuda: the short-span scan convolves {DELTA_NET_SHORT_SPAN_CONV} wide over \
             {} channels, got a {:?} kernel",
            p.conv_dim,
            kernel.dims()
        );
    }
    let dev = match qkv.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: qkv must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    let x_p = f32_ptr(qkv, "qkv")?;
    let k_p = f32_ptr(kernel, "kernel")?;
    let ptrs_p = typed_ptr(&table.ptrs, DType::I64, "span ptrs")?;
    let spans_p = typed_ptr(&table.spans, DType::U32, "span extents")?;
    let status = unsafe {
        run_delta_net_short_span_f32(
            x_p as *const f32,
            k_p as *const f32,
            p.alpha_p as *const f32,
            p.blin_p as *const f32,
            p.dt_p as *const f32,
            p.a_p as *const f32,
            p.o_p as *mut f32,
            ptrs_p as *const i64,
            spans_p as *const u32,
            table.n as i32,
            table.max_len as i32,
            p.h_v as i32,
            p.h_k as i32,
            p.conv_dim as i32,
            DELTA_NET_SHORT_SPAN_CONV as i32,
            eps,
            fused.q_scale,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    if status != 0 {
        candle::bail!(
            "delta_net cuda: the short-span scan refused {} spans of up to {} rows over \
             h_v {} / h_k {} / {} channels and wrote nothing",
            table.n,
            table.max_len,
            p.h_v,
            p.h_k,
            p.conv_dim
        );
    }
    Ok(())
}
