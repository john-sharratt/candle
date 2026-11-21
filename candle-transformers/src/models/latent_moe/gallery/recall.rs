//! Exact top-M selection and sign-agreement recall over packed keys, batched
//! across sessions.
//!
//! Recall scores every entry by the XNOR-and-popcount agreement of its packed sign
//! bits with the packed query heads, which has no depth limit. Top-M selection is
//! exact over any entry count and runs on the device. Batched across sessions, each
//! session's sign index is read in place through a pointer table rather than
//! concatenated, so the cost does not grow with depth. A session's shortlist is any
//! M-superset of its top entries, because the float rescore that follows picks the
//! final set.

use super::*;

/// Exact top-`m` entry ids by bounded u32 key (recall shortlist selector) —
/// valid at any entry count, fully on-device. Tie order is arbitrary: any
/// M-superset is a valid recall shortlist (the float rescore re-ranks it).
#[cfg(feature = "cuda")]
pub fn topm_select(counts: &Tensor, m: usize, bins: usize) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let g = counts.dim(0)?;
    let m = m.min(g);
    let dev = match counts.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("topm_select requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let hist = Tensor::zeros(bins, DType::U32, counts.device())?;
    let meta = Tensor::zeros(4, DType::U32, counts.device())?;
    let out = Tensor::zeros(m, DType::U32, counts.device())?;
    {
        let (sc, _) = counts.storage_and_layout();
        let (sh, _) = hist.storage_and_layout();
        let (sm, _) = meta.storage_and_layout();
        let (so, _) = out.storage_and_layout();
        let (cp, _g1) = match &*sc {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (hp, _g2) = match &*sh {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (mp, _g3) = match &*sm {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (op, _g4) = match &*so {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let code = unsafe {
            candle_kernels::simple::bdp::run_topm_select(
                cp as *const u32,
                hp as *mut u32,
                mp as *mut u32,
                op as *mut u32,
                g as i32,
                m as i32,
                bins as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("deepseek_topm_select launch failed: cuda error {code}");
        }
    }
    Ok(out)
}

/// Pack sign bits of `[n, dim]` f32 rows into `[n, ceil(dim/32)]` u32 — the
/// on-device index the recall stage scans.
#[cfg(feature = "cuda")]
pub fn sign_pack(x: &Tensor) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let (n, dim) = x.dims2()?;
    let dev = match x.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("sign_pack requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let x = x.contiguous()?;
    // Every word is written by the kernel's grid-stride loop, so the output
    // is allocated uninitialised (hot-path invariant 6).
    let out = Tensor::empty((n, sign_words(dim)), DType::U32, x.device())?;
    {
        // `contiguous()` hands a dense view back unchanged, offset and all, so
        // the input is addressed from its own first element rather than its
        // storage's.
        let (sx, x_layout) = x.storage_and_layout();
        let x_off = x_layout.start_offset();
        let (so, _) = out.storage_and_layout();
        let x_view = match &*sx {
            Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.slice(x_off..),
            _ => unreachable!(),
        };
        let (xp, _g1) = x_view.device_ptr(&stream);
        let (op, _g2) = match &*so {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let code = unsafe {
            candle_kernels::simple::bdp::run_sign_pack(
                xp as *const f32,
                op as *mut u32,
                n as i32,
                dim as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("deepseek_sign_pack launch failed: cuda error {code}");
        }
    }
    Ok(out)
}

/// Sign-agreement counts `[g]` of every packed entry row against the packed
/// query heads (XNOR+popcount summed over heads and words).
#[cfg(feature = "cuda")]
pub fn bdp_recall(q_signs: &Tensor, signs: &Tensor, dim: usize) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let (h, w1) = q_signs.dims2()?;
    let (g, w2) = signs.dims2()?;
    if w1 != w2 || w1 != sign_words(dim) {
        candle::bail!("bdp_recall word mismatch: q {w1}, entries {w2}, dim {dim}");
    }
    let dev = match signs.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("bdp_recall requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let counts = Tensor::zeros(g, DType::U32, signs.device())?;
    {
        let (sq, _) = q_signs.storage_and_layout();
        let (ss, _) = signs.storage_and_layout();
        let (sc, _) = counts.storage_and_layout();
        let (qp, _g1) = match &*sq {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (sp, _g2) = match &*ss {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let (cp, _g3) = match &*sc {
            Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
            _ => unreachable!(),
        };
        let code = unsafe {
            candle_kernels::simple::bdp::run_bdp_recall(
                qp as *const u32,
                sp as *const u32,
                cp as *mut u32,
                h as i32,
                g as i32,
                w1 as i32,
                dim as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("bdp_recall launch failed: cuda error {code}");
        }
    }
    Ok(counts)
}

/// Batched BDP recall across `n_sess` concurrent decode sessions in ONE launch.
/// `q_signs` `[n_sess·n_heads, words]` are the sessions' packed query heads (row
/// `s·n_heads + h`); `sign_tensors[s]` `[cnt[s], words]` is session `s`'s own
/// resident packed-sign index, **read in place** via a device pointer table (no
/// concatenation — that copy grew `O(Σlen·words)` with context depth). `off`/`cnt`
/// `[n_sess]` u32 give each session's base/count in the concatenated `counts`
/// output. Returns `counts` `[total_g]` — byte-identical per session to the
/// per-session [`bdp_recall`]. `max_g` = the largest `cnt[s]`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn bdp_recall_batched(
    q_signs: &Tensor,
    sign_tensors: &[Tensor],
    off: u64,
    cnt: u64,
    n_sess: usize,
    n_heads: usize,
    max_g: usize,
    dim: usize,
    generation: &Generation,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let words = sign_words(dim);
    let total_g: usize = sign_tensors
        .iter()
        .map(|t| t.dim(0))
        .sum::<Result<usize>>()?;
    let dev = match q_signs.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("bdp_recall_batched requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let counts = Tensor::zeros(total_g.max(1), DType::U32, q_signs.device())?;
    // Per-session base pointer table: each session's signs are read in place (the
    // sign index is always GPU-resident), so no depth-scaling `cat`.
    let mut sign_ptrs: Vec<i64> = Vec::with_capacity(n_sess);
    for t in sign_tensors {
        let (st, layout) = t.storage_and_layout();
        let p = match &*st {
            Storage::Cuda(c) => {
                c.as_cuda_slice::<u32>()?
                    .slice(layout.start_offset()..)
                    .device_ptr(&stream)
                    .0
            }
            _ => candle::bail!("bdp_recall_batched requires CUDA sign storage"),
        };
        sign_ptrs.push(p as i64);
    }
    // The per-gallery sign-pointer table rides the wave arena; it is read once
    // per block, which is what the bump arena is for.
    let sign_ptrs_t = desc::stage(&sign_ptrs, generation)?;
    {
        let (sq, _) = q_signs.storage_and_layout();
        let (sc, _) = counts.storage_and_layout();
        let ptr_u32 = |st: &Storage| -> Result<u64> {
            match st {
                Storage::Cuda(c) => Ok(c.as_cuda_slice::<u32>()?.device_ptr(&stream).0),
                _ => unreachable!(),
            }
        };
        let table = sign_ptrs_t.ptr();
        let (qp, op, cp, outp) = (ptr_u32(&sq)?, off, cnt, ptr_u32(&sc)?);
        let code = unsafe {
            candle_kernels::simple::bdp::run_bdp_recall_batched(
                qp as *const u32,
                table as *const u64,
                op as *const u32,
                cp as *const u32,
                outp as *mut u32,
                n_sess as i32,
                n_heads as i32,
                max_g as i32,
                words as i32,
                dim as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("bdp_recall_batched launch failed: cuda error {code}");
        }
    }
    Ok(counts)
}

/// Batched exact top-`max_m` (per session `min(max_m, cnt[s])`) over per-session
/// count segments in ONE launch per stage. `counts` `[total_g]` concatenated,
/// `off`/`cnt` `[n_sess]` u32. Returns `[n_sess, max_m]` u32 of SESSION-RELATIVE
/// ids (`0..cnt[s]`); session `s`'s first `min(max_m, cnt[s])` columns are its
/// shortlist (remaining columns undefined). Byte-identical per session to
/// [`topm_select`]. `bins` = the agreement range (`n_heads·dim + 1`).
#[cfg(feature = "cuda")]
/// `off`/`cnt` are DEVICE ADDRESSES of `[n_sess]` u32 segment tables, not
/// tensors: the caller stages them once (from the wave arena where it has one)
/// and hands the same pair to this and to [`bdp_recall_batched`], instead of
/// each helper re-uploading its own copy.
#[allow(clippy::too_many_arguments)]
pub fn topm_select_batched(
    counts: &Tensor,
    off: u64,
    cnt: u64,
    n_sess: usize,
    max_g: usize,
    max_m: usize,
    bins: usize,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let dev = match counts.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("topm_select_batched requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let hist = Tensor::zeros(n_sess * bins, DType::U32, counts.device())?;
    let meta = Tensor::zeros(n_sess * 4, DType::U32, counts.device())?;
    let out = Tensor::zeros((n_sess, max_m), DType::U32, counts.device())?;
    {
        let (sc, _) = counts.storage_and_layout();
        let (sh, _) = hist.storage_and_layout();
        let (sm, _) = meta.storage_and_layout();
        let (so, _) = out.storage_and_layout();
        let ptr = |st: &Storage| -> Result<u64> {
            match st {
                Storage::Cuda(c) => Ok(c.as_cuda_slice::<u32>()?.device_ptr(&stream).0),
                _ => unreachable!(),
            }
        };
        let (cp, offp, cntp, hp, mp, outp) = (ptr(&sc)?, off, cnt, ptr(&sh)?, ptr(&sm)?, ptr(&so)?);
        let code = unsafe {
            candle_kernels::simple::bdp::run_topm_select_batched(
                cp as *const u32,
                offp as *const u32,
                cntp as *const u32,
                hp as *mut u32,
                mp as *mut u32,
                outp as *mut u32,
                n_sess as i32,
                max_g as i32,
                max_m as i32,
                bins as i32,
                stream.cu_stream() as *mut core::ffi::c_void,
            )
        };
        if code != 0 {
            candle::bail!("topm_select_batched launch failed: cuda error {code}");
        }
    }
    Ok(out)
}
