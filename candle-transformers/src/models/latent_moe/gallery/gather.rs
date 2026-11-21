//! Gathering every decode session's selected corpus rows in one launch.
//!
//! Each session's selected entries are written at its own row offset in a shared
//! block, so one launch reads the whole wave's corpus through a table of region
//! addresses instead of one launch per session. A spilled gallery's rows are
//! re-heated on the host and scattered in, which is the bounded warm-tier path.
//! Both tables are staged in the wave's arena, because they are read once and
//! never touched again; uploading them per call would cost more than the kernel.

use super::*;

/// Gather EVERY decode session's selected corpus rows into a shared pre-allocated
/// block in ONE kernel launch (across all HOT galleries), instead of a launch per
/// session. `gids[i]` are session `i`'s selected entry ids, placed at output row
/// `row_offsets[i]`. Hot galleries batch through one launch via a device pointer
/// table (each gallery's region base addresses); a spilled gallery re-heats its
/// rows on CPU and `slice_set`s them in (the bounded warm-tier path). Byte-
/// identical per row to [`FloatGallery::gather_corpus_into`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gather_corpus_batched(
    galleries: &[&FloatGallery],
    gids: &[Tensor],
    row_offsets: &[u32],
    out_nope: &Tensor,
    out_scale: &Tensor,
    out_rope: &Tensor,
    out_pos: &Tensor,
    generation: &Generation,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    let n = galleries.len();
    if n == 0 {
        return Ok(());
    }
    // Spilled sessions take the per-session CPU path; collect the hot ones.
    let mut hot: Vec<usize> = Vec::new();
    for i in 0..n {
        if gids[i].dim(0)? == 0 {
            continue;
        }
        let g = galleries[i];
        if g.spilled {
            g.gather_corpus_into(
                &gids[i],
                out_nope,
                out_scale,
                out_rope,
                out_pos,
                row_offsets[i] as usize,
                generation,
            )?;
        } else {
            hot.push(i);
        }
    }
    if hot.is_empty() {
        return Ok(());
    }
    let device = galleries[hot[0]].device.clone();
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("gather_corpus_batched requires CUDA"),
    };
    let stream = dev.cuda_stream();
    macro_rules! dp {
        ($t:expr, $ty:ty) => {{
            let (s, _) = $t.storage_and_layout();
            match &*s {
                Storage::Cuda(c) => c.as_cuda_slice::<$ty>()?.device_ptr(&stream).0,
                _ => candle::bail!("gather_corpus_batched: non-CUDA storage"),
            }
        }};
    }

    // Packed metadata (exactly 2 small uploads): `ptrs` = the device pointer
    // table [nope|scale|rope|pos|gids] (region addresses cached per gallery; the
    // gid address is this step's selection tensor), `meta` = [out_off|cnt]. No
    // per-session `narrow`/lock for regions, no gid `cat`, no per-array upload.
    let n_hot = hot.len();
    let mut ptrs = vec![0i64; 5 * n_hot];
    let mut meta = vec![0u32; 2 * n_hot];
    let mut max_k = 0usize;
    // Keep the (possibly freshly-made-contiguous) gid tensors alive until the
    // launch is enqueued — their addresses are in the pointer table.
    let mut gid_keep: Vec<Tensor> = Vec::with_capacity(n_hot);
    for (col, &i) in hot.iter().enumerate() {
        let g = galleries[i];
        let rp = g.hot_region_ptrs()?;
        ptrs[col] = rp[0] as i64;
        ptrs[n_hot + col] = rp[1] as i64;
        ptrs[2 * n_hot + col] = rp[2] as i64;
        ptrs[3 * n_hot + col] = rp[3] as i64;
        let gk = gids[i].contiguous()?;
        ptrs[4 * n_hot + col] = dp!(gk, u32) as i64;
        meta[col] = row_offsets[i]; // out_off
        meta[n_hot + col] = gk.dim(0)? as u32; // cnt
        max_k = max_k.max(gk.dim(0)?);
        gid_keep.push(gk);
    }
    // Both tables ride the wave arena — they are read once per block and never
    // touched again, which is exactly what the bump arena is for. Uploading them
    // per call cost more than this kernel does (see `desc.rs`).
    let ptrs_t = desc::stage(&ptrs, generation)?;
    let meta_t = desc::stage_slice(&meta, generation)?;

    let code = unsafe {
        candle_kernels::simple::corpus_gather::run_corpus_gather_rows_batched(
            ptrs_t.ptr() as *const i64,
            meta_t.ptr() as *const u32,
            dp!(out_nope, u8) as *mut u32,
            dp!(out_scale, f32) as *mut u32,
            dp!(out_rope, half::bf16) as *mut u32,
            dp!(out_pos, u32) as *mut u32,
            n_hot as i32,
            max_k as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    if code != 0 {
        candle::bail!("corpus_gather_rows_batched launch failed: cuda error {code}");
    }
    Ok(())
}
