//! Scatters glue latents into the gap chunks reserved for them, in one launch
//! per wave.
//!
//! Glue tokens are written into chunks reserved before attention runs, so the
//! scatter must land before any attention pass of the same layer reads those
//! gaps. Each run names its own destination by address, which lets the whole
//! wave's scatter go in a single launch. The kernel addresses rows as one flat
//! block, so a run must be dense: a run that is not is refused, because a
//! transposed or strided run would otherwise scatter garbage without failing.

use super::*;

/// One sequence's scatter: `kv`'s rows into the gap chunks named by `slices`
/// (block index) and `in_blk` (in-block offset), in the slot at `headers_ptr`.
///
/// The two index arrays are HOST vectors, not tensors: they are computed
/// host-side from the reprojection's descriptors and read once by the kernel, so
/// they go straight into the wave's pinned arena. Materialising them as tensors
/// was an allocation and a pageable upload each, per sequence, per layer.
#[cfg(feature = "cuda")]
pub struct GlueRun {
    pub kv: Tensor,
    pub headers_ptr: u64,
    pub slices: Vec<u32>,
    pub in_blk: Vec<u32>,
}

/// Scatter glue latents into their RESERVED gap chunks. Launch BEFORE any
/// attention pass of the same layer on the same stream.
///
/// Batched over runs (invariant 2b): each run names its own destination slot by
/// address, so the wave's whole scatter — every glue island, every prefill
/// writeback, every speculative-verify writeback — is ONE launch. A run with no
/// rows is skipped; an empty batch is a no-op.
#[cfg(feature = "cuda")]
pub fn paged_latent_glue_scatter(runs: &[GlueRun], generation: &Generation) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    // Skip on ROW COUNT, not on the index arrays: a run with rows but no slices
    // is a caller bug, and filtering on `slices` would drop it silently instead
    // of reaching the length check below.
    let live: Vec<&GlueRun> = runs
        .iter()
        .filter(|r| r.kv.dims().first().copied().unwrap_or(0) > 0)
        .collect();
    if live.is_empty() {
        return Ok(());
    }
    let dev = match live[0].kv.device() {
        candle::Device::Cuda(d) => d.clone(),
        _ => candle::bail!("glue scatter requires CUDA tensors"),
    };
    let stream = dev.cuda_stream();

    let mut table = vec![0i64; GLUE_SCATTER_WORDS * live.len()];
    // The per-run index tables are staged individually; hold them so the arena
    // regions stay live until the launch below is enqueued.
    let mut staged_idx = Vec::with_capacity(2 * live.len());
    let mut max_rows = 0usize;
    for (i, run) in live.iter().enumerate() {
        let (rows, width) = run.kv.dims2()?;
        if run.slices.len() != rows || run.in_blk.len() != rows {
            candle::bail!(
                "glue scatter: {rows} latent rows against {} slices / {} offsets",
                run.slices.len(),
                run.in_blk.len()
            );
        }
        // The kernel addresses row `r` as `kv + r*HEAD_DIM`, so a run must be a
        // dense `[rows, HEAD_DIM]` block. A dim-0 narrow of one satisfies this;
        // a transpose or a dim-1 narrow does not, and would scatter garbage
        // into the arena rather than fail. Cheap to check, and the single-run
        // API this replaced only had one caller to reason about.
        if width != HEAD_DIM || run.kv.stride()[0] != HEAD_DIM || run.kv.stride()[1] != 1 {
            candle::bail!(
                "glue scatter: run {i} is [{rows}, {width}] stride {:?}, not a dense \
                 [rows, {HEAD_DIM}] block",
                run.kv.stride()
            );
        }
        let kv_p = {
            let (storage, layout) = run.kv.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let slice = c.as_cuda_slice::<half::bf16>()?;
                    let (p, _guard) = slice.device_ptr(&stream);
                    p + (layout.start_offset() * std::mem::size_of::<half::bf16>()) as u64
                }
                _ => candle::bail!("expected CUDA storage for glue latents"),
            }
        };
        let sl = desc::stage_slice(&run.slices, generation)?;
        let ib = desc::stage_slice(&run.in_blk, generation)?;
        let w = &mut table[i * GLUE_SCATTER_WORDS..][..GLUE_SCATTER_WORDS];
        w[0] = kv_p as i64;
        w[1] = run.headers_ptr as i64;
        w[2] = sl.ptr() as i64;
        w[3] = ib.ptr() as i64;
        w[4] = rows as i64;
        staged_idx.push(sl);
        staged_idx.push(ib);
        max_rows = max_rows.max(rows);
    }
    let staged = desc::stage(&table, generation)?;
    unsafe {
        candle_kernels::paged_latent::run_paged_latent_glue_scatter_bf16(
            staged.ptr() as *const i64,
            live.len() as i32,
            max_rows as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(())
}
