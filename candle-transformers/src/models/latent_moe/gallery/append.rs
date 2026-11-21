//! Appending a wave's completed groups to every session's gallery in one pass.
//!
//! A wave's pooled block already holds each session's groups as a contiguous run,
//! so the append is a scatter with one destination per session. The wave signs and
//! positions its new keys once, seals them into the two-region cache in one
//! launch, and settles every gallery's capacity and tier before any row is
//! written, so each row is written once to its final home. A warm-tier gallery
//! keeps its float pair in RAM, which the scatter kernel cannot reach, so those
//! four arrays take a device-to-host copy each. The two-region cache built here is
//! the stored latent itself.

use super::*;

/// Build the two-region position-free cache (`latent_build_corpus_cache_kernel`)
/// for `n` rows of `attn_gpu` `[n, HEAD_DIM]` f32 into the gallery buffers at
/// `[row_lo, row_lo+n)`. The output views carry the row offset in their device
/// pointer, so the kernel (gid `0..n`, reading `attn_gpu[gid]`) writes exactly
/// those rows.
#[cfg(feature = "cuda")]
/// Append the raw bytes of `ids`' rows of a CPU-resident `[rows, w]` tensor to
/// `out` — the host-side row gather for the warm tier, straight into the packed
/// staging buffer so the re-heat never materialises a per-region tensor.
///
/// `live` bounds the gather to the tensor's occupied prefix, matching the
/// `narrow(0, 0, live)` the device path applies.
#[cfg(feature = "cuda")]
pub(super) fn pack_rows(
    t: &Tensor,
    ids: &[u32],
    live: usize,
    row_bytes: usize,
    out: &mut Vec<u8>,
) -> Result<()> {
    use candle::{CpuStorage, Storage};
    let (storage, layout) = t.storage_and_layout();
    let cols = t.dim(1)?;
    let stride = layout.stride()[0];
    let base = layout.start_offset();
    // Row `id` of a CPU tensor as bytes, in the tensor's own element type.
    macro_rules! rows {
        ($v:expr, $to_bytes:expr) => {{
            let data = $v;
            for &id in ids {
                let id = id as usize;
                if id >= live {
                    candle::bail!("pack_rows: entry {id} past the live prefix ({live})");
                }
                let row = &data[base + id * stride..][..cols];
                for x in row {
                    out.extend_from_slice(&$to_bytes(*x));
                }
            }
        }};
    }
    let before = out.len();
    match &*storage {
        Storage::Cpu(CpuStorage::U8(v)) => {
            for &id in ids {
                let id = id as usize;
                if id >= live {
                    candle::bail!("pack_rows: entry {id} past the live prefix ({live})");
                }
                out.extend_from_slice(&v[base + id * stride..][..cols]);
            }
        }
        Storage::Cpu(CpuStorage::F32(v)) => rows!(v, |x: f32| x.to_ne_bytes()),
        Storage::Cpu(CpuStorage::BF16(v)) => rows!(v, |x: half::bf16| x.to_bits().to_ne_bytes()),
        Storage::Cpu(CpuStorage::U32(v)) => rows!(v, |x: u32| x.to_ne_bytes()),
        _ => candle::bail!(
            "pack_rows: unsupported warm-tier element type {:?}",
            t.dtype()
        ),
    }
    debug_assert_eq!(out.len() - before, ids.len() * row_bytes);
    Ok(())
}

/// One session's share of a wave's fleet-wide pooled block, for
/// [`append_batch_all`]: the gallery to append to, and which rows of the block
/// are its groups.
#[cfg(feature = "cuda")]
pub struct AppendSlice<'a> {
    pub gallery: &'a mut FloatGallery,
    /// `(first row of the block, rows)` runs, in the order they must land.
    ///
    /// Usually one. A SPECULATIVE VERIFY wave carries a whole drafted block as
    /// several consecutive decode rows of the SAME sequence, so that sequence
    /// occupies several wave slots and closes a group in more than one of them;
    /// every such run appends to this one gallery, back to back, exactly as the
    /// per-slot loop did.
    pub runs: Vec<(usize, usize)>,
}

#[cfg(feature = "cuda")]
impl AppendSlice<'_> {
    pub(super) fn rows(&self) -> usize {
        self.runs.iter().map(|(_, n)| n).sum()
    }
}

/// Append EVERY session's completed groups in one pass over the wave.
///
/// A wave's compressor pool already emits one fleet-wide `[ΣG, d]` block with
/// each session's groups as a contiguous run of it, so the append is a scatter:
/// six arrays × one destination per session. Done per session that is six
/// `slice_set` launches times the wave's width times the compression layers,
/// each moving a few kilobytes — pure launch overhead, and exactly the
/// allocate-plus-copy-per-consumer shape hot-path invariant 2 forbids.
///
/// Here it is a fixed handful of launches for the WHOLE wave, independent of
/// session count: one sign-pack, one position upload, one latent seal, and one
/// [`rows_scatter`] whose descriptor table names every destination by address.
///
/// Each gallery's tier is settled BEFORE anything is written (`grow_to` then
/// `maybe_spill` over the post-append length), so every row is written once, to
/// its final home. That is the same bytes as building hot and spilling after —
/// a spill moves whole regions and does not care whether the new rows are in
/// them yet — and it removes the need to scatter twice around the crossing.
#[cfg(feature = "cuda")]
pub fn append_batch_all(
    slices: &mut [AppendSlice<'_>],
    attn_rows: &Tensor,
    key_rows: &Tensor,
    positions: &[u32],
    generation: &Generation,
) -> Result<()> {
    let (total, hd) = attn_rows.dims2()?;
    if total == 0 || slices.is_empty() {
        return Ok(());
    }
    // Every gallery of a wave shares one device; the seal and the scatter both
    // run there, so the incoming rows are placed on it once (a no-op when they
    // already are, which is the case for the wave's own pooled block).
    let device = slices[0].gallery.device.clone();
    let attn_rows = &attn_rows.to_device(&device)?;
    let key_rows = &key_rows.to_device(&device)?;
    let (kn, kd) = key_rows.dims2()?;
    if kn != total || positions.len() != total {
        candle::bail!(
            "append_batch_all: attn {:?}, keys {:?}, pos {} disagree on the row count",
            attn_rows.dims(),
            key_rows.dims(),
            positions.len()
        );
    }
    let covered: usize = slices.iter().map(|s| s.rows()).sum();
    if covered != total {
        candle::bail!("append_batch_all: slices cover {covered} rows of a {total}-row block");
    }
    for s in slices.iter() {
        for &(src_row, rows) in &s.runs {
            if src_row + rows > total {
                candle::bail!(
                    "append_batch_all: slice rows {src_row}..{} of a {total}-row block",
                    src_row + rows
                );
            }
        }
        if hd != s.gallery.head_dim || kd != s.gallery.index_head_dim {
            candle::bail!(
                "append_batch_all: attn {:?} / keys {:?} do not match the gallery \
                 (head_dim {}, index_head_dim {})",
                attn_rows.dims(),
                key_rows.dims(),
                s.gallery.head_dim,
                s.gallery.index_head_dim
            );
        }
    }

    // Sign-pack the new keys and upload their positions ONCE for the whole wave.
    // Both indexes are permanently GPU-resident (only the float pair spills), so
    // neither depends on any gallery's tier.
    let all_signs = sign_pack(key_rows)?;
    let all_pos = Tensor::from_vec(positions.to_vec(), total, &device)?;

    // ONE seal over the whole block. The two-region cache it builds IS the
    // stored latent — there is no separate f32 archive (the old canonical-f32
    // CPU copy was reference-only and its blocking D2H drained the pipeline on
    // every append). The kernel writes every row, so the scratch is
    // uninitialised (hot-path invariant 6).
    let all_nope = Tensor::empty((total, NOPE_DIM), DType::U8, &device)?;
    let all_scale = Tensor::empty((total, NOPE_BANDS), DType::F32, &device)?;
    let all_rope = Tensor::empty((total, ROPE_DIM), DType::BF16, &device)?;
    build_corpus_cache_into(attn_rows, &all_nope, &all_scale, &all_rope, 0, total)?;

    // Settle every gallery's capacity and tier before writing a single row.
    for s in slices.iter_mut() {
        let end = s.gallery.len + s.rows();
        s.gallery.grow_to(end)?;
        s.gallery.maybe_spill(end)?;
    }

    let mut runs: Vec<RowRun> = Vec::with_capacity(6 * slices.len());
    for s in slices.iter() {
        let g = &*s.gallery;
        let mut dst = g.len;
        for &(src_row, rows) in &s.runs {
            let take = |t: &Tensor| -> Result<Tensor> { t.narrow(0, src_row, rows) };
            runs.push(RowRun::new(take(&all_pos)?, &g.pos, dst));
            runs.push(RowRun::new(take(&all_signs)?, &g.signs, dst));
            if g.spilled {
                // Warm tier: the float pair lives in RAM, which the scatter
                // kernel cannot reach, so those four arrays take a
                // device-to-host copy each.
                let cpu = |t: &Tensor| -> Result<Tensor> { take(t)?.to_device(&Device::Cpu) };
                g.keys.slice_set(&cpu(key_rows)?, 0, dst)?;
                g.nope_i8.slice_set(&cpu(&all_nope)?, 0, dst)?;
                g.nope_scale.slice_set(&cpu(&all_scale)?, 0, dst)?;
                g.rope_bf.slice_set(&cpu(&all_rope)?, 0, dst)?;
            } else {
                runs.push(RowRun::new(take(key_rows)?, &g.keys, dst));
                runs.push(RowRun::new(take(&all_nope)?, &g.nope_i8, dst));
                runs.push(RowRun::new(take(&all_scale)?, &g.nope_scale, dst));
                runs.push(RowRun::new(take(&all_rope)?, &g.rope_bf, dst));
            }
            dst += rows;
        }
    }
    rows_scatter(&runs, generation)?;

    for s in slices.iter_mut() {
        s.gallery.len += s.rows();
    }
    Ok(())
}

pub(super) fn build_corpus_cache_into(
    attn_gpu: &Tensor,
    nope_i8: &Tensor,
    nope_scale: &Tensor,
    rope_bf: &Tensor,
    row_lo: usize,
    n: usize,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle::Storage;
    if n == 0 {
        return Ok(());
    }
    let dev = match attn_gpu.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("build_corpus_cache_into requires CUDA"),
    };
    let stream = dev.cuda_stream();
    let ni8 = nope_i8.narrow(0, row_lo, n)?;
    let nsc = nope_scale.narrow(0, row_lo, n)?;
    let rbf = rope_bf.narrow(0, row_lo, n)?;
    macro_rules! p {
        ($t:expr, $ty:ty) => {{
            let (storage, layout) = $t.storage_and_layout();
            match &*storage {
                Storage::Cuda(c) => {
                    let (ptr, _g) = c.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                    ptr + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
                }
                _ => candle::bail!("expected CUDA storage"),
            }
        }};
    }
    let comp_p = p!(attn_gpu, f32);
    let ni8_p = p!(&ni8, u8);
    let nsc_p = p!(&nsc, f32);
    let rbf_p = p!(&rbf, half::bf16);
    unsafe {
        candle_kernels::paged_latent::run_latent_build_corpus_cache(
            comp_p as *const f32,
            ni8_p as *mut u8,
            nsc_p as *mut f32,
            rbf_p as *mut core::ffi::c_void,
            0,
            n as i32,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
    Ok(())
}
