//! Stages selection tables and inputs on the device through the least-costly path
//! the caller allows.
//!
//! A caller's pinned, device-mapped generation is used when given, since the kernel
//! then reads over the bus with no device copy. Otherwise the bytes go into the
//! persistence domain's transient span with one bump and one host-to-device copy,
//! and when there is no stream to bump against, a plain device allocation is made.
//! The span path returns a guard that keeps the span open for as long as the
//! buffers borrow it. Tables that are not bounded by a guard are owned outright,
//! because an unguarded range could be reset by another thread and handed out
//! again while the kernel is still reading it. An empty job list still uploads one
//! zeroed row, so the launch has a valid pointer.

use super::*;

/// Stage `bytes` where a kernel can read them.
///
/// Three paths, in preference order:
///
/// 1. A caller-supplied [`Generation`] — the host-pinned, device-mapped arena.
///    The GPU reads over PCIe with no device copy at all, so this stays the
///    best option when the caller has one.
/// 2. The **persistence domain's transient span** — one bump and an H2D. This
///    replaced `memcpy_stod`, which allocated and freed a device buffer per
///    selection table, per call.
/// 3. A plain device allocation, when there is no stream to bump against
///    (CPU-backed tests).
///
/// The returned `GpuBuf` borrows in case 2: the span outlives it, and the
/// caller's generation guard is what keeps the cursor from rewinding
/// underneath it.
/// The transient-span guard a freshly-built input set must hold.
///
/// `Some` exactly when [`stage_bytes_as_gpu_buf`] took the bump path — i.e.
/// when the caller supplied no pinned generation — because that is the only
/// case where the returned `GpuBuf`s borrow the span.
pub(super) fn bump_generation(
    generation: Option<&Generation>,
    dev: &candle::CudaDevice,
) -> Option<super::super::super::bump_arena::Generation> {
    if generation.is_some() {
        return None;
    }
    // `0` planned span: this staging is sized per call rather than by the wave
    // plan, so the whole domain stays cursor-managed.
    super::super::super::bump_arena::persistence_domain(&dev.cuda_stream())
        .and_then(|b| b.generation(NOT_A_WAVE, NOT_A_WAVE))
        .ok()
}

pub(super) fn stage_bytes_as_gpu_buf(
    bytes: &[u8],
    generation: Option<&Generation>,
    bump: Option<&super::super::super::bump_arena::Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    if let Some(generation) = generation {
        let mut pinned = generation.alloc(bytes.len())?;
        pinned.as_mut_slice().copy_from_slice(bytes);
        return generation.submit(pinned);
    }
    let Some(bump) = bump else {
        // No guard to bound a span range, so own the bytes outright rather than
        // bump: an unguarded range is one another thread's last-generation drop
        // can reset and re-hand-out under this kernel. Owned costs an
        // allocation, which these tables can afford — they are small and
        // uploaded once per selector.
        let slice = dev.memcpy_stod(bytes)?;
        return Ok(GpuBuf::from_raw_owned(slice, dev));
    };
    // Allocated *through* the guard, so the range provably belongs to a
    // generation that is still open — `BumpRange<'_>` borrows `bump`. The
    // caller keeps `bump` alive for as long as the returned `GpuBuf`, and the
    // upload is ordered behind everything issued before it.
    let range = bump.alloc(bytes.len(), 256)?;
    dev.upload_raw(range.ptr, bytes)?;
    Ok(GpuBuf::from_borrowed(range.ptr, range.len))
}

pub(super) fn stage_i64_slice(
    values: &[i64],
    generation: Option<&Generation>,
    bump: Option<&super::super::super::bump_arena::Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    stage_bytes_as_gpu_buf(bytes, generation, bump, dev)
}

pub(super) fn stage_i32_slice(
    values: &[i32],
    generation: Option<&Generation>,
    bump: Option<&super::super::super::bump_arena::Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    stage_bytes_as_gpu_buf(bytes, generation, bump, dev)
}

pub(super) fn stage_f32_slice(
    values: &[f32],
    generation: Option<&Generation>,
    bump: Option<&super::super::super::bump_arena::Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    stage_bytes_as_gpu_buf(bytes, generation, bump, dev)
}

/// Upload a selection table to the device as an owned buffer.
///
/// Owned, never the pinned-stager arena: the selection kernel reads the table
/// asynchronously, and an arena-backed `GpuBuf` is valid only while its staging
/// generation chain lives — a last-generation drop on another thread syncs one
/// stream and resets the arena, recycling these bytes under the in-flight
/// kernel. The table is small; the owned slice is freed stream-ordered after
/// the kernel. Same reasoning as the `head_gids` upload beside it.
///
/// An empty job list still needs a non-null pointer for the launch, so it
/// uploads one zeroed row.
pub(super) fn stage_table(
    table: &[i64],
    bump: Option<&super::super::super::bump_arena::Generation>,
    dev: &candle::CudaDevice,
) -> Result<GpuBuf> {
    if table.is_empty() {
        return stage_i64_slice(&vec![0i64; PerHeadEntry::COLS], None, bump, dev);
    }
    stage_i64_slice(table, None, bump, dev)
}
