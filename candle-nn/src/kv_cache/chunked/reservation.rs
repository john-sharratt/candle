//! The span's memory: one device allocation, claimed once, never moved.
//!
//! Everything above this file assumes one contiguous span whose address never
//! changes. It is a single `cuMemAlloc`, made once when the region pool is
//! created and freed only when the pool goes. It needs nothing beyond plain
//! device allocation, so it runs under every driver model — WDDM, TCC and Linux
//! alike (`docs/driver_model.md` §4); Windows' TCC driver offers no virtual
//! memory management, which an address-space reservation would need.
//!
//! # Capacity is the allocation that succeeds
//!
//! The span asks for what it wants, capped at the free memory the driver reports
//! less [`FREE_MARGIN`] for the driver's own allocations, and steps down on a
//! refusal until one is granted ([`step_down`]). Under TCC and Linux the driver
//! refuses honestly. Under WDDM an allocation past the OS's residency budget is
//! granted and demoted instead, so the cap — and the governor that sized what is
//! asked for from DXGI's budget — is what keeps the span resident there.
//!
//! # A guard granule in front
//!
//! The allocation's first [`GRANULARITY`] bytes are never handed out: the span's
//! base is one granule in. A slice viewing span memory that reached the driver's
//! free would free the whole span if it held the allocation's base, and nothing
//! can hold that address now — an interior pointer is refused by the driver.
//!
//! # Zeroed on claim
//!
//! Each range is zero-filled as it is claimed ([`Reservation::map_range`]),
//! which the region tier relies on: `RegionPool` hands a region whose
//! `dirty_epoch` is still zero to its first tenant *without* cleaning it, on the
//! strength of this write. The fill is queued rather than awaited; its one hard
//! rule is that the span must not be freed while it is still in flight, which
//! [`Reservation::fence`] states and the teardown obeys.

use std::sync::Arc;

use candle::cuda_backend::cudarc::driver::result::{
    free_sync, malloc_sync, mem_get_info, memset_d8_async, DriverError,
};
use candle::cuda_backend::cudarc::driver::sys::{CUdeviceptr, CUresult};
use candle::cuda_backend::cudarc::driver::{CudaContext, CudaStream};
use candle::cuda_backend::graph::without_recording;
use candle::Result;

/// The unit every span size is a whole number of, and the guard in front of it.
pub(crate) const GRANULARITY: usize = 2 << 20;

/// Free memory the span leaves to the driver: its own allocations — module
/// loads, a context's local memory, a library's workspace — are taken after the
/// span is claimed and would otherwise find the card full.
const FREE_MARGIN: usize = 256 << 20;

/// The most of `want` bytes the span may ask for with `free` bytes free on the
/// card: [`FREE_MARGIN`] and the guard granule left over, in whole granules.
fn granted_at_most(want: usize, free: usize) -> usize {
    let room = free.saturating_sub(FREE_MARGIN + GRANULARITY);
    (want.min(room) / GRANULARITY) * GRANULARITY
}

/// The next size to ask for after `bytes` was refused: a thirty-second less,
/// at least one granule, in whole granules — `0` when nothing smaller is left.
fn step_down(bytes: usize) -> usize {
    let less = (bytes / 32).max(GRANULARITY);
    (bytes.saturating_sub(less) / GRANULARITY) * GRANULARITY
}

/// One contiguous device allocation and the span it carries.
///
/// Dropping it frees the allocation. Nothing hands out sub-ranges here — carving
/// is the region layer's job; this owns only the memory.
pub(crate) struct Reservation {
    /// The allocation the driver returned: the guard granule's first byte.
    alloc_base: CUdeviceptr,
    /// The span's first byte, one granule past `alloc_base`.
    base: CUdeviceptr,
    /// Bytes of span, past the guard.
    reserved: usize,
    /// Bytes from `base` claimed and zero-filled so far.
    mapped: usize,
    context: Arc<CudaContext>,
    /// The device's stream, which the zero-fill is queued on.
    stream: Arc<CudaStream>,
}

impl Reservation {
    /// Allocate a span of up to `bytes`, rounded down to whole granules, on
    /// `stream`'s device — the most of it the card grants. Claims nothing yet:
    /// [`Self::map_range`] zero-fills what the span hands out.
    pub(crate) fn reserve(stream: &Arc<CudaStream>, bytes: usize) -> Result<Self> {
        let context = stream.context().clone();
        context
            .bind_to_thread()
            .map_err(|e| candle::Error::Msg(format!("reservation: bind: {e}")))?;
        let (free, _) =
            mem_get_info().map_err(|e| candle::Error::Msg(format!("cuMemGetInfo: {e}")))?;
        let mut size = granted_at_most(bytes, free);
        loop {
            if size == 0 {
                candle::bail!(
                    "reservation: the card granted no span of the {bytes} B asked for \
                     ({free} B free, {FREE_MARGIN} B left to the driver)"
                );
            }
            // The guard granule rides in front of the span (see the module
            // header). Straight to the driver: this is the span itself.
            // SAFETY: a plain device allocation, freed in `drop`.
            match unsafe { malloc_sync(size + GRANULARITY) } {
                Ok(alloc_base) => {
                    return Ok(Self {
                        alloc_base,
                        base: alloc_base + GRANULARITY as u64,
                        reserved: size,
                        mapped: 0,
                        context,
                        stream: stream.clone(),
                    })
                }
                Err(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY)) => size = step_down(size),
                Err(e) => candle::bail!("reservation: cuMemAlloc of {size} B failed: {e}"),
            }
        }
    }

    /// Device address of the span's first byte.
    pub(crate) fn base(&self) -> u64 {
        self.base
    }

    /// Bytes of span the allocation holds (a whole number of granules).
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.reserved
    }

    /// The unit every span size is a whole number of.
    pub(crate) fn granularity(&self) -> usize {
        GRANULARITY
    }

    /// Bytes from the span's base claimed so far.
    #[cfg(test)]
    pub(crate) fn mapped_bytes(&self) -> usize {
        self.mapped
    }

    /// Claim `[offset, offset + bytes)` of the span — rounded up to whole
    /// granules and clipped at its end — zero-filling it. Returns the bytes of
    /// the range claimed: the request, or less where it ran past the span.
    pub(crate) fn map_range(&mut self, offset: usize, bytes: usize) -> Result<usize> {
        let start = (offset / GRANULARITY) * GRANULARITY;
        let end = (offset + bytes)
            .div_ceil(GRANULARITY)
            .saturating_mul(GRANULARITY)
            .min(self.reserved);
        if end <= start {
            return Ok(0);
        }
        let len = end - start;
        self.context
            .bind_to_thread()
            .map_err(|e| candle::Error::Msg(format!("bind_to_thread: {e}")))?;
        let stream = self.stream.cu_stream();
        let at = self.base + start as u64;
        // Queued with no capture in flight (`without_recording`), which a span
        // being claimed never waits on: it is claimed at creation, before the
        // first forward.
        // SAFETY: `[at, at + len)` lies inside this allocation.
        without_recording(|_| unsafe { memset_d8_async(at, 0, len, stream) })?
            .map_err(|e| candle::Error::Msg(format!("reservation: zero-fill: {e}")))?;
        self.mapped = self.mapped.max(end);
        Ok(len)
    }

    /// Wait for every write this span has queued, before it is freed.
    ///
    /// **Through the record gate, waiting.** A context-wide synchronise
    /// invalidates any capture in flight, so it waits for none to be recording.
    /// Waiting is safe here because the one caller is the drop at the end of
    /// the process. Errors are swallowed on purpose — this runs from `Drop`, and
    /// a context already faulted has a real failure being reported elsewhere.
    fn fence(&self) {
        let _ = without_recording(|quiet| quiet.synchronize_context(&self.context));
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // Nothing may still be writing into the span about to be freed.
        self.fence();
        // SAFETY: the allocation `reserve` made, freed once.
        let _ = unsafe { free_sync(self.alloc_base) };
    }
}

#[cfg(test)]
mod sizing_tests {
    use super::{granted_at_most, step_down, FREE_MARGIN, GRANULARITY};

    const MIB: usize = 1 << 20;

    /// **A span asks for at most the card's free memory, less the driver's
    /// margin and the guard granule, in whole granules.** 1 GiB free leaves
    /// 1024 − 256 − 2 = 766 MiB, granule-aligned to 766; asking for less gets
    /// what was asked, rounded down.
    #[test]
    fn a_span_asks_for_the_free_memory_less_the_margin_and_the_guard() {
        assert_eq!(granted_at_most(usize::MAX, 1024 * MIB), 766 * MIB);
        assert_eq!(granted_at_most(100 * MIB + 1, 1024 * MIB), 100 * MIB);
        assert_eq!(granted_at_most(usize::MAX, FREE_MARGIN + GRANULARITY), 0);
        assert_eq!(granted_at_most(usize::MAX, 0), 0);
    }

    /// **A refusal steps down by a thirty-second, at least a granule**, to
    /// nothing once a granule is refused.
    #[test]
    fn a_refused_size_steps_down_a_thirty_second_at_least_a_granule() {
        assert_eq!(step_down(64 * 1024 * MIB), 62 * 1024 * MIB);
        assert_eq!(step_down(32 * MIB), 30 * MIB);
        assert_eq!(step_down(2 * MIB), 0);
        assert_eq!(step_down(0), 0);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::{Reservation, GRANULARITY};
    use candle::cuda_backend::cudarc::driver::result::{memcpy_dtoh_async, memcpy_htod_async};
    use candle::cuda_backend::StreamLease;
    use candle::{Device, Result};

    use crate::kv_cache::chunked::gpu_test_lock::{gpu_serial, GpuGuard};

    /// A CUDA stream **and** the crate-wide GPU serialisation guard, returned
    /// together so the lock cannot be forgotten: these tests allocate device
    /// memory, which every `ChunkedKvBacking` draws on too.
    fn stream() -> Option<(StreamLease, GpuGuard)> {
        let guard = gpu_serial();
        match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => Some((d.cuda_stream(), guard)),
            _ => None,
        }
    }

    /// Sizes are whole granules, and the span starts one guard granule into
    /// its allocation.
    #[test]
    fn a_span_is_whole_granules_behind_a_guard() -> Result<()> {
        let Some((s, _gpu)) = stream() else {
            return Ok(());
        };
        let r = Reservation::reserve(&s, 3 * GRANULARITY + 1)?;
        assert_eq!(r.reserved_bytes(), 3 * GRANULARITY);
        assert_eq!(r.base(), r.alloc_base + GRANULARITY as u64);
        assert_eq!(r.mapped_bytes(), 0, "reserving claims nothing");
        Ok(())
    }

    /// Claimed memory is usable at the span's address and comes back zeroed.
    #[test]
    fn claimed_memory_is_zeroed_and_usable_at_the_span_address() -> Result<()> {
        let Some((s, _gpu)) = stream() else {
            return Ok(());
        };
        let mut r = Reservation::reserve(&s, 4 * GRANULARITY)?;
        assert_eq!(r.map_range(0, GRANULARITY)?, GRANULARITY);
        let mut zero = vec![0xFFu8; 4096];
        // SAFETY: the first granule is claimed and 4096 is well inside it.
        unsafe { memcpy_dtoh_async(&mut zero, r.base(), s.cu_stream()) }
            .map_err(|e| candle::Error::Msg(format!("dtoh: {e}")))?;
        let pattern = vec![0xA5u8; 4096];
        // SAFETY: as above.
        unsafe { memcpy_htod_async(r.base(), &pattern, s.cu_stream()) }
            .map_err(|e| candle::Error::Msg(format!("htod: {e}")))?;
        let mut back = vec![0u8; 4096];
        // SAFETY: as above, reading the range just written.
        unsafe { memcpy_dtoh_async(&mut back, r.base(), s.cu_stream()) }
            .map_err(|e| candle::Error::Msg(format!("dtoh: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        assert_eq!(zero, vec![0u8; 4096], "a claimed range must be zero");
        assert_eq!(back, pattern, "the span did not hold what was written");
        Ok(())
    }

    /// A claim stops at the span's end, and a range claimed twice is claimed.
    #[test]
    fn a_claim_stops_at_the_span_end() -> Result<()> {
        let Some((s, _gpu)) = stream() else {
            return Ok(());
        };
        let mut r = Reservation::reserve(&s, 2 * GRANULARITY)?;
        assert_eq!(r.map_range(GRANULARITY, 4 * GRANULARITY)?, GRANULARITY);
        assert_eq!(r.map_range(4 * GRANULARITY, GRANULARITY)?, 0);
        assert_eq!(r.mapped_bytes(), 2 * GRANULARITY);
        Ok(())
    }
}
