//! GPU spans: per-region device timing with no host synchronisation.
//!
//! The host-sync timer this replaced measured GPU time by draining the device at
//! every span boundary, which made each span's number accurate and the run as a
//! whole a fiction: it serialised the pipeline, cost ~20% of bulk throughput on
//! Qwen3.6-35B, and — worst for a launch-bound model — folded the inter-op gaps
//! that ARE the problem into the span totals. A profiler that hides the class of
//! bug you are hunting is worse than none.
//!
//! A GPU span instead brackets the work with two CUDA events recorded INTO the
//! stream. `cuEventRecord` is enqueued like a kernel: the host does not wait, and
//! the ordering that makes the measurement meaningful is the stream's own. The
//! elapsed time between the pair is read later, at [`gpu_drain`], which the
//! caller places on a boundary that already synchronises — so the measurement
//! adds no synchronisation of its own.
//!
//! Both events are held in a per-thread pool and recycled, because a pair is
//! created per span per layer per wave and creating them on the hot path would
//! trade one driver cost for another.

#[cfg(all(feature = "profile", feature = "cuda"))]
use super::pipeline::pipeline_record_duration;

/// A named GPU span, bracketed by two enqueued CUDA events.
///
/// Recorded on [`Self::end`] (or on drop) and accumulated later by
/// [`gpu_drain`]. Zero-sized and free when `profile` or `cuda` is off.
pub struct GpuSpan {
    #[cfg(all(feature = "profile", feature = "cuda"))]
    inner: Option<GpuSpanInner>,
}

#[cfg(all(feature = "profile", feature = "cuda"))]
struct GpuSpanInner {
    name: &'static str,
    /// Index of the borrowed event pair in the pool's `lent` slot list.
    slot: usize,
    stream: std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>,
}

#[cfg(all(feature = "profile", feature = "cuda"))]
mod gpu_pool {
    use candle::cuda_backend::cudarc::driver::{sys, CudaEvent, CudaStream};
    use std::cell::RefCell;
    use std::sync::Arc;

    pub(super) struct Pair {
        pub start: CudaEvent,
        pub stop: CudaEvent,
    }

    pub(super) struct Pending {
        pub name: &'static str,
        pub pair: Pair,
    }

    #[derive(Default)]
    pub(super) struct Pool {
        /// Pairs currently lent to an open span, by slot index.
        pub lent: Vec<Option<Pair>>,
        /// Pairs whose stop event is recorded and awaiting a drain.
        pub pending: Vec<Pending>,
        /// Recycled pairs ready to lend again.
        pub free: Vec<Pair>,
    }

    thread_local! {
        pub(super) static POOL: RefCell<Pool> = RefCell::new(Pool::default());
    }

    /// Above this many un-harvested pairs, opening a span harvests first.
    ///
    /// A pair is only recycled at a drain, and a drain belongs on a boundary that
    /// already synchronises — which for a wave engine is a phase boundary, not a
    /// layer. Without a cap every span between two boundaries allocates a fresh
    /// pair, so a long prefill would create driver events without bound and the
    /// "pool" would never once hand a pair back. The cap is what makes this a
    /// pool rather than an allocator, and it bounds the event count of a
    /// profiling build no matter where the caller chooses to drain.
    pub(super) const HIGH_WATER: usize = 4096;

    /// Number of pairs recorded but not yet harvested.
    pub(super) fn pending_len() -> usize {
        POOL.with(|p| p.borrow().pending.len())
    }

    /// Borrow a pair and record its start event into `stream`.
    ///
    /// Returns the slot index, or `None` if the driver refuses an event — in
    /// which case the span degrades to nothing rather than failing the forward.
    pub(super) fn open(stream: &Arc<CudaStream>) -> Option<usize> {
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            let pair = match p.free.pop() {
                Some(pair) => pair,
                None => {
                    let ctx = stream.context();
                    // CU_EVENT_DEFAULT, not the crate default: timing is the
                    // entire point, and `CU_EVENT_DISABLE_TIMING` would make
                    // `elapsed_ms` fail.
                    let start = ctx
                        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .ok()?;
                    let stop = ctx
                        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                        .ok()?;
                    Pair { start, stop }
                }
            };
            if pair.start.record(stream).is_err() {
                p.free.push(pair);
                return None;
            }
            let slot = p.lent.iter().position(|s| s.is_none()).unwrap_or_else(|| {
                p.lent.push(None);
                p.lent.len() - 1
            });
            p.lent[slot] = Some(pair);
            Some(slot)
        })
    }

    /// Record the stop event and move the pair to the pending list.
    pub(super) fn close(name: &'static str, slot: usize, stream: &Arc<CudaStream>) {
        POOL.with(|p| {
            let mut p = p.borrow_mut();
            let Some(pair) = p.lent.get_mut(slot).and_then(Option::take) else {
                return;
            };
            if pair.stop.record(stream).is_err() {
                p.free.push(pair);
                return;
            }
            p.pending.push(Pending { name, pair });
        });
    }
}

/// Open a GPU span on `device`'s stream. See [`GpuSpan`].
///
/// Costs one enqueued `cuEventRecord`. The host does not wait, so this may be
/// placed anywhere on the hot path.
#[cfg(all(feature = "profile", feature = "cuda"))]
#[inline(always)]
pub fn gpu_span(name: &'static str, device: &candle::Device) -> GpuSpan {
    let candle::Device::Cuda(dev) = device else {
        return GpuSpan { inner: None };
    };
    // Recycle before borrowing. Non-blocking, so a pair whose work is still in
    // flight stays pending and this costs a few `cuEventQuery` calls once every
    // HIGH_WATER spans — not per span, and never a stall.
    if gpu_pool::pending_len() >= gpu_pool::HIGH_WATER {
        drain_inner(false);
    }
    let stream = dev.cuda_stream();
    let inner = gpu_pool::open(&stream).map(|slot| GpuSpanInner { name, slot, stream });
    GpuSpan { inner }
}

/// No-op without both `profile` and `cuda`.
#[cfg(not(all(feature = "profile", feature = "cuda")))]
#[inline(always)]
pub fn gpu_span(_name: &'static str, _device: &candle::Device) -> GpuSpan {
    GpuSpan {}
}

/// Open a GPU span only when `cond` holds — the
/// [`span_if`](super::span_if) analogue.
#[inline(always)]
pub fn gpu_span_if(cond: bool, name: &'static str, device: &candle::Device) -> Option<GpuSpan> {
    if cond {
        Some(gpu_span(name, device))
    } else {
        None
    }
}

/// Open a GPU span whose name depends on the wave phase.
///
/// A host-timed span could pick its name at the *record*, by which time the
/// phase was long known; an event span needs the name at the *open*. Without
/// this every phase-dependent site grew a seven-line `if decode { .. } else
/// { .. }` around the name, which is how a dozen of them looked before it
/// existed.
#[inline(always)]
pub fn gpu_span_phase(
    decode: bool,
    decode_name: &'static str,
    prefill_name: &'static str,
    device: &candle::Device,
) -> GpuSpan {
    gpu_span(if decode { decode_name } else { prefill_name }, device)
}

impl GpuSpan {
    /// Close the span at this exact point rather than at end of scope.
    #[cfg(all(feature = "profile", feature = "cuda"))]
    #[inline(always)]
    pub fn end(self) {
        drop(self);
    }

    /// Consumes the (zero-sized) span and does nothing. Split from the timing
    /// arm because `GpuSpan` only implements `Drop` when there are events to
    /// record, and `drop()` on a non-`Drop` type is a lint, not a no-op.
    #[cfg(not(all(feature = "profile", feature = "cuda")))]
    #[inline(always)]
    pub fn end(self) {}
}

#[cfg(all(feature = "profile", feature = "cuda"))]
impl Drop for GpuSpan {
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            gpu_pool::close(inner.name, inner.slot, &inner.stream);
        }
    }
}

/// Harvest every GPU span whose stop event has completed, accumulating each
/// into the pipeline profiler.
///
/// Non-blocking: a pair whose work is still in flight is left pending and picked
/// up by a later drain, so calling this mid-wave costs a few `cuEventQuery`
/// calls and never stalls. Call [`gpu_drain_blocking`] where the totals must be
/// complete.
#[cfg(all(feature = "profile", feature = "cuda"))]
pub fn gpu_drain() {
    drain_inner(false);
}

/// Drain every pending GPU span, waiting for those still in flight.
///
/// For a boundary that already synchronises — the end of a wave, the end of a
/// benchmark config — where leaving spans unharvested would under-report.
#[cfg(all(feature = "profile", feature = "cuda"))]
pub fn gpu_drain_blocking() {
    drain_inner(true);
}

/// The un-harvested-pair cap, for the test that asserts the pool bounds itself.
#[cfg(all(feature = "profile", feature = "cuda", test))]
pub(super) fn high_water_for_test() -> usize {
    gpu_pool::HIGH_WATER
}

/// Current un-harvested pair count, for the same test.
#[cfg(all(feature = "profile", feature = "cuda", test))]
pub(super) fn pending_len_for_test() -> usize {
    gpu_pool::pending_len()
}

#[cfg(all(feature = "profile", feature = "cuda"))]
fn drain_inner(block: bool) {
    let harvested: Vec<(&'static str, std::time::Duration)> = gpu_pool::POOL.with(|p| {
        let mut p = p.borrow_mut();
        let mut out = Vec::new();
        let mut still_pending = Vec::with_capacity(p.pending.len());
        for entry in std::mem::take(&mut p.pending) {
            if !entry.pair.stop.is_complete() {
                if !block {
                    still_pending.push(entry);
                    continue;
                }
                if entry.pair.stop.synchronize().is_err() {
                    p.free.push(entry.pair);
                    continue;
                }
            }
            // A pair the driver cannot time is dropped rather than recorded as
            // zero: a silent zero would read as "this phase is free", which is
            // the one wrong answer a profiler must not give.
            if let Ok(ms) = entry.pair.start.elapsed_ms(&entry.pair.stop) {
                out.push((
                    entry.name,
                    std::time::Duration::from_secs_f64(ms as f64 / 1e3),
                ));
            }
            p.free.push(entry.pair);
        }
        p.pending = still_pending;
        out
    });
    // Accumulate outside the pool borrow — `pipeline_record_duration` takes its
    // own thread-local borrow, and holding both invites a panic if the two ever
    // become the same cell.
    for (name, elapsed) in harvested {
        pipeline_record_duration(name, elapsed, 1);
    }
}

/// No-op without both `profile` and `cuda`.
#[cfg(not(all(feature = "profile", feature = "cuda")))]
#[inline(always)]
pub fn gpu_drain() {}

/// No-op without both `profile` and `cuda`.
#[cfg(not(all(feature = "profile", feature = "cuda")))]
#[inline(always)]
pub fn gpu_drain_blocking() {}
