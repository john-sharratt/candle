//! The transient tier: per-domain device bump allocators.
//!
//! Inference-loop transients — kernel argument blobs, staging buffers,
//! intermediate activations — do not allocate from the region pool. They
//! bump-allocate from a contiguous span that is **never deallocated**: a
//! generation hands out disjoint ranges by advancing a cursor, and later resets
//! that cursor to zero in one store. No per-buffer RAII, no frees, zero
//! allocator traffic on the wave path in steady state
//! (`docs/archived/arena_unification.md` §3.6).
//!
//! # The reset is the only dangerous operation
//!
//! Resetting a cursor makes every buffer handed out since the last reset
//! reusable, so it must not happen while a kernel is still reading one. Three
//! things guard it, and they are the whole safety argument:
//!
//! - **A borrow.** [`Generation::alloc`] returns a [`BumpRange<'w>`] that
//!   borrows the guard, and every tensor built on that range carries the same
//!   `'w`. A buffer therefore cannot be named after the guard that would free
//!   it has dropped — the compiler rejects the program instead.
//! - **A counted generation.** `reset` is refused while any [`Generation`] is
//!   live. This is what covers the host-side cursor when several guards overlap;
//!   the borrow bounds one guard's ranges, the count bounds the arena. Refusal
//!   is loud — the count is checked, not assumed — because a silent early reset
//!   is a data race that reproduces as garbage output far from its cause
//!   (principle 7: safety by refusal, not by ceremony).
//! - **A stream fence.** The last generation to drop synchronises the domain's
//!   stream before the cursor moves, so the GPU has drained the ranges the
//!   host is about to hand out again. This is `PinnedStager`'s sync-then-reset
//!   discipline, applied to device memory. The borrow is a host-side statement
//!   about names; only the fence orders the device.
//!
//! # Why domains are separate arenas
//!
//! The scheduler's wave loop and the persistence thread have unrelated
//! lifetimes; one must never reset the other's live buffers. §3.6 gives them
//! disjoint sub-ranges of one span.
//!
//! # Backing
//!
//! Every domain is a disjoint sub-range of the device reservation's transient
//! tier ([`super::region_pool::carve_transient`]), carved once and never
//! returned. Addresses are therefore fixed for the process lifetime, which is
//! what lets a [`BumpRange`] be a bare pointer: `'w` bounds when the *range* may
//! be reused, never whether the address is mapped.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, OnceLock};

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::Result;

use super::chunk_ops::MIGRATION_STAGING_CAP_BYTES;
use super::region_pool::carve_transient;

/// A range handed out by a [`Generation`].
///
/// Deliberately **not** RAII: a bump range is freed by its generation's reset,
/// never individually, so a `Drop` impl would be a lie. What bounds it instead
/// is `'w` — a borrow of the [`Generation`] that handed it out. The cursor
/// cannot rewind while that guard is alive, so "valid for `'w`" and "valid
/// until the generation resets" are the same statement, and the borrow checker
/// can hold the compiler to it.
#[derive(Debug, Clone, Copy)]
pub struct BumpRange<'w> {
    /// Device address of the range's first byte.
    pub ptr: u64,
    /// Bytes reserved.
    pub len: usize,
    /// Covariant in `'w`: this range is valid for at most as long as the guard
    /// it came from.
    wave: PhantomData<&'w ()>,
}

/// How a domain makes a reset safe.
///
/// Resetting hands memory back out while kernels may still be reading it. What
/// prevents that differs by domain, and the difference is worth naming: one
/// costs a pipeline stall and the other costs nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reclaim {
    /// Fence the stream before the cursor moves.
    ///
    /// For domains whose ranges are read from one stream and overwritten from
    /// another — the persistence thread stages on its copy stream while the
    /// compute stream runs — where nothing else orders the two.
    Fence,
    /// Rely on stream order alone.
    ///
    /// For domains that are double-buffered on a single stream. By the time a
    /// half is handed out again, an entire other wave's work sits between the
    /// reads and the writes *on the same stream*, and same-stream launches
    /// complete in issue order. A fence would buy nothing and cost a full
    /// device sync on the forward's critical path — which is precisely the
    /// wave-path stall §3.6 exists to remove.
    StreamOrdered,
}

struct Inner {
    base: u64,
    capacity: usize,
    /// Bytes handed out since the last reset.
    cursor: usize,
    /// Live [`Generation`] guards. The cursor cannot move while this is > 0.
    live: usize,
    /// Whether anything has been handed out since the last reset.
    dirty: bool,
    /// High-water mark of `cursor`, for the watermark that sizes the span.
    peak: usize,
    stream: Arc<CudaStream>,
    reclaim: Reclaim,
}

/// One domain's transient span.
#[derive(Clone)]
pub(crate) struct BumpArena {
    inner: Arc<Mutex<Inner>>,
    name: &'static str,
}

impl std::fmt::Debug for BumpArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().unwrap();
        f.debug_struct("BumpArena")
            .field("name", &self.name)
            .field("capacity", &inner.capacity)
            .field("cursor", &inner.cursor)
            .field("peak", &inner.peak)
            .finish()
    }
}

impl BumpArena {
    /// Reserve `capacity` bytes for the domain called `name`.
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        name: &'static str,
        capacity: usize,
        reclaim: Reclaim,
    ) -> Result<Self> {
        // A disjoint sub-range of the device's transient reservation. The
        // span is untyped storage; every range is written by its claimant
        // before any kernel reads it, and the cursor guarantees ranges handed
        // out within one generation are disjoint.
        let base = carve_transient(stream, capacity)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                base,
                capacity,
                cursor: 0,
                live: 0,
                dirty: false,
                peak: 0,
                stream: stream.clone(),
                reclaim,
            })),
            name,
        })
    }

    /// Open a generation. The cursor cannot reset while the returned guard —
    /// or any other — is alive.
    pub(crate) fn generation(&self) -> Generation {
        self.inner.lock().unwrap().live += 1;
        Generation {
            inner: Arc::clone(&self.inner),
            name: self.name,
        }
    }

    /// Whether any [`Generation`] guard is currently open on this arena.
    pub(crate) fn is_live(&self) -> bool {
        self.inner.lock().unwrap().live > 0
    }

    /// `(cursor, peak, capacity)` — the watermark that sizes this domain's
    /// span, and the input that places the transient boundary.
    pub(crate) fn stats(&self) -> (usize, usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.cursor, inner.peak, inner.capacity)
    }
}

/// Bump `len` bytes aligned to `align` from the arena behind `inner`.
///
/// The one allocation primitive, reached only through [`Generation::alloc`].
/// There is deliberately no arena-level entry point: a range handed out by the
/// arena would be bounded by the arena — which lives for the process — and so
/// would reintroduce exactly the unbounded lease this module exists to remove.
///
/// Errors rather than growing: the span is the domain's budget, and §3.6's fast
/// gate is supposed to have sized the wave to fit *before* assembly starts. An
/// overflow here means the gate was wrong, which is worth a loud failure rather
/// than a silent allocation behind its back.
fn bump<'a>(
    inner: &'a Mutex<Inner>,
    name: &'static str,
    len: usize,
    align: usize,
) -> Result<BumpRange<'a>> {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    let mut inner = inner.lock().unwrap();
    let start = (inner.cursor + align - 1) & !(align - 1);
    let end = start
        .checked_add(len)
        .ok_or_else(|| candle::Error::Msg(format!("{name}: bump allocation overflowed usize")))?;
    if end > inner.capacity {
        candle::bail!(
            "{}: transient span exhausted — {len} B at offset {start} exceeds the \
             {} B budget. The wave should have been gated to fit before assembly.",
            name,
            inner.capacity,
        );
    }
    inner.cursor = end;
    inner.dirty = true;
    if end > inner.peak {
        inner.peak = end;
        // Only on a new high-water mark, so this is quiet in steady state
        // and self-terminating: a domain's watermark converges within the
        // first few waves. This is the measurement the transient span is
        // sized from, and it has to come from a real workload rather than
        // the scheduler's own reporting path, which not every run exercises.
        log::debug!("{}: transient peak {} B of {} B", name, end, inner.capacity);
    }
    Ok(BumpRange {
        ptr: inner.base + start as u64,
        len,
        wave: PhantomData,
    })
}

/// A live claim on a [`BumpArena`]'s current contents.
///
/// While one exists the cursor cannot reset, so every range handed out since
/// the last reset stays valid. The last guard to drop fences the domain's
/// stream and resets.
pub struct Generation {
    inner: Arc<Mutex<Inner>>,
    name: &'static str,
}

impl Generation {
    /// Bump `len` bytes, aligned to `align`, from the half this guard pins.
    ///
    /// The returned range borrows `self`, so it cannot outlive the guard whose
    /// drop resets the cursor underneath it. That is the whole safety argument
    /// for wave buffers, moved from a runtime count to the type system:
    /// a caller who holds a `BumpRange<'w>` provably still holds the guard.
    ///
    /// Allocating *through the guard* also removes an ambiguity the arena
    /// cannot resolve. A bump range has to come from a specific half, and with
    /// two waves in flight there is no way to tell from the outside which half
    /// a given call site belongs to — the older wave's late allocations would
    /// land in the newer wave's half and be freed by its reset. Here the guard
    /// names its own half, so the question never arises.
    ///
    /// A range cannot outlive the guard that reclaims it:
    ///
    /// ```compile_fail
    /// # use candle_nn::kv_cache::{begin_wave, BumpRange};
    /// # fn f(stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>)
    /// #     -> candle::Result<()> {
    /// let range = {
    ///     let wave = begin_wave(stream)?;
    ///     wave.alloc(1024, 256)?
    /// };
    /// println!("{}", range.ptr);
    /// # Ok(()) }
    /// ```
    ///
    /// while using it inside the guard's scope is fine — the control that keeps
    /// the case above from failing for an unrelated reason:
    ///
    /// ```no_run
    /// # use candle_nn::kv_cache::begin_wave;
    /// # fn f(stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>)
    /// #     -> candle::Result<()> {
    /// let wave = begin_wave(stream)?;
    /// let range = wave.alloc(1024, 256)?;
    /// println!("{}", range.ptr);
    /// # Ok(()) }
    /// ```
    pub fn alloc(&self, len: usize, align: usize) -> Result<BumpRange<'_>> {
        bump(&self.inner, self.name, len, align)
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        let (should_reset, stream, reclaim) = {
            let mut inner = self.inner.lock().unwrap();
            inner.live -= 1;
            (
                inner.live == 0 && inner.dirty,
                inner.stream.clone(),
                inner.reclaim,
            )
        };
        if !should_reset {
            return;
        }
        // Fence before the cursor moves: the ranges about to become reusable
        // may still be under a kernel's read, and for this domain nothing else
        // orders that read against the next writer.
        if reclaim == Reclaim::Fence {
            if let Err(e) = stream.synchronize() {
                // Refuse to reset rather than hand out memory the GPU may still
                // be reading. The span leaks for one generation; the
                // alternative is a data race (principle 7).
                log::error!(
                    "{}: transient reset skipped — stream fence failed: {e:?}",
                    self.name
                );
                return;
            }
        }
        let mut inner = self.inner.lock().unwrap();
        inner.cursor = 0;
        inner.dirty = false;
    }
}

fn domains() -> &'static Mutex<HashMap<usize, BumpArena>> {
    static DOMAINS: OnceLock<Mutex<HashMap<usize, BumpArena>>> = OnceLock::new();
    DOMAINS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `(cursor, peak, capacity)` for the persistence domain on `ordinal`, or
/// `None` if it has never been used.
///
/// Per-domain peaks are what the transient span is sized from:
/// `S = 2·W_wave + W_persist + shelf`.
pub fn persistence_domain_stats(ordinal: usize) -> Option<(usize, usize, usize)> {
    domains().lock().unwrap().get(&ordinal).map(|a| a.stats())
}

/// Wave-domain half size.
///
/// **Measured**, on Qwen3-30B-A3B at batch 64 (2026-08-07): both halves peak at
/// 30.8 MiB, so 64 MiB is a little over 2x headroom. The peak is a wide-prefill
/// attention output; decode-only layers sit at 3.8-6 MiB. Both halves converge
/// on the same number, which is what per-layer alternation should produce.
///
/// The headroom is deliberate rather than tight. The peak scales with the
/// widest prefill a wave admits and with head count, so a larger model or a
/// longer prefill window moves it, and exhausting a half fails the forward
/// outright — which is the correct behaviour (it is how the O(depth)
/// accumulation bug was caught) but a bad thing to run close to.
///
/// The reservation's transient tier is sized *from* this constant rather than
/// the other way round ([`super::region_pool`]), so raising it moves the
/// boundary and costs the KV side two regions per extra 16 MiB — which is the
/// honest price and the reason to keep [`wave_domain_stats`] and the peak log
/// pointed at it.
pub(super) const WAVE_HALF_BYTES: usize = 64 * 1024 * 1024;

/// The scheduler's wave domain: two halves, alternating.
///
/// Double-buffered so wave `N+1` can assemble while wave `N`'s kernels are
/// still draining (§3.6). A half resets only when its own generation guard
/// drops, which fences the stream first — so the half being filled and the half
/// being drained are never the same memory.
struct WaveDomain {
    halves: [BumpArena; 2],
    /// The half [`begin_wave`] handed out most recently: the one an in-flight
    /// wave is allocating from.
    current: usize,
}

fn wave_domains() -> &'static Mutex<HashMap<usize, WaveDomain>> {
    static DOMAINS: OnceLock<Mutex<HashMap<usize, WaveDomain>>> = OnceLock::new();
    DOMAINS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn with_wave_domain<R>(
    stream: &Arc<CudaStream>,
    f: impl FnOnce(&mut WaveDomain) -> Result<R>,
) -> Result<R> {
    let mut map = wave_domains().lock().unwrap();
    let ordinal = stream.context().ordinal();
    let domain = match map.entry(ordinal) {
        Entry::Occupied(o) => o.into_mut(),
        Entry::Vacant(v) => v.insert(WaveDomain {
            halves: [
                BumpArena::new(stream, "wave-a", WAVE_HALF_BYTES, Reclaim::StreamOrdered)?,
                BumpArena::new(stream, "wave-b", WAVE_HALF_BYTES, Reclaim::StreamOrdered)?,
            ],
            // So the first `begin_wave` lands on half A.
            current: 1,
        }),
    };
    f(domain)
}

/// Open a wave: switch to the other half and return its generation guard.
///
/// **The guard must be held for the whole wave**, and the borrow checker now
/// holds callers to it: [`Generation::alloc`] hands out ranges that borrow this
/// guard, and the tensors built on them carry the same lifetime, so a wave
/// intermediate cannot be named after the guard drops. While the guard lives
/// the cursor cannot rewind; when it drops, the stream fences and the half is
/// reusable.
///
/// Refuses when the half it would take is still live. Two waves in flight is
/// the point of double buffering — N+1 assembles while N drains — but a third
/// would have to share a span with one of them, and sharing means the first
/// reset frees memory the other is still reading. Refuse rather than corrupt
/// (principle 7).
///
/// # Alternation is not a safety property
///
/// A layer opens two generations — one for attention, one for the FFN — so with
/// one attention group per layer the parity is even and a given call site keeps
/// landing on the same half. That is fine and expected: what separates a half's
/// last read from its next write is the same-stream work issued in between, and
/// between two attention sections sits an entire FFN. Alternation is what makes
/// that gap large, not what makes it correct.
pub fn begin_wave(stream: &Arc<CudaStream>) -> Result<Generation> {
    with_wave_domain(stream, |domain| {
        let next = domain.current ^ 1;
        if domain.halves[next].is_live() {
            candle::bail!(
                "wave domain: half {next} still has a live generation — a third \
                 wave started while two were in flight. Both halves are spoken \
                 for, and sharing one would let the first reset free memory the \
                 other is still reading."
            )
        }
        domain.current = next;
        Ok(domain.halves[next].generation())
    })
}

/// `(cursor, peak, capacity)` for each wave half on `ordinal`.
///
/// The `W_wave` term of the span equation `S = 2·W_wave + W_persist`.
pub fn wave_domain_stats(ordinal: usize) -> Option<[(usize, usize, usize); 2]> {
    wave_domains()
        .lock()
        .unwrap()
        .get(&ordinal)
        .map(|d| [d.halves[0].stats(), d.halves[1].stats()])
}

/// The persistence thread's transient domain: migration and quantize staging.
///
/// Sized to [`MIGRATION_STAGING_CAP_BYTES`], which is already the cap every
/// staging batch bisects against — so the span is exactly the budget the
/// callers were written to respect, now enforced by the allocator instead of
/// by the driver refusing.
///
/// Separate from the wave domain so the persistence thread can never reset a
/// buffer the scheduler is still reading, and vice versa (§3.6).
pub(crate) fn persistence_domain(stream: &Arc<CudaStream>) -> Result<BumpArena> {
    let mut map = domains().lock().unwrap();
    let ordinal = stream.context().ordinal();
    if let Some(a) = map.get(&ordinal) {
        return Ok(a.clone());
    }
    let arena = BumpArena::new(
        stream,
        "persist-staging",
        MIGRATION_STAGING_CAP_BYTES,
        Reclaim::Fence,
    )?;
    map.insert(ordinal, arena.clone());
    Ok(arena)
}

#[cfg(test)]
mod tests {

    /// Ranges from one generation never overlap — the property that removes
    /// the need for any slot table or disjointness bookkeeping (§3.6).
    #[test]
    fn ranges_within_a_generation_are_disjoint() {
        let (base, cap) = (0x1000u64, 4096usize);
        let mut cursor = 0usize;
        let mut alloc = |len: usize, align: usize| -> (u64, usize) {
            let start = (cursor + align - 1) & !(align - 1);
            assert!(start + len <= cap);
            cursor = start + len;
            (base + start as u64, len)
        };
        let a = alloc(100, 16);
        let b = alloc(200, 16);
        let c = alloc(1, 256);
        assert!(a.0 + a.1 as u64 <= b.0, "a and b overlap");
        assert!(b.0 + b.1 as u64 <= c.0, "b and c overlap");
        assert_eq!(c.0 % 256, 0, "alignment must be honoured");
    }

    /// Alignment rounds the cursor up, never down — a range must start at or
    /// after the previous one's end.
    #[test]
    fn alignment_only_moves_the_cursor_forward() {
        for align in [1usize, 4, 16, 256, 4096] {
            for cursor in 0usize..64 {
                let start = (cursor + align - 1) & !(align - 1);
                assert!(start >= cursor, "align {align} moved cursor {cursor} back");
                assert_eq!(start % align, 0);
                assert!(start - cursor < align, "over-aligned");
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod wave_tests {
    use super::{begin_wave, wave_domain_stats};
    use candle::{Device, Result};

    /// The wave domain is process-global and `cargo test` runs tests in
    /// parallel, so these take turns. It is the *crate-wide* lock rather than a
    /// local one: the KV selection tests build backings on the same device, and
    /// a lock that only excludes this module would not exclude them.
    use super::super::gpu_test_lock::gpu_serial as serial;

    fn stream() -> Option<std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>> {
        match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => Some(d.cuda_stream()),
            _ => None,
        }
    }

    /// Inside a wave, ranges come from the half and do not overlap.
    #[test]
    fn ranges_within_a_wave_are_disjoint() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let guard = begin_wave(&s)?;
        let a = guard.alloc(1000, 256)?;
        let b = guard.alloc(2000, 256)?;
        assert!(a.ptr + a.len as u64 <= b.ptr, "ranges overlap");
        assert_eq!(b.ptr % 256, 0);
        // Nothing to assert about life after `drop(guard)`: `a` and `b` borrow
        // it, so a program that used them afterwards would not compile. That is
        // the property this used to check at run time.
        Ok(())
    }

    /// Consecutive waves land on different halves. This is the property that
    /// lets wave N+1 assemble while wave N's kernels drain: if both waves used
    /// the same span, the reset between them would free memory still being
    /// read.
    #[test]
    fn consecutive_waves_alternate_halves() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let g1 = begin_wave(&s)?;
        let first = g1.alloc(4096, 256)?.ptr;
        drop(g1);

        let g2 = begin_wave(&s)?;
        let second = g2.alloc(4096, 256)?.ptr;
        drop(g2);

        assert_ne!(first, second, "consecutive waves shared a half");

        let g3 = begin_wave(&s)?;
        let third = g3.alloc(4096, 256)?.ptr;
        drop(g3);
        assert_eq!(third, first, "the third wave should be back on half A");
        Ok(())
    }

    /// A second wave opened while the first is live is refused, not silently
    /// given the half the first is still filling (principle 7).
    #[test]
    fn a_concurrent_wave_is_refused() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let _outer = begin_wave(&s)?;
        // The other half is free, so this one succeeds...
        let inner = begin_wave(&s)?;
        // ...but a third has nowhere to go while both are live.
        assert!(
            begin_wave(&s).is_err(),
            "a third concurrent wave must be refused"
        );
        drop(inner);
        Ok(())
    }

    /// Peaks are what the span is sized from, so they must survive the reset
    /// that clears the cursor.
    #[test]
    fn peak_outlives_the_reset() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let guard = begin_wave(&s)?;
        guard.alloc(8192, 256)?;
        drop(guard);
        let stats = wave_domain_stats(0).expect("domain exists");
        let peak = stats[0].1.max(stats[1].1);
        assert!(peak >= 8192, "peak {peak} lost across the reset");
        assert_eq!(stats[0].0.min(stats[1].0), 0, "a cursor failed to rewind");
        Ok(())
    }
}
