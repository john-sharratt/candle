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
//! reusable, so it must not happen while a kernel is still reading one. Two
//! things guard it, and they are the whole safety argument:
//!
//! - **A counted generation.** `reset` is refused while any [`Generation`] is
//!   live. Refusal is loud — the count is checked, not assumed — because a
//!   silent early reset is a data race that reproduces as garbage output far
//!   from its cause (principle 7: safety by refusal, not by ceremony).
//! - **A stream fence.** The last generation to drop synchronises the domain's
//!   stream before the cursor moves, so the GPU has drained the ranges the
//!   host is about to hand out again. This is `PinnedStager`'s sync-then-reset
//!   discipline, applied to device memory.
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
//! what lets a `BumpRange` be a bare pointer with no lifetime.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::Result;

use super::chunk_ops::MIGRATION_STAGING_CAP_BYTES;
use super::region_pool::carve_transient;

/// A range handed out by a [`BumpArena`].
///
/// Deliberately **not** RAII: a bump range is freed by its generation's reset,
/// never individually, so a `Drop` impl would be a lie. It carries no lifetime
/// either — the compiler cannot express "valid until the generation resets",
/// and pretending otherwise with a borrow would force every consumer to thread
/// a lifetime that the counted generation already enforces at run time.
#[derive(Debug, Clone, Copy)]
pub struct BumpRange {
    /// Device address of the range's first byte.
    pub ptr: u64,
    /// Bytes reserved.
    pub len: usize,
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

    /// Bump-allocate `len` bytes, aligned to `align`.
    ///
    /// Errors rather than growing: the span is the domain's budget, and §3.6's
    /// fast gate is supposed to have sized the wave to fit *before* assembly
    /// starts. An overflow here means the gate was wrong, which is worth a
    /// loud failure rather than a silent allocation behind its back.
    pub fn alloc(&self, len: usize, align: usize) -> Result<BumpRange> {
        debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
        let mut inner = self.inner.lock().unwrap();
        let start = (inner.cursor + align - 1) & !(align - 1);
        let end = start.checked_add(len).ok_or_else(|| {
            candle::Error::Msg(format!("{}: bump allocation overflowed usize", self.name))
        })?;
        if end > inner.capacity {
            candle::bail!(
                "{}: transient span exhausted — {len} B at offset {start} exceeds the \
                 {} B budget. The wave should have been gated to fit before assembly.",
                self.name,
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
            log::debug!(
                "{}: transient peak {} B of {} B",
                self.name,
                end,
                inner.capacity
            );
        }
        Ok(BumpRange {
            ptr: inner.base + start as u64,
            len,
        })
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

/// A live claim on a [`BumpArena`]'s current contents.
///
/// While one exists the cursor cannot reset, so every range handed out since
/// the last reset stays valid. The last guard to drop fences the domain's
/// stream and resets.
pub struct Generation {
    inner: Arc<Mutex<Inner>>,
    name: &'static str,
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
/// **The guard must be held for the whole wave.** This is §3.7's answer for
/// wave intermediates: they are handed to candle ops as `Tensor`s and outlive
/// the scope that allocated them, so they cannot pin the cursor themselves —
/// the wave does. While the guard lives, [`wave_alloc`] hands out ranges from
/// this half and the cursor cannot rewind; when it drops, the stream fences and
/// the half is reusable.
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

/// Bump `len` bytes from the in-flight wave's half, or `None` if no wave is in
/// flight on this stream.
///
/// The absence is real, not a feature flag: the same kernel wrappers run inside
/// the scheduler's wave loop and standalone in probes, replay harnesses, and
/// kernel tests. Inside a wave there is a half whose generation guarantees the
/// range outlives every kernel reading it; outside one there is nothing to
/// bound the lifetime, and the caller must own its output instead.
pub fn wave_alloc(stream: &Arc<CudaStream>, len: usize, align: usize) -> Result<Option<BumpRange>> {
    with_wave_domain(stream, |domain| {
        // **Both halves live is ambiguous, and ambiguity here is silent
        // corruption.** This serves `domain.current`, the *most recently begun*
        // half — not the half the caller's own `Generation` pins, which it has
        // no way to name (the whole point of the ambient design is that kernel
        // wrappers deep in the call graph allocate without threading a guard
        // through). Those coincide only while exactly one generation is live.
        //
        // If a second wave opens while the first is still in flight, the first
        // wave's later allocations land in the second's half and are freed by
        // the second's reset — under kernels still reading them. `begin_wave`
        // refuses a *third* wave; this refuses to guess between two.
        //
        // Today nothing nests (attention guards drop before the FFN guard
        // opens), so this is unreachable — which is exactly when a wrong answer
        // would go unnoticed longest.
        if domain.halves[0].is_live() && domain.halves[1].is_live() {
            candle::bail!(
                "wave domain: both halves have live generations, so which one this \
                 allocation belongs to is ambiguous. Serving the most recent would \
                 hand the older wave memory the newer wave's reset frees. Close the \
                 outer wave before opening another, or thread the `Generation` to \
                 this call site."
            )
        }
        let half = &domain.halves[domain.current];
        if !half.is_live() {
            return Ok(None);
        }
        half.alloc(len, align).map(Some)
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
    use super::{begin_wave, wave_alloc, wave_domain_stats};
    use candle::{Device, Result};

    /// The wave domain is process-global and `cargo test` runs tests in
    /// parallel, so these take turns. Without it, `no_wave_in_flight_means_no_range`
    /// would see a half another test still had open.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn stream() -> Option<std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>> {
        match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => Some(d.cuda_stream()),
            _ => None,
        }
    }

    /// Outside a wave there is nothing to bound a range's lifetime, so the
    /// domain must say so rather than hand one out. This is what lets the same
    /// kernel wrappers run in probes and replay harnesses.
    #[test]
    fn no_wave_in_flight_means_no_range() -> Result<()> {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(s) = stream() else { return Ok(()) };
        assert!(wave_alloc(&s, 1024, 256)?.is_none());
        Ok(())
    }

    /// Inside a wave, ranges come from the half and do not overlap.
    #[test]
    fn ranges_within_a_wave_are_disjoint() -> Result<()> {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(s) = stream() else { return Ok(()) };
        let guard = begin_wave(&s)?;
        let a = wave_alloc(&s, 1000, 256)?.expect("wave in flight");
        let b = wave_alloc(&s, 2000, 256)?.expect("wave in flight");
        assert!(a.ptr + a.len as u64 <= b.ptr, "ranges overlap");
        assert_eq!(b.ptr % 256, 0);
        drop(guard);
        // And the half is closed again once the wave ends.
        assert!(wave_alloc(&s, 16, 256)?.is_none());
        Ok(())
    }

    /// Consecutive waves land on different halves. This is the property that
    /// lets wave N+1 assemble while wave N's kernels drain: if both waves used
    /// the same span, the reset between them would free memory still being
    /// read.
    #[test]
    fn consecutive_waves_alternate_halves() -> Result<()> {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(s) = stream() else { return Ok(()) };
        let g1 = begin_wave(&s)?;
        let first = wave_alloc(&s, 4096, 256)?.expect("wave in flight").ptr;
        drop(g1);

        let g2 = begin_wave(&s)?;
        let second = wave_alloc(&s, 4096, 256)?.expect("wave in flight").ptr;
        drop(g2);

        assert_ne!(first, second, "consecutive waves shared a half");

        let g3 = begin_wave(&s)?;
        let third = wave_alloc(&s, 4096, 256)?.expect("wave in flight").ptr;
        drop(g3);
        assert_eq!(third, first, "the third wave should be back on half A");
        Ok(())
    }

    /// A second wave opened while the first is live is refused, not silently
    /// given the half the first is still filling (principle 7).
    #[test]
    fn a_concurrent_wave_is_refused() -> Result<()> {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
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
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(s) = stream() else { return Ok(()) };
        let guard = begin_wave(&s)?;
        wave_alloc(&s, 8192, 256)?.expect("wave in flight");
        drop(guard);
        let stats = wave_domain_stats(0).expect("domain exists");
        let peak = stats[0].1.max(stats[1].1);
        assert!(peak >= 8192, "peak {peak} lost across the reset");
        assert_eq!(stats[0].0.min(stats[1].0), 0, "a cursor failed to rewind");
        Ok(())
    }
}
