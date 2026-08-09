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
use candle::cuda_backend::wave_provenance::WaveTicket;
use candle::Result;

use super::chunk_ops::MIGRATION_STAGING_CAP_BYTES;
use super::region_pool::carve_transient;
use super::wave_plan::LayerPhase;

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
    /// Bumped every time the cursor rewinds.
    ///
    /// A [`WaveTicket`] carries the epoch it was minted in, so a ticket from a
    /// closed generation resolves to nothing rather than carving from whatever
    /// occupies the span now. `LiveTensor<'w>` already makes that unreachable at
    /// compile time; this is the backstop for the `unsafe` constructors that
    /// mint leases from raw pointers, where the compiler has nothing to check.
    epoch: u64,
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
                epoch: 0,
                stream: stream.clone(),
                reclaim,
            })),
            name,
        })
    }

    /// Open a generation reserving `planned_span` bytes at the base of the span
    /// for layout slots.
    ///
    /// The cursor starts above the reserved region, so bumped ranges and planned
    /// slots partition the span between them instead of competing for it.
    /// Refuses a reservation larger than the arena, since every slot in it would
    /// otherwise address memory the domain does not own.
    pub(crate) fn generation(&self, domain: u32, arena: u32) -> Result<Generation> {
        let mut inner = self.inner.lock().unwrap();
        if inner.cursor > inner.peak {
            inner.peak = inner.cursor;
        }
        inner.live += 1;
        drop(inner);
        Ok(Generation {
            inner: Arc::clone(&self.inner),
            name: self.name,
            domain,
            arena,
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

/// [`bump`] without the borrow, for the ticket resolver.
///
/// [`BumpRange`] ties its lifetime to the mutex guard, which is exactly right at
/// a call site holding a [`Generation`] and useless here: the resolver is called
/// from candle-core, which has no guard to borrow from and gets its bound from
/// the `'w` already on the operand instead. Returning `None` on exhaustion
/// rather than erroring, for the reason given on [`resolve_wave_alloc`].
fn bump_raw(inner: &Mutex<Inner>, name: &'static str, len: usize, align: usize) -> Option<u64> {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    let mut inner = inner.lock().ok()?;
    let start = (inner.cursor + align - 1) & !(align - 1);
    let end = start.checked_add(len)?;
    if end > inner.capacity {
        return None;
    }
    inner.cursor = end;
    inner.dirty = true;
    if end > inner.peak {
        inner.peak = end;
        log::debug!("{}: transient peak {} B of {} B", name, end, inner.capacity);
    }
    Some(inner.base + start as u64)
}

/// A live claim on a [`BumpArena`]'s current contents.
///
/// While one exists the cursor cannot reset, so every range handed out since
/// the last reset stays valid. The last guard to drop fences the domain's
/// stream and resets.
pub struct Generation {
    inner: Arc<Mutex<Inner>>,
    name: &'static str,
    /// Coordinates of the arena this guard is open on, so the generation can
    /// mint a [`WaveTicket`] for the storages allocated inside it.
    domain: u32,
    arena: u32,
}

impl Generation {
    /// The coordinate an op allocating from this generation should carry.
    ///
    /// Stamped with the arena's current epoch, so the ticket stops resolving the
    /// moment this generation's cursor rewinds.
    pub fn ticket(&self) -> WaveTicket {
        WaveTicket {
            domain: self.domain,
            arena: self.arena,
            epoch: self.inner.lock().unwrap().epoch,
        }
    }

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
    /// # use candle_nn::kv_cache::{begin_wave, BumpRange, LayerPhase};
    /// # fn f(stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>)
    /// #     -> candle::Result<()> {
    /// let range = {
    ///     let wave = begin_wave(stream, LayerPhase::Attention)?;
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
    /// # use candle_nn::kv_cache::{begin_wave, LayerPhase};
    /// # fn f(stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>)
    /// #     -> candle::Result<()> {
    /// let wave = begin_wave(stream, LayerPhase::Attention)?;
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
        inner.epoch += 1;
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

/// The attention phase's span.
///
/// Sized from measurement, not symmetry. A generation's bump cursor never
/// rewinds, so a phase needs the **sum** of everything it allocates, not its
/// peak. On Qwen3-30B-A3B at ten concurrent contexts the attention phase sums to
/// 297 MiB, which this rounds up to leave headroom for a wider wave.
///
/// Read that number against the one this constant used to hold. It was sized at
/// 128 MiB from a measured 32 MiB peak — but that peak was the *paged-attention
/// context buffer alone*, because `attention_norm` was accepting a wave and
/// passing `Backing::Owned`, so the norm, the QKV projection, its cast and
/// `o_proj` all inherited the pool instead. A span sized from a chain that is
/// not running measures the chain that is not running, and it reads exactly like
/// a comfortable fit: 24% utilisation, no error, no fallback warning. Seeding
/// the chain moved the phase to 297 MiB in one step.
pub const WAVE_ATTN_BYTES: usize = 384 * 1024 * 1024;

/// The FFN phase's span.
///
/// Four times the attention span because the expert chain is four buffers deep
/// over `rows x experts_per_tok` rows — gather, gate, up, SwiGLU, down, cast —
/// and none of them is reclaimed until the guard drops. This is the number that
/// was clipping: the measured peak sat at 99.87% of a 64 MiB span, so the true
/// demand was unknown until the cap was lifted off it.
pub const WAVE_FFN_BYTES: usize = 512 * 1024 * 1024;

/// The forward-scoped span.
///
/// Three orders of magnitude smaller than the layer phases, because it holds a
/// different *kind* of thing. The phase spans carry activations, which scale
/// with wave width times model width; this carries the metadata a forward builds
/// once and every layer reads — ragged prefill offsets, RoPE tables, gathered
/// position ids — which scale with the *sequence count*. A 64-sequence wave puts
/// a few kilobytes here.
///
/// **One region**, which is also the floor: the transient tier is carved in
/// [`super::region_pool::REGION_BYTES`] units and a compile-time assert requires
/// the total to stay region-aligned, so a smaller span would not buy back any
/// memory — it would round up to this anyway. The measured need is ~3 KB, so
/// this is ~5000x headroom, and none of it is wasted: a span that cannot be
/// subdivided cannot be spent on anything else.
pub const WAVE_FORWARD_BYTES: usize = 16 * 1024 * 1024;

/// The scheduler's wave domain: one arena per layer phase.
///
/// A layer opens exactly two generations — attention, then the FFN — so with the
/// arenas named by phase, attention reuses one span every layer and the FFN
/// reuses the other. That is what the alternating pair already produced by
/// parity; naming it removes the dependence on the count of `begin_wave` calls
/// per layer staying even, and lets each span be sized from *its own* phase
/// rather than both from the larger.
///
/// The sizing difference is not small: on Qwen3-30B-A3B at ten concurrent
/// contexts the attention phase sums to 297 MiB and the FFN to at least 511 MiB,
/// so two max-phase spans would reserve 1022 MiB where 808 MiB does.
///
/// # Why resetting a phase's span every layer is safe
///
/// Unchanged from the alternating pair, because the ordering argument was never
/// about which half: a span is reset when its guard drops and handed out again
/// one layer later, and between those two points sits the *other* phase's entire
/// kernel sequence on the same stream. Same-stream launches complete in issue
/// order, so the previous layer's reads have drained before the next layer's
/// writes are issued ([`Reclaim::StreamOrdered`]).
struct WaveDomain {
    /// Indexed by [`LayerPhase`] via [`phase_index`].
    arenas: [BumpArena; 3],
}

/// Arena coordinate for a domain that is **not** a wave half.
///
/// The persistence domain stages on the copy stream and reclaims by fence, so a
/// range from it must never be inherited by an op the way a wave range is: the
/// two are not ordered against each other. A ticket carrying this resolves to
/// nothing, so an op reading persistence-staged memory allocates from the pool —
/// which is the correct answer, not a fallback.
pub(crate) const NOT_A_WAVE: u32 = u32::MAX;

/// Slot for a phase's arena.
///
/// A `match` rather than a cast so adding a phase is a compile error here
/// instead of an out-of-bounds index at run time.
fn phase_index(phase: LayerPhase) -> usize {
    match phase {
        LayerPhase::Attention => 0,
        LayerPhase::Ffn => 1,
        LayerPhase::Forward => 2,
    }
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
            arenas: [
                BumpArena::new(stream, "wave-attn", WAVE_ATTN_BYTES, Reclaim::StreamOrdered)?,
                BumpArena::new(stream, "wave-ffn", WAVE_FFN_BYTES, Reclaim::StreamOrdered)?,
                BumpArena::new(
                    stream,
                    "wave-forward",
                    WAVE_FORWARD_BYTES,
                    Reclaim::StreamOrdered,
                )?,
            ],
        }),
    };
    f(domain)
}

/// Open a generation on `phase`'s span.
///
/// **The guard must be held for the whole phase**, and the borrow checker holds
/// callers to it: [`Generation::alloc`] hands out ranges that borrow this guard,
/// and the tensors built on them carry the same lifetime, so a wave intermediate
/// cannot be named after the guard drops. While the guard lives the cursor
/// cannot rewind; when it drops, the span is reusable.
///
/// Refuses when that phase's span is already live. A layer runs its two phases
/// in sequence — attention's guard drops before the FFN's opens — so an overlap
/// means a phase was entered twice without leaving it, and the inner drop would
/// reset the span the outer one is still allocating from. Refuse rather than
/// corrupt (principle 7).
pub fn begin_wave(stream: &Arc<CudaStream>, phase: LayerPhase) -> Result<Generation> {
    // Idempotent, and done here rather than at load: this is the first moment a
    // wave arena can exist, so it is the earliest point the resolver could be
    // useful and the latest it could be needed.
    candle::cuda_backend::wave_provenance::install_wave_allocator(resolve_wave_alloc);
    let ordinal = stream.context().ordinal() as u32;
    let arena_idx = phase_index(phase) as u32;
    with_wave_domain(stream, |domain| {
        let arena = &domain.arenas[phase_index(phase)];
        if arena.is_live() {
            candle::bail!(
                "wave domain: the {phase:?} span already has a live generation. A \
                 layer leaves one phase before entering the next, so this is a \
                 phase re-entered while still open — the inner guard's drop would \
                 reset the span the outer one is still handing out."
            )
        }
        arena.generation(ordinal, arena_idx)
    })
}

/// Carve `bytes` from the arena a [`WaveTicket`] names — the resolver candle-core
/// calls when an op inherits its operand's arena.
///
/// `None` rather than an error for every miss, because each one has an
/// unremarkable meaning and a pool allocation is always a correct answer:
/// an unknown domain (no wave has run on that stream), a stale epoch (the
/// generation rewound), or an exhausted span. Only the last is interesting, and
/// it is already reported by the span-exhausted path the planned allocations
/// take — turning it into a hard failure here would abort a forward that could
/// have completed on pool memory.
fn resolve_wave_alloc(ticket: WaveTicket, bytes: usize, align: usize) -> Option<u64> {
    let map = wave_domains().lock().ok()?;
    let domain = map.get(&(ticket.domain as usize))?;
    let arena = domain.arenas.get(ticket.arena as usize)?;
    let inner = Arc::clone(&arena.inner);
    let name = arena.name;
    // Drop the registry lock before touching the arena: the two are always taken
    // in this order and never the reverse, which is what keeps this free of the
    // deadlock an allocation path called from arbitrary op code could otherwise
    // introduce.
    drop(map);
    {
        let guard = inner.lock().ok()?;
        if guard.epoch != ticket.epoch || guard.live == 0 {
            return None;
        }
    }
    bump_raw(&inner, name, bytes, align)
}

/// `(cursor, peak, capacity)` per wave arena on `ordinal`, ordered by
/// [`phase_index`]: attention, FFN, forward.
///
/// The `W_wave` term of the span equation
/// `S = W_attn + W_ffn + W_forward + W_persist`.
///
/// **`peak` is a process-lifetime high-water mark**, not a per-wave or
/// per-request figure: it is raised when the cursor passes it and never lowered,
/// because a span has to be sized for the worst moment it ever sees, not the
/// most recent one. Reading it as "how full is the arena now" is wrong — that is
/// `cursor`, and between waves it is zero.
pub fn wave_domain_stats(ordinal: usize) -> Option<[(usize, usize, usize); 3]> {
    wave_domains().lock().unwrap().get(&ordinal).map(|d| {
        [
            d.arenas[0].stats(),
            d.arenas[1].stats(),
            d.arenas[2].stats(),
        ]
    })
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
    use super::{begin_wave, wave_domain_stats, LayerPhase};
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
        let guard = begin_wave(&s, LayerPhase::Attention)?;
        let a = guard.alloc(1000, 256)?;
        let b = guard.alloc(2000, 256)?;
        assert!(a.ptr + a.len as u64 <= b.ptr, "ranges overlap");
        assert_eq!(b.ptr % 256, 0);
        // Nothing to assert about life after `drop(guard)`: `a` and `b` borrow
        // it, so a program that used them afterwards would not compile. That is
        // the property this used to check at run time.
        Ok(())
    }

    /// The two phases must not share a span. If they did, the FFN's reset would
    /// free the attention output the residual add is still reading, and each
    /// span would have to be sized for the larger of the two phases.
    #[test]
    fn the_phases_have_separate_spans() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let g1 = begin_wave(&s, LayerPhase::Attention)?;
        let attn = g1.alloc(4096, 256)?.ptr;
        drop(g1);

        let g2 = begin_wave(&s, LayerPhase::Ffn)?;
        let ffn = g2.alloc(4096, 256)?.ptr;
        drop(g2);

        assert_ne!(attn, ffn, "attention and FFN shared a span");
        Ok(())
    }

    /// The next layer reuses its phase's span rather than accumulating — the
    /// whole point of resetting on guard drop. Without it a 48-layer wave would
    /// hold 48 layers of transients instead of one layer's.
    #[test]
    fn the_next_layer_reuses_the_same_phase_span() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let mut first = None;
        for _layer in 0..3 {
            let g = begin_wave(&s, LayerPhase::Attention)?;
            let ptr = g.alloc(4096, 256)?.ptr;
            drop(g);
            match first {
                None => first = Some(ptr),
                Some(f) => assert_eq!(ptr, f, "a later layer moved off the span"),
            }
        }
        Ok(())
    }

    /// Both phases may be open at once — they are different spans — but
    /// re-entering one while it is still open is refused (principle 7): the
    /// inner guard's drop would reset the span the outer one is filling.
    #[test]
    fn re_entering_a_live_phase_is_refused() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let _attn = begin_wave(&s, LayerPhase::Attention)?;
        let ffn = begin_wave(&s, LayerPhase::Ffn)?;
        assert!(
            begin_wave(&s, LayerPhase::Attention).is_err(),
            "re-entering the live attention span must be refused"
        );
        assert!(
            begin_wave(&s, LayerPhase::Ffn).is_err(),
            "re-entering the live FFN span must be refused"
        );
        drop(ffn);
        Ok(())
    }

    /// Peaks are what the span is sized from, so they must survive the reset
    /// that clears the cursor.
    #[test]
    fn peak_outlives_the_reset() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let guard = begin_wave(&s, LayerPhase::Attention)?;
        guard.alloc(8192, 256)?;
        drop(guard);
        let stats = wave_domain_stats(0).expect("domain exists");
        let peak = stats[0].1.max(stats[1].1);
        assert!(peak >= 8192, "peak {peak} lost across the reset");
        assert_eq!(stats[0].0.min(stats[1].0), 0, "a cursor failed to rewind");
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod provenance_tests {
    use super::super::gpu_test_lock::gpu_serial as serial;
    use super::{begin_wave, LayerPhase};
    use candle::cuda_backend::wave_provenance::LeaseOrigin;
    use candle::{DType, Device, Result, Storage, Tensor};

    fn cuda() -> Option<Device> {
        match Device::cuda_if_available(0) {
            Ok(d @ Device::Cuda(_)) => Some(d),
            _ => None,
        }
    }

    /// The backing a tensor's storage carries.
    fn backing_of(t: &Tensor) -> candle::cuda_backend::Backing {
        let (storage, _) = t.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => c.backing,
            _ => panic!("expected CUDA storage"),
        }
    }

    /// **The rule.** An op reading a wave-backed operand allocates its output
    /// from the same generation — not from the pool.
    ///
    /// This is what makes the `'w` on the result true rather than merely
    /// permitted, so it is worth pinning directly: the type system cannot check
    /// that the *bytes* came from the arena, only that the lifetime is not
    /// widened. Asserting on the cursor as well as the backing catches a ticket
    /// that resolves to the right arena but never actually carves from it.
    #[test]
    fn an_op_inherits_its_operands_arena() -> Result<()> {
        let _gpu = serial();
        let Some(dev) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &dev else {
            return Ok(());
        };
        let stream = cd.cuda_stream();

        // A real allocation to point the lease at, so the cast reads live bytes.
        let owner = Tensor::zeros((256,), DType::F32, &dev)?;

        let wave = begin_wave(&stream, LayerPhase::Attention)?;
        let before = {
            let inner = wave.inner.lock().unwrap();
            inner.cursor
        };

        let range = wave.alloc(256 * 4, 256)?;
        let operand = unsafe {
            Tensor::from_leased_cuda_ptr(
                range.ptr,
                DType::F32,
                (256,),
                &dev,
                LeaseOrigin::Wave(wave.ticket()),
            )?
        };
        let cursor_after_operand = {
            let inner = wave.inner.lock().unwrap();
            inner.cursor
        };

        let cast = operand.to_dtype(DType::BF16)?;

        match backing_of(&cast) {
            candle::cuda_backend::Backing::Lease(LeaseOrigin::Wave(t)) => {
                assert_eq!(t, wave.ticket(), "inherited the wrong generation");
            }
            other => panic!("cast fell back to the pool: {other:?}"),
        }
        let cursor_after_cast = {
            let inner = wave.inner.lock().unwrap();
            inner.cursor
        };
        assert!(
            cursor_after_cast > cursor_after_operand,
            "the cast claimed no bytes from the arena ({cursor_after_operand} -> {cursor_after_cast})"
        );
        assert!(cursor_after_operand > before);

        drop(cast);
        drop(operand);
        drop(wave);
        // The owner must survive every one of those drops: a wave-backed result
        // that reached `cuMemFreeAsync` would take the pool with it.
        assert_eq!(owner.to_vec1::<f32>()?.len(), 256);
        Ok(())
    }

    /// A pool-backed operand stays on the pool. The rule inherits an arena; it
    /// does not invent one, so an ordinary tensor is untouched by any of this
    /// even while a generation happens to be open.
    #[test]
    fn a_pool_operand_does_not_acquire_a_wave() -> Result<()> {
        let _gpu = serial();
        let Some(dev) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &dev else {
            return Ok(());
        };
        let wave = begin_wave(&cd.cuda_stream(), LayerPhase::Ffn)?;
        let ordinary = Tensor::zeros((128,), DType::F32, &dev)?;
        let cast = ordinary.to_dtype(DType::BF16)?;
        assert!(
            matches!(backing_of(&cast), candle::cuda_backend::Backing::Owned),
            "a pool operand must not pick up a wave it never came from"
        );
        drop(wave);
        Ok(())
    }

    /// A ticket from a closed generation resolves to nothing, so an op holding
    /// a stale lease allocates from the pool instead of carving from whatever
    /// occupies that span now. `LiveTensor<'w>` makes this unreachable in safe
    /// code; the backstop exists for the `unsafe` raw-pointer constructors.
    #[test]
    fn a_stale_ticket_does_not_resolve() -> Result<()> {
        let _gpu = serial();
        let Some(dev) = cuda() else { return Ok(()) };
        let Device::Cuda(cd) = &dev else {
            return Ok(());
        };
        let stream = cd.cuda_stream();

        let stale = {
            let wave = begin_wave(&stream, LayerPhase::Attention)?;
            let t = wave.ticket();
            let _r = wave.alloc(1024, 256)?;
            t
        };
        assert!(
            candle::cuda_backend::wave_provenance::wave_alloc(stale, 1024, 256).is_none(),
            "a ticket outlived its generation and still carved from the span"
        );
        let _ = dev;
        Ok(())
    }
}
