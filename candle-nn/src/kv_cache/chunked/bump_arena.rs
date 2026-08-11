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
use super::region_pool::{carve_persist, place_transient, release_transient};
use super::wave_census::{self, Carve};
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
    /// The largest generation the census has already itemised, **per carve
    /// count**.
    ///
    /// Kept apart from `peak` because the two answer different questions: `peak`
    /// is what the span must hold, this is what has already been reported.
    ///
    /// Keyed by carve count rather than held as one number because an arena
    /// serves more than one chain, and the widest is not the only one worth
    /// seeing. The attention span carries a twelve-carve prefill chain and a
    /// narrower decode chain that reaches `o_proj` through a different operand;
    /// a single high-water mark would only ever print the first, and the plan
    /// has to upper-bound both. The count is what distinguishes them, and it
    /// takes a handful of distinct values, so this stays small.
    reported_peak: HashMap<usize, usize>,
    /// Ranges carved in the current generation, when
    /// [`super::wave_census::enabled`]. Empty in every ordinary run.
    census: Vec<Carve>,
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
    /// Reserve `capacity` bytes for the domain called `name`, out of the fixed
    /// persistence block.
    ///
    /// The **only** fixed-carve domain. The persistence thread stages on its own
    /// copy stream, on a schedule that has nothing to do with a wave's, so its
    /// ranges can be live at any moment a forward begins — which is exactly when
    /// the wave tier moves. It therefore lives at the far left of the span,
    /// where neither boundary can reach it, and it is the one domain whose
    /// address is fixed for the process lifetime.
    ///
    /// The wave domains are the opposite: [`BumpArena::detached`], placed per
    /// forward, gone in between.
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        name: &'static str,
        capacity: usize,
        reclaim: Reclaim,
    ) -> Result<Self> {
        // A disjoint sub-range of the device's reservation. The span is untyped
        // storage; every range is written by its claimant before any kernel
        // reads it, and the cursor guarantees ranges handed out within one
        // generation are disjoint.
        let base = carve_persist(stream, capacity)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                base,
                capacity,
                cursor: 0,
                live: 0,
                dirty: false,
                peak: 0,
                reported_peak: HashMap::new(),
                census: Vec::new(),
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

    /// A domain with no backing yet, to be positioned by [`Self::rebase`].
    ///
    /// The wave domains are built this way because the tier they live in does
    /// not exist between forwards. A zero capacity is not a degenerate case to
    /// guard against — it is the correct state of a wave span when no wave is
    /// running, and any allocation against it fails loudly on the ordinary
    /// exhausted-span path.
    pub(crate) fn detached(stream: &Arc<CudaStream>, name: &'static str) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                base: 0,
                capacity: 0,
                cursor: 0,
                live: 0,
                dirty: false,
                peak: 0,
                reported_peak: HashMap::new(),
                census: Vec::new(),
                epoch: 0,
                stream: stream.clone(),
                reclaim: Reclaim::StreamOrdered,
            })),
            name,
        }
    }

    /// Move this domain to `base` with `capacity` bytes, for the wave about to
    /// start.
    ///
    /// **Only legal while no generation is open**, which is exactly when the
    /// domain holds nothing: the transient tier vanishes between forwards, so
    /// there is no live range whose address this could invalidate. That is what
    /// lets the tier be the one variable-size, movable block in the span — and
    /// it is why it can sit between the arenas and the weights, absorbing both
    /// sides' movement without a hole and without relocating anything.
    ///
    /// A `BumpRange` is still a bare pointer. `'w` bounds it to its generation,
    /// and a generation never spans a rebase.
    pub(crate) fn rebase(&self, base: u64, capacity: usize) -> Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if inner.live > 0 {
            candle::bail!(
                "{}: refusing to move a domain with {} live generation(s) — its \
                 ranges are named by pointers that would silently move",
                self.name,
                inner.live,
            )
        }
        inner.base = base;
        inner.capacity = capacity;
        inner.cursor = 0;
        inner.dirty = false;
        inner.census.clear();
        // The epoch is what invalidates a `WaveTicket` minted before the move.
        inner.epoch += 1;
        Ok(())
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
    if wave_census::enabled() {
        // The carve is recorded whether or not a frame could be named: a census
        // that dropped the unattributable ones would under-count the phase,
        // which is the one thing it exists to get right.
        inner.census.push(Carve {
            len,
            label: wave_census::label(),
        });
    }
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
    if wave_census::enabled() {
        // The carve is recorded whether or not a frame could be named: a census
        // that dropped the unattributable ones would under-count the phase,
        // which is the one thing it exists to get right.
        inner.census.push(Carve {
            len,
            label: wave_census::label(),
        });
    }
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
        // The tier's reference count moves on **every** drop, not only the ones
        // that reset a cursor: a clean generation still held the tier open.
        let release = |g: &Generation| {
            if g.domain != NOT_A_WAVE {
                release_if_last(g.domain as usize);
            }
        };
        if !should_reset {
            release(self);
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
                release(self);
                return;
            }
        }
        let mut inner = self.inner.lock().unwrap();
        // The generation is closing, so its cursor is this layer phase's whole
        // cost. Itemise it if it is the worst yet for its chain — see
        // `wave_census`.
        if wave_census::enabled() {
            let (cursor, carves, capacity) = (inner.cursor, inner.census.len(), inner.capacity);
            let seen = inner.reported_peak.entry(carves).or_insert(0);
            if cursor > *seen {
                *seen = cursor;
                wave_census::report(self.name, cursor, capacity, &inner.census);
            }
        }
        inner.census.clear();
        inner.cursor = 0;
        inner.dirty = false;
        inner.epoch += 1;
        drop(inner);
        release(self);
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
    /// Generations open across **all three** arenas.
    ///
    /// The per-arena `live` count bounds one span's cursor; this bounds the
    /// tier's *existence*. It goes 0 → 1 when a forward opens its first phase
    /// and 1 → 0 when the last guard drops, and those two moments are exactly
    /// where the tier is placed and where it vanishes.
    live_generations: usize,
    /// What the next placement should reserve per phase, set by the forward
    /// before it opens anything ([`plan_wave_transient`]).
    ///
    /// `None` until a forward has priced itself, which is the case for every
    /// path that opens a wave without going through the batched forward — tests,
    /// the migration helpers — and those get [`fallback_plan`].
    planned: Option<[usize; 3]>,
    /// Where the tier currently sits, while it exists.
    placed_at: Option<u64>,
    /// Whether that placement belongs to a **forward** or to a lone guard.
    ///
    /// [`plan_wave_transient`] reserves for the forward, so the reservation has
    /// to outlive the phase boundaries inside it and is released by the *next*
    /// forward. A caller that opens a guard without ever pricing itself — a
    /// test, a migration helper, anything at load — has no forward to hand it
    /// back, so its reservation is scoped to the guard and
    /// [`release_if_last`] returns it.
    ///
    /// Without the distinction a single unplanned `begin_wave` at load pins the
    /// tier at whatever the frontier was then, for the life of the process.
    reserved_by_forward: bool,
}

/// What to reserve per phase when a wave opens without having priced itself.
///
/// The old fixed constants. A caller that goes through the batched forward
/// always sets a real plan; this covers the paths that do not, and it is
/// deliberately the worst case rather than a guess — an under-sized fallback
/// would exhaust a span rather than merely waste one.
fn fallback_plan() -> [usize; 3] {
    [WAVE_ATTN_BYTES, WAVE_FFN_BYTES, WAVE_FORWARD_BYTES]
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
            // Created with no backing at all: the tier does not exist between
            // forwards, so there is nothing to carve until one opens.
            arenas: [
                BumpArena::detached(stream, "wave-attn"),
                BumpArena::detached(stream, "wave-ffn"),
                BumpArena::detached(stream, "wave-forward"),
            ],
            live_generations: 0,
            planned: None,
            placed_at: None,
            reserved_by_forward: false,
        }),
    };
    f(domain)
}

/// Price **and reserve** the transient tier for the forward about to run.
///
/// Called once per forward, after its KV is admitted and before it opens its
/// first phase. The three figures come from `WavePlan::phase_bytes` at the
/// admitted row count, so the tier is sized to **this** wave rather than to the
/// widest one the engine can run — a twenty-session decode needs a few megabytes
/// where the fixed constants reserved 912 MiB, and the difference is what the
/// weight side gets to hold.
///
/// # The reservation is per forward, and that is the whole point
///
/// This used to only record a size, with [`begin_wave`] deciding the address on
/// each phase and [`release_if_last`] giving the ground back when that phase's
/// last guard dropped. A forward opens and closes its phases many times, so the
/// tier's ground was released and re-taken between every one of them — and in
/// each of those gaps the region ceiling lifted, another thread could claim a
/// fresh region, and the next phase's tier landed one region further along.
///
/// A tier that walks forward mid-sweep puts layer *N*'s FFN extent and layer
/// *N+1*'s attention extent at **different** offsets, which is not the situation
/// [`Reclaim::StreamOrdered`] argues about: that argument is a phase reusing
/// *its own* span with a whole other phase's same-stream work in between. Two
/// different extents, on two streams, have nothing ordering them. It cost
/// `Q8_0 x20` every session's output while the region pool looked untouched
/// (`docs/elastic_vram_partition.md` §13b).
///
/// So the address is decided **here**, once, and held for the forward: the
/// ceiling stands across the inter-phase gaps, `begin_wave` rebases onto an
/// address it does not choose, and `release_if_last` returns the arenas to
/// nothing without returning the ground.
///
/// A forward that begins while a guard from the previous one is still live keeps
/// that reservation rather than moving it — the tier cannot move under a live
/// generation, and refusing here would fail a forward for a bookkeeping reason.
/// Hand back the previous forward's tier **before this one claims its KV.**
///
/// The tier's ground is deliberately held past the guard's drop
/// ([`release_if_last`]) so a region freed between forwards cannot be handed to
/// someone else and move the next tier. [`plan_wave_transient`] is what returns
/// it — and that is one phase too late.
///
/// A forward runs admit *first*: §7 phase 1 claims every KV slot the wave will
/// write so the frontier is final when phase 2 prices the tier against it. But
/// the previous tier is still standing during phase 1, and
/// [`super::region_pool::claim_region`] caps the pool at its base — so admit is
/// refused by a reservation belonging to a forward that has already finished.
///
/// While every wave succeeds this is invisible: the tier is re-placed each time
/// at the same width, and admit's claims fit below it. It becomes fatal the
/// moment one wave fails. The failed wave's tier stays, the retry's admit is
/// refused by it, that attempt fails too, and the engine spins — which is
/// exactly what the daemon did, at 1,500 refusals a second, with `class 4096 B`
/// named in the error because active K is what admit claims.
///
/// So the release moves here, ahead of admit. The guard is the same one
/// `plan_wave_transient` applies: a forward that begins while a generation from
/// the previous one is still live keeps that reservation rather than moving it,
/// because the tier cannot move under a live generation.
///
/// # Every caller outside a forward, not just the next one
///
/// Admit is the *first* thing a stale tier blocks, not the only one. The
/// scheduler's relief ladder runs in the same inter-forward gap and claims
/// regions on nearly every rung, and one consequence there is worse than a
/// refused claim: the pool's region ceiling reads `transient_base` while a tier
/// is placed, and that is an *address*, fixed where the last forward put it.
/// Move the weight boundary and it does not follow — so **a placed tier makes
/// the ceiling deaf to the boundary**, and the ladder's last
/// rung, the only one that adds ground rather than recycling it, hands over
/// weight-side ground that no rung above it can reach.
///
/// The contract is therefore "call this before touching the region pool from
/// outside a forward", and the `live_generations` guard is what makes it safe to
/// state that broadly: a caller that is wrong about being outside a forward
/// finds a live generation and changes nothing.
pub fn end_wave_transient(stream: &Arc<CudaStream>) {
    let stream = stream.clone();
    let _ = with_wave_domain(&stream, |domain| {
        if domain.live_generations > 0 {
            return Ok(());
        }
        if domain.placed_at.take().is_some() {
            release_transient(&stream);
            domain.reserved_by_forward = false;
        }
        Ok(())
    });
}

/// Price and reserve this forward's transient tier.
///
/// # Three lock scopes, and the middle one is deliberate
///
/// The release and the record happen under the wave-domain lock; the
/// **placement does not**, and it must not. [`place_transient`] buys ground from
/// the weight side when live arenas hold the tier's footprint, that purchase
/// blocks on the expert pipeline thread, and the boundary move it performs calls
/// [`wave_is_live`] — which takes this very lock. Holding it across the purchase
/// deadlocks the two threads against each other.
///
/// So there is a window between releasing the old tier and placing the new one
/// in which another thread may claim a region into the ground the placement
/// wants. That is not a hazard, it is the same case the placement already
/// handles: it buys what the claim took. The window is precisely what the old
/// `tier_reserve` existed to close, and closing it that way cost more than
/// leaving it open — a standing withholding of the last tier's width, refusing
/// arena claims against memory nobody owned.
pub fn plan_wave_transient(stream: &Arc<CudaStream>, per_phase: [usize; 3]) -> Result<()> {
    let stream = stream.clone();
    // The previous forward's ground goes back before this one's frontier is
    // measured, so the reservation is taken against a partition that already
    // reflects everything admit claimed.
    let place = with_wave_domain(&stream, |domain| {
        domain.planned = Some(per_phase);
        if domain.live_generations > 0 {
            return Ok(false);
        }
        if domain.placed_at.take().is_some() {
            release_transient(&stream);
        }
        Ok(true)
    })?;
    if !place {
        return Ok(());
    }
    // Outside the lock. A failure here leaves no tier standing and
    // `reserved_by_forward` clear, so the forward that fails takes nothing with
    // it — the next one starts against the whole partition.
    let base = place_transient(&stream, per_phase.iter().sum())?;
    with_wave_domain(&stream, |domain| {
        domain.placed_at = Some(base);
        domain.reserved_by_forward = true;
        Ok(())
    })
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
    let stream = stream.clone();
    // **Any placement happens before the lock is taken.** [`place_transient`] can
    // buy ground from the weight side, that purchase blocks on the expert
    // pipeline thread, and the boundary move it performs re-enters this very
    // mutex through [`wave_is_live`]. Placing from inside the closure below
    // deadlocks the two threads against each other —
    // [`plan_wave_transient`] is restructured for the same reason.
    //
    // Only the unplanned caller reaches it: a guard opened with no forward
    // behind it (a test, a migration helper, a `rows == 0` forward that skipped
    // phase 2) owns its own reservation, and this is where that is taken.
    let unplanned_base = match with_wave_domain(&stream, |domain| {
        Ok((domain.live_generations == 0 && domain.placed_at.is_none())
            .then(|| domain.planned.unwrap_or_else(fallback_plan)))
    })? {
        Some(plan) => Some(place_transient(&stream, plan.iter().sum())?),
        None => None,
    };
    with_wave_domain(&stream, |domain| {
        let arena = &domain.arenas[phase_index(phase)];
        if arena.is_live() {
            candle::bail!(
                "wave domain: the {phase:?} span already has a live generation. A \
                 layer leaves one phase before entering the next, so this is a \
                 phase re-entered while still open — the inner guard's drop would \
                 reset the span the outer one is still handing out."
            )
        }
        // **The arenas come into existence here; the ground they sit on does
        // not.** [`plan_wave_transient`] reserved that once for the whole
        // forward, so this only lays the three spans out inside it — at an
        // address this function does not choose and cannot change. That is what
        // keeps every phase of a sweep at the same offsets.
        //
        // **The plan is the size.** `WavePlan` prices this wave's buffers from
        // the model's geometry, and every variant in it was read off
        // `super::wave_census` on a real run rather than inferred, so a
        // twenty-session decode reserves the few megabytes it needs where the
        // fixed constants reserved 912 MiB — and a wide prefill reserves more
        // than 912 MiB rather than silently spilling the tail of its expert
        // chain onto the pool, which is what the constants were doing.
        //
        // The fallback covers callers that never priced themselves — tests, the
        // migration helpers — and it is deliberately the old worst case rather
        // than a guess. Those callers never reached `plan_wave_transient`
        // either, so this is also where their reservation is taken.
        if domain.live_generations == 0 {
            let plan = domain.planned.unwrap_or_else(fallback_plan);
            let base = match domain.placed_at {
                Some(base) => base,
                None => {
                    // No forward priced this one, so nothing will hand the
                    // ground back for it — the reservation is the guard's and
                    // drops with it. Placed above, outside the lock.
                    //
                    // Absent only if the domain gained a generation or a tier
                    // between that placement and this line, which would mean
                    // laying these spans out on ground priced for a different
                    // occupant. Refuse rather than corrupt (principle 7).
                    let base = unplanned_base.ok_or_else(|| {
                        candle::Error::Msg(
                            "wave domain: the tier's ground was priced with no \
                             generation live and one appeared before it could be \
                             laid out. The spans this guard would hand out are not \
                             the ones that were reserved."
                                .into(),
                        )
                    })?;
                    domain.placed_at = Some(base);
                    domain.reserved_by_forward = false;
                    base
                }
            };
            let mut at = base;
            for (i, arena) in domain.arenas.iter().enumerate() {
                arena.rebase(at, plan[i])?;
                at += plan[i] as u64;
            }
        }
        // Count only once the guard exists. The count is what `release_if_last`
        // decrements on drop, so incrementing before a fallible call would strand
        // the tier permanently if that call ever failed — there would be no guard
        // to drop and nothing to bring the count back down.
        let generation = arena.generation(ordinal, arena_idx)?;
        domain.live_generations += 1;
        Ok(generation)
    })
}

/// Release the tier if `ordinal`'s last wave generation has just dropped.
///
/// Called from [`Generation::drop`]. Reference-counted rather than scoped
/// because a forward's head span is *returned to its caller* — the tier's
/// lifetime is the union of the guards, not any one function's body.
fn release_if_last(ordinal: usize) {
    let mut map = match wave_domains().lock() {
        Ok(m) => m,
        Err(e) => e.into_inner(),
    };
    let Some(domain) = map.get_mut(&ordinal) else {
        return;
    };
    domain.live_generations = domain.live_generations.saturating_sub(1);
    if domain.live_generations > 0 {
        return;
    }
    // Nothing is live in any of the three spans, so detach them — a cursor that
    // still names an address it no longer owns is the one thing a bump allocator
    // must not have.
    //
    // **A forward's ground stays reserved.** This runs at every phase boundary,
    // not only at the end of a forward: a layer's attention guard drops before
    // its FFN guard opens, and the head span opens after the last layer's has
    // gone. Returning the reservation there would lift the region ceiling in
    // each of those gaps, let another thread claim into it, and move the next
    // phase's tier — which is exactly the failure §13b records. The reservation
    // belongs to the forward, and [`plan_wave_transient`] is what returns it.
    //
    // **A lone guard's ground does not.** A caller that never priced itself has
    // no forward to hand it back, so leaving the reservation standing pins the
    // tier for the life of the process — at whatever the frontier happened to be
    // when some helper opened a span at load.
    for arena in &domain.arenas {
        let _ = arena.rebase(0, 0);
    }
    if !domain.reserved_by_forward && domain.placed_at.take().is_some() {
        release_transient(&stream_of(domain));
    }
}

/// The stream a wave domain was built against.
///
/// Every arena in a domain is created from the same stream, so any of them
/// answers — arena 0 by convention.
fn stream_of(domain: &WaveDomain) -> Arc<CudaStream> {
    domain.arenas[0].inner.lock().unwrap().stream.clone()
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

/// Whether any wave generation is open on `ordinal`.
///
/// **The one thing the moving boundary must never do is move mid-wave.** A
/// retraction evicts experts and relocates others, and a wave in flight may be
/// reading either; the design's answer is that the boundary moves only at the
/// expert pipeline's end-of-pass, where no GEMM for the pass is still being
/// issued. That is a structural property of where `renegotiate_boundary` is
/// called from, and this is what lets `set_weight_floor` check it rather than
/// trust it (principle 7: refuse rather than corrupt).
pub fn wave_is_live(ordinal: usize) -> bool {
    wave_domains()
        .lock()
        .map(|map| {
            map.get(&ordinal)
                .is_some_and(|d| d.arenas.iter().any(|a| a.is_live()))
        })
        .unwrap_or(false)
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
/// Separate from the wave domains so the persistence thread can never reset a
/// buffer the scheduler is still reading, and vice versa — and carved from the
/// **fixed** left block rather than the floating wave tier, because its ranges
/// live on a copy stream whose schedule has nothing to do with a wave's.
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
