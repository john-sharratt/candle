//! The bump arena each transient domain carves its device ranges from.
//!
//! Every domain of the transient tier is a `BumpArena`: a cursor over one
//! contiguous range that moves forward within a generation and rewinds when
//! the last generation closes. The constructor a domain is built with says
//! which ground backs it: the persistence block, the wave tier placed for each
//! forward, or ground a guest's drain has claimed. A carve past the capacity
//! is refused, and alignment applies to the absolute address, so the span's
//! base does not change what a kernel receives.

use super::*;

/// One domain's transient span.
#[derive(Clone)]
pub(crate) struct BumpArena {
    pub(super) inner: Arc<Mutex<Inner>>,
    pub(super) name: &'static str,
    /// `Some` for a domain several threads stage through: see [`Turn`].
    pub(super) turn: Option<Arc<Turn>>,
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
                carry: 0,
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
            // The persistence block is reached from more than one thread.
            turn: Some(Turn::new()),
        })
    }

    /// An arena over a range the caller already owns.
    ///
    /// The third way a domain gets its backing, and the only one that does not
    /// come from the transient tier. [`BumpArena::new`] carves the persistence
    /// block; [`BumpArena::detached`] is placed into the wave tier per forward.
    /// A co-resident guest is neither: its drain runs *between* waves with the
    /// tier already handed back, so there is no tier to place into. What it does
    /// hold is its own claimed ground — span regions the drain took from the KV
    /// side and keeps until it ends — and that is what this borrows.
    ///
    /// The range therefore outlives every generation opened on it and dies with
    /// the guest's ground, which is why nothing here carves or releases: the
    /// caller's `GuestGround` owns the memory and this only hands out cursors
    /// inside it.
    pub(crate) fn over(
        stream: &Arc<CudaStream>,
        name: &'static str,
        base: u64,
        capacity: usize,
        reclaim: Reclaim,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                base,
                capacity,
                cursor: 0,
                carry: 0,
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
            // A guest's drain stages on its own thread alone.
            turn: None,
        }
    }

    /// Open a generation reserving `planned_span` bytes at the base of the span
    /// for layout slots.
    ///
    /// The cursor starts above the reserved region, so bumped ranges and planned
    /// slots partition the span between them instead of competing for it.
    /// Refuses a reservation larger than the arena, since every slot in it would
    /// otherwise address memory the domain does not own.
    pub(crate) fn generation(&self, domain: u32, arena: u32) -> Result<Generation> {
        // Before the cursor is read: the entry mark must be taken with the
        // domain already this thread's.
        let turn = self.turn.as_ref().map(Turn::enter);
        let mut inner = self.inner.lock().unwrap();
        if inner.cursor > inner.peak {
            inner.peak = inner.cursor;
        }
        let entry_cursor = inner.cursor;
        let entry_carry = inner.carry;
        inner.live += 1;
        drop(inner);
        Ok(Generation {
            inner: Arc::clone(&self.inner),
            name: self.name,
            domain,
            arena,
            entry_cursor,
            entry_carry,
            _turn: turn,
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
                carry: 0,
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
            // Opened by the wave thread only.
            turn: None,
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
        // **Where the arena is placed is the only bound its carves have.**
        //
        // `bump` refuses a carve past `capacity`, so every range this domain
        // hands out lies inside `[base, base + capacity)` by construction. That
        // makes the placement, not the carve, the thing that can be wrong: an
        // arena rebased over the expert weight zone passes its own bound check
        // on every allocation while handing out weights, and the activation
        // written through the range replaces them.
        //
        // The two share one reservation — arenas carve upward from `span_base`,
        // the weight zone downward from `span_end` — so an overlap is a
        // placement that crossed `weight_floor`, not an impossible address.
        // Checked once here rather than per carve: one comparison at a rebase
        // covers every range the domain will ever produce.
        // Diagnostic only, and gated so the production build evaluates none of
        // it. `forbid_write` is a no-op without the feature, but its ARGUMENTS
        // are still evaluated — a `format!` here allocated a `String` on every
        // rebase, which is once per layer phase per wave, in a build that then
        // threw it away.
        #[cfg(feature = "tensor-assert")]
        if candle::readonly_regions::hits(base, capacity) {
            // The partition as it stands at the violation, not as it stood at
            // load: an arena inside the declared region is only a fault if it
            // is also at or above the LIVE `weight_floor`. The weight side
            // releases ground legitimately, and a region declared once at load
            // goes stale the moment it does.
            let l = super::super::region_pool::span_layout(inner.stream.context().ordinal());
            log::error!(
                "arena {} rebase [{:#x}, {:#x}) hits the declared weight region. \
                 Live partition: {l:?}",
                self.name,
                base,
                base + capacity as u64,
            );
            if l.is_some_and(|l| base + capacity as u64 > l.weight_floor) {
                candle::readonly_regions::forbid_write(
                    &format!("bump arena rebase ({})", self.name),
                    base,
                    capacity,
                );
            }
        }
        inner.base = base;
        inner.capacity = capacity;
        inner.cursor = 0;
        inner.carry = 0;
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
/// The relative offset of the next carve, aligned as an **absolute address**.
///
/// `(base + cursor)` is what the kernel dereferences, so that is what alignment
/// has to hold on — rounding the offset alone assumes the span's base is itself
/// aligned, and the bases are not: `begin_wave` lays the three phase spans out
/// back-to-back at the plan's raw byte counts, so any unaligned plan entry
/// shifts every span after it. On the dense Qwen3 plan that put the FFN span at
/// an address ≡ 8 (mod 16), and every "256-aligned" activation carved from it
/// inherited the skew — which the tensor-core kernels' 16-byte `cp.async`
/// reads fault on (`CUDA_ERROR_MISALIGNED_ADDRESS`). The plans are also
/// normalised at recording so this costs no padding in practice; doing it here
/// as well makes the guarantee local to the carve instead of hostage to every
/// layout site.
pub(super) fn aligned_start(base: u64, cursor: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two(), "alignment must be a power of two");
    let align = align as u64;
    let abs = base + cursor as u64;
    let aligned = (abs + align - 1) & !(align - 1);
    (aligned - base) as usize
}

pub(super) fn bump<'a>(
    inner: &'a Mutex<Inner>,
    name: &'static str,
    len: usize,
    align: usize,
) -> Result<BumpRange<'a>> {
    let mut inner = inner.lock().unwrap();
    let start = aligned_start(inner.base, inner.cursor, align);
    let end = start
        .checked_add(len)
        .ok_or_else(|| candle::Error::Msg(format!("{name}: bump allocation overflowed usize")))?;
    if end > inner.capacity - inner.carry {
        // Itemise the generation that ran out, not just the carve that noticed.
        // A span exhausts because the plan is short, and "short by what" is the
        // only question worth asking here — the ordinary report fires when a
        // generation *completes* and sets a new high-water mark, which this one
        // never will.
        if wave_census::enabled() {
            wave_census::report(name, inner.cursor, inner.capacity, &inner.census);
        }
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
    if end + inner.carry > inner.peak {
        inner.peak = end + inner.carry;
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

/// [`bump`] without the borrow, for the ticket resolver, against a lock the
/// caller already holds.
///
/// [`BumpRange`] ties its lifetime to the mutex guard, which is exactly right at
/// a call site holding a [`Generation`] and useless here: the resolver is called
/// from candle-core, which has no guard to borrow from and gets its bound from
/// the `'w` already on the operand instead. Under the caller's guard so it can
/// *decide* and carve without dropping it in between — see
/// [`resolve_wave_alloc`], where releasing the lock between reading `live` and
/// bumping would leave a window for the generation to close underneath it.
/// `None` is an exhausted span, which the resolver reports as such.
pub(super) fn bump_locked(
    inner: &mut Inner,
    name: &'static str,
    len: usize,
    align: usize,
) -> Option<u64> {
    let start = aligned_start(inner.base, inner.cursor, align);
    let end = start.checked_add(len)?;
    if end > inner.capacity - inner.carry {
        // The generation that ran out is the one worth itemising: what it had
        // carved, and the request that did not fit.
        if wave_census::enabled() {
            let mut carves = inner.census.clone();
            carves.push(Carve {
                len,
                label: wave_census::label(),
            });
            wave_census::report(name, end, inner.capacity, &carves);
        }
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
    if end + inner.carry > inner.peak {
        inner.peak = end + inner.carry;
        log::debug!("{}: transient peak {} B of {} B", name, end, inner.capacity);
    }
    Some(inner.base + start as u64)
}

/// Carry `len` bytes at the top of the span, every byte of it above `above`.
///
/// `above` is the stage's high-water: its result still sits in the range the
/// stage released, and a carry that reached down into it would be written by
/// the copy it is the destination of. `None` when the two ends would meet.
pub(super) fn carry_locked(
    inner: &mut Inner,
    len: usize,
    align: usize,
    above: usize,
) -> Option<u64> {
    let top = (inner.capacity - inner.carry).checked_sub(len)?;
    let abs = (inner.base + top as u64) & !(align as u64 - 1);
    let start = (abs.checked_sub(inner.base)?) as usize;
    if start < above.max(inner.cursor) {
        return None;
    }
    inner.carry = inner.capacity - start;
    inner.dirty = true;
    if inner.cursor + inner.carry > inner.peak {
        inner.peak = inner.cursor + inner.carry;
    }
    Some(abs)
}

pub(super) fn domains() -> &'static Mutex<HashMap<usize, BumpArena>> {
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
