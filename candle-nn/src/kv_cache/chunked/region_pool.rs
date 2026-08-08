//! The device reservation and its two sides.
//!
//! One VA span per device, claimed once and never given back
//! (`docs/archived/arena_unification.md` §3.1). It is divided at a fixed,
//! region-aligned boundary:
//!
//! ```text
//! [ region 0 | region 1 | ... | region k-1 ]  [ transient tier ]
//!   KV side, 16 MiB each, lowest-first         bump domains
//! ```
//!
//! **The KV side** hands out whole regions. `create_arena` takes one,
//! `release_arena` gives it back, and neither touches the CUDA allocator: a
//! region is an address, and the memory behind it was claimed at startup. That
//! is what makes an arena base pointer permanently valid — the property the
//! arena-topology guard used to defend with a process-global lock.
//!
//! **The transient tier** is carved into the per-domain bump spans of
//! [`super::bump_arena`]. It sits at the right end because that is the edge a
//! moving boundary would grow into; the boundary is fixed in this step, so the
//! layout is what carries the segregation (§9 S6).
//!
//! # Sizing
//!
//! The whole reservation — both sides — is `kv_floor`, the governor's partition
//! knob, and the KV side is what is left of it once the transient tier is taken
//! out. The cushion the governor also holds back (`scratch_margin`) stays
//! *outside* the reservation, for the allocations still served by the CUDA pool:
//! the gallery arena, the grow-only scratches, the threaded expert pipeline's
//! combine target.
//!
//! The claim is then made granule by granule with a write to each one, so the
//! reservation ends wherever the driver actually refuses rather than wherever a
//! subtraction predicted it would. Measuring and claiming stay one act — and
//! because the ask is now honest, a refusal *means* something: it says the
//! governor's `usable()` over-read the card, which is a fact worth logging
//! rather than the expected outcome of asking for more than exists.

use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use candle::cuda_backend::cudarc::driver::result::memset_d8_sync;
use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::vram::AllocClass;
use candle::Result;

use super::chunk_ops::MIGRATION_STAGING_CAP_BYTES;
use super::reservation::Reservation;
use super::types::TARGET_ARENA_BYTES;

/// One region of the KV side. Every size class carves its arenas at this size.
pub const REGION_BYTES: usize = TARGET_ARENA_BYTES;

/// The transient tier's share of the span: `S = 2·W_wave + W_persist` (§3.6),
/// built from each domain's own budget so the two cannot drift apart.
///
/// **Every byte here is a byte the KV side does not get**, and on a 16 GiB card
/// the KV side is what the expert cache competes with, so this is sized to what
/// the domains need and never to what looks safe. Both terms are held against a
/// measurement (Qwen3-30B-A3B, batch 64):
///
/// | term | reserved | measured peak |
/// |---|---|---|
/// | `2·W_wave` | 128 MiB | 61.6 MiB — 30.8 MiB per half |
/// | `W_persist` | 64 MiB | 29,696 B, but bounded by the batch cap it spans |
///
/// The wave halves keep ~2x headroom deliberately: the peak scales with the
/// widest prefill a wave admits, and exhausting a half fails the forward. The
/// persistence term is a declared budget rather than a watermark — a batch
/// bisects itself to fit it, so the span buys DtoH syncs on the hot→warm path,
/// not correctness.
///
/// There is no shelf term. It reserved 64 MiB for a static-shelf allocator that
/// was never built — the sampler, provenance and MoE routing scratches still
/// grow from the CUDA pool, where `scratch_margin` already covers them — so it
/// was 4 regions of KV backing nothing. Reinstate it when something allocates
/// from it, priced in regions like everything else here.
const TRANSIENT_SPAN_BYTES: usize =
    2 * super::bump_arena::WAVE_HALF_BYTES + MIGRATION_STAGING_CAP_BYTES;

const _: () = assert!(
    TRANSIENT_SPAN_BYTES % REGION_BYTES == 0,
    "the boundary between the two sides must be region-aligned"
);

/// The KV side when no governor has measured the card.
///
/// A fixed number rather than a fraction of what the driver reports free. The
/// fraction was tried and is wrong here: the only processes that get this far
/// without a governor are test binaries, which run hundreds of GPU tests
/// concurrently, so "half of free" depends on which tests happen to be in
/// flight when the first KV cache is built — and a run that lost the race
/// claimed nothing at all. This is large enough for the suite's peak arena
/// working set and small enough to leave the card to the tests themselves.
const TEST_KV_SPAN_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// A claimed region, released back to the free list when dropped.
///
/// Holding one is what keeps a region out of circulation, so an [`Arena`] can
/// simply own its handle and the free list stays correct through every path
/// that drops an arena — release, truncate, or the backing going away.
///
/// **Dropping this does not mean the GPU is finished with the region.** It means
/// no host-side gid names it any more; kernels launched earlier can still be
/// reading those bytes. The wait belongs to whoever re-tenants the region, and
/// [`claim_region`] is where it lives — do not add a path that hands a released
/// region to a new owner without going through it.
///
/// [`Arena`]: super::arena::Arena
#[derive(Debug)]
pub(crate) struct RegionHandle {
    ordinal: usize,
    index: usize,
    base: u64,
}

impl RegionHandle {
    /// Device address of the region's first byte.
    pub(crate) fn base(&self) -> u64 {
        self.base
    }

    /// Position of the region in the span. Rises left to right, so the highest
    /// index is the cheapest to evacuate (§3.10, problem 1).
    pub(crate) fn index(&self) -> usize {
        self.index
    }
}

impl Drop for RegionHandle {
    fn drop(&mut self) {
        let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pool) = map.get_mut(&self.ordinal) {
            // Stamp the quiesce epoch current at release. `claim_region` reads it
            // to decide whether this region still needs a wait.
            pool.released_epoch[self.index] = pool.quiesce_epoch;
            pool.free.push(Reverse(self.index));
            pool.live -= 1;
            log::debug!("region {} released, {} live", self.index, pool.live);
        }
    }
}

/// Occupancy of one device's KV side.
#[derive(Debug, Clone, Copy)]
pub struct RegionStats {
    /// Regions the reservation actually claimed.
    pub total: usize,
    /// Regions held by an arena right now.
    pub live: usize,
    /// Regions available without evicting anything — the pressure signal.
    pub free: usize,
    /// Most regions ever live at once — how close the partition came to full.
    pub peak_live: usize,
    /// Bytes of transient span, and of it, bytes carved into bump domains.
    pub transient_bytes: usize,
    pub transient_carved: usize,
}

struct RegionPool {
    reservation: Reservation,
    /// Regions the claim actually backed: the KV side is `[0, total)`.
    total: usize,
    /// Lowest region index never yet handed out.
    next: usize,
    /// Returned regions, lowest first (principle 5: keep live data left-packed).
    free: BinaryHeap<Reverse<usize>>,
    live: usize,
    peak_live: usize,
    /// Count of whole-device quiesces this pool has performed.
    ///
    /// A `cuCtxSynchronize` waits for every task on every stream of the context,
    /// so once it returns, **every** kernel launched before it has retired — not
    /// just the ones reading the region that triggered it. That makes one sync a
    /// blanket safety statement about all regions released before it, which is
    /// what lets the counter below turn a per-claim wait into a per-batch one.
    quiesce_epoch: u64,
    /// Per region, the epoch current when it was last released. A claim needs a
    /// wait only while this still equals [`Self::quiesce_epoch`] — otherwise a
    /// quiesce has happened since, and the region's readers are long gone.
    released_epoch: Vec<u64>,
    /// One past the last byte of the transient tier.
    transient_end: u64,
    transient_bytes: usize,
    /// Transient bytes already carved into domains. Never returned.
    transient_carved: usize,
}

fn pools() -> &'static Mutex<HashMap<usize, RegionPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, RegionPool>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The KV side's target size.
///
/// With a governor — every production load installs one — this is everything
/// left: `usable` is what remains of the balloon-measured capacity after the
/// weights and the expert cache, less the cushion the forward's own activations
/// need. Claiming the remainder is the point of consuming the memory at startup
/// rather than competing for it later.
///
/// Without one, [`TEST_KV_SPAN_BYTES`].
///
/// **A governor that fails to answer is an error, not a fallback.** These two
/// cases used to share one `_` arm, so a transient `usable()` failure on a
/// production card sized the whole KV side at the 2 GiB test constant and said
/// nothing — the reservation would come up a quarter of its intended size and
/// every downstream number would be quietly wrong. Absence of a governor is a
/// test binary; a governor that errors is a fault.
fn kv_span_target(ordinal: usize) -> Result<usize> {
    let Some(gov) = candle::vram::get(ordinal) else {
        return Ok(TEST_KV_SPAN_BYTES);
    };
    let usable = gov.usable().map_err(|e| {
        candle::Error::Msg(format!(
            "reservation: the VRAM governor could not report usable bytes ({e}). \
             Refusing to size the KV side from the {TEST_KV_SPAN_BYTES} B test \
             constant on a card that has a governor — that would silently claim a \
             fraction of the intended reservation."
        ))
    })?;
    Ok(kv_span_from(usable as usize, gov.scratch_margin() as usize))
}

/// The KV side, given what is left of `C` and the cushion to leave on the pool.
///
/// Split out from [`kv_span_target`] so the identity below can be asserted
/// without a governor.
///
/// **The transient tier comes out of this budget.** The reservation claims
/// `kv_span + TRANSIENT_SPAN_BYTES` in one act, so the transient side is part
/// of the same claim rather than something added on top of it.
///
/// It used to be added on top, and the result over-asked by the whole transient
/// span. `scratch_margin` was subtracted here as "the cushion the first
/// forward's scratch lands in" — but the forward's scratch *is* the transient
/// tier now, which is what §3.6 moved it to, so the same memory was reserved
/// twice by two places that did not know about each other. The reservation then
/// asked for `usable − margin + 704 MiB`, the granule touch refused the excess,
/// and the KV side came out at whatever the driver happened to leave. The
/// partition was being decided by the refusal point instead of by policy.
///
/// The same mistake, one layer up, is recorded against `balloon_headroom_abs`:
/// reserving the transient peak in the balloon cap as well as in
/// `expert_budget` "booked the same bytes twice".
fn kv_span_from(usable: usize, pool_reserve: usize) -> usize {
    usable
        .saturating_sub(pool_reserve)
        .saturating_sub(TRANSIENT_SPAN_BYTES)
}

impl RegionPool {
    fn create(stream: &std::sync::Arc<CudaStream>) -> Result<Self> {
        let ordinal = stream.context().ordinal();
        // Read before the claim: mapping granules consumes headroom, so asking
        // afterwards reports what is left rather than what was available.
        let usable_before = candle::vram::get(ordinal)
            .and_then(|g| g.usable().ok())
            .unwrap_or(0) as usize;
        let want_regions = kv_span_target(ordinal)? / REGION_BYTES;
        let kv_span = want_regions * REGION_BYTES;
        let mut reservation = Reservation::reserve(stream, kv_span + TRANSIENT_SPAN_BYTES)?;

        // The transient tier is claimed first even though it sits to the right.
        // A forward that cannot allocate its intermediates cannot run at all,
        // while a KV side that comes up short merely holds fewer sequences — so
        // if the card can only satisfy one of them, it must be this one.
        let transient = reservation.map_range(kv_span, TRANSIENT_SPAN_BYTES)?;
        if transient < TRANSIENT_SPAN_BYTES {
            candle::bail!(
                "reservation: the card refused the {TRANSIENT_SPAN_BYTES} B transient tier \
                 after {transient} B — there is not enough VRAM left to run a forward"
            )
        }
        let claimed = reservation.map_range(0, kv_span)?;
        let total = claimed / REGION_BYTES;
        // The partition, on the one channel that survives a test binary (which
        // installs no tracing subscriber, so the log lines below are invisible
        // there). Step 7 tunes these numbers, and it needs them from the gate as
        // well as from the daemon.
        // What the governor meant KV to own, against what it actually got. The
        // two differ whenever `kv_floor` exceeds what survived the weights and
        // the expert cache, and nothing else reports it: `shortfall` measures
        // only the granule touch refusing, so a floor that was never achievable
        // reads as a clean reservation. The floor is the partition knob, so a
        // silent gap between asking and getting makes it un-tunable.
        let gov = candle::vram::get(ordinal);
        let intended = gov.as_ref().map_or(0, |g| g.kv_floor() as usize);
        let floor_deficit = intended.saturating_sub(claimed + TRANSIENT_SPAN_BYTES);
        // The class tallies go out alongside so `floor_deficit` is
        // *attributable*, not merely visible. `kv_floor` names the reserve the
        // expert budget must LEAVE, not what KV ends up owning; the two differ
        // by everything that allocated in between. Without the breakdown the gap
        // reads as an unexplained ~1 GiB, and the first thing anyone tuning from
        // first principles does is mistrust the knob. `residual` is
        // `capacity - weights - experts - usable` — pool growth, the gallery,
        // per-class overhead — and is the term to chase if the deficit moves.
        let weights = gov
            .as_ref()
            .map_or(0, |g| g.class_reserved(AllocClass::Weights) as usize);
        let experts = gov
            .as_ref()
            .map_or(0, |g| g.class_reserved(AllocClass::Expert) as usize);
        let capacity = gov.as_ref().map_or(0, |g| g.capacity() as usize);
        let residual = capacity
            .saturating_sub(weights)
            .saturating_sub(experts)
            .saturating_sub(usable_before);
        if super::alloc::arena_stats_enabled() {
            let mib = |b: usize| b / (1024 * 1024);
            eprintln!(
                "[reservation] capacity_c={}MiB usable={}MiB kv_floor={}MiB \
                 scratch_margin={}MiB | asked={}MiB claimed={}MiB ({total} regions) \
                 transient={}MiB | shortfall={}MiB floor_deficit={}MiB \
                 | weights={}MiB experts={}MiB residual={}MiB",
                mib(capacity),
                mib(usable_before),
                mib(intended),
                gov.as_ref().map_or(0, |g| mib(g.scratch_margin() as usize)),
                mib(kv_span),
                mib(claimed),
                mib(TRANSIENT_SPAN_BYTES),
                mib(kv_span.saturating_sub(claimed)),
                mib(floor_deficit),
                mib(weights),
                mib(experts),
                mib(residual),
            );
        }
        if floor_deficit > 0 {
            log::warn!(
                "reservation: KV owns {} MiB of the {} MiB `kv_floor` reserves, \
                 short by {} MiB. `kv_floor` is the reserve the expert budget \
                 must LEAVE, not what KV receives — the gap is whatever \
                 allocated between `expert_budget()` and this claim: weights \
                 {} MiB, experts {} MiB, residual {} MiB (pool growth, gallery, \
                 per-class overhead). Raising CANDLE_VRAM_KV_FLOOR_MB shrinks \
                 the expert budget by the same amount, which is the trade this \
                 knob makes.",
                (claimed + TRANSIENT_SPAN_BYTES) / (1024 * 1024),
                intended / (1024 * 1024),
                floor_deficit / (1024 * 1024),
                weights / (1024 * 1024),
                experts / (1024 * 1024),
                residual / (1024 * 1024),
            );
        }
        if claimed < kv_span {
            log::warn!(
                "reservation: KV side claimed {} MiB of the {} MiB asked for — \
                 {total} regions",
                claimed / (1024 * 1024),
                kv_span / (1024 * 1024),
            );
        } else {
            log::info!(
                "reservation: {total} regions ({} MiB) KV + {} MiB transient, \
                 {} MiB of address space in {} KiB granules",
                claimed / (1024 * 1024),
                TRANSIENT_SPAN_BYTES / (1024 * 1024),
                reservation.reserved_bytes() / (1024 * 1024),
                reservation.granularity() / 1024,
            );
        }

        let transient_end = reservation.base() + (kv_span + TRANSIENT_SPAN_BYTES) as u64;
        Ok(Self {
            reservation,
            total,
            next: 0,
            free: BinaryHeap::new(),
            live: 0,
            peak_live: 0,
            // Epoch 0 is "no quiesce yet". A region that has never been released
            // keeps its slot at 0, which is only ever read after a release
            // overwrites it, so the initial value is never load-bearing.
            quiesce_epoch: 0,
            released_epoch: vec![0; total],
            transient_end,
            transient_bytes: TRANSIENT_SPAN_BYTES,
            transient_carved: 0,
        })
    }

    fn region_base(&self, index: usize) -> u64 {
        self.reservation.base() + (index * REGION_BYTES) as u64
    }

    fn free_count(&self) -> usize {
        self.free.len() + (self.total - self.next)
    }
}

/// Run `f` against this device's pool, creating the reservation on first use.
/// Recycled-claim cost, split into its two halves because they have different
/// answers if either ever gets expensive.
///
/// The **sync** is the whole-device quiesce of [`claim_region`], amortised by
/// the epoch stamp: `quiesced` counts the claims that actually paid it. If that
/// ratio climbs back toward 1 the batching has stopped working, and the next
/// move is a per-region release event. The **fill** is a synchronous 16 MiB
/// memset; if *it* dominates the answer is to zero on release instead, off the
/// allocation path.
///
/// Reported under `KV_ARENA_STATS`, next to `[arena-create]`, so one gate run
/// with the flag set gives the whole allocation fast path.
static RECYCLE_COUNT: AtomicU64 = AtomicU64::new(0);
static RECYCLE_QUIESCED: AtomicU64 = AtomicU64::new(0);
static RECYCLE_SYNC_NS: AtomicU64 = AtomicU64::new(0);
static RECYCLE_FILL_NS: AtomicU64 = AtomicU64::new(0);
static RECYCLE_SYNC_MAX_NS: AtomicU64 = AtomicU64::new(0);

fn record_recycle(quiesced: bool, sync_ns: u64, fill_ns: u64) {
    if !super::alloc::arena_stats_enabled() {
        return;
    }
    let n = RECYCLE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let q =
        RECYCLE_QUIESCED.fetch_add(u64::from(quiesced), Ordering::Relaxed) + u64::from(quiesced);
    let sync_total = RECYCLE_SYNC_NS.fetch_add(sync_ns, Ordering::Relaxed) + sync_ns;
    let fill_total = RECYCLE_FILL_NS.fetch_add(fill_ns, Ordering::Relaxed) + fill_ns;
    RECYCLE_SYNC_MAX_NS.fetch_max(sync_ns, Ordering::Relaxed);
    let ms = |ns: u64| ns as f64 / 1_000_000.0;
    eprintln!(
        "[region-recycle] quiesced={quiesced} sync_ms={:.3} fill_ms={:.3} | n={n} \
         quiesces={q} sync_total_ms={:.1} sync_max_ms={:.3} fill_total_ms={:.1} \
         fill_avg_ms={:.3}",
        ms(sync_ns),
        ms(fill_ns),
        ms(sync_total),
        ms(RECYCLE_SYNC_MAX_NS.load(Ordering::Relaxed)),
        ms(fill_total),
        ms(fill_total) / n as f64,
    );
}

fn with_pool<R>(
    stream: &std::sync::Arc<CudaStream>,
    f: impl FnOnce(&mut RegionPool) -> Result<R>,
) -> Result<R> {
    let mut map: MutexGuard<'_, HashMap<usize, RegionPool>> =
        pools().lock().unwrap_or_else(|e| e.into_inner());
    let ordinal = stream.context().ordinal();
    let pool = match map.entry(ordinal) {
        Entry::Occupied(o) => o.into_mut(),
        Entry::Vacant(v) => v.insert(RegionPool::create(stream)?),
    };
    f(pool)
}

/// Claim this device's reservation if it does not exist yet.
///
/// Called once per KV backing so the claim is a startup cost and the free-region
/// count — which admission reads before any arena exists — always has an answer.
pub(crate) fn ensure(stream: &std::sync::Arc<CudaStream>) -> Result<()> {
    with_pool(stream, |_| Ok(()))
}

/// Take a region for an arena, or `None` when the KV side is full.
///
/// `None` is the pressure signal, not an error: the caller's response is to
/// evacuate a region (§3.8), which is the demotion the hot tier does anyway.
///
/// The region's bytes are zero on return — freshly-claimed ones from the
/// mapping touch, recycled ones from an explicit fill here, so a new tenant
/// never reads the last one's data (invariant 4).
pub(crate) fn claim_region(stream: &std::sync::Arc<CudaStream>) -> Result<Option<RegionHandle>> {
    with_pool(stream, |pool| {
        let (index, recycled) = match pool.free.pop() {
            Some(Reverse(idx)) => (idx, true),
            None if pool.next < pool.total => {
                pool.next += 1;
                (pool.next - 1, false)
            }
            None => return Ok(None),
        };
        let base = pool.region_base(index);
        if recycled {
            // **Quiesce before re-tenanting.** A region returns to the free list
            // the moment its arena drops, and that says only that no host-side
            // gid still names it — kernels launched earlier can still be reading
            // it, on the compute stream or the persistence thread's copy stream.
            // Handing those bytes to a new owner and memsetting them is then a
            // read-after-free, and it reports as `CUDA_ERROR_ILLEGAL_ADDRESS` in
            // whichever unrelated kernel happens to be in flight.
            //
            // The allocator used to supply this ordering for free: an arena's
            // slab was released with `cuMemFreeAsync` and re-allocated with
            // `cuMemAllocAsync` on the same stream, so reuse could not overtake
            // the reads. A free-list push has no such ordering, so the wait is
            // explicit — device-wide, because the copy stream is one of the
            // readers and draining the compute stream alone would not cover it.
            //
            // **But at most once per batch of releases.** A quiesce retires every
            // kernel launched before it, on every stream, so it discharges the
            // debt of *all* regions released up to that point — not just the one
            // that triggered it. Regions come back in bulk (an eviction pass
            // drops a turn's arenas together) and are re-claimed in bulk, so the
            // epoch stamp turns what was a wait per claim into a wait per batch.
            //
            // This was measured, not assumed. Per-claim, the gate run paid
            // **2,837 ms** across 395 recycled claims — 7.2 ms average, 35.3 ms
            // worst — which is 2.4 % of the run spent stalled on an allocation.
            // (Step 4 measured this as free, at 0.029 ms per arena creation. It
            // was: back then the scheduler synchronised the device every wave in
            // `release_empty_arenas`, so the queue was always shallow here. Step
            // 5 removed that sweep-path sync — correctly, it was guarding an
            // unmap that no longer happens — and this became the real cost of
            // the ordering rather than a free ride on someone else's.)
            //
            // SAFETY: the region is inside the reservation and mapped, and no
            // kernel is reading it — either because the sync below just retired
            // them all, or because an earlier one did and nothing has re-read
            // the region since (it has been on the free list throughout).
            // Bound unconditionally: the fill below needs a current context on
            // this thread whether or not the sync ran.
            let bound = stream.context().bind_to_thread();
            let needs_sync = pool.released_epoch[index] == pool.quiesce_epoch;
            let t_sync = Instant::now();
            let synced = bound.and_then(|()| {
                if !needs_sync {
                    return Ok(());
                }
                let r = stream.context().synchronize();
                if r.is_ok() {
                    pool.quiesce_epoch += 1;
                }
                r
            });
            let sync_ns = t_sync.elapsed().as_nanos() as u64;
            let t_fill = Instant::now();
            let zeroed = synced.and_then(|()| unsafe { memset_d8_sync(base, 0, REGION_BYTES) });
            record_recycle(needs_sync, sync_ns, t_fill.elapsed().as_nanos() as u64);
            if let Err(e) = zeroed {
                // Put it back: a region that could not be cleaned is still the
                // pool's, and losing it here would shrink the span silently.
                // Its epoch stamp stands, so the retry re-evaluates the wait
                // against whatever has happened in the meantime.
                pool.free.push(Reverse(index));
                candle::bail!("recycling region {index}: {e}")
            }
        }
        pool.live += 1;
        pool.peak_live = pool.peak_live.max(pool.live);
        // Paired with the release log below: a fault correlated between a
        // release of region N and its re-claim is the re-tenancy signature, and
        // the region index is the only handle a log has on which bytes those
        // were.
        log::debug!(
            "region {index} claimed ({}{}), {} live of {}",
            if recycled { "recycled" } else { "fresh" },
            if recycled { ", zeroed" } else { "" },
            pool.live,
            pool.total,
        );
        Ok(Some(RegionHandle {
            ordinal: stream.context().ordinal(),
            index,
            base,
        }))
    })
}

/// Carve a bump domain's span out of the transient tier.
///
/// Carved downward from the right end, once per domain and never returned, so
/// every domain's address is fixed for the process lifetime — which is what
/// lets a `BumpRange` be a bare pointer. Its `'w` bounds when the range may be
/// handed out again, not whether the address is mapped.
pub(crate) fn carve_transient(stream: &std::sync::Arc<CudaStream>, bytes: usize) -> Result<u64> {
    with_pool(stream, |pool| {
        let len = bytes.div_ceil(REGION_BYTES) * REGION_BYTES;
        if pool.transient_carved + len > pool.transient_bytes {
            candle::bail!(
                "transient tier exhausted: {len} B on top of {} B carved exceeds the {} B span",
                pool.transient_carved,
                pool.transient_bytes,
            )
        }
        pool.transient_carved += len;
        Ok(pool.transient_end - pool.transient_carved as u64)
    })
}

/// Occupancy of a device's KV side, or `None` if it has no reservation yet.
pub fn region_stats(ordinal: usize) -> Option<RegionStats> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| RegionStats {
        total: pool.total,
        live: pool.live,
        free: pool.free_count(),
        peak_live: pool.peak_live,
        transient_bytes: pool.transient_bytes,
        transient_carved: pool.transient_carved,
    })
}

#[cfg(test)]
mod tests {
    use super::{claim_region, region_stats, REGION_BYTES};
    use candle::{Device, Result};
    use std::sync::Arc;

    use candle::cuda_backend::cudarc::driver::CudaStream;

    /// The pool is process-global and cargo runs tests in parallel, so region
    /// counts are only stable while one test at a time is looking at them — and
    /// that includes tests outside this module, which is why it is the
    /// crate-wide lock and not a local `static`.
    use super::super::gpu_test_lock::gpu_serial as serial;

    fn stream() -> Option<Arc<CudaStream>> {
        match Device::new_cuda(0) {
            Ok(Device::Cuda(d)) => Some(d.cuda_stream()),
            _ => None,
        }
    }

    /// Regions come out of the span in ascending order and are disjoint at the
    /// region stride — the arithmetic every base pointer depends on.
    #[test]
    fn regions_are_disjoint_and_ascending() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let a = claim_region(&s)?.expect("a region");
        let b = claim_region(&s)?.expect("a second region");
        assert!(b.index() > a.index(), "regions ascend");
        assert_eq!(
            b.base() - a.base(),
            ((b.index() - a.index()) * REGION_BYTES) as u64,
            "region bases step by exactly the region stride"
        );
        Ok(())
    }

    /// Dropping a handle returns its region, and the next claim takes the
    /// lowest free one back — the left-packing the evacuation order relies on.
    ///
    /// Asserted as an ordering, not as two exact indices. The lock serialises
    /// this module's tests against each other, but the pool is process-global
    /// and other modules' tests claim from it in parallel, so a specific region
    /// freed here can be taken by one of them before the re-claim. What must
    /// hold regardless of that interleaving is that claims come back ascending:
    /// a concurrent thief changes *which* region is lowest, never that the
    /// lowest is the one handed out.
    #[test]
    fn a_released_region_comes_back_lowest_first() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let low = claim_region(&s)?.expect("a region");
        let high = claim_region(&s)?.expect("a second region");
        let low_idx = low.index();
        assert!(
            low.index() < high.index(),
            "claims ascend while the pool is fresh"
        );
        let before = region_stats(0).expect("a pool").free;
        drop(high);
        drop(low);
        assert_eq!(region_stats(0).expect("a pool").free, before + 2);
        let again = claim_region(&s)?.expect("a recycled region");
        assert!(
            again.index() <= low_idx,
            "a recycled region must come back no higher than the lowest just freed: \
             got {}, freed {low_idx}",
            again.index()
        );
        let then = claim_region(&s)?.expect("the other recycled region");
        assert!(
            then.index() > again.index(),
            "the free list is lowest-first: {} followed {}",
            then.index(),
            again.index()
        );
        Ok(())
    }

    /// A recycled region is handed over zeroed, whatever its last tenant left.
    #[test]
    fn a_recycled_region_is_zeroed() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let region = claim_region(&s)?.expect("a region");
        let base = region.base();
        let dirt = vec![0xC3u8; 8192];
        // SAFETY: the region is claimed by this test and mapped read/write.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                base,
                &dirt,
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("dirtying a region: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        drop(region);

        let again = claim_region(&s)?.expect("the recycled region");
        assert_eq!(again.base(), base, "the same region should come back");
        let mut back = vec![0xFFu8; 8192];
        // SAFETY: as above, reading the region this test holds.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                &mut back,
                base,
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("reading a region: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        assert!(
            back.iter().all(|&b| b == 0),
            "a recycled region still held its last tenant's bytes"
        );
        Ok(())
    }

    /// A batch of regions released together costs **one** quiesce between them,
    /// not one each — the whole point of the epoch stamp. The claim that pays it
    /// advances the epoch, which discharges every other region released before
    /// that moment.
    ///
    /// Asserted through the zero guarantee rather than by reading the counter,
    /// because the guarantee is what the skip must not break: a region that
    /// takes the fast path is still handed over clean.
    #[test]
    fn one_quiesce_covers_a_whole_batch_of_releases() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };

        let batch: Vec<_> = (0..4)
            .map(|_| claim_region(&s).unwrap().expect("a region"))
            .collect();
        let bases: Vec<u64> = batch.iter().map(|r| r.base()).collect();
        let dirt = vec![0x5Au8; 4096];
        for &base in &bases {
            // SAFETY: each region is claimed by this test and mapped read/write.
            unsafe {
                candle::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                    base,
                    &dirt,
                    s.cu_stream(),
                )
            }
            .map_err(|e| candle::Error::Msg(format!("dirtying a region: {e}")))?;
        }
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        drop(batch);

        // Re-claim all four. The first pays the quiesce; the rest skip it — and
        // every one of them must still come back zeroed.
        let again: Vec<_> = (0..4)
            .map(|_| claim_region(&s).unwrap().expect("a region"))
            .collect();
        for region in &again {
            let mut back = vec![0xFFu8; 4096];
            // SAFETY: as above, reading a region this test holds.
            unsafe {
                candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_async(
                    &mut back,
                    region.base(),
                    s.cu_stream(),
                )
            }
            .map_err(|e| candle::Error::Msg(format!("reading a region: {e}")))?;
            s.synchronize()
                .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
            assert!(
                back.iter().all(|&b| b == 0),
                "region {} skipped its quiesce and kept the last tenant's bytes",
                region.index()
            );
        }
        Ok(())
    }

    /// **The whole reservation is exactly `kv_floor`.**
    ///
    /// The governor hands the expert cache `usable − kv_floor − scratch_margin`,
    /// so what is left when the first KV cache is built is
    /// `kv_floor + scratch_margin`. Out of that, the reservation claims its two
    /// sides and leaves the cushion on the CUDA pool — where the gallery arena,
    /// the grow-only scratches and the threaded pipeline's combine target still
    /// live. The identity is what makes `kv_floor` mean one thing:
    /// **the VRAM the KV subsystem owns, transient tier included.**
    ///
    /// A pure-arithmetic test, so it holds on a machine with no GPU.
    #[test]
    fn the_reservation_claims_exactly_the_kv_floor() {
        let margin = 1024 * 1024 * 1024;
        for floor_mib in [3072usize, 5120, 6144] {
            let floor = floor_mib * 1024 * 1024;
            let usable = floor + margin; // what the expert loader leaves behind
            let claim = super::kv_span_from(usable, margin) + super::TRANSIENT_SPAN_BYTES;
            assert_eq!(
                claim,
                floor,
                "a {floor_mib} MiB floor must claim exactly that, not {} MiB",
                claim / (1024 * 1024)
            );
        }
    }

    /// A card too small to hold the transient tier asks for no KV rather than
    /// wrapping into a colossal span.
    #[test]
    fn a_span_smaller_than_the_transient_tier_yields_no_kv() {
        assert_eq!(super::kv_span_from(super::TRANSIENT_SPAN_BYTES / 2, 0), 0);
        assert_eq!(super::kv_span_from(0, 1024), 0);
    }

    /// Live + free accounts for every region: nothing leaks out of the span.
    #[test]
    fn every_region_is_either_live_or_free() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let held = claim_region(&s)?.expect("a region");
        let stats = region_stats(0).expect("a pool");
        assert_eq!(stats.live + stats.free, stats.total);
        assert!(stats.live >= 1);
        drop(held);
        let after = region_stats(0).expect("a pool");
        assert_eq!(after.live + after.free, after.total);
        Ok(())
    }
}
