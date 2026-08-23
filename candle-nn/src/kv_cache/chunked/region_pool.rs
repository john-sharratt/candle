//! The device reservation and the four things that live in it.
//!
//! One VA span per device, claimed once and never given back, covering
//! **everything the engine owns except the dense weights**
//! (`docs/elastic_vram_partition.md` §2):
//!
//! ```text
//!            while a forward runs:                between forwards:
//!  span_base            T        W    span_end     span_base       W    span_end
//!      ├────────┬───────┼────────┼────────┤            ├────────┬──────┼────────┤
//!      │persist │regions│transnt │ expert │            │persist │regions│ expert │
//!      │ fixed  │  ───► │        │ ◄───   │            │ fixed  │  ───► │ ◄───   │
//!      └────────┴───────┴────────┴────────┘            └────────┴──────┴────────┘
//! ```
//!
//! **The persistence staging block** is the one span with a fixed address. The
//! persistence thread stages on its own copy stream, on a schedule that has
//! nothing to do with a wave's, so its ranges can be live at any moment a
//! forward begins — it therefore sits at the far left, where nothing reaches it.
//!
//! **The KV side** hands out whole regions, lowest-index-first, so live arenas
//! stay left-packed and `W` can move. `create_arena` takes one, `release_arena`
//! gives it back, and neither touches the CUDA allocator: a region is an
//! address, and the memory behind it was claimed at startup.
//!
//! **The wave transient tier vanishes between forwards.** It is placed by
//! [`place_transient`] at the width *this* wave prices, immediately left of `W`,
//! and released when the last generation drops. That is what makes it the only
//! variable-size block in the span: at the moment its extent changes it holds
//! nothing, so a resize leaves no hole and moves no data. A block at a fixed
//! address could not do this — making it variable there would either strand a
//! gap between it and the arenas, or move the arena base and relocate every
//! region. It has to be the block adjacent to the moving boundary.
//!
//! **The weight side** is expert slots, filled right-to-left
//! ([`super::weight_zone`]). It is owned by the expert cache; this module knows
//! only `W`, its leftmost edge, which [`set_weight_floor`] places at load and
//! moves between waves.
//!
//! **`W` is the only boundary that moves**, and it moves in one place, at one
//! time: the expert pipeline thread, where no expert GEMM for the pass is still
//! being issued. The KV side never moves it directly — it *buys* ground through
//! [`set_ground_broker`], which sends the request to that thread and blocks on
//! the answer, so the eviction still happens where it is safe while the
//! arithmetic stays with the claim that knows the number.
//!
//! Outside a wave that purchase is the KV side's whole answer to running out: the
//! span is one reservation, the cold tier holds a valid copy of every expert, and
//! so a region can always be had for the price of a reload. The only refusals
//! left are a tier standing over the ground (which no concession can lift — the
//! wave must narrow) and the weight zone's own floor, the fewest slots the expert
//! cache can serve a token with.
//!
//! # Sizing
//!
//! The span is everything `usable()` reports at the moment it is claimed, less
//! the cushion left to the CUDA pool. Nothing is subtracted for the dense
//! weights, and that is not an oversight: they are already resident when this
//! runs, so `usable()` — which counts the *drop in headroom since `C` was
//! measured* — has already netted them out. Subtracting them again would be the
//! same bytes booked twice, which this codebase has done twice before
//! (`balloon_headroom_abs` against `expert_budget`; `scratch_margin` against the
//! transient tier, "same bytes, two places, opposite signs").
//!
//! The claim is then made granule by granule with a write to each one, so the
//! reservation ends wherever the driver actually refuses rather than wherever a
//! subtraction predicted it would. A short claim costs **regions**, not the
//! transient tier or the weight side: everything except the region count is
//! positioned relative to the span's right edge, so shrinking the span squeezes
//! the KV count and leaves a forward able to run.

use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use candle::cuda_backend::cudarc::driver::result::memset_d8_sync;
use candle::cuda_backend::cudarc::driver::CudaStream;
use candle::vram::AllocClass;
use candle::Result;

use super::chunk_ops::MIGRATION_STAGING_CAP_BYTES;
use super::reservation::Reservation;
use super::types::TARGET_ARENA_BYTES;
use super::weight_zone::{INITIAL_KV_RESERVE, MIN_ELASTIC_RESERVE};

/// One region of the KV side. Every size class carves its arenas at this size.
pub const REGION_BYTES: usize = TARGET_ARENA_BYTES;

/// The widest the wave transient tier can ever be — the old fixed reservation,
/// kept only as the worst case the elastic middle must be able to *reach*.
///
/// Nothing reserves this any more. The tier is placed per forward at the width
/// that forward actually needs (`WavePlan::phase_bytes`), which for a
/// twenty-session decode is a few megabytes against the 912 MiB here. This
/// constant survives so [`MIN_ELASTIC_RESERVE`] can be checked against it: the
/// floor has to leave room for the widest wave, even though no wave but the
/// widest will use it.
const MAX_WAVE_TRANSIENT_BYTES: usize = super::wave_spans::WAVE_ATTN_BYTES
    + super::wave_spans::WAVE_FFN_BYTES
    + super::wave_spans::WAVE_FORWARD_BYTES;

/// The persistence thread's staging block, at the far **left** of the span.
///
/// Fixed for the process lifetime and out of both frontiers' way. It is 4
/// regions' worth; putting it left rather than letting it float is what lets the
/// weight boundary move without ever having to reason about a copy stream that
/// is not synchronised to the wave.
const PERSIST_SPAN_BYTES: usize = MIGRATION_STAGING_CAP_BYTES;

const _: () = assert!(
    MAX_WAVE_TRANSIENT_BYTES.is_multiple_of(REGION_BYTES)
        && PERSIST_SPAN_BYTES.is_multiple_of(REGION_BYTES),
    "every fixed block must be region-aligned so region carving stays exact"
);

const _: () = assert!(
    MIN_ELASTIC_RESERVE > MAX_WAVE_TRANSIENT_BYTES + PERSIST_SPAN_BYTES,
    "the floor must leave room for the widest wave's tier, or a forward cannot run"
);

/// The span when no governor has measured the card.
///
/// A fixed number rather than a fraction of what the driver reports free. The
/// fraction was tried and is wrong here: the only processes that get this far
/// without a governor are test binaries, which run hundreds of GPU tests
/// concurrently, so "half of free" depends on which tests happen to be in
/// flight when the first KV cache is built — and a run that lost the race
/// claimed nothing at all. This is large enough for the suite's peak arena
/// working set and small enough to leave the card to the tests themselves.
const TEST_SPAN_BYTES: usize =
    2 * 1024 * 1024 * 1024 + MAX_WAVE_TRANSIENT_BYTES + PERSIST_SPAN_BYTES;

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
            // to decide whether this region still needs a wait, and its being
            // non-zero is what tells the next tenant to clean it at all.
            pool.dirty_epoch[self.index] = pool.quiesce_epoch;
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
    /// Regions claimable right now without evicting anything — the pressure
    /// signal, and the admission budget. Excludes anything the transient tier's
    /// ceiling forbids; see [`blocked`](Self::blocked).
    pub free: usize,
    /// Regions nobody owns that the tier's ceiling puts out of reach anyway.
    ///
    /// Reported apart from `free` because the two call for opposite responses: a
    /// low `free` with a low `blocked` is genuine KV pressure and wants the
    /// boundary moved, while a low `free` with a high `blocked` is the wave's own
    /// tier standing on the ground and wants a narrower wave.
    pub blocked: usize,
    /// Most regions ever live at once — how close the partition came to full.
    pub peak_live: usize,
    /// Bytes the wave transient tier occupies **right now**, and the ceiling it
    /// imposes on the region count while it does.
    ///
    /// Both are zero between forwards: the tier does not exist then, and the
    /// whole span below the weight floor is the KV side's. A non-zero reading is
    /// a snapshot taken mid-wave, not a reservation.
    pub transient_bytes: usize,
    pub transient_ceiling: usize,
    /// Fresh regions claimed since boot while a tier was placed — claims that
    /// arrived after the wave's admit phase. Zero is the precondition for
    /// anchoring the tier at the arena frontier.
    pub fresh_claims_during_wave: usize,
    /// Claims refused because the tier's base was the ceiling — room existed,
    /// the tier's position hid it. A non-zero reading says the admit phase let a
    /// claim through to the wave.
    pub refusals_during_wave: usize,
    /// Bytes of the span the weight side is permitted to hold.
    pub weight_bytes: usize,
    /// Bytes of the span nothing has been able to use — the tail left over when
    /// the region count rounds down.
    pub slack_bytes: usize,
    /// Bytes of address space reserved, and the driver's mapping granule. The
    /// two together say whether a shortfall is the driver refusing or the
    /// layout rounding.
    pub reserved_bytes: usize,
    pub granularity: usize,
}

struct RegionPool {
    reservation: Reservation,
    /// First byte of the span.
    span_base: u64,
    /// Bytes of the span actually backed by physical memory. May be less than
    /// what was reserved, when the driver refused part-way.
    span_bytes: usize,
    /// First byte available to regions — past both fixed blocks.
    region_base: u64,
    /// The leftmost byte the weight side occupies: **the moving boundary**.
    /// `span_end` until an expert cache installs a zone.
    weight_floor: u64,
    /// Regions that fit between [`Self::region_base`] and the transient tier:
    /// the KV side is `[0, total)`.
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
    /// Per region, the epoch current when its bytes last belonged to someone
    /// else. **Zero means never** — the region has been untouched since the
    /// reservation was mapped, so it is already zeroed and has no readers.
    ///
    /// A claim needs a *wait* only while this still equals
    /// [`Self::quiesce_epoch`] — otherwise a quiesce has happened since and the
    /// region's readers are long gone — but any non-zero value means it needs
    /// *cleaning*, because the last occupant's bytes are still there.
    ///
    /// # Three ways ground changes hands, and all three land here
    ///
    /// An arena releasing its region is the obvious one. The other two are the
    /// reason this is no longer called `released_epoch`: the **wave transient
    /// tier** stands on KV ground for the length of a forward and writes
    /// intermediates all over it, and the **weight boundary** hands over ground
    /// that was holding expert weights. A region arriving from either is exactly
    /// as dirty as a recycled one, and until this was written only the recycled
    /// path checked.
    dirty_epoch: Vec<u64>,
    /// Where the wave transient tier sits **while a forward is running**, and
    /// how big it is. `None` between forwards, when it occupies nothing.
    ///
    /// This is the one variable-size block in the span, and it is variable
    /// precisely because it vanishes: nothing is live in it at the moment its
    /// position or extent changes, so a resize leaves no hole and moves no data.
    /// That is why it sits *between* the arenas and the weights — the only place
    /// a block that changes size can absorb both sides' movement without one.
    transient_base: Option<u64>,
    transient_bytes: usize,
    /// Persistence-staging bytes carved from the fixed left block.
    persist_carved: usize,
    /// Fresh regions claimed while a wave's transient tier was placed.
    ///
    /// **A tripwire that must read zero.** A region claim creates an arena, and
    /// an arena may only be created between forwards —
    /// `BackingInner::arena_window` blocks until then. A non-zero reading names
    /// a creation path that skipped that gate and moved the arena frontier under
    /// a tier already placed against it.
    ///
    /// It used to be a *measurement*: the count of claims that would have to fit
    /// in a gap above the frontier for the tier to be anchored there, which on
    /// the quantized path ran into the hundreds because the compressor creates
    /// size-class arenas as it chooses formats. That number was read as proof
    /// the frontier anchor was impossible. It was proof the sealing thread was
    /// allocating arenas inside the wave.
    fresh_claims_during_wave: usize,
    /// Region claims refused because the tier's base was the ceiling.
    ///
    /// The same tripwire from the other side: a claim that ran out of ground
    /// *with a tier standing* is a claim that arrived inside a wave. A claim
    /// that runs out with no tier standing is ordinary pressure and buys more
    /// (`set_ground_broker`).
    refusals_during_wave: usize,
    /// Highest demand seen in the window that is closing, and in the one before
    /// it. The mark the weight side measures "spare" against is the larger of
    /// the two — see [`RegionPool::spare_regions`].
    kv_peak_window: usize,
    kv_peak_prev_window: usize,
    /// When the current window opened.
    kv_peak_window_opened: Instant,
    /// When the KV side was last refused a region.
    ///
    /// The one signal that is neither history nor occupancy: a refusal is the KV
    /// side *saying* it wanted more. Neither of the other two catches an ingest,
    /// where demand only ever grows — at any instant there is room, and a moment
    /// later there is not.
    last_pressure_at: Option<Instant>,
}

/// How far back the KV side's high-water mark looks.
///
/// The mark is a **sliding-window maximum**, not a decaying estimate: the
/// largest demand seen in the window now closing or the one before it. It rises
/// the instant demand does, and falls exactly one window after the peak that set
/// it stops being current. **Fast to concede, deliberate to take** — being short
/// of KV throttles admission, being short of experts is a slowdown, and the two
/// are not worth trading symmetrically.
///
/// Two earlier shapes and what they cost:
///
/// - **Per-pass decay** (×0.9) let the weight side take 224 of 291 regions
///   during the gate's single-context configs and the twenty-context config that
///   followed failed outright. Passes are not time; a benchmark answers "has KV
///   been idle for a few forwards?" yes constantly.
/// - **Exponential decay on a five-minute clock** was safe and nearly inert. It
///   converges by halving, so a cold-start mark of 332 regions needs three
///   intervals — a quarter of an hour — to reach a steady state the daemon
///   reaches in seconds. Measured on a live rebuild: 19 live regions of 332,
///   5008 MiB idle, and the boundary had not moved.
///
/// A window maximum has neither fault. Convergence is one window, exactly, and
/// there is no rate constant to pick — only how long a quiet period must last
/// before it counts as quiet, which is a question about the workload and has an
/// answer.
///
/// **Sixty seconds** is that answer for an interactive daemon: long enough that
/// the gap between two turns of a conversation does not read as idle, short
/// enough that a batch of small prefills hands the weight side its ground while
/// the batch is still running. The safety net underneath is admission — the
/// scheduler's ceiling is `(free regions − setpoint) × region_bytes`, read live,
/// so ground given to the weights narrows what admission accepts rather than
/// failing anything.
const KV_PEAK_WINDOW: Duration = Duration::from_secs(60);

/// Most regions the weight side may take in one pass.
///
/// Growth is a step, not a jump, because each region it takes may have to be
/// given back — and giving back costs an eviction or a relocation, while not
/// taking costs only the residency it would have bought for one more pass.
const KV_GROW_STEP: usize = 8;

/// Regions bought in one go when a claim runs the KV side out of ground.
///
/// Symmetric with [`KV_GROW_STEP`], and for the same reason inverted: a purchase
/// costs a device-wide quiesce (the weight side cannot hand over ground while a
/// kernel might still be reading it), so buying one region per claim would pay
/// that sync per arena. A section-quantize drain claimed eighteen regions in one
/// pass; at a region apiece that is eighteen full device syncs.
const KV_BUY_STEP: usize = 8;

fn pools() -> &'static Mutex<HashMap<usize, RegionPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, RegionPool>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Buys KV ground from the weight side: `regions` in, bytes conceded out.
///
/// Installed once by the expert cache, which owns the only thing that can pay —
/// expert residency, whose cold tier holds a valid copy of every expert, so a
/// slot given up costs a reload and never a loss.
type GroundBroker = Arc<dyn Fn(usize) -> u64 + Send + Sync>;

/// Sellers by device ordinal.
///
/// **Keyed, and replaceable.** A `OnceLock<GroundBroker>` was neither, and both
/// were wrong: one broker for the process misroutes every purchase on the second
/// GPU to the first one's expert cache, and a set-once cell means a second model
/// load leaves the first model's dead `Weak` installed — every later purchase
/// answers zero and the partition silently reverts to refusing claims it could
/// have paid for.
fn ground_brokers() -> &'static Mutex<HashMap<usize, GroundBroker>> {
    static BROKERS: OnceLock<Mutex<HashMap<usize, GroundBroker>>> = OnceLock::new();
    BROKERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Install the weight side's seller. Called once, as the expert cache opens.
///
/// # Why this is a callback where the old design insisted on a counter
///
/// [`RegionPool::kv_pressure`] used to record refusals for the weight side to
/// read at its own safe point, because `claim_region` runs on whichever thread
/// wanted an arena while the weight side belongs to the expert pipeline thread,
/// and evicting from here would be a cross-thread call into a cache that may be
/// mid-wave. Every word of that is still true — which is why the broker does not
/// evict anything. It **sends a message and blocks on the reply**, so the
/// eviction still happens on the pipeline thread, behind its own quiesce, at a
/// point it chose. The only thing that changed sides is the arithmetic.
///
/// And the arithmetic is why it had to. A counter records *events*, and the
/// weight side spent them as *regions*: one failed section-quantize drain walked
/// the size-class ladder and left 4,436 units of demand behind it, against a KV
/// side that was twenty-eight regions short. The boundary paid all of it. A
/// claim that buys its own ground cannot make that error, because the claim and
/// the demand are the same object — the allocation *is* the measurement.
///
/// Absent broker (CPU builds, inline mode, tests) means no seller: a claim that
/// runs out then refuses exactly as it always did.
///
/// **Replaces any previous seller for `ordinal`.** The newest expert cache on a
/// device is the one that owns its weight zone, so it is the one that can sell;
/// keeping an older registration would leave purchases going to a cache whose
/// model is gone.
pub fn set_ground_broker(ordinal: usize, broker: impl Fn(usize) -> u64 + Send + Sync + 'static) {
    let mut map = ground_brokers().lock().unwrap_or_else(|e| e.into_inner());
    map.insert(ordinal, Arc::new(broker));
}

thread_local! {
    /// Whether this thread is already inside a purchase.
    ///
    /// The broker blocks on the expert pipeline, which moves the boundary and
    /// calls back into this pool. Nothing on that path claims a region today, but
    /// a purchase that re-entered would deadlock on the pool mutex rather than
    /// fail visibly, so the invariant is enforced rather than assumed.
    static BUYING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Ask the weight side for `regions` of ground; answers the bytes it conceded.
///
/// **Called with no pool lock held.** The broker blocks on the pipeline thread,
/// which takes that lock to move the floor.
fn buy_ground(stream: &std::sync::Arc<CudaStream>, regions: usize) -> Result<u64> {
    // Cloned out of the map, so the broker's own lock is not held across a call
    // that blocks on another thread.
    let broker = {
        let map = ground_brokers().lock().unwrap_or_else(|e| e.into_inner());
        map.get(&stream.context().ordinal()).cloned()
    };
    let Some(broker) = broker else {
        return Ok(0);
    };
    if BUYING.with(|b| b.replace(true)) {
        return Ok(0);
    }
    // Cleared on the way out of this scope however it is left. A flag reset only
    // on the success path would, after one unwind, leave this thread unable to
    // buy ground for the life of the process — and it would fail as a throughput
    // collapse with no error, which is the worst way for anything here to fail.
    struct Buying;
    impl Drop for Buying {
        fn drop(&mut self) {
            BUYING.with(|b| b.set(false));
        }
    }
    let _buying = Buying;
    let conceded = broker(regions);
    if conceded > 0 {
        // The KV side just proved it wants everything it has and more. Stamping
        // the moment keeps the weight side from reading the ground it has only
        // just handed over as spare on its very next pass.
        with_pool(stream, |pool| {
            pool.last_pressure_at = Some(Instant::now());
            pool.kv_peak_window = pool.total;
            Ok(())
        })?;
    }
    Ok(conceded)
}

/// The span's target size: everything left, less the cushion the CUDA pool keeps.
///
/// With a governor — every production load installs one — `usable` is what
/// remains of the balloon-measured capacity `C` right now. The dense weights are
/// already resident when this runs, so they are already inside `usable`'s
/// accounting and **nothing further is subtracted for them** (see
/// [`span_from`]).
///
/// Without a governor, [`TEST_SPAN_BYTES`].
///
/// **A governor that fails to answer is an error, not a fallback.** These two
/// cases used to share one `_` arm, so a transient `usable()` failure on a
/// production card sized the whole span at the 2 GiB test constant and said
/// nothing — the reservation would come up a quarter of its intended size and
/// every downstream number would be quietly wrong. Absence of a governor is a
/// test binary; a governor that errors is a fault.
fn span_target(ordinal: usize) -> Result<usize> {
    let Some(gov) = candle::vram::get(ordinal) else {
        return Ok(TEST_SPAN_BYTES);
    };
    let usable = gov.usable().map_err(|e| {
        candle::Error::Msg(format!(
            "reservation: the VRAM governor could not report usable bytes ({e}). \
             Refusing to size the span from the {TEST_SPAN_BYTES} B test constant \
             on a card that has a governor — that would silently claim a fraction \
             of the intended reservation."
        ))
    })?;
    Ok(span_from(usable as usize, gov.pool_cushion() as usize))
}

/// The span, given what is left of `C` and the cushion to leave on the pool.
///
/// **The dense weights are subtracted exactly once, and not here.** `usable()`
/// is `headroom.min(C − spent_by_us)` where `spent_by_us` is the drop in
/// headroom since `C` was measured — and the dense weights are resident by the
/// time this is called, so they are already inside that drop. Subtracting
/// `class_reserved(Weights)` on top would book the same bytes twice.
///
/// That is not a hypothetical. The load order now makes the weights both
/// resident *and* tallied at this moment, so the second subtraction looks like
/// prudence; it is the exact mistake this codebase has already made twice.
/// `balloon_headroom_abs` reserved the transient peak that `expert_budget` was
/// also reserving — 1,104 MiB no allocator was permitted to touch — and this
/// function's predecessor subtracted `scratch_margin` against a transient tier
/// that was then added back on top, "same bytes, two places, opposite signs".
///
/// Split out so the identity can be asserted without a governor.
fn span_from(usable: usize, pool_reserve: usize) -> usize {
    usable.saturating_sub(pool_reserve)
}

/// How the span divides, once its true backed size is known.
///
/// **This is the layout at rest**, between forwards. Only the persistence block
/// is here; the wave transient tier occupies nothing until a forward places it
/// ([`place_transient`]), so the regions run straight to the weight floor:
///
/// ```text
/// | persist 64 MiB | regions … | ← W → weight slots |
/// ```
///
/// The tier is deliberately absent. It is the one **variable-size** block in the
/// span — priced per wave rather than at the widest wave — and it is safe to be
/// variable precisely because it vanishes here: at the moment its extent changes
/// it holds nothing, so a resize leaves no hole and moves no data.
///
/// That is also why it belongs between the regions and the weights rather than
/// at a fixed address beside the persistence block. A variable-size block at a
/// fixed address is a choice between two failures: leave the region base fixed
/// and every resize strands a hole (fragmentation), or pack the regions after it
/// and every resize relocates every region (thrashing). Only the position
/// adjacent to the moving boundary disturbs nothing.
///
/// A span that came up short costs **regions** and never the persistence block:
/// it is positioned from the left and the region count is what absorbs the
/// difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    region_base: u64,
    total: usize,
    transient_end: u64,
    /// The weight floor as clamped into the span — what the caller asked for,
    /// corrected. Returned rather than recomputed so nothing downstream has to
    /// repeat the clamp and risk disagreeing about it.
    weight_floor: u64,
    slack: usize,
}

/// Divide a span of `span_bytes` at `span_base` with the weight side occupying
/// `[weight_floor, span_end)`.
fn layout_span(span_base: u64, span_bytes: usize, weight_floor: u64) -> Layout {
    let region_base = span_base + PERSIST_SPAN_BYTES as u64;
    let transient_end = region_base;
    let span_end = span_base + span_bytes as u64;
    // Clamp rather than trust: the floor arrives from arithmetic done a crate
    // away, and a wrapped subtraction here would produce an address that looks
    // plausible. `.min(span_end)` keeps the lower bound meaningful on a span too
    // small to hold the fixed blocks; `RegionPool::create` refuses that span
    // outright, so this only has to stay arithmetically sane.
    let weight_floor = weight_floor.clamp(region_base.min(span_end), span_end);
    let usable_for_regions = weight_floor.saturating_sub(region_base) as usize;
    let total = usable_for_regions / REGION_BYTES;
    Layout {
        region_base,
        total,
        transient_end,
        weight_floor,
        slack: usable_for_regions - total * REGION_BYTES,
    }
}

impl RegionPool {
    fn create(stream: &std::sync::Arc<CudaStream>) -> Result<Self> {
        let ordinal = stream.context().ordinal();
        // Read before the claim: mapping granules consumes headroom, so asking
        // afterwards reports what is left rather than what was available.
        let usable_before = candle::vram::get(ordinal)
            .and_then(|g| g.usable().ok())
            .unwrap_or(0) as usize;
        let want = (span_target(ordinal)? / REGION_BYTES) * REGION_BYTES;
        let mut reservation = Reservation::reserve(stream, want)?;

        // Claim left to right across the whole span. A refusal part-way ends the
        // span there, and because every other block is positioned relative to
        // the right edge, the shortfall lands on the region count — the one
        // tenant that degrades gracefully.
        let claimed = reservation.map_range(0, want)?;
        let span_bytes = (claimed / REGION_BYTES) * REGION_BYTES;
        let span_base = reservation.base();
        let span_end = span_base + span_bytes as u64;
        let minimum = PERSIST_SPAN_BYTES + MAX_WAVE_TRANSIENT_BYTES;
        if span_bytes < minimum {
            candle::bail!(
                "reservation: the card backed only {span_bytes} B of the {want} B asked \
                 for, below the {minimum} B of fixed staging and transient span a \
                 forward needs — there is not enough VRAM left to run one"
            )
        }

        // No weight side yet: an expert cache installs one at load
        // (`set_weight_floor`), and a process without experts leaves the whole
        // span to KV.
        let layout = layout_span(span_base, span_bytes, span_end);
        let total = layout.total;
        // The partition, on the one channel that survives a test binary (which
        // installs no tracing subscriber, so the log lines below are invisible
        // there).
        //
        // `residual` is `capacity − weights − usable` — CUDA pool growth, the
        // gallery arena, per-class overhead — and is the term to chase when the
        // span comes out smaller than the card suggests it should. `weights` is
        // reported next to it precisely so that anyone reading this can confirm
        // it was subtracted **once**: it is inside `usable`, not applied on top
        // of it.
        let gov = candle::vram::get(ordinal);
        let weights = gov
            .as_ref()
            .map_or(0, |g| g.class_reserved(AllocClass::Weights) as usize);
        let capacity = gov.as_ref().map_or(0, |g| g.capacity() as usize);
        let residual = capacity
            .saturating_sub(weights)
            .saturating_sub(usable_before);
        if super::alloc::arena_stats_enabled() {
            let mib = |b: usize| b / (1024 * 1024);
            eprintln!(
                "[reservation] capacity_c={}MiB usable={}MiB pool_cushion={}MiB \
                 weights={}MiB residual={}MiB | asked={}MiB span={}MiB \
                 (shortfall {}MiB) = persist {}MiB + {total} regions ({}MiB) \
                 + transient {}MiB + weights-side {}MiB, slack {}B",
                mib(capacity),
                mib(usable_before),
                gov.as_ref().map_or(0, |g| mib(g.pool_cushion() as usize)),
                mib(weights),
                mib(residual),
                mib(want),
                mib(span_bytes),
                mib(want.saturating_sub(span_bytes)),
                mib(PERSIST_SPAN_BYTES),
                mib(total * REGION_BYTES),
                mib(MAX_WAVE_TRANSIENT_BYTES),
                mib((span_end - layout.weight_floor) as usize),
                layout.slack,
            );
        }
        if span_bytes < want {
            log::warn!(
                "reservation: the card backed {} MiB of the {} MiB span asked for. \
                 The shortfall lands on the KV side ({total} regions) — the fixed \
                 staging, the transient tier and the weight side are positioned \
                 from the span's right edge and are unaffected.",
                span_bytes / (1024 * 1024),
                want / (1024 * 1024),
            );
        } else {
            log::info!(
                "reservation: {} MiB span = {} MiB persist + {total} regions \
                 ({} MiB), transient placed per forward (≤{} MiB), in {} KiB granules",
                span_bytes / (1024 * 1024),
                PERSIST_SPAN_BYTES / (1024 * 1024),
                (total * REGION_BYTES) / (1024 * 1024),
                MAX_WAVE_TRANSIENT_BYTES / (1024 * 1024),
                reservation.granularity() / 1024,
            );
        }

        Ok(Self {
            reservation,
            span_base,
            span_bytes,
            region_base: layout.region_base,
            weight_floor: span_end,
            total,
            next: 0,
            free: BinaryHeap::new(),
            live: 0,
            peak_live: 0,
            // **Epochs start at one**, so that zero can mean "never dirtied" in
            // `dirty_epoch` without colliding with a real epoch. A pristine
            // region is zero from the mapping touch and has no readers, and that
            // has to be distinguishable from one dirtied before the first
            // quiesce — otherwise every region on the card claims as dirty once.
            quiesce_epoch: 1,
            dirty_epoch: vec![0; total],
            transient_base: None,
            transient_bytes: 0,
            persist_carved: 0,
            fresh_claims_during_wave: 0,
            refusals_during_wave: 0,
            // Opens at the top: the weight side must watch the KV side stay
            // small before it may take anything, never assume it will.
            // Both windows open at the top: the weight side must watch the KV
            // side stay small before it may take anything, never assume it will.
            kv_peak_window: total,
            kv_peak_prev_window: total,
            kv_peak_window_opened: Instant::now(),
            last_pressure_at: None,
        })
    }

    fn span_end(&self) -> u64 {
        self.span_base + self.span_bytes as u64
    }

    /// The lowest address the wave transient tier may not reach below: one past
    /// the last byte any live arena holds.
    ///
    /// A region is not preemptible — an arena keeps its address for as long as
    /// it lives — so this is a hard floor for the tier, not a preference.
    fn live_end(&self) -> u64 {
        self.region_base + (self.live_watermark() * REGION_BYTES) as u64
    }

    /// Regions the KV side may hand out **right now**.
    ///
    /// `total` is what fits below the weight floor. While a forward is running,
    /// the transient tier sits on top of the live arenas, so the ceiling drops to
    /// where it starts — a region stamped past that would be inside the tier.
    ///
    /// Between forwards the tier is gone and the ceiling **is** the boundary.
    /// Nothing is withheld against the next placement: [`place_transient`] buys
    /// the ground it needs when it needs it, which is both exact and current
    /// where a withheld reserve was a one-sample guess at a quantity that varies
    /// eightfold between consecutive waves.
    fn region_ceiling(&self) -> usize {
        ceiling_regions(
            self.transient_base,
            self.weight_floor,
            self.region_base,
            self.total,
        )
    }

    fn region_base(&self, index: usize) -> u64 {
        self.region_base + (index * REGION_BYTES) as u64
    }

    /// Regions [`claim_region`] would actually hand out right now.
    ///
    /// **Not "regions nobody owns".** The two differ whenever a wave's transient
    /// tier is placed, because the tier caps the pool at
    /// [`Self::region_ceiling`] and everything above that is unclaimable however
    /// free it looks. Counting the difference matters because this number
    /// answers "what would `claim_region` hand out right now" — a claim made
    /// while the tier stands must be refused at the ceiling, and reporting the
    /// tier's own ground as claimable is a promise the allocator then refuses
    /// to keep. (A wide section prefill once did exactly that: a 496 MiB tier
    /// capped the ceiling at 293 regions, the 31 above it were reported free,
    /// and admission planned against them — 12,488 refusals in four seconds
    /// with no forward completing.)
    ///
    /// Callers whose decision lands *after* the tier releases — admission
    /// budgets, scheduler pressure, the weight-side negotiation — must add
    /// [`Self::ceiling_blocked`] back on top; [`RegionStats`] carries both
    /// numbers so each consumer picks the horizon it is actually deciding for.
    fn free_count(&self) -> usize {
        // The heap is unordered under iteration, so this counts rather than
        // short-circuits — a few hundred entries on a path that runs per wave,
        // not per claim.
        let below = self
            .free
            .iter()
            .filter(|Reverse(i)| *i < self.region_ceiling())
            .count();
        claimable(below, self.next, self.region_ceiling())
    }

    /// Regions that are unowned but sit above the ceiling, so cannot be claimed
    /// while the tier stands. Zero between waves; the gap between "free" and
    /// "free to use" when a wave is placed.
    fn ceiling_blocked(&self) -> usize {
        let ceiling = self.region_ceiling();
        let above = self.free.iter().filter(|Reverse(i)| *i >= ceiling).count();
        blocked(above, self.next, self.total, ceiling)
    }

    /// Mark every region overlapping `[base, base + len)` as holding someone
    /// else's bytes, so the next claim quiesces and zeroes it.
    ///
    /// Rounds **outward**: a region half-covered is still dirty, and a byte of
    /// stale data is as bad as a region of it.
    fn mark_dirty_span(&mut self, base: u64, len: usize) {
        let start = base.saturating_sub(self.region_base) as usize / REGION_BYTES;
        let end = (base
            .saturating_add(len as u64)
            .saturating_sub(self.region_base) as usize)
            .div_ceil(REGION_BYTES);
        for slot in self
            .dirty_epoch
            .iter_mut()
            .take(end.min(self.total))
            .skip(start)
        {
            *slot = self.quiesce_epoch;
        }
    }

    /// One past the highest region currently held by an arena.
    ///
    /// The KV side cannot shrink below this, and it is meaningfully smaller than
    /// `next` because the free list is lowest-first: live arenas stay
    /// left-packed, which is the property that makes the boundary movable at all.
    fn live_watermark(&self) -> usize {
        let free: std::collections::HashSet<usize> =
            self.free.iter().map(|Reverse(i)| *i).collect();
        (0..self.next)
            .rev()
            .find(|i| !free.contains(i))
            .map_or(0, |i| i + 1)
    }

    /// Move the weight side's left edge and re-derive the region count.
    ///
    /// Runs at model load to place the boundary, and between waves to move it.
    /// Refuses rather than corrupts (principle 7): a floor past
    /// [`MIN_ELASTIC_RESERVE`], past the span end, or one that would strand a
    /// live region is an error, not a clamp.
    ///
    /// The fixed blocks do not move with it — the transient tier is a constant
    /// size at a constant address (see [`layout_span`]) — so this changes exactly
    /// two things: how many regions exist, and where the weight side starts.
    fn set_weight_floor(&mut self, floor: u64) -> Result<usize> {
        let span_end = self.span_end();
        let min_floor = self.span_base + MIN_ELASTIC_RESERVE as u64;
        if floor < min_floor {
            candle::bail!(
                "weight floor {floor:#x} would leave the elastic middle below the \
                 {} MiB reserve (floor must be at least {min_floor:#x})",
                MIN_ELASTIC_RESERVE / (1024 * 1024),
            )
        }
        if floor > span_end {
            candle::bail!("weight floor {floor:#x} is past the span end {span_end:#x}")
        }
        let layout = layout_span(self.span_base, self.span_bytes, floor);
        let watermark = self.live_watermark();
        if layout.total < watermark {
            candle::bail!(
                "weight floor would cut the KV side to {} regions, but region {} is \
                 live. The weight side may only take regions the KV side is not \
                 using; ask it to release them first.",
                layout.total,
                watermark - 1,
            )
        }
        let gained = layout.total.saturating_sub(self.total);
        self.weight_floor = floor;
        self.total = layout.total;
        // **Ground arriving from the weight side is dirty.** It was holding
        // expert weights a moment ago, and `resize(.., 0)` would declare it
        // pristine — the state reserved for a region untouched since the mapping
        // touch, which is the one case a claim may skip cleaning. A KV arena
        // would then be handed a region full of expert bytes.
        //
        // Stamped one epoch *behind* the current one, which is the exact truth:
        // the handover already quiesced (`quiesce_before_handover`), so there are
        // no readers left to wait for — but the bytes are still there, so it
        // still needs zeroing.
        self.dirty_epoch.resize(layout.total, 0);
        if gained > 0 {
            let stale = self.quiesce_epoch.saturating_sub(1);
            for slot in self.dirty_epoch.iter_mut().skip(layout.total - gained) {
                *slot = stale;
            }
        }
        // Regions past the new total no longer exist: drop them from the free
        // list and pull `next` back so the fresh-region path cannot hand one out.
        self.free.retain(|Reverse(i)| *i < layout.total);
        self.next = self.next.min(layout.total);
        Ok(layout.total)
    }

    /// Regions the weight side may take, against the KV side's **recent**
    /// high-water mark rather than its free count right now.
    ///
    /// Free-right-now is the wrong signal and taking it was measured to be
    /// catastrophic: the gate's first configs run one context, so nearly the
    /// whole KV side reads as free, the weight side takes it, and the 20-context
    /// config that follows exhausts the 67 regions left. Free means "not in use
    /// this instant", and the weight side needs "not in use lately".
    ///
    /// So the mark is a sliding-window maximum over [`KV_PEAK_WINDOW`]: it
    /// tracks demand up the instant it rises, and falls exactly one window after
    /// the peak that set it stops being current — ageing a boot transient out
    /// without ever forgetting a demand the KV side is still making.
    ///
    /// # Why this is still an estimate, and why that is not a failure of nerve
    ///
    /// `docs/elastic_vram_partition.md` §7 claims the phase-locked forward
    /// "recomputes the partition exactly and needs no decay at all". That is
    /// half true, and the half it gets wrong is the half that matters here.
    ///
    /// What a forward can compute exactly is *its own* demand: admit makes the
    /// KV claims, `WavePlan` prices the tier, and both are known before a byte
    /// computes. The boundary is not set against this forward's demand. It is
    /// set against the **next** one's — the weight side is being asked to take
    /// ground on the promise that nothing will want it back — and no exactness
    /// about the present makes the future knowable.
    ///
    /// That was established by building the exact version and running it. A
    /// monotone watermark has no seed that works: started at zero it offers the
    /// whole span on the first negotiation and every config of the gate fails;
    /// started at `total` it never falls, so the weight side never gains a
    /// region and the mechanism is inert. Anything in between is a decay wearing
    /// a different name.
    ///
    /// So the estimate stays, and what the exact figures buy is the *other*
    /// direction: the tier term below is this wave's real price rather than the
    /// old fixed 912 MiB, so the mark it is added to is no longer inflated by a
    /// reservation nobody uses.
    fn spare_regions(&mut self, slack: usize) -> usize {
        // Demand is arenas **plus the tier standing on top of them** — this runs
        // at the pipeline's end of pass, while the tier is still reserved, so
        // both are visible. Measuring against `live` alone would let the weight
        // side grow into ground the very next wave's transient needs.
        //
        // `transient_bytes` alone, now that there is no reserve outliving the
        // placement. A call from between forwards therefore sees a tier of zero —
        // and the sliding window below is what covers that, because it is a
        // sixty-second maximum and the tier was in the sum every time one stood.
        // A window is the better instrument in any case: it carries the *largest*
        // tier of the last minute across the gap, where the reserve carried only
        // the most recent one.
        let demand = self.live + self.transient_bytes.div_ceil(REGION_BYTES);
        // Rises the instant demand does; falls exactly one window after the peak
        // that set it passes out of view.
        self.kv_peak_window = self.kv_peak_window.max(demand);
        if self.kv_peak_window_opened.elapsed() >= KV_PEAK_WINDOW {
            self.kv_peak_prev_window = self.kv_peak_window;
            self.kv_peak_window = demand;
            self.kv_peak_window_opened = Instant::now();
        }
        // **Two independent permissions, and the weight side needs both.**
        //
        // The window peak guards against demand the KV side made *recently* and
        // will plausibly make again. It says nothing about right now — a peak
        // that has aged out leaves the mark low even while every region is
        // occupied, because the mark measures history and occupancy measures the
        // present.
        //
        // A live rebuild showed exactly that: sixty seconds in, the KV side was
        // climbing toward saturation, the window had not yet caught up, and the
        // weight side took four regions out from under it. Small in itself, and
        // precisely the wrong direction — those were regions the KV side was
        // about to need and could only get back by evicting experts, which the
        // pinned pool had no room for.
        //
        // So ground must be free in *both* senses: unused across the window, and
        // unused this instant.
        //
        // And neither is enough on its own during an **ingest**, where demand
        // climbs monotonically: the window has not caught up, this instant has
        // room, and a moment later every region is taken. Measured — with both
        // guards in place the weight side still took eight regions ninety
        // seconds into a rebuild that then saturated.
        //
        // The third guard is the KV side's own voice. A refusal is it asking for
        // more and being told no, and a side that has been refused inside the
        // last window has no spare ground by definition, whatever the other two
        // say.
        if self
            .last_pressure_at
            .is_some_and(|t| t.elapsed() < KV_PEAK_WINDOW)
        {
            return 0;
        }
        let by_history = self
            .total
            .saturating_sub(self.kv_peak_window.max(self.kv_peak_prev_window));
        // Occupancy counts tier-blocked ground as unoccupied: the boundary only
        // moves with no wave open, so a standing tier is idle — its bytes are
        // dead until the next forward's phase 0 releases it — and its *future*
        // demand is already priced into the window peak above (`demand` includes
        // `transient_bytes` every time a tier stands). Counting the blocked
        // ground as occupied here would charge the tier twice and pin the
        // boundary wherever a tier happened to be standing.
        let by_occupancy = self.free_count() + self.ceiling_blocked();
        by_history.min(by_occupancy).saturating_sub(slack)
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
    // **Keep the frontier honest before taking virgin ground.** Claims recycle
    // the free list lowest-first, so the frontier only advances when the list
    // is empty — and if chunk-empty arenas exist at that moment, "empty" is a
    // bookkeeping artifact of churn, not real occupancy. Advancing anyway is
    // what pushes long-lived arenas into the top regions, and the transient
    // tier — anchored at the weight floor, measuring down — then loses that
    // ground for as long as those arenas live (measured on the Llama-2 MHA
    // gate: a span-end arena claimed at a churn-inflated 566/566 peak blocked
    // a 4-region tier with only 205 regions genuinely live). Sweep first, so
    // the frontier tracks the true live watermark.
    let free_list_empty = with_pool(stream, |pool| Ok(pool.free.peek().is_none()))?;
    if free_list_empty {
        super::backing::global_release_empty_arenas();
    }
    match with_pool(stream, |pool| try_claim(pool, stream))? {
        Claim::Got(handle) => return Ok(Some(handle)),
        // A tier is standing: the ground above it exists but belongs to a
        // running wave, and no amount of weight-side concession reaches it. The
        // answer is a narrower wave, not more ground.
        Claim::TierBlocked => return Ok(None),
        Claim::Exhausted => {}
    }
    // **Outside a wave, running out is a price, not a wall.** The span is one
    // reservation with a moving boundary, the cold tier holds a valid copy of
    // every expert, and so the KV side may buy the ground it needs by evicting
    // expert residency — which costs a reload and nothing else. The weight
    // zone's own floor is what finally refuses (`WeightZone::new`), and it is the
    // only refusal left in this path.
    if buy_ground(stream, KV_BUY_STEP)? == 0 {
        return Ok(None);
    }
    match with_pool(stream, |pool| try_claim(pool, stream))? {
        Claim::Got(handle) => Ok(Some(handle)),
        _ => Ok(None),
    }
}

/// What one attempt at [`claim_region`] found.
enum Claim {
    Got(RegionHandle),
    /// Refused by a placed tier's base — unreachable until the wave ends.
    TierBlocked,
    /// No region below the boundary, and no tier to blame.
    Exhausted,
}

fn try_claim(pool: &mut RegionPool, stream: &std::sync::Arc<CudaStream>) -> Result<Claim> {
    {
        // **The ceiling binds the free list too.** A region freed while a tier
        // stands keeps its index, and that tier is placed below it. Handing it
        // out would put KV writes inside a running wave's intermediates.
        //
        // Between forwards there is no tier and the ceiling is the boundary
        // itself. It used to be the boundary less `tier_reserve` — the last
        // tier's width, held back so the next placement would find its ground
        // empty — and that reserve is gone. It could not do the job: the width
        // it withheld was one sample of a quantity that ranged from 4 to 34
        // regions across a single run, so it neither protected a wider tier nor
        // released the ground a narrower one left behind. What it did do was
        // refuse claims against unowned memory, and then the refusal escalated
        // into a boundary renegotiation that handed over the same ground by the
        // most expensive route available. [`place_transient`] buys what it needs
        // instead, on the same terms as any other claim.
        //
        // One comparison suffices: the heap is ordered lowest-index-first, so if
        // the lowest free region is above the ceiling then every free region is.
        let ceiling = pool.region_ceiling();
        let lowest_free_usable = pool.free.peek().is_some_and(|Reverse(i)| *i < ceiling);
        let (index, recycled) = if lowest_free_usable {
            let Reverse(idx) = pool.free.pop().expect("peeked");
            (idx, true)
        } else if pool.next < ceiling {
            // **A region claimed while a tier stands is an invariant breach.**
            //
            // Every region claim creates an arena, and an arena may only be
            // created between forwards — `BackingInner::arena_window` blocks
            // until then. So reaching here with a tier placed means some path
            // creates arenas without that gate, and the tier the running wave is
            // using was placed against a frontier that has just moved under it.
            //
            // Counted, loudly, rather than refused: refusing would turn a
            // bookkeeping hole into a failed forward, and the claim itself is
            // still served from ground nobody holds. The counter is a tripwire
            // that must read zero, not a tuning input — it used to be the latter,
            // and `fresh_per_forward_peak` grew a gap above the frontier to
            // accommodate exactly the claims this now forbids.
            if pool.transient_base.is_some() {
                pool.fresh_claims_during_wave += 1;
                log::warn!(
                    "reservation: fresh region claimed with the wave tier placed at {:#x} \
                     — an arena was created inside a wave (count {}). Every creation must \
                     pass BackingInner::arena_window, which waits for the gap between \
                     forwards.",
                    pool.transient_base.unwrap_or(0),
                    pool.fresh_claims_during_wave,
                );
            }
            pool.next += 1;
            (pool.next - 1, false)
        } else if pool.transient_base.is_some() {
            // **Blocked by the tier, which the boundary cannot fix.** The
            // ceiling is the tier's base while one is placed, and moving the
            // weight floor does not move a tier that is already standing — so
            // this shortfall is not demand the weight side can satisfy, and
            // asking it to pay is asking it to pay for nothing.
            //
            // It did exactly that. A section prefill wedged, the scheduler's
            // relief asked for ground on every retry, and each tier-blocked
            // refusal had counted as a region of boundary demand — so the weight
            // side conceded, one region at a time, evicting six experts a step,
            // until the expert zone was **empty**: `slots=0 weights=0MiB`, every
            // expert gone, and the ceiling still exactly where it started
            // because `transient_base` had never moved.
            //
            // What this shortfall calls for is a narrower wave or a released
            // tier, and `refusals_during_wave` is what says so.
            //
            // **Like the branch above, this now reads zero or something is
            // wrong.** A claim reaching the ceiling with a tier standing is a
            // claim that arrived inside a wave, which `arena_window` forbids.
            pool.refusals_during_wave += 1;
            if super::alloc::arena_stats_enabled() {
                eprintln!(
                    "[tier-refusal] region claim refused with the tier placed \
                     at {:#x} (ceiling {ceiling}, next {}, total {}) — count {}",
                    pool.transient_base.unwrap_or(0),
                    pool.next,
                    pool.total,
                    pool.refusals_during_wave,
                );
            }
            return Ok(Claim::TierBlocked);
        } else {
            // Exhausted with no tier standing. The caller buys ground and comes
            // back; this attempt only reports what it found.
            //
            // **The weight side must hear about it.** A refusal is the
            // KV side asking for more and being told no, and
            // [`RegionPool::spare_regions`]'s third guard reads this stamp to
            // decide that a side refused inside the last window has no spare
            // ground whatever its occupancy says. Stamping only on a *completed*
            // purchase would leave the guard dead in the one case that matters
            // most — the KV side refused and the weight side unwilling to sell —
            // and let the weight side take ground from a KV side that had just
            // been turned down.
            pool.last_pressure_at = Some(Instant::now());
            return Ok(Claim::Exhausted);
        };
        let base = pool.region_base(index);
        // **Whether it needs cleaning, not where it came from.** This used to key
        // on `recycled` — a region taken off the free list — which silently
        // assumed the free list was the only way a region could arrive with
        // someone else's bytes in it. It is not, and both of the others are on
        // the hot path:
        //
        // - **The wave transient tier.** It stands on KV ground for a whole
        //   forward and writes intermediates across all of it. When it is
        //   released, `next` may then advance straight into that ground as a
        //   *fresh* claim — no free-list entry, so no quiesce and no zeroing,
        //   while the forward's attention and FFN kernels are still in flight
        //   writing there. A `tier_reserve` used to keep the KV side off that
        //   band until the next placement moved it, which hid the omission; with
        //   the reserve gone the race is direct, and it produces exactly what it
        //   sounds like — KV chunks interleaved with activation bytes, decoding
        //   to NaN.
        // - **The weight boundary.** Ground conceded by the expert zone held
        //   expert weights a moment ago. The handover quiesces
        //   (`quiesce_before_handover`), so there are no readers left, but the
        //   bytes are still there and a fresh claim would hand them to a KV
        //   arena unzeroed.
        let dirty = pool.dirty_epoch[index] != 0;
        if dirty {
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
            let needs_sync = pool.dirty_epoch[index] == pool.quiesce_epoch;
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
            // Clean now, and its next tenant learns that from the stamp rather
            // than from having watched it happen.
            pool.dirty_epoch[index] = 0;
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
            if dirty { ", cleaned" } else { "" },
            pool.live,
            pool.total,
        );
        Ok(Claim::Got(RegionHandle {
            ordinal: stream.context().ordinal(),
            index,
            base,
        }))
    }
}

/// Regions [`claim_region`] can hand out: the free ones below the ceiling, plus
/// the fresh ones the ceiling still leaves ahead of `next`.
///
/// Split out from [`RegionPool::free_count`] so the rule is arithmetic that
/// tests without a device — the same reason [`tier_fits`] is a free function.
fn claimable(free_below_ceiling: usize, next: usize, ceiling: usize) -> usize {
    free_below_ceiling + ceiling.saturating_sub(next)
}

/// Regions nobody owns that the ceiling nonetheless forbids: the free ones at or
/// above it, plus everything past `next` that the ceiling has cut off.
///
/// `next.max(ceiling)` is the subtle term. Fresh ground runs from `next` to the
/// ceiling; above *both* is what neither path can reach. Taking `next` alone
/// would double-count the fresh regions [`claimable`] already returned, and
/// taking `ceiling` alone would count regions below `next` that are live.
fn blocked(free_at_or_above: usize, next: usize, total: usize, ceiling: usize) -> usize {
    free_at_or_above + total.saturating_sub(next.max(ceiling))
}

/// How many regions the KV side may reach, given where the tier stands.
///
/// A placed tier answers with its own base — real memory a running wave is
/// writing into, where a claim would be corruption. An absent one answers with
/// the boundary, because outside a wave nothing is standing there at all.
///
/// **A placed tier makes the ceiling deaf to the boundary**, which is the second
/// half of the section-prefill wedge. `transient_base` is an address, fixed when
/// the tier was placed; move the weight floor and only the absent arm follows it.
/// So a tier left standing past its forward caps the KV side at wherever it was
/// put, and no concession the weight side makes can lift that cap — the daemon
/// conceded itself down to its floor over thousands of retries while this
/// returned 293 every time. [`super::bump_arena::end_wave_transient`] closes it
/// by ending the tier's lifetime with its forward.
///
/// The absent arm used to subtract a `tier_reserve` — the last tier's width, held
/// back so the next placement would find its ground empty. It is gone: the widths
/// it was predicting ranged from 4 to 34 regions inside one run, so it withheld
/// ground a narrow wave did not want while failing to protect a wide one, and its
/// only reliable effect was refusing claims against memory nobody owned.
/// [`place_transient`] buys its ground at placement instead.
///
/// Split out from [`RegionPool::region_ceiling`] so that relationship is
/// arithmetic that tests without a device — the same reason [`claimable`] and
/// [`tier_fits`] are free functions.
fn ceiling_regions(
    transient_base: Option<u64>,
    weight_floor: u64,
    region_base: u64,
    total: usize,
) -> usize {
    let top = match transient_base {
        Some(base) => base,
        None => weight_floor.max(region_base),
    };
    let usable = top.saturating_sub(region_base) as usize;
    (usable / REGION_BYTES).min(total)
}

/// Whether `[base, base + len)` is ground the tier may stand on, or the bytes
/// by which it is not.
///
/// **The tier may only stand on ground no arena is using.** Bounding it below by
/// the start of the KV side is not enough, and that was the defect: a region
/// already handed out keeps its address for the life of its arena, so a tier
/// measured down from `W` lands on top of the *highest* live regions long before
/// it reaches region zero. [`RegionPool::region_ceiling`] stops later claims from
/// entering the tier but cannot revoke a claim already made, so the placement is
/// where the two have to be reconciled — and the only sound answer is to refuse
/// (principle 7: refuse rather than corrupt). [`RegionPool::set_weight_floor`]
/// guards the same invariant from the other side, for the same reason.
///
/// Split out from [`place_transient`] so the rule is arithmetic that tests
/// without a device.
fn tier_fits(base: u64, len: usize, live_end: u64, floor: u64) -> std::result::Result<(), u64> {
    let top = base.saturating_add(len as u64);
    // Both directions are reported, because either can be the binding one: the
    // frontier anchor computes a base upward from the arenas and overshoots
    // `floor`, while the default placement measures down from `floor` and
    // undershoots `live_end`. Taking the larger keeps the pressure figure honest
    // whichever it was — and taking a `saturating_sub` in each direction is what
    // stops the unsigned wrap the single-sided version had.
    let short = live_end.saturating_sub(base).max(top.saturating_sub(floor));
    if short > 0 {
        return Err(short);
    }
    Ok(())
}

/// **Place the wave transient tier for the forward about to run.**
///
/// The tier does not exist between forwards. This brings it into being with
/// `bytes` of extent, packed against the arena frontier, and returns its base.
/// The region ceiling drops to that base for the duration, so no arena can be
/// stamped into the ground the tier now occupies.
///
/// # Anchored at the arena frontier `A`
///
/// `docs/elastic_vram_partition.md` §7 phase 2. This leaves the whole remainder
/// in one contiguous run adjacent to the weight side, so the boundary can be
/// moved to `A + T` in the same operation rather than through a control loop
/// hunting for bytes stranded mid-span:
///
/// ```text
/// | persist | KV regions … | transient | expert slots |
/// ```
///
/// # It took two failed attempts to find what was actually in the way
///
/// **First attempt.** `Q8_0 x20` produced silently wrong output on every
/// session, reproducibly, while the region pool looked untouched: identical arena
/// creations (577, max index 163), zero class promotions, zero refusals, zero
/// fresh claims taken while the tier stood. That last figure was misleading —
/// the counters only observe the window in which the tier is *placed*, and the
/// tier was being released and re-taken between every layer phase, so claims
/// arriving in those gaps were invisible and still advanced `next` and moved the
/// tier under the sweep.
///
/// **Second attempt**, after [`super::bump_arena::plan_wave_transient`] began
/// reserving the tier for the whole forward — which closes that blind spot and
/// holds the address fixed for a sweep. The silent corruption became a loud
/// refusal, and the refusals said what the first attempt could not:
///
/// ```text
/// [tier-refusal] ... (ceiling 16, next 16, total 332) — count 1025
/// ```
///
/// **1025 refusals inside one forward, at the same placement.** That was read as
/// a bound on how wide a gap the anchor would need — hundreds of regions — and
/// the anchor was abandoned for a placement measured down from the weight floor,
/// with a `claim_reserve` gap sized to absorb the claims.
///
/// It was the wrong reading. Those 1025 claims are the compressor creating
/// size-class arenas *while the forward runs*, on the persistence thread, and a
/// wave is meant to be pre-allocated end to end: everything it writes is claimed
/// by admit and everything it attends over is resident before it opens. The
/// sealing thread may fill arenas throughout the wave — that is the point of it
/// running concurrently — but creating one moves the very frontier this
/// placement is measured against. `BackingInner::arena_window` now makes that
/// creation wait for the gap between forwards, and
/// [`super::bump_arena::plan_wave_transient`] waits for any such window to close
/// before it reads the frontier. With no claims arriving after the placement,
/// the gap that absorbed them is not needed and the anchor holds.
///
/// # It buys its ground rather than having it withheld
///
/// The tier needs `[W − len, W)` clear of live arenas, and an arena cannot be
/// asked to move — it holds its region for as long as it lives. The KV side used
/// to be kept off that ground continuously by `tier_reserve`, a standing
/// withholding of the *last* tier's width. That could not work: the widths it was
/// predicting ranged from 4 to 34 regions inside one run, so it neither covered a
/// wider tier nor released what a narrower one had left, and its reliable effect
/// was refusing arena claims against ground nobody owned.
///
/// So the ground is bought at the moment it is needed. If live arenas reach into
/// the tier's footprint, this asks the weight side for the shortfall — the same
/// purchase any claim makes — and places into what that frees. The KV side keeps
/// every region up to the boundary in the meantime, and the tier is priced
/// against what this wave actually needs rather than what the last one did.
pub(crate) fn place_transient(stream: &std::sync::Arc<CudaStream>, bytes: usize) -> Result<u64> {
    let short = match try_place(stream, bytes)? {
        Placed::At(base) => return Ok(base),
        // Not "the KV side is short of regions" — the tier itself does not fit,
        // and the regions in its way are live. One purchase, for exactly the
        // shortfall the placement measured.
        Placed::Short(regions) => regions,
    };
    buy_ground(stream, short)?;
    if let Placed::At(base) = try_place(stream, bytes)? {
        return Ok(base);
    }
    // **Sweep the empties before declaring the partition dead.** The footprint
    // check above counts *claimed* regions, and a region whose arena went
    // chunk-empty since the last periodic sweep still reads as claimed — on a
    // churn-heavy config (seal/requantize under repeats) hundreds of such
    // arenas can stand in the tier's ground while holding nothing. Releasing
    // them is a free-list push per arena, after which the retry re-measures
    // against only the genuinely live arenas.
    super::backing::global_release_empty_arenas();
    match try_place(stream, bytes)? {
        Placed::At(base) => Ok(base),
        Placed::Short(still_short) => {
            let len = bytes.div_ceil(REGION_BYTES) * REGION_BYTES;
            let (span_bytes, floor_off, live_off, total, live) = with_pool(stream, |pool| {
                Ok((
                    pool.span_bytes,
                    pool.weight_floor.saturating_sub(pool.span_base),
                    pool.live_end().saturating_sub(pool.span_base),
                    pool.total,
                    pool.live,
                ))
            })?;
            candle::bail!(
                "wave transient tier needs {len} B below the weight floor and is \
                 {still_short} regions into ground live KV arenas hold, which cannot \
                 move. The weight side could not concede them — it is at its own \
                 floor — so this wave is too wide for a partition that has nothing \
                 left to trade. (span {span_bytes} B, weight floor at +{floor_off} B, \
                 arena frontier at +{live_off} B, {live}/{total} regions live)"
            )
        }
    }
}

/// The outcome of one attempt at [`place_transient`].
enum Placed {
    /// The tier stands at this address.
    At(u64),
    /// Live arenas reach this many regions into its footprint.
    Short(usize),
}

fn try_place(stream: &std::sync::Arc<CudaStream>, bytes: usize) -> Result<Placed> {
    with_pool(stream, |pool| {
        let len = bytes.div_ceil(REGION_BYTES) * REGION_BYTES;
        let floor = pool.weight_floor;
        // **The tier sits at the arena frontier, with no gap above it.**
        //
        // A gap would be room for claims arriving after the placement, and there
        // are none: every region claim creates an arena,
        // `BackingInner::arena_window` makes arena creation wait for the gap
        // between forwards, and `plan_wave_transient` waits for any window still
        // open before it reads the frontier here. So the frontier this measures
        // is the frontier for the whole forward.
        //
        // The gap it replaces was seeded per forward (`set_claim_reserve`) and
        // grown from observation (`fresh_per_forward_peak`), and both existed to
        // absorb the compressor's mid-wave arena creation. With that forbidden,
        // the tier packs directly against the arenas and every byte between it
        // and the weight floor stays claimable — which is the ground the failing
        // ingest ran out of while `blocked` said the tier was standing on it.
        //
        // `live_end` rather than the `next` watermark: a region freed from the
        // top of the span is ground the tier may stand on, and measuring from
        // the bump pointer would strand it for the life of the process.
        let live_end = pool.live_end();
        let base = live_end;
        if let Err(short) = tier_fits(base, len, live_end, floor) {
            return Ok(Placed::Short(
                (short as usize).div_ceil(REGION_BYTES).max(1),
            ));
        }
        if tier_poison_enabled() {
            // SAFETY: `[base, base+len)` is mapped device memory inside the
            // reservation, and the quiesce above retired every reader of it.
            //
            // The context is bound first and the result checked, because a
            // memset that silently fails is a diagnostic that silently lies —
            // and this one is used to decide whether a bug is real.
            stream
                .context()
                .bind_to_thread()
                .and_then(|()| unsafe { memset_d8_sync(base, 0xCD, len) })
                .map_err(candle::Error::wrap)?;
        }
        pool.transient_base = Some(base);
        pool.transient_bytes = len;
        Ok(Placed::At(base))
    })
}

/// Whether to fill the transient tier with a poison byte as it is placed.
///
/// `KV_TIER_POISON=1`. Off by default: it costs a synchronous device memset of
/// the whole tier on every forward.
fn tier_poison_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("KV_TIER_POISON")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// The tier vanishes: nothing is live in it, the arenas that were laid out inside
/// it are detached, and **its ground goes back to the KV side in full**.
///
/// The release used to hold a `tier_reserve` behind it, so the ceiling stayed
/// where the tier had been until the next placement moved it. That reserve is
/// gone (see [`ceiling_regions`]); the ceiling is the boundary the instant the
/// tier is released, and the next [`place_transient`] buys back whatever it needs
/// from the weight side rather than being kept a standing allowance.
///
/// # The ground comes back dirty, and saying so is what makes that safe
///
/// A tier writes activations across every byte it stands on, and its kernels are
/// still in flight when this runs — a dropped [`super::bump_arena::Generation`]
/// says only that no *host-side* range names the span. So every region the tier
/// covered is marked as needing the same quiesce-and-zero a recycled one gets,
/// and [`claim_region`] will not hand one out until it has had it.
///
/// **Without this the reserve was load-bearing for a reason nobody had written
/// down.** It kept the KV side off the tier's band, which meant the omission — a
/// *fresh* claim skipping the clean entirely — could not be reached. Removing the
/// reserve made it reachable in one step: `next` advances into ground the last
/// forward is still writing, the arena and the activations interleave, and the
/// KV decodes to NaN.
pub(crate) fn release_transient(stream: &std::sync::Arc<CudaStream>) {
    let _ = with_pool(stream, |pool| {
        if let (Some(base), true) = (pool.transient_base, pool.transient_bytes > 0) {
            pool.mark_dirty_span(base, pool.transient_bytes);
        }
        pool.transient_base = None;
        pool.transient_bytes = 0;
        Ok(())
    });
}

/// Carve the persistence thread's staging block, out of the fixed left end.
///
/// Separate from [`place_transient`] because the block is: it is carved **once**
/// and never moves, where the wave tier is placed and released every forward.
/// The persistence thread's copy stream is not synchronised to a wave, so its
/// ranges can be live at any moment one begins — it is the one domain that
/// cannot be allowed to vanish, and therefore the one that needs a fixed
/// address.
pub(crate) fn carve_persist(stream: &std::sync::Arc<CudaStream>, bytes: usize) -> Result<u64> {
    with_pool(stream, |pool| {
        let len = bytes.div_ceil(REGION_BYTES) * REGION_BYTES;
        if pool.persist_carved + len > PERSIST_SPAN_BYTES {
            candle::bail!(
                "persistence staging block exhausted: {len} B on top of {} B carved \
                 exceeds the {PERSIST_SPAN_BYTES} B block",
                pool.persist_carved,
            )
        }
        let base = pool.span_base + pool.persist_carved as u64;
        pool.persist_carved += len;
        Ok(base)
    })
}

/// Claim the reservation if it does not exist, and report the span's right edge.
///
/// The address an expert cache builds its [`super::weight_zone::WeightZone`]
/// from: slots fill leftward from here.
pub fn span_end(stream: &std::sync::Arc<CudaStream>) -> Result<u64> {
    with_pool(stream, |pool| Ok(pool.span_end()))
}

/// Bytes the weight side may **ever** hold — the span less
/// [`MIN_ELASTIC_RESERVE`].
///
/// The cap the expert cache sizes itself against, replacing
/// `VramGovernor::expert_budget`. It is a *position*, not a budget: there is no
/// arithmetic here about what anything else might want, because everything else
/// is on the other side of the floor by construction.
pub fn weight_capacity_bytes(stream: &std::sync::Arc<CudaStream>) -> Result<usize> {
    with_pool(stream, |pool| {
        Ok(pool.span_bytes.saturating_sub(MIN_ELASTIC_RESERVE))
    })
}

/// Bytes the weight side takes **at load** — the span less
/// [`INITIAL_KV_RESERVE`].
///
/// The opening position, not the limit. [`weight_capacity_bytes`] is how far the
/// zone may grow once the KV side has shown what it actually uses; this is where
/// it starts, before any wave has said anything.
pub fn initial_weight_bytes(stream: &std::sync::Arc<CudaStream>) -> Result<usize> {
    with_pool(stream, |pool| {
        Ok(pool.span_bytes.saturating_sub(INITIAL_KV_RESERVE))
    })
}

/// Place the weight side's left edge at `floor` and re-derive the KV side.
///
/// Called at model load to open the boundary, and between waves to move it.
/// Returns the region count that survives.
pub fn set_weight_floor(stream: &std::sync::Arc<CudaStream>, floor: u64) -> Result<usize> {
    let ordinal = stream.context().ordinal();
    // **The latch.** Moving the boundary evicts experts and relocates others; a
    // wave in flight may be reading either. The design's answer is that this
    // only ever runs between forwards, which is a property of the call site — so
    // it is checked here rather than trusted, in the one place every move goes
    // through.
    //
    // It has already caught a call site that was wrong about that. The weight
    // side's take-back was driven from the expert pipeline's end-of-pass, which
    // reads as "outside a wave" and is not: the pipeline answers a MoE layer
    // while the forward thread sits inside `ffn_forward`, under that layer's FFN
    // wave guard. Callers must treat a refusal as a refusal — see
    // `renegotiate_boundary`, which for a while applied its half of the move and
    // then let this bail propagate as a warning.
    if super::bump_arena::wave_is_live(ordinal) {
        candle::bail!(
            "refusing to move the weight boundary while a wave generation is open: \
             a retraction evicts and relocates expert slots, and a wave in flight \
             may be reading them. The boundary moves at the expert pipeline's \
             end-of-pass, where no GEMM for the pass is still being issued."
        )
    }
    with_pool(stream, |pool| {
        let total = pool.set_weight_floor(floor)?;
        log::debug!(
            "reservation: weight floor at {floor:#x} — {} MiB to the weight side, \
             {total} regions to KV",
            (pool.span_end() - floor) / (1024 * 1024),
        );
        Ok(total)
    })
}

/// **The negotiation, from the KV side — and only in the growing direction.**
///
/// Regions the KV side is holding free beyond `slack` that the weight side could
/// take back, bounded by [`KV_GROW_STEP`].
///
/// It used to answer `(wanted, spare)`, where `wanted` drained an accumulated
/// count of refused claims. That half is gone, and its removal is the point.
/// **A counter records events; the weight side spent them as regions.** One
/// failed section-quantize drain walked the size-class ladder and left 4,436
/// units of demand behind it against a KV side that was twenty-eight regions
/// short — and the boundary paid all of it, evicting 1,598 experts and taking the
/// zone to a capacity below its own pinned working set, from which every
/// subsequent forward failed. Now a claim buys exactly the ground it needs at the
/// moment it needs it ([`set_ground_broker`]), so there is nothing to accumulate
/// and nothing to convert: the allocation *is* the measurement.
pub fn kv_spare_regions(stream: &std::sync::Arc<CudaStream>, slack: usize) -> Result<usize> {
    with_pool(stream, |pool| {
        Ok(pool.spare_regions(slack).min(KV_GROW_STEP))
    })
}

/// The address the weight floor would sit at if the weight side gave up
/// `regions` more regions (positive) or took `regions` fewer (negative).
///
/// Expressed as an address rather than a count so the caller — which thinks in
/// slots, not regions — converts once, in one direction, using
/// [`super::weight_zone::WeightZone::capacity_for_frontier`].
pub fn weight_floor_after(stream: &std::sync::Arc<CudaStream>, delta: isize) -> Result<u64> {
    with_pool(stream, |pool| {
        let shift = (delta.unsigned_abs() * REGION_BYTES) as u64;
        let floor = if delta >= 0 {
            (pool.weight_floor + shift).min(pool.span_end())
        } else {
            pool.weight_floor.saturating_sub(shift)
        };
        Ok(floor.max(pool.span_base + MIN_ELASTIC_RESERVE as u64))
    })
}

/// The reservation's addresses, as they stand right now.
///
/// Every device pointer this engine hands a KV kernel has to land somewhere in
/// here, and which somewhere decides whether it is legal. Published so
/// [`super::guard`] can answer that question without duplicating the layout —
/// there is exactly one authority for where the boundary is, and it is this
/// module.
#[derive(Debug, Clone, Copy)]
pub struct SpanLayout {
    /// First byte of the whole reservation.
    pub span_base: u64,
    /// One past its last byte.
    pub span_end: u64,
    /// First byte of the region range — the persistence staging block sits
    /// between `span_base` and here.
    pub region_base: u64,
    /// Bytes carved from the staging block so far.
    pub persist_carved: usize,
    /// Regions the KV side currently has.
    pub total: usize,
    /// The weight side's left edge: KV below, expert slots above.
    pub weight_floor: u64,
    /// The wave transient tier, while one stands.
    pub transient_base: Option<u64>,
    pub transient_bytes: usize,
}

impl SpanLayout {
    /// One past the last byte of the region range.
    pub fn region_end(&self) -> u64 {
        self.region_base + (self.total * REGION_BYTES) as u64
    }
}

/// The reservation's current addresses, or `None` if this device has none yet.
pub fn span_layout(ordinal: usize) -> Option<SpanLayout> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| SpanLayout {
        span_base: pool.span_base,
        span_end: pool.span_end(),
        region_base: pool.region_base,
        persist_carved: pool.persist_carved,
        total: pool.total,
        weight_floor: pool.weight_floor,
        transient_base: pool.transient_base,
        transient_bytes: pool.transient_bytes,
    })
}

/// Occupancy of a device's KV side, or `None` if it has no reservation yet.
pub fn region_stats(ordinal: usize) -> Option<RegionStats> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| RegionStats {
        total: pool.total,
        live: pool.live,
        free: pool.free_count(),
        blocked: pool.ceiling_blocked(),
        peak_live: pool.peak_live,
        transient_bytes: pool.transient_bytes,
        transient_ceiling: pool.region_ceiling(),
        fresh_claims_during_wave: pool.fresh_claims_during_wave,
        refusals_during_wave: pool.refusals_during_wave,
        weight_bytes: (pool.span_end() - pool.weight_floor) as usize,
        slack_bytes: layout_span(pool.span_base, pool.span_bytes, pool.weight_floor).slack,
        reserved_bytes: pool.reservation.reserved_bytes(),
        granularity: pool.reservation.granularity(),
    })
}

#[cfg(test)]
mod tests {
    use super::{claim_region, place_transient, region_stats, release_transient, REGION_BYTES};
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

        // The free list is shared with every other test in this binary, so the
        // NEXT claim is not necessarily the region just released — whichever
        // tests ran earlier may have left their own queued ahead of it. Claim
        // until the dirtied one comes back, holding the others so they cannot be
        // handed out again, and check THAT one.
        //
        // Asserting identity on the first claim made this test depend on
        // execution order (it failed with two bases exactly one region apart),
        // and on the failing path it went on to read from `base` after the pool
        // had handed that region to someone else.
        // Small on purpose: each iteration HOLDS a region, and the pool's
        // sub-tier class is only a few dozen deep, so a large scan would starve
        // whatever runs next instead of skipping a couple of stale entries.
        const MAX_SCAN: usize = 8;
        let mut held = Vec::new();
        let mut recycled = None;
        for _ in 0..MAX_SCAN {
            let Some(r) = claim_region(&s)? else { break };
            if r.base() == base {
                recycled = Some(r);
                break;
            }
            held.push(r);
        }
        let again = recycled.expect("the dirtied region is returned to the pool");
        assert_eq!(again.base(), base);
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

    /// **Ground the transient tier stood on is cleaned before it is re-let**,
    /// even though it never passed through the free list.
    ///
    /// This is the regression that produced NaN in a live conversation, and the
    /// path is entirely *fresh* claims. The tier writes activations across its
    /// whole span for the length of a forward; when it is released the ceiling
    /// rises to the boundary and `next` may advance straight into that band. A
    /// fresh claim used to skip both the quiesce and the zero — the cleaning was
    /// keyed on "came off the free list" rather than "holds someone else's
    /// bytes" — so a KV arena was handed a region full of activation data while
    /// the kernels that wrote it were still in flight.
    ///
    /// A `tier_reserve` had been keeping the KV side off that band, which is why
    /// the omission was unreachable until the reserve was removed. It was load
    /// bearing for a reason its own doc comment never stated.
    #[test]
    fn tier_ground_is_cleaned_before_a_fresh_claim_takes_it() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };

        // Place a tier, dirty its first region the way a forward's intermediates
        // would, then release it — the state at the top of the next forward.
        let base = place_transient(&s, REGION_BYTES)?;
        let dirt = vec![0xA7u8; 8192];
        // SAFETY: `[base, base + REGION_BYTES)` is mapped device memory inside
        // the reservation, reserved to this tier and to nothing else.
        unsafe {
            candle::cuda_backend::cudarc::driver::result::memcpy_htod_async(
                base,
                &dirt,
                s.cu_stream(),
            )
        }
        .map_err(|e| candle::Error::Msg(format!("dirtying the tier: {e}")))?;
        s.synchronize()
            .map_err(|e| candle::Error::Msg(format!("sync: {e}")))?;
        release_transient(&s);

        // Claim until one lands in the tier's span. Every claim here is fresh:
        // nothing was released, so the free list is empty of these indices.
        let mut held = Vec::new();
        let mut seen = false;
        while let Some(region) = claim_region(&s)? {
            let hit = region.base() == base;
            held.push(region);
            if hit {
                seen = true;
                break;
            }
        }
        assert!(seen, "no claim reached the tier's ground — test is inert");

        let mut back = vec![0xFFu8; 8192];
        // SAFETY: reading the region this test now holds.
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
            "a region the tier had written was handed out still holding its \
             activation bytes"
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

    /// **A region the ceiling forbids is not free**, and the daemon wedged on
    /// the difference.
    ///
    /// `free` answers "what would a claim get right now", so counting
    /// unreachable ground there is a promise the allocator then refuses to
    /// keep. That is what happened: a wide section prefill placed a 496 MiB
    /// tier, the ceiling landed at 293 of 324 regions, live reached 293, and
    /// the 31 regions above the ceiling were reported free. Admission planned
    /// against 496 MiB it could not have and every claim was refused — 12,488
    /// of them in four seconds, with no forward completing to move the
    /// boundary. Consumers deciding for *after* the tier releases add
    /// `blocked` back on (`vram_budget_available` is
    /// `(free + blocked) × REGION_BYTES`), which is honest precisely because
    /// `free` itself is not inflated.
    #[test]
    fn ground_under_the_tier_is_not_counted_as_free() {
        // The shape at the wedge: 324 regions all handed out at least once
        // (`next` = 324), 293 live, 31 on the free list at indices 293..323,
        // and a 496 MiB tier putting the ceiling at 293.
        assert_eq!(super::claimable(0, 324, 293), 0, "nothing is claimable");
        assert_eq!(
            super::blocked(31, 324, 324, 293),
            31,
            "and the 31 are blocked, not free"
        );

        // Between waves the tier is gone, the ceiling is the whole pool, and
        // the same 31 become spendable again.
        assert_eq!(super::claimable(31, 324, 324), 31);
        assert_eq!(super::blocked(0, 324, 324, 324), 0);
    }

    /// The fresh-region term is counted once, by whichever side owns it.
    #[test]
    fn fresh_ground_is_claimable_or_blocked_but_never_both() {
        // 100 regions, next=40, ceiling=70: 30 fresh are claimable, 30 are cut
        // off above the ceiling, and no free-list entries exist either side.
        assert_eq!(super::claimable(0, 40, 70), 30);
        assert_eq!(super::blocked(0, 40, 100, 70), 30);
        assert_eq!(
            super::claimable(0, 40, 70) + super::blocked(0, 40, 100, 70) + 40,
            100,
            "claimable + blocked + allocated must be the whole pool"
        );

        // A ceiling below `next` cuts off nothing fresh — `next` has already
        // passed it — and leaves the tail beyond `next` blocked.
        assert_eq!(super::claimable(0, 80, 70), 0);
        assert_eq!(super::blocked(0, 80, 100, 70), 20);
    }

    /// **A tier caps the KV side only while it is standing**, and the boundary
    /// is the cap the rest of the time.
    ///
    /// Both halves of this were defects. A tier left standing past its forward
    /// pins the ceiling at an address, so the weight side could concede itself
    /// down to its floor and this still returned 293 —
    /// `bump_arena::end_wave_transient` ends the tier's life with its forward.
    /// And once released, the ceiling used to stop short by `tier_reserve`, the
    /// last tier's width, held against the next placement: that refused claims
    /// against ground nobody owned, and the refusal escalated into a boundary
    /// renegotiation which handed over the same regions by the most expensive
    /// route there is.
    #[test]
    fn the_ceiling_is_the_tier_while_it_stands_and_the_boundary_after() {
        let base = 0u64;
        let total = 324;
        // A 496 MiB tier placed against a floor 324 regions up: 31 regions of
        // tier, so it stands at region 293.
        let tier = 31 * REGION_BYTES;
        let floor = (total * REGION_BYTES) as u64;
        let placed = Some(floor - tier as u64);

        assert_eq!(
            super::ceiling_regions(placed, floor, base, total),
            293,
            "a standing tier caps the KV side at its own base — real memory a \
             running wave is writing into"
        );
        assert_eq!(
            super::ceiling_regions(None, floor, base, total),
            324,
            "released, every region up to the boundary is claimable: nothing is \
             standing there, so nothing is withheld"
        );

        // The boundary moves. A placed tier is deaf to it; an absent one is not.
        let moved = floor + (40 * REGION_BYTES) as u64;
        assert_eq!(
            super::ceiling_regions(placed, moved, base, total),
            293,
            "a placed tier pins the ceiling at its base, whatever the floor does"
        );
        assert_eq!(
            super::ceiling_regions(None, moved, base, total),
            324,
            "and without one the ceiling follows the boundary, capped by the pool"
        );
    }

    /// **The tier never stands on a live region**, however full the KV side is.
    ///
    /// The regression this pins: the fit test used to bound the tier below by
    /// the *start* of the KV side, so a tier measured down from `W` was accepted
    /// while it sat on top of the highest live arenas. `region_ceiling` refuses
    /// later claims but cannot revoke one already made, so the wave wrote its
    /// intermediates into live KV bytes — silently, with no error anywhere.
    ///
    /// Pure arithmetic, so it holds on a machine with no GPU.
    #[test]
    fn the_tier_never_overlaps_a_live_region() {
        let r = REGION_BYTES as u64;
        let region_base = 0x1000_0000u64;
        let floor = region_base + 332 * r; // 332 regions
        let len = 4 * REGION_BYTES; // a decode-sized tier
        let base = floor - len as u64; // an arbitrary base, hard against the floor

        // Empty KV side: the tier sits above every region and fits.
        assert!(super::tier_fits(base, len, region_base, floor).is_ok());

        // 328 live regions — the tier lands exactly on the frontier, no overlap.
        let live_end = region_base + 328 * r;
        assert!(
            super::tier_fits(base, len, live_end, floor).is_ok(),
            "abutting the frontier is not overlapping it"
        );

        // 329 live: one region of the tier is inside live KV. Refuse.
        let live_end = region_base + 329 * r;
        assert_eq!(
            super::tier_fits(base, len, live_end, floor),
            Err(r),
            "one region short must be reported as one region short"
        );

        // The old bound — the start of the KV side — accepted every one of
        // these. The whole span being live is the extreme case.
        let live_end = region_base + 332 * r;
        assert_eq!(super::tier_fits(base, len, live_end, floor), Err(4 * r));
    }

    /// **The placement anchors at the arena frontier, and only the floor can
    /// refuse it.**
    ///
    /// `try_place` sets `base = live_end`, so the live-region side of
    /// [`tier_fits`] is satisfied by construction — a tier that begins where the
    /// arenas end cannot overlap them — and every byte between the tier's top
    /// and the weight floor stays claimable. What binds instead is the floor:
    /// when the tier does not fit in the run above the frontier, the shortfall
    /// is what the boundary must move by, which is exactly what `Placed::Short`
    /// asks the weight side for.
    ///
    /// The placement this replaced measured *down* from the floor, which left
    /// the free run stranded between the arenas and the tier where only a
    /// control loop could find it.
    #[test]
    fn the_tier_anchors_at_the_frontier_and_only_the_floor_refuses() {
        let r = REGION_BYTES as u64;
        let region_base = 0x1000_0000u64;
        let floor = region_base + 100 * r;

        // 40 regions live, an 8-region tier: it abuts the arenas and there are
        // 52 regions of claimable ground above it.
        let live_end = region_base + 40 * r;
        let len = 8 * REGION_BYTES;
        assert!(
            super::tier_fits(live_end, len, live_end, floor).is_ok(),
            "a tier beginning where the arenas end never overlaps them"
        );

        // An empty KV side is the same rule with the frontier at the bottom.
        assert!(super::tier_fits(region_base, len, region_base, floor).is_ok());

        // 95 live regions leave 5 for an 8-region tier: three short, and the
        // floor is the only side that can be.
        let live_end = region_base + 95 * r;
        assert_eq!(
            super::tier_fits(live_end, len, live_end, floor),
            Err(3 * r),
            "the shortfall is what the weight boundary has to concede"
        );
    }

    /// The shortfall is reported from **whichever** side binds, and never wraps.
    ///
    /// The frontier anchor computes its base upward from the arenas, so it
    /// overshoots `floor` rather than undershooting `live_end` — the direction
    /// the single-sided `region_base - base` could not express, and underflowed
    /// on: in release it wrapped to ~2^64 and drove `kv_pressure` to a retraction
    /// request larger than the span.
    #[test]
    fn the_tier_shortfall_reports_both_directions_without_wrapping() {
        let r = REGION_BYTES as u64;
        let region_base = 0x1000_0000u64;
        let floor = region_base + 100 * r;

        // Anchored above the frontier and running two regions past the floor.
        let base = region_base + 99 * r;
        let len = 3 * REGION_BYTES;
        assert_eq!(
            super::tier_fits(base, len, region_base, floor),
            Err(2 * r),
            "overshooting the floor is a two-region shortfall, not a wrap"
        );

        // Both bounds violated at once: the larger of the two is what the
        // boundary has to move by.
        let base = region_base + 10 * r;
        let live_end = region_base + 12 * r;
        let len = 95 * REGION_BYTES;
        assert_eq!(
            super::tier_fits(base, len, live_end, floor),
            Err(5 * r),
            "past the floor by 5 regions, into live by 2 — the binding one is 5"
        );
    }

    /// **The span is `usable` less the pool cushion — and nothing else.**
    ///
    /// Specifically it is *not* also less the dense weights. They are resident
    /// by the time this runs, so `usable()` has already netted them out of `C`;
    /// subtracting `class_reserved(Weights)` here as well would book the same
    /// bytes twice. The load order is what makes that mistake look like
    /// prudence, so the identity is pinned rather than left to a comment.
    ///
    /// A pure-arithmetic test, so it holds on a machine with no GPU.
    #[test]
    fn the_span_is_usable_less_the_cushion_and_nothing_else() {
        let mib = 1024 * 1024;
        let cushion = 512 * mib;
        let capacity = 14_592 * mib;
        for dense_mib in [0usize, 1024, 4096] {
            let dense = dense_mib * mib;
            // What `usable()` reports once `dense` bytes are resident: the drop
            // in headroom since C was measured is exactly the weights.
            let usable = capacity - dense;
            assert_eq!(
                super::span_from(usable, cushion),
                capacity - dense - cushion,
                "the weights must be subtracted once, not twice, at {dense_mib} MiB"
            );
        }
    }

    /// **The tier's ceiling binds the free list, not just fresh regions.**
    ///
    /// A region freed while the tier was small — or absent, between forwards —
    /// keeps its index, and the next wave's tier can be placed below it. Handing
    /// it out then puts KV writes inside the wave's own intermediates, and
    /// nothing downstream would notice until the output was wrong.
    ///
    /// Pure arithmetic against `region_ceiling`, so it holds with no GPU: what
    /// the bug was is a `free.pop()` that never consulted the ceiling at all.
    #[test]
    fn the_tier_ceiling_binds_recycled_regions() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 4096 * mib;
        let l = super::layout_span(base, span, base + span as u64);
        // A tier placed 8 regions below the weight floor drops the ceiling by 8.
        let tier = 8 * REGION_BYTES;
        let tier_base = (base + span as u64) - tier as u64;
        let ceiling = ((tier_base - l.region_base) as usize) / REGION_BYTES;
        assert_eq!(
            ceiling,
            l.total - 8,
            "the tier must remove exactly its own regions from the ceiling"
        );
        // Every index at or above the ceiling is inside the tier — whether it is
        // fresh or recycled is not a property the address knows about.
        for idx in ceiling..l.total {
            let addr = l.region_base + (idx * REGION_BYTES) as u64;
            assert!(
                addr >= tier_base,
                "region {idx} sits inside the tier at {tier_base:#x}"
            );
        }
        assert!(
            l.region_base + ((ceiling - 1) * REGION_BYTES) as u64 + REGION_BYTES as u64
                <= tier_base,
            "the last usable region must end at or before the tier"
        );
    }

    /// A span too small for the persist block yields no regions rather than
    /// wrapping into a colossal one. (`span_from` itself is only
    /// `usable − cushion`; a span that cannot hold the fixed block is refused
    /// outright in `create`.)
    ///
    /// Note what is *not* subtracted: the wave tier. At rest it occupies
    /// nothing, so its ground belongs to the KV side until a forward places it.
    #[test]
    fn a_span_smaller_than_the_persist_block_yields_no_regions() {
        assert_eq!(super::span_from(0, 1024), 0, "a cushion larger than usable");
        let base = 0x1_0000_0000u64;
        let persist = super::PERSIST_SPAN_BYTES;
        let l = super::layout_span(base, persist, base + persist as u64);
        assert_eq!(l.total, 0, "the persist block leaves nothing for KV");
        assert_eq!(l.slack, 0);
        // A span below even that stays sane rather than wrapping.
        let l = super::layout_span(base, persist / 2, base + (persist / 2) as u64);
        assert_eq!(l.total, 0);
        assert_eq!(l.slack, 0);
        // And the tier's worth of ground *is* the KV side's while nothing runs.
        let with_tier = persist + super::MAX_WAVE_TRANSIENT_BYTES;
        let l = super::layout_span(base, with_tier, base + with_tier as u64);
        assert_eq!(
            l.total,
            super::MAX_WAVE_TRANSIENT_BYTES / REGION_BYTES,
            "at rest the tier's ground belongs to the KV side"
        );
    }

    /// The layout is positional from the right: the persist block is at the
    /// left, the transient tier is measured back from the weight floor, and the
    /// regions are what is left in the middle. Nothing overlaps and nothing
    /// escapes the span.
    #[test]
    fn the_layout_partitions_the_span_without_overlap() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        for span_mib in [3072usize, 8192, 14336] {
            for weight_mib in [0usize, 1024, 4096] {
                let span = span_mib * mib;
                let span_end = base + span as u64;
                let floor = span_end - (weight_mib * mib) as u64;
                let l = super::layout_span(base, span, floor);

                // Only the persist block is fixed; the regions start straight
                // after it and run up to the weight floor. The wave tier is not
                // in the layout at all — it exists only while a forward runs.
                assert_eq!(l.region_base, base + super::PERSIST_SPAN_BYTES as u64);
                let region_end = l.region_base + (l.total * REGION_BYTES) as u64;
                assert!(
                    region_end <= l.weight_floor,
                    "regions ran into the weight side at {span_mib}/{weight_mib}"
                );
                assert!(l.weight_floor <= span_end, "weight side past the span end");
                assert_eq!(
                    region_end + l.slack as u64,
                    l.weight_floor,
                    "slack must account for exactly the rounding remainder"
                );
            }
        }
    }

    /// A weight floor beyond either edge is clamped rather than wrapped. The
    /// floor arrives from arithmetic done a crate away, and a wrapped
    /// subtraction here would produce an address that looks plausible.
    #[test]
    fn the_layout_clamps_a_floor_outside_the_span() {
        let base = 0x1_0000_0000u64;
        let span = 4096 * 1024 * 1024;
        let past_end = super::layout_span(base, span, base + span as u64 + 4096);
        assert_eq!(past_end, super::layout_span(base, span, base + span as u64));
        let before_start = super::layout_span(base, span, 0);
        assert_eq!(
            before_start.total, 0,
            "a floor left of the regions leaves none"
        );
    }

    /// **The transient tier is not in the resting layout at all.**
    ///
    /// Between forwards it occupies nothing, so the regions run from the persist
    /// block straight to the weight floor. That is what makes the tier the one
    /// block in the span that can change size: at the moment its extent changes
    /// it holds nothing, so a resize leaves no hole and moves no data — and it is
    /// why it can sit *between* the arenas and the weights, absorbing both sides'
    /// movement, which nothing at a fixed address could do.
    #[test]
    fn the_resting_layout_has_no_transient_tier() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 12288 * mib;
        let span_end = base + span as u64;
        let expected = base + super::PERSIST_SPAN_BYTES as u64;
        for weight_mib in [0usize, 512, 2048, 6144, 9216] {
            let l = super::layout_span(base, span, span_end - (weight_mib * mib) as u64);
            assert_eq!(l.region_base, expected, "regions start after persist");
            // Every byte between the persist block and the weight floor is
            // available to regions when no forward is running.
            let floor = l.weight_floor;
            assert_eq!(
                l.total * REGION_BYTES + l.slack,
                (floor - expected) as usize,
                "resting layout left a gap at weight={weight_mib} MiB"
            );
        }
    }

    /// Moving the boundary changes exactly one thing: how many regions exist.
    /// Every megabyte the weight side gives up becomes KV regions, one for one
    /// (modulo the rounding the slack accounts for).
    #[test]
    fn giving_ground_becomes_regions_one_for_one() {
        let base = 0x1_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 12288 * mib;
        let span_end = base + span as u64;
        let at = |weight_mib: usize| {
            super::layout_span(base, span, span_end - (weight_mib * mib) as u64)
        };

        let wide = at(6144);
        let narrow = at(4096);
        assert_eq!(
            narrow.total - wide.total,
            (2048 * mib) / REGION_BYTES,
            "2048 MiB of weights is exactly that many regions of KV"
        );
    }

    /// The whole span is accounted for: persist + regions + slack + weights is
    /// exactly the span, at every boundary position. (No transient term — at
    /// rest it has none.)
    #[test]
    fn every_byte_of_the_span_is_accounted_for() {
        let base = 0x2_0000_0000u64;
        let mib = 1024 * 1024;
        let span = 9216 * mib;
        let span_end = base + span as u64;
        for weight_mib in [0usize, 512, 2048, 6144] {
            let floor = span_end - (weight_mib * mib) as u64;
            let l = super::layout_span(base, span, floor);
            let weights = (span_end - l.weight_floor) as usize;
            let total = super::PERSIST_SPAN_BYTES + l.total * REGION_BYTES + l.slack + weights;
            assert_eq!(total, span, "span unaccounted at weight={weight_mib} MiB");
        }
    }

    /// Live + free + blocked accounts for every region: nothing leaks out of
    /// the span.
    ///
    /// **Three buckets, not two.** `free` used to mean "no arena owns it" and
    /// this asserted `live + free == total`; it now means "claimable right now",
    /// which is what the admission budget needs it to mean, and the ground the
    /// tier's ceiling puts out of reach is counted separately. A region is
    /// therefore in exactly one of three states, and the sum is still the span.
    #[test]
    fn every_region_is_live_free_or_blocked() -> Result<()> {
        let _serial = serial();
        let Some(s) = stream() else { return Ok(()) };
        let held = claim_region(&s)?.expect("a region");
        let stats = region_stats(0).expect("a pool");
        assert_eq!(stats.live + stats.free + stats.blocked, stats.total);
        assert!(stats.live >= 1);
        drop(held);
        let after = region_stats(0).expect("a pool");
        assert_eq!(after.live + after.free + after.blocked, after.total);
        Ok(())
    }
}
