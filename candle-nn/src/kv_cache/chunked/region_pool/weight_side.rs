//! The weight side's seller and the moves of its floor, which decide how much ground the KV side may buy.
//!
//! The KV side buys ground through the broker installed for its device. The
//! broker blocks while the expert cache evicts slots on its own pipeline thread,
//! so the eviction happens where it is safe. The floor moves only between
//! forwards, and `set_weight_floor` refuses a move that would cut live KV or a
//! standing tier rather than clamping it. Ground the weight side gives up is
//! marked dirty. The read-only declaration for diagnostics follows the floor, so
//! a region the tier now stands on is not reported as corrupted.

use super::*;

/// Buys KV ground from the weight side: `regions` in, bytes conceded out.
///
/// Installed once by the expert cache, which owns the only thing that can pay —
/// expert residency, whose cold tier holds a valid copy of every expert, so a
/// slot given up costs a reload and never a loss.
pub(super) type GroundBroker = Arc<dyn Fn(usize) -> u64 + Send + Sync>;

/// Sellers by device ordinal.
///
/// **Keyed, and replaceable.** A `OnceLock<GroundBroker>` was neither, and both
/// were wrong: one broker for the process misroutes every purchase on the second
/// GPU to the first one's expert cache, and a set-once cell means a second model
/// load leaves the first model's dead `Weak` installed — every later purchase
/// answers zero and the partition silently reverts to refusing claims it could
/// have paid for.
pub(super) fn ground_brokers() -> &'static Mutex<HashMap<usize, GroundBroker>> {
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

/// Run `f` with the weight side closed to purchase on this thread: a claim
/// inside it takes free ground only, and one that finds none refuses exactly as
/// it would with no seller. For allocations made on a guess — ground they may
/// take if it stands free, never ground worth evicting an expert for.
pub(crate) fn without_purchase<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            BUYING.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(BUYING.with(|b| b.replace(true)));
    f()
}

/// Whether this thread's claims are closed to purchase right now.
#[cfg(test)]
pub(crate) fn purchase_closed() -> bool {
    BUYING.with(|b| b.get())
}

/// Ask the weight side for `regions` of ground; answers the bytes it conceded.
///
/// **Called with no pool lock held.** The broker blocks on the pipeline thread,
/// which takes that lock to move the floor.
pub(super) fn buy_ground(stream: &std::sync::Arc<CudaStream>, regions: usize) -> Result<u64> {
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
    // Whether the weight side can be made to concede is the question a blocking
    // claim would exist to answer: a claim only has something to wait *for* if
    // concessions are being refused for a reason that passes. Counted so that is
    // measured rather than assumed.
    SPARE_TALLY[if conceded > 0 { 4 } else { 5 }].fetch_add(1, Ordering::Relaxed);
    if conceded > 0 {
        // The KV side just proved it wants more than it holds. Stamping the
        // moment keeps the weight side from reading the ground it has only just
        // handed over as spare on its very next pass.
        //
        // A purchase is the KV side's own voice, and the next negotiation must
        // hear it: `spare_regions` refuses once on `kv_asked_since_negotiation`,
        // which is a statement about *now* — the KV side just ran out — rather
        // than a forecast. That guard is all this needs to record. The demand
        // peak it also used to raise is gone: the weight side measures live
        // occupancy, so a purchase shows up in `live` on its own.
        with_pool(stream, |pool| {
            pool.growth.note_demand();
            Ok(())
        })?;
    }
    Ok(conceded)
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
        Ok(pool.tradeable_bytes().saturating_sub(MIN_ELASTIC_RESERVE))
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
        Ok(pool.tradeable_bytes().saturating_sub(INITIAL_KV_RESERVE))
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
    // while the forward thread sits inside `ffn_residual`, under that layer's FFN
    // wave guard. Callers must treat a refusal as a refusal — see
    // `renegotiate_boundary`, which for a while applied its half of the move and
    // then let this bail propagate as a warning.
    if load_scratch::is_open(ordinal) {
        candle::bail!(
            "refusing to place a weight side while the load's scratch is open at the span's \
             top: the load's transients stand there until `close_load` freezes the dense block \
             (docs/driver_model.md §4)"
        )
    }
    if super::super::bump_arena::wave_is_live(ordinal) {
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
        // The boundary is the truth about what the weight side owns, so the
        // read-only declaration follows it. Ground released to KV stops being
        // immutable the instant it is released — the wave tier stands on it and
        // writes there, entirely legitimately — and a declaration frozen at
        // load reports exactly that as corruption. Diagnostic-only, and a no-op
        // without `tensor-assert`.
        candle::readonly_regions::release_below(floor);
        Ok(total)
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
        // The same base `set_weight_floor` clamps against — the reserve belongs
        // to the elastic middle, which starts at `region_base`. Clamping to a
        // different point than the guard would propose floors the guard refuses.
        Ok(floor.max(pool.region_base + MIN_ELASTIC_RESERVE as u64))
    })
}
