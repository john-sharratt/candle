//! Moving the weight/KV boundary for a streamed dense model.
//!
//! `docs/qwen38_layer_streaming.md` §2.1 is the defect this closes: *"A dense
//! model has no slots, so the boundary has nothing to trade and the partition is
//! inert. KV pressure on a dense model can only be answered by refusing work."*
//! Once a dense model's layers are slot tenants it has exactly what an expert
//! cache has — uniform, droppable ground — and the two directions of
//! `BatchedModelCore` (`request_kv_ground`, `reclaim_spare_ground`) can be
//! answered rather than defaulted to zero.
//!
//! Measured before this existed, on Qwen3.8-27B at 20 contexts:
//!
//! ```text
//! recurrent state: no region for this sequence after 4 claimed — no tier
//! stands, so the span is genuinely full — the weight side is at its own floor
//! and could not concede ground.
//! ```
//!
//! The weight side was holding 6,120 MiB of layer slots and declining to give
//! any of it back, because nothing asked it to.
//!
//! # Between forwards only
//!
//! Both directions hand ground across the boundary, and `set_weight_floor`
//! refuses while a wave generation is open on the span — which is the check that
//! makes this safe rather than an inconvenience. The caller is the wave loop's
//! inter-forward gap, alongside the transient tier's hand-back, for the reason
//! `ExpertCache::reclaim_spare_ground` gives at length: driven from inside a
//! forward it is refused in the common case and, in the narrow window where it
//! is not, costs a device-wide quiesce mid-sweep.
//!
//! # Why this is so much shorter than the expert cache's
//!
//! `ExpertCache::renegotiate_boundary` relocates hot experts out of the ground
//! it is surrendering, so it needs a plan, a device-to-device copy per relocated
//! slot, and a wait before the boundary may move. A layer move relocates
//! nothing, and that is a property of [`super::order`] rather than of anything
//! here: the resident set is a prefix of the protection order at every budget, so
//! a layer that survives the move keeps its **address**. Moving a tenant between
//! addresses without copying its weights would point a live view at another
//! layer's bytes; not having to is what makes this a re-plan and a floor write.
//!
//! # A move is a budget, not a slot count
//!
//! Layers are not the same size, so ground is traded in **bytes**: the caller
//! proposes a budget, [`LayerCache::plan_for`] answers with the layout it buys,
//! and the difference in [`super::zone::ZonePlan::floor`] is what actually
//! changes hands. A byte budget also states the floor honestly — the plan refuses
//! outright below the pinned head plus one streaming cell, which is the point at
//! which the model cannot run rather than merely runs slowly.
//!
//! # What is given up first
//!
//! The head of the eviction order, which is the spread's answer for the new size.
//! So a concession does not merely cost residency, it costs the *least* residency
//! it can: the layers surrendered are the ones whose absence leaves the widest
//! runs of resident layers between them, and therefore the most time to hide the
//! transfers that replace them.

use candle::Result;
use candle_nn::kv_cache::{
    kv_spare_regions, set_weight_floor, span_end, wave_is_live, REGION_BYTES,
};

use super::cache::{LayerCache, SlotAssembler};
use std::sync::atomic::{AtomicU64, Ordering};

/// Why the growth direction answered as it did.
///
/// `[called, whole, no_spare, no_gain, under_floor, applied, wave_live]`
///
/// **Every refusal in `reclaim_spare_ground` is a bare `return Ok(0)`, and only
/// the successes log.** That asymmetry cost three rounds of arithmetic guessing
/// at which branch was firing during a residency collapse — the same shape of
/// blindness `spare_regions`' own tally was added to end, one layer down. The
/// pool's counters describe what the pool offered; these describe what this
/// consumer did with it, and the gap between the two is exactly where a grant
/// goes to die.
static GROWTH_TALLY: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

fn growth_note(idx: usize) {
    GROWTH_TALLY[idx].fetch_add(1, Ordering::Relaxed);
}

/// See [`GROWTH_TALLY`].
pub fn growth_tally() -> [u64; 7] {
    std::array::from_fn(|i| GROWTH_TALLY[i].load(Ordering::Relaxed))
}

/// Regions of headroom left to the KV side when the weight side takes ground.
///
/// The expert cache's `KV_REGION_SLACK`, for the same reason: `spare_regions`
/// answers from what the KV side holds *now*, and taking every region it calls
/// spare leaves the next demand spike nothing to grow into. It is also what
/// covers the one quantity genuinely in the future — persistence's quantize
/// destinations, which have not been claimed when the boundary moves.
const KV_REGION_SLACK: usize = 32;

/// Layers the growth direction must gain before it will move the boundary.
///
/// Hysteresis, and it is about *traffic* rather than about the cost of the move.
/// Growth and retraction are driven by different signals at different moments —
/// spare regions between forwards, a transient tier's shortfall inside one — so
/// a zone sitting near the balance point can take a slot back every forward and
/// give it up again on the next. Each of those retractions drops a layer that
/// must then be re-streamed, which is the one cost this whole subsystem exists
/// to minimise, and the pair nets zero residency.
///
/// Two is the smallest value that makes an oscillation cost more to start than
/// the single layer it would win.
const MIN_GROWTH_LAYERS: usize = 2;

/// Drain every copy the cache has in flight and stop the world.
///
/// A handover in either direction changes who owns an address. Slot transfers
/// run on the cache's own copy stream, which a compute-stream synchronize does
/// not cover, so this synchronizes the **context** — the same thing
/// `ExpertCache::quiesce_before_handover` does and for the same reason.
fn quiesce(device: &candle::CudaDevice) -> Result<()> {
    let stream = device.cuda_stream();
    let ctx = stream.context();
    ctx.bind_to_thread().map_err(candle::Error::wrap)?;
    ctx.synchronize().map_err(candle::Error::wrap)?;
    Ok(())
}

/// Give the KV side `regions` more regions by surrendering layer ground.
///
/// Answers with the bytes actually conceded, which is zero when the zone is
/// already on its floor — the pinned head, which has no copy in any tier and so
/// cannot be reloaded if it is dropped, plus the one cell a streamed layer must
/// land in.
///
/// Refuses **before touching anything**. The expert cache learned this the hard
/// way: its check used to sit after the retraction had already run, so a refusal
/// left the zone believing it was smaller while slots past the new capacity were
/// still live, and the next pass handed a grouped GEMM an address below the
/// weight floor. Here the order is: decide, quiesce, publish, then drop.
pub fn concede_kv_ground<T, A: SlotAssembler<T>>(
    cache: &mut LayerCache<T, A>,
    regions: usize,
) -> Result<u64> {
    if regions == 0 {
        return Ok(0);
    }
    let device = cache.device().clone();
    let stream = device.cuda_stream();
    let end = span_end(&stream)?;
    let held = cache.zone_bytes();
    let want = regions * REGION_BYTES;

    // What the smaller budget buys. `plan_for` refusing *is* the floor check —
    // there is no separate minimum to keep in step with the planner's own rule.
    let budget = held.saturating_sub(want);
    let Ok(plan) = cache.plan_for(end, budget) else {
        tracing::debug!(
            target: "candle_transformers::layer_stream",
            wanted = regions,
            held_mib = held >> 20,
            "layer zone is on its floor and can concede no further ground"
        );
        return Ok(0);
    };
    let before = cache.residency().homed();
    let conceded = held.saturating_sub(plan.used_bytes(end)) as u64;
    if conceded == 0 {
        return Ok(0);
    }

    // **Ask whether the publish can land before paying to make it safe.**
    //
    // `set_weight_floor` refuses while a wave generation is open, and the ground
    // broker's own path reaches here from a KV claim *inside* a forward — so on
    // that path the refusal is certain. The quiesce below is a full
    // `ctx.synchronize()` across both streams: paying it and then being refused
    // drains the device mid-sweep for nothing, once per short claim, which on a
    // wide cohort is a stall per claim in the middle of decode. The latch stays
    // where it is — this is not a substitute for it, and a wave that opens
    // between here and there is still caught there, with nothing moved.
    if wave_is_live(stream.context().ordinal()) {
        tracing::debug!(
            target: "candle_transformers::layer_stream",
            wanted = regions,
            "a wave is open, so the weight floor cannot move; not draining the device for it"
        );
        return Ok(0);
    }

    quiesce(&device)?;

    // **The floor moves first, and the tenants follow it.**
    //
    // `set_weight_floor` is refusable — it declines while a wave generation is
    // open on the span — so anything done before it is done on a coin flip. The
    // other order drops the tenants and then discovers it may not publish, which
    // leaves the cache believing it is smaller while the boundary still says
    // that ground is the weight side's: the layers are gone, the ground is
    // stranded, and the reason is invisible because the caller sees a plain
    // "conceded 0".
    //
    // This is the growth path's order and it is the same lesson
    // `ExpertCache::renegotiate_boundary` writes up at length after a
    // half-applied retraction handed a grouped GEMM an address a slot below the
    // floor. A refusal here lands with nothing moved, and the next pass retries.
    //
    // Publishing first is safe in its own right: the quiesce above means no
    // kernel is reading these slots, the pool stamps ground arriving from the
    // weight side as dirty so a KV claim cleans it before use, and the drop
    // below is the very next statement on this thread.
    let kv_regions = set_weight_floor(&stream, plan.floor)?;
    let dropped = cache.reshape(&plan)?;

    tracing::debug!(
        target: "candle_transformers::layer_stream",
        conceded_mib = conceded >> 20,
        homed = plan.resident(),
        was = before,
        dropped = dropped.len(),
        kv_regions,
        "layer zone conceded ground to the KV side"
    );
    Ok(conceded)
}

/// Take back KV regions that are standing free.
///
/// The growth direction. Concedes nothing, so it answers with zero — the value
/// exists only so the two directions share a signature.
///
/// **The floor moves first and the table follows it.** `set_weight_floor` is
/// refusable, and doing it the other way round leaves a zone one slot wider than
/// the boundary the KV side was told about — the next miss then allocates that
/// top slot and hands a GEMM an address below the floor, which is KV ground.
/// That exact bug is written up on `ExpertCache::renegotiate_boundary`; the
/// order here is the one it arrived at.
pub fn reclaim_spare_ground<T, A: SlotAssembler<T>>(cache: &mut LayerCache<T, A>) -> Result<u64> {
    growth_note(0);
    let before = cache.residency().homed();
    let num_layers = cache.residency().num_layers();
    if cache.residency().is_whole() {
        // Every layer already has a home. More ground would be addresses nothing
        // can ever occupy, so the KV side keeps it.
        growth_note(1);
        return Ok(0);
    }
    let device = cache.device().clone();
    let stream = device.cuda_stream();

    // Sweep first: `spare_regions` reads `live` as the KV side's demand, and an
    // arena that went chunk-empty several waves ago is still live until
    // something sweeps it. Without this the answer comes from a demand figure
    // that includes arenas holding nothing.
    candle_nn::kv_cache::reclaim_empty_arenas();
    // **Ask for a grant this consumer can actually spend.**
    //
    // Ground is taken in whole layers and the test below refuses a gain under
    // `MIN_GROWTH_LAYERS`, so anything less than that many layers' worth of
    // regions is not a small win — it is nothing, discarded without being
    // applied and without the pool learning that it was. `kv_grow_step`'s
    // geometric convergence assumes every grant is taken; a consumer that
    // silently drops its grants accumulates no evidence and is offered the same
    // unusable number on every forward thereafter. Measured on the 27B before
    // this was passed: 396 regions granted over one gate run, 198 applied, the
    // other 3.1 GiB discarded in grants too small to buy one ~154 MiB layer.
    let per_layer = cache.widest_layer_bytes().div_ceil(REGION_BYTES);
    let spare = kv_spare_regions(&stream, KV_REGION_SLACK, MIN_GROWTH_LAYERS * per_layer)?;
    if spare == 0 {
        growth_note(2);
        return Ok(0);
    }
    let end = span_end(&stream)?;
    let plan = cache.plan_for(end, cache.zone_bytes() + spare * REGION_BYTES)?;
    let gained = plan.resident().saturating_sub(before);
    if gained == 0 {
        growth_note(3);
        tracing::debug!(
            target: "candle_transformers::layer_stream",
            spare,
            widest_layer_mib = cache.widest_layer_bytes() >> 20,
            homed = before,
            zone_mib = cache.zone_bytes() >> 20,
            budget_mib = (cache.zone_bytes() + spare * REGION_BYTES) >> 20,
            "spare KV ground bought no layer"
        );
        return Ok(0);
    }
    // The last layers of a model that nearly fits are worth taking however few
    // they are — that case ends the streaming entirely rather than trading one
    // layer back and forth, so `plan.resident() < num_layers` exempts it.
    if gained < MIN_GROWTH_LAYERS && plan.resident() < num_layers {
        growth_note(4);
        return Ok(0);
    }
    // **Ask whether the publish can land before paying to make it safe** — the
    // same pre-check `concede_kv_ground` makes, and for the same reason. This
    // runs from wave phase 0, on the line after `end_wave_transient`, and
    // `wave_is_live` also counts `live_generations > 0` — which covers the tail
    // after a forward returns while its logits still sit on the head span. So a
    // generation from the previous wave is routinely still held here, and
    // `set_weight_floor` below then bails *after* a device-wide drain has been
    // paid, once per forward, in the middle of decode.
    if wave_is_live(stream.context().ordinal()) {
        growth_note(6);
        return Ok(0);
    }
    growth_note(5);

    quiesce(&device)?;
    set_weight_floor(&stream, plan.floor)?;
    // **The layers this brings in are the ones most recently given up**, because
    // the resident set is a prefix of the protection order at every budget. So
    // growth restores the spread the concession cost rather than filling the new
    // ground with whatever happens to fault next — the missing set stays a prefix
    // of the eviction order, and therefore stays as evenly spaced as the new size
    // allows. `LayerResidency::plan_into` then backfills the new homes,
    // nearest-to-the-wave first, so the forward in progress sees the gain.
    cache.reshape(&plan)?;

    tracing::debug!(
        target: "candle_transformers::layer_stream",
        gained,
        homed = plan.resident(),
        was = before,
        spare,
        whole = plan.is_whole(),
        "layer zone took free KV regions"
    );
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::{MIN_GROWTH_LAYERS, REGION_BYTES};

    /// Layer images this model family actually produces, in bytes. The first
    /// three are read off the 27B's own concession log (`conceded_mib=154`
    /// dropping one layer, 173 and 179 likewise); the last two bracket the range
    /// from "barely over a region" to a much larger model's layer.
    const LAYER_BYTES: [usize; 5] = [
        154 << 20,
        173 << 20,
        202 << 20,
        REGION_BYTES + 1,
        1024 << 20,
    ];

    /// **The grant this module asks for must cover the hysteresis it will then
    /// be tested against.**
    ///
    /// `reclaim_spare_ground` requests ground from the pool, then refuses to
    /// apply a gain under `MIN_GROWTH_LAYERS`. If the request can be satisfied
    /// by less than that many layers' worth of regions, the two disagree and
    /// every grant in the gap is computed, discarded, and forgotten — the pool
    /// learns nothing, so the next negotiation produces the same unusable number.
    /// That is the ratchet: retraction has no minimum and moves a layer at a
    /// time, while growth could not reach its own threshold.
    #[test]
    fn the_grant_asked_for_covers_the_hysteresis_it_is_tested_against() {
        for bytes in LAYER_BYTES {
            let per_layer = bytes.div_ceil(REGION_BYTES);
            let min_grant = MIN_GROWTH_LAYERS * per_layer;
            assert!(
                min_grant * REGION_BYTES >= MIN_GROWTH_LAYERS * bytes,
                "a floor of {min_grant} regions cannot buy {MIN_GROWTH_LAYERS} layers of {bytes} B"
            );
        }
    }

    /// The specific number that was wrong: `kv_grow_step`'s old hard floor of
    /// eight regions is the expert cache's allocation unit, and it cannot buy one
    /// layer of any size this family produces.
    #[test]
    fn the_expert_caches_unit_could_not_buy_a_single_layer_here() {
        const OLD_HARD_FLOOR: usize = 8;
        for bytes in LAYER_BYTES {
            if bytes <= REGION_BYTES * OLD_HARD_FLOOR {
                continue; // a layer that small was never the problem
            }
            assert!(
                OLD_HARD_FLOOR * REGION_BYTES < bytes,
                "this case is meant to demonstrate the old floor falling short"
            );
            // ...and the floor this module now passes clears it by construction.
            let min_grant = MIN_GROWTH_LAYERS * bytes.div_ceil(REGION_BYTES);
            assert!(min_grant > OLD_HARD_FLOOR);
        }
    }
}
