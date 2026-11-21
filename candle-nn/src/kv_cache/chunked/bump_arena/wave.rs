//! Opens a generation on a layer phase's span, placing the tier if it is missing.
//!
//! The guard holds its phase's span for the whole phase, and the borrow checker
//! keeps every intermediate allocated from it within that guard. Opening it
//! first settles the tier: a plan the standing tier cannot hold is re-placed,
//! and a plan whose spans a live generation still holds is refused, since
//! laying out a new plan over arenas sized for another forward would run past
//! them. Every span is checked against the weight floor before it is laid out,
//! so none can reach into the expert zone.

use super::*;

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
#[track_caller]
pub fn begin_wave(stream: &Arc<CudaStream>, phase: LayerPhase) -> Result<Generation> {
    let opener = std::panic::Location::caller();
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
    // Two callers reach it. One is the unplanned guard — opened with no forward
    // behind it (a test, a migration helper, a `rows == 0` forward that skipped
    // phase 2) — which owns its own reservation, taken here.
    //
    // **The other is a plan that has outgrown the tier standing under it**, and
    // that one is a silent-corruption path, not a convenience.
    // [`plan_wave_transient`] records `planned` *before* it checks whether it may
    // place, so a forward that arrives while the previous one's outputs are still
    // held updates the plan and leaves the old, narrower tier where it is. If the
    // generations then drop before this guard opens, the rebase below hands the
    // arenas capacities from the new plan on top of the old tier's base — and
    // nothing downstream re-checks, so the spans simply run off the tier's top
    // into the weight zone and write over expert slots.
    //
    // So the tier is re-placed whenever it cannot hold the plan. Re-placing is
    // safe here for the same reason the rebase is: no generation is live.
    //
    // **And the decision is taken again if it went stale.** It is made under
    // one lock and acted on under another, with the placement between them, so
    // a generation live at the decision — the previous forward's logits rows,
    // still held — can drop before the layout. The layout then finds no
    // generation, no tier, and no placement of its own: a plan with no ground.
    // Measured in zend's ingest as draft walks refused and their files failed,
    // a few times a minute. That state is the decision being out of date, not a
    // fault, so it goes round again; only if it keeps recurring is it refused.
    const MAX_REDECIDE: usize = 4;
    for _ in 0..MAX_REDECIDE {
        let replan = {
            // Same reason as [`plan_wave_transient`]: a placement reads the arena
            // frontier, so no arena may be half-created when it does.
            let mut map = await_arena_windows(lock_domains(), &stream);
            let (_, domain) = domain_entry(&mut map, &stream);
            if domain.live_generations > 0 {
                None
            } else {
                let plan = domain.planned.unwrap_or_else(fallback_plan);
                let wanted = tier_footprint(plan);
                // A tier whose top already crosses the weight floor must be
                // re-placed however large it is, so the `bytes >= wanted` arm below
                // cannot keep it. `plan_wave_transient` leaves exactly this state
                // behind when a forward arrives while the previous one is still
                // live: the wider plan is recorded, the narrower tier stays put, and
                // the two are only reconciled here.
                let crosses_floor = domain.placed_at.is_some_and(|b| {
                    super::super::region_pool::span_layout(stream.context().ordinal())
                        .is_some_and(|l| b.saturating_add(wanted as u64) > l.weight_floor)
                });
                // **The reservation this domain holds, not what the pool happens to
                // be occupying.**
                //
                // This read `region_stats(..).transient_bytes`, which is a different
                // quantity in two ways that both matter: it is *live occupancy* —
                // its own docs call it "a snapshot taken mid-wave, not a
                // reservation" — and it is a property of the POOL rather than of
                // this domain's tier. So it could report a figure at least as large
                // as the plan while this domain's own tier was narrower than the
                // plan; the replan was then skipped, and the layout below wrote the
                // new plan's widths onto the old tier's base and ran off its top
                // into the expert weight zone. The comment above this block
                // describes that failure exactly — the check simply was not asking
                // the question the comment assumed.
                let standing = domain.placed_at.and(domain.placed_bytes);
                match standing {
                    Some(bytes) if bytes >= wanted && !crosses_floor => None,
                    _ => Some(plan),
                }
            }
        };
        wave_gate().notify_all();
        let unplanned_base = match replan {
            Some(plan) => {
                // Hand the old one back first when there is one, so its ground is
                // marked dirty and the next tenant cleans it.
                release_transient(&stream);
                let bought = tier_footprint(plan);
                Some((place_transient(&stream, bought)?, bought))
            }
            None => None,
        };
        let opened = with_wave_domain(&stream, |domain| {
            let arena = &domain.arenas[phase_index(phase)];
            if arena.is_live() {
                candle::bail!(
                    "wave domain: the {phase:?} span already has a live generation. A \
                 layer leaves one phase before entering the next, so this is a \
                 phase re-entered while still open — the inner guard's drop would \
                 reset the span the outer one is still handing out."
                )
            }
            // **The two layer phases share one region** (`phase_offsets`), so one
            // may not open while the other is live: its first carve would land on
            // the other's buffers. A layer runs them in sequence, so this is a
            // phase entered out of turn — refuse rather than corrupt (principle 7).
            if let Some(other) = other_layer_phase(phase) {
                if domain.arenas[phase_index(other)].is_live() {
                    candle::bail!(
                        "wave domain: the {phase:?} span shares its ground with the \
                         {other:?} span, which still has a live generation. A layer \
                         leaves one phase before entering the next; opening this one \
                         would carve over the other's buffers."
                    )
                }
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
                // **A re-placement supersedes the recorded base.** The replan path
                // above released the standing tier and placed a new one at the
                // CURRENT frontier, but `placed_at` still holds where the OLD tier
                // stood — using it would lay the spans at the old base with the new
                // plan's widths, writing wave intermediates over whatever now owns
                // the ground between the two addresses (live KV chunks when the
                // frontier moved up, expert slots when the plan overruns the old
                // extent). Silent corruption both ways, found independently by
                // three reviewers. The freshly returned address is authoritative
                // whenever it exists.
                if let Some((base, bought)) = unplanned_base {
                    domain.placed_at = Some(base);
                    domain.placed_bytes = Some(bought);
                    // A replan belongs to the forward whose plan outgrew the tier;
                    // an unplanned guard's placement is its own. Preserve whichever
                    // ownership was already recorded when a tier stood, and default
                    // a fresh placement to guard-owned exactly as before.
                    if domain.planned.is_none() {
                        domain.reserved_by_forward = false;
                    }
                }
                let base = match domain.placed_at {
                    Some(base) => base,
                    // No tier recorded and none was just placed: the decision above
                    // was taken while a generation was live, and it has dropped
                    // since. Decide again — see the loop.
                    None => return Ok(None),
                };
                // **Refuse a layout that does not fit the ground it was sold.**
                //
                // The replan above is what keeps these in step, so reaching this
                // with a plan too large means that logic failed. Laying it out
                // anyway is the corruption: the last span runs off the tier's top
                // into whatever is above it, which on this span is the expert weight
                // zone. Refusing costs one wave; the alternative cost a model.
                let need = tier_footprint(plan);
                match domain.placed_bytes {
                    Some(bytes) if need > bytes => {
                        return Err(candle::Error::Msg(format!(
                            "wave domain: the plan needs {need} B but the tier standing at \
                         {base:#x} bought only {bytes} B. Laying the spans out would run \
                         {} B past its top — into the expert weight zone.",
                            need - bytes,
                        )));
                    }
                    _ => {}
                }
                // **The floor is the invariant; the tier is only a proxy for it.**
                //
                // Comparing the plan against the bytes the tier bought cannot catch
                // the case this guard exists for, because both figures are derived
                // from the same `planned` array — they agree by construction and the
                // comparison always passes. What actually went wrong is one step
                // further out: `plan_wave_transient` records a forward's plan even
                // when it cannot place a tier for it (`live_generations > 0`), so a
                // wider forward arriving behind a live one leaves its plan standing
                // over the previous, narrower tier. The layout below then walks
                // `plan` widths from a base chosen for a smaller purchase, and the
                // last span runs past the tier's top into the expert weight zone.
                //
                // So the layout is checked against the thing it must never cross,
                // read live rather than inferred. `weight_floor` moves only through
                // `set_weight_floor`, which refuses to cut live KV, so it is a
                // sound bound at any instant.
                if let Some(l) = super::super::region_pool::span_layout(stream.context().ordinal())
                {
                    let top = base.saturating_add(need as u64);
                    if top > l.weight_floor {
                        return Err(candle::Error::Msg(format!(
                            "wave domain: laying {need} B of spans from {base:#x} reaches \
                         {top:#x}, past the weight floor {:#x} — that is expert-slot \
                         ground, and the FFN span would write activations over resident \
                         weights. Refusing the wave; the plan needs a tier of its own.",
                            l.weight_floor,
                        )));
                    }
                }
                // The two layer phases at the base, sharing it — never live at
                // once, which the check below enforces — and the forward span past
                // the larger of them (`phase_offsets`, the layout `tier_footprint`
                // priced).
                let offsets = phase_offsets(plan);
                for (i, arena) in domain.arenas.iter().enumerate() {
                    arena.rebase(base + offsets[i] as u64, plan[i])?;
                }
                domain.plan_pending = false;
            } else if let Some(plan) = domain.planned {
                // **A generation is live, so the spans stand as an earlier plan
                // laid them out — and they must hold this one.** Inside one forward
                // they always do: its phases open on the layout its own plan made.
                // A forward opening while an earlier one's returned span is still
                // held does not get a layout of its own, and when its plan has
                // outgrown the spans standing, handing them out runs it on arenas
                // sized for someone else: its first wide buffer asks past them. On
                // BlackWorld (c673b63c9, 2026-10-19) a step priced at 340 MB of
                // forward phase opened on a 16 MiB span a decode wave had laid out,
                // and the creep segment's 320 MiB residual failed the fourteen
                // turns its wave carried. Refused here instead, before anything
                // runs: the generation drops, and the step placed again lays out
                // its own spans.
                let laid = arena.stats().2;
                let wants = plan[phase_index(phase)];
                if wants > laid {
                    // Named: whether the holder is the failing step's own stack —
                    // gone when the step returns, so its retry finds the spans free —
                    // or something kept between steps, which a retry would only meet
                    // again.
                    let holder = domain
                        .forward_opener
                        .map_or_else(|| "an unknown site".to_string(), |l| l.to_string());
                    return Err(candle::Error::Msg(format!(
                        "{SPANS_OUTGROWN_MARKER}: the {phase:?} span is laid out at {laid} B \
                         for an earlier plan, and this forward's plan asks {wants} B of it; \
                         a generation still live holds that layout — the forward span \
                         opened at {holder}"
                    )));
                }
            }
            // Count only once the guard exists. The count is what `release_if_last`
            // decrements on drop, so incrementing before a fallible call would strand
            // the tier permanently if that call ever failed — there would be no guard
            // to drop and nothing to bring the count back down.
            let generation = arena.generation(ordinal, arena_idx)?;
            domain.live_generations += 1;
            // The forward-wide span opening is the forward's scratch scope
            // opening: its launchers carve from this span until it closes.
            if arena_idx == FORWARD_ARENA {
                domain.forward_opener = Some(opener);
                let budget = if domain.planned.is_some() {
                    domain.scratch_priced
                } else {
                    0
                };
                open_forward_scratch(ordinal as usize, Arc::clone(&arena.inner), budget);
            }
            Ok(Some(generation))
        })?;
        if let Some(generation) = opened {
            return Ok(generation);
        }
    }
    // Laying these spans out now would put them on ground priced for a
    // different occupant. Refuse rather than corrupt (principle 7).
    Err(candle::Error::Msg(format!(
        "wave domain: the tier's ground was priced with no generation live, and \
         {MAX_REDECIDE} times running a generation had come or gone before it could \
         be laid out. The spans this guard would hand out are not the ones that were \
         reserved."
    )))
}

/// Release the tier if `ordinal`'s last wave generation has just dropped.
///
/// Called from [`Generation::drop`]. Reference-counted rather than scoped
/// because a forward's head span is *returned to its caller* — the tier's
/// lifetime is the union of the guards, not any one function's body.
pub(super) fn release_if_last(ordinal: usize) {
    release_if_last_locked(ordinal);
    // A caller waiting in `enter_arena_window` is waiting on exactly this: the
    // last generation dropping. Signalled outside the lock so it wakes into a
    // free mutex.
    wave_gate().notify_all();
}

pub(super) fn release_if_last_locked(ordinal: usize) {
    let mut map = lock_domains();
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
        domain.placed_bytes = None;
        release_transient(&stream_of(domain));
    }
}

/// The stream a wave domain was built against.
///
/// Every arena in a domain is created from the same stream, so any of them
/// answers — arena 0 by convention.
pub(super) fn stream_of(domain: &WaveDomain) -> Arc<CudaStream> {
    domain.arenas[0].inner.lock().unwrap().stream.clone()
}

/// Carve `bytes` from the arena a [`WaveTicket`] names — the resolver candle-core
/// calls when an op inherits its operand's arena.
///
/// Three answers, and only one of them goes to the pool. An unknown domain (no
/// wave has run on that stream) or a stale epoch (the generation rewound) is
/// [`WaveCarve::Closed`]: the operand's phase is over, and its derived data has
/// nowhere else to live. An open generation whose span cannot hold the request
/// is [`WaveCarve::Exhausted`], which the caller turns into an error — a buffer
/// the wave plan did not price, named where it was asked for.
pub(super) fn resolve_wave_alloc(ticket: WaveTicket, bytes: usize, align: usize) -> WaveCarve {
    // A refusal says where the span stood, read under the same guard as the
    // carve it refused.
    let carve = |guard: &mut Inner, name: &'static str| match bump_locked(guard, name, bytes, align)
    {
        Some(ptr) => WaveCarve::Carved(ptr),
        None => WaveCarve::Exhausted(SpanState {
            cursor: guard.cursor,
            capacity: guard.capacity,
        }),
    };
    // A guest ticket names no phase, so it is answered from the guest registry
    // rather than the wave domain's three arenas. Checked first because it is
    // the cheaper lookup and because `arenas.get(3)` would otherwise decide it
    // by returning `None` — the right answer for the wrong reason, and one that
    // would silently stop working if a fourth phase were ever added.
    if ticket.arena == ASIDE_ARENA {
        return super::super::aside::carve(ticket, bytes, align);
    }
    if ticket.arena == GUEST_ARENA {
        // A guest that opened a heap carves there: it frees as tensors die,
        // which a graph without stages needs (`guest_heap`).
        if let Some(carved) = super::super::guest_heap::carve(ticket, bytes, align) {
            return carved;
        }
        let Ok(map) = guest_domains().lock() else {
            return WaveCarve::Closed;
        };
        let Some(arena) = map.get(&(ticket.domain as usize)) else {
            return WaveCarve::Closed;
        };
        let inner = Arc::clone(&arena.inner);
        let name = arena.name;
        drop(map);
        {
            // Decided and carved under one guard. Dropping it between the two
            // would let the generation close in the window and hand this carve a
            // range the arena has already rewound past.
            let Ok(mut guard) = inner.lock() else {
                return WaveCarve::Closed;
            };
            // **A guest's weights carry a seed, not a provenance.** They are
            // placed once and read by every stage — the encode, each denoise
            // step, each decode tile — and each stage is its own generation, so
            // a real epoch would route the first stage and nothing after it.
            // [`GUEST_ANY_EPOCH`] means "whichever generation is open", and
            // `live == 0` still refuses when none is: the seed never names a
            // range, so it cannot alias one.
            //
            // Tensors carved *inside* a stage do not use that seed — `stage_in`
            // stamps the open generation's real epoch, so the chain descending
            // from a stage stops resolving the moment that stage's arena
            // rewinds, and only the weights keep the any-epoch licence.
            let epoch_ok = ticket.epoch == GUEST_ANY_EPOCH || guard.epoch == ticket.epoch;
            if !epoch_ok || guard.live == 0 {
                return WaveCarve::Closed;
            }
            return carve(&mut guard, name);
        }
    }
    let Ok(map) = wave_domains().lock() else {
        return WaveCarve::Closed;
    };
    let Some(arena) = map
        .get(&(ticket.domain as usize))
        .and_then(|d| d.arenas.get(ticket.arena as usize))
    else {
        return WaveCarve::Closed;
    };
    let inner = Arc::clone(&arena.inner);
    let name = arena.name;
    // Drop the registry lock before touching the arena: the two are always taken
    // in this order and never the reverse, which is what keeps this free of the
    // deadlock an allocation path called from arbitrary op code could otherwise
    // introduce.
    drop(map);
    // Decided and carved under one guard, as the guest branch is: released
    // between the two, the generation could close in the window and the carve
    // land in a range the arena has already rewound past.
    let Ok(mut guard) = inner.lock() else {
        return WaveCarve::Closed;
    };
    if guard.epoch != ticket.epoch || guard.live == 0 {
        return WaveCarve::Closed;
    }
    carve(&mut guard, name)
}

/// Whether any wave generation is open on `ordinal`.
///
/// **The one thing the moving boundary must never do is move mid-wave.** A
/// retraction evicts experts and relocates others, and a wave in flight may be
/// reading either; the design's answer is that the boundary moves only between
/// forwards — the wave loop's own gap, and the arena claim that has run the KV
/// side out before a wave has begun. That is a structural property of where
/// `renegotiate_boundary` is called from, and this is what lets
/// `set_weight_floor` check it rather than
/// trust it (principle 7: refuse rather than corrupt).
pub fn wave_is_live(ordinal: usize) -> bool {
    // One definition of "a wave is running", shared by the arena gate and the
    // boundary latch. Arena liveness alone is FALSE at every phase boundary —
    // a layer's attention guard drops before its FFN guard opens — which is
    // the §13b hole `forward_open` was introduced to close for the arena gate;
    // reading only the arenas here left `set_weight_floor`'s latch with that
    // same hole, so a boundary move reached from off the wave thread (a fork's
    // buy_ground, the scheduler's relief) could retract the zone mid-forward
    // in the gap and evict expert slots the next phase names via `slot_base`.
    // `live_generations` covers the tail after the forward returns, while the
    // logits still sit on the head span.
    wave_domains()
        .lock()
        .map(|map| {
            map.get(&ordinal).is_some_and(|d| {
                d.forward_open || d.live_generations > 0 || d.arenas.iter().any(|a| a.is_live())
            })
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
