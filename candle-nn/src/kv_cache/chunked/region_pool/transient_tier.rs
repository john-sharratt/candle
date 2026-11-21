//! Places the wave transient tier for a forward and releases it when the forward ends.
//!
//! A tier is placed directly above the live arenas and below the weight floor,
//! with its base at the arena frontier, so the ground between it and the floor
//! stays claimable. If the tier does not fit, the shortfall is bought from the
//! weight side and the placement retried, for as long as purchases keep landing.
//! Empty arenas are swept before the partition is declared full. On release the
//! ground returns to the KV side marked dirty, so the next claim cleans it before
//! reuse. The widths of recent forwards are remembered, so the growth
//! negotiation keeps the widest of them free.

use super::*;

/// Forwards whose tiers the growth negotiation remembers, leaving the widest free.
///
/// **A step, not a forward.** A speculative decode step is several forwards of
/// unlike width — draft forwards a few regions wide, then the verify of every
/// block, then the next step's drafts — so the last forward's tier is a draft's
/// as often as not. Predicting the next forward from it handed the verify's ground
/// to the weight side between two drafts, and the verify's placement then had to
/// buy it back mid-step: on Flash-Next at ×8, once the span was packed after the
/// prefill, a 912 MiB verify tier refused 256 MiB short with the weight zone 84
/// regions fuller. Sixteen covers a step at the widest draft ladder with room for
/// a segmented sweep's forwards; a prefill's chunks are each wider than the last,
/// so the window's widest is still the next chunk's.
pub(super) const RECENT_FORWARDS: usize = 16;

/// **Place the wave transient tier for the forward about to run.**
///
/// The tier does not exist between forwards. This brings it into being with
/// `bytes` of extent, packed against the arena frontier, and returns its base.
/// The region ceiling drops to that base for the duration, so no arena can be
/// stamped into the ground the tier now occupies.
///
/// # Anchored at the arena frontier `A`
///
/// `docs/archived/elastic_vram_partition.md` §7 phase 2. This leaves the whole remainder
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
///
/// # More than one purchase, while purchases keep landing
///
/// A purchase runs outside the pool lock and blocks on the expert pipeline — a
/// quiesce, the relocations, the floor move — and arena creation is not held off
/// across it, so a region the persistence thread claims in that window raises
/// `live_end` into the ground just bought. The retry then measured itself short
/// again and the wave failed: measured on the Qwen3-30B engine probe, a purchase
/// conceded 41 slots (~113 MiB) and the tier was still one region into live
/// arenas, in 4 of 12 runs, with an error blaming a weight floor that had just
/// moved. So the placement buys again for whatever it is still short, for as long
/// as the weight side keeps conceding.
pub(crate) fn place_transient(stream: &std::sync::Arc<CudaStream>, bytes: usize) -> Result<u64> {
    /// Purchases one placement may make. Each is a full boundary move on the
    /// pipeline thread, so this bounds the wait, not the ground.
    const MAX_PURCHASES: usize = 4;
    let mut short = match try_place(stream, bytes)? {
        Placed::At(base) => return Ok(base),
        // Not "the KV side is short of regions" — the tier itself does not fit,
        // and the regions in its way are live. A purchase for exactly the
        // shortfall the placement measured.
        Placed::Short(regions) => regions,
    };
    let mut conceded = 0u64;
    for _ in 0..MAX_PURCHASES {
        let got = buy_ground(stream, short)?;
        conceded += got;
        match try_place(stream, bytes)? {
            Placed::At(base) => return Ok(base),
            Placed::Short(regions) => short = regions,
        }
        if got == 0 {
            break;
        }
    }
    // **Sweep the empties before declaring the partition dead.** The footprint
    // check above counts *claimed* regions, and a region whose arena went
    // chunk-empty since the last periodic sweep still reads as claimed — on a
    // churn-heavy config (seal/requantize under repeats) hundreds of such
    // arenas can stand in the tier's ground while holding nothing. Releasing
    // them is a free-list push per arena, after which the retry re-measures
    // against only the genuinely live arenas.
    super::super::backing::global_release_empty_arenas();
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
            let short_bytes = still_short * REGION_BYTES;
            candle::bail!(
                "{TIER_REFUSED_MARKER}: {TIER_SHORT_BY}{short_bytes} B — its transient \
                 tier needs {len} B below the weight floor and is {still_short} regions \
                 into ground live KV arenas hold, which cannot move. The weight side conceded {conceded} B to this \
                 placement and would give no more — at its own floor, or refusing to \
                 move while a wave or an expert invocation is in flight. (span \
                 {span_bytes} B, weight floor at \
                 +{floor_off} B, arena frontier at +{live_off} B, {live}/{total} \
                 regions live)"
            )
        }
    }
}

/// The outcome of one attempt at [`place_transient`].
pub(super) enum Placed {
    /// The tier stands at this address.
    At(u64),
    /// Live arenas reach this many regions into its footprint.
    Short(usize),
}

pub(super) fn try_place(stream: &std::sync::Arc<CudaStream>, bytes: usize) -> Result<Placed> {
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
                .and_then(|()| unsafe { memset_d8_async(base, 0xCD, len, stream.cu_stream()) })
                .map_err(candle::Error::wrap)?;
        }
        pool.transient_base = Some(base);
        pool.transient_bytes = len;
        // Survives the release, so the between-forwards demand reading still
        // knows a tier of this size is about to want its ground back.
        pool.transient_high_water = pool.transient_high_water.max(len);
        pool.tier_peak = pool.tier_peak.max(len);
        pool.recent_tiers[pool.recent_at] = len;
        pool.recent_at = (pool.recent_at + 1) % RECENT_FORWARDS;
        if pool.prompts_waiting {
            pool.waiting_tier_bytes = pool.waiting_tier_bytes.max(len);
        }
        pool.tiers_placed += 1;
        Ok(Placed::At(base))
    })
}

/// Whether to fill the transient tier with a poison byte as it is placed.
///
/// `KV_TIER_POISON=1`. Off by default: it costs a synchronous device memset of
/// the whole tier on every forward.
pub(super) fn tier_poison_enabled() -> bool {
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
            // Written and read by the forwards alone, on this stream.
            pool.mark_dirty_span(base, pool.transient_bytes, stream_id(stream));
        }
        pool.transient_base = None;
        pool.transient_bytes = 0;
        Ok(())
    });
}

/// Forget the tiers the recent forwards placed, so the growth negotiation stops
/// keeping their ground free for a forward that is not coming.
///
/// The negotiation leaves the recent forwards' widest tier free because the next
/// forward is one of them again: the next chunk of a prefill, a little deeper, or
/// the next verify of a decode. With nothing in flight that reasoning has nothing
/// to apply to: the work is over, and a turn whose last forward was a 128K prefill
/// chunk — a section ingest, which decodes nothing — would otherwise hold the
/// weight side back by that whole tier for as long as the engine sat idle. The
/// next forward prices and places its own tier, buying back what it needs.
///
/// A placed tier is left alone: a caller that is wrong about being outside a
/// forward changes nothing.
pub fn forget_recent_tiers(stream: &std::sync::Arc<CudaStream>) {
    let _ = with_pool(stream, |pool| {
        if pool.transient_base.is_none() {
            pool.recent_tiers = [0; RECENT_FORWARDS];
        }
        Ok(())
    });
}
