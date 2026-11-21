//! The width of the folder pool: how many conversations the card lets it hold
//! open at once, and the ceiling and fallback that bound it.
//!
//! The width is a count, evaluated under the pool's lock against memory that is
//! not KV. The fixed footprint is the expert cache and the dense weights, which a
//! scan can never free, and the per-conversation cost is measured live from the
//! pool's own growth. Holding back the governor's scratch margin, rather than a
//! fraction of capacity, avoids charging the expert cache twice. The weight side
//! may concede only down to its floor, and the scan takes about half of what it
//! could concede, because the floor is a limit and not an optimum. The constant
//! ceiling is a backstop on the derived width, and the blind fallback is kept
//! small so a card that reports nothing is not over-subscribed.

use super::*;

/// Directories ingested concurrently. Each unit is ONE conversation running a
/// short chain (one prefill + one bounded decode), so this is the whole
/// concurrent conversation count for the layer.
///
/// Sized to feed BOTH row-groups. A directory's worker is a three-phase state
/// machine — prefill, prefill, decode — so width buys sequences in each. At 8,
/// 45% of waves ran a prefill with the decode row idle, and the extra width
/// exists to fill those.
///
/// Width also widens prefill itself, which is where most of a directory's wall
/// clock goes. A prefill wave carries as many sequences as are ready, and its
/// throughput scales close to linearly with that count — measured across 981
/// waves on Qwen3-30B-A3B: 188 t/s at one sequence, 357 at two, 457 at three,
/// 699 at four, 1427 at five. A forward carried 10.9x the tokens for 7.2x the
/// time as the batch grew, because the expert weight load amortizes across the
/// batch. Sustaining four or more ready sequences takes roughly twice that many
/// open conversations, since each spends part of its chain decoding.
///
/// **This is a ceiling on [`scan_width`], not the width itself.** The live value
/// is derived from the card; this only stops that derivation running away. It
/// was previously 24, matched to the scheduler's `MAX_PREFILL_WIDTH` on the
/// reasoning that the pool should never ask for more concurrency than a wave can
/// carry — but that ceiling is a *prefill* backstop, and it was sizing the pool
/// for the wrong phase.
///
/// Measured over a full workspace ingest (Qwen3.6-35B-A3B, 201 waves): decode is
/// **61% of the phase's wall time against prefill's 24%**, and decode has no
/// width cap of its own. At a pool of 24 both phases averaged only ~9 wide
/// (max 23) — roughly a third of workers in a forward at any instant, since each
/// unit's chain then ran two prefills before its decode. Prefill is capped at 24 per wave
/// regardless, so workers beyond that queue for prefill but still add to decode
/// width, which is the phase that dominates.
///
/// Decode throughput is width-bound, not depth-bound: banding the same waves by
/// width gives 35 t/s at 1.2 sequences, 124 at 8.7, and 429 at 18.4, while
/// per-forward time FALLS from 338 ms to 43 ms — the MoE expert-weight load
/// amortising across the batch, exactly as the prefill note above describes.
///
/// **96 is a measured optimum, not a headroom guess.** Swept on a 72 GB card
/// (RTX PRO 5000, governor capacity 70.7 GB), whole-workspace ingest, decode
/// rate measured against forward time:
///
/// | ceiling | decode t/s | decode width | ingest phase |
/// |--------:|-----------:|-------------:|-------------:|
/// |      24 |      122.3 |          8.6 |        563 s |
/// |      64 |      240.2 |         32.6 |        453 s |
/// |  **96** |  **288.4** |     **56.2** |    **446 s** |
/// |     128 |      265.2 |         28.4 |        468 s |
///
/// 128 REGRESSES: the achieved width collapses to 28 because the extra workers
/// contend for admission rather than adding concurrency, so more of them sit
/// blocked than decoding. Past the knee, raising this number costs throughput.
///
/// The trade is still VRAM: every admitted conversation pins its K/V, so on a
/// tight card the useful width is bounded by eviction churn — and by the
/// warm-tier drain, which an over-wide ingest can outrun.
///
/// **Caveat worth knowing before changing this.** [`scan_width`]'s per-conversation
/// costing is meant to be the real governor, with this constant only a backstop.
/// On the 72 GB card it has never bound: the pool logged `n_workers == ceiling`
/// at every value swept above (24, 64, 96, 128), so the ceiling — not the
/// memory estimate — is what actually limits width here, and the 128 regression
/// was found by throughput rather than refused by the costing. On a smaller card
/// the estimate does bind and picks the width; on a large one, treat this
/// constant as the live limit.
pub const REPO_MAP_PARALLELISM: usize = 96;

/// Width used when the card can say NOTHING — neither the memory report nor the
/// governor is available, which is the normal state at scan start.
///
/// Deliberately not [`REPO_MAP_PARALLELISM`]: that is a measured ceiling for a
/// 72 GB card, and using it as the blind default would open 96 conversations on
/// a 16 GB one. This is the pre-tuning value, which ran without a single
/// transient-tier failure on every card in the fleet; the runtime gate widens
/// from here once the governor reports.
pub(super) const REPO_MAP_BLIND_PARALLELISM: usize = 24;

/// How many conversations this pool may hold open at once, derived from the
/// card rather than from a thread count.
///
/// Two earlier shapes failed, and both failures are instructive:
///
/// 1. A snapshot test ("is the pool full?") blocks new claims but cannot
///    un-open the conversations already running, so the pool sailed past the
///    mark while 24 chains were in flight.
/// 2. A forward-looking test ("would ONE more fit?") is still evaluated by
///    every worker against the same pre-allocation state. At scan start all 24
///    read a nearly-empty pool, all passed, and all allocated together — a
///    thundering herd that put the whole burst over the wall at once. That is
///    why every configuration produced exactly `n_failed = 24`: one burst, one
///    wall, 24 casualties.
///
/// So the bound is a COUNT, evaluated under a lock, against memory that is not
/// KV: `(limit - fixed_footprint) / per_conversation`. `fixed_footprint` is the
/// expert cache plus dense weights — everything the pool holds that a scan can
/// never free — and `per_conversation` is measured live. Workers then queue on
/// the count instead of racing a gauge that lags them.
pub(super) fn max_live_conversations() -> Option<usize> {
    let (report, age_ms) = candle_conversation::memory_report::latest()?;
    if age_ms > 10_000 {
        return None;
    }
    let kv = kv_reserved(&report);
    let vram = report.vram?;
    let governor = vram.governor?;
    if governor.capacity_bytes == 0 {
        return None;
    }
    Some(scan_width(
        governor.capacity_bytes,
        governor.pool_cushion_bytes,
        vram.pool_used_bytes,
        kv,
        SCAN_KV_BASELINE.load(Ordering::Relaxed),
        SCAN_LIVE_CONVS.load(Ordering::Relaxed),
        // Zero when the model reports no zone to defend — a dense checkpoint or
        // a CPU device, where the weight side has no elastic boundary and
        // nothing is taken from it by a K/V claim.
        report.weights.floor_bytes.unwrap_or(0),
        weights_now(&report.weights),
    ))
}

/// The count bound, as arithmetic over the report's figures alone.
///
/// `(capacity - scratch_margin - fixed - baseline) / per_conversation`, where
/// `fixed` is what the pool holds that no scan decision can release — the expert
/// cache plus dense weights — `baseline` is the inherited pre-scan KV corpus
/// (see [`SCAN_KV_BASELINE`]), and `per_conversation` is measured from the
/// pool's own growth.
///
/// Numerator and denominator must price the same bytes. `fixed` is
/// `pool_used - kv`, so subtracting it alone hands *every* KV byte back as room
/// for new conversations — including the inherited corpus, which is resident,
/// is not the scan's to spend, and is precisely what `per_conversation_kv`
/// excludes on the other side of the divide. On the measured report below that
/// counted ~2 GiB of standing arenas as free space and opened ten directories
/// against room for six.
///
/// The thing that costs VRAM is an *open conversation*: it pins its K/V until
/// its chain completes, and while it is live its K sits in `R16` (4 bytes per
/// element) and its V in F16, so a directory costs several times what the same
/// turns cost once sealed and quantized. Admission cannot substitute for this
/// bound — it throttles prefill *submission*, and by then the conversation
/// exists and its K/V is already pinned.
///
/// The margin held back is the governor's OWN `scratch_margin`, not a fraction
/// of capacity. A fraction double-charges the expert cache: `fixed` subtracts it
/// explicitly, then the fraction holds back a share of capacity that is mostly
/// the same bytes again. Measured on the 16 GiB card, a 0.70 fraction left the
/// gate 1.87 GiB for scan KV while the governor had ~5 GiB floored for exactly
/// that — under half the room, and width is what the whole pool is for: up to
/// the point the card sustains, a wave's throughput scales with the sequences it
/// carries, because the expert load amortizes across them (prefill 188 t/s at
/// one sequence against 699 at four; decode 3.4 against 15.9 at nine). Holding
/// back the scratch margin instead spends the room the governor already reserved
/// for KV — while [`SCAN_CONV_KV_MIN`] keeps the resulting width on the safe
/// side of that point.
#[allow(clippy::too_many_arguments)]
pub(super) fn scan_width(
    capacity: u64,
    scratch_margin: u64,
    pool_used: u64,
    kv: u64,
    baseline: u64,
    live: usize,
    weight_floor: u64,
    weights_now: u64,
) -> usize {
    let fixed = pool_used.saturating_sub(kv);
    // **The scan may take what the weight zone can concede, and not a byte
    // more.**
    //
    // `capacity` is the governor's balloon-measured resident capacity — the
    // whole elastic span, not a K/V-only share — and the weight zone lives in
    // that same span. Ground K/V claims is ground the weight side loses, so a
    // width planned from capacity alone plans to evict the model.
    //
    // It did. At 96 directory conversations the pool's K/V reached 48 GB, the
    // weight zone fell from 53.5 GB to 10.9 GB — 19 GB below its floor — and
    // admission refused every prefill for the rest of the run, which is both
    // correct and unrecoverable: the floor check compares residency against the
    // floor, and no number of refusals hands K/V back.
    //
    // The bound is `weights_now - floor`, not `capacity - floor`. Subtracting
    // the floor from capacity charges the scan for ground the weights are
    // already standing on, which it can never take anyway; measured, that cut
    // the pool from 96 to 7 on a card with 21 GB genuinely available and left
    // admission idle at `queued=0` while decode ran at the pool's width.
    // **Half of what it could concede, because the floor is a limit and not an
    // optimum.**
    //
    // Conceding weight ground is permitted down to the floor and is not free on
    // the way there: a smaller expert zone pages more, and the paging costs the
    // forwards the extra conversations were opened to fill. Measured on the
    // 72 GB card, over the same directory corpus:
    //
    //   pool 7   zone stable at 50.2 GB   12.2 units/min
    //   pool 53  zone pinned at the floor  ~9.0 units/min   decode width 50
    //
    // The wide pool got everything it asked for — decode ran 50 sequences to the
    // narrow pool's 4-7 — and still finished less work, because the zone spent
    // the run at 28 GB. Taking half leaves the expert cache its working set and
    // still opens several times the conversations a capacity-minus-floor bound
    // allowed.
    let concedeable = weights_now.saturating_sub(weight_floor) / 2;
    let for_kv = capacity
        .saturating_sub(scratch_margin)
        .saturating_sub(fixed)
        .saturating_sub(baseline)
        .min(concedeable);
    let per_conv = per_conversation_kv(kv, baseline, live);
    ((for_kv / per_conv.max(1)) as usize).clamp(1, REPO_MAP_PARALLELISM)
}

/// Pool width from the VRAM governor, for the moment BEFORE any memory report
/// exists.
///
/// The scan starts before the scheduler has published its first report — the
/// walk runs between calibration and the pool — so [`max_live_conversations`]
/// has nothing to read and every worker would fall back to the full constant.
/// The governor is live from model load, and its headroom at this instant is
/// exactly the VRAM left after weights and the expert cache: the room a scan's
/// conversations have to share.
///
/// A slice is held back for wave activations (measured at ~1 GiB for a wide
/// prefill, and the wall is only reached when a wave lands on a high base), and
/// the rest is divided by the per-conversation KV estimate.
pub(super) fn scan_width_from_governor() -> Option<usize> {
    let gov = candle::vram::get(0)?;
    let headroom = gov.measure().ok()?.headroom;
    let for_kv = headroom.saturating_sub(SCAN_ACTIVATION_RESERVE);
    Some(((for_kv / SCAN_CONV_KV_MIN) as usize).clamp(1, REPO_MAP_PARALLELISM))
}

/// Held back from the scan's share for a wave's transient activations. Measured
/// at 666-1005 MiB across prefill waves on Qwen3-30B-A3B; the failures all
/// occurred when a wave of that size landed on an already-high base.
pub(super) const SCAN_ACTIVATION_RESERVE: u64 = 1536 * 1024 * 1024;
