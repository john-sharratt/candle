//! The estimate of what one more open folder conversation costs the pool's KV
//! arenas, which the width is sized from.
//!
//! The estimate prices only the KV the pool itself adds: the arenas standing
//! before the pass opened, which a scan cannot free by finishing a directory, are
//! an inherited baseline and are anchored once per pass. The estimate is measured
//! and clamped to a floor and a ceiling, so a cold start cannot admit without
//! limit and one atypical directory cannot stall the scan. It charges reserved
//! arena rather than the live half, because reserved is what the allocator's
//! ceiling counts and the unfilled slabs are a real cost of each open
//! conversation. On the weight side the floor is measured against the expert zone
//! and not against base plus zone, because the dense base lives outside the span
//! no concession can release.

use super::*;

/// What the weight side is holding right now, in the units the floor is
/// expressed in.
///
/// **A zone figure, not base + zone.** `floor_bytes` is
/// `interleave::optimal_weight_bytes` — a fraction of the achievable residency
/// of the *elastic expert zone*, measured inside a span that is the KV regions
/// plus that zone and nothing else. Dense base weights are allocated outside
/// that span and no concession can ever release them, so counting them here
/// reports ground the scan cannot have: `(base + zone) − zone_floor`, halved by
/// the caller, overstates the concession by `base_bytes / 2`. It read correct on
/// this card only because a wave stack reports no separate base at all.
///
/// With no floor to defend the answer is UNBOUNDED rather than zero: a stack
/// that reports no zone has none for K/V to take from, so capacity alone is the
/// bound and the caller's `min` must not clamp it to nothing.
pub(super) fn weights_now(weights: &candle_conversation::memory_report::WeightSection) -> u64 {
    match weights.floor_bytes {
        Some(_) => weights.resident_expert_bytes.unwrap_or(0),
        None => u64::MAX,
    }
}

/// Total reserved KV arena bytes in a memory report — the quantity the arena
/// allocator's ceiling actually counts, summed over every size class.
pub(super) fn kv_reserved(report: &MemoryReport) -> u64 {
    report
        .kv
        .classes
        .iter()
        .fold(0u64, |acc, c| acc.saturating_add(c.reserved_bytes))
}

/// What one more open directory conversation is expected to cost the pool.
///
/// Measured rather than assumed, and measured against the pool's OWN growth:
/// the estimate prices `kv - baseline`, never the whole process's arenas (see
/// [`SCAN_KV_BASELINE`]). Clamped so a cold start cannot admit unboundedly and
/// one atypical directory cannot stall the scan.
pub(super) fn per_conversation_kv(kv: u64, baseline: u64, live: usize) -> u64 {
    let grown = kv.saturating_sub(baseline);
    if live > 0 && grown > 0 {
        (grown / live as u64).clamp(SCAN_CONV_KV_MIN, SCAN_CONV_KV_MAX)
    } else {
        SCAN_CONV_KV_MIN
    }
}

/// Reserved KV present when the pool opened — the corpus the scan did not
/// create, and cannot free by finishing a directory.
///
/// The tool sections, the base builder's prefill and the calibration exemplars
/// all live in the same arenas the report totals, and together they run to
/// gigabytes before a single directory is opened. Charging that to the handful
/// of conversations in flight pins [`per_conversation_kv`] to
/// [`SCAN_CONV_KV_MAX`] and throttles the pool to a width of two on a card with
/// room for more — and the error compounds in the wrong direction, since fewer
/// live conversations divide the same fixed corpus into a larger per-conversation
/// estimate. Pricing the delta instead leaves exactly the part a scan decision
/// influences: a completed directory demotes to the warm tier and gives its
/// arenas back, so the delta tracks what is genuinely in flight.
pub(super) static SCAN_KV_BASELINE: AtomicU64 = AtomicU64::new(0);

/// Re-anchor [`SCAN_KV_BASELINE`] on the arenas already in place, so the pool
/// about to open prices only the KV it goes on to create. Called once per pass,
/// before any worker claims a directory.
pub(super) fn anchor_scan_kv_baseline() {
    let baseline = candle_conversation::memory_report::latest()
        .map(|(report, _age_ms)| kv_reserved(&report))
        .unwrap_or(0);
    SCAN_KV_BASELINE.store(baseline, Ordering::Relaxed);
    tracing::debug!(
        target: "zend::repo_scan",
        baseline_bytes = baseline,
        "scan pool: per-conversation KV estimate anchored on the pre-scan corpus",
    );
}

/// Floor/ceiling on the per-conversation KV estimate.
///
/// The floor is measured, not guessed: a 24-conversation burst drove KV arenas
/// to 5760 MiB, i.e. ~240 MiB of LIVE KV each — and arenas reserve roughly
/// twice what they hold live (2560 MiB reserved against 1360 live in a steady
/// sample), because each format keeps its own partially-filled 16 MiB slabs.
/// So a conversation costs the pool about 480 MiB of *reserved* arena, which is
/// the quantity the allocator's ceiling actually counts.
///
/// The doubling is NOT slack, and halving it to price only the live half is a
/// measured dead end. The argument for halving is that slab waste belongs to a
/// *format* rather than to a conversation, so the marginal directory should not
/// pay it twice; the card says otherwise. At 256 MiB the gate opened 19
/// directories on the 16 GiB card and the pass hit the wall inside two minutes:
/// `pool_reserved=14976MiB` against `pool_used=13560MiB` — a 1415 MiB gap of
/// reserved-but-unfilled arena — 18 MiB free, a 16 MiB quantized arena refused
/// with `arenas_freed=0`, a device OOM that halved the admission budget, and 15
/// directories lost in a single millisecond. The gap is exactly the slab
/// overhead, and it scales with concurrently-live conversations, because each
/// one's chunks land in different format arenas at different fill levels.
///
/// It was also slower, which is the part worth remembering: 10 s per directory
/// against 6.82 at width 10, with decode throughput at nine co-batched sequences
/// falling from 15.9 tok/s to 7.9 and degrading further past ten. Past the wall
/// the extra width buys eviction churn, not throughput.
pub(super) const SCAN_CONV_KV_MIN: u64 = 480 * 1024 * 1024;

pub(super) const SCAN_CONV_KV_MAX: u64 = 768 * 1024 * 1024;
