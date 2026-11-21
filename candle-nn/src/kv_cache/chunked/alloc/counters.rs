//! Process-wide counters for arena creation and for class promotions.
//!
//! Arena creation is timed and class promotions are counted, so the allocator's
//! cost and the fit of the size ladder can be read from the stats log. A steady
//! trickle of promotions means the mechanism is working; a flood means the
//! ladder or the reservation is wrong for the workload.

use super::*;

pub(super) static ARENA_STATS_ENABLED: OnceLock<bool> = OnceLock::new();

pub(super) static ARENA_CREATE_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static ARENA_CREATE_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

pub(in super::super) fn arena_stats_enabled() -> bool {
    *ARENA_STATS_ENABLED.get_or_init(|| std::env::var("KV_ARENA_STATS").is_ok())
}

pub(super) fn record_arena_create(
    kind: &str,
    location: ArenaLocation,
    index: usize,
    elapsed_ns: u64,
) {
    if !arena_stats_enabled() {
        return;
    }
    let total_count = ARENA_CREATE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let total_ns = ARENA_CREATE_TOTAL_NS.fetch_add(elapsed_ns, Ordering::Relaxed) + elapsed_ns;
    let took_ms = elapsed_ns as f64 / 1_000_000.0;
    let total_ms = total_ns as f64 / 1_000_000.0;
    let avg_ms = total_ms / (total_count as f64);
    eprintln!(
        "[arena-create] kind={kind} location={location:?} index={index} took_ms={took_ms:.3} total_count={total_count} total_ms={total_ms:.3} avg_ms={avg_ms:.3}"
    );
}

pub(super) static CLASS_PROMOTIONS: AtomicU64 = AtomicU64::new(0);

/// Count a scarcity promotion and, under `KV_ARENA_STATS`, name it.
///
/// Worth its own counter rather than folding into the arena-create line: a
/// promotion is the allocator reporting that a class could not get a region,
/// which is the signal the ladder's shape is tuned against. A steady trickle is
/// the mechanism working; a flood means the ladder's shape is wrong for the
/// workload, or the reservation is undersized.
pub(super) fn record_class_promotion(from: SizeClass, to: SizeClass) {
    let total = CLASS_PROMOTIONS.fetch_add(1, Ordering::Relaxed) + 1;
    if arena_stats_enabled() {
        eprintln!(
            "[class-promote] {} B -> {} B (no region for the smaller class) total={total}",
            from.bytes(),
            to.bytes(),
        );
    }
}

/// Total scarcity promotions since process start.
pub fn class_promotion_count() -> u64 {
    CLASS_PROMOTIONS.load(Ordering::Relaxed)
}
