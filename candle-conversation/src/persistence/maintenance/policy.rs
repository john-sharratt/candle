//! Picks at most one maintenance op from per-segment stats.
//!
//! A fully dead segment is dropped first, since that costs nothing. Otherwise
//! the segment with the most dead weight is compacted once it is dead enough
//! and has settled, and only then are small adjacent segments combined.
//! Compaction rewrites a segment's live bytes, so these thresholds keep a store
//! from churning segments that would reclaim little. The choice is a pure
//! function of the stats, and a forced pass waives the age and ratio gates.

use super::*;

/// Compact trigger: a sealed segment must be at least this fraction dead
/// weight before relocating it is worth the I/O (§8). Relocating rewrites the
/// live bytes to reclaim the dead, so this bounds the write amplification at
/// its inverse: at 10% a 357 GB store rewrote 3.3–3.9 GB every 15–60 s to
/// reclaim ~0.6 GB (BlackWorld, 2026-10-27).
pub const SEGMENT_COMPACT_MIN_DEAD: f64 = 0.20;

/// Compact trigger: a sealed segment must have settled at least this long
/// since it was sealed (or last compacted) — a rate-limit so a just-sealed
/// segment accumulates deaths before it is churned (§8).
pub const SEGMENT_COMPACT_MIN_AGE_SECS: u64 = 60;

/// Combine trigger: two adjacent sealed segments each below this live size
/// merge into the active, cutting the segment count after compaction has
/// shrunk neighbours (§6).
pub const COMBINE_SEGMENT_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// One sealed segment's stats, fed to [`pick_maintenance_op`].
#[derive(Clone, Copy, Debug)]
pub struct SegmentStat {
    pub id: SegmentId,
    /// Record bytes on disk (excluding the superblock).
    pub total_bytes: u64,
    /// Live read-back record bytes (superseded / tombstoned / distilled
    /// excluded).
    pub live_bytes: u64,
    /// Seconds since the segment was sealed (file mtime).
    pub age_secs: u64,
}

/// Choose at most one maintenance op from per-segment stats, in priority
/// order: drop a fully-dead segment (cheapest, biggest win); else compact the
/// segment that most exceeds the savings + settle thresholds; else combine the
/// first adjacent pair of small segments.
///
/// Pure — the caller supplies the stats (ages included) so the policy is
/// unit-testable without a clock or disk. `stats` is assumed ascending by id,
/// so `windows(2)` are the adjacency candidates for combine.
pub fn pick_maintenance_op(stats: &[SegmentStat], force: bool) -> Option<MaintenanceOp> {
    // 1. Drop: any fully-dead segment (unconditional — always worth deleting).
    if let Some(s) = stats.iter().find(|s| s.live_bytes == 0) {
        return Some(MaintenanceOp::Drop(s.id));
    }
    // 2. Compact the deadest segment past the ratio + settle thresholds. When
    //    `force` is set (the manual `POST /v1/substrate/zend/maintenance` trigger) the age
    //    and ratio gates are waived — any segment carrying *some* dead weight
    //    qualifies — so a just-sealed segment holding a freshly-archived
    //    conversation's now-dead records is compacted immediately.
    let mut best: Option<(SegmentId, f64)> = None;
    for s in stats {
        if s.total_bytes == 0 {
            continue;
        }
        if !force && s.age_secs < SEGMENT_COMPACT_MIN_AGE_SECS {
            continue;
        }
        let dead = s.total_bytes.saturating_sub(s.live_bytes);
        let ratio = dead as f64 / s.total_bytes as f64;
        let qualifies = if force {
            dead > 0
        } else {
            ratio >= SEGMENT_COMPACT_MIN_DEAD
        };
        if qualifies && best.is_none_or(|(_, r)| ratio > r) {
            best = Some((s.id, ratio));
        }
    }
    if let Some((id, _)) = best {
        return Some(MaintenanceOp::Compact(id));
    }
    // 3. Combine: first adjacent pair both below the size threshold.
    for w in stats.windows(2) {
        if w[0].live_bytes < COMBINE_SEGMENT_BYTES && w[1].live_bytes < COMBINE_SEGMENT_BYTES {
            return Some(MaintenanceOp::Combine(w[0].id, w[1].id));
        }
    }
    None
}

/// Drop the first fully-dead segment, and nothing else: no relocation, so no
/// scratch space.
pub fn pick_drop(stats: &[SegmentStat]) -> Option<MaintenanceOp> {
    stats
        .iter()
        .find(|s| s.live_bytes == 0)
        .map(|s| MaintenanceOp::Drop(s.id))
}
