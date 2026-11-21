//! Unit tests for the pure policy that picks a maintenance op.
//!
//! They pin the priority of drop, compact and combine, the dead-ratio and settle
//! thresholds, the forced pass, and the rule that a drop-only pass never
//! relocates. Segment stats are built by hand, so no clock or disk is involved.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in super::super) fn stat(id: u64, total: u64, live: u64, age: u64) -> SegmentStat {
        SegmentStat {
            id: SegmentId(id),
            total_bytes: total,
            live_bytes: live,
            age_secs: age,
        }
    }

    #[test]
    fn drop_wins_over_everything() {
        // A fully-dead segment is dropped even if a compact candidate exists.
        let stats = [stat(1, 1000, 0, 0), stat(2, 1000, 100, 999)];
        assert_eq!(
            pick_maintenance_op(&stats, false),
            Some(MaintenanceOp::Drop(SegmentId(1)))
        );
    }

    #[test]
    fn compact_needs_dead_ratio_and_settle() {
        // Dead enough but not settled → no op.
        let young = [stat(1, 1000, 100, 10)];
        assert_eq!(pick_maintenance_op(&young, false), None);
        // Settled but not dead enough → no op: 5%, and the 15% BlackWorld
        // rewrote a 4.3 GB segment for every 15–60 s.
        let clean = [stat(1, 1000, 950, 999)];
        assert_eq!(pick_maintenance_op(&clean, false), None);
        let churn = [stat(1, 1000, 850, 999)];
        assert_eq!(pick_maintenance_op(&churn, false), None);
        // A fifth dead → compact.
        let fifth = [stat(1, 1000, 800, 999)];
        assert_eq!(
            pick_maintenance_op(&fifth, false),
            Some(MaintenanceOp::Compact(SegmentId(1)))
        );
        // Settled and 90% dead → compact.
        let dead = [stat(1, 1000, 100, 999)];
        assert_eq!(
            pick_maintenance_op(&dead, false),
            Some(MaintenanceOp::Compact(SegmentId(1)))
        );
    }

    /// A full volume's pass drops a dead segment and never relocates.
    #[test]
    fn a_drop_only_pass_drops_and_nothing_else() {
        let stats = [stat(1, 1000, 100, 999), stat(2, 1000, 0, 0)];
        assert_eq!(pick_drop(&stats), Some(MaintenanceOp::Drop(SegmentId(2))));
        assert_eq!(pick_drop(&stats[..1]), None, "a compaction is not a drop");
    }

    #[test]
    fn force_waives_age_and_ratio_gates() {
        // Young (age 10 < 60) and only 5% dead — no op on a normal pass.
        let s = [stat(1, 1000, 950, 10)];
        assert_eq!(pick_maintenance_op(&s, false), None);
        // A forced pass compacts it: any dead weight qualifies, age ignored.
        assert_eq!(
            pick_maintenance_op(&s, true),
            Some(MaintenanceOp::Compact(SegmentId(1)))
        );
        // A fully-live segment is never compacted, even forced (nothing to shed).
        let live = [stat(1, 1000, 1000, 0)];
        assert_eq!(pick_maintenance_op(&live, true), None);
    }

    #[test]
    fn compact_prefers_the_deadest_settled_segment() {
        let stats = [
            stat(1, 1000, 500, 999), // 50% dead
            stat(2, 1000, 100, 999), // 90% dead — deadest
            stat(3, 1000, 800, 999), // 20% dead
        ];
        assert_eq!(
            pick_maintenance_op(&stats, false),
            Some(MaintenanceOp::Compact(SegmentId(2)))
        );
    }

    #[test]
    fn combine_pairs_adjacent_small_segments() {
        // No drop, no compact candidate (all clean + settled), two small
        // adjacent segments → combine.
        let stats = [stat(1, 1000, 1000, 999), stat(2, 1000, 1000, 999)];
        assert_eq!(
            pick_maintenance_op(&stats, false),
            Some(MaintenanceOp::Combine(SegmentId(1), SegmentId(2)))
        );
    }

    #[test]
    fn no_op_when_nothing_qualifies() {
        // One big fully-live settled segment: nothing to drop/compact, and a
        // single segment has no adjacent pair to combine.
        let big = [stat(
            1,
            COMBINE_SEGMENT_BYTES + 1,
            COMBINE_SEGMENT_BYTES + 1,
            999,
        )];
        assert_eq!(pick_maintenance_op(&big, false), None);
    }

    #[test]
    fn op_targets_enumerate_segments() {
        assert_eq!(
            MaintenanceOp::Drop(SegmentId(3)).targets(),
            vec![SegmentId(3)]
        );
        assert_eq!(
            MaintenanceOp::Combine(SegmentId(1), SegmentId(2)).targets(),
            vec![SegmentId(1), SegmentId(2)]
        );
    }
}
