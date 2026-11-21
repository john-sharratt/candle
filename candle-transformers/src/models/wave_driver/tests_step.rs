//! Tests for how a step is cut into segments and how the head's rows map back to
//! their members.
//!
//! A verify block beside a folded single-token prefill must keep all of its scored
//! rows, and every member after it must keep its own. The segment tests cover a
//! creep in the middle of the stack, one that starts at layer zero, one that
//! reaches the head, and one that runs alone. The tier test checks that layer
//! phases take the widest segment and the forward span takes the sum.

use super::*;

#[cfg(test)]
pub(super) mod step_tests {
    use super::*;

    /// **A verify block beside a folded single-token prefill keeps all its
    /// rows, and every member after it its own.** The head scores the caller's
    /// decode rows, the folded prefill among them, then the verify block's
    /// every row and the prompt's last, then glue; back in the caller's order
    /// each member takes exactly its scored rows, and the glue still trails.
    #[test]
    fn a_verify_block_beside_a_folded_prefill_keeps_its_rows() {
        // Prefill members in caller order: a 5-token verify block, a 1-token
        // prefill (folded into the decode rows), a 100-token prompt.
        let (single, multi) = (vec![1usize], vec![0usize, 2]);
        let scored = vec![5usize, 1, 1];
        // Head rows: decode 0 | folded 1 | verify 2..7 | prompt 7 | glue 8.
        let order = super::caller_order_rows(1, &single, &multi, &scored, 9).unwrap();
        assert_eq!(order, vec![0..1, 2..7, 1..2, 7..8, 8..9]);
        let total: usize = order.iter().map(|r| r.len()).sum();
        assert_eq!(total, 9, "every row the head scored, once");
        // A head short of what the members need is refused, not misassigned.
        assert!(super::caller_order_rows(1, &single, &multi, &scored, 7).is_err());
    }

    pub(in super::super) fn seg(start: usize, end: usize, creep: bool) -> Segment {
        Segment { start, end, creep }
    }

    #[test]
    fn a_creep_mid_stack_splits_the_step_in_three() {
        assert_eq!(
            segments(40, 10, 20, true),
            vec![seg(0, 10, false), seg(10, 20, true), seg(20, 40, false)]
        );
    }

    #[test]
    fn a_creep_from_layer_zero_to_the_head_is_one_segment() {
        assert_eq!(segments(40, 0, 40, true), vec![seg(0, 40, true)]);
    }

    #[test]
    fn a_creep_reaching_the_head_ends_the_step() {
        assert_eq!(
            segments(40, 30, 40, true),
            vec![seg(0, 30, false), seg(30, 40, true)]
        );
    }

    #[test]
    fn a_creep_alone_runs_only_its_window() {
        assert_eq!(segments(40, 10, 20, false), vec![seg(10, 20, true)]);
    }

    /// Layer-phase spans at the widest segment; the forward span at the sum.
    #[test]
    fn the_step_tier_shares_layer_phases_and_sums_forward_spans() {
        assert_eq!(
            step_plan(&[[10, 20, 5], [30, 15, 7], [5, 5, 3]]),
            [30, 20, 15]
        );
        assert_eq!(step_plan(&[]), [0, 0, 0]);
    }
}
