//! Tests the prefill slab packer's cap, slack and oversize rules on concrete
//! sequence lengths.
//!
//! A straggler is absorbed when it lands at or under the slack ceiling and is
//! split one token past it. Mid-fleet slabs close at the bare cap, a tail too
//! large to absorb splits normally, and a sequence beyond the slack ceiling
//! travels alone uncut. A further test fixes the slack at a quarter of the cap.

use super::*;

#[cfg(test)]
pub(super) mod slab_tests {
    use super::{pack_prefill_slabs, prefill_slack_cap};

    #[test]
    fn straggler_is_absorbed_within_slack() {
        // 8192 + 128 = 8320 ≤ 10240 slack: ONE slab, no straggler wave.
        assert_eq!(pack_prefill_slabs(&[8192, 128], 8192), vec![(0, 2)]);
        // Right at the slack ceiling: still one slab.
        assert_eq!(pack_prefill_slabs(&[8192, 2048], 8192), vec![(0, 2)]);
        // One past the ceiling: split.
        assert_eq!(
            pack_prefill_slabs(&[8192, 2049], 8192),
            vec![(0, 1), (1, 2)]
        );
    }

    #[test]
    fn mid_fleet_slabs_respect_the_bare_cap() {
        // Only the FINAL slab may overshoot: slab 1 closes at the cap because
        // absorbing the remaining 8500 would blow past the slack ceiling; the
        // tail then packs together (500 + 8000 = 8500 ≤ 10240 absorbs at ITS
        // closing point).
        assert_eq!(
            pack_prefill_slabs(&[8000, 500, 8000], 8192),
            vec![(0, 1), (1, 3)]
        );
    }

    #[test]
    fn tail_larger_than_slack_splits_normally() {
        // 10 × 1030: greedy packs 7 (7210); the 3-seq tail (3090) would land at
        // 10300 > 10240, so no absorb — but the tail then fits one slab alone.
        let lens = [1030usize; 10];
        assert_eq!(pack_prefill_slabs(&lens, 8192), vec![(0, 7), (7, 10)]);
    }

    #[test]
    fn oversize_sequences_travel_alone_and_uncut() {
        // A single sequence past even the slack cap is never split.
        assert_eq!(pack_prefill_slabs(&[30000], 8192), vec![(0, 1)]);
        assert_eq!(
            pack_prefill_slabs(&[30000, 100, 30000], 8192),
            vec![(0, 1), (1, 2), (2, 3)]
        );
    }

    #[test]
    fn slack_is_a_quarter_of_the_cap() {
        assert_eq!(prefill_slack_cap(8192), 10240);
        assert_eq!(prefill_slack_cap(100), 125);
    }
}
