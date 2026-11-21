//! Tests for the ring's stock arithmetic and for which pending promotions survive
//! a forward fault.
//!
//! The stock tests pin the ring's size with room and without it, against
//! decode's reserve, the sweep word and the read-ahead window, including the
//! floor and the ceiling at half the ring. The promotion tests check that a fault
//! drops only the demand claims, and that a claim that has not completed never
//! lands.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::{lands, ring_stock, split_ahead, DevicePromotion, RingStock};

    use std::cell::Cell;

    pub(in super::super) fn promo(
        row: usize,
        expert: usize,
        slot: usize,
        ahead: bool,
    ) -> DevicePromotion {
        DevicePromotion {
            row,
            expert,
            slot,
            ticket: 40 + row as u64,
            ahead,
        }
    }

    /// A fault drops the demand claims and keeps the read-ahead ones, each side
    /// in pending order.
    #[test]
    fn a_fault_keeps_read_ahead_claims_and_drops_demand_ones() {
        let (ahead, demand) = split_ahead(vec![
            promo(3, 7, 100, false),
            promo(5, 2, 101, true),
            promo(4, 9, 102, false),
            promo(6, 1, 103, true),
        ]);
        let key = |v: &[DevicePromotion]| {
            v.iter()
                .map(|p| (p.row, p.expert, p.slot))
                .collect::<Vec<_>>()
        };
        assert_eq!(key(&ahead), vec![(5, 2, 101), (6, 1, 103)]);
        assert_eq!(key(&demand), vec![(3, 7, 100), (4, 9, 102)]);
    }

    /// A complete claim lands unless it is a demand claim under a fault; an
    /// incomplete one never lands, and the fault word is read only for a
    /// complete demand claim — after the completion it is ordered behind.
    #[test]
    fn a_demand_claim_waits_out_a_fault_and_a_read_ahead_claim_does_not() {
        let reads = Cell::new(0);
        let word = |set: bool| {
            let reads = &reads;
            move || {
                reads.set(reads.get() + 1);
                set
            }
        };
        assert!(lands(true, false, word(false)));
        assert!(!lands(true, false, word(true)));
        assert_eq!(reads.get(), 2);
        assert!(lands(true, true, word(true)));
        assert!(!lands(false, false, word(false)));
        assert!(!lands(false, true, word(false)));
        assert_eq!(reads.get(), 2, "read only for a complete demand claim");
    }

    pub(in super::super) fn stock(target: usize, reserve: u32, sweep: u32) -> RingStock {
        RingStock {
            target,
            reserve,
            sweep,
        }
    }

    /// With room the ring stocks for every miss, ×1.25, reserves nothing, and
    /// lets a launch claim up to that prediction.
    #[test]
    fn with_room_the_ring_stocks_for_all_misses_and_reserves_nothing() {
        assert_eq!(ring_stock(8, 100, true, 0, 0, 1024), stock(125, 0, 125));
        // The floor and the half-capacity ceiling.
        assert_eq!(ring_stock(0, 4, true, 0, 0, 1024), stock(32, 0, 32));
        assert_eq!(ring_stock(0, 1_000, true, 0, 0, 1024), stock(512, 0, 512));
        // More empty slots than the prediction: every one is stocked, and the
        // sweep word stays the prediction.
        assert_eq!(ring_stock(8, 100, true, 300, 0, 1024), stock(300, 0, 125));
    }

    /// Without room decode's target is the reserve and the sweep word, and the
    /// stock past it is empty slots only — a full zone stocks exactly decode's
    /// target.
    #[test]
    fn without_room_the_stock_past_decodes_reserve_is_empty_slots() {
        // Full zone: stock = reserve = sweep = decode's target.
        assert_eq!(ring_stock(40, 100, false, 0, 0, 1024), stock(50, 50, 50));
        // Fewer empties than decode's target: the target, made of lazy victims.
        assert_eq!(ring_stock(40, 100, false, 25, 0, 1024), stock(50, 50, 50));
        // Plenty empty: every one stocked, decode's target still reserved, and
        // a prompt claiming past it still a sweep.
        assert_eq!(ring_stock(40, 100, false, 300, 0, 1024), stock(300, 50, 50));
        // Never past half the ring.
        assert_eq!(
            ring_stock(40, 1_000, false, 1_000, 0, 256),
            stock(128, 50, 50)
        );
        // A prompt-only launch: decode's floor is still reserved.
        assert_eq!(ring_stock(0, 100, false, 10, 0, 1024), stock(32, 32, 32));
    }

    /// Read-ahead's window is stocked past the misses' target, never past half
    /// the ring, and moves neither the reserve nor the sweep word.
    #[test]
    fn the_read_ahead_window_is_stocked_on_top() {
        assert_eq!(ring_stock(40, 100, false, 0, 14, 1024), stock(64, 50, 50));
        assert_eq!(ring_stock(8, 100, true, 0, 14, 1024), stock(139, 0, 125));
        assert_eq!(ring_stock(40, 100, false, 0, 100, 256), stock(128, 50, 50));
        // Empty slots past target + window are still all stocked.
        assert_eq!(
            ring_stock(40, 100, false, 300, 14, 1024),
            stock(300, 50, 50)
        );
    }
}
