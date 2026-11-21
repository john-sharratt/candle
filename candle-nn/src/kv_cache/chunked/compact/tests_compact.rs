//! Tests the planner's cap, the slot extents, the collision refusal and the
//! report.
//!
//! The plan cap must cover a whole arena and scale with the budget. A slot past
//! an arena's capacity must have no address, and a plan that reads what it writes
//! must be refused before launch. An empty pass must read as empty, and a
//! frontier that rose must not report a reclaim.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// A pool's plan cap covers at least one whole arena, so a pass can empty the
    /// arena holding the frontier rather than leave it part-drained for the next
    /// allocations to refill: the smallest rung's arena (16 MiB of 320 B slots)
    /// raises the cap, the largest rung's (16 MiB of 16 KiB slots) does not.
    #[test]
    fn a_pools_plan_cap_covers_one_whole_arena() {
        let smallest = ArenaKey::new(SizeClass::from_index(0).unwrap(), ArenaLocation::Gpu);
        let largest = ArenaKey::new(
            SizeClass::from_index(SizeClass::COUNT - 1).unwrap(),
            ArenaLocation::Gpu,
        );
        assert_eq!(smallest.chunks(), 52_428);
        assert_eq!(planned_moves_cap(smallest, CAP_BUDGET), 52_428);
        assert_eq!(largest.chunks(), 1_024);
        assert_eq!(planned_moves_cap(largest, CAP_BUDGET), 32_768);
    }

    /// A pass of three times the budget plans three times the moves a pool, so it
    /// claims what three passes would and its holders are swept once; the one-arena
    /// floor still holds under a short budget.
    #[test]
    fn a_pools_plan_cap_scales_with_the_budget() {
        let smallest = ArenaKey::new(SizeClass::from_index(0).unwrap(), ArenaLocation::Gpu);
        let largest = ArenaKey::new(
            SizeClass::from_index(SizeClass::COUNT - 1).unwrap(),
            ArenaLocation::Gpu,
        );
        let triple = Duration::from_millis(240);
        assert_eq!(planned_moves_cap(largest, triple), 98_304);
        assert_eq!(planned_moves_cap(smallest, triple), 98_304);
        assert_eq!(
            planned_moves_cap(largest, Duration::from_millis(40)),
            16_384
        );
        assert_eq!(
            planned_moves_cap(smallest, Duration::from_millis(40)),
            52_428
        );
    }

    /// The figure the pass is judged by is the frontier's fall, not the move count
    /// — a pass that copies ten thousand chunks and leaves the frontier where it
    /// was has achieved nothing, and the report must make that legible.
    #[test]
    fn the_report_measures_the_frontier_not_the_work() {
        let busy_but_useless = CompactionReport {
            moves: 10_000,
            frontier_before: 1727,
            frontier_after: 1727,
            ..Default::default()
        };
        assert_eq!(busy_but_useless.regions_reclaimed(), 0);
        assert!(!busy_but_useless.is_empty(), "it did do work");

        let cheap_and_good = CompactionReport {
            moves: 40,
            frontier_before: 1727,
            frontier_after: 200,
            ..Default::default()
        };
        assert_eq!(cheap_and_good.regions_reclaimed(), 1527);
    }

    /// A slot index past the arena's capacity has no address, and the address of
    /// an in-range slot carries that arena's own extent.
    ///
    /// **The raw-gid namespace is wider than an arena.** `GID_STRIDE` is one fixed
    /// power of two while an arena holds `chunks_per_region` slots for its class, so
    /// an index can be a valid gid and still name ground past the arena's end.
    /// Unbounded, the copy planner turns that into a write into the next region —
    /// another arena, the wave transient tier, or a sequence's recurrent store — and
    /// because every address in the reservation is mapped it does not fault. It
    /// surfaces as whatever reads those bytes next getting a wrong number, in a
    /// different subsystem from the writer.
    ///
    /// Pure arithmetic over a hand-built table: no device, no pool, and no
    /// allocation — the bound is the whole behaviour under test.
    #[test]
    fn a_slot_past_the_arenas_capacity_has_no_address() {
        use crate::kv_cache::arena_table::ResolvedArenaInfo;
        let info = vec![ResolvedArenaInfo {
            base_ptr: 0x1_0000_0000,
            chunk_byte_stride: 512,
            chunk_capacity: 4,
        }];

        assert_eq!(slot_addr(&info, 0, 0), Some((0x1_0000_0000, 512)));
        assert_eq!(slot_addr(&info, 0, 3), Some((0x1_0000_0000 + 3 * 512, 512)));
        // One past the end is the first address that belongs to somebody else.
        assert_eq!(slot_addr(&info, 0, 4), None, "capacity is exclusive");
        assert_eq!(
            slot_addr(&info, 0, super::super::super::types::GID_STRIDE - 1),
            None,
            "a valid gid index is not a valid slot index"
        );
        // An arena with no device slab cannot supply an address at all.
        let cpu = vec![ResolvedArenaInfo {
            base_ptr: 0,
            chunk_byte_stride: 512,
            chunk_capacity: 4,
        }];
        assert_eq!(slot_addr(&cpu, 0, 0), None);
        assert_eq!(slot_addr(&info, 1, 0), None, "no such arena");
    }

    /// Every slot an in-range index names lies wholly inside its arena, when the
    /// copy length is the arena's own stride.
    ///
    /// This is the property the planner needs and the reason the length no longer
    /// comes from the size class: `[addr, addr + stride)` is inside
    /// `[base, base + capacity * stride)` for every valid index, which is only true
    /// while the stride used for the length is the stride used for the address.
    #[test]
    fn a_slots_extent_never_leaves_its_arena() {
        use crate::kv_cache::arena_table::ResolvedArenaInfo;
        let stride = 320i64;
        let capacity = 7u32;
        let base = 0x2_0000_0000u64;
        let info = vec![ResolvedArenaInfo {
            base_ptr: base,
            chunk_byte_stride: stride,
            chunk_capacity: capacity,
        }];
        let end = base + capacity as u64 * stride as u64;
        for idx in 0..capacity as usize {
            let (addr, len) = slot_addr(&info, 0, idx).expect("an in-range slot");
            assert!(addr >= base, "slot {idx} starts inside the arena");
            assert!(
                addr + len as u64 <= end,
                "slot {idx} ends at {} which is past the arena's end {end}",
                addr + len as u64,
            );
        }
    }

    /// A plan that reads a slot another record writes is refused before the launch.
    ///
    /// **This is the corruption, reduced to arithmetic.** The migrate kernel is one
    /// concurrent block per record, so a slot appearing on both sides is a
    /// read/write race and the losing record copies the other's chunk — which is then
    /// installed under the losing chunk's holder. It reached production because the
    /// two halves are separately legitimate: `plan_pool` keeps its sources and
    /// destinations disjoint, and `claim_moves` then asks the allocator for any free
    /// slot in the destination arena, which is allowed to be a planned source whose
    /// chunk was released after the census sampled it.
    ///
    /// Asserted on the records rather than on the plan, because the records are what
    /// the kernel is handed and the plan's own disjointness is already pinned
    /// elsewhere — it was true, and not enough.
    #[test]
    fn a_plan_that_reads_what_it_writes_is_refused() {
        // Disjoint: three moves packing downward, nothing read and written.
        let ok = CopyRecords {
            srcs: vec![0x3000, 0x3800, 0x4000],
            dsts: vec![0x1000, 0x1800, 0x2000],
            lens: vec![0x800, 0x800, 0x800],
            source_collisions: 0,
        };
        assert!(disjoint_ends(&ok).is_ok());

        // Record 0 reads 0x2000 while record 2 writes it — the collision the census
        // snapshot allows.
        let racy = CopyRecords {
            srcs: vec![0x2000, 0x3800, 0x4000],
            dsts: vec![0x1000, 0x1800, 0x2000],
            lens: vec![0x800, 0x800, 0x800],
            source_collisions: 0,
        };
        let err = disjoint_ends(&racy).expect_err("a slot read and written must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("READS a range another WRITES"),
            "the refusal must say a read raced a write, got: {msg}",
        );

        // Two records writing one slot: the same hazard with both ends inverted, and
        // the case a source-only check cannot see.
        let double_write = CopyRecords {
            srcs: vec![0x3000, 0x3800],
            dsts: vec![0x1000, 0x1000],
            lens: vec![0x800, 0x800],
            source_collisions: 0,
        };
        let err = disjoint_ends(&double_write).expect_err("one slot written twice must refuse");
        assert!(
            err.to_string()
                .contains("two records WRITE overlapping ranges"),
            "got: {err}",
        );

        // A destination landing PART-way into another slot — what a wrong length or
        // stride produces, and what an equality test on bare addresses misses.
        let partial = CopyRecords {
            srcs: vec![0x3000, 0x3800],
            dsts: vec![0x1000, 0x1400],
            lens: vec![0x800, 0x800],
            source_collisions: 0,
        };
        assert!(
            disjoint_ends(&partial).is_err(),
            "overlap is about ranges, not equal start addresses"
        );

        // Two records reading the same bytes is harmless — both copy the same content
        // out — so it must NOT be refused.
        let shared_read = CopyRecords {
            srcs: vec![0x3000, 0x3000],
            dsts: vec![0x1000, 0x1800],
            lens: vec![0x800, 0x800],
            source_collisions: 0,
        };
        assert!(
            disjoint_ends(&shared_read).is_ok(),
            "two reads of one range are not a collision"
        );

        // An empty plan is trivially disjoint — a pass that moves nothing is not a
        // pass that races.
        assert!(disjoint_ends(&CopyRecords::default()).is_ok());
    }

    /// A pass that moved nothing reads empty, so a caller can skip its log line and
    /// its `reclaim_spare_ground` without inspecting anything else.
    #[test]
    fn a_pass_that_moved_nothing_is_empty() {
        assert!(CompactionReport::default().is_empty());
    }

    /// A frontier that somehow rose does not underflow into a vast fake reclaim.
    #[test]
    fn a_risen_frontier_reports_no_reclaim() {
        let grew = CompactionReport {
            moves: 1,
            frontier_before: 100,
            frontier_after: 140,
            ..Default::default()
        };
        assert_eq!(grew.regions_reclaimed(), 0);
    }
}
