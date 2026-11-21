//! Tests for the per-class occupancy report.
//!
//! A class's live bytes are its slot count times its stride. Two formats that
//! share a class share one pool and one row, and never register a second arena. A
//! pool with registered but unallocated classes reports no resident arenas.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Occupancy is reported **per size class**, and a class's live bytes are
    /// its slot count times its stride — whatever formats happen to occupy it.
    #[test]
    fn gpu_class_stats_report_per_class_occupancy() {
        let pool = ChunkGidPool::new();
        let small = ArenaKey::new(SizeClass::at(0), ArenaLocation::Gpu);
        let large = ArenaKey::new(SizeClass::at(6), ArenaLocation::Gpu);

        // Two arenas in the small class with three live slots; one arena in
        // the large class with five.
        pool.register_arena(small);
        pool.register_arena(small);
        pool.register_arena(large);
        let _s: Vec<_> = (0..3).map(|_| pool.allocate_for(small).unwrap()).collect();
        let _l: Vec<_> = (0..5).map(|_| pool.allocate_for(large).unwrap()).collect();

        let stats = pool.gpu_class_stats();
        let row = |c: SizeClass| stats.classes[c.index()];

        assert_eq!(row(small.class).slot_bytes, small.slot_stride());
        assert_eq!(row(small.class).arenas, 2);
        assert_eq!(row(small.class).reserved_bytes, 2 * TARGET_ARENA_BYTES);
        assert_eq!(row(small.class).live_bytes, 3 * small.slot_stride());

        assert_eq!(row(large.class).arenas, 1);
        assert_eq!(row(large.class).reserved_bytes, TARGET_ARENA_BYTES);
        assert_eq!(row(large.class).live_bytes, 5 * large.slot_stride());

        assert_eq!(stats.total_arenas(), 3);
        assert_eq!(
            stats.total_live_bytes(),
            3 * small.slot_stride() + 5 * large.slot_stride()
        );
    }

    /// **Two formats sharing a class share a pool.** This is the property the
    /// whole initiative exists to obtain, and it is visible here: allocating
    /// for two different formats that map to one class grows a single row, not
    /// two, and never registers a second arena.
    #[test]
    fn formats_sharing_a_class_share_one_pool() {
        use crate::kv_cache::{KvFormat, QuantFormat};
        const ELEMS: usize = 1024;
        let pool = ChunkGidPool::new();
        // Q4_1 and Q4_KS are both 640 B payloads, so one rung, one key.
        let a = ArenaKey::for_format(
            KvFormat::Quantized(QuantFormat::Q4_1),
            ELEMS,
            ArenaLocation::Gpu,
        )
        .unwrap();
        let b = ArenaKey::for_format(
            KvFormat::Quantized(QuantFormat::Q4_KS),
            ELEMS,
            ArenaLocation::Gpu,
        )
        .unwrap();
        assert_eq!(a, b, "the two formats must resolve to one key");

        pool.register_arena(a);
        let g1 = pool.allocate_for(a).expect("first format claims a slot");
        let g2 = pool
            .allocate_for(b)
            .expect("second format claims from the SAME pool");
        assert_eq!(g1.arena_idx(), g2.arena_idx());

        let stats = pool.gpu_class_stats();
        assert_eq!(stats.total_arenas(), 1, "one arena serves both formats");
        assert_eq!(
            stats.classes[a.class.index()].live_bytes,
            2 * a.slot_stride()
        );
    }

    #[test]
    fn gpu_class_stats_on_an_empty_pool_are_zero() {
        // A pool with registered-but-unallocated classes reports nothing: the
        // per-wave diagnostic must not count preallocated empty pool-table
        // entries as resident arenas.
        let stats = ChunkGidPool::new().gpu_class_stats();
        assert_eq!(stats.total_arenas(), 0);
        assert_eq!(stats.total_reserved_bytes(), 0);
        assert_eq!(stats.total_live_bytes(), 0);
        // The slot_bytes column is still populated, so a report shows the
        // whole ladder rather than a ragged subset.
        for (i, c) in stats.classes.iter().enumerate() {
            assert_eq!(c.slot_bytes, SizeClass::at(i).bytes());
        }
    }
}
