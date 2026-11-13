//! A gap's lease against real pools (`docs/kv_quantize_in_the_gap.md` §3.2):
//! exactly the demand, each slot at its arena address, the lowest-placed arena
//! first, and every slot back in its pool once released.

use super::super::arena::ArenaKey;
use super::super::compress::band_ptr;
use super::super::compression_policy::CompressionPolicy;
use super::super::gid_pool::ChunkGid;
use super::super::size_class::{class_for_format, SizeClass};
use super::super::ChunkedKvBacking;
use super::demand::PerClass;
use super::launch_tests::{backing, cuda};
use super::lease::Lease;
use super::lease_alloc::lease_demand;
use crate::kv_cache::arena_table::ArenaLocation;
use crate::kv_cache::{KvFormat, QuantFormat};

/// The classes of two formats the production level selects, and a demand of
/// `a` and `b` slots in them.
fn two_classes(backing: &ChunkedKvBacking, a: usize, b: usize) -> (usize, usize, PerClass<usize>) {
    let epc = backing.elems_per_chunk();
    let q8 = class_for_format(KvFormat::Quantized(QuantFormat::Q8_0), epc)
        .unwrap()
        .index();
    let q4 = class_for_format(KvFormat::Quantized(QuantFormat::Q4_0), epc)
        .unwrap()
        .index();
    assert_ne!(q8, q4);
    let mut demand = [0; SizeClass::COUNT];
    demand[q8] = a;
    demand[q4] = b;
    (q8, q4, demand)
}

fn key(class: usize) -> ArenaKey {
    ArenaKey::new(SizeClass::from_index(class).unwrap(), ArenaLocation::Gpu)
}

#[test]
fn a_lease_holds_exactly_the_demand_each_slot_at_its_address() {
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let b = backing(&device, &CompressionPolicy::new(5));
    let (q8, q4, demand) = two_classes(&b, 37, 5);
    let (mut lease, short) = lease_demand(&b, &demand).unwrap();
    assert_eq!(short, [0; SizeClass::COUNT]);
    assert_eq!(lease.held(), demand);
    let image = lease.stack_image().unwrap();
    assert_eq!(image.addrs.len(), 42);
    for c in [q8, q4] {
        let (off, len) = (image.off[c] as usize, image.len[c] as usize);
        let gids = lease.take(c, &(0..len as u32).collect::<Vec<_>>()).unwrap();
        let addrs: Vec<u64> = b
            .inner
            .storage
            .read(|s| {
                gids.iter()
                    .map(|g| band_ptr(s, g, "test"))
                    .collect::<candle::Result<Vec<_>>>()
            })
            .unwrap()
            .unwrap();
        assert_eq!(addrs, image.addrs[off..off + len], "class {c}");
    }
}

#[test]
fn an_empty_demand_leases_nothing() {
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let b = backing(&device, &CompressionPolicy::new(5));
    let (lease, short) = lease_demand(&b, &[0; SizeClass::COUNT]).unwrap();
    assert!(lease.is_empty());
    assert_eq!(short, [0; SizeClass::COUNT]);
    assert!(lease.stack_image().unwrap().addrs.is_empty());
}

#[test]
fn released_slots_are_back_in_their_pools() {
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let b = backing(&device, &CompressionPolicy::new(5));
    let (q8, q4, demand) = two_classes(&b, 300, 17);
    let (mut lease, _) = lease_demand(&b, &demand).unwrap();
    let leased = [q8, q4].map(|c| b.inner.pool.free_slots(key(c)));
    drop(lease.release());
    for (i, c) in [q8, q4].into_iter().enumerate() {
        assert_eq!(
            b.inner.pool.free_slots(key(c)),
            leased[i] + demand[c],
            "class {c}"
        );
    }
}

/// The slots a lease names, as a set: the pool takes the lowest-placed arena
/// first, and within an arena hands out its free slots in its own order.
fn slots(lease: &Lease<ChunkGid>) -> Vec<u64> {
    let mut a = lease.stack_image().unwrap().addrs;
    a.sort_unstable();
    a
}

/// Lowest-placed arena first, every gap: a lease released and taken again
/// lands on the same slots, and one whose slot is still held takes the next
/// free one instead.
#[test]
fn each_lease_takes_the_lowest_free_places() {
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let b = backing(&device, &CompressionPolicy::new(5));
    let (q8, _, demand) = two_classes(&b, 24, 0);
    let (mut first, _) = lease_demand(&b, &demand).unwrap();
    let a = slots(&first);
    drop(first.release());

    let (mut again, _) = lease_demand(&b, &demand).unwrap();
    assert_eq!(slots(&again), a, "the same places");

    // Keep one slot, as an installed block would.
    let kept_addr = again.stack_image().unwrap().addrs[0];
    let kept = again.take(q8, &[0]).unwrap();
    drop(again.release());
    let (third, _) = lease_demand(&b, &demand).unwrap();
    let c = slots(&third);
    assert!(!c.contains(&kept_addr), "a held slot is not leased");
    let common = c.iter().filter(|x| a.contains(x)).count();
    assert_eq!(
        common, 23,
        "the other places again, and one more beside them"
    );
    drop(kept);
}

/// A slot freed low in the arena — a truncate, an eviction — is the next
/// lease's first, ahead of anything higher.
#[test]
fn a_hole_opened_below_is_filled_first() {
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let b = backing(&device, &CompressionPolicy::new(5));
    let (q8, _, demand) = two_classes(&b, 8, 0);
    let (mut l, _) = lease_demand(&b, &demand).unwrap();
    let a = l.stack_image().unwrap();
    let mut held = l.take(q8, &(0..8).collect::<Vec<_>>()).unwrap();
    // Free the third slot only.
    drop(held.remove(2));
    let mut one = [0; SizeClass::COUNT];
    one[q8] = 1;
    let (next, _) = lease_demand(&b, &one).unwrap();
    assert_eq!(next.stack_image().unwrap().addrs, vec![a.addrs[2]]);
}
