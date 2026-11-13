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

/// A lease for `demand` and each class's shortfall — and every class that came
/// up short reported with its reason, exactly once.
pub(super) fn leased(
    b: &ChunkedKvBacking,
    demand: &PerClass<usize>,
) -> candle::Result<(Lease<ChunkGid>, PerClass<usize>)> {
    let l = lease_demand(b, demand)?;
    let short_classes: Vec<usize> = (0..SizeClass::COUNT).filter(|&c| l.short[c] > 0).collect();
    let refused: Vec<usize> = l.refusals.iter().map(|r| r.class).collect();
    assert_eq!(refused, short_classes, "a shortfall without its reason");
    for r in &l.refusals {
        assert_eq!(r.short, l.short[r.class]);
    }
    Ok((l.lease, l.short))
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
    let (mut lease, short) = leased(&b, &demand).unwrap();
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
    let (lease, short) = leased(&b, &[0; SizeClass::COUNT]).unwrap();
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
    let (mut lease, _) = leased(&b, &demand).unwrap();
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
    let (mut first, _) = leased(&b, &demand).unwrap();
    let a = slots(&first);
    drop(first.release());

    let (mut again, _) = leased(&b, &demand).unwrap();
    assert_eq!(slots(&again), a, "the same places");

    // Keep one slot, as an installed block would.
    let kept_addr = again.stack_image().unwrap().addrs[0];
    let kept = again.take(q8, &[0]).unwrap();
    drop(again.release());
    let (third, _) = leased(&b, &demand).unwrap();
    let c = slots(&third);
    assert!(!c.contains(&kept_addr), "a held slot is not leased");
    let common = c.iter().filter(|x| a.contains(x)).count();
    assert_eq!(
        common, 23,
        "the other places again, and one more beside them"
    );
    drop(kept);
}

/// The gids of every slot `lease` holds in `class`, taken out in stack order.
fn take_all(lease: &mut Lease<ChunkGid>, class: usize) -> Vec<ChunkGid> {
    let n = lease.held()[class] as u32;
    lease.take(class, &(0..n).collect::<Vec<_>>()).unwrap()
}

/// Each gid's address as the convert would resolve it.
fn band_addrs(b: &ChunkedKvBacking, gids: &[ChunkGid]) -> Vec<u64> {
    b.inner
        .storage
        .read(|s| {
            gids.iter()
                .map(|g| band_ptr(s, g, "test"))
                .collect::<candle::Result<Vec<_>>>()
        })
        .unwrap()
        .unwrap()
}

/// A demand wider than one arena fills the pool's free slots, stamps fresh
/// regions for the rest, and resolves every address across the arena
/// changes.
#[test]
fn a_demand_wider_than_an_arena_spans_arenas_at_the_right_addresses() {
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let b = backing(&device, &CompressionPolicy::new(5));
    let (q8, _, _) = two_classes(&b, 0, 0);
    let per_arena = key(q8).chunks();
    let mut demand = [0; SizeClass::COUNT];
    demand[q8] = 3 * per_arena + 7;
    let (mut lease, short) = leased(&b, &demand).unwrap();
    assert_eq!(short, [0; SizeClass::COUNT]);
    let image = lease.stack_image().unwrap();
    let gids = take_all(&mut lease, q8);
    let mut arenas: Vec<usize> = gids.iter().map(ChunkGid::arena_idx).collect();
    arenas.dedup();
    assert!(arenas.len() >= 4, "{} arenas", arenas.len());
    assert_eq!(band_addrs(&b, &gids), image.addrs);
    let mut distinct = image.addrs.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), demand[q8], "no slot leased twice");
}

/// Every class the deepest level reaches, leased at once: each run in its
/// class, each address its gid's.
#[test]
fn every_class_a_level_reaches_is_leased_in_one_call() {
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let policy = CompressionPolicy::new(10);
    let b = backing(&device, &policy);
    let reach = super::launch_tests::reachable(&b, &policy);
    let demand: PerClass<usize> = std::array::from_fn(|c| if reach[c] { 5 + c } else { 0 });
    assert!(reach.iter().filter(|&&r| r).count() > 2);
    let (mut lease, short) = leased(&b, &demand).unwrap();
    assert_eq!(short, [0; SizeClass::COUNT]);
    assert_eq!(lease.held(), demand);
    let image = lease.stack_image().unwrap();
    for c in (0..SizeClass::COUNT).filter(|&c| reach[c]) {
        let (off, len) = (image.off[c] as usize, image.len[c] as usize);
        let gids = take_all(&mut lease, c);
        assert_eq!(
            band_addrs(&b, &gids),
            image.addrs[off..off + len],
            "class {c}"
        );
        for g in &gids {
            let arena_class = b
                .inner
                .storage
                .read(|s| s.arena_key(g.arena_idx()).map(|k| k.class.index()))
                .unwrap();
            assert_eq!(arena_class, Some(c), "a slot of class {c} in its own class");
        }
    }
}

/// Lease and release, gap after gap: the pools end where they started, so the
/// cycle leaks no slot.
#[test]
fn repeated_leases_and_releases_leak_nothing() {
    let Some((device, _cuda, _gpu)) = cuda() else {
        return;
    };
    let b = backing(&device, &CompressionPolicy::new(5));
    let (q8, q4, demand) = two_classes(&b, 90, 40);
    // Warm once so any arena the demand needs exists before the baseline.
    drop(leased(&b, &demand).unwrap().0.release());
    let base = [q8, q4].map(|c| b.inner.pool.free_slots(key(c)));
    for round in 0..5 {
        let (mut lease, _) = leased(&b, &demand).unwrap();
        // An installed block keeps a few slots for the round, as the catch-up's take does.
        let kept = lease.take(q8, &[0, 1, 2]).unwrap();
        drop(lease.release());
        assert_eq!(
            b.inner.pool.free_slots(key(q8)),
            base[0] - 3,
            "round {round}"
        );
        drop(kept);
        assert_eq!(
            [q8, q4].map(|c| b.inner.pool.free_slots(key(c))),
            base,
            "round {round}"
        );
    }
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
    let (mut l, _) = leased(&b, &demand).unwrap();
    let a = l.stack_image().unwrap();
    let mut held = l.take(q8, &(0..8).collect::<Vec<_>>()).unwrap();
    // Free the third slot only.
    drop(held.remove(2));
    let mut one = [0; SizeClass::COUNT];
    one[q8] = 1;
    let (next, _) = leased(&b, &one).unwrap();
    assert_eq!(next.stack_image().unwrap().addrs, vec![a.addrs[2]]);
}
