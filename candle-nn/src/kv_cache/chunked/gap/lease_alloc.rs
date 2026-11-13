//! A gap's lease, allocated from the pools (`docs/kv_quantize_in_the_gap.md`
//! §3.2).
//!
//! Each class's demand is claimed the lowest-placed arena first — the pool's
//! free slots in existing arenas, then fresh regions, widening the class only when no region
//! can be had — and a claim the ground refuses stops that class short rather
//! than fail the gap: the class defers blocks for a gap, and the lease never
//! buys ground from the experts. Each slot's address is resolved once, as it is
//! allocated, so the stack the claim reads is a copy of the lease.

use candle::Result;

use super::super::arena::{Arena, ArenaKey, ArenaStorageState};
use super::super::backing::ChunkedKvBacking;
use super::super::gid_pool::ChunkGid;
use super::super::size_class::SizeClass;
use super::demand::PerClass;
use super::lease::Lease;
use crate::kv_cache::arena_table::ArenaLocation;

/// A lease for `demand`, from `backing`'s pools, and each class's shortfall:
/// what the ground could not give. The pools are the model's, shared by every
/// layer, so any backing of the model serves.
pub(crate) fn lease_demand(
    backing: &ChunkedKvBacking,
    demand: &PerClass<usize>,
) -> Result<(Lease<ChunkGid>, PerClass<usize>)> {
    let mut lease = Lease::default();
    let mut short = [0usize; SizeClass::COUNT];
    for (c, &want) in demand.iter().enumerate() {
        if want == 0 {
            continue;
        }
        let class = SizeClass::from_index(c)
            .ok_or_else(|| candle::Error::Msg(format!("gap lease: no size class {c}")))?;
        let key = ArenaKey::new(class, ArenaLocation::Gpu);
        let (gids, _refused) = backing.inner.alloc_chunks_for_key_upto(key, want);
        short[c] = want - gids.len();
        let addrs = backing.inner.storage.read(|s| slot_addresses(s, &gids))??;
        lease.add(c, gids.into_iter().zip(addrs));
    }
    Ok((lease, short))
}

/// Each gid's device address, the arena looked up once per run of slots in
/// one arena — which is how the pool hands them out, lowest arena first.
fn slot_addresses(storage: &ArenaStorageState, gids: &[ChunkGid]) -> Result<Vec<u64>> {
    let arenas = storage.arenas();
    let mut last: Option<(usize, &Arena)> = None;
    gids.iter()
        .map(|gid| {
            let ai = gid.arena_idx();
            let arena = match last {
                Some((idx, arena)) if idx == ai => arena,
                _ => {
                    let arena = arenas.get(&ai).ok_or_else(|| {
                        candle::Error::Msg(format!("gap lease: arena {ai} not in storage"))
                    })?;
                    last = Some((ai, arena));
                    arena
                }
            };
            arena.slot_ptr(gid.chunk_idx()).ok_or_else(|| {
                candle::Error::Msg(format!("gap lease: arena {ai} is not GPU-resident"))
            })
        })
        .collect()
}
