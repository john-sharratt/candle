//! A gap's lease, allocated from the pools (`docs/kv_quantize_in_the_gap.md`
//! §3.2).
//!
//! Each class's demand is claimed the lowest-placed arena first — the pool's
//! free slots in existing arenas, then fresh regions, widening the class only
//! when no region can be had — and a claim the ground refuses stops that class
//! short rather than fail the gap: the class defers blocks for a gap, and the
//! lease never buys ground from the experts. Why it stopped is kept beside the
//! shortfall, so the gap's log line says whether the ground ran out or
//! something failed. Each slot's address is resolved once, as it is
//! allocated, so the stack the claim reads is a copy of the lease.

use candle::Result;

use super::super::arena::ArenaKey;
use super::super::backing::ChunkedKvBacking;
use super::super::band_addrs::BandAddrs;
use super::super::gid_pool::ChunkGid;
use super::super::size_class::SizeClass;
use super::demand::PerClass;
use super::lease::Lease;
use crate::kv_cache::arena_table::ArenaLocation;

/// A class the pools could not fully serve: how many slots short, and the
/// refusal that stopped the claim.
#[derive(Debug)]
pub struct LeaseRefusal {
    pub class: usize,
    pub short: usize,
    pub why: candle::Error,
}

/// A gap's lease and what the ground could not give.
#[derive(Debug)]
pub(crate) struct Leased {
    pub lease: Lease<ChunkGid>,
    /// Each class's shortfall, zero where the demand was met.
    pub short: PerClass<usize>,
    /// One entry per class that came up short, with its reason.
    pub refusals: Vec<LeaseRefusal>,
}

/// A lease for `demand`, from `backing`'s pools. The pools are the model's,
/// shared by every layer, so any backing of the model serves.
pub(crate) fn lease_demand(backing: &ChunkedKvBacking, demand: &PerClass<usize>) -> Result<Leased> {
    let mut lease = Lease::default();
    let mut short = [0usize; SizeClass::COUNT];
    let mut refusals = Vec::new();
    for (c, &want) in demand.iter().enumerate() {
        if want == 0 {
            continue;
        }
        let class = SizeClass::from_index(c)
            .ok_or_else(|| candle::Error::Msg(format!("gap lease: no size class {c}")))?;
        let key = ArenaKey::new(class, ArenaLocation::Gpu);
        let (gids, stopped) = backing.inner.alloc_chunks_for_key_upto(key, want);
        short[c] = want - gids.len();
        if let Some(why) = stopped {
            refusals.push(LeaseRefusal {
                class: c,
                short: short[c],
                why,
            });
        }
        let addrs = backing.inner.storage.read(|s| -> Result<Vec<u64>> {
            let mut addrs = BandAddrs::new(s);
            gids.iter().map(|g| addrs.addr(g, "gap lease")).collect()
        })??;
        lease.add(c, gids.into_iter().zip(addrs));
    }
    Ok(Leased {
        lease,
        short,
        refusals,
    })
}
