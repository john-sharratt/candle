//! A gap's lease, allocated from the pools (`docs/kv_quantize_in_the_gap.md`
//! §3.2).
//!
//! Each class's demand is claimed the lowest-placed arena first — the pool's
//! free slots in existing arenas, then fresh regions, widening the class only
//! when no region can be had — and a claim the ground refuses stops that class
//! short rather than fail the gap: the class defers blocks for a gap.
//!
//! The firm demand may buy ground from the weight side: it is measured, and the
//! blocks it converts give back more native ground than they take. The spare
//! demand — a cold gap's guess — takes only ground standing free
//! ([`without_purchase`]), and a spare part the ground cannot give is no
//! shortfall: no block was counted on it. A firm shortfall's reason is kept
//! beside it, so the gap's log line says whether the ground ran out or
//! something failed. Each slot's address is resolved once, as it is allocated,
//! so the stack the claim reads is a copy of the lease.

use candle::Result;

use super::super::arena::ArenaKey;
use super::super::backing::ChunkedKvBacking;
use super::super::band_addrs::BandAddrs;
use super::super::gid_pool::ChunkGid;
use super::super::region_pool::without_purchase;
use super::super::size_class::SizeClass;
use super::demand::{GapDemand, PerClass};
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
pub(crate) fn lease_demand(backing: &ChunkedKvBacking, demand: &GapDemand) -> Result<Leased> {
    let mut lease = Lease::default();
    let mut short = [0usize; SizeClass::COUNT];
    let mut refusals = Vec::new();
    for (c, short_c) in short.iter_mut().enumerate() {
        if demand.total(c) == 0 {
            continue;
        }
        let (firm, spare) = (demand.firm[c], demand.spare[c]);
        let class = SizeClass::from_index(c)
            .ok_or_else(|| candle::Error::Msg(format!("gap lease: no size class {c}")))?;
        let key = ArenaKey::new(class, ArenaLocation::Gpu);
        let (mut gids, stopped) = backing.inner.alloc_chunks_for_key_upto(key, firm);
        *short_c = firm - gids.len();
        if let Some(why) = stopped {
            refusals.push(LeaseRefusal {
                class: c,
                short: *short_c,
                why,
            });
        } else if spare > 0 {
            let (more, _) =
                without_purchase(|| backing.inner.alloc_chunks_for_key_upto(key, spare));
            gids.extend(more);
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
