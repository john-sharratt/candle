//! Reports the reservation's current addresses and the regions a tenant may move.
//!
//! Every device pointer a KV kernel is handed has to fall inside the
//! reservation, and the layout is the one place that says where the boundary
//! is. Callers that check a batch of addresses take one snapshot rather than
//! fetching it per pointer, because the layout is behind the pool's lock. The
//! relocation plan for a tenant only empties regions whose emptying closes a
//! hole below the frontier, since any other move merely shifts the hole.

use super::*;

/// The reservation's addresses, as they stand right now.
///
/// Every device pointer this engine hands a KV kernel has to land somewhere in
/// here, and which somewhere decides whether it is legal. Published so
/// [`super::guard`] can answer that question without duplicating the layout —
/// there is exactly one authority for where the boundary is, and it is this
/// module.
#[derive(Debug, Clone, Copy)]
pub struct SpanLayout {
    /// First byte of the whole reservation.
    pub span_base: u64,
    /// One past its last byte.
    pub span_end: u64,
    /// First byte of the region range — the persistence staging block sits
    /// between `span_base` and here.
    pub region_base: u64,
    /// Bytes carved from the staging block so far.
    pub persist_carved: usize,
    /// Regions the KV side currently has.
    pub total: usize,
    /// The weight side's left edge: KV below, expert slots above.
    pub weight_floor: u64,
    /// The wave transient tier, while one stands.
    pub transient_base: Option<u64>,
    pub transient_bytes: usize,
}

impl SpanLayout {
    /// One past the last byte of the region range.
    pub fn region_end(&self) -> u64 {
        self.region_base + (self.total * REGION_BYTES) as u64
    }
}

/// The reservation's current addresses, or `None` if this device has none yet.
///
/// **Takes the pool's global lock — hoist it out of loops.** The layout is read
/// far more often than it changes, but it is not a constant (the weight floor
/// moves, the transient tier comes and goes, growth changes `total`), so it
/// lives behind the same mutex as the mutation. A caller that checks a *batch*
/// of addresses takes one snapshot and reuses it; a caller that fetches per
/// address turns a lock designed for arena claims into per-pointer traffic.
///
/// That is not hypothetical: the slot-metadata pointer guard fetched per
/// pointer, at `N_PALETTE` K and V addresses per KV head per slice — 4,608
/// acquisitions per attention layer, 98% of the metadata pack. Both other
/// callers already snapshot once and loop
/// (`expert_lre::compute`, `SlotStateHost::span_layout_for_checks`).
pub fn span_layout(ordinal: usize) -> Option<SpanLayout> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| SpanLayout {
        span_base: pool.span_base,
        span_end: pool.span_end(),
        region_base: pool.region_base,
        persist_carved: pool.persist_carved,
        total: pool.total,
        weight_floor: pool.weight_floor,
        transient_base: pool.transient_base,
        transient_bytes: pool.transient_bytes,
    })
}

/// The regions a tenant's compaction may usefully empty into fresh low arenas:
/// the span-level two-cursor plan, restricted to the regions the tenant holds.
///
/// The top cursor walks down from the frontier over live regions while the
/// tenant holds them, and stops at the first live region it does not; the bottom
/// cursor walks up over free regions. They pair while each hole lies below the
/// region it would receive. Answers the paired regions, highest first — each one
/// emptied lowers the frontier.
///
/// **Only those, because emptying anything else moves a hole rather than
/// closing one.** A region below another tenant's live region leaves, when
/// emptied, a hole under a frontier that tenant still holds; the next pass finds
/// that hole and fills it from the next region down. Planned per tenant against
/// "any hole below the frontier", Qwen3.8-Flash-Next's recurrent-state passes
/// moved 912 slots each with the region count unchanged, KV arenas above them
/// holding the frontier where it was.
pub(crate) fn frontier_relocations(
    watermark: usize,
    is_free: impl Fn(usize) -> bool,
    held: impl Fn(usize) -> bool,
) -> Vec<usize> {
    let mut run = Vec::new();
    for i in (0..watermark).rev() {
        if is_free(i) {
            continue;
        }
        if !held(i) {
            break;
        }
        run.push(i);
    }
    let holes = (0..watermark).filter(|&i| is_free(i));
    let pairs = holes
        .zip(run.iter())
        .take_while(|(hole, region)| hole < *region)
        .count();
    run.truncate(pairs);
    run
}

/// [`frontier_relocations`] on `ordinal`'s pool, for a tenant holding the regions
/// `held` names (by index). `None` without a reservation.
///
/// The caller gathers `held` under its own lock first: this takes the region
/// pool's, and no tenant lock is ever taken inside it.
pub fn regions_to_relocate(
    ordinal: usize,
    held: &std::collections::HashSet<usize>,
) -> Option<Vec<usize>> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    let pool = map.get(&ordinal)?;
    let free: std::collections::HashSet<usize> = pool.free.iter().map(|Reverse(i)| *i).collect();
    Some(frontier_relocations(
        pool.live_watermark(),
        |i| free.contains(&i),
        |i| held.contains(&i),
    ))
}

/// What a device's held ground holds, or `None` if it has no reservation yet.
pub fn held_ground_stats(ordinal: usize) -> Option<HeldStats> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| pool.held.stats())
}

/// The highest region a device's held ground stands on, or `None` when it holds
/// no arena (or has no reservation yet).
pub fn held_ground_top_rank(ordinal: usize) -> Option<usize> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).and_then(|pool| pool.held.top_rank())
}

/// What a device's held ground will claim at its next provisioning pass, or
/// `None` if it has no reservation yet — ground a forward priced now should
/// count as taken.
pub fn held_ground_next_claim(ordinal: usize) -> Option<HeldClaim> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| pool.held.next_claim())
}

/// `device`'s held ground, or `None` if it has no reservation yet. Cloned out of
/// the pools lock, which provisioning takes again to claim and release regions.
pub(super) fn held_ground_of(device: &candle::CudaDevice) -> Option<Arc<SpanHeldGround>> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&device.cuda_stream().context().ordinal())
        .map(|pool| pool.held.clone())
}

/// **Between forwards:** fit `device`'s held ground arenas to what recent
/// forwards took — see `SpanHeldGround::provision`. Does nothing before a
/// reservation exists or while a model is loading.
pub fn provision_held_ground(device: &candle::CudaDevice) {
    if let Some(held) = held_ground_of(device) {
        held.provision(device);
    }
}

/// **From a known point between forwards:** make room on `device`'s held ground
/// for one take of `bytes` — see `SpanHeldGround::make_room`. Does nothing
/// before a reservation exists.
pub(crate) fn make_held_room(device: &candle::CudaDevice, bytes: usize) {
    if let Some(held) = held_ground_of(device) {
        held.make_room(device, bytes);
    }
}

/// **With nothing in flight:** let `device`'s held ground go back to what it
/// holds plus one region — see `SpanHeldGround::settle`. The idle counterpart of
/// [`provision_held_ground`].
pub fn settle_held_ground(device: &candle::CudaDevice) {
    if let Some(held) = held_ground_of(device) {
        held.settle(device);
    }
}

/// Who holds a device's held ground, by call site and largest first, or `None`
/// if it has no reservation yet.
pub fn held_ground_sites(ordinal: usize) -> Option<Vec<HeldSite>> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| pool.held.sites())
}
