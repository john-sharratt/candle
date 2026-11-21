//! Reports how the KV side's regions are used, so admission and the weight side read one set of figures.
//!
//! Regions fall into three disjoint states: claimable now, blocked by a standing
//! tier's ceiling, and held by a tenant. A low `free` with a high `blocked` means
//! the wave's own tier stands on the ground and the wave should be narrower,
//! while a low `free` with a low `blocked` is genuine KV pressure. The span
//! partition cuts the reservation into bands that sum to the span exactly, so a
//! reading can show where the card's memory went.

use super::*;

/// Occupancy of one device's KV side.
#[derive(Debug, Clone, Copy)]
pub struct RegionStats {
    /// Regions the reservation actually claimed.
    pub total: usize,
    /// Regions held by any tenant right now — a KV arena, a record arena, or a
    /// span tenant such as the recurrent state store or the provenance gallery.
    ///
    /// **Not "held by an arena", which is what this said.** Three unlike tenants
    /// claim from one free list, and reading this as the arena count is how the
    /// fragmentation efficiency figure came to divide the KV pools' packed arena
    /// count by a frontier that a recurrent state store was holding up: on
    /// Qwen3.8-Flash-Next that reads 4% with the KV pools packed to within two
    /// arenas of perfect. [`Self::span_tenant`] is the part that is not an arena.
    pub live: usize,
    /// Of [`Self::live`], the regions held by a span tenant rather than an arena.
    ///
    /// Counted where the handles are minted rather than derived by subtracting the
    /// arena pools' totals, because those come from a different snapshot behind a
    /// different lock — and a figure formed by subtracting two populations is the
    /// shape that produced the wrong reading in the first place.
    pub span_tenant: usize,
    /// Regions claimable right now without evicting anything — the pressure
    /// signal, and the admission budget. Excludes anything the transient tier's
    /// ceiling forbids; see [`blocked`](Self::blocked).
    pub free: usize,
    /// Regions nobody owns that the tier's ceiling puts out of reach anyway.
    ///
    /// Reported apart from `free` because the two call for opposite responses: a
    /// low `free` with a low `blocked` is genuine KV pressure and wants the
    /// boundary moved, while a low `free` with a high `blocked` is the wave's own
    /// tier standing on the ground and wants a narrower wave.
    /// **Every blocked region is the tier's; the weight side holds none of them,
    /// and it cannot.** `total` *is* `(weight_floor − region_base)/REGION_BYTES`,
    /// recomputed by `layout_span` on every path that moves the floor, so the
    /// weight side's ground lies outside `[0, total)` by construction and can
    /// never appear in a count of regions the KV side owns.
    ///
    /// That is worth stating because a `weight_blocked` field was added here to
    /// separate the two, and subtracted from `vram_budget_available` to close an
    /// OOM. It was identically zero on every reachable pool state — verified
    /// structurally and then in every line of a full gate run — so the
    /// subtraction was a no-op and the admission budget it was meant to correct
    /// never changed. Its test passed only by calling the free functions with a
    /// `(total, ceiling)` pair the pool cannot produce.
    ///
    /// The over-promise it was aimed at is real and still open: admission spends
    /// `(free + blocked)`, and the blocked half does not come back until the wave
    /// ends. The fix belongs in *when* a budget may count tier ground, not in a
    /// split of `blocked` that has no second half.
    pub blocked: usize,
    /// Most regions ever live at once — how close the partition came to full.
    pub peak_live: usize,
    /// One past the **highest live region index** — the arena frontier, in
    /// regions, and the figure a compaction exists to reduce.
    ///
    /// `live_watermark - live` is the number of free regions stranded *below* the
    /// frontier: ground the KV side owns and reports as available, which the wave
    /// transient tier cannot use because it must stand above every live arena, and
    /// which therefore pushes `weight_floor` right and costs expert residency.
    /// Zero holes means `live_watermark == live`, which is what a perfect
    /// compaction pass leaves behind and what a test can assert exactly — unlike a
    /// throughput figure.
    pub live_watermark: usize,
    /// Bytes the wave transient tier occupies **right now**, and the ceiling it
    /// imposes on the region count while it does.
    ///
    /// Both are zero between forwards: the tier does not exist then, and the
    /// whole span below the weight floor is the KV side's. A non-zero reading is
    /// a snapshot taken mid-wave, not a reservation.
    pub transient_bytes: usize,
    pub transient_ceiling: usize,
    /// Fresh regions claimed since boot while a tier was placed — claims that
    /// arrived after the wave's admit phase. Zero is the precondition for
    /// anchoring the tier at the arena frontier.
    pub fresh_claims_during_wave: usize,
    /// Claims refused because the tier's base was the ceiling — room existed,
    /// the tier's position hid it. A non-zero reading says the admit phase let a
    /// claim through to the wave.
    pub refusals_during_wave: usize,
    /// Bytes of the span the weight side is permitted to hold.
    pub weight_bytes: usize,
    /// Bytes of the span nothing has been able to use — the tail left over when
    /// the region count rounds down.
    pub slack_bytes: usize,
    /// Bytes of address space reserved, and the driver's mapping granule. The
    /// two together say whether a shortfall is the driver refusing or the
    /// layout rounding.
    pub reserved_bytes: usize,
    pub granularity: usize,
    /// Device bytes already in use the instant **before** this span was
    /// reserved — the CUDA context, the driver's own working set, the module
    /// images, and anything another process holds.
    ///
    /// Captured here because this is the only moment it is measurable in
    /// isolation: the reservation runs before the weight load, so nothing of
    /// ours is on the card yet. Without it the memory report can only say
    /// `in_use − span − pool` and must lump this constant in with the thing a
    /// reader actually wants — an allocation the report does not know about.
    /// That residual read 1.4 GB on a 16 GB card while being almost entirely
    /// this baseline, which makes a real leak of a few hundred MB invisible.
    ///
    /// `None` when no VRAM gauge was registered for the device, which the
    /// reservation itself tolerates — so this is a state a live pool reaches,
    /// not just a CPU-test artifact. It stays an `Option` all the way to the
    /// memory report because a `0` here would read as "the card was empty
    /// before us" and license a subtraction that silently did nothing.
    pub pre_reservation_in_use_bytes: Option<u64>,
    /// Transient tiers placed since the pool was made — one per forward, so a
    /// reader polling it sees each wave go by.
    pub tiers_placed: u64,
}

/// Occupancy of a device's KV side, or `None` if it has no reservation yet.
/// The lowest free region of `ordinal`'s KV side — the lowest hole below its
/// frontier, the region the next claim takes — or `None` when no region is
/// free or the device has no reservation. Reads the pool without making one.
pub fn lowest_free_region(ordinal: usize) -> Option<usize> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal)
        .and_then(|pool| pool.free.peek().map(|Reverse(i)| *i))
}

/// One past the highest region of `0..next` not in `free` — the arena
/// frontier — or `0` when every one is free.
///
/// **The free regions at the top, walked down, and no set built.** The
/// frontier is found from the free indices sorted high to low: each one
/// that is the region just under the frontier so far moves it down, and the
/// first that is not ends the walk. It used to collect every free index into
/// a `HashSet` and probe it from `next` down, on every reading — and admission
/// read it several times per offer, under the pool's lock.
pub(super) fn watermark_of(next: usize, free: impl Iterator<Item = usize>) -> usize {
    let mut top: Vec<usize> = free.filter(|&i| i < next).collect();
    top.sort_unstable_by(|a, b| b.cmp(a));
    let mut watermark = next;
    for i in top {
        if i + 1 != watermark {
            break;
        }
        watermark = i;
    }
    watermark
}

/// What admission reads of a device's KV side, and nothing else: the regions
/// there are, those live, the arena frontier, and the weight side's bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionGround {
    pub total: usize,
    pub live: usize,
    pub live_watermark: usize,
    pub weight_bytes: usize,
}

/// [`region_stats`]'s four figures admission reads per offer, without the free
/// list's census, the span's layout or the reservation's queries the rest of
/// it computes — a lock and a walk of the free regions at the frontier.
/// `None` without a reservation.
pub fn region_ground(ordinal: usize) -> Option<RegionGround> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| RegionGround {
        total: pool.total,
        live: pool.live,
        live_watermark: pool.live_watermark(),
        weight_bytes: (pool.span_end() - pool.weight_floor) as usize,
    })
}

pub fn region_stats(ordinal: usize) -> Option<RegionStats> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    map.get(&ordinal).map(|pool| RegionStats {
        total: pool.total,
        live: pool.live,
        span_tenant: pool.span_live,
        free: pool.free_count(),
        blocked: pool.ceiling_blocked(),
        peak_live: pool.peak_live,
        live_watermark: pool.live_watermark(),
        transient_bytes: pool.transient_bytes,
        transient_ceiling: pool.region_ceiling(),
        fresh_claims_during_wave: pool.fresh_claims_during_wave,
        refusals_during_wave: pool.refusals_during_wave,
        weight_bytes: (pool.span_end() - pool.weight_floor) as usize,
        slack_bytes: layout_span(
            pool.span_base,
            pool.persist_bytes,
            pool.span_bytes,
            pool.dense_bytes,
            pool.weight_floor,
        )
        .slack,
        reserved_bytes: pool.reservation.reserved_bytes(),
        granularity: pool.reservation.granularity(),
        pre_reservation_in_use_bytes: pool.pre_reservation_in_use,
        tiers_placed: pool.tiers_placed,
    })
}

/// The span cut by what holds each byte, at one moment — every band in bytes,
/// summing to the span exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpanPartition {
    /// The held ground: launch tables, resident headers and partials that
    /// outlive every forward. Fixed at reservation.
    pub held_ground: usize,
    /// The persistence staging block beside it. Fixed at reservation.
    pub persist_staging: usize,
    /// The model's dense weights.
    pub dense: usize,
    /// The dense block's rounding up to a whole region.
    pub dense_padding: usize,
    /// Regions held by KV arenas.
    pub kv: usize,
    /// Regions held by span tenants — recurrent state, the provenance
    /// gallery, index caches, creep residuals.
    pub tenants: usize,
    /// The largest wave transient tier placed since the last reading: the
    /// activations of the widest forward. It stands on regions nobody holds, so
    /// it is carved out of them rather than added beside them.
    pub transient: usize,
    /// Regions nobody holds, less the tier's peak.
    pub regions_free: usize,
    /// The tail of the region range too short for a region.
    pub slack: usize,
    /// The weight side's ground, `[weight_floor, span_end)`.
    pub weight_zone: usize,
}

impl SpanPartition {
    /// Every band added up — the span, by construction.
    pub fn sum(&self) -> usize {
        self.held_ground
            + self.persist_staging
            + self.dense
            + self.dense_padding
            + self.kv
            + self.tenants
            + self.transient
            + self.regions_free
            + self.slack
            + self.weight_zone
    }
}

/// [`SpanPartition`] of a span of `span_bytes` whose fixed left block is
/// `persist` bytes — `held` of them held ground — holding `dense_bytes` of
/// dense weights, its weight side from `weight_floor_offset` bytes into it,
/// `live` of whose regions are held — `span_live` of them by span tenants — and
/// whose widest tier since the last reading was `tier_peak` bytes.
#[allow(clippy::too_many_arguments)]
pub(super) fn partition(
    persist: usize,
    held_ground: usize,
    span_bytes: usize,
    dense_bytes: usize,
    weight_floor_offset: usize,
    live: usize,
    span_live: usize,
    tier_peak: usize,
) -> SpanPartition {
    let l = layout_span(
        0,
        persist,
        span_bytes,
        dense_bytes,
        weight_floor_offset as u64,
    );
    let regions = l.total * REGION_BYTES;
    let held = live.min(l.total) * REGION_BYTES;
    let kv = live.saturating_sub(span_live).min(l.total) * REGION_BYTES;
    let unheld = regions - held;
    let transient = tier_peak.min(unheld);
    SpanPartition {
        held_ground,
        persist_staging: persist - held_ground,
        dense: dense_bytes,
        dense_padding: (l.region_base - l.dense_base) as usize - dense_bytes,
        kv,
        tenants: held - kv,
        transient,
        regions_free: unheld - transient,
        slack: l.slack,
        weight_zone: span_bytes - l.weight_floor as usize,
    }
}

/// The span on device `ordinal`, cut by what holds each byte — see
/// [`SpanPartition`]. Reading it restarts the tier's peak, so each reading
/// covers the forwards since the one before.
pub fn span_partition(ordinal: usize) -> Option<SpanPartition> {
    let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
    let pool = map.get_mut(&ordinal)?;
    let peak = std::mem::replace(&mut pool.tier_peak, pool.transient_bytes);
    Some(pool_partition(pool, peak))
}

/// [`span_partition`] without restarting the tier's peak — for a report that
/// reads the fixed blocks and must not take the peak from the per-wave reading.
pub fn span_partition_now(ordinal: usize) -> Option<SpanPartition> {
    let map = pools().lock().unwrap_or_else(|e| e.into_inner());
    let pool = map.get(&ordinal)?;
    Some(pool_partition(pool, pool.tier_peak))
}

/// `pool` cut by what holds each byte, with `peak` as the tier's widest.
pub(super) fn pool_partition(pool: &RegionPool, peak: usize) -> SpanPartition {
    partition(
        pool.persist_bytes,
        pool.held_bytes,
        pool.span_bytes,
        pool.dense_bytes,
        (pool.weight_floor - pool.span_base) as usize,
        pool.live,
        pool.span_live,
        peak.max(pool.transient_bytes),
    )
}
