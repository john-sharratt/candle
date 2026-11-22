//! Describes the KV arenas, the warm tier, host RAM and admission as the
//! scheduler reads them.
//!
//! The frontier and the class rows are taken as one sample, because read apart
//! they disagree by hundreds of regions and that reads as an unaccounted tenant.
//! The span tenant rows say which holder owns the ground, since a total alone
//! cannot attribute it. Host RAM is reported against the budget the ingest
//! throttle gates on, not against the operating system's free figure, because
//! our own resident weights push that figure down.

use super::*;

/// KV arena occupancy, whole-process.
#[derive(Debug, Clone, Serialize)]
pub struct KvSection {
    /// One row per occupied size class of the resident GPU arenas.
    ///
    /// This replaced a float-vs-quant split, which is no longer a question an
    /// arena can answer: a size-class arena holds whatever fits its slots. The
    /// ladder carries the same signal — compression moves occupancy down it.
    pub classes: Vec<KvClassRow>,
    /// Per-backing per-format rows from every registered `ChunkedKvBacking`.
    pub arenas: Vec<ArenaRow>,
    /// One past the highest live region — the arena frontier, in regions.
    ///
    /// **The figure that costs weights, and the denominator of every efficiency
    /// number computed from this report.** The wave transient tier stands above the
    /// highest live arena and `weight_floor` is measured from there, so expert
    /// residency is set by this one index rather than by how many arenas are live
    /// or how full they are.
    ///
    /// Carried here rather than left to the consumer's own `region_stats` call so
    /// that the frontier and the class rows are ONE sample. Read apart, they are up
    /// to a publish interval out of step, and the mismatch does not look like skew:
    /// it appears as regions the class rows cannot account for, which reads exactly
    /// like a second tenant holding ground. Measured — 734 phantom regions.
    pub frontier_regions: usize,
    /// Who stands on the region just below [`Self::frontier_regions`] — the KV pools,
    /// held ground, or a span tenant by its label — in the same breath. What a
    /// frontier that stays up with the pools packed is waiting on.
    pub frontier_holder: &'static str,
    /// Regions held by any tenant at that same moment — arenas and span tenants.
    pub live_regions: usize,
    /// Of [`Self::live_regions`], the ones a span tenant holds: a sequence's
    /// recurrent state store, the provenance gallery. Ground in use that no
    /// compaction can pack, so a consumer measuring fragmentation must not charge it.
    pub span_regions: usize,
    /// Of [`Self::live_regions`], the ones **record** arenas hold. In use, and absent from
    /// [`Self::classes`] — which reports band pools only — so a consumer summing those rows
    /// must add this or it charges live records as waste.
    pub record_regions: usize,
    /// [`Self::span_regions`] broken down by tenant, from the SAME publish.
    ///
    /// **Which tenant holds the ground is not answerable from the total.** A tenant
    /// on thirty mostly-empty arenas and one on thirty full ones give the same
    /// `span_regions`, so an optimisation cannot be aimed at a tenant or
    /// attributed to one. Each row also carries what its holders occupy and how
    /// many strides it spans, which is what separates ground a packing walk could
    /// reclaim from ground only a different slot layout could.
    ///
    /// Published here rather than sampled by the consumer because the two figures
    /// must be one moment: read a publish interval apart, the difference between
    /// the total and the rows reads exactly like an uncounted tenant instead of
    /// like sampling skew — the same trap [`Self::frontier_regions`] records.
    pub span_tenants: Vec<SpanTenantRow>,
}

/// One span tenant's arenas at the moment of a publish.
#[derive(Debug, Clone, Serialize)]
pub struct SpanTenantRow {
    /// The tenant's name, as `SlotTenant::label` gives it.
    pub tenant: &'static str,
    /// Whole regions its arenas stand on.
    pub regions: usize,
    /// Bytes its holders actually occupy. The shortfall against `regions` is free
    /// slots plus every arena's unused tail.
    pub held_bytes: usize,
    /// Distinct strides. A tenant spread over several pools pays at least one
    /// region per pool however small its slots are, which no packing walk
    /// recovers — only giving it fewer strides does.
    pub pools: usize,
}

/// One size class's share of the resident GPU arenas.
#[derive(Debug, Clone, Serialize)]
pub struct KvClassRow {
    /// Slot stride in bytes — the class's identity.
    pub slot_bytes: usize,
    pub arenas: usize,
    pub reserved_bytes: u64,
    pub live_bytes: u64,
    /// Arenas this class's live chunks would occupy if they were packed.
    pub packed_arenas: usize,
    /// Arenas a perfect pack would empty — regions that are recoverable and that
    /// the empty-arena sweep structurally cannot recover, because an arena keeps
    /// its region until its *last* chunk goes and nothing moves chunks between
    /// arenas. This is the fragmentation figure; region holes are not (they are
    /// taken by the next claim, the free list being lowest-index-first).
    pub freeable_arenas: usize,
}

/// One `(backing, format)` arena row.
#[derive(Debug, Clone, Serialize)]
pub struct ArenaRow {
    pub backing: usize,
    pub format: String,
    pub arenas: usize,
    pub bytes: u64,
}

/// Warm (host RAM) KV tier and its drain state.
#[derive(Debug, Clone, Serialize)]
pub struct WarmSection {
    /// Residences holding a warm copy (the purge population), as of the last
    /// persistence pass — gauge-stamped, so this never takes the substrate lock.
    pub resident_count: usize,
    /// Drainable hot→warm deficit (the ingest backpressure signal).
    pub pending_warm_bytes: u64,
    /// Hot KV the drain is skipping because it is pinned — unreclaimable while
    /// the pin holds, so an eviction pass cannot turn it back into free regions.
    pub pinned_undrainable_bytes: u64,
    /// Bytes held by warm copies — the one host-side quantity admission can shrink.
    pub resident_bytes: u64,
    /// Host bytes of the per-turn payloads the substrate holds whole — wide-Q
    /// signature windows and projection events — outside the warm budget and
    /// growing with the corpus.
    pub held_payload_bytes: u64,
}

/// Host RAM through the budget model (`candle::vram::host_ram_budget`).
#[derive(Debug, Clone, Serialize)]
pub struct HostSection {
    pub total_bytes: u64,
    /// OS "available" (free + standby + zeroed) from the scheduler's cached
    /// probe. Diagnostic only — the throttle gates on the BUDGET below, because
    /// this number is pushed down by our own resident weights.
    pub available_bytes: u64,
    /// The OS/other-process buffer the budget holds back.
    pub buffer_bytes: u64,
    /// Weights reserved in full (mmap-backed but budgeted resident).
    pub weights_reserved_bytes: u64,
    /// True when the model exceeds `total − buffer` and weight pages will swap.
    pub weights_capped: bool,
    /// What the warm KV tier may occupy: total − buffer − pinned − weights.
    pub kv_warm_budget_bytes: u64,
    /// Drain-pipeline slack the throttle allows above the budget.
    pub pipeline_slack_bytes: u64,
    /// Live warm usage (resident + pending hot→warm) vs that ceiling.
    pub warm_usage_bytes: u64,
    /// The throttle predicate: `warm_usage > kv_warm_budget + slack`.
    pub over_budget: bool,
    /// Pages read from disk per second — the DIRECT thrash signal. A rate, not a
    /// level: high while available is low means genuine paging; near-zero while
    /// available is low means the low level is structural (pinned pools) and is
    /// not pressure. `None` off-Windows or on the first sample.
    pub pages_in_per_sec: Option<f64>,
    /// Commit charge / limit (RAM + pagefile) — the ceiling host allocations
    /// actually fail against. `None` off-Windows.
    pub commit_total_bytes: Option<u64>,
    pub commit_limit_bytes: Option<u64>,
    /// The QSA index page cache (`docs/hit_driven_projection.md` §7.6): its
    /// slab, what it holds, and its hits and misses since start. All zero when
    /// the process holds none.
    pub index_cache: IndexCacheSection,
}

/// The index page cache's slab and traffic.
#[derive(Debug, Clone, Serialize)]
pub struct IndexCacheSection {
    /// The share of host RAM the partition gives it.
    pub target_bytes: u64,
    pub pinned_bytes: u64,
    pub pageable_bytes: u64,
    /// Bytes of the blocks its entries hold.
    pub held_bytes: u64,
    pub entries: u64,
    pub hits: u64,
    pub misses: u64,
    /// Pages read and not cached: larger than the largest block, or no block
    /// could be had.
    pub uncached: u64,
    pub evicted: u64,
}

/// The admission throttle's live inputs and setpoint.
#[derive(Debug, Clone, Serialize)]
pub struct AdmissionSection {
    /// The regulated setpoint (AIMD).
    pub setpoint_bytes: u64,
    /// Live ceiling the setpoint is clamped to (headroom + evictable − pinned,
    /// clamped to unreserved device memory).
    pub ceiling_bytes: u64,
    pub quantum_bytes: u64,
    /// In-flight widths at capture.
    pub prefill_width: usize,
    pub section_width: usize,
    pub decode_width: usize,
    pub queued_prefills: usize,
}

/// MoE expert residency.
#[derive(Debug, Clone, Serialize)]
pub struct ExpertSection {
    /// Host-pinned expert pool (`cuMemAllocHost`) — non-pageable, structural.
    pub host_pinned_bytes: u64,
    /// VRAM-resident expert slots (governor class tally).
    pub vram_reserved_bytes: u64,
}
