//! Full-system memory report: one snapshot covering VRAM, KV arenas, the warm
//! tier, host RAM, and the admission throttle — published by the scheduler at
//! its ~2 s telemetry cadence, logged, and served over HTTP.
//!
//! Exists because tonight's failures were all *accounting* failures before they
//! were memory failures: admission read the pool reuse gap as free VRAM, the
//! host-RAM floor read an 11 GB pinned expert pool as pressure, and none of the
//! numbers involved were visible in one place. This report puts every quantity
//! the throttles reason about side by side — what the driver says, what the
//! pool has reserved, how many KV regions are free, what is pinned and can
//! never move — so a wrong inference is attributable from a single log
//! line instead of a night of cross-referencing.
//!
//! Process-global slot (the [`phase_ring`](super::phase_ring) pattern): the
//! scheduler lives in this crate, the HTTP layer in `zend`, and both link this
//! module — `GET /v1/memory` reads [`latest`] with no engine lock. Every
//! publish also emits the full report as one JSON debug line on
//! `candle_conversation::scheduler::memory`, so a run's memory history is
//! reconstructable from the log alone.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use candle::vram::{host_pinned_bytes, AllocClass};
use candle::wave_provenance::last_wave_declines;
use candle::Device;
use candle_nn::kv_cache::global_arena_memory_report;
use serde::Serialize;

use super::Scheduler;

/// One full memory snapshot. All byte quantities are raw bytes; consumers
/// derive MiB. `None` sections mean the underlying source is unavailable in
/// this build/run (e.g. no CUDA governor), never that the value is zero.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryReport {
    /// Milliseconds since the Unix epoch at capture.
    pub captured_unix_ms: u64,
    pub vram: Option<VramSection>,
    pub kv: KvSection,
    pub warm: WarmSection,
    pub host: HostSection,
    pub admission: AdmissionSection,
    pub experts: ExpertSection,
    pub weights: WeightSection,
    pub gallery: GallerySection,
    pub accounting: AccountingSection,
    /// What the wave throughput planner currently believes, and the budget the
    /// next fill will open with.
    pub planner: PlannerSection,
    /// Who is holding the span and why — `None` until the first census, which
    /// is taken only when the engine is actually short of ground.
    pub holdings: Option<super::holdings::Holdings>,
    /// What the waves in the last window actually spent their time on, and
    /// which of the three regimes the engine is in.
    pub latency: LatencySection,
}

/// Which resource the waves in the last window were actually waiting on.
///
/// **The three are different problems with different fixes**, and two of them
/// look identical from throughput alone. A wave whose forward time rises with
/// the sequence count is either doing more arithmetic (compute) or waiting on
/// more separate expert loads (latency) — the same curve, opposite remedies.
/// More residency helps the second and does nothing for the first; a wider wave
/// helps the first and makes the second worse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bound {
    /// The bus was busy most of the window: expert bytes are the cost, and
    /// residency or wider waves are the levers.
    Bandwidth,
    /// Little crossed the bus, but the engine waited on it — many separate
    /// misses, each a round trip that cannot be amortised across the wave.
    /// Widening makes this worse: more sequences route to more distinct
    /// experts, and the working set outgrows what the cache holds and the
    /// predictor anticipates.
    Latency,
    /// Neither: the time went into arithmetic.
    Compute,
    /// Nothing ran in the window.
    Idle,
}

/// Where a window's wave time went, in the domains the budget can act on.
///
/// Deliberately **not** a generic span dump. The profiler already provides one,
/// behind a feature that is off in production builds, and a hundred span names
/// do not answer "what is the engine waiting on". These are the domains a
/// budget decision can actually move — bytes over the bus, waits on those
/// bytes, and everything else — read from counters that are always on.
#[derive(Debug, Clone, Serialize)]
pub struct LatencySection {
    /// Wall-clock the window covers.
    pub window_ms: u64,
    /// Expert bytes that crossed the bus in it, and what that implies about
    /// the achieved rate.
    pub transfer_bytes: u64,
    pub achieved_bytes_per_s: f64,
    /// The rate the planner believes the bus can do, for the ratio below.
    pub link_bytes_per_s: f64,
    /// Achieved over link. **The bandwidth question**: near 1 and the bus is
    /// the wall; near 0 and it is not, whatever else is slow.
    pub bus_utilisation: f64,
    /// Expert loads and how they went. `misses` is the count of round trips —
    /// the quantity latency is paid per, where bandwidth is paid per byte.
    pub expert_hits: u64,
    pub expert_misses: u64,
    pub cold_loads: u64,
    pub warm_loads: u64,
    /// Times the forward blocked on a transfer that had not landed, and
    /// prefetches that were issued but arrived late. **The latency question**:
    /// these rise when the working set outgrows what the predictor can stay
    /// ahead of, and they rise with the number of distinct experts routed —
    /// which is to say, with the width of the decode wave.
    pub fence_stalls: u64,
    pub late_loads: u64,
    /// Misses per second of window — the round-trip rate the engine is paying.
    pub misses_per_s: f64,
    /// The fraction of loads that arrived too late to be used when wanted.
    pub late_fraction: f64,
    /// Which regime the numbers above put the window in.
    pub bound: Bound,
}

/// The admission planner's internal state — the estimates it decides with, and
/// the budget it would open a wave against right now.
///
/// **Published because a decision is not reviewable from its outcome.** A fill
/// that admits nothing looks identical whether the model refused on the floor,
/// refused on throughput, or was never offered anything; and a model that has
/// learned a wrong copy rate produces perfectly reasonable-looking refusals
/// from it. The estimates are the state that makes a refusal legible, and they
/// move slowly enough that sampling them at the report cadence catches
/// everything that matters.
#[derive(Debug, Clone, Serialize)]
pub struct PlannerSection {
    /// Bytes a second the planner believes expert weights cross the bus at,
    /// and the link it is bounded by.
    pub copy_bytes_per_s: f64,
    pub link_bytes_per_s: f64,
    /// Seconds a MoE layer takes when the bus is not the limit.
    pub layer_secs: f64,
    /// The coefficient on the resident fraction that gives the expert hit rate.
    pub hit_coefficient: f64,
    /// Observations folded into each estimate so far. Zero samples with a
    /// confident-looking estimate means the seed, not a measurement.
    pub copy_samples: u64,
    pub layer_samples: u64,
    pub hit_samples: u64,
    /// The model's whole expert footprint, for the resident fraction.
    pub expert_total_bytes: u64,
    /// The narrowest forward the copy rate may learn from — below this a wave
    /// has not routed across a layer and its implied rate is nonsense.
    pub min_learn_rows: usize,
    /// The budget a fill opening now would reset the planner with.
    pub budget_resident_bytes: u64,
    pub budget_floor_bytes: u64,
    pub budget_max_rows: usize,
    pub budget_max_decodes: usize,
    /// The frontier gap the wave's transient tier is placed in, what the weight
    /// side is owed of it, and what is left to size a wave against.
    pub tier_gap_bytes: u64,
    pub tier_owed_bytes: u64,
    pub tier_budget_bytes: u64,
}

/// Model weights resident in VRAM, split into the two parts that behave
/// differently: the dense tensors, which are permanent, and the expert slots,
/// which page against host RAM.
///
/// Without this the dense half is invisible to every consumer of the report —
/// it sits outside the KV pool (so it is absent from `pool_used_bytes`) and its
/// [`AllocClass::Weights`] tally is never credited, so a whole-card accounting
/// could only reach it by subtraction.
#[derive(Debug, Clone, Serialize)]
pub struct WeightSection {
    /// Dense tensors: embeddings, attention projections, norms, output head.
    /// `None` when the model cannot report it (non-MoE or no expert cache).
    pub base_bytes: Option<u64>,
    /// Expert slots currently resident in VRAM — moves as experts page.
    pub resident_expert_bytes: Option<u64>,
}

/// The provenance gallery arena's VRAM slabs.
///
/// The arena allocates 16 MiB device slabs directly and holds them for its
/// lifetime, outside both the KV pool and the governor's class tallies, so it
/// appears in no other field of this report. Its pages are evictable and rebuild
/// from the substrate on demand, which is why the relief sequence sheds them
/// before it touches model KV.
#[derive(Debug, Clone, Serialize)]
pub struct GallerySection {
    /// VRAM held by the arena's slabs right now.
    pub resident_bytes: u64,
    /// Turns currently paged into the arena.
    pub resident_turns: usize,
}

/// Device + pool + governor view of the card.
#[derive(Debug, Clone, Serialize)]
pub struct VramSection {
    /// Driver-reported free/total (DXGI headroom path on WDDM).
    pub driver_free_bytes: u64,
    pub driver_total_bytes: u64,
    /// CUDA stream-ordered pool: live allocations vs OS-reserved footprint.
    /// `reserved - used` is the reuse gap — reusable for pool allocations that
    /// fit existing arenas, NOT usable for fresh arenas, and the first thing
    /// WDDM spills to host memory near the ceiling.
    pub pool_used_bytes: u64,
    pub pool_reserved_bytes: u64,
    /// Pool-aware availability (`init_free - pool_used - reserve`).
    pub budget_available_bytes: Option<u64>,
    /// Governor view — absent when no governor is installed (non-CUDA).
    pub governor: Option<GovernorSection>,
}

/// The VRAM governor's budget model.
#[derive(Debug, Clone, Serialize)]
pub struct GovernorSection {
    /// Balloon-measured resident capacity `C`.
    pub capacity_bytes: u64,
    /// Live measured headroom (honest, excludes the pool reuse gap).
    pub headroom_bytes: u64,
    /// Cushion left outside the reservation for the CUDA pool.
    ///
    /// Replaces `kv_floor_bytes`, which reported the static KV reserve. There is
    /// no static reserve now — the span holds KV, transients and experts, and
    /// the boundary between the last two moves — so the weight side's own extent
    /// is what says where the partition currently sits, and that is reported
    /// with the arena occupancy rather than here.
    pub pool_cushion_bytes: u64,
    /// Loose per-class reserved tallies (reporting, not availability gates).
    pub reserved_weights_bytes: u64,
    pub reserved_expert_bytes: u64,
    pub reserved_scratch_bytes: u64,
    pub reserved_kv_bytes: u64,
}

/// Whole-card reconciliation: what this report can account for, and what it
/// cannot.
///
/// Every other section names one consumer. None of them adds up, and that is
/// how several GiB stayed invisible: the dense weights, the gallery arena's
/// slabs and the CUDA pool each sat outside the KV pool and outside each
/// other's tallies, so no reader could have summed them without knowing to look
/// in three places and a fourth for the total. All three are inside the
/// reservation now, and this section is where that is checked rather than
/// assumed.
///
/// [`Self::outside_span_bytes`] is the number that matters. The reservation is
/// the budget; memory allocated outside it competes with it, and on WDDM the
/// loser is paged to host RAM — measured at 3.7 GiB on the 3.6-35B, which cost
/// 17x on decode with nothing anywhere saying why. Driving it to zero is the
/// goal, and this is the meter for it.
#[derive(Debug, Clone, Serialize)]
pub struct AccountingSection {
    /// The VMM reservation: pinned, non-migratable, holds KV + experts +
    /// transients. `None` without a region pool.
    pub span_bytes: Option<u64>,
    /// Ours, but NOT in the span — the demotable set.
    ///
    /// This **is** [`Self::outside_pool_bytes`], because on this device every
    /// device allocation the engine makes outside the reservation is a CUDA pool
    /// allocation: `CudaDevice::{alloc, alloc_zeros, memcpy_stod}` all reach
    /// `CudaStream::alloc`, which calls `cuMemAllocAsync` whenever the context
    /// reports `has_async_alloc`. The pool's OS-reserved footprint therefore
    /// covers the whole outside set already.
    pub outside_span_bytes: u64,
    /// The CUDA async pool's OS-reserved footprint — the whole outside set.
    pub outside_pool_bytes: u64,
    // There is deliberately no `outside_dense_bytes`. It was wrong three times,
    // and the third is the instructive one.
    //
    // First it was an ADDEND to the pool, as though the weights sat beside it
    // rather than in it — inflating the demotable set by the size of the model.
    // Then a named PART of the pool, correct until the weights moved into the
    // span, at which point it reported a component larger than the whole
    // containing it. Then a SUBTRACTION, `total_dense - inside_dense`, which
    // looked principled and was not: `total_dense` sums the raw GGUF bytes the
    // loader read, `inside_dense` sums the repacked twins that stayed, and a
    // Q4_K source is not the size of its Q4_KO twin. Subtracting two different
    // populations produced a confident 135 MiB of weights that did not exist.
    //
    // What answers the question exactly is already here: [`Self::inside_dense_bytes`]
    // is what the allocator handed out, and [`Self::outside_pool_bytes`] is what
    // the driver says the pool holds. Right after load the second is the residual
    // weight footprint — measured at 32 MiB, against 1,952 before the weights
    // moved into the span. Neither is derived from the other.
    /// The gallery arena, which is **inside** the span: its slabs are claimed
    /// regions, the same 16 MiB unit a KV arena takes.
    ///
    /// Reported here beside the outside set rather than only in
    /// [`GallerySection`] because it used to be part of it — the slabs came
    /// from the CUDA pool and were, in the arena's own words, "never returned".
    /// A reader comparing two runs needs to see that this moved rather than
    /// vanished, and a regression that put it back outside would otherwise show
    /// up only as `outside_pool_bytes` quietly growing.
    pub inside_gallery_bytes: u64,
    /// The model's dense weights, **inside** the span — the dense block, locked
    /// at the end of load.
    ///
    /// The whole point of the reservation-before-load ordering: these bytes were
    /// the single largest thing outside the span, and being outside it meant
    /// WDDM could demote the model itself to host RAM. Inside, they are pinned
    /// device allocations the driver may not migrate.
    pub inside_dense_bytes: u64,
    /// Per-sequence recurrent state, also **inside** the span: each store's
    /// buffers are carved from claimed regions.
    ///
    /// Named for the same reason [`Self::inside_gallery_bytes`] is, and more
    /// urgently — it is the largest thing in the span that is neither KV nor
    /// weights (~126 MiB per sequence on a hybrid stack, several GiB across a
    /// wide wave) and it moves with wave width, so a run that grows it and a run
    /// that leaks it look identical in the span total alone.
    pub inside_recurrent_bytes: u64,
    /// Bytes that went to the pool because their origin carried no wave ticket,
    /// **over the last wave**.
    ///
    /// A provenance break: something upstream produced a tensor with no wave
    /// backing and everything derived from it inherited the pool, so the fix is
    /// at that root — which may be many frames above whatever site a
    /// forbidden-allocation report names.
    ///
    /// Per-wave rather than cumulative, and the distinction is load-bearing. The
    /// lifetime total is dominated by declines that are *correct*: an op on the
    /// residual stream has nothing to inherit, because the residual crosses
    /// layers and belongs on the pool by design, and model loading has no wave at
    /// all. Read cumulatively this number is large, unfalsifiable, and says
    /// nothing. Scoped to a wave — where every allocation is supposed to inherit
    /// — a non-zero reading is a defect.
    pub decline_no_ticket_bytes: u64,
    /// Bytes that went to the pool because the arena had no room, over the last
    /// wave.
    ///
    /// **A sizing problem, not a provenance one.** Nothing about the call site is
    /// wrong; the wave arena was too narrow for the work. Reported beside
    /// [`Self::decline_no_ticket_bytes`] because the two land on the same
    /// `CudaDevice::alloc` and are otherwise indistinguishable — and they have
    /// opposite fixes, so a reader who cannot separate them chases the wrong one.
    pub decline_arena_full_bytes: u64,
    /// Driver-reported in use across the whole card (`total - free`).
    pub device_in_use_bytes: u64,
    /// The **expert weight zone's extent** inside the span — `span_end −
    /// weight_floor`, the ground the weight side is permitted to hold.
    ///
    /// **Capacity, not occupancy.** It counts the zone's free slots too, so it
    /// is always ≥ [`WeightSection::resident_expert_bytes`], which is what is
    /// actually loaded. The two answer different questions and the report needs
    /// both: this one says how much of the span the weight side owns (and is
    /// therefore the tenant figure that belongs beside `inside_dense_bytes` and
    /// friends), while the resident figure says how much of that ground is in
    /// use. Naming it `inside_expert_bytes` and calling it "resident" — as the
    /// first version of this field did — puts two different quantities under one
    /// heading and silently over-counts occupancy by the zone's free capacity.
    ///
    /// This is also the tenant that *moves*: the elastic boundary concedes it to
    /// the KV side under pressure, so watching it fall across a run is how zone
    /// erosion becomes visible in the report instead of only in the log.
    pub expert_zone_bytes: u64,
    /// The wave transient tier, **inside** the span — this forward's activation
    /// ground, between the KV regions and the weight zone.
    ///
    /// Transient by construction (released between forwards), so a reading here
    /// is a point-in-time sample rather than a standing cost. Named because it
    /// is the fourth span tenant and its absence made the span's own arithmetic
    /// impossible to close.
    pub inside_transient_bytes: u64,
    /// Device bytes in use before the span was reserved — the CUDA context, the
    /// driver's working set, the module images, and any other process. Measured
    /// at reservation time, which is the only moment it stands alone (the
    /// reservation precedes the weight load).
    ///
    /// Published so [`Self::unaccounted_bytes`] can mean one thing instead of
    /// three.
    ///
    /// **`None` when it could not be measured**, which is a real state: the
    /// reservation runs whether or not a VRAM gauge was registered (it branches
    /// on exactly that), so on a gauge-less run there is no baseline to take. It
    /// is an `Option` rather than a `0` because the two readings demand opposite
    /// conclusions — a genuine zero means the card was empty before us and
    /// `unaccounted` is trustworthy, while an unmeasured one means
    /// `unaccounted` has silently reverted to including the whole CUDA context,
    /// which is the 1.4 GB of noise this field exists to remove. Reporting both
    /// as `0` hides that distinction at exactly the moment a reader is trying to
    /// decide whether a residual is a leak.
    pub driver_baseline_bytes: Option<u64>,
    /// `device_in_use - (span + outside + driver_baseline)` — **an allocation
    /// this report does not know about, and nothing else** — but only when
    /// [`Self::driver_baseline_bytes`] is `Some`. With `None` the baseline could
    /// not be subtracted and this reverts to the older, inflated meaning; read
    /// that field before reading this one.
    ///
    /// It used to fold in the driver baseline too, and read 1.4 GB on a 16 GB
    /// card while being almost entirely that constant. A number that large is
    /// unreadable as a signal: a genuine few-hundred-MB leak moves it by a
    /// fraction and nobody can tell. With the baseline named separately this is
    /// meant to sit near zero, and any material reading is a defect to chase.
    pub unaccounted_bytes: i64,
}

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
}

/// One size class's share of the resident GPU arenas.
#[derive(Debug, Clone, Serialize)]
pub struct KvClassRow {
    /// Slot stride in bytes — the class's identity.
    pub slot_bytes: usize,
    pub arenas: usize,
    pub reserved_bytes: u64,
    pub live_bytes: u64,
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
}

/// The wave fill's live inputs.
#[derive(Debug, Clone, Serialize)]
pub struct AdmissionSection {
    /// Free KV bytes admission could spend right now — the free regions less
    /// the relief pass's working room.
    pub ceiling_bytes: u64,
    /// In-flight widths at capture.
    pub prefill_width: usize,
    pub section_width: usize,
    pub decode_width: usize,
    pub queued_prefills: usize,
    /// Not-yet-fed prefill tokens: the queued FIFO plus the unfed remainder of
    /// everything in flight.
    ///
    /// Published beside `queued_prefills` because a **count cannot express the
    /// work it stands for**, which is the same argument `admission` makes about
    /// sequence counts — this layer's queue holds scopes from 48 to 5,980
    /// tokens, so ten queued items is either a trickle or a flood. A producer
    /// that paces itself on depth has to pace on tokens.
    ///
    /// It is also the one queue signal that is meaningful on **every** model and
    /// card: it says nothing about VRAM, expert residency or partitioning, so a
    /// producer keyed on it behaves identically for a dense 3B on a 72 GB card
    /// and a streaming MoE on a tight one. Only the rate at which the engine
    /// drains it differs.
    pub queued_prefill_tokens: u64,
    /// Tokens completed since the process started. Cumulative rather than a
    /// rate so a consumer picks its own window and a missed report costs
    /// nothing.
    pub completed_tokens: u64,
    /// Decodes the last wave fill admitted — the conversations the engine is
    /// carrying to completion right now.
    pub decode_carried: usize,
    /// Eligible decodes the last wave fill could **not** admit: resident
    /// conversations whose KV chunk or model state the allocators refused.
    ///
    /// **The engine's "full" signal, and the one a producer paces on.** It is
    /// the real allocation's answer, not an estimate — a decode is refused only
    /// when the device has no ground for it — so it means the same thing on
    /// every model and card: non-zero, and one more open conversation would
    /// wait behind ones that already cannot step; zero, and everything resident
    /// is being carried. A producer that opens only at zero cannot pile
    /// conversations onto a device that has stopped stepping the ones it has,
    /// which is the wedge this field exists to close.
    pub decode_starved: usize,
    /// Conversation slots open on the engine right now — every sequence the
    /// scheduler holds a conversation for, whoever opened it.
    ///
    /// A producer pacing on this report opens conversations faster than the
    /// report can reflect them: the backlog it reads is a wave behind, so a
    /// pool of 96 workers passed a gate that read "empty" 96 times in one
    /// second. This lets the gate count what it has opened that the engine
    /// has not yet seen, and hold until the engine catches up.
    pub open_slots: usize,
    /// Sequences a wave carries right now — the engine's width, which follows
    /// its expert hit rate. The producer paces its queue and its open count on
    /// this rather than on a constant: a wave's worth of work waiting is what
    /// keeps every wave full, and more is K/V waiting.
    pub wave_width: usize,
    /// Milliseconds between this report and the one before it — the engine's
    /// publishing cadence, one wave. A reader deciding whether a report is
    /// stale compares its age to this, not to a constant: a wave here runs
    /// three to five seconds and a five-second constant read a report a wave
    /// old as stale for 95 fills of one run, holding the producer for nothing.
    pub publish_interval_ms: u64,
    /// The weight zone as it stands, in bytes, and the hold beneath which the
    /// model would start streaming its experts.
    ///
    /// **The one resource a producer must not spend.** Every open conversation
    /// takes K/V, the KV side and the weight side share one elastic span, so
    /// enough of them push the boundary down and evict the resident experts —
    /// and an engine that streams its experts is slower at everything,
    /// including finishing the conversations that would give the ground back.
    ///
    /// Published so the producer can read the real quantity instead of a
    /// proxy for it. Counting open conversations against a width was the proxy
    /// tried before this, and it throttled a run for 409 seconds while the
    /// zone it was meant to protect never came near its hold.
    pub weight_zone_bytes: u64,
    pub weight_hold_bytes: u64,
}

/// MoE expert residency.
#[derive(Debug, Clone, Serialize)]
pub struct ExpertSection {
    /// Host-pinned expert pool (`cuMemAllocHost`) — non-pageable, structural.
    pub host_pinned_bytes: u64,
    /// VRAM-resident expert slots (governor class tally).
    pub vram_reserved_bytes: u64,
}

impl LatencySection {
    /// The share of the window the bus must have been busy for bandwidth to be
    /// the binding cost.
    ///
    /// Half. Below it the bus spent most of the window idle, so whatever the
    /// engine was waiting on, it was not the width of the pipe — and calling it
    /// bandwidth-bound sends the next fix (more residency, wider waves) in a
    /// direction that cannot help. Above it the transfers alone account for
    /// most of the elapsed time and nothing else needs to be true.
    const BANDWIDTH_SHARE: f64 = 0.5;

    /// Round trips per second past which the engine is paying more in
    /// per-miss latency than the bytes can explain.
    ///
    /// Derived rather than picked: a miss costs at least one PCIe round trip,
    /// and at the ~10 µs order that a small transfer's launch-and-fence costs
    /// on this class of link, 20,000 of them is 0.2 s of every second spent in
    /// round trips regardless of how few bytes moved. That is the point where
    /// the count, not the volume, is the thing to fix.
    const LATENCY_MISSES_PER_S: f64 = 20_000.0;

    /// Classify the window.
    ///
    /// Order matters: bandwidth is checked first because a saturated bus also
    /// produces stalls — everything waits when the pipe is full — and reporting
    /// that as latency would point at the predictor when the answer is the
    /// link. Latency is only claimed when the bus was demonstrably *not* busy.
    fn classify(bus_utilisation: f64, misses_per_s: f64, late_fraction: f64, ran: bool) -> Bound {
        if !ran {
            return Bound::Idle;
        }
        if bus_utilisation >= Self::BANDWIDTH_SHARE {
            return Bound::Bandwidth;
        }
        // A high round-trip rate, or prefetches that keep arriving late, with
        // the bus idle: the engine is waiting on the *number* of transfers.
        if misses_per_s >= Self::LATENCY_MISSES_PER_S || late_fraction >= 0.25 {
            return Bound::Latency;
        }
        Bound::Compute
    }
}

/// Latest published report plus its capture `Instant` (for age).
static LATEST: OnceLock<Mutex<Option<(MemoryReport, Instant)>>> = OnceLock::new();

fn slot() -> &'static Mutex<Option<(MemoryReport, Instant)>> {
    LATEST.get_or_init(|| Mutex::new(None))
}

/// Store `report` as the latest snapshot.
pub fn publish(report: MemoryReport) {
    *slot().lock().unwrap() = Some((report, Instant::now()));
}

/// The latest snapshot and its age in milliseconds, if one has been published.
pub fn latest() -> Option<(MemoryReport, u64)> {
    slot()
        .lock()
        .unwrap()
        .as_ref()
        .map(|(r, at)| (r.clone(), at.elapsed().as_millis() as u64))
}

impl Scheduler {
    /// Build and publish the full memory report, and log it as one JSON line.
    ///
    /// Called at the wave-telemetry cadence (~2 s). The ceiling read performs
    /// one device query — the same cost an admission pass already pays — so
    /// this adds no new class of work to the loop.
    ///
    /// Logged at TRACE: it fires every wave and the line is a full JSON dump of
    /// every arena, so at DEBUG it drowns out everything else in the log. The
    /// report is always published to [`publish`] regardless of level, so
    /// `/v1/memory` and the perf view keep their data whether or not anyone is
    /// capturing the line.
    pub(super) fn publish_memory_report(&mut self) {
        let report = self.build_memory_report();
        self.last_report_publish = Some(std::time::Instant::now());
        match serde_json::to_string(&report) {
            Ok(json) => tracing::trace!(
                target: "candle_conversation::scheduler::memory",
                %json,
                "memory report"
            ),
            Err(e) => tracing::warn!("memory report serialization failed: {e}"),
        }
        publish(report);
    }

    fn build_memory_report(&mut self) -> MemoryReport {
        let captured_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // ── VRAM ────────────────────────────────────────────────────────────
        let vram = self.session.vram_free_total().map(|(free, total)| {
            let (pool_used, pool_reserved) = self.session.vram_pool_stats().unwrap_or((0, 0));
            let governor = self.session.vram_governor().map(|gov| GovernorSection {
                capacity_bytes: gov.capacity(),
                headroom_bytes: gov.measure().map(|r| r.headroom).unwrap_or(0),
                pool_cushion_bytes: gov.pool_cushion(),
                reserved_weights_bytes: gov.class_reserved(AllocClass::Weights),
                reserved_expert_bytes: gov.class_reserved(AllocClass::Expert),
                reserved_scratch_bytes: gov.class_reserved(AllocClass::Scratch),
                reserved_kv_bytes: gov.class_reserved(AllocClass::Kv),
            });
            VramSection {
                driver_free_bytes: free as u64,
                driver_total_bytes: total as u64,
                pool_used_bytes: pool_used as u64,
                pool_reserved_bytes: pool_reserved as u64,
                budget_available_bytes: self.session.vram_budget_available().map(|b| b as u64),
                governor,
            }
        });

        // ── KV arenas ───────────────────────────────────────────────────────
        let classes = self
            .session
            .kv_gpu_class_stats()
            .map(|cs| {
                cs.classes
                    .iter()
                    .filter(|c| c.arenas > 0)
                    .map(|c| KvClassRow {
                        slot_bytes: c.slot_bytes,
                        arenas: c.arenas,
                        reserved_bytes: c.reserved_bytes as u64,
                        live_bytes: c.live_bytes as u64,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let arenas = global_arena_memory_report()
            .into_iter()
            .map(|(backing, format, arenas, bytes)| ArenaRow {
                backing,
                format,
                arenas,
                bytes: bytes as u64,
            })
            .collect();
        let kv = KvSection { classes, arenas };

        // ── Warm tier ───────────────────────────────────────────────────────
        let warm = WarmSection {
            resident_count: self.persist_trigger.warm_resident_count() as usize,
            pending_warm_bytes: self.persist_trigger.pending_warm_bytes(),
            pinned_undrainable_bytes: self.persist_trigger.pinned_undrainable_bytes(),
            resident_bytes: self.persist_trigger.warm_resident_bytes(),
        };

        // ── Host RAM ────────────────────────────────────────────────────────
        // `warm_over_budget` refreshes the cached probe (≤1 Hz syscall) and IS
        // the ingest throttle's predicate — report exactly what it decided on.
        let over_budget = self.warm_over_budget();
        let (available, total) = self
            .host_ram_probe
            .map(|(_, a, t)| (a, t))
            .unwrap_or((0, 0));
        let budget = candle::vram::host_ram_budget(total);
        let warm_usage = self
            .persist_trigger
            .warm_resident_bytes()
            .saturating_add(self.persist_trigger.pending_warm_bytes());
        let perf = candle::vram::host_perf();
        let host = HostSection {
            total_bytes: total,
            available_bytes: available,
            buffer_bytes: budget.buffer_bytes,
            weights_reserved_bytes: budget.weights_reserved_bytes,
            weights_capped: budget.weights_capped,
            kv_warm_budget_bytes: budget.kv_warm_budget_bytes,
            pipeline_slack_bytes: super::prefill::WARM_PIPELINE_SLACK_BYTES,
            warm_usage_bytes: warm_usage,
            over_budget,
            pages_in_per_sec: candle::vram::pages_in_per_sec(),
            commit_total_bytes: perf.map(|p| p.commit_total_bytes),
            commit_limit_bytes: perf.map(|p| p.commit_limit_bytes),
        };

        // ── Admission ───────────────────────────────────────────────────────
        let admission = AdmissionSection {
            ceiling_bytes: self.admit_budget_ceiling(),
            prefill_width: self.prefill_width(),
            section_width: self.section_ingest_width(),
            decode_width: self.decode_width(),
            queued_prefills: self.prefill_queue.len(),
            queued_prefill_tokens: self.pending_prefill_tokens(),
            completed_tokens: super::PREFILL_OK_TOKENS.load(std::sync::atomic::Ordering::Relaxed),
            // An ungated fill (no weight zone) carries every eligible decode.
            decode_carried: self
                .wave_decode_set
                .as_ref()
                .map_or_else(|| self.decode_width(), Vec::len),
            decode_starved: self.wave_decode_starved,
            open_slots: self.slot_conversations.len(),
            wave_width: self.wave_width,
            publish_interval_ms: self
                .last_report_publish
                .map_or(0, |t| t.elapsed().as_millis() as u64),
            // The live-region view, which is what the fill's own residency stop
            // reads — not the extent, which includes ground a standing tier has
            // already bought and would read as headroom that is not there.
            // The reported zone is the identity at its floor reserve — what the
            // weight side could hold with no wave standing. A report of the
            // partition, not of one wave's moment.
            weight_zone_bytes: super::interleave::effective_weight_zone_bytes(0).unwrap_or(0),
            weight_hold_bytes: super::interleave::optimal_weight_bytes().unwrap_or(0),
        };

        // ── Experts ─────────────────────────────────────────────────────────
        let experts = ExpertSection {
            host_pinned_bytes: host_pinned_bytes(),
            vram_reserved_bytes: self
                .session
                .vram_governor()
                .map(|g| g.class_reserved(AllocClass::Expert))
                .unwrap_or(0),
        };

        // ── Model weights ───────────────────────────────────────────────────
        // `resident_weight_bytes` is dense + resident experts; the expert half
        // is already reported above, so split it back out rather than double
        // count it in a whole-card sum.
        let resident_expert_bytes = self
            .model
            .expert_stats()
            .map(|s| s.resident_vram_bytes as u64);
        let weights = WeightSection {
            base_bytes: self
                .model
                .resident_weight_bytes()
                .map(|total| (total as u64).saturating_sub(resident_expert_bytes.unwrap_or(0))),
            resident_expert_bytes,
        };

        // ── Gallery arena ───────────────────────────────────────────────────
        let gallery = GallerySection {
            resident_bytes: self
                .gallery_arena
                .as_ref()
                .map_or(0, |a| a.resident_bytes()),
            resident_turns: self
                .gallery_arena
                .as_ref()
                .map_or(0, |a| a.resident_turns()),
        };

        // ── Whole-card reconciliation ───────────────────────────────────────
        //
        // What lives outside the reservation, totalled HERE because nothing else
        // does. Each part is already reported above; the point is the total, and
        // the residual after it.
        //
        // **The pool IS the outside set.** Every device allocation the engine
        // makes outside the span goes through `CudaDevice::{alloc, alloc_zeros,
        // memcpy_stod}` → `CudaStream::alloc` → `cuMemAllocAsync`, so the pool's
        // reserved footprint already covers all of it. The dense weights are no
        // longer among them — they are carved from the span at load — so nothing
        // is added on top of this figure.
        // What the dense block actually handed out — asked of the allocator, not
        // inferred from the model's own byte count. The two measure different
        // populations (raw checkpoint bytes read versus repacked twins kept), so
        // deriving one from the other is what produced a confident figure for
        // weights that did not exist.
        let inside_dense_bytes = match &self.device {
            Device::Cuda(cuda) => {
                candle_nn::kv_cache::dense_bytes(&cuda.cuda_stream()).unwrap_or(0) as u64
            }
            _ => 0,
        };
        let outside_pool_bytes = vram.as_ref().map_or(0, |v| v.pool_reserved_bytes);
        let outside_span_bytes = outside_pool_bytes;
        // Inside the span — claimed regions, so already counted in `span_bytes`
        // and deliberately NOT added to the outside set.
        let inside_gallery_bytes = gallery.resident_bytes;
        let inside_recurrent_bytes = self.model.recurrent_reserved_bytes() as u64;
        // Why the pool was reached at all, split by the two causes that have
        // opposite fixes — over the last wave, not the run. See the field docs
        // for why the lifetime totals answer nothing.
        let (decline_no_ticket_bytes, decline_arena_full_bytes) = last_wave_declines();
        let span_bytes = match self.device.location() {
            candle::DeviceLocation::Cuda { gpu_id } => candle_nn::kv_cache::span_layout(gpu_id)
                .map(|l| l.span_end.saturating_sub(l.span_base)),
            _ => None,
        };
        let device_in_use_bytes = vram.as_ref().map_or(0, |v| {
            v.driver_total_bytes.saturating_sub(v.driver_free_bytes)
        });
        // Captured at reservation time by the region pool — see
        // `RegionStats::pre_reservation_in_use_bytes`. Without subtracting it,
        // `unaccounted` reports the driver's own footprint as if it were a leak.
        // One `region_stats` read for all three span figures the pool owns: the
        // driver baseline it captured at reservation, the weight zone, and the
        // standing transient tier.
        // The pool records `None` when no VRAM gauge was registered at
        // reservation, which is a different answer from "the card was empty"
        // and has to survive to the reader — see `driver_baseline_bytes`.
        let (driver_baseline_bytes, expert_zone_bytes, inside_transient_bytes) =
            match self.device.location() {
                candle::DeviceLocation::Cuda { gpu_id } => {
                    candle_nn::kv_cache::region_stats(gpu_id).map_or((None, 0, 0), |s| {
                        (
                            s.pre_reservation_in_use_bytes,
                            s.weight_bytes as u64,
                            s.transient_bytes as u64,
                        )
                    })
                }
                _ => (None, 0, 0),
            };
        let accounting = AccountingSection {
            span_bytes,
            outside_span_bytes,
            outside_pool_bytes,
            inside_dense_bytes,
            inside_gallery_bytes,
            inside_recurrent_bytes,
            expert_zone_bytes,
            inside_transient_bytes,
            decline_no_ticket_bytes,
            decline_arena_full_bytes,
            device_in_use_bytes,
            driver_baseline_bytes,
            unaccounted_bytes: device_in_use_bytes as i64
                - span_bytes.unwrap_or(0) as i64
                - outside_span_bytes as i64
                - driver_baseline_bytes.unwrap_or(0) as i64,
        };

        // ── the planner ─────────────────────────────────────────────────────
        // Read through the same `Ground` the fill uses, so what is published is
        // what the next admission pass will actually open with rather than a
        // reconstruction of it.
        let optimal = super::interleave::optimal_weight_bytes().unwrap_or(0);
        let budget = {
            use super::admit::Ground;
            super::prefill::WaveFill::new(self, optimal).budget()
        };
        let (tier_gap, tier_owed, tier_budget) = self.tier_budget_now();
        let r = &self.wave_rate;
        let planner = PlannerSection {
            copy_bytes_per_s: r.effective_bytes_per_s(),
            link_bytes_per_s: r.link_bytes_per_s(),
            layer_secs: r.layer_secs(),
            hit_coefficient: r.hit_rate(),
            copy_samples: r.samples(),
            layer_samples: r.decode_samples(),
            hit_samples: r.hit_samples(),
            expert_total_bytes: r.expert_total_bytes(),
            min_learn_rows: r.min_learn_rows(),
            budget_resident_bytes: budget.resident,
            budget_floor_bytes: budget.floor,
            budget_max_rows: budget.max_rows,
            budget_max_decodes: budget.max_decodes,
            tier_gap_bytes: tier_gap as u64,
            tier_owed_bytes: tier_owed as u64,
            tier_budget_bytes: tier_budget as u64,
        };

        // ── what the window waited on ───────────────────────────────────────
        // Deltas against the last report, so the section describes the interval
        // rather than an average since process start that buries the moment a
        // regime changed.
        let now = std::time::Instant::now();
        let window_ms = self
            .last_latency_at
            .map(|t| now.duration_since(t).as_millis() as u64)
            .unwrap_or(0);
        self.last_latency_at = Some(now);
        let stats = self.model.expert_stats();
        let slot_bytes = stats.as_ref().map_or(0, |s| s.slot_bytes) as u64;
        let d = |now: usize, was: &mut usize| -> u64 {
            let delta = now.saturating_sub(*was) as u64;
            *was = now;
            delta
        };
        let (hits, misses, cold, warm_loads, fences, late) = match stats.as_ref() {
            Some(s) => (
                d(s.expert_hits, &mut self.latency_prev.hits),
                d(s.expert_misses, &mut self.latency_prev.misses),
                d(s.cold_loads, &mut self.latency_prev.cold),
                d(s.warm_loads, &mut self.latency_prev.warm),
                d(s.fence_stalls, &mut self.latency_prev.fences),
                d(s.late_loads, &mut self.latency_prev.late),
            ),
            None => (0, 0, 0, 0, 0, 0),
        };
        // A miss is one expert slot crossing the bus.
        let transfer_bytes = misses.saturating_mul(slot_bytes);
        let secs = (window_ms as f64) / 1000.0;
        let achieved = if secs > 0.0 {
            transfer_bytes as f64 / secs
        } else {
            0.0
        };
        let link = self.wave_rate.link_bytes_per_s();
        let bus_utilisation = if link > 0.0 { achieved / link } else { 0.0 };
        let misses_per_s = if secs > 0.0 {
            misses as f64 / secs
        } else {
            0.0
        };
        let late_fraction = if misses > 0 {
            late as f64 / misses as f64
        } else {
            0.0
        };
        let latency = LatencySection {
            window_ms,
            transfer_bytes,
            achieved_bytes_per_s: achieved,
            link_bytes_per_s: link,
            bus_utilisation,
            expert_hits: hits,
            expert_misses: misses,
            cold_loads: cold,
            warm_loads,
            fence_stalls: fences,
            late_loads: late,
            misses_per_s,
            late_fraction,
            bound: LatencySection::classify(
                bus_utilisation,
                misses_per_s,
                late_fraction,
                secs > 0.0 && (hits + misses) > 0,
            ),
        };

        MemoryReport {
            captured_unix_ms,
            vram,
            kv,
            warm,
            host,
            admission,
            experts,
            weights,
            gallery,
            accounting,
            planner,
            holdings: super::holdings::latest(),
            latency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A saturated bus is reported as bandwidth even though it also stalls.**
    /// Everything waits when the pipe is full, so a classifier that checked
    /// stalls first would blame the predictor for a link that is simply busy —
    /// and send the next fix at prefetch depth instead of at residency.
    #[test]
    fn a_busy_bus_is_bandwidth_bound_whatever_else_it_looks_like() {
        // Bus at 80%, and stalling hard on top of it.
        assert_eq!(
            LatencySection::classify(0.80, 500_000.0, 0.9, true),
            Bound::Bandwidth,
        );
        // Exactly at the share is still bandwidth.
        assert_eq!(
            LatencySection::classify(LatencySection::BANDWIDTH_SHARE, 0.0, 0.0, true),
            Bound::Bandwidth,
        );
    }

    /// **Latency is an idle bus that is nonetheless waited on.** Many small
    /// round trips, few bytes — the signature of a working set that has
    /// outgrown what the cache holds and the predictor stays ahead of.
    #[test]
    fn many_round_trips_over_an_idle_bus_is_latency_bound() {
        assert_eq!(
            LatencySection::classify(0.05, LatencySection::LATENCY_MISSES_PER_S, 0.0, true),
            Bound::Latency,
        );
        // Or prefetches that keep arriving too late to use, at any miss rate.
        assert_eq!(
            LatencySection::classify(0.05, 100.0, 0.30, true),
            Bound::Latency,
        );
    }

    /// **Neither** — the bus is quiet and nothing is waiting on it, so the time
    /// went into arithmetic. The regime widening a wave actually helps.
    #[test]
    fn a_quiet_bus_with_few_misses_is_compute_bound() {
        assert_eq!(
            LatencySection::classify(0.10, 2_000.0, 0.01, true),
            Bound::Compute,
        );
    }

    /// A window in which nothing routed says so, rather than reporting the
    /// regime of an engine that was not running.
    #[test]
    fn a_window_with_no_work_is_idle() {
        assert_eq!(LatencySection::classify(0.0, 0.0, 0.0, false), Bound::Idle);
    }

    /// The three working regimes are genuinely distinguishable from the same
    /// two axes — which is the whole point, since throughput alone cannot tell
    /// compute from latency.
    #[test]
    fn the_three_regimes_are_separable() {
        let cases = [
            (0.9, 1_000.0, 0.0, Bound::Bandwidth),
            (0.02, 80_000.0, 0.5, Bound::Latency),
            (0.02, 100.0, 0.0, Bound::Compute),
        ];
        for (bus, misses, late, want) in cases {
            assert_eq!(
                LatencySection::classify(bus, misses, late, true),
                want,
                "bus={bus} misses/s={misses} late={late}",
            );
        }
    }

    fn sample() -> MemoryReport {
        MemoryReport {
            captured_unix_ms: 1_700_000_000_000,
            vram: Some(VramSection {
                driver_free_bytes: 1,
                driver_total_bytes: 2,
                pool_used_bytes: 3,
                pool_reserved_bytes: 4,
                budget_available_bytes: Some(5),
                governor: Some(GovernorSection {
                    capacity_bytes: 6,
                    headroom_bytes: 7,
                    pool_cushion_bytes: 8,
                    reserved_weights_bytes: 11,
                    reserved_expert_bytes: 12,
                    reserved_scratch_bytes: 13,
                    reserved_kv_bytes: 14,
                }),
            }),
            kv: KvSection {
                classes: vec![KvClassRow {
                    slot_bytes: 1152,
                    arenas: 1,
                    reserved_bytes: 2,
                    live_bytes: 3,
                }],
                arenas: vec![ArenaRow {
                    backing: 0,
                    format: "R16".into(),
                    arenas: 2,
                    bytes: 256,
                }],
            },
            warm: WarmSection {
                resident_count: 7,
                pending_warm_bytes: 8,
                pinned_undrainable_bytes: 9,
                resident_bytes: 10,
            },
            host: HostSection {
                total_bytes: 32,
                available_bytes: 2,
                buffer_bytes: 4,
                weights_reserved_bytes: 17,
                weights_capped: false,
                kv_warm_budget_bytes: 0,
                pipeline_slack_bytes: 1,
                warm_usage_bytes: 0,
                over_budget: false,
                pages_in_per_sec: Some(12.5),
                commit_total_bytes: Some(30),
                commit_limit_bytes: Some(40),
            },
            admission: AdmissionSection {
                ceiling_bytes: 11,
                prefill_width: 1,
                section_width: 2,
                decode_width: 3,
                queued_prefills: 4,
                queued_prefill_tokens: 4096,
                completed_tokens: 1_000_000,
                decode_carried: 3,
                decode_starved: 0,
                open_slots: 7,
                wave_width: 8,
                publish_interval_ms: 3_000,
                weight_zone_bytes: 7 << 30,
                weight_hold_bytes: 5 << 30,
            },
            experts: ExpertSection {
                host_pinned_bytes: 11_000_000_000,
                vram_reserved_bytes: 5_100_000_000,
            },
            weights: WeightSection {
                // OCCUPANCY, and so strictly inside the zone extent
                // (`accounting.expert_zone_bytes` below) that holds it. A
                // fixture with more resident experts than zone to put them in
                // describes a machine that cannot exist, and would let a
                // regression that wires both fields to the zone pass.
                base_bytes: Some(1_100_000_000),
                resident_expert_bytes: Some(5_100_000_000),
            },
            gallery: GallerySection {
                resident_bytes: 268_435_456,
                resident_turns: 42,
            },
            // The planner as run 22 had learned it: a ~15 GB/s effective copy
            // rate under a 25 GB/s link, an 18 ms layer, and the hit
            // coefficient the expert cache's counters imply at ~39% residency.
            planner: PlannerSection {
                copy_bytes_per_s: 15.0e9,
                link_bytes_per_s: 25.0e9,
                layer_secs: 18.5e-3,
                hit_coefficient: 1.65,
                copy_samples: 120,
                layer_samples: 40,
                hit_samples: 300,
                expert_total_bytes: 20_292_042_752,
                min_learn_rows: 126,
                budget_resident_bytes: 5_600 << 20,
                budget_floor_bytes: 5_385 << 20,
                budget_max_rows: 450,
                budget_max_decodes: 10,
                tier_gap_bytes: 366 << 20,
                tier_owed_bytes: 0,
                tier_budget_bytes: 302 << 20,
            },
            holdings: None,
            // A window that saw a busy bus: 8 GB of expert traffic in 2 s
            // against a 25 GB/s link is 16% utilisation, which with a low
            // round-trip rate is the compute regime.
            latency: LatencySection {
                window_ms: 2_000,
                transfer_bytes: 8_000_000_000,
                achieved_bytes_per_s: 4.0e9,
                link_bytes_per_s: 25.0e9,
                bus_utilisation: 0.16,
                expert_hits: 120_000,
                expert_misses: 4_138,
                cold_loads: 12,
                warm_loads: 4_126,
                fence_stalls: 30,
                late_loads: 40,
                misses_per_s: 2_069.0,
                late_fraction: 0.0097,
                bound: Bound::Compute,
            },
            // Shaped like the measured 3.6-35B so the sums mean something: a
            // 64 GiB span holding the model, a pool reduced to per-wave churn,
            // and ~2 GiB the driver's own context holds that we cannot name.
            //
            // The pool figure is the one that carries the intent. It was 4 GiB
            // when the weights loaded into it; the whole reservation-before-load
            // change is what took it to a few hundred MiB, and a fixture still
            // shaped around the old number would let the containment assertions
            // below pass while asserting the opposite of the design.
            accounting: AccountingSection {
                span_bytes: Some(68_719_476_736),
                // The pool IS the outside set — but the weights are no longer
                // in it.
                outside_span_bytes: 234_881_024,
                outside_pool_bytes: 234_881_024,
                // The model, inside the span.
                inside_dense_bytes: 1_914_699_776,
                // Also inside, so neither may appear in the outside sum.
                inside_gallery_bytes: 268_435_456,
                // 16 sequences of hybrid recurrent state.
                inside_recurrent_bytes: 2_113_929_216,
                // The elastic weight zone — the largest span tenant on a MoE
                // model, and the one that shrinks as the boundary concedes.
                // The zone's EXTENT — deliberately larger than
                // `weights.resident_expert_bytes` above, which is what is
                // actually loaded into it.
                expert_zone_bytes: 5_939_691_520,
                // One forward's activation ground, released between waves.
                inside_transient_bytes: 67_108_864,
                // One wave's worth: provenance breaks dominating, the arena
                // itself never refusing — the shape measured on the 3.6-35B.
                decline_no_ticket_bytes: 205_959_852,
                decline_arena_full_bytes: 0,
                device_in_use_bytes: 71_015_942_144,
                // The context + driver working set, measured before the span was
                // reserved. Naming it is what leaves `unaccounted` meaning only
                // "an allocation the report does not know about".
                driver_baseline_bytes: Some(1_900_000_000),
                unaccounted_bytes: 71_015_942_144i64
                    - 68_719_476_736i64
                    - 234_881_024i64
                    - 1_900_000_000i64,
            },
        }
    }

    /// The JSON shape is the API contract — every section and the fields other
    /// tooling greps for must be present by name.
    #[test]
    fn report_serializes_with_every_section() {
        let json = serde_json::to_string(&sample()).unwrap();
        for key in [
            "captured_unix_ms",
            "driver_free_bytes",
            "pool_reserved_bytes",
            "headroom_bytes",
            "\"arenas\"",
            "pending_warm_bytes",
            "pinned_undrainable_bytes",
            "available_bytes",
            "kv_warm_budget_bytes",
            "weights_reserved_bytes",
            "warm_usage_bytes",
            "over_budget",
            "ceiling_bytes",
            "host_pinned_bytes",
            "pages_in_per_sec",
            "commit_limit_bytes",
            "resident_bytes",
            // The two consumers a whole-card sum could otherwise only reach by
            // subtraction: dense weights sit outside the KV pool, and the
            // gallery arena allocates outside the pool AND the class tallies.
            "base_bytes",
            "resident_expert_bytes",
            "resident_turns",
            // The reconciliation. `outside_span_bytes` is the meter for the
            // demotable set — the thing that cost 17x on decode while every
            // individual section read as healthy.
            "outside_span_bytes",
            "outside_pool_bytes",
            "inside_dense_bytes",
            "inside_gallery_bytes",
            "inside_recurrent_bytes",
            "decline_no_ticket_bytes",
            "decline_arena_full_bytes",
            "unaccounted_bytes",
            "span_bytes",
        ] {
            assert!(json.contains(key), "missing {key} in {json}");
        }
    }

    /// The named parts must be **contained** in the total, not summed into it.
    ///
    /// Trivial arithmetic, and worth a gate precisely because the bug it guards
    /// was arithmetic somebody performed wrongly. The first version of this test
    /// asserted `outside_span == dense + pool` and passed for exactly as long as
    /// the code made the same mistake: the dense weights are allocated through
    /// `CudaDevice::alloc` → `cuMemAllocAsync`, so they are *in* the pool, and
    /// adding them to it counted the model twice — 4,746 MiB of demotable memory
    /// reported where 2,784 MiB existed, with the difference silently deducted
    /// from `unaccounted_bytes`.
    ///
    /// A sum test cannot catch that; a containment test can, which is why the
    /// assertion is now an inequality on each part.
    #[test]
    fn the_outside_span_parts_are_contained_in_the_whole() {
        let r = sample();
        let a = &r.accounting;
        assert_eq!(
            a.outside_span_bytes, a.outside_pool_bytes,
            "the CUDA pool is the whole outside set: every non-span device \
             allocation goes through `cuMemAllocAsync`",
        );
        assert!(
            a.inside_dense_bytes > a.outside_pool_bytes,
            "the span must hold the model, and hold more of it than the pool holds \
             of anything ({} in the span, {} in the whole pool) — this is the point \
             of claiming the reservation before the load, and a fixture that did \
             not exercise it would assert nothing",
            a.inside_dense_bytes,
            a.outside_pool_bytes,
        );
        assert!(
            a.inside_gallery_bytes > 0,
            "the fixture must exercise a gallery that holds something, or this \
             says nothing about where its bytes are counted",
        );
        assert!(
            a.inside_recurrent_bytes > 0,
            "likewise the recurrent state — it is the largest in-span tenant that \
             is neither KV nor weights, and a zero fixture would not exercise the \
             one thing this section is for",
        );
        // The in-span tenants are named PARTS of the span, never addends to it.
        // Same containment rule as the dense weights against the pool, in the
        // other half of the card: the two halves fail the same way, by summing
        // a component with the total that already includes it.
        let named_in_span = a.inside_gallery_bytes
            + a.inside_recurrent_bytes
            + a.inside_dense_bytes
            + a.expert_zone_bytes
            + a.inside_transient_bytes;
        assert!(
            named_in_span < a.span_bytes.expect("the fixture reserves a span"),
            "the gallery, recurrent state, dense weights, expert zone and transient \
             tier are tenants OF the span ({named_in_span} of {:?}), so they must fit \
             inside it",
            a.span_bytes,
        );
        // **Every large span tenant is reported.** Deliberately not a coverage
        // ratio: the span is a *reservation*, and its unclaimed KV regions are
        // legitimately empty, so the tenants can sum to a fraction of it and
        // nothing is wrong. What can go wrong is a tenant having no field at
        // all — the expert zone was exactly that, the biggest one on a MoE model
        // and entirely absent, which left `dense + recurrent + gallery` several
        // gigabytes short of the truth with no way to tell that from a leak.
        //
        // So the check is presence, not proportion: each named tenant must
        // actually be populated, and a regression that quietly stops reading one
        // (a renamed `region_stats` field, say) trips here rather than showing up
        // as memory that vanished.
        for (name, v) in [
            ("dense", a.inside_dense_bytes),
            ("gallery", a.inside_gallery_bytes),
            ("recurrent", a.inside_recurrent_bytes),
            ("expert zone", a.expert_zone_bytes),
            ("transient", a.inside_transient_bytes),
        ] {
            assert!(
                v > 0,
                "span tenant {name:?} reports zero — the fixture exercises every \
                 tenant, so a zero here means the field is no longer being read",
            );
        }
        // **The zone is capacity; the resident figure is occupancy.** Two
        // different questions, and the report answers both — the first version
        // of `expert_zone_bytes` read the zone extent while documenting itself
        // as "resident expert weights", which filed capacity under an occupancy
        // heading and over-counted the span's tenants by the zone's free slots.
        //
        // The invariant is containment, `resident <= zone`, and deliberately not
        // a strict `<`: a fully-packed zone reports the two as EQUAL, and that
        // is the healthy steady state, not a defect — measured 6,724 MiB for
        // both on a loaded 3.6-35B, because the elastic boundary grows the
        // weight zone to fit exactly what loads into it.
        //
        // The fixture models a partially-packed zone (5.1 GB resident in 5.94 GB
        // of ground) so the gap between the two is real here and a reading that
        // collapses them is visible.
        let resident = r
            .weights
            .resident_expert_bytes
            .expect("the fixture reports resident experts");
        assert!(
            resident <= a.expert_zone_bytes,
            "resident experts ({resident}) must fit in the zone that holds them \
             ({}) — occupancy above capacity means one of the two is reading the \
             wrong source",
            a.expert_zone_bytes,
        );
        assert!(
            resident < a.expert_zone_bytes,
            "this FIXTURE is meant to have free slots in the zone, so that the \
             containment check above is exercised rather than trivially met by \
             equality; if the fixture changed, fix the fixture",
        );
        // The residual is the card minus every total we can name — the span, the
        // pool, and the driver's own baseline — and specifically NOT minus the
        // parts *within* those totals, which would double-count.
        //
        // The baseline belongs in this subtraction and used to be missing, which
        // published the driver's footprint as if it were an unexplained
        // allocation: 1.4 GB of "unaccounted" on a 16 GB card that was almost
        // entirely context. A residual that large cannot be read as a signal —
        // a real leak moves it by a fraction of itself.
        //
        // `expect` rather than `unwrap_or(0)`: on a fixture that names a baseline
        // the two spellings agree, so a silent default here would let the field
        // regress to `None` — the state where the subtraction stops happening and
        // the residual quietly means the old thing again — without failing.
        let baseline = a
            .driver_baseline_bytes
            .expect("the fixture measures a driver baseline");
        assert_eq!(
            a.unaccounted_bytes,
            a.device_in_use_bytes as i64
                - a.span_bytes.unwrap_or(0) as i64
                - a.outside_span_bytes as i64
                - baseline as i64,
            "the residual must be what the card holds minus everything we can name",
        );
    }

    /// `latest` hands back what `publish` stored, with a sane age.
    #[test]
    fn publish_then_latest_round_trips() {
        publish(sample());
        let (got, age_ms) = latest().expect("just published");
        assert_eq!(got.captured_unix_ms, 1_700_000_000_000);
        assert_eq!(got.experts.host_pinned_bytes, 11_000_000_000);
        assert!(age_ms < 60_000, "freshly published, age {age_ms}ms");
    }
}
