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
use candle_nn::kv_cache::global_arena_memory_report;
use serde::Serialize;

use super::admission::admit_quantum;
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
/// slabs and the CUDA pool are each documented as sitting outside the KV pool
/// and outside each other's tallies, so no reader could have summed them
/// without knowing to look in three places and a fourth for the total.
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
    /// The dense model tensors, which are **inside** [`Self::outside_pool_bytes`]
    /// rather than beside it: they are loaded through the same `alloc` path as
    /// everything else, so their bytes are pool bytes.
    ///
    /// Reported as a named component for the same reason
    /// [`Self::inside_gallery_bytes`] is — a reader needs to see which part of
    /// the pool is the permanent model and which part is per-wave churn, and
    /// moving the weights into the span must show up as this shrinking rather
    /// than as an unexplained drop in the total.
    ///
    /// **It used to be an addend**, summed with the pool as though the two were
    /// disjoint. That inflated the demotable set by the size of the model —
    /// 1,962 MiB of 4,746 reported — and correspondingly hid the same amount in
    /// [`Self::unaccounted_bytes`], which read a reassuring 93 MiB while ~2 GiB
    /// of the card was genuinely unexplained. The tell was already in the tree:
    /// `vram::trim_pool_after_load` records the pool holding 1,952 MiB right
    /// after load, against 1,921 MiB of dense weights. Same bytes.
    pub outside_dense_bytes: u64,
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
    /// Per-sequence recurrent state, also **inside** the span: each store's
    /// buffers are carved from claimed regions.
    ///
    /// Named for the same reason [`Self::inside_gallery_bytes`] is, and more
    /// urgently — it is the largest thing in the span that is neither KV nor
    /// weights (~126 MiB per sequence on a hybrid stack, several GiB across a
    /// wide wave) and it moves with wave width, so a run that grows it and a run
    /// that leaks it look identical in the span total alone.
    pub inside_recurrent_bytes: u64,
    /// Driver-reported in use across the whole card (`total - free`).
    pub device_in_use_bytes: u64,
    /// `device_in_use - (span + outside)`. Anything here is held by another
    /// process, by the driver's own context, or by an allocation this report
    /// still does not know about — the last being the interesting case.
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
            pipeline_slack_bytes: super::prefill::warm_pipeline_slack_bytes(),
            warm_usage_bytes: warm_usage,
            over_budget,
            pages_in_per_sec: candle::vram::pages_in_per_sec(),
            commit_total_bytes: perf.map(|p| p.commit_total_bytes),
            commit_limit_bytes: perf.map(|p| p.commit_limit_bytes),
        };

        // ── Admission ───────────────────────────────────────────────────────
        let admission = AdmissionSection {
            setpoint_bytes: self.admit_budget,
            ceiling_bytes: self.admit_budget_ceiling(),
            quantum_bytes: admit_quantum(),
            prefill_width: self.prefill_width(),
            section_width: self.section_ingest_width(),
            decode_width: self.decode_width(),
            queued_prefills: self.prefill_queue.len(),
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
        // reserved footprint already covers all of it — the dense weights
        // included. Adding the weights on top double-books them.
        let outside_dense_bytes = weights.base_bytes.unwrap_or(0);
        let outside_pool_bytes = vram.as_ref().map_or(0, |v| v.pool_reserved_bytes);
        let outside_span_bytes = outside_pool_bytes;
        // Inside the span — claimed regions, so already counted in `span_bytes`
        // and deliberately NOT added to the outside set.
        let inside_gallery_bytes = gallery.resident_bytes;
        let inside_recurrent_bytes = self.model.recurrent_reserved_bytes() as u64;
        let span_bytes = match self.device.location() {
            candle::DeviceLocation::Cuda { gpu_id } => candle_nn::kv_cache::span_layout(gpu_id)
                .map(|l| l.span_end.saturating_sub(l.span_base)),
            _ => None,
        };
        let device_in_use_bytes = vram.as_ref().map_or(0, |v| {
            v.driver_total_bytes.saturating_sub(v.driver_free_bytes)
        });
        let accounting = AccountingSection {
            span_bytes,
            outside_span_bytes,
            outside_dense_bytes,
            outside_pool_bytes,
            inside_gallery_bytes,
            inside_recurrent_bytes,
            device_in_use_bytes,
            unaccounted_bytes: device_in_use_bytes as i64
                - span_bytes.unwrap_or(0) as i64
                - outside_span_bytes as i64,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                setpoint_bytes: 10,
                ceiling_bytes: 11,
                quantum_bytes: 12,
                prefill_width: 1,
                section_width: 2,
                decode_width: 3,
                queued_prefills: 4,
            },
            experts: ExpertSection {
                host_pinned_bytes: 11_000_000_000,
                vram_reserved_bytes: 8_000_000_000,
            },
            weights: WeightSection {
                base_bytes: Some(1_100_000_000),
                resident_expert_bytes: Some(8_000_000_000),
            },
            gallery: GallerySection {
                resident_bytes: 268_435_456,
                resident_turns: 42,
            },
            // Shaped like the real thing so the sums mean something: a 64 GiB
            // span, a 4 GiB pool holding 2 GiB of dense weights inside it, the
            // gallery's own figure from above, and a card holding 75 GiB.
            accounting: AccountingSection {
                span_bytes: Some(68_719_476_736),
                // The pool IS the outside set — the weights are in it, not
                // beside it.
                outside_span_bytes: 4_294_967_296,
                outside_pool_bytes: 4_294_967_296,
                outside_dense_bytes: 2_147_483_648,
                // Inside the span, so it must NOT appear in the outside sum.
                inside_gallery_bytes: 268_435_456,
                // Likewise inside: 16 sequences of hybrid recurrent state.
                inside_recurrent_bytes: 2_113_929_216,
                device_in_use_bytes: 80_530_636_800,
                unaccounted_bytes: 80_530_636_800i64 - 68_719_476_736i64 - 4_294_967_296i64,
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
            "setpoint_bytes",
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
            "outside_dense_bytes",
            "outside_pool_bytes",
            "inside_gallery_bytes",
            "inside_recurrent_bytes",
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
        let a = sample().accounting;
        assert_eq!(
            a.outside_span_bytes, a.outside_pool_bytes,
            "the CUDA pool is the whole outside set: every non-span device \
             allocation goes through `cuMemAllocAsync`",
        );
        assert!(
            a.outside_dense_bytes > 0 && a.outside_dense_bytes < a.outside_pool_bytes,
            "the dense weights are a PART of the pool ({} of {}), never an \
             addend to it — the fixture must exercise that containment",
            a.outside_dense_bytes,
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
        let named_in_span = a.inside_gallery_bytes + a.inside_recurrent_bytes;
        assert!(
            named_in_span < a.span_bytes.expect("the fixture reserves a span"),
            "the gallery and recurrent state are tenants OF the span ({named_in_span} \
             of {:?}), so they must fit inside it",
            a.span_bytes,
        );
        // The residual is the card minus the two totals — and specifically NOT
        // minus the named parts, which are already inside them.
        assert_eq!(
            a.unaccounted_bytes,
            a.device_in_use_bytes as i64
                - a.span_bytes.unwrap_or(0) as i64
                - a.outside_span_bytes as i64,
            "the residual must be what the card holds minus what we can name",
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
