//! Full-system memory report: one snapshot covering VRAM, KV arenas, the warm
//! tier, host RAM, and the admission throttle — published by the scheduler at
//! its ~2 s telemetry cadence, logged, and served over HTTP.
//!
//! Exists because tonight's failures were all *accounting* failures before they
//! were memory failures: admission read the pool reuse gap as free VRAM, the
//! host-RAM floor read an 11 GB pinned expert pool as pressure, and none of the
//! numbers involved were visible in one place. This report puts every quantity
//! the throttles reason about side by side — what the driver says, what the
//! pool has reserved, what the governor believes is evictable, what is pinned
//! and can never move — so a wrong inference is attributable from a single log
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

use candle::vram::{host_pinned_bytes, AllocClass, Criticality};
use candle_nn::kv_cache::{global_arena_memory_report, GpuArenaFormatStats};
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
/// appears in no other field of this report. It is registered for relief (its
/// pages are evictable and rebuild from the substrate on demand), but relief
/// registration reports what *could* be freed, not what is currently held.
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
    /// KV floor the relief ladder never evicts below.
    pub kv_floor_bytes: u64,
    pub scratch_margin_bytes: u64,
    /// What registered relievers report they could reversibly free (≤ Moderate).
    pub evictable_moderate_bytes: u64,
    /// Loose per-class reserved tallies (reporting, not availability gates).
    pub reserved_weights_bytes: u64,
    pub reserved_expert_bytes: u64,
    pub reserved_scratch_bytes: u64,
    pub reserved_kv_bytes: u64,
}

/// KV arena occupancy, whole-process.
#[derive(Debug, Clone, Serialize)]
pub struct KvSection {
    /// Float vs quant split of the resident GPU arenas.
    pub float_arenas: usize,
    pub float_reserved_bytes: u64,
    pub float_live_bytes: u64,
    pub quant_arenas: usize,
    pub quant_reserved_bytes: u64,
    pub quant_live_bytes: u64,
    /// Per-backing per-format rows from every registered `ChunkedKvBacking`.
    pub arenas: Vec<ArenaRow>,
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
    /// Hot KV the drain is skipping because it is pinned — counted evictable by
    /// the forecast but unreclaimable while the pin holds.
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
                kv_floor_bytes: gov.kv_floor(),
                scratch_margin_bytes: gov.scratch_margin(),
                evictable_moderate_bytes: gov.evictable_estimate(Criticality::Moderate),
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
        let fs = self
            .session
            .kv_gpu_format_stats()
            .unwrap_or(GpuArenaFormatStats {
                float_arenas: 0,
                float_reserved_bytes: 0,
                float_live_bytes: 0,
                quant_arenas: 0,
                quant_reserved_bytes: 0,
                quant_live_bytes: 0,
            });
        let arenas = global_arena_memory_report()
            .into_iter()
            .map(|(backing, format, arenas, bytes)| ArenaRow {
                backing,
                format,
                arenas,
                bytes: bytes as u64,
            })
            .collect();
        let kv = KvSection {
            float_arenas: fs.float_arenas,
            float_reserved_bytes: fs.float_reserved_bytes as u64,
            float_live_bytes: fs.float_live_bytes as u64,
            quant_arenas: fs.quant_arenas,
            quant_reserved_bytes: fs.quant_reserved_bytes as u64,
            quant_live_bytes: fs.quant_live_bytes as u64,
            arenas,
        };

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
                    kv_floor_bytes: 8,
                    scratch_margin_bytes: 9,
                    evictable_moderate_bytes: 10,
                    reserved_weights_bytes: 11,
                    reserved_expert_bytes: 12,
                    reserved_scratch_bytes: 13,
                    reserved_kv_bytes: 14,
                }),
            }),
            kv: KvSection {
                float_arenas: 1,
                float_reserved_bytes: 2,
                float_live_bytes: 3,
                quant_arenas: 4,
                quant_reserved_bytes: 5,
                quant_live_bytes: 6,
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
            "evictable_moderate_bytes",
            "float_live_bytes",
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
        ] {
            assert!(json.contains(key), "missing {key} in {json}");
        }
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
