//! Assembles the whole-process memory report from the sections that describe
//! each part of the card and the host.
//!
//! The report exists so every quantity the throttles reason about is read from
//! one publish: the driver's own figures, the pool, the KV arenas, the warm
//! tier, host RAM and admission. The sections are sampled together, so a figure
//! divided by another is two halves of one snapshot. The outside-the-span total
//! is reconciled here because nothing else does it, and the residual is taken
//! against the driver's baseline rather than the parts within each total, so no
//! byte is counted twice. The report is logged at trace level because a full
//! dump on every wave would drown the debug log.

use super::*;

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
    pub(in super::super) fn publish_memory_report(&mut self) {
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

    pub(super) fn build_memory_report(&mut self) -> MemoryReport {
        let captured_unix_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // ── VRAM ────────────────────────────────────────────────────────────
        let vram = self.session.vram_free_total().map(|(free, total)| {
            let (pool_used, pool_reserved) = self.session.vram_pool_stats().unwrap_or((0, 0));
            let governor = self.session.vram_governor().map(|gov| {
                let reading = gov.measure().ok();
                GovernorSection {
                    driver_model: gov.driver_model().map(DriverModel::name),
                    probe: reading.map_or("unread", |r| r.source.name()),
                    capacity_bytes: gov.capacity(),
                    headroom_bytes: reading.map_or(0, |r| r.headroom),
                    pool_cushion_bytes: gov.pool_cushion(),
                    reserved_weights_bytes: gov.class_reserved(AllocClass::Weights),
                    reserved_expert_bytes: gov.class_reserved(AllocClass::Expert),
                    reserved_scratch_bytes: gov.class_reserved(AllocClass::Scratch),
                    reserved_kv_bytes: gov.class_reserved(AllocClass::Kv),
                }
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
        //
        // Carries the fragmentation figure alongside the occupancy, from the one
        // accessor the scheduler's own log line reads (`kv_fragmentation`), so the
        // report, the log and any harness agree by construction rather than by
        // three implementations of the same arithmetic.
        let frag = self.session.kv_fragmentation();
        let classes = self
            .session
            .kv_gpu_class_stats()
            .map(|cs| {
                cs.classes
                    .iter()
                    .filter(|c| c.arenas > 0)
                    .map(|c| {
                        let f = frag
                            .iter()
                            .find(|(k, _)| k.class.bytes() == c.slot_bytes)
                            .map(|(_, f)| *f)
                            .unwrap_or_default();
                        KvClassRow {
                            slot_bytes: c.slot_bytes,
                            arenas: c.arenas,
                            reserved_bytes: c.reserved_bytes as u64,
                            live_bytes: c.live_bytes as u64,
                            packed_arenas: f.packed_arenas,
                            freeable_arenas: f.freeable_arenas(),
                        }
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
        // Sampled here, in the same breath as the class rows above, so a consumer
        // dividing one by the other is dividing two halves of one snapshot.
        let ground = self.session.kv_ground_lost().unwrap_or_default();
        // In the same breath as `span_regions` above, for the reason on
        // `KvSection::span_tenants`: the total and its breakdown have to be one
        // moment or their difference reads as an uncounted tenant.
        let span_tenants: Vec<SpanTenantRow> = candle_nn::kv_cache::arena_census(&self.device)
            .into_iter()
            .map(|t| SpanTenantRow {
                tenant: t.tenant.label(),
                regions: t.regions,
                held_bytes: t.held_bytes,
                pools: t.pools,
            })
            .collect();
        let recurrent_held = span_tenants
            .iter()
            .find(|t| t.tenant == SlotTenant::RecurrentState.label())
            .map_or(0, |t| t.held_bytes as u64);
        let state = self.state_section(recurrent_held);
        let kv = KvSection {
            classes,
            arenas,
            frontier_regions: ground.watermark,
            frontier_holder: self.frontier_holder(ground.watermark),
            live_regions: ground.live_arenas,
            span_regions: ground.span_regions,
            record_regions: ground.record_regions,
            span_tenants,
        };

        // ── Warm tier ───────────────────────────────────────────────────────
        let warm = WarmSection {
            resident_count: self.persist_triggers.warm_resident_count() as usize,
            pending_warm_bytes: self.persist_triggers.pending_warm_bytes(),
            pinned_undrainable_bytes: self.persist_triggers.pinned_undrainable_bytes(),
            resident_bytes: self.persist_triggers.warm_resident_bytes(),
            held_payload_bytes: self.persist_triggers.held_payload_bytes(),
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
            .persist_triggers
            .warm_resident_bytes()
            .saturating_add(self.persist_triggers.pending_warm_bytes());
        let perf = candle::vram::host_perf();
        let cache = index_page_cache::global()
            .map(|c| c.stats())
            .unwrap_or_default();
        let host = HostSection {
            total_bytes: total,
            available_bytes: available,
            buffer_bytes: budget.buffer_bytes,
            weights_reserved_bytes: budget.weights_reserved_bytes,
            weights_capped: budget.weights_capped,
            kv_warm_budget_bytes: budget.kv_warm_budget_bytes,
            pipeline_slack_bytes: super::super::prefill::WARM_PIPELINE_SLACK_BYTES,
            warm_usage_bytes: warm_usage,
            over_budget,
            pages_in_per_sec: candle::vram::pages_in_per_sec(),
            commit_total_bytes: perf.map(|p| p.commit_total_bytes),
            commit_limit_bytes: perf.map(|p| p.commit_limit_bytes),
            index_cache: IndexCacheSection {
                target_bytes: budget.index_cache_bytes,
                pinned_bytes: cache.pinned_bytes,
                pageable_bytes: cache.pageable_bytes,
                held_bytes: cache.held_bytes,
                entries: cache.entries,
                hits: cache.hits,
                misses: cache.misses,
                uncached: cache.uncached,
                evicted: cache.evicted,
            },
        };

        // ── Admission ───────────────────────────────────────────────────────
        let admission = AdmissionSection {
            setpoint_bytes: self.admit_budget,
            ceiling_bytes: self.admit_budget_ceiling(),
            quantum_bytes: admit_quantum(),
            prefill_width: self.prefill_width(),
            section_width: self.section_ingest_width(),
            decode_width: self.decode_width(),
            queued_prefills: self.prefill_queue.len() + self.pending_submits.len(),
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
            floor_bytes: super::super::interleave::optimal_weight_bytes(),
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
        let inside_held_bytes = match self.device.location() {
            candle::DeviceLocation::Cuda { gpu_id } => {
                candle_nn::kv_cache::held_ground_stats(gpu_id).map_or(0, |s| s.used as u64)
            }
            _ => 0,
        };
        // Why the pool was reached at all, split by its two causes — over the
        // last wave, not the run. See the field docs for why the lifetime
        // totals answer nothing.
        let (decline_no_ticket_bytes, decline_closed_bytes) = last_wave_declines();
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
            inside_held_bytes,
            expert_zone_bytes,
            inside_transient_bytes,
            decline_no_ticket_bytes,
            decline_closed_bytes,
            device_in_use_bytes,
            driver_baseline_bytes,
            unaccounted_bytes: device_in_use_bytes as i64
                - span_bytes.unwrap_or(0) as i64
                - outside_span_bytes as i64
                - driver_baseline_bytes.unwrap_or(0) as i64,
        };

        // ── Held ground, by holder ──────────────────────────────────────────
        let held = match self.device.location() {
            candle::DeviceLocation::Cuda { gpu_id } => {
                candle_nn::kv_cache::held_ground_stats(gpu_id).map(|s| HeldSection {
                    used_bytes: s.used as u64,
                    pending_bytes: s.pending as u64,
                    peak_bytes: s.peak as u64,
                    capacity_bytes: s.capacity as u64,
                    largest_free_bytes: s.largest_free as u64,
                    sites: candle_nn::kv_cache::held_ground_sites(gpu_id)
                        .unwrap_or_default()
                        .into_iter()
                        .map(|h| HeldSiteRow {
                            site: h.site,
                            live_bytes: h.live_bytes as u64,
                            live_count: h.live_count,
                            peak_bytes: h.peak_bytes as u64,
                            takes: h.takes,
                            declines: h.declines,
                            declined_bytes: h.declined_bytes,
                            largest_decline_bytes: h.largest_decline as u64,
                        })
                        .collect(),
                })
            }
            _ => None,
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
            state,
            accounting,
            held,
        }
    }
}
