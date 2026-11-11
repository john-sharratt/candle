//! The wave window's close: the telemetry the run loop publishes once every
//! [`WAVE_WINDOW`](super::WAVE_WINDOW) — the wave sample the performance page
//! draws, the memory report, the per-wave sweep of emptied arenas and the
//! fragmentation and planner lines.
//!
//! Called from the busy loop after each iteration's phase attribution, and
//! from the idle wait each time a window falls due there (`idle_wait`), so an
//! idle card closes its windows on time as a busy one does: one row per slot
//! of `/v1/telemetry`, never a gap while nothing runs.

use super::*;
use candle::DeviceLocation;
use candle_nn::kv_cache::{region_stats, span_partition, vram_bands, REGION_BYTES};

impl Scheduler {
    /// Close the wave window: publish its sample and the reports that share its
    /// cadence. Call only when `wave_stats.due()`.
    pub(super) fn flush_wave_telemetry(&mut self) {
        // The 2 s telemetry window: the memory report, the class census, the
        // fragmentation log. Cheap per wave only because it is rare, which is
        // exactly the claim a span is needed to check.
        let _g = super::profile::span("loop:telemetry");

        // **What the rate planner currently believes a forward costs.**
        //
        // Published because every one of these was invisible while being wrong.
        // The planner decides how wide a wave is, which is the whole of decode
        // throughput, and it does so from four learned numbers — none of which
        // appeared in any log. A zero `fixed`+`overhead` makes the projected rate
        // flat in width and narrows every wave to one row, and that state was
        // reached on a card whose only distinguishing feature was being large
        // enough to hold the checkpoint. It read as healthy throughout.
        if let Some(rate) = self.wave_rate.as_ref() {
            tracing::trace!(
                target: "candle_conversation::scheduler::throttle",
                fixed_ms = rate.fixed_secs() * 1e3,
                overhead_ms = rate.overhead_secs() * 1e3,
                compute_us_per_token = rate.compute_secs_per_token() * 1e6,
                link_gbps = rate.effective_bytes_per_s() / 1e9,
                samples = rate.samples(),
                "wave rate model",
            );
        }
        // Our eviction gate's own view of VRAM: the pool budget we
        // defend (vram_budget_available) and pool_used — queried only
        // on the wave we emit, not every iteration.
        let kv_vram = self
            .session
            .vram_budget_available()
            .zip(self.session.vram_pool_stats())
            .map(|(budget, (used, _reserved))| (budget, used));
        let backlog = self.pending_prefill_tokens();
        // The whole card, cut by what holds each byte: the driver's total
        // and free, what was on the card before the span, the CUDA pool
        // outside it, the span's own partition and the resident experts
        // in its weight zone. The bands add up to the card's total, so
        // the panel stacking them can show where every byte went.
        let pool_reserved = self.session.vram_pool_stats().map_or(0, |(_, r)| r as u64);
        let (free, total) = self
            .session
            .vram_free_total()
            .map_or((0, 0), |(f, t)| (f as u64, t as u64));
        let (context, span) = match self.device.location() {
            DeviceLocation::Cuda { gpu_id } => (
                region_stats(gpu_id).and_then(|s| s.pre_reservation_in_use_bytes),
                span_partition(gpu_id),
            ),
            _ => (None, None),
        };
        let resident_experts = self
            .model
            .expert_stats()
            .map_or(0, |s| s.resident_vram_bytes as u64);
        let vram = vram_bands(total, free, context, pool_reserved, span, resident_experts);
        // Active-work counts at this wave for the slots panel.
        let slots = (
            self.slot_conversations.len() as u32,
            self.decode_width() as u32,
            self.prefill_width() as u32,
            self.section_ingest_width() as u32,
        );
        self.wave_stats.flush(
            kv_vram,
            backlog,
            vram,
            slots,
            self.model.expert_stats().map(|s| ExpertCounters::of(&s)),
            match &self.device {
                Device::Cuda(d) => Some(d.capture_stats()),
                _ => None,
            },
            self.wave_rate.as_mut().map(|r| r.take_trades()),
            self.model.recurrent_census(),
        );
        self.report_waiting_submits();
        // Same cadence: publish the full memory report (global slot for
        // `GET /v1/memory` + one JSON debug line). See `memory_report`.
        self.publish_memory_report();
        // Return emptied regions to the free list. Compression
        // (hot→warm) and eviction free chunks scattered across arenas,
        // and an arena only gives its region back once its last chunk
        // goes — so nothing surfaces without a sweep looking for it.
        // Running it here, ahead of pressure, is what keeps the
        // free-region count honest: the setpoint is compared against
        // regions that are genuinely claimable, not against a count
        // that would only be right after the next relief pass.
        let swept = self.session.release_empty_arenas().unwrap_or(0);
        if swept > 0 {
            relief_trace::note("sched", "arena_sweep", swept as u64, 0);
            tracing::trace!(
                target: "candle_conversation::scheduler::vram_relief",
                arenas_swept = swept,
                "proactive empty-arena sweep (per-wave)"
            );
        }
        self.log_kv_memory();
        // **How fragmented the KV pools are — measured where a compaction
        // would run, at the cadence it would run at.**
        //
        // Deliberately beside `release_empty_arenas` above, because the two
        // are the same job at different granularities and the contrast is
        // the point: the sweep returns an arena whose LAST chunk has gone,
        // and nothing moves a chunk between arenas, so an arena holding a
        // handful of live chunks keeps its whole 16 MiB region indefinitely.
        // `freeable` is the regions that are recoverable and that the sweep
        // structurally cannot recover.
        //
        // This is the figure that is NOT region holes. A region freed below
        // the arena frontier is taken by the next claim — the free list is
        // lowest-index-first — so holes are self-correcting and measured to
        // peak in the tens and settle at one. Sparsity does not
        // self-correct, which is why it is what gets logged and what a
        // compaction is judged by.
        // **The frontier first, because the frontier is what costs
        // weights.** The tier stands above the highest live arena and
        // `weight_floor` is measured from there, so expert residency is
        // set by that one index — not by how many arenas are live and not
        // by how full they are. `could_be` is where the frontier would sit
        // with everything packed, and the gap between them is the ground a
        // perfect compaction hands back, split into its two causes: the
        // holes that hold the frontier up, and the sparsity that fills it
        // with air.
        if let Some(g) = self.session.kv_ground_lost() {
            if g.total() > 0 {
                let mib = REGION_BYTES >> 20;
                tracing::trace!(
                    target: "candle_conversation::scheduler::vram_relief",
                    frontier = g.watermark,
                    could_be = g.packed_arenas,
                    live_arenas = g.live_arenas,
                    arena_holes = g.arena_holes(),
                    sparsity = g.sparsity(),
                    denied_mib = g.total() * mib,
                    // What reclaiming it would cost. Logged beside the
                    // gain because the ratio is the only thing that says
                    // whether a pass is worth its bandwidth: four arenas
                    // for forty thousand copies is not.
                    planned_moves = self.session.kv_planned_moves(),
                    "kv ground denied to the weight side by fragmentation",
                );
            }
        }
        let frag = self.session.kv_fragmentation();
        let freeable: usize = frag.iter().map(|(_, f)| f.freeable_arenas()).sum();
        if freeable > 0 {
            let rows: Vec<String> = frag
                .iter()
                .filter(|(_, f)| f.freeable_arenas() > 0)
                .map(|(k, f)| {
                    format!(
                        "{}B={}a/{}pack({}%)",
                        k.class.bytes(),
                        f.arenas,
                        f.packed_arenas,
                        f.occupancy_pct(),
                    )
                })
                .collect();
            tracing::trace!(
                target: "candle_conversation::scheduler::vram_relief",
                freeable_arenas = freeable,
                freeable_mib = freeable * (REGION_BYTES >> 20),
                "kv fragmentation (a perfect pack would return this): {}",
                rows.join(" "),
            );
        }
        // **Is the planner still running on its seeds?** Every figure
        // it decides with is learned, and the seeds are a measurement of
        // one card with one checkpoint — the decode one was found 26x
        // optimistic. A run that admits oddly is asked this first, and
        // without the line the answer is unobtainable after the fact.
        if let Some(r) = self.wave_rate.as_ref() {
            tracing::trace!(
                target: "candle_conversation::scheduler::admission",
                link_gbps = r.link_bytes_per_s() / 1e9,
                effective_gbps = r.effective_bytes_per_s() / 1e9,
                link_fraction = r.link_fraction(),
                compute_us_per_row = r.compute_secs_per_token() * 1e6,
                layer_us = r.layer_secs() * 1e6,
                hit_exponent = r.hit_exponent(),
                // The two-cost fit only answers once the observations
                // have enough spread to separate the copy from the
                // compute; until then both are seeds.
                fit_converged = r.cost_fit_converged(),
                prefill_samples = r.samples(),
                decode_samples = r.decode_samples(),
                hit_samples = r.hit_samples(),
                "wave rate planner",
            );
        }
        // Wake the sealing pass now that its ground exists, rather
        // than leaving it to the 5 s tick. Guarded on there being work —
        // an atomic load — so an idle engine is not woken once per wave
        // to find nothing. The trigger coalesces on a one-slot channel,
        // so a pass already running or already queued absorbs this.
        if self.persist_triggers.pending_warm_bytes() > 0 {
            self.persist_triggers.fire();
        }
    }
}
