//! Wall-clock attribution for each phase of a wave, filed with the activity
//! census.
//!
//! Each quantum's wall time is filed under its own phase, so a profile can say
//! where a wave's time went. Cleanup, rates and the gallery are timed in the same
//! way. The KV memory picture is published at the loop's slow cadence and covers
//! the CUDA pool, the reservation's regions, the transient domains and the slot
//! state slabs. The pool holds only the model, the expert cache and the scratch
//! buffers that stop growing, so a flat pool is the healthy shape, and growth
//! means something outside the reservation is still allocating per wave.

use super::*;

impl Scheduler {
    /// Run the decode quantum, attributing its wall-clock to [`WavePhase::Decode`]
    /// (the reprojection drain inside it is separately attributed to
    /// [`WavePhase::Reproject`] as a sub-slice).
    pub(super) fn timed_decode(&mut self) {
        let _stage = census_stage::enter(Stage::Decode);
        let _g = profile::span("loop:decode");
        let t = Instant::now();
        self.wave_stats.in_decode_quantum = true;
        self.run_decode_until_budget();
        self.wave_stats.in_decode_quantum = false;
        self.wave_stats
            .add_phase(WavePhase::Decode, t.elapsed().as_millis() as u64);
    }

    /// Run `work`, filing its wall with the census under `activity`.
    pub(super) fn timed_activity(
        &mut self,
        activity: super::super::step_census::Activity,
        work: fn(&mut Self),
    ) {
        let t = Instant::now();
        work(self);
        self.step_census.spend(activity, t.elapsed());
    }

    /// Clean up finished turns — their seals — filed with the census.
    pub(super) fn timed_cleanup(&mut self) {
        let _stage = census_stage::enter(Stage::Seal);
        let t = Instant::now();
        self.cleanup_finished();
        self.step_census
            .spend(super::super::step_census::Activity::Seal, t.elapsed());
    }

    /// Update the gallery's keep trade, give up its dead turns, and publish
    /// the learned rates, each filed with the census (`step_census`).
    pub(super) fn timed_rates_and_gallery(&mut self) {
        let _stage = census_stage::enter(Stage::Gallery);
        let t = Instant::now();
        self.update_gallery_keep();
        self.sweep_gallery();
        let gallery = t.elapsed();
        self.publish_learned_rates();
        self.step_census
            .spend(super::super::step_census::Activity::Gallery, gallery);
        self.step_census.spend(
            super::super::step_census::Activity::Rates,
            t.elapsed().saturating_sub(gallery),
        );
    }

    /// Run the prefill quantum, attributing its wall-clock to [`WavePhase::Prefill`].
    pub(super) fn timed_prefill(&mut self) {
        let _g = profile::span("loop:prefill");
        let t = Instant::now();
        self.timed_activity(
            super::super::step_census::Activity::Withdraw,
            Self::withdraw_abandoned_prefills,
        );
        self.timed_rates_and_gallery();
        let _stage = census_stage::enter(Stage::Prefill);
        self.run_prefill_until_budget();
        self.wave_stats
            .add_phase(WavePhase::Prefill, t.elapsed().as_millis() as u64);
    }

    /// Run the section-ingest quantum, attributing its wall-clock to
    /// [`WavePhase::Section`].
    pub(super) fn timed_section(&mut self) {
        let _stage = census_stage::enter(Stage::Seal);
        let _g = profile::span("loop:section");
        let t = Instant::now();
        self.timed_activity(
            super::super::step_census::Activity::Seal,
            Self::run_section_ingest_until_budget,
        );
        self.wave_stats
            .add_phase(WavePhase::Section, t.elapsed().as_millis() as u64);
    }

    /// Publish the KV memory picture at the wave loop's slow cadence: the CUDA
    /// pool, the reservation's regions, the transient domains, and the
    /// slot-state slabs.
    ///
    /// This used to also *trim* the pool — return its reserved-but-free
    /// fragmentation to the OS, keeping a `pool_used + slack` floor — and every
    /// relief rung called it after freeing anything. That mattered when KV was
    /// pool memory: freed arenas stayed reserved, `pool_reserved` climbed away
    /// from `pool_used`, and the driver's real free went to zero. None of it
    /// applies now. KV comes from the reservation, so releasing an arena moves a
    /// region between two lists and changes no pool accounting at all; what is
    /// left in the pool is the model, the expert cache and the few remaining
    /// grow-only scratches, which reach their size and stay there.
    /// `cuMemPoolTrimTo` on that is work with nothing to reclaim — and it
    /// synchronously unmaps, which is why it needed a guard against in-flight
    /// kernels holding captured pointers. Dropping the trim drops the guard.
    pub(in super::super) fn log_kv_memory(&self) {
        let mib = |b: usize| b / (1024 * 1024);
        if let Some((used, reserved)) = self.session.vram_pool_stats() {
            // What is left of the pool once KV moved out. Flat is the healthy
            // shape: growth here means something outside the reservation is
            // still allocating per-wave.
            tracing::trace!(
                "kv-pool: used={}MiB reserved={}MiB gap={}MiB",
                mib(used),
                mib(reserved),
                mib(reserved.saturating_sub(used)),
            );
        }
        // Resident GPU arena occupancy, one line per occupied size class. Watch
        // occupancy move from the large classes to the small ones across a
        // pressure episode to confirm compress-to-free is bringing
        // quantization forward.
        if let Some(cs) = self.session.kv_gpu_class_stats() {
            let rows: Vec<String> = cs
                .classes
                .iter()
                .filter(|c| c.arenas > 0)
                .map(|c| {
                    format!(
                        "{}B={}a/{}MiB(live {}MiB)",
                        c.slot_bytes,
                        c.arenas,
                        mib(c.reserved_bytes),
                        mib(c.live_bytes)
                    )
                })
                .collect();
            tracing::trace!("kv-pool classes: {}", rows.join(" "));
        }
        // The reservation's KV side. `free` is the pressure signal admission
        // reads; `peak_live` against `total` says how close the startup
        // partition came to binding, which is what step 7 tunes.
        //
        // The last two must read zero. A region claim creates an arena, arena
        // creation waits for the gap between forwards, and the wave's tier is
        // placed against the arena frontier as it stands at that moment — so a
        // claim arriving with a tier standing is an arena created inside a wave.
        if let Some(r) = candle_nn::kv_cache::region_stats(0) {
            tracing::trace!(
                "kv-regions: live={} peak={} free={} of {} ({}MiB) | tier={}MiB \
                 (ceiling {} regions) | weights={}MiB | in-wave-arenas={} in-wave-refusals={}",
                r.live,
                r.peak_live,
                r.free,
                r.total,
                mib(r.total * candle_nn::kv_cache::REGION_BYTES),
                mib(r.transient_bytes),
                r.transient_ceiling,
                mib(r.weight_bytes),
                r.fresh_claims_during_wave,
                r.refusals_during_wave,
            );
        }
        // Per-domain transient peaks, the terms the transient tier is sized
        // from: `S = 2*W_wave + W_persist + shelf`.
        if let Some((cursor, peak, cap)) = candle_nn::kv_cache::persistence_domain_stats(0) {
            if peak > 0 {
                tracing::trace!(
                    "kv-transient persist: cursor={}MiB peak={}MiB cap={}MiB",
                    mib(cursor),
                    mib(peak),
                    mib(cap),
                );
            }
        }
        // The `W_wave` term of the same equation. Both halves, because the span
        // has to hold the larger of the two.
        if let Some(halves) = candle_nn::kv_cache::wave_domain_stats(0) {
            let peak = halves.iter().map(|h| h.1).max().unwrap_or(0);
            if peak > 0 {
                tracing::trace!(
                    "kv-transient wave: peak={}MiB (a={}MiB b={}MiB) cap={}MiB each",
                    mib(peak),
                    mib(halves[0].1),
                    mib(halves[1].1),
                    mib(halves[0].2),
                );
            }
        }
        // Slot-state slabs: the region tier's answer to audit A13. `slabs`
        // settling and then staying flat while sequences deepen is the evidence
        // that the decode path has stopped calling the allocator — it used to
        // free and re-allocate this buffer every time a sequence crossed a
        // 32-token boundary, per layer.
        let (live, slabs, bytes) = candle_nn::kv_cache::slot_state_stats();
        if slabs > 0 {
            tracing::trace!(
                "kv-slotstate: live={live} slabs={slabs} reserved={}MiB promotions={}",
                mib(bytes),
                candle_nn::kv_cache::class_promotion_count(),
            );
        }
    }
}
