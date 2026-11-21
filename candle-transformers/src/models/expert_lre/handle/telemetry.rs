//! The pipeline's telemetry: the counters and profiles it keeps, and the hit
//! references it can replay from the routing it served.
//!
//! A settled read waits for the pipeline thread to serve every layer sent before
//! it, so its counters are complete; a read without a settle trails the forward
//! by the layers still queued, which suits an estimate that folds intervals. The
//! residency references replay the routing of an interval under least-recently-used
//! eviction and under Belady's optimum, which no policy can beat, at the weight
//! zone's present capacity. They are a reporting call and not one made per step.

use super::*;

impl ExpertCache {
    /// Snapshot and reset the pipeline thread's profile accumulator (`pipe_*`
    /// spans), and — in a profile build — the expert GEMMs' worker counters:
    /// `moe:worker copy` (summed worker time copying remote experts into
    /// scratch, one count per item) and `moe:worker cold wait` (summed worker
    /// time waiting on the stager). The worker sums are block-time across the
    /// `WORKERS` blocks of a launch, not wall time.
    pub fn snapshot_profiles(&self) -> ProfileSnapshot {
        let (resp_tx, resp_rx) = mpsc::sync_channel(1);
        #[allow(unused_mut)]
        let mut snap = if self.send(PipelineMessage::SnapshotProfile {
            response_tx: resp_tx,
        }) {
            resp_rx.recv().unwrap_or_default()
        } else {
            ProfileSnapshot::default()
        };
        #[cfg(all(feature = "cuda", feature = "profile"))]
        match self.dispatch.drain_worker_counters() {
            Ok(c) => {
                snap.entries
                    .push(("moe:worker copy".into(), c.copy_ns as f64 / 1e6, c.items));
                snap.entries.push((
                    "moe:worker cold wait".into(),
                    c.cold_wait_ns as f64 / 1e6,
                    c.items,
                ));
                tracing::info!(
                    target: "candle_transformers::expert_lre",
                    items = c.items,
                    launches = c.launches,
                    gib = c.bytes as f64 / (1u64 << 30) as f64,
                    copy_ms = c.copy_ns as f64 / 1e6,
                    cold_wait_ms = c.cold_wait_ns as f64 / 1e6,
                    ahead_items = c.ahead_items,
                    ahead_gib = c.ahead_bytes as f64 / (1u64 << 30) as f64,
                    "expert workers since the last snapshot"
                );
            }
            Err(e) => tracing::warn!("expert worker counters unreadable: {e}"),
        }
        snap
    }

    /// The counters' shared home, for a reader on another thread that must
    /// not wait for the pipeline thread — a gauge such as
    /// `resident_vram_bytes` read while forwards run. [`Self::expert_stats`]
    /// is the settled read.
    pub fn stats_shared(&self) -> Arc<Mutex<PipelineStats>> {
        self.stats.clone()
    }

    /// The pipeline telemetry counters, complete for every routed layer sent
    /// before the call.
    ///
    /// The pipeline thread serves routed layers concurrently with the forward,
    /// so a counter read the moment a forward returns could still be missing
    /// its last layers; this settles the thread first — one channel round
    /// trip, which after the sampler's synchronize finds the thread idle.
    pub fn expert_stats(&self) -> PipelineStats {
        self.settle();
        #[allow(unused_mut)]
        let mut s = PipelineStats::snapshot(&self.stats);
        #[cfg(feature = "cuda")]
        {
            let (ns, n) = self.dispatch.cold_wait.since_reset();
            s.cold_wait_ns = ns;
            s.cold_waits = n as usize;
        }
        s
    }

    /// The routed experts' hit and miss tallies as the pipeline thread has
    /// counted them so far, `(hits, misses)` — without settling it, so they
    /// trail the forward by the routed layers still queued. For an estimate
    /// that folds intervals, read once per step: the settle [`Self::expert_stats`]
    /// makes waits for every queued layer's summary word, and so for the device
    /// to reach the forward's last MoE layer, and then for the pipeline thread
    /// to serve the backlog.
    pub fn hit_counts(&self) -> (usize, usize) {
        self.stats
            .lock()
            .map_or((0, 0), |s| (s.expert_hits, s.expert_misses))
    }

    /// Reset all pipeline telemetry counters to zero, and start a new interval
    /// of the routing trace the residency ceiling replays.
    pub fn reset_expert_stats(&self) {
        self.settle();
        PipelineStats::reset(&self.stats);
        #[cfg(feature = "cuda")]
        {
            self.dispatch.cold_wait.reset();
            if let Ok(mut t) = self.trace.lock() {
                t.reset();
            }
        }
    }

    /// Write the routing served since the start of the previous interval to
    /// `path`, in `routing_trace`'s file format, for offline replay.
    pub fn write_routing_trace(&self, path: &Path) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.settle();
            let capacity = self.evictable_slots().unwrap_or(0);
            let t = self
                .trace
                .lock()
                .map_err(|_| candle::Error::Msg("expert cache: routing trace poisoned".into()))?;
            t.write(path, capacity).map_err(candle::Error::wrap)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = path;
            Ok(())
        }
    }

    /// The residency references for the interval since the last
    /// [`Self::reset_expert_stats`], as hit rates in percent over the routing
    /// served in it, at the weight zone's present capacity, on the measured
    /// hit rate's terms: `(lru, min)` — least-recently-used eviction, and
    /// Belady's optimum (`belady`), which no policy can beat. A replay of the
    /// whole interval: a reporting call, never a per-step one. `None` before
    /// any routing.
    pub fn hit_references(&self) -> Option<(f64, f64)> {
        #[cfg(feature = "cuda")]
        {
            self.settle();
            let capacity = self.evictable_slots()?;
            let t = self.trace.lock().ok()?;
            Some((t.lru(capacity)?, t.ceiling(capacity)?))
        }
        #[cfg(not(feature = "cuda"))]
        None
    }
}
