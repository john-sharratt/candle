//! Moves the boundary between the weight zone and the KV side, always on the
//! pipeline thread and only between forwards.
//!
//! The KV side states the number of regions it needs, because a request does not
//! outlive the call that made it, and zero is an ordinary answer: the zone may
//! already sit at its floor, a wave may be live, or an expert layer may be
//! unserved. A guest's concession may take the zone below its serving floor, and
//! the caller owns the restore, because a zone below its floor cannot serve a
//! token. Taking back free KV ground is the reverse direction, asked only between
//! forwards, and the arenas are swept first, since an unswept arena is counted
//! neither free nor spare.

use super::*;

impl ExpertCache {
    /// Buy `regions` of weight-side ground for the KV side, and answer with the
    /// bytes it conceded.
    ///
    /// **The caller states the quantity.** It is either an arena claim that has
    /// run the KV side out and is asking for what it is about to allocate, or the
    /// scheduler's relief asking for its measured setpoint shortfall. Both know
    /// the number; neither can accumulate one, because the request does not
    /// outlive the call that made it. What this replaced — a running count of
    /// refused claims drained here — could and did: 4,436 regions against a
    /// twenty-eight-region need, paid in full.
    ///
    /// **For a caller that is stuck.** The other direction — the weight side
    /// taking back ground the KV side is not using — only runs between forwards
    /// ([`Self::reclaim_spare_ground`]), and a KV side that cannot allocate the
    /// arenas a wave needs never reaches the next one. This is the path that
    /// breaks that.
    ///
    /// Zero is an ordinary answer: the zone may already sit at its floor, a
    /// wave may be live, or an expert layer the forward thread has begun may
    /// still be unserved — the boundary moves only with none in flight. The
    /// caller decides whether to retry on that basis rather than spinning on a
    /// claim that cannot succeed.
    ///
    /// Blocks on the pipeline thread's reply — it owns the cache state and is
    /// the only place a boundary move is safe.
    pub fn request_kv_ground(&self, regions: usize) -> u64 {
        if regions == 0 {
            return 0;
        }
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if !self.send(PipelineMessage::RenegotiateBoundary {
            ask: BoundaryAsk::Sell(regions),
            response_tx,
        }) {
            return 0;
        }
        response_rx.recv().unwrap_or(0)
    }

    /// [`Self::request_kv_ground`] while the engine is paused for a guest: the
    /// zone may go below its serving floor, down to the pinned layers — the
    /// experts with no copy in the pack or the warm tier. Every other expert
    /// reloads from its tier when next routed.
    ///
    /// **The caller owns the restore.** A zone left below its floor cannot serve
    /// a token, so before the engine forwards again the caller must call
    /// [`Self::restore_serving_floor`] until it answers `true`.
    pub fn concede_for_guest(&self, regions: usize) -> u64 {
        if regions == 0 {
            return 0;
        }
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if !self.send(PipelineMessage::RenegotiateBoundary {
            ask: BoundaryAsk::SellPaused(regions),
            response_tx,
        }) {
            tracing::warn!(
                target: "candle_transformers::expert_lre",
                regions,
                "a guest's concession could not reach the expert pipeline: its thread is gone"
            );
            return 0;
        }
        let conceded = response_rx.recv().unwrap_or(0);
        tracing::info!(
            target: "candle_transformers::expert_lre",
            regions,
            conceded_mib = conceded >> 20,
            "the expert pipeline answered a guest's concession"
        );
        conceded
    }

    /// After [`Self::concede_for_guest`] left the zone on its pinned set and
    /// the guest is still short: copy the pinned experts to host memory and
    /// concede their ground too ([`BoundaryAsk::StashPinned`]). Answers the
    /// bytes conceded. The caller owns the restore, as for the concession.
    pub fn stash_pinned_for_guest(&self) -> u64 {
        let (response_tx, response_rx) = mpsc::sync_channel(1);
        if !self.send(PipelineMessage::RenegotiateBoundary {
            ask: BoundaryAsk::StashPinned,
            response_tx,
        }) {
            return 0;
        }
        response_rx.recv().unwrap_or(0)
    }

    /// Grow the zone back to its serving floor after [`Self::concede_for_guest`],
    /// from the ground the guest has handed back. Answers whether the zone now
    /// stands at or above the floor — `true` at once for a zone that never went
    /// below it, and for a build without `cuda`.
    pub fn restore_serving_floor(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            let at_floor = |s: &PipelineStats| {
                !s.pinned_stashed
                    && (s.expert_slot_bytes == 0
                        || s.zone_bytes >= self.serving_floor_slots * s.expert_slot_bytes)
            };
            if at_floor(&PipelineStats::snapshot(&self.stats)) {
                return true;
            }
            candle_nn::kv_cache::reclaim_empty_arenas();
            let (response_tx, response_rx) = mpsc::sync_channel(1);
            if !self.send(PipelineMessage::RenegotiateBoundary {
                ask: BoundaryAsk::RestoreFloor,
                response_tx,
            }) {
                return false;
            }
            let _ = response_rx.recv();
            at_floor(&PipelineStats::snapshot(&self.stats))
        }
        #[cfg(not(feature = "cuda"))]
        {
            true
        }
    }

    /// The other direction: take back KV regions that are standing free.
    ///
    /// **Call this only between forwards.** Moving the boundary evicts and
    /// relocates expert slots, and a wave in flight may be reading either, so
    /// `set_weight_floor` refuses while a wave generation is open on the span.
    ///
    /// The caller is the wave loop's own inter-forward gap, alongside the
    /// transient tier's hand-back — never the pipeline thread. That thread
    /// serves a routed layer while the forward thread is still inside
    /// `ffn_residual` holding the layer's FFN wave guard, so asking from there
    /// would be refused in the common case and, in the narrow window between one
    /// layer's guard dropping and the next one's opening, granted at the cost of
    /// a device-wide quiesce in the middle of a forward.
    ///
    /// Answers with the bytes taken — always zero, since this direction concedes
    /// nothing; the value exists so the two directions share a signature.
    pub fn reclaim_spare_ground(&self) -> u64 {
        // **Sweep before asking, because the answer is computed from `live`.**
        //
        // `spare_regions` offers the weight side what occupancy says is free, and
        // an unswept arena is counted neither free nor spare. A region whose
        // arena went chunk-empty several waves ago is still `live` until
        // something sweeps it, so without this the negotiation is answered
        // against ground that came free some time ago and nobody has noticed.
        //
        // The reactive sweeps cannot cover it: `claim_region` sweeps only when
        // the free list is empty and `place_transient` only after a placement
        // came up short, and a workload with spare regions reaches neither. The
        // 3.6-35B gate runs at `free 18` throughout, so nothing swept between
        // configs at all.
        //
        // Here rather than in each model's wave loop so every MoE model gets it,
        // and so the sweep and the question it informs cannot drift apart.
        #[cfg(feature = "cuda")]
        {
            candle_nn::kv_cache::reclaim_empty_arenas();
            // The growth question — "how much is the KV side holding that I
            // could take?" — asked on this thread, between forwards, where its
            // answer is the pool as it stands (`spare_for_growth`). The pipeline
            // thread is asked only to make a move there is ground for.
            let spare = match spare_for_growth(&self.device.cuda_stream()) {
                Ok(spare) => spare,
                Err(e) => {
                    tracing::warn!("expert cache: growth question failed: {e}");
                    return 0;
                }
            };
            if spare == 0 {
                return 0;
            }
            // A zone at its limit takes nothing however much ground is offered —
            // a cache holding every expert sits there for good — so it is not
            // worth the pipeline thread's backlog to hear so. Read from the
            // gauges that thread publishes each routed layer.
            let at_limit = self
                .stats
                .lock()
                .is_ok_and(|s| s.zone_max_bytes > 0 && s.zone_bytes >= s.zone_max_bytes);
            if at_limit {
                grow_note(GrowOutcome::AtLimit);
                return 0;
            }
            let (response_tx, response_rx) = mpsc::sync_channel(1);
            if !self.send(PipelineMessage::RenegotiateBoundary {
                ask: BoundaryAsk::Take(spare),
                response_tx,
            }) {
                return 0;
            }
            response_rx.recv().unwrap_or(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Span bytes the weight zone could concede to the KV side on demand —
    /// the gauge the pipeline thread publishes each routed layer
    /// (`PipelineStats::zone_cedeable_bytes`). Feeds the prefill width cap:
    /// the elastic boundary cedes this ground to stuck KV claims
    /// (`request_kv_ground`), so a wave sized against it is admissible even
    /// when little KV ground is standing free. Reads 0 before the first
    /// routed layer — the cold-start waves are far below any cap that matters.
    pub fn cedeable_span_bytes(&self) -> usize {
        PipelineStats::snapshot(&self.stats).zone_cedeable_bytes
    }

    /// The weight zone's slots now, less the permanently resident layers'.
    #[cfg(feature = "cuda")]
    pub(super) fn evictable_slots(&self) -> Option<usize> {
        let s = PipelineStats::snapshot(&self.stats);
        let slots = s.zone_bytes.checked_div(s.expert_slot_bytes)?;
        Some(slots.saturating_sub(self.pinned_slots))
    }
}
