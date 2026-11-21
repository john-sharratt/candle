//! Moves a timeline's K/V between the GPU, warm RAM and cold disk tiers, and sets
//! how long it is kept.
//!
//! `demote_timelines_hot` drops device copies that are already warm, keeping the
//! RAM copy. `evict_ingest_timeline` frees both resident tiers once a completed
//! ingest is durable on disk. A transient timeline never reaches cold storage, and
//! an unresumed one keeps no recurrent snapshot because it is read and never
//! extended. `settle_evicted_timeline` waits for the device copies to leave, and
//! fails rather than returning early while a turn still holds one.

use super::*;

impl ConversationEngine {
    /// Set (or clear) the per-conversation KV-compression override for
    /// `timeline` at runtime — used by the daemon to flag a forked capture
    /// conversation as lossless (native R16/F16, no quantize) before its first
    /// turn migrates hot→warm. See [`crate::substrate::ConvCompression`].
    pub fn set_timeline_compression(
        &self,
        timeline: TimelineId,
        compression: Option<ConvCompression>,
    ) {
        self.conversation
            .set_timeline_compression(timeline, compression);
    }

    /// Mark `timeline` as scratch: its turns never reach cold storage.
    ///
    /// Sets `no_cold_persist` on every residence the timeline holds, so a
    /// conversation opened for one question and thrown away costs GPU and RAM
    /// for as long as it runs and nothing on disk afterwards. The alternative —
    /// writing it and tombstoning it — pays the whole write and then asks
    /// compaction to take it back.
    ///
    /// Call it immediately after minting, before the first turn seals: a turn
    /// that has already been written is already on the cold path and this does
    /// not retract it.
    pub fn mark_timeline_transient(&self, timeline: TimelineId) {
        self.conversation.mark_timeline_transient(timeline);
    }

    /// Mark `timeline` as kept but never resumed: its turns stay — sealed,
    /// persisted, projectable — but it keeps no recurrent snapshot, and the one
    /// it already has is retired. For content that is read and never extended:
    /// a document ingested as a conversation, a kept dream. A snapshot is a
    /// whole recurrent state per conversation (~112 MiB on Flash-Next) whatever
    /// the conversation's length. See
    /// [`crate::projection::Conversation::mark_timeline_unresumed`].
    ///
    /// In-memory as a mark, durable as the retirement: a caller marks its
    /// timelines again after a restart (they then write nothing new), and a
    /// fork of such a timeline starts from its nearest ancestor that kept one.
    pub fn mark_timeline_unresumed(&self, timeline: TimelineId) {
        self.conversation.mark_timeline_unresumed(timeline);
    }

    /// [`Self::mark_timeline_unresumed`] without the retiring write, for a
    /// timeline whose tail is known to be retired already (or absent): later
    /// seals write no snapshot, and nothing is appended now.
    pub fn note_timeline_unresumed(&self, timeline: TimelineId) {
        self.conversation.note_timeline_unresumed(timeline);
    }

    /// Bytes of the recurrent snapshot `timeline` keeps, if any — a retired
    /// tail reads as a few dozen. Read from the index, not the record.
    pub fn recurrent_snapshot_bytes(&self, timeline: TimelineId) -> Option<u64> {
        self.conversation.recurrent_snapshot_bytes(timeline)
    }

    /// Mark `timeline` for distillation at `mode` (shed content at compaction) —
    /// see [`crate::projection::Conversation::distill_timeline`]. A later call may
    /// upgrade the mode; gate on [`Self::is_timeline_distilled`] only to avoid
    /// re-marking at the same mode.
    pub fn distill_timeline(&self, timeline: TimelineId, mode: DistillMode) -> crate::Result<()> {
        self.conversation
            .distill_timeline(timeline, mode)
            .map_err(ConversationError::Model)
    }

    /// Whether `timeline` is already marked for distillation.
    pub fn is_timeline_distilled(&self, timeline: TimelineId) -> bool {
        self.conversation.is_timeline_distilled(timeline)
    }

    /// Demote the hot K/V of `timelines` to the warm (RAM) tier, keeping the
    /// warm copy — the VRAM the hot copies held returns to the pool. The demote
    /// itself runs on the scheduler thread (single-owner GPU-pool mutation).
    /// Used by the loader's calibration phase to keep VRAM flat: reclaim each
    /// throwaway case's K/V as it retires rather than letting it accumulate hot.
    /// Idempotent — a turn already demoted (or not yet warm) is skipped.
    ///
    /// `flush` selects the mode:
    /// - `true` (boundary sweep): first drain the hot→warm migration **on this
    ///   thread** so the whole tail is warm-backed (hence demotable), then issue
    ///   the demote and **block** until it completes — the caller needs the VRAM
    ///   reclaimed before the next phase prefills. The flush is done here rather
    ///   than inside the scheduler handler so its (≤30 s) wait can't stall the
    ///   scheduler's decode/prefill loop. Returns the number of residences
    ///   demoted.
    /// - `false` (incremental sweep): **fire-and-forget** — issue the demote and
    ///   return immediately without blocking the caller (case submission must not
    ///   stall). Only already-warm-backed turns are dropped; any not-yet-warm
    ///   tail is caught by the next sweep or the boundary flush. Returns `0`.
    pub fn demote_timelines_hot(
        &self,
        timelines: &[TimelineId],
        flush: bool,
    ) -> crate::Result<usize> {
        if flush {
            self.persist_thread
                .trigger_handle()
                .flush_blocking(std::time::Duration::from_secs(30));
        }
        if timelines.is_empty() {
            return Ok(0);
        }
        let (response_tx, response_rx) = flume::bounded(1);
        self.core
            .scheduler_tx
            .send(SchedulerRequest::DemoteTimelinesHot {
                conversation: self.conversation.clone(),
                timelines: timelines.to_vec(),
                response_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        if flush {
            // Boundary: wait for the demote to complete (VRAM must be reclaimed
            // before the next phase).
            let demoted = response_rx
                .recv()
                .map_err(|_| ConversationError::SchedulerGone)??;
            Ok(demoted)
        } else {
            // Incremental: fire-and-forget. Dropping `response_rx` makes the
            // handler's reply a silent no-op; the demote still runs.
            drop(response_rx);
            Ok(0)
        }
    }

    /// Fully evict a **completed ingest** `timeline`'s KV from VRAM + RAM once
    /// it is durable on disk: flag every turn residence `evict_when_cold` and
    /// wake the persistence thread so the hot→warm→cold pipeline runs promptly.
    /// As each turn migrates, its VRAM is freed at warm-land and its RAM copy at
    /// cold-land, leaving it cold-only on NVMe — `elevate_to_hot` pulls it back
    /// on demand if a later projection re-selects it. Returns the number of turn
    /// residences flagged.
    ///
    /// Unlike [`Self::demote_timelines_hot`] (hot→warm, keeps the warm RAM
    /// copy), this reclaims **both** resident tiers: a completed code_read file
    /// is not attended again until retrieval, so keeping it warm only wastes RAM
    /// and PCIe migration bandwidth. Fire-and-forget — the actual frees happen
    /// on the persistence thread as durability lands.
    pub fn evict_ingest_timeline(&self, timeline: TimelineId) -> usize {
        let flagged = self.conversation.mark_timeline_evict_when_cold(timeline);
        if flagged > 0 {
            // Wake the persistence thread so the flagged turns migrate → persist
            // → evict now, rather than waiting for the next periodic tick.
            self.persist_thread.trigger();
        }
        flagged
    }

    /// Wait for an evicted `timeline` to leave the device, then settle it.
    ///
    /// [`Self::evict_ingest_timeline`] only flags the timeline: the persistence
    /// thread copies each turn warm and drops its device copy as the warm copy
    /// lands, on its own schedule. A caller that wants its next request to start
    /// on the ground the last one gave back calls this after it — the Strata
    /// harness does, standing in for the idle time a daemon has between one
    /// user's requests. It drives the persistence thread until no turn of
    /// `timeline` holds a device copy, then has the scheduler sweep the emptied
    /// arenas, pack the pools and let the weight side grow back into what that
    /// freed (`Scheduler::settle_device`), and answers the partition as it then
    /// stands.
    ///
    /// **Fails rather than returning early** when a turn still holds a device
    /// copy after `timeout`: a turn the persistence thread cannot migrate is a
    /// fault to name, not a state to measure the next request through.
    pub fn settle_evicted_timeline(
        &self,
        timeline: TimelineId,
        timeout: Duration,
    ) -> crate::Result<SettleReport> {
        let trigger = self.persist_thread.trigger_handle();
        let deadline = Instant::now() + timeout;
        loop {
            let hot = self.conversation.timeline_hot_residences(timeline);
            if hot == 0 {
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(ConversationError::Other(format!(
                    "timeline {timeline:?} still holds {hot} turn(s) on the device {timeout:?} \
                     after its eviction: the persistence thread did not migrate them to warm"
                )));
            }
            // A flush that cannot be queued is one already in flight, whose pass
            // serves this wait as well; look again shortly rather than spin.
            if !trigger.flush_blocking(deadline - now) {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        let (response_tx, response_rx) = flume::bounded(1);
        self.core
            .scheduler_tx
            .send(SchedulerRequest::Settle { response_tx })
            .map_err(|_| ConversationError::SchedulerGone)?;
        response_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)
    }
}
