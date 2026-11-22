//! Prepares ingested content for retrieval: append-only layers, score
//! normalisation, and preloading the galleries.
//!
//! Ingest layers are marked append-only, so their projections read their own
//! scope rather than material from other files. Score normalisation is warmed per
//! file or per collection, so scores sit on the same scale across process starts.
//! The summariser these calls once drove is disconnected, so `pending_summary_len`
//! is always zero and `trigger_summariser` does nothing. The warmers and the
//! preload run on the scheduler thread, which owns the GPU gallery arena.

use super::*;

impl ConversationEngine {
    /// Backpressure metric — turns awaiting summariser absorption for
    /// `timeline`. **Always zero**: the summariser is disconnected, so nothing
    /// enqueues (see [`Self::new`]). It was zero in steady state before, too, so
    /// this reads the same either way.
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.conversation.pending_summary_len(timeline)
    }

    /// Wake the summariser thread now instead of waiting for its next tick.
    ///
    /// **A no-op while the summariser is disconnected** — the disabled handle has
    /// no receiver, so the send fails silently. Callers (the ingest pipeline
    /// kicks it after a scope lands) need no knowledge of that.
    pub fn trigger_summariser(&self) {
        self.core.summariser_thread.trigger();
    }

    /// Enable or disable AVL summarisation for `timeline`.
    ///
    /// **Every timeline now defaults to `false`** — the summariser is
    /// disconnected (see [`Self::new`]), so setting `true` here would queue turns
    /// onto `pending_summary_queue` that nothing drains. Nothing in the engine or
    /// `zend` calls it with `true`; the remaining production callers pass `false`
    /// on ingest timelines, which is redundant against the default but states the
    /// intent at the site.
    ///
    /// It exists as the single re-enabling point: deciding *which* timelines opt
    /// in is this call plus spawning the thread in [`Self::new`]. See
    /// [`crate::summary_tree`].
    pub fn set_timeline_summarize(&self, timeline: TimelineId, summarize: bool) {
        self.conversation
            .set_timeline_summarize(timeline, summarize);
    }

    /// Mark `layer` as an append-only ingest layer (code_reading/repo_map): a
    /// projection targeting it is scored/selected self-local (belief groups masked
    /// to the target timeline), so an ingest scope-summary is grounded in its own
    /// scope rather than cross-file retrieval. Called once per ingest layer at
    /// setup. See [`crate::substrate::Substrate::mark_layer_append_only`].
    pub fn mark_layer_append_only(&self, layer: LayerId) {
        self.conversation.mark_layer_append_only(layer);
    }

    /// Let an append-only `layer`'s projections draw `groups` — groups of the
    /// layers it compounds on — across conversations, its own layer staying
    /// self-local. See [`crate::substrate::Substrate::set_ingest_sees`].
    pub fn set_ingest_sees(&self, layer: LayerId, groups: std::collections::HashSet<GroupId>) {
        self.conversation.set_ingest_sees(layer, groups);
    }

    /// Re-arm the score-normalization warm-up so the next projection re-learns
    /// per-file backgrounds — call after an ingest reconcile mints fresh timelines
    /// the prior warm never scanned. See
    /// [`crate::projection::Conversation::reset_normalization_warm`].
    pub fn reset_normalization_warm(&self) {
        self.conversation.reset_normalization_warm();
    }

    /// A handle that warms the ingest layers' per-file backgrounds on the
    /// scheduler's GPU gallery arena, a slice of files at a time, without
    /// holding this engine. Run it AFTER an ingest pass / reconcile finishes
    /// (never concurrently — it would starve the ingest writer).
    pub fn ingest_warmer(&self) -> IngestWarmer {
        IngestWarmer {
            conversation: self.conversation.clone(),
            scheduler_tx: self.core.scheduler_tx.clone(),
        }
    }

    /// A handle that scores probes raw against a belief group on the
    /// scheduler's GPU gallery arena without holding the engine — see
    /// [`GroupScanner::scan`].
    pub fn group_scanner(&self) -> GroupScanner {
        GroupScanner {
            conversation: self.conversation.clone(),
            scheduler_tx: self.core.scheduler_tx.clone(),
        }
    }

    /// Warm one belief group's per-timeline backgrounds by self-match — for a
    /// group nothing else teaches, such as a tag-scoped turn group, which
    /// learns only from probes inside its scope. Returns how many timelines
    /// were warmed. See
    /// [`crate::projection::Conversation::warm_group_normalization`].
    pub fn warm_group_normalization(
        &self,
        schema: &crate::projection::Schema,
        group: GroupId,
    ) -> usize {
        self.conversation.warm_group_normalization(schema, group)
    }

    /// Warm one timeline's backgrounds by self-match — a conversation written
    /// onto a belief group after the group was warmed. See
    /// [`crate::projection::Conversation::warm_timeline_normalization`].
    pub fn warm_timeline_normalization(
        &self,
        schema: &crate::projection::Schema,
        timeline: TimelineId,
    ) -> bool {
        self.conversation
            .warm_timeline_normalization(schema, timeline)
    }

    /// Warm the belief-driven section collections' per-member backgrounds from
    /// their own tag-scoped corpus. Call after load, once the corpus is stable —
    /// without it a collection's levels are cold on every process start but the
    /// one that built the corpus, which changes both the scale and the RANKING of
    /// its scores. Runs on the scheduler thread, where the GPU gallery arena
    /// scores the corpus in batch, and blocks until it is done. See
    /// [`Conversation::warm_collection_normalization`].
    pub fn warm_collection_normalization(&self, schema: &Schema) -> CollectionWarm {
        let (tx, rx) = flume::bounded(1);
        if self
            .core
            .scheduler_tx
            .send(SchedulerRequest::WarmCollectionNormalization {
                conversation: self.conversation.clone(),
                schema: Box::new(schema.clone()),
                response_tx: tx,
            })
            .is_err()
        {
            return CollectionWarm::default();
        }
        rx.recv().unwrap_or_default()
    }

    /// Queue the dialogue normalization replay now, rather than at the first
    /// reprojection that would ask for it, so a restarted daemon re-learns its
    /// score backgrounds before conversations need them. Returns at once; the
    /// scheduler scores the replay a turn per loop iteration, and
    /// [`Self::warm_replay_progress`] reports how far it has got. Scored for
    /// the timeline of the last dialogue turn the replay takes
    /// ([`Conversation::warm_replay_target`]); returns whether one was sent —
    /// `false` with no dialogue turn to replay, or the scheduler gone.
    pub fn warm_dialogue_normalization(
        &self,
        projection: &Builder,
        max_probe_tokens: usize,
    ) -> bool {
        let Some(target) = self.conversation.warm_replay_target() else {
            return false;
        };
        self.core
            .scheduler_tx
            .send(SchedulerRequest::WarmDialogueNormalization {
                conversation: self.conversation.clone(),
                projection: Arc::new(projection.clone()),
                target,
                max_probe_tokens: max_probe_tokens.max(1),
            })
            .is_ok()
    }

    /// `(scored, total)` of the dialogue normalization replay now draining;
    /// `None` when none is.
    pub fn warm_replay_progress(&self) -> Option<(usize, usize)> {
        self.conversation.warm_replay_progress()
    }

    /// Preload what a projection's belief scan reads — every scanned group's
    /// signatures and every collection's gallery, decoded and resident on the
    /// GPU gallery arena — so the first projection after a start does none of
    /// it. Runs on the scheduler thread, which owns the arena, and blocks until
    /// it is done. See [`Conversation::preload_galleries`].
    pub fn preload_galleries(&self, schema: &Schema) -> GalleryPreload {
        let (tx, rx) = flume::bounded(1);
        if self
            .core
            .scheduler_tx
            .send(SchedulerRequest::PreloadGalleries {
                conversation: self.conversation.clone(),
                schema: Box::new(schema.clone()),
                response_tx: tx,
            })
            .is_err()
        {
            return GalleryPreload::default();
        }
        rx.recv().unwrap_or_default()
    }
}
