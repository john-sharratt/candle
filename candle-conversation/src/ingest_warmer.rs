//! Warming the ingest layers' per-file normalization hit levels where the GPU
//! gallery arena is — the scheduler thread — a slice of files at a time.
//!
//! Every ingested file's hit levels are learned by self-match: a few of its own
//! turns scored against its own exchanges. On the host that is a probe × window
//! scan per probe, and over 1,344 files it held eight cores for 270 s after
//! every start. On the arena each file costs one batched launch for all of its
//! probes. The arena belongs to the scheduler thread, so the work is sent there
//! in slices small enough that decode waves keep running between them; a
//! scheduler without an arena (a host with no CUDA) answers so, and the whole
//! warm-up then runs on the host's warm pool as before.

use std::sync::Arc;
use std::time::Instant;

use flume::Sender;

use crate::cancel::ingest_cancelled;
use crate::projection::{Conversation, GroupId, LayerId, Schema, TimelineId};
use crate::scheduler::SchedulerRequest;

/// Files per slice. One slice is one uninterrupted stretch of scheduler time,
/// a few milliseconds per file on the arena, so this bounds how long a decode
/// wave can wait behind the warm-up.
const WARM_SLICE_TIMELINES: usize = 16;

/// A handle that runs the ingest warm-up without holding the engine: the
/// conversation it warms and the scheduler that owns the arena. Taken from
/// [`crate::ConversationEngine::ingest_warmer`].
pub struct IngestWarmer {
    pub(crate) conversation: Conversation,
    pub(crate) scheduler_tx: Sender<SchedulerRequest>,
}

impl IngestWarmer {
    /// Warm every ingest group's per-file hit levels. Blocks until done. Call
    /// after an ingest pass has finished — never concurrently with an ingest
    /// writer, whose substrate lock the assembly reads under.
    pub fn run(&self, schema: &Schema) {
        let started = Instant::now();
        let schema = Arc::new(schema.clone());
        let mut warmed = 0usize;
        for (layer, group, timelines) in self.conversation.ingest_warm_work(&schema) {
            for slice in timelines.chunks(WARM_SLICE_TIMELINES) {
                if ingest_cancelled() {
                    return;
                }
                match self.warm_slice(&schema, layer, group, slice) {
                    Slice::Warmed(n) => warmed += n,
                    Slice::NoArena => {
                        // A scheduler without an arena: the host path, once,
                        // for everything.
                        self.conversation.warm_ingest_normalization(&schema);
                        return;
                    }
                    // The scheduler has shut down; nothing is left to serve.
                    Slice::Gone => return,
                }
            }
        }
        tracing::info!(
            timelines = warmed,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "normalization warm-up: learned per-file hit levels for ingest-layer timelines \
             on the GPU gallery arena"
        );
    }

    /// One slice on the scheduler thread.
    fn warm_slice(
        &self,
        schema: &Arc<Schema>,
        layer: LayerId,
        group: GroupId,
        timelines: &[TimelineId],
    ) -> Slice {
        let (tx, rx) = flume::bounded(1);
        let sent = self.scheduler_tx.send(SchedulerRequest::WarmIngestSlice {
            conversation: self.conversation.clone(),
            schema: Arc::clone(schema),
            layer,
            group,
            timelines: timelines.to_vec(),
            response_tx: tx,
        });
        if sent.is_err() {
            return Slice::Gone;
        }
        match rx.recv() {
            Ok(Some(n)) => Slice::Warmed(n),
            Ok(None) => Slice::NoArena,
            Err(_) => Slice::Gone,
        }
    }
}

/// What one slice came back as.
enum Slice {
    /// Warmed on the arena: how many timelines.
    Warmed(usize),
    /// The scheduler has no arena.
    NoArena,
    /// The scheduler has shut down.
    Gone,
}
