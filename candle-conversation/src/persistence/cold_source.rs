//! What a cold load reads through, taken from the log's owner in one call and
//! read without it.
//!
//! A cold load moves a stream's chunks from disk to the device: a pool of
//! reader threads, a host-side allocator and the device's own thread
//! (`pipeline`). None of that runs on the log's owner. The owner plans the
//! read and hands over the direct-I/O handles it needs — after flushing what it
//! staged, so every chunk the plan names is in its file — and the load runs on
//! the caller's threads. The handles are shared owners of the files: a segment
//! a compaction replaces in the meantime stays readable through them.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;

use candle::direct_io::DirectFile;

use super::chunk_plan::{ChunkedReadPlan, SourceLog};
use super::inherit::InheritedSubstrate;
use super::segment::SegmentId;
use super::streams::StreamId;
use super::{Result, SubstratePersistence};
use crate::substrate::Substrate;

/// See the module documentation.
pub struct ColdSource {
    active: Arc<DirectFile>,
    sealed: HashMap<SegmentId, Arc<DirectFile>>,
    inherited: Vec<Arc<InheritedSubstrate>>,
}

impl ColdSource {
    /// The handle set a planned read of `source` goes through, or `None` for a
    /// sealed segment the plan did not name.
    pub fn direct(&self, source: SourceLog) -> Option<&DirectFile> {
        match source {
            SourceLog::Active => Some(&self.active),
            SourceLog::Sealed(id) => self.sealed.get(&id).map(|d| d.as_ref()),
            SourceLog::Inherited(i) => self.inherited.get(i).map(|s| s.direct_file()),
        }
    }

    /// Reader threads a read may run: one per handle of the active set.
    pub fn readers(&self) -> usize {
        self.active.n_handles()
    }
}

impl SubstratePersistence {
    /// Plan a cold read of `stream_id` in batches of at most `buffer_size`
    /// bytes, with what it reads through. Flushes anything staged first, so
    /// every chunk `substrate` names is in its file.
    pub fn cold_read(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
        buffer_size: usize,
    ) -> Result<(ChunkedReadPlan, ColdSource)> {
        self.flush_staged()?;
        let plan = self.plan_chunked_read(substrate, stream_id, buffer_size);
        let mut sealed = HashMap::new();
        for batch in &plan.chunks {
            if let SourceLog::Sealed(id) = batch.source {
                if let Entry::Vacant(slot) = sealed.entry(id) {
                    slot.insert(self.segments.open_sealed_direct(id)?);
                }
            }
        }
        let source = ColdSource {
            active: self.segments.active_shared_direct(),
            sealed,
            inherited: self.inherited.clone(),
        };
        Ok((plan, source))
    }
}
