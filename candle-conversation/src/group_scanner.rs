//! Scoring probes raw against a belief group where the GPU gallery arena is —
//! the scheduler thread — for a caller outside the engine: an offline harness
//! that weighs, gates and folds the votes itself, against the gallery the
//! daemon already holds resident, instead of uploading its own beside it.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use flume::Sender;

use crate::persistence::streams::StreamId;
use crate::projection::{Conversation, GroupId, GroupVotes};
use crate::provenance::WideQSig;
use crate::scheduler::SchedulerRequest;

/// A handle that scans without holding the engine: the conversation it reads
/// and the scheduler that owns the arena. Taken from
/// [`crate::ConversationEngine::group_scanner`].
pub struct GroupScanner {
    pub(crate) conversation: Conversation,
    pub(crate) scheduler_tx: Sender<SchedulerRequest>,
}

/// Why a scan answered nothing.
#[derive(Debug, PartialEq, Eq)]
pub enum ScanRefused {
    /// The scheduler has no GPU gallery arena.
    NoArena,
    /// The scheduler has shut down.
    Gone,
}

impl GroupScanner {
    /// `probes`' raw votes against every active file of `group` — see
    /// [`Conversation::scan_group_votes`]. Blocks until the scheduler, at
    /// ingest priority, has run it.
    pub fn scan(
        &self,
        group: GroupId,
        calibrations: bool,
        masked: HashSet<StreamId>,
        excluded: HashMap<StreamId, Vec<Range<usize>>>,
        probes: Vec<Vec<WideQSig>>,
    ) -> Result<candle::Result<GroupVotes>, ScanRefused> {
        let (tx, rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::ScanGroupVotes {
                conversation: self.conversation.clone(),
                group,
                calibrations,
                masked,
                excluded,
                probes,
                response_tx: tx,
            })
            .map_err(|_| ScanRefused::Gone)?;
        match rx.recv() {
            Ok(Some(votes)) => Ok(votes),
            Ok(None) => Err(ScanRefused::NoArena),
            Err(_) => Err(ScanRefused::Gone),
        }
    }
}
