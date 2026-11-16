//! Scoring probes raw against a belief group where the GPU gallery arena is —
//! the scheduler thread — for a caller outside the engine: an offline harness
//! that weighs, gates and folds the votes itself, against the gallery the
//! daemon already holds resident, instead of uploading its own beside it.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use flume::Sender;

use crate::persistence::streams::StreamId;
use crate::projection::{Conversation, ExclusionLearner, FoldReport, GroupId, GroupVotes};
use crate::provenance::gallery_arena::{GroupSet, Keep};
use crate::provenance::WideQSig;
use crate::scheduler::{RemoteArena, SchedulerRequest};

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
    /// Fold what changed in the learner's group into its anomalous-chunk
    /// exclusion, and write its records and cut — see
    /// [`ExclusionLearner::fold`]. Its launches run on the scheduler's wave
    /// thread, each a small job of its own (`RemoteArena`); the assembly, the
    /// merging and the cut run on the calling thread, holding no lock.
    ///
    /// Each competition is scanned in the fold groups its live scans weigh, of
    /// the tokens they keep — `content` for the files' own turns,
    /// `calibration` for their calibrations — so the learner reads what those
    /// scans read, and never widens a turn they keep narrow
    /// (`docs/gallery_residency.md` §4, `docs/gallery_dedup.md` §3).
    pub fn fold_chunk_exclusion(
        &self,
        learner: &mut ExclusionLearner,
        content: (GroupSet, Keep),
        calibration: (GroupSet, Keep),
    ) -> Result<candle::Result<FoldReport>, ScanRefused> {
        let arena = RemoteArena::connect(&self.scheduler_tx).ok_or(ScanRefused::NoArena)?;
        Ok(learner.fold(&self.conversation, &arena, content, calibration))
    }

    /// `probes`' raw votes against every active file of `group` — see
    /// [`Conversation::scan_group_votes`] — in `groups` (`None` for every fold
    /// group), of the tokens `keep` has the card hold. Blocks until the
    /// scheduler, at ingest priority, has run it.
    #[allow(clippy::too_many_arguments)]
    pub fn scan(
        &self,
        group: GroupId,
        calibrations: bool,
        masked: HashSet<StreamId>,
        excluded: HashMap<StreamId, Vec<Range<usize>>>,
        probes: Vec<Vec<WideQSig>>,
        groups: Option<GroupSet>,
        keep: Keep,
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
                groups,
                keep,
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
