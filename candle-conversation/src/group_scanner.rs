//! Scoring probes raw against a belief group where the GPU gallery arena is —
//! the scheduler thread — for a caller outside the engine: an offline harness
//! that weighs, gates and folds the votes itself, against the gallery the
//! daemon already holds resident, instead of uploading its own beside it.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use flume::Sender;

use crate::persistence::streams::StreamId;
use crate::projection::{Conversation, ExclusionReport, GroupId, GroupVotes};
use crate::provenance::gallery_arena::{GroupSet, Keep};
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
    /// Learn `group`'s anomalous-chunk exclusion from up to `per_file` of each
    /// file's calibration questions, scanned here against its content alone,
    /// and set it for every live scan — see
    /// [`Conversation::learn_chunk_exclusion`]. Blocks for the scan; the
    /// attribution runs on the calling thread, holding no lock.
    ///
    /// Each competition is scanned in the fold groups its live scans weigh, of
    /// the tokens they keep — `content` for the files' own turns,
    /// `calibration` for their calibrations — so the learn reads what those
    /// scans read, and never widens a turn they keep narrow
    /// (`docs/gallery_residency.md` §4, `docs/gallery_dedup.md` §3).
    pub fn learn_chunk_exclusion(
        &self,
        group: GroupId,
        per_file: usize,
        content: (GroupSet, Keep),
        calibration: (GroupSet, Keep),
    ) -> candle::Result<ExclusionReport> {
        self.conversation
            .learn_chunk_exclusion(group, per_file, |calibrations, masked, probes| {
                let (groups, keep) = if calibrations { calibration } else { content };
                match self.scan(
                    group,
                    calibrations,
                    masked,
                    HashMap::new(),
                    probes,
                    Some(groups),
                    keep,
                ) {
                    Ok(votes) => votes,
                    Err(refused) => Err(candle::Error::Msg(format!(
                        "the chunk-exclusion scan was refused: {refused:?}"
                    ))),
                }
            })
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
