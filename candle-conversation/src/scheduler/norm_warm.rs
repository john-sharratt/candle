//! The dialogue normalization warm-up, scored on the scheduler's gallery arena.
//!
//! After a load, and after every ingest reconcile, the score-normalization hit
//! levels are re-learned by replaying recent sealed dialogue turns as
//! seal-style observations ([`Conversation::take_warm_replay`]). Each replayed
//! turn is a full belief scan — every collection and every layer's groups —
//! and scanned on the CPU that was ~12 s a probe over a 7M-token code-reading
//! layer: 402 s for a 32-turn replay, burning cores beside live turns after
//! every restart. Here each probe is the paged GPU launch the reprojection
//! uses, over the same resident pages.
//!
//! The replay is background work, so it is paced by the loop rather than by the
//! decode step: one probe per loop iteration, which a decode quantum bounds to
//! one per `WAVE_SLICE`. A live turn waits on at most one probe a quantum while
//! a warm-up drains, and the warm-up drains whatever else is running.
//!
//! The ingest layers' per-file warm-up ([`crate::IngestWarmer`]) arrives here
//! as slices and waits in the same backlog, under the same rule. Served the
//! moment it arrived, each slice was ~300 ms of self-match scans the decode
//! loop stood behind, once per decode step: a dialogue opened right after a
//! load decoded at a fraction of its rate until the ~30 s warm-up was done.
//!
//! A raw group scan ([`Conversation::scan_group_votes`]) — an offline caller
//! scoring its own probes against the resident gallery — waits in the same
//! loop under the same rule, after the replay and before ingest slices.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::sync::Arc;
use std::time::{Duration, Instant};

use flume::Sender;

use super::Scheduler;
use crate::persistence::streams::StreamId;
use crate::projection::{
    Builder, Conversation, GroupId, GroupVotes, LayerId, ProjectionTarget, Schema, TimelineId,
    WarmProbe,
};
use crate::provenance::gallery_arena::{GroupSet, Keep};
use crate::provenance::WideQSig;

/// One replayed turn waiting to be scored, with what scoring it needs.
struct WarmJob {
    substrate: Conversation,
    projection: Arc<Builder>,
    target: ProjectionTarget,
    probe: WarmProbe,
}

/// One slice of an ingest group's per-file warm-up, waiting to be scored on the
/// arena — the [`super::SchedulerRequest::WarmIngestSlice`] it arrived as.
pub(super) struct IngestSliceJob {
    pub(super) conversation: Conversation,
    pub(super) schema: Arc<Schema>,
    pub(super) layer: LayerId,
    pub(super) group: GroupId,
    pub(super) timelines: Vec<TimelineId>,
    /// The timelines warmed, or `None` when this scheduler has no arena.
    pub(super) response_tx: Sender<Option<usize>>,
}

/// A batch of probes scored raw against a belief group's files — the
/// [`super::SchedulerRequest::ScanGroupVotes`] it arrived as.
pub(super) struct ScanVotesJob {
    pub(super) conversation: Conversation,
    pub(super) group: GroupId,
    pub(super) calibrations: bool,
    pub(super) masked: HashSet<StreamId>,
    pub(super) excluded: HashMap<StreamId, Vec<Range<usize>>>,
    pub(super) probes: Vec<Vec<WideQSig>>,
    /// The fold groups to scan; `None` for every group.
    pub(super) groups: Option<GroupSet>,
    /// Which of the turns' tokens the card holds for the scan.
    pub(super) keep: Keep,
    /// The votes, or `None` when this scheduler has no arena.
    pub(super) response_tx: Sender<Option<candle::Result<GroupVotes>>>,
}

/// The replay's backlog, and what the current one has done so far.
#[derive(Default)]
pub(super) struct NormWarm {
    backlog: VecDeque<WarmJob>,
    /// Raw group scans, served after the replay's turns and before ingest
    /// slices.
    scans: VecDeque<ScanVotesJob>,
    /// Ingest warm-up slices, served after the replay's turns.
    ingest: VecDeque<IngestSliceJob>,
    /// When the replay now draining was queued.
    started: Option<Instant>,
    probes: usize,
    probe_windows: usize,
}

impl Scheduler {
    /// How long an otherwise idle loop may wait for a request before it owes
    /// the warm-up its next turn: `None` with nothing queued (block as usual),
    /// zero while anything is.
    pub(super) fn norm_warm_wait(&self) -> Option<Duration> {
        let warm = &self.norm_warm;
        if warm.backlog.is_empty() && warm.scans.is_empty() && warm.ingest.is_empty() {
            return None;
        }
        Some(Duration::ZERO)
    }

    /// Queue the dialogue warm-up replay when it is armed — see
    /// [`Conversation::take_warm_replay`]. A no-op otherwise, so it is cheap to
    /// call on every reprojection.
    pub(super) fn queue_norm_warm(
        &mut self,
        substrate: &Conversation,
        projection: &Arc<Builder>,
        target: ProjectionTarget,
    ) {
        let Some(probes) = substrate.take_warm_replay(target) else {
            return;
        };
        let warm = &mut self.norm_warm;
        warm.started.get_or_insert_with(Instant::now);
        warm.backlog.extend(probes.into_iter().map(|probe| WarmJob {
            substrate: substrate.clone(),
            projection: Arc::clone(projection),
            target,
            probe,
        }));
    }

    /// Queue one ingest warm-up slice; [`Self::step_norm_warm`] scores it on a
    /// later loop iteration.
    pub(super) fn queue_ingest_warm_slice(&mut self, job: IngestSliceJob) {
        self.norm_warm.ingest.push_back(job);
    }

    /// Queue one raw group scan; [`Self::step_norm_warm`] runs it on a later
    /// loop iteration.
    pub(super) fn queue_group_scan(&mut self, job: ScanVotesJob) {
        self.norm_warm.scans.push_back(job);
    }

    /// Drop every queued job for `conversation`'s substrate — it has detached,
    /// and nothing on this thread may refer to it afterwards. A dropped
    /// slice's reply channel closes, which its warmer reads as the engine
    /// having gone.
    pub(super) fn forget_norm_warm(&mut self, conversation: &Conversation) {
        let warm = &mut self.norm_warm;
        warm.backlog
            .retain(|job| !job.substrate.same_substrate(conversation));
        warm.scans
            .retain(|scan| !scan.conversation.same_substrate(conversation));
        warm.ingest
            .retain(|slice| !slice.conversation.same_substrate(conversation));
        if warm.backlog.is_empty() {
            warm.started = None;
        }
    }

    /// Score one queued replay turn — or, with none left, one raw group scan,
    /// then one ingest slice. Returns whether one was scored.
    pub(super) fn step_norm_warm(&mut self) -> bool {
        let Some(job) = self.norm_warm.backlog.pop_front() else {
            if let Some(scan) = self.norm_warm.scans.pop_front() {
                let votes = self.gallery_arena.as_deref().map(|arena| {
                    scan.conversation.scan_group_votes(
                        scan.group,
                        scan.calibrations,
                        &scan.masked,
                        &scan.excluded,
                        &scan.probes,
                        scan.groups,
                        scan.keep,
                        arena,
                    )
                });
                // A caller that has gone away has nothing left to wait for.
                let _ = scan.response_tx.send(votes);
                return true;
            }
            let Some(slice) = self.norm_warm.ingest.pop_front() else {
                return false;
            };
            let warmed = self.gallery_arena.as_deref().map(|arena| {
                slice.conversation.warm_ingest_timelines(
                    &slice.schema,
                    slice.layer,
                    slice.group,
                    &slice.timelines,
                    arena,
                )
            });
            // A warmer that has gone away has nothing left to wait for.
            let _ = slice.response_tx.send(warmed);
            return true;
        };
        job.substrate.score_warm_probe(
            job.projection.schema(),
            job.target,
            &job.probe,
            self.gallery_arena.as_deref(),
        );
        let warm = &mut self.norm_warm;
        warm.probes += 1;
        warm.probe_windows += job.probe.probe.len();
        if warm.backlog.is_empty() {
            // Its cost depends on which stored turns it replays, so it is
            // logged: a warm-up runs beside live turns and is invisible
            // otherwise.
            tracing::info!(
                target: "candle_conversation::provenance",
                probes = warm.probes,
                probe_windows = warm.probe_windows,
                elapsed_ms = warm
                    .started
                    .take()
                    .map_or(0, |t| t.elapsed().as_millis() as u64),
                "normalization warm-up: replayed dialogue turns"
            );
            warm.probes = 0;
            warm.probe_windows = 0;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_substrate::{open, record_turns, YAML};
    use super::super::tests::make_test_scheduler;
    use super::IngestSliceJob;
    use crate::projection::Builder;
    use std::sync::Arc;
    use std::time::Duration;

    /// **The replay queues once per arming, and each step scores one turn.**
    /// Two dialogue turns and one tool exemplar: the replay is the two untagged
    /// turns. Each step scores one, and the drained backlog stops shortening
    /// the idle wait. A re-arm queues the replay again.
    #[test]
    fn the_warm_replay_scores_a_turn_a_step_until_drained() {
        let dir = tempfile::tempdir().unwrap();
        let conv = open(dir.path());
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let target = record_turns(
            &conv,
            &builder,
            &[
                (&[], 0xAAAA_AAAA_AAAA_AAAA, 20),
                (&["tool", "alpha"], 0x5555_5555_5555_5555, 20),
                (&[], 0xABAB_ABAB_ABAB_ABAB, 12),
            ],
        );
        let (mut sched, _tx) = make_test_scheduler();
        assert_eq!(
            sched.norm_warm_wait(),
            None,
            "nothing queued blocks as usual"
        );

        sched.queue_norm_warm(&conv, &builder, target);
        assert_eq!(sched.norm_warm.backlog.len(), 2, "the two dialogue turns");
        sched.queue_norm_warm(&conv, &builder, target);
        assert_eq!(
            sched.norm_warm.backlog.len(),
            2,
            "an armed replay is taken once"
        );

        assert_eq!(sched.norm_warm_wait(), Some(Duration::ZERO));
        assert!(sched.step_norm_warm());
        assert_eq!(sched.norm_warm.backlog.len(), 1);
        assert!(sched.step_norm_warm());
        assert!(!sched.step_norm_warm(), "the backlog is drained");
        assert_eq!(sched.norm_warm_wait(), None);

        conv.reset_normalization_warm();
        sched.queue_norm_warm(&conv, &builder, target);
        assert_eq!(sched.norm_warm.backlog.len(), 2, "a re-arm queues it again");
    }

    /// **A raw group scan is answered on a later step, not when it arrives** —
    /// here `None`, a scheduler with no arena — and the drained queue stops
    /// shortening the idle wait.
    #[test]
    fn a_group_scan_is_answered_on_the_next_step() {
        use super::ScanVotesJob;
        use crate::projection::GroupId;
        use std::collections::{HashMap, HashSet};

        let dir = tempfile::tempdir().unwrap();
        let conv = open(dir.path());
        let (mut sched, _tx) = make_test_scheduler();
        let (reply_tx, reply_rx) = flume::bounded(1);
        sched.queue_group_scan(ScanVotesJob {
            conversation: conv,
            group: GroupId::from_raw(1).unwrap(),
            calibrations: true,
            masked: HashSet::new(),
            excluded: HashMap::new(),
            probes: Vec::new(),
            groups: None,
            keep: super::Keep::All,
            response_tx: reply_tx,
        });
        assert_eq!(sched.norm_warm_wait(), Some(Duration::ZERO));
        assert!(reply_rx.try_recv().is_err(), "the caller is still waiting");

        assert!(sched.step_norm_warm());
        assert!(
            reply_rx.try_recv().unwrap().is_none(),
            "no arena on this scheduler"
        );
        assert!(!sched.step_norm_warm(), "the scan was the whole backlog");
        assert_eq!(sched.norm_warm_wait(), None);
    }

    /// **An ingest warm-up slice is answered on a later step, not when it
    /// arrives.** Its warmer hears nothing until the step answers it — here
    /// `None`, a scheduler with no arena — and the backlog then stops
    /// shortening the idle wait.
    #[test]
    fn an_ingest_warm_slice_is_answered_on_the_next_step() {
        let dir = tempfile::tempdir().unwrap();
        let conv = open(dir.path());
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let layer = &builder.schema().layers[0];
        let (mut sched, _tx) = make_test_scheduler();
        let (reply_tx, reply_rx) = flume::bounded(1);
        sched.queue_ingest_warm_slice(IngestSliceJob {
            conversation: conv.clone(),
            schema: Arc::new(builder.schema().clone()),
            layer: layer.id,
            group: layer.groups[0].id,
            timelines: Vec::new(),
            response_tx: reply_tx,
        });
        assert_eq!(sched.norm_warm_wait(), Some(Duration::ZERO));
        assert!(reply_rx.try_recv().is_err(), "the warmer is still waiting");

        assert!(sched.step_norm_warm());
        assert_eq!(
            reply_rx.try_recv().unwrap(),
            None,
            "no arena on this scheduler"
        );
        assert!(!sched.step_norm_warm(), "the slice was the whole backlog");
        assert_eq!(sched.norm_warm_wait(), None);
    }
}
