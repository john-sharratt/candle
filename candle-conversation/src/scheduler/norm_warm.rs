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
//! The replay is background work, so it runs at ingest priority: one probe per
//! loop iteration, and none while a higher-priority conversation holds the
//! device or is in its cooldown ([`super::priority_pause`]). A live turn never
//! waits on it.
//!
//! The ingest layers' per-file warm-up ([`crate::IngestWarmer`]) arrives here
//! as slices and waits in the same backlog, under the same rule. Served the
//! moment it arrived, each slice was ~300 ms of self-match scans the decode
//! loop stood behind, once per decode step: a dialogue opened right after a
//! load decoded at a fraction of its rate until the ~30 s warm-up was done.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use flume::Sender;

use super::priority_pause::PAUSED_POLL;
use super::Scheduler;
use crate::projection::{
    Builder, Conversation, DecodePriority, GroupId, LayerId, ProjectionTarget, Schema, TimelineId,
    WarmProbe,
};

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

/// The replay's backlog, and what the current one has done so far.
#[derive(Default)]
pub(super) struct NormWarm {
    backlog: VecDeque<WarmJob>,
    /// Ingest warm-up slices, served after the replay's turns.
    ingest: VecDeque<IngestSliceJob>,
    /// When the replay now draining was queued.
    started: Option<Instant>,
    probes: usize,
    probe_windows: usize,
}

impl NormWarm {
    /// Whether any replayed turn or ingest slice is waiting.
    pub(super) fn pending(&self) -> bool {
        !self.backlog.is_empty() || !self.ingest.is_empty()
    }
}

impl Scheduler {
    /// How long an otherwise idle loop may wait for a request before it owes
    /// the warm-up its next turn: `None` with nothing queued (block as usual),
    /// zero when a turn can run now, and [`PAUSED_POLL`] while ingest-priority
    /// work is paused — so a paused replay neither spins nor delays a request.
    pub(super) fn norm_warm_wait(&self) -> Option<Duration> {
        if !self.norm_warm.pending() {
            return None;
        }
        let paused = self
            .priority_pause
            .paused(DecodePriority::Low, Instant::now());
        Some(if paused { PAUSED_POLL } else { Duration::ZERO })
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

    /// Queue one ingest warm-up slice; [`Self::step_norm_warm`] scores it when
    /// ingest-priority work may run.
    pub(super) fn queue_ingest_warm_slice(&mut self, job: IngestSliceJob) {
        self.norm_warm.ingest.push_back(job);
    }

    /// Score one queued replay turn — or, with none left, one ingest slice —
    /// unless ingest-priority work is paused right now. Returns whether one was
    /// scored.
    pub(super) fn step_norm_warm(&mut self) -> bool {
        if self.norm_warm_wait() != Some(Duration::ZERO) {
            return false;
        }
        let Some(job) = self.norm_warm.backlog.pop_front() else {
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
    use super::{IngestSliceJob, PAUSED_POLL};
    use crate::projection::{Builder, DecodePriority};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// **The replay queues once per arming, and is scored only in the gaps a
    /// dialogue leaves.** Two dialogue turns and one tool exemplar: the replay
    /// is the two untagged turns. While a higher-priority conversation holds
    /// the device the loop may wait the pause poll and scores nothing; once it
    /// has gone, each step scores one turn, and the drained backlog stops
    /// shortening the idle wait. A re-arm queues the replay again.
    #[test]
    fn the_warm_replay_waits_behind_a_dialogue_and_drains_after_it() {
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

        sched
            .priority_pause
            .observe(DecodePriority::High, Instant::now());
        assert_eq!(sched.norm_warm_wait(), Some(PAUSED_POLL));
        assert!(!sched.step_norm_warm(), "a dialogue holds the device");
        assert_eq!(sched.norm_warm.backlog.len(), 2);

        sched.priority_pause = Default::default();
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

    /// **An ingest warm-up slice is answered in the gaps a dialogue leaves, not
    /// when it arrives.** While a higher-priority conversation holds the device
    /// the slice waits and its warmer hears nothing; once it has gone the next
    /// step answers it — here `None`, a scheduler with no arena — and the
    /// backlog stops shortening the idle wait.
    #[test]
    fn an_ingest_warm_slice_waits_behind_a_dialogue() {
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

        sched
            .priority_pause
            .observe(DecodePriority::High, Instant::now());
        assert_eq!(sched.norm_warm_wait(), Some(PAUSED_POLL));
        assert!(!sched.step_norm_warm(), "a dialogue holds the device");
        assert!(reply_rx.try_recv().is_err(), "the warmer is still waiting");

        sched.priority_pause = Default::default();
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
