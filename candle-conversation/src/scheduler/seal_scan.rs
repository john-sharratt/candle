//! The seal scan, run on the scheduler's thread.
//!
//! When a turn seals, its stored wide-Q signature is scored against every
//! belief node once more, and that scan — unlike every live reprojection —
//! teaches the score-normalization hit levels. The caller holds the turn; the
//! scheduler holds the gallery arena, and the arena has exactly one scan thread
//! (a re-seal frees a turn's pages on the grounds that nobody else can be
//! holding a scan pin on them). So the caller sends the probe here
//! ([`SchedulerRequest::ScoreSealedTurn`](super::SchedulerRequest::ScoreSealedTurn))
//! and the scan runs as the same paged launch the reprojection uses, over the
//! same resident pages and usually the same cached index.

use super::Scheduler;
use crate::projection::{Builder, Conversation, Observe, ProjectionTarget};
use crate::provenance::WideQSig;
use crate::substrate::ProjectionScores;

/// One sealed turn's scan inputs, as the caller gathered them from the
/// substrate.
pub(crate) struct SealedProbe {
    /// The turn's head + tail signature window.
    pub probe: Vec<WideQSig>,
    /// The turn's question window; empty when it has none.
    pub probe_q: Vec<WideQSig>,
    /// The turn's gather-scope tags — the scopes this observation may teach.
    pub tags: Vec<String>,
    /// The turn's stream id, so a replay of the same turn folds nothing twice.
    pub source: u64,
}

impl Scheduler {
    /// Score `sealed` against every belief node of `projection`'s schema and
    /// fold it into the normalization levels of the scopes it belongs to.
    pub(super) fn score_sealed_turn(
        &self,
        substrate: &Conversation,
        projection: &Builder,
        target: ProjectionTarget,
        sealed: &SealedProbe,
    ) -> ProjectionScores {
        let (scores, _) = substrate.score_beliefs(
            projection.schema(),
            target,
            &sealed.probe,
            (!sealed.probe_q.is_empty()).then_some(sealed.probe_q.as_slice()),
            Observe::Yes {
                tags: &sealed.tags,
                source: sealed.source,
            },
            self.gallery_arena.as_deref(),
        );
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_substrate::{open, record_turns, sig, YAML};
    use super::super::tests::make_test_scheduler;
    use super::super::SchedulerRequest;
    use super::SealedProbe;
    use crate::projection::{Builder, Conversation, Observe, ProjectionTarget};
    use crate::provenance::WideQSig;
    use std::sync::Arc;

    /// Two exemplars per member; returns the target.
    fn record_corpus(conv: &Conversation, builder: &Builder) -> ProjectionTarget {
        record_turns(
            conv,
            builder,
            &[
                (&["tool", "alpha"], 0xAAAA_AAAA_AAAA_AAAA, 40),
                (&["tool", "alpha"], 0xABAB_ABAB_ABAB_ABAB, 17),
                (&["tool", "beta"], 0x5555_5555_5555_5555, 33),
                (&["tool", "beta"], 0x1234_5678_9ABC_DEF0, 8),
            ],
        )
    }

    /// **A seal scan sent to the scheduler teaches exactly what scanning in
    /// place taught.** The request carries the probe, the question window, the
    /// tags and the source across the channel; losing any of them scores or
    /// teaches differently, and the levels read back afterwards say so.
    #[test]
    fn a_seal_scan_through_the_scheduler_learns_what_an_in_place_scan_learned() {
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let sp = &builder.schema().system_prompt;
        let sealed = SealedProbe {
            probe: (0..12).map(|_| sig(0xAAAA_AAAA_AAAA_AAAA)).collect(),
            probe_q: (0..5).map(|_| sig(0xABAB_ABAB_ABAB_ABAB)).collect(),
            tags: vec!["tool".to_string()],
            source: 0x5EA1,
        };
        let later: Vec<WideQSig> = (0..6).map(|_| sig(0x5555_5555_5555_5555)).collect();

        // The reference: the scan as the caller used to run it, in place.
        let ref_dir = tempfile::tempdir().unwrap();
        let reference = open(ref_dir.path());
        let target = record_corpus(&reference, &builder);
        let (want, _) = reference.score_beliefs(
            builder.schema(),
            target,
            &sealed.probe,
            Some(sealed.probe_q.as_slice()),
            Observe::Yes {
                tags: &sealed.tags,
                source: sealed.source,
            },
            None,
        );

        let dir = tempfile::tempdir().unwrap();
        let conv = open(dir.path());
        record_corpus(&conv, &builder);
        let cold = conv.score_belief_collections(sp, &later, None, Observe::No, None);
        let (mut sched, _tx) = make_test_scheduler();
        let (tx, rx) = flume::bounded(1);
        sched.handle_request(SchedulerRequest::ScoreSealedTurn {
            substrate: conv.clone(),
            projection: Arc::clone(&builder),
            target,
            sealed: SealedProbe {
                probe: sealed.probe.clone(),
                probe_q: sealed.probe_q.clone(),
                tags: sealed.tags.clone(),
                source: sealed.source,
            },
            response_tx: tx,
        });
        let got = rx.recv().expect("the scheduler answers");

        let coll = sp.collection_named("tools").expect("tools collection");
        let learned_want = reference.score_belief_collections(sp, &later, None, Observe::No, None);
        let learned_got = conv.score_belief_collections(sp, &later, None, Observe::No, None);
        for s in &coll.sections {
            assert_eq!(
                got.section(s.id).to_bits(),
                want.section(s.id).to_bits(),
                "section {}: the seal scan scored differently through the scheduler",
                s.name
            );
            assert_eq!(
                learned_got.section(s.id).to_bits(),
                learned_want.section(s.id).to_bits(),
                "section {}: the seal scan taught a different level",
                s.name
            );
        }
        assert!(
            coll.sections
                .iter()
                .any(|s| cold.section(s.id) != learned_got.section(s.id)),
            "the seal scan taught nothing — a later probe scores exactly as it did cold"
        );
    }
}
