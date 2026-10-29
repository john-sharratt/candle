//! The ingest demote, sent to the substrate rather than applied under its lock
//! by the scheduler (`docs/substrate_mutation_queue.md` §2).
//!
//! Applied inline, the demote waited for the substrate's write lock on the
//! scheduler thread behind maintenance's scans — 56 waits of up to 715 ms in
//! ten minutes on BlackWorld — and passed over when the lock was held, it was
//! passed over nearly every wave and the ingest backed up behind it. Sent, it
//! is applied by the substrate's next writer or its applier: never skipped, and
//! the scheduler never waits. What it took out of the substrate comes back
//! here, and the scheduler drops the hot copies on its own thread.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_nn::kv_cache::SealedSequence;
use flume::{Receiver, Sender};

use crate::projection::{Conversation, SectionId, TimelineId, TurnKey};
use crate::substrate::EvictionReport;

/// What a demote sheds toward, and what it must keep — the arguments of
/// [`Substrate::demote_cold_ingest`](crate::substrate::Substrate::demote_cold_ingest).
pub(super) struct DemoteOrder {
    pub ingest: HashSet<TimelineId>,
    pub keep_turns: Vec<TurnKey>,
    pub keep_sections: Vec<SectionId>,
    pub window: usize,
    pub target: u64,
}

/// One applied demote: what it shed, toward what target, how long the apply
/// held the substrate, and the hot copies it took out of it.
pub(super) struct ColdDemoted {
    pub report: EvictionReport,
    pub target: u64,
    pub took: Duration,
    pub hot: Vec<Vec<SealedSequence>>,
}

/// See the module documentation.
pub(super) struct IngestDemotes {
    tx: Sender<ColdDemoted>,
    rx: Receiver<ColdDemoted>,
    /// Demotes sent and not yet applied.
    in_flight: Arc<AtomicUsize>,
}

impl Default for IngestDemotes {
    fn default() -> Self {
        let (tx, rx) = flume::unbounded();
        IngestDemotes {
            tx,
            rx,
            in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl IngestDemotes {
    /// Whether a demote is sent and not yet applied. One is enough: another,
    /// sent from the same pressure toward the same target, would shed it twice.
    pub fn in_flight(&self) -> bool {
        self.in_flight.load(Ordering::Acquire) > 0
    }

    /// Send `order` to `conversation`'s substrate.
    pub fn send(&self, conversation: &Conversation, order: DemoteOrder) {
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        let tx = self.tx.clone();
        let in_flight = Arc::clone(&self.in_flight);
        conversation.send(move |substrate| {
            let applied = Instant::now();
            let (report, hot) = substrate.demote_cold_ingest(
                &order.ingest,
                &order.keep_turns,
                &order.keep_sections,
                order.window,
                order.target,
            );
            // Back before it stops counting as in flight, so a scheduler that
            // finds none in flight has already got every result.
            let _ = tx.send(ColdDemoted {
                report,
                target: order.target,
                took: applied.elapsed(),
                hot,
            });
            in_flight.fetch_sub(1, Ordering::AcqRel);
        });
    }

    /// Every demote applied since the last call, oldest first.
    pub fn landed(&self) -> Vec<ColdDemoted> {
        self.rx.try_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A sent demote is in flight until it is applied, then comes back
    /// once** — with its target, whatever it shed — and the next write
    /// applies it if the applier has not.
    #[test]
    fn a_sent_demote_comes_back_once_applied() {
        let conversation = Conversation::new();
        let demotes = IngestDemotes::default();
        demotes.send(
            &conversation,
            DemoteOrder {
                ingest: HashSet::new(),
                keep_turns: Vec::new(),
                keep_sections: Vec::new(),
                window: 0,
                target: 4_096,
            },
        );
        drop(conversation.write());
        assert!(!demotes.in_flight());
        let landed = demotes.landed();
        assert_eq!(landed.len(), 1);
        assert_eq!(landed[0].target, 4_096);
        assert_eq!((landed[0].report.count, landed[0].report.bytes), (0, 0));
        assert!(landed[0].hot.is_empty());
        assert!(demotes.landed().is_empty(), "a result comes back once");
    }
}
