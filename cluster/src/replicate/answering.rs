//! The catch-up answers this member is running, one per requester (§7.2).
//!
//! Answering a `CATCHUP` lists every record the requester lacks, a header read
//! per record, and a member far behind lacks all of them: the answer can run
//! for minutes. A requester whose catch-up goes quiet asks again, and its gap
//! watch asks again while this member keeps writing — so an answer that is
//! never stopped runs beside each one asked after it, every one of them
//! re-listing the same history. They stacked to dozens on one member, read its
//! disk flat out and starved everything else that needs its log.
//!
//! So a requester has at most one answer running: a newer `CATCHUP` from it
//! cancels the one before, and an answer stops before its next range once it
//! is cancelled.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// One running answer: who it is for, and whether it should stop.
#[derive(Clone)]
pub struct Ticket {
    requester: u64,
    request: u64,
    cancelled: Arc<AtomicBool>,
}

impl Ticket {
    /// Whether the answer should go on to its next range.
    pub fn live(&self) -> bool {
        !self.cancelled.load(Ordering::Relaxed)
    }
}

/// The answer running for each requester.
#[derive(Default)]
pub struct Answering {
    running: HashMap<u64, Ticket>,
}

impl Answering {
    /// Start answering `request` from `requester`, cancelling the answer
    /// already running for it.
    pub fn begin(&mut self, requester: u64, request: u64) -> Ticket {
        let ticket = Ticket {
            requester,
            request,
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        if let Some(before) = self.running.insert(requester, ticket.clone()) {
            before.cancelled.store(true, Ordering::Relaxed);
        }
        ticket
    }

    /// `ticket`'s answer ended. A newer answer for the same requester is left
    /// running.
    pub fn end(&mut self, ticket: &Ticket) {
        if self
            .running
            .get(&ticket.requester)
            .is_some_and(|t| t.request == ticket.request)
        {
            self.running.remove(&ticket.requester);
        }
    }

    /// Cancel every answer whose requester `keep` no longer names: it left the
    /// group, or its socket went away.
    pub fn retain(&mut self, keep: impl Fn(u64) -> bool) {
        self.running.retain(|requester, t| {
            let kept = keep(*requester);
            if !kept {
                t.cancelled.store(true, Ordering::Relaxed);
            }
            kept
        });
    }

    /// How many answers are running.
    pub fn len(&self) -> usize {
        self.running.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A requester has one answer running.** Asking again cancels the
    /// answer before; the burst that stacked dozens of full listings now
    /// leaves one.
    #[test]
    fn a_newer_request_cancels_the_answer_before_it() {
        let mut a = Answering::default();
        let first = a.begin(7, 1);
        let second = a.begin(7, 2);
        let third = a.begin(7, 3);
        assert!(!first.live());
        assert!(!second.live());
        assert!(third.live());
        assert_eq!(a.len(), 1);
    }

    /// Requesters are independent: one asking again leaves the other's answer
    /// running.
    #[test]
    fn another_requesters_answer_runs_on() {
        let mut a = Answering::default();
        let mine = a.begin(7, 1);
        let theirs = a.begin(9, 1);
        a.begin(7, 2);
        assert!(!mine.live());
        assert!(theirs.live());
        assert_eq!(a.len(), 2);
    }

    /// A superseded answer that ends late does not clear the answer that
    /// replaced it.
    #[test]
    fn a_superseded_answer_ending_leaves_its_successor() {
        let mut a = Answering::default();
        let old = a.begin(7, 1);
        let new = a.begin(7, 2);
        a.end(&old);
        assert_eq!(a.len(), 1);
        assert!(new.live());
        a.end(&new);
        assert_eq!(a.len(), 0);
    }

    /// A requester that left has its answer cancelled.
    #[test]
    fn a_requester_gone_has_its_answer_cancelled() {
        let mut a = Answering::default();
        let gone = a.begin(7, 1);
        let here = a.begin(9, 1);
        a.retain(|r| r == 9);
        assert!(!gone.live());
        assert!(here.live());
        assert_eq!(a.len(), 1);
    }
}
