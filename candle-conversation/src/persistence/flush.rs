//! A blocking request for the persistence thread's hot→warm drain.
//!
//! The scheduler asks for a flush when VRAM is short and the turns it could
//! evict have no warm copy yet: a turn can leave VRAM only once one exists.
//! What it needs is enough of them, not all of them. Answered only after the
//! whole backlog had moved, a flush kept the scheduler — blocked in relief —
//! waiting through every pending residence: 57 of them and 578 MiB on the
//! Qwen3-30B-A3B engine probe, to free the 288 MiB that relief had asked for.

use crossbeam::channel::{Receiver, Sender};

/// Who to answer, and how much hot→warm the answer waits for.
///
/// **Taken up by a pass already running**, at its next group, not left for the
/// next pass: the persistence thread is usually mid-pass when relief asks, and a
/// request read only between passes made relief wait out the whole pass in
/// flight — 3.1 s for 33 residences on the Qwen3-30B-A3B probe — before its own
/// began. Bytes are counted from the moment it is taken up ([`Self::taken_at`]),
/// because what was installed before then was already warm when relief looked.
pub struct FlushRequest {
    ack: Sender<()>,
    /// Answer once this many bytes are installed warm; `None` waits for the
    /// whole backlog.
    at_least: Option<u64>,
    /// The pass's installed-bytes count when it took this request up.
    from: u64,
    /// Taken up by a pass already under way rather than at its start.
    mid_pass: bool,
}

impl FlushRequest {
    /// A flush answered once the whole pending backlog is warm.
    pub fn drain(ack: Sender<()>) -> Self {
        Self {
            ack,
            at_least: None,
            from: 0,
            mid_pass: false,
        }
    }

    /// A flush answered once `bytes` are installed warm, or the backlog is.
    pub fn at_least(ack: Sender<()>, bytes: u64) -> Self {
        Self {
            ack,
            at_least: Some(bytes),
            from: 0,
            mid_pass: false,
        }
    }

    /// This request, taken up by a pass already under way that has installed
    /// `installed` bytes so far.
    pub fn taken_at(mut self, installed: u64) -> Self {
        self.from = installed;
        self.mid_pass = true;
        self
    }

    /// At the end of a pass's hot→warm phase: answer, or hand the request back
    /// for the next pass.
    ///
    /// A full drain taken up mid-pass is handed back. Its caller is promised a
    /// pass that drains the backlog, and the pass that took it up planned its
    /// work before the request arrived — so the next pass, which starts with it,
    /// is the one that keeps the promise.
    pub fn answer_or_carry(self) -> Option<Self> {
        if self.at_least.is_none() && self.mid_pass {
            return Some(Self {
                from: 0,
                mid_pass: false,
                ..self
            });
        }
        self.answer();
        None
    }

    /// Whether `installed` bytes moved warm so far in the pass answer this
    /// request before the pass's hot→warm phase is through.
    pub fn satisfied_by(&self, installed: u64) -> bool {
        self.at_least
            .is_some_and(|want| installed.saturating_sub(self.from) >= want)
    }

    /// Answer the waiting caller.
    pub fn answer(self) {
        let _ = self.ack.send(());
    }
}

/// One pass's view of the flush channel: the request it is answering, where a new
/// one arrives while it runs, and a drain it hands on to the next pass.
pub struct PassFlush<'a> {
    current: Option<FlushRequest>,
    incoming: Option<&'a Receiver<FlushRequest>>,
    carry: Option<FlushRequest>,
}

impl<'a> PassFlush<'a> {
    /// A pass started with `current`, taking new requests up from `incoming`.
    pub fn new(
        current: Option<FlushRequest>,
        incoming: Option<&'a Receiver<FlushRequest>>,
    ) -> Self {
        Self {
            current,
            incoming,
            carry: None,
        }
    }

    /// At a group boundary, with `installed` bytes moved warm so far: take up a
    /// waiting request, and answer one that is met.
    pub fn at_group(&mut self, installed: u64) {
        if self.current.is_none() {
            self.current = self
                .incoming
                .and_then(|rx| rx.try_recv().ok())
                .map(|f| f.taken_at(installed));
        }
        if self
            .current
            .as_ref()
            .is_some_and(|f| f.satisfied_by(installed))
        {
            if let Some(f) = self.current.take() {
                f.answer();
            }
        }
    }

    /// At the end of the pass's hot→warm phase: answer what is outstanding, or
    /// keep a mid-pass drain for the next pass ([`FlushRequest::answer_or_carry`]).
    pub fn finish(&mut self) {
        self.carry = self.current.take().and_then(FlushRequest::answer_or_carry);
    }

    /// The drain the next pass starts with, if this one handed one on.
    pub fn carried(self) -> Option<FlushRequest> {
        self.carry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam::channel;

    #[test]
    fn a_full_drain_is_never_satisfied_early() {
        let (tx, _rx) = channel::bounded(1);
        let f = FlushRequest::drain(tx);
        assert!(!f.satisfied_by(0));
        assert!(!f.satisfied_by(u64::MAX));
    }

    #[test]
    fn a_sized_flush_is_satisfied_at_its_bytes() {
        let (tx, rx) = channel::bounded(1);
        let f = FlushRequest::at_least(tx, 300 << 20);
        assert!(!f.satisfied_by(299 << 20));
        assert!(f.satisfied_by(300 << 20));
        f.answer();
        assert!(rx.try_recv().is_ok());
    }

    /// Taken up mid-pass, a request counts only what the pass installs after.
    #[test]
    fn a_request_taken_up_mid_pass_counts_from_there() {
        let (tx, _rx) = channel::bounded(1);
        let f = FlushRequest::at_least(tx, 100).taken_at(250);
        assert!(!f.satisfied_by(300));
        assert!(f.satisfied_by(350));
    }

    /// A full drain taken up mid-pass is carried to the next pass, which answers
    /// it; one taken up at a pass's start, or a sized one, is answered at once.
    #[test]
    fn a_drain_taken_mid_pass_is_answered_by_the_next_pass() {
        let (tx, rx) = channel::bounded(1);
        let carried = FlushRequest::drain(tx)
            .taken_at(10)
            .answer_or_carry()
            .expect("a mid-pass drain is carried");
        assert!(rx.try_recv().is_err());
        assert!(carried.answer_or_carry().is_none());
        assert!(rx.try_recv().is_ok());

        let (tx, rx) = channel::bounded(1);
        assert!(FlushRequest::at_least(tx, 5)
            .taken_at(10)
            .answer_or_carry()
            .is_none());
        assert!(rx.try_recv().is_ok());
    }

    /// A sized request sent while a pass runs is taken up at the next group and
    /// answered once the bytes after it are in, before the pass is through.
    #[test]
    fn a_pass_answers_a_request_that_arrives_while_it_runs() {
        let (req_tx, req_rx) = channel::bounded(1);
        let mut pass = PassFlush::new(None, Some(&req_rx));
        pass.at_group(100);
        let (ack_tx, ack_rx) = channel::bounded(1);
        req_tx.try_send(FlushRequest::at_least(ack_tx, 50)).unwrap();
        pass.at_group(120);
        assert!(ack_rx.try_recv().is_err(), "20 bytes since it arrived");
        pass.at_group(170);
        assert!(ack_rx.try_recv().is_ok());
        pass.finish();
        assert!(pass.carried().is_none());
    }

    /// A drain that arrives mid-pass goes on to the next pass unanswered.
    #[test]
    fn a_drain_arriving_mid_pass_is_carried() {
        let (req_tx, req_rx) = channel::bounded(1);
        let mut pass = PassFlush::new(None, Some(&req_rx));
        let (ack_tx, ack_rx) = channel::bounded(1);
        req_tx.try_send(FlushRequest::drain(ack_tx)).unwrap();
        pass.at_group(10);
        pass.finish();
        assert!(ack_rx.try_recv().is_err());
        let mut next = PassFlush::new(pass.carried(), Some(&req_rx));
        next.finish();
        assert!(ack_rx.try_recv().is_ok());
    }
}
