//! The queue the scheduler looks at between waves.
//!
//! # Why the depth is an atomic and not a lock
//!
//! [`GuestQueue::depth`] is read once per wave, on the scheduler's critical
//! path, and answers "no" almost every time. A mutex there would put lock
//! traffic into the loop that the hot-path invariants exist to keep clear — and
//! the answer it protects is one integer. So the count is an atomic the
//! scheduler loads, and the mutex is taken only on the rare pass where there is
//! actually work to take.
//!
//! # Why a drain takes everything
//!
//! Loading a guest is the expensive half: its weights cross the PCIe link into
//! ground the engine has just evicted, and the engine then pays to fetch that
//! working set back. Serving one job per load would pay that twice for two
//! jobs. So the drain takes the entire backlog for one guest at a time — which
//! is also what makes the queue a queue rather than a mailbox.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crossbeam::channel::{bounded, Receiver, Sender};

use super::progress::{GuestEvent, GuestSink};
use super::work::{Guest, GuestError, GuestOutcome, GuestRequest};

/// A job waiting for its guest, with the channel its answer goes back on.
pub struct Pending {
    pub request: GuestRequest,
    /// Monotonic submission order, for the log and for FIFO within a guest.
    pub seq: u64,
    /// Where this job's progress goes while it runs. [`GuestSink::none`] for a
    /// caller that only wants the answer.
    pub sink: GuestSink,
    reply: Sender<Result<GuestOutcome, GuestError>>,
}

impl Pending {
    /// Answer the caller.
    ///
    /// A send that fails is an ordinary outcome, not an error: the caller's HTTP
    /// request may have been cancelled while the job was queued, and the work
    /// was still worth doing for whatever else the drain was already loaded for.
    pub fn answer(self, outcome: Result<GuestOutcome, GuestError>) {
        let _ = self.reply.send(outcome);
    }

    /// Tell a watching caller what is happening. Nothing when none is.
    pub fn emit(&self, event: GuestEvent) {
        self.sink.emit(event);
    }
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending")
            .field("seq", &self.seq)
            .field("guest", &self.request.guest())
            .finish()
    }
}

/// A caller's handle on one submitted job.
pub struct GuestReceipt {
    rx: Receiver<Result<GuestOutcome, GuestError>>,
    pub seq: u64,
    pub guest: Guest,
}

impl GuestReceipt {
    /// Block until the job finishes.
    ///
    /// The wait is unbounded on purpose. A guest drain evicts the engine's
    /// working set, loads a model and serves a whole backlog; a timeout here
    /// would abandon a job that is going to complete, after the engine has
    /// already paid for it. Callers that need a deadline own one at their own
    /// layer, where cancelling is free.
    pub fn wait(self) -> Result<GuestOutcome, GuestError> {
        self.rx.recv().unwrap_or(Err(GuestError::Abandoned))
    }

    /// The answer if it is ready, without blocking.
    pub fn try_take(&self) -> Option<Result<GuestOutcome, GuestError>> {
        self.rx.try_recv().ok()
    }
}

impl std::fmt::Debug for GuestReceipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestReceipt")
            .field("seq", &self.seq)
            .field("guest", &self.guest)
            .finish()
    }
}

/// Work queued for the co-resident models.
///
/// Shared between the API threads that submit and the scheduler thread that
/// drains.
#[derive(Debug, Default)]
pub struct GuestQueue {
    inner: Mutex<VecDeque<Pending>>,
    /// Mirrors `inner.len()`. The scheduler's per-wave poll reads only this.
    depth: AtomicUsize,
    next_seq: AtomicU64,
    /// Set at shutdown so a submission after it is refused rather than queued
    /// against a scheduler that will never drain again.
    closed: AtomicBool,
}

impl GuestQueue {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a job whose caller wants only the answer.
    pub fn submit(&self, request: GuestRequest) -> Result<GuestReceipt, GuestError> {
        self.submit_watched(request, GuestSink::none())
    }

    /// Queue a job, or refuse it outright, reporting progress to `sink`.
    ///
    /// The refusal path never touches the mutex and never queues: a bad ask
    /// must not be able to cost the engine an eviction. It also never emits —
    /// a caller that is refused gets the refusal from this call, and a
    /// `Loading` on a job that will never load would be a lie.
    pub fn submit_watched(
        &self,
        request: GuestRequest,
        sink: GuestSink,
    ) -> Result<GuestReceipt, GuestError> {
        if let Err(why) = request.check() {
            return Err(GuestError::Refused(why));
        }
        if self.closed.load(Ordering::Acquire) {
            return Err(GuestError::Abandoned);
        }
        let guest = request.guest();
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = bounded(1);
        {
            let mut q = self.inner.lock().unwrap();
            // Re-checked under the lock. Without it a `close` between the check
            // above and the push leaves a job in a queue nothing will drain,
            // and its caller waits for an answer that cannot come.
            if self.closed.load(Ordering::Acquire) {
                return Err(GuestError::Abandoned);
            }
            q.push_back(Pending {
                request,
                seq,
                sink,
                reply: tx,
            });
            self.depth.store(q.len(), Ordering::Release);
        }
        Ok(GuestReceipt { rx, seq, guest })
    }

    /// How many jobs are waiting. One atomic load — this is the per-wave poll.
    pub fn depth(&self) -> usize {
        self.depth.load(Ordering::Acquire)
    }

    /// Whether the scheduler should stop and drain.
    pub fn has_work(&self) -> bool {
        self.depth() > 0
    }

    /// Take every job for `guest`, in submission order, leaving the rest.
    ///
    /// One guest at a time because loading is what a drain is paying for, and
    /// two guests cannot be resident at once in ground that was only just
    /// evicted for one.
    pub fn take(&self, guest: Guest) -> Vec<Pending> {
        let mut q = self.inner.lock().unwrap();
        let mut mine = Vec::new();
        let mut rest = VecDeque::with_capacity(q.len());
        while let Some(p) = q.pop_front() {
            if p.request.guest() == guest {
                mine.push(p);
            } else {
                rest.push_back(p);
            }
        }
        *q = rest;
        self.depth.store(q.len(), Ordering::Release);
        mine
    }

    /// The guest with the oldest waiting job, or `None` when the queue is empty.
    ///
    /// Oldest-first across guests, so a backlog of one kind cannot starve the
    /// other: whichever has been waiting longest is loaded next, and the whole
    /// of its backlog goes with it.
    pub fn next_guest(&self) -> Option<Guest> {
        let q = self.inner.lock().unwrap();
        q.front().map(|p| p.request.guest())
    }

    /// Refuse everything queued and everything submitted afterwards.
    ///
    /// Called on scheduler shutdown. Without it every caller blocked in
    /// [`GuestReceipt::wait`] waits forever on a thread that has gone.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let drained: Vec<Pending> = {
            let mut q = self.inner.lock().unwrap();
            let all = std::mem::take(&mut *q);
            self.depth.store(0, Ordering::Release);
            all.into()
        };
        for p in drained {
            p.answer(Err(GuestError::Abandoned));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest::work::{GuestImage, ImageLora, ImageRequest, ProseRequest};

    fn image() -> GuestRequest {
        GuestRequest::Image(ImageRequest {
            prompt: "a lantern".into(),
            width: 512,
            height: 512,
            steps: 8,
            seed: None,
            lora: ImageLora::default(),
            reference: None,
            shift: None,
        })
    }

    fn prose() -> GuestRequest {
        GuestRequest::Prose(ProseRequest {
            system: String::new(),
            prompt: "the yard".into(),
            max_tokens: 32,
            temperature: None,
            seed: None,
            choices: None,
        })
    }

    fn an_image() -> GuestOutcome {
        GuestOutcome::Image(GuestImage {
            width: 512,
            height: 512,
            png: vec![1, 2, 3],
            seed: 7,
        })
    }

    #[test]
    fn a_submitted_job_is_visible_to_the_scheduler_poll() {
        let q = GuestQueue::new();
        assert!(!q.has_work());
        let r = q.submit(image()).unwrap();
        assert_eq!(q.depth(), 1);
        assert!(q.has_work());

        let taken = q.take(Guest::Image);
        assert_eq!(taken.len(), 1);
        assert_eq!(q.depth(), 0);
        assert!(!q.has_work());

        taken.into_iter().next().unwrap().answer(Ok(an_image()));
        assert_eq!(r.wait().unwrap().guest(), Guest::Image);
    }

    /// **A drain takes one guest's whole backlog and leaves the other's.**
    /// Loading is what the drain is paying for, so serving one job per load
    /// would pay it once per job — and two guests cannot both be resident in
    /// ground that was evicted for one.
    #[test]
    fn a_drain_takes_one_guests_backlog_and_leaves_the_rest() {
        let q = GuestQueue::new();
        q.submit(image()).unwrap();
        q.submit(prose()).unwrap();
        q.submit(image()).unwrap();
        assert_eq!(q.depth(), 3);

        let images = q.take(Guest::Image);
        assert_eq!(images.len(), 2);
        assert_eq!(
            images.iter().map(|p| p.seq).collect::<Vec<_>>(),
            vec![0, 2],
            "a guest's own jobs must stay in submission order"
        );
        assert_eq!(q.depth(), 1, "the other guest's job was taken too");
        assert_eq!(q.next_guest(), Some(Guest::Prose));
    }

    /// Oldest-first across guests, so a steady stream of one kind cannot leave
    /// the other waiting forever.
    #[test]
    fn the_longest_waiting_guest_is_loaded_next() {
        let q = GuestQueue::new();
        q.submit(prose()).unwrap();
        q.submit(image()).unwrap();
        assert_eq!(q.next_guest(), Some(Guest::Prose));

        q.take(Guest::Prose);
        assert_eq!(q.next_guest(), Some(Guest::Image));
        q.take(Guest::Image);
        assert_eq!(q.next_guest(), None);
    }

    /// **A refused ask never reaches the queue.** The drain evicts the engine's
    /// working set before a guest runs, so a request that was never servable
    /// must not be able to trigger one.
    #[test]
    fn a_refused_ask_is_not_queued() {
        let q = GuestQueue::new();
        let bad = GuestRequest::Image(ImageRequest {
            prompt: String::new(),
            width: 512,
            height: 512,
            steps: 8,
            seed: None,
            lora: ImageLora::default(),
            reference: None,
            shift: None,
        });
        assert!(matches!(q.submit(bad), Err(GuestError::Refused(_))));
        assert_eq!(q.depth(), 0);
        assert!(!q.has_work());
    }

    /// **Shutdown answers everyone.** A caller blocked in `wait` is blocked on
    /// the scheduler thread; if that thread goes without answering, the caller
    /// waits for the life of the process and the HTTP request never completes.
    #[test]
    fn shutdown_answers_every_waiting_caller() {
        let q = GuestQueue::new();
        let a = q.submit(image()).unwrap();
        let b = q.submit(prose()).unwrap();
        q.close();

        assert_eq!(a.wait(), Err(GuestError::Abandoned));
        assert_eq!(b.wait(), Err(GuestError::Abandoned));
        assert_eq!(q.depth(), 0);
        assert!(
            matches!(q.submit(image()), Err(GuestError::Abandoned)),
            "a submission after shutdown queued against a scheduler that has gone"
        );
    }

    /// A caller that goes away releases its receiver, and the answer send is an
    /// ordinary miss rather than a panic — the job was still worth running for
    /// whatever else the drain had already loaded for.
    #[test]
    fn answering_a_caller_that_left_is_not_an_error() {
        let q = GuestQueue::new();
        let r = q.submit(image()).unwrap();
        drop(r);
        let taken = q.take(Guest::Image);
        taken.into_iter().next().unwrap().answer(Ok(an_image()));
    }

    #[test]
    fn submission_order_is_a_total_order_across_guests() {
        let q = GuestQueue::new();
        let a = q.submit(image()).unwrap();
        let b = q.submit(prose()).unwrap();
        let c = q.submit(image()).unwrap();
        assert_eq!((a.seq, b.seq, c.seq), (0, 1, 2));
    }
}
