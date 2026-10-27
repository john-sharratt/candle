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
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use flume::{bounded, Receiver, Sender};

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
    /// When the job was first refused for room, if it has been.
    waiting_since: Option<Instant>,
}

impl Pending {
    /// How long the job has waited for room as of `now`, starting the clock
    /// on its first refusal.
    pub fn waited(&mut self, now: Instant) -> Duration {
        now.saturating_duration_since(*self.waiting_since.get_or_insert(now))
    }

    /// Start the job's wait `ago` in the past, for a test of the patience.
    #[cfg(test)]
    pub(crate) fn refused_ago(&mut self, ago: Duration) {
        self.waiting_since = Instant::now().checked_sub(ago);
    }

    /// Answer the caller.
    ///
    /// A send that fails is an ordinary outcome, not an error: the caller's HTTP
    /// request may have been cancelled while the job was queued, and the work
    /// was still worth doing for whatever else the drain was already loaded for.
    ///
    /// # Why this borrows rather than consuming
    ///
    /// Consuming was the better signature while a drain answered its jobs by
    /// walking them: taking `self` made "answered exactly once" a property of
    /// the type. A drain that hands its whole backlog to the guest and is called
    /// back per job cannot move out of the list it is still lending, so the
    /// once-ness moved to [`super::drain`], which tracks what it has answered
    /// and reports a job the guest never answered for.
    ///
    /// `try_send`, never `send`: the reply channel is `bounded(1)`, so a
    /// blocking send of a second answer to a caller that has not yet read the
    /// first would park the thread making it — and this is called from the
    /// scheduler's drain, where a parked thread is the whole engine stopped.
    /// A second answer is dropped instead, which makes the weaker once-ness
    /// guarantee above safe rather than merely likely.
    pub fn answer(&self, outcome: Result<GuestOutcome, GuestError>) {
        let _ = self.reply.try_send(outcome);
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

    /// Await the job without holding a thread — [`Self::wait`]'s async
    /// counterpart, with the same deliberately unbounded wait.
    pub async fn wait_async(self) -> Result<GuestOutcome, GuestError> {
        self.rx
            .recv_async()
            .await
            .unwrap_or(Err(GuestError::Abandoned))
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
#[derive(Debug)]
pub struct GuestQueue {
    inner: Mutex<VecDeque<Pending>>,
    /// Mirrors `inner.len()`. The scheduler's per-wave poll reads only this.
    depth: AtomicUsize,
    next_seq: AtomicU64,
    /// Set at shutdown so a submission after it is refused rather than queued
    /// against a scheduler that will never drain again.
    closed: AtomicBool,
    /// Whether a job put back by [`Self::requeue`] is waiting for room. While
    /// it is, the scheduler admits no new prefill — see
    /// [`Self::waiting_for_room`].
    waiting: AtomicBool,
    /// When a waiting job may be tried again, as milliseconds since `epoch`.
    retry_at_ms: AtomicU64,
    epoch: Instant,
    /// The largest square side this node's card draws ([`super::reach`]).
    /// The request's own per-side limit until the engine states its card.
    reach_side: AtomicU32,
}

impl Default for GuestQueue {
    fn default() -> Self {
        Self {
            inner: Mutex::default(),
            depth: AtomicUsize::new(0),
            next_seq: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            waiting: AtomicBool::new(false),
            retry_at_ms: AtomicU64::new(0),
            epoch: Instant::now(),
            reach_side: AtomicU32::new(super::work::MAX_IMAGE_SIDE),
        }
    }
}

impl GuestQueue {
    pub fn new() -> Self {
        Self::default()
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// State this node's reach: the largest square side its card draws.
    pub fn set_reach(&self, side: u32) {
        self.reach_side.store(side, Ordering::Relaxed);
    }

    /// The largest square side this node draws — what a caller is told.
    pub fn reach(&self) -> u32 {
        self.reach_side.load(Ordering::Relaxed)
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
        if let GuestRequest::Image(r) = &request {
            let side = self.reach();
            if !super::reach::within(r.width, r.height, side) {
                return Err(GuestError::BeyondReach {
                    width: r.width,
                    height: r.height,
                    side,
                });
            }
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
                waiting_since: None,
            });
            self.depth.store(q.len(), Ordering::Release);
        }
        Ok(GuestReceipt { rx, seq, guest })
    }

    /// How many jobs are waiting. One atomic load — this is the per-wave poll.
    pub fn depth(&self) -> usize {
        self.depth.load(Ordering::Acquire)
    }

    /// Whether the scheduler should stop and drain: something is queued, and
    /// no job waiting for room is still inside its retry interval.
    pub fn has_work(&self) -> bool {
        self.depth() > 0
            && (!self.waiting.load(Ordering::Acquire)
                || self.now_ms() >= self.retry_at_ms.load(Ordering::Acquire))
    }

    /// Whether a job is waiting for work in flight to give its ground back.
    /// One atomic load: the scheduler reads it on every admission pass, which
    /// runs after every decode step.
    pub fn waiting_for_room(&self) -> bool {
        self.waiting.load(Ordering::Acquire)
    }

    /// How long until a waiting job is due, for a scheduler deciding how long
    /// it may park. `None` when nothing waits.
    pub fn retry_wait(&self) -> Option<Duration> {
        if !self.waiting_for_room() {
            return None;
        }
        let at = self.retry_at_ms.load(Ordering::Acquire);
        Some(Duration::from_millis(at.saturating_sub(self.now_ms())))
    }

    /// Put jobs refused for room back at the head of the queue, in their own
    /// order, to be tried again after `retry_in`.
    pub fn requeue(&self, jobs: Vec<Pending>, retry_in: Duration) {
        if jobs.is_empty() {
            return;
        }
        let mut q = self.inner.lock().unwrap();
        if self.closed.load(Ordering::Acquire) {
            drop(q);
            for p in jobs {
                p.answer(Err(GuestError::Abandoned));
            }
            return;
        }
        for p in jobs.into_iter().rev() {
            q.push_front(p);
        }
        self.retry_at_ms.store(
            self.now_ms() + retry_in.as_millis() as u64,
            Ordering::Release,
        );
        self.waiting.store(true, Ordering::Release);
        self.depth.store(q.len(), Ordering::Release);
    }

    /// Take every job for `guest`, in submission order, leaving the rest.
    ///
    /// One guest at a time because loading is what a drain is paying for, and
    /// two guests cannot be resident at once in ground that was only just
    /// evicted for one.
    ///
    /// Taking clears the wait: the drain the jobs are taken for either serves
    /// them, answers them, or puts them back with [`Self::requeue`], all on the
    /// scheduler thread before admission runs again.
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
        self.waiting.store(false, Ordering::Release);
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
            self.waiting.store(false, Ordering::Release);
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
    use crate::guest::work::{GuestImage, ImageLora, ImageRequest, MatteRequest};
    use futures::executor::block_on;

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

    fn matte() -> GuestRequest {
        GuestRequest::Matte(MatteRequest {
            pixels: vec![0; 4 * 4 * 3],
            width: 4,
            height: 4,
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

    /// `wait_async` is `wait` without the parked thread: the same answer
    /// arrives on the same one-shot, and a queue that closes under a waiting
    /// caller still answers `Abandoned` rather than hanging.
    #[test]
    fn a_receipt_can_be_awaited_instead_of_waited_on() {
        let q = GuestQueue::new();
        let r = q.submit(image()).unwrap();
        q.take(Guest::Image)
            .into_iter()
            .next()
            .unwrap()
            .answer(Ok(an_image()));
        let outcome = block_on(r.wait_async()).unwrap();
        assert_eq!(outcome.guest(), Guest::Image);

        let r = q.submit(image()).unwrap();
        q.close();
        assert_eq!(block_on(r.wait_async()), Err(GuestError::Abandoned));
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
        q.submit(matte()).unwrap();
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
        assert_eq!(q.next_guest(), Some(Guest::Matte));
    }

    /// Oldest-first across guests, so a steady stream of one kind cannot leave
    /// the other waiting forever.
    #[test]
    fn the_longest_waiting_guest_is_loaded_next() {
        let q = GuestQueue::new();
        q.submit(matte()).unwrap();
        q.submit(image()).unwrap();
        assert_eq!(q.next_guest(), Some(Guest::Matte));

        q.take(Guest::Matte);
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

    /// **A draw past the card's reach is refused at once, naming the reach.**
    /// Queued, it would evict the engine and wait out its patience for ground
    /// the card does not have. Within the reach, any shape of the area is fine.
    #[test]
    fn a_draw_past_the_reach_is_refused_with_the_reach() {
        let q = GuestQueue::new();
        q.set_reach(512);
        let draw = |width, height| {
            GuestRequest::Image(ImageRequest {
                prompt: "a lantern".into(),
                width,
                height,
                steps: 8,
                seed: None,
                lora: ImageLora::default(),
                reference: None,
                shift: None,
            })
        };
        let e = q.submit(draw(1024, 1024)).map(|_| ()).unwrap_err();
        assert!(
            matches!(
                e,
                GuestError::BeyondReach {
                    width: 1024,
                    height: 1024,
                    side: 512
                }
            ),
            "{e:?}"
        );
        assert!(
            e.to_string()
                .contains("only 512x512 is currently available"),
            "{e}"
        );
        assert_eq!(q.depth(), 0);
        assert!(q.submit(draw(256, 1024)).is_ok());
        assert_eq!(q.depth(), 1);
    }

    /// **Shutdown answers everyone.** A caller blocked in `wait` is blocked on
    /// the scheduler thread; if that thread goes without answering, the caller
    /// waits for the life of the process and the HTTP request never completes.
    #[test]
    fn shutdown_answers_every_waiting_caller() {
        let q = GuestQueue::new();
        let a = q.submit(image()).unwrap();
        let b = q.submit(matte()).unwrap();
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

    /// **A job put back to wait for room holds admission closed and is not due
    /// until its retry.** It goes back at the head, ahead of anything queued
    /// since; `take` clears the wait, because the drain it is taken for either
    /// answers it or puts it back again.
    #[test]
    fn a_requeued_job_waits_for_its_retry_and_closes_admission() {
        let q = GuestQueue::new();
        q.submit(image()).unwrap();
        let taken = q.take(Guest::Image);
        assert!(!q.waiting_for_room());
        assert_eq!(q.retry_wait(), None);

        q.submit(matte()).unwrap();
        q.requeue(taken, Duration::from_secs(60));
        assert!(q.waiting_for_room());
        assert_eq!(q.depth(), 2);
        assert!(!q.has_work(), "a waiting job was due before its retry");
        assert!(q.retry_wait().is_some_and(|w| w > Duration::from_secs(50)));
        assert_eq!(
            q.next_guest(),
            Some(Guest::Image),
            "it went back behind newer work"
        );

        q.requeue(q.take(Guest::Image), Duration::ZERO);
        assert!(q.has_work(), "a job past its retry was not due");
        assert_eq!(q.take(Guest::Image).len(), 1);
        assert!(
            !q.waiting_for_room(),
            "taking the job left admission closed"
        );
    }

    /// Shutdown answers a waiting job too, and a requeue after it answers
    /// rather than queueing against a scheduler that has gone.
    #[test]
    fn shutdown_answers_a_job_waiting_for_room() {
        let q = GuestQueue::new();
        let r = q.submit(image()).unwrap();
        q.requeue(q.take(Guest::Image), Duration::from_secs(60));
        q.close();
        assert_eq!(r.wait(), Err(GuestError::Abandoned));
        assert!(!q.waiting_for_room());

        let r = q.submit(image());
        assert!(matches!(r, Err(GuestError::Abandoned)));
    }

    #[test]
    fn submission_order_is_a_total_order_across_guests() {
        let q = GuestQueue::new();
        let a = q.submit(image()).unwrap();
        let b = q.submit(matte()).unwrap();
        let c = q.submit(image()).unwrap();
        assert_eq!((a.seq, b.seq, c.seq), (0, 1, 2));
    }
}
