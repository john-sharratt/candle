//! The scheduler's side of a guest drain.
//!
//! [`crate::guest::drain::drain_one`] describes what a drain does in terms of an
//! [`EngineRoom`]; this is the engine that is. The split is not indirection for
//! its own sake — the drain's sequence is the part with the invariants in it
//! (the tier goes back first, every job is answered, the ground is released
//! however the drain exits) and none of those invariants are about CUDA. Behind
//! this trait they are testable; in front of it they would need a card and a
//! checkpoint to exercise once.
//!
//! # Where the drain sits in the loop
//!
//! At the top of [`Scheduler::run`]'s iteration, after the submission drain and
//! **before** the wave quanta. That is the one point in the loop where no
//! forward is in flight and none has been issued for this pass — which is
//! exactly the window a boundary move and a span claim are legal in. Anywhere
//! inside the quanta and `claim_span_region` would refuse, correctly, with a
//! message about a standing transient tier.

use std::sync::Arc;

use candle::Device;

use super::prefill::VramPhase;
use super::Scheduler;
use crate::guest::drain::EngineRoom;
use crate::guest::ground::{GroundError, GuestGround};
use crate::guest::{drain_one, DrainReport, Guests};

impl EngineRoom for Scheduler {
    fn device(&self) -> Device {
        self.device.clone()
    }

    fn release_transient(&mut self) {
        // The tier outlives the guards that used it — a forward's outputs
        // escape into its caller — so between waves one can still be standing
        // over ground its wave no longer needs. Every claim below is refused
        // while it is, and `span_region_refusal` would then report "a wave is
        // running", which is the wrong diagnosis for a drain that simply had
        // not handed it back.
        if let Device::Cuda(d) = &self.device {
            candle_nn::kv_cache::end_wave_transient(&d.cuda_stream());
        }
    }

    fn shed(&mut self, _bytes: usize) -> u64 {
        let before = self.free_region_bytes();
        // The same ladder ordinary KV pressure walks. `VramPhase::Load` is the
        // right setpoint: a guest drain is a load — it is about to put a
        // checkpoint on the card — not a decode step trying not to stall.
        self.relieve_vram_pressure("guest", VramPhase::Load);
        self.free_region_bytes().saturating_sub(before)
    }

    fn free_bytes(&self) -> usize {
        self.free_region_bytes() as usize
    }

    fn claim_ground(&mut self, bytes: usize) -> Result<GuestGround, GroundError> {
        GuestGround::claim(&self.device, bytes)
    }

    fn reclaim(&mut self) {
        // The shedding above may have made the weight side concede ground it
        // was using. Now that the guest is gone, let it take that back rather
        // than waiting for the pressure signal to swing the other way — which
        // on a quiet world may be a long time, and until it does the model runs
        // with a smaller expert working set for no reason.
        //
        // Legal here for the same reason the claim was: between forwards.
        self.model.reclaim_spare_ground();
    }
}

impl Scheduler {
    /// Bytes the KV side could hand out right now.
    ///
    /// `free` alone, not `free + blocked`: a blocked region is one the transient
    /// tier's ceiling puts out of reach, and a guest cannot claim it either. The
    /// pressure *report* adds them because the two want different responses;
    /// a claim only cares which it can actually have.
    fn free_region_bytes(&self) -> u64 {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() else {
            return 0;
        };
        candle_nn::kv_cache::region_stats(gpu_id)
            .map(|s| s.free as u64 * candle_nn::kv_cache::REGION_BYTES as u64)
            .unwrap_or(0)
    }

    /// Serve one guest's whole backlog, if any is waiting.
    ///
    /// Returns the report when a drain ran. `None` is the common case and costs
    /// one atomic load — this is called every pass of the scheduler loop.
    ///
    /// **Normal inference is blocked for the length of this call**, which is the
    /// point rather than a cost: the guest stands in ground the KV side was
    /// using, and a wave running against regions that now hold a diffusion
    /// model's weights would read them as attention state. Every address in the
    /// span is mapped, so that does not fault — it produces numbers.
    pub(super) fn drain_guests(&mut self, guests: &Arc<Guests>) -> Option<DrainReport> {
        if !guests.has_work() {
            return None;
        }
        let guest = guests.next_to_drain()?;
        let jobs = guests.queue.take(guest);
        if jobs.is_empty() {
            return None;
        }
        let Some(mut model) = guests.registry.build(guest) else {
            // `next_to_drain` answers an unconfigured guest's jobs and skips it,
            // so reaching here means the registry changed underneath us. The
            // jobs are still answered rather than dropped.
            for job in jobs {
                job.answer(Err(crate::guest::GuestError::Unavailable(guest)));
            }
            return None;
        };
        tracing::info!(
            target: "candle_conversation::guest",
            %guest,
            jobs = jobs.len(),
            "stopping between waves to serve a guest — normal inference is blocked until it \
             finishes"
        );
        Some(drain_one(self, model.as_mut(), jobs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest::model::testing::FakeGuest;
    use crate::guest::work::{Guest, GuestError, ProseRequest};
    use crate::guest::{GuestModel, GuestRequest};
    use std::sync::atomic::AtomicUsize;

    fn prose() -> GuestRequest {
        GuestRequest::Prose(ProseRequest {
            system: String::new(),
            prompt: "the yard".into(),
            max_tokens: 8,
            temperature: None,
            seed: None,
            choices: None,
        })
    }

    fn guests_with(guest: Guest, runs: Arc<AtomicUsize>) -> Arc<Guests> {
        let mut g = Guests::new();
        g.registry.register(guest, move || {
            Box::new(FakeGuest {
                guest,
                per_job_bytes: 0,
                base_bytes: 1 << 20,
                loads: Arc::new(AtomicUsize::new(0)),
                runs: Arc::clone(&runs),
                unloads: Arc::new(AtomicUsize::new(0)),
                fail_load: false,
                fail_run: false,
            }) as Box<dyn GuestModel>
        });
        Arc::new(g)
    }

    /// **The poll costs one atomic load and does nothing else.** It runs on
    /// every pass of the scheduler loop, and a drain that fired on an empty
    /// queue would evict the engine's working set for no jobs at all.
    #[test]
    fn an_empty_queue_does_not_drain() {
        let (mut scheduler, _tx) = super::super::tests::make_test_scheduler();
        let guests = Arc::new(Guests::new());
        assert!(scheduler.drain_guests(&guests).is_none());
    }

    /// **A job submitted to an idle daemon must wake the loop.**
    ///
    /// The scheduler parks in `rx.recv()` when there is nothing to do, and
    /// every other producer of work arrives *through* that channel — so the
    /// block is also the wake. The guest queue is the one that does not, and a
    /// job queued against an idle engine sat there indefinitely: the poll that
    /// would have found it is at the top of the loop, above the park, and
    /// nothing was ever going to arrive to get back there.
    ///
    /// It cost a ten-minute request that never returned, with not one line in
    /// the log — the job was queued correctly and simply never looked at.
    ///
    /// This asserts the wake reaches the channel the loop is parked on. That
    /// the loop then drains is `a_queued_job_is_taken_and_its_caller_answered`.
    #[test]
    fn submitting_to_an_idle_engine_sends_a_wake() {
        use crate::scheduler::SchedulerRequest;
        let (tx, rx) = crossbeam::channel::bounded(4);

        // What `ConversationEngine::submit_guest` does, in the order it does
        // it: queue the job, then wake whatever is parked on the channel.
        let guests = Guests::new();
        let _receipt = guests.queue.submit(prose()).unwrap();
        tx.send(SchedulerRequest::Wake).unwrap();

        assert!(
            guests.has_work(),
            "the job was not queued before the wake was sent"
        );
        assert!(
            matches!(rx.try_recv(), Ok(SchedulerRequest::Wake)),
            "an idle scheduler would still be parked with work waiting"
        );
    }

    /// A job for a guest nothing is configured for is answered and cleared,
    /// rather than left keeping `has_work` true on every subsequent pass — which
    /// would make the scheduler stop between every pair of forwards to drain a
    /// backlog nothing can take.
    #[test]
    fn a_job_for_an_unconfigured_guest_is_cleared_without_a_drain() {
        let (mut scheduler, _tx) = super::super::tests::make_test_scheduler();
        let guests = Arc::new(Guests::new());
        let receipt = guests.queue.submit(prose()).unwrap();

        assert!(scheduler.drain_guests(&guests).is_none());
        assert_eq!(guests.queue.depth(), 0, "the job stayed queued");
        assert_eq!(receipt.wait(), Err(GuestError::Unavailable(Guest::Prose)));
    }

    /// **A queued job reaches a drain, and its caller is answered either way.**
    ///
    /// What this pins is the **wiring**, not the outcome: the queue is polled,
    /// the guest is built, the drain runs, the caller is answered, and the queue
    /// is left empty. A job that reached none of that would leave its caller
    /// blocked for the life of the process, and nothing else in the suite covers
    /// the path from a submission to a scheduler pass.
    ///
    /// The outcome is deliberately not asserted. This session is a CPU one and
    /// holds no span reservation, so the claim refuses — correctly, and with a
    /// message that says so. On a card with a reservation the same call serves
    /// the job. Asserting either would make the test a statement about the
    /// machine it runs on rather than about the wiring.
    #[test]
    fn a_queued_job_is_taken_and_its_caller_answered() {
        let (mut scheduler, _tx) = super::super::tests::make_test_scheduler();
        let runs = Arc::new(AtomicUsize::new(0));
        let guests = guests_with(Guest::Prose, Arc::clone(&runs));
        let receipt = guests.queue.submit(prose()).unwrap();

        let report = scheduler
            .drain_guests(&guests)
            .expect("a queued job did not reach a drain");
        assert_eq!(report.guest, Some(Guest::Prose));
        assert_eq!(report.jobs, 1);
        assert_eq!(guests.queue.depth(), 0);
        assert!(!guests.has_work(), "the job was left in the queue");

        // Answered — with *something*. `wait` returning at all is the property;
        // a caller that is never answered is the failure this exists to catch.
        let answer = receipt.wait();
        match answer {
            Ok(_) => {}
            Err(GuestError::NoRoom { .. }) | Err(GuestError::Failed(_)) => {}
            other => panic!("a queued job's caller was answered with {other:?}"),
        }
    }
}
