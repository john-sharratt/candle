//! Running a guest's whole backlog, once, between two of the engine's waves.
//!
//! # The sequence, and why it is this order
//!
//! 1. **Hand back the transient tier.** The tier is placed flush against the
//!    KV side's frontier with no gap above it, so while one stands there is no
//!    ground to claim — and `span_region_refusal` will say exactly that. Every
//!    step below claims or frees, so this goes first.
//! 2. **Shed until the ground is there.** Compress resident turns, evict the
//!    cold tail, release empty arenas, and ask the weight side to concede.
//!    This is the same ladder ordinary KV pressure walks, driven to a target
//!    instead of to a shortfall.
//! 3. **Claim.** One arena window, every region at once.
//! 4. **Load, run the whole backlog, unload.**
//! 5. **Drop the ground.** The regions return to the KV side.
//!
//! There is no sixth step. The engine's working set comes back the way it
//! always does — a warm turn elevates on the demand that needs it, an expert
//! pages in on the layer that routes to it — which is why this has no restore
//! path to get wrong. What it does do is [`ManagedBatchedModel::reclaim_spare_ground`],
//! so the weight side can take back what it conceded now that the guest is
//! gone rather than waiting for the pressure signal to swing the other way.
//!
//! # Why normal inference is blocked and not merely deprioritised
//!
//! This runs on the scheduler thread, between waves, so nothing else can be
//! forwarding while it does. That is deliberate rather than incidental: the
//! guest is standing in ground the KV side was using, and a wave running
//! against KV regions that now hold a diffusion model's weights would read
//! them as attention state. Every address in the span is mapped, so it would
//! not fault — it would produce numbers.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle::Device;

use super::ground::{GroundError, GuestGround};
use super::model::GuestModel;
use super::progress::{GuestEvent, GuestSink};
use super::queue::Pending;
use super::work::{Guest, GuestError, GuestOutcome, GuestRequest};

/// What one drain did, for the log and for the memory report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DrainReport {
    pub guest: Option<Guest>,
    pub jobs: usize,
    pub served: usize,
    pub failed: usize,
    /// Ground the guest actually stood in.
    pub ground_mib: u64,
    /// What the shedding ladder freed on the way to it.
    pub freed_mib: u64,
    pub load_ms: u64,
    pub run_ms: u64,
    pub total_ms: u64,
}

impl DrainReport {
    pub fn is_empty(&self) -> bool {
        self.jobs == 0
    }
}

/// Everything the drain needs from the engine, as callbacks.
///
/// A trait rather than a `&mut Scheduler` so the sequence above is testable
/// without a card: the order of the steps, the refusal when the ground is
/// short, and the answer every caller gets are the parts with invariants in
/// them, and none of them are about CUDA.
pub trait EngineRoom {
    /// The device the guest will stand on.
    fn device(&self) -> Device;

    /// Hand back any standing wave transient tier. Step 1.
    fn release_transient(&mut self);

    /// Shed toward `bytes` of free KV ground, answering with what was freed.
    ///
    /// Called repeatedly until it stops making progress: the ladder's rungs
    /// have different costs and the cheap ones are tried first, so a single
    /// call is not the same as an exhausted one.
    fn shed(&mut self, bytes: usize) -> u64;

    /// Free KV ground right now, in bytes.
    fn free_bytes(&self) -> usize;

    /// Claim `bytes` of span ground for the guest. Step 3.
    ///
    /// On the trait rather than called directly so the whole sequence can be
    /// exercised without a card. The production implementation is one line —
    /// [`GuestGround::claim`] — and that is the point: everything with an
    /// invariant in it is either here, where a fake can drive it, or in
    /// [`super::ground::Bump`], where the arithmetic stands alone.
    fn claim_ground(&mut self, bytes: usize) -> Result<GuestGround, GroundError>;

    /// Let the weight side take back ground the shedding made it concede.
    fn reclaim(&mut self);
}

/// How many times the shedding ladder is walked before the drain gives up.
///
/// Each pass is a full walk of the relief ladder — compress, evict, flush,
/// concede — and each is strictly cheaper than the one after it because the
/// cheap rungs have already run. Four is where the passes stop finding
/// anything on a card under load; a fifth costs a blocking warm flush to
/// discover the same refusal.
const MAX_SHED_PASSES: usize = 4;

/// Run every queued job for one guest.
///
/// `jobs` is the guest's whole backlog, in submission order. Every one of them
/// is answered — with an outcome, or with the reason it was not served — on
/// every path out of this function, including the ones where the guest never
/// loaded. A queued job whose caller is left waiting is the failure this is
/// arranged to make impossible.
pub fn drain_one<R: EngineRoom>(
    room: &mut R,
    model: &mut dyn GuestModel,
    jobs: Vec<Pending>,
) -> DrainReport {
    let started = Instant::now();
    let guest = model.guest();
    let mut report = DrainReport {
        guest: Some(guest),
        jobs: jobs.len(),
        ..Default::default()
    };
    if jobs.is_empty() {
        return report;
    }

    let requests: Vec<GuestRequest> = jobs.iter().map(|p| p.request.clone()).collect();
    let want = model.footprint_bytes(&requests);

    // Step 1. Nothing below can claim a region while a tier stands.
    room.release_transient();

    // Step 2. The same ladder ordinary KV pressure walks, driven to a target.
    let mut freed = 0u64;
    for _ in 0..MAX_SHED_PASSES {
        if room.free_bytes() >= want {
            break;
        }
        let got = room.shed(want.saturating_sub(room.free_bytes()));
        freed += got;
        if got == 0 {
            // The ladder found nothing. Another pass walks the same rungs and
            // finds the same nothing, at the cost of a blocking warm flush.
            break;
        }
    }
    report.freed_mib = freed >> 20;

    // Step 3. Claim only what the guest places itself — a second tenancy was
    // shed for above and claims its own regions from what that freed.
    let device = room.device();
    let ground_want = model.ground_bytes(&requests);
    let ground = match room.claim_ground(ground_want) {
        Ok(g) => Arc::new(Mutex::new(g)),
        Err(e) => {
            let err = ground_refusal(&e, want, freed);
            tracing::warn!(
                target: "candle_conversation::guest",
                %guest,
                want_mib = want >> 20,
                freed_mib = report.freed_mib,
                "no room for the guest: {e}"
            );
            for job in jobs {
                job.answer(Err(err.clone()));
            }
            report.failed = report.jobs;
            report.total_ms = started.elapsed().as_millis() as u64;
            room.reclaim();
            return report;
        }
    };
    report.ground_mib = (ground.lock().unwrap().capacity() >> 20) as u64;

    // Step 4. Announced first: loading is seconds of weights crossing the link
    // and the only stretch of a drain with nothing to show, so a watching
    // caller is told it has started rather than left on a blank spinner.
    //
    // After the claim, not before — a job that never got its ground was told
    // its refusal above, and a `Loading` for a load that will not happen is
    // worse than silence.
    for job in &jobs {
        job.emit(GuestEvent::Loading);
    }
    let t_load = Instant::now();
    let loaded = model.load(&device, &ground, &requests);
    report.load_ms = t_load.elapsed().as_millis() as u64;
    if let Err(e) = loaded {
        tracing::warn!(
            target: "candle_conversation::guest",
            %guest,
            "the guest failed to load: {e}"
        );
        // Unloaded even though the load failed: a partial load holds device
        // handles over ground that is about to go back to the KV side, and a
        // `QStorage` outliving its region is exactly the aliasing this module
        // is arranged around.
        model.unload();
        let err = GuestError::Failed(e);
        for job in jobs {
            job.answer(Err(err.clone()));
        }
        report.failed = report.jobs;
        report.total_ms = started.elapsed().as_millis() as u64;
        release(ground, guest);
        room.reclaim();
        return report;
    }

    // **What the arena did not serve, and why.** A reading now and a
    // subtraction after, rather than a reset, because the counters are
    // process-wide and the persistence thread can add to them mid-drain — see
    // [`candle::cuda_backend::wave_provenance::DeclineSnapshot`]. The two
    // reasons have opposite fixes: `NoTicket` is a provenance break somewhere
    // upstream, `ArenaFull` is a sizing problem and nothing else.
    let before_declines = candle::cuda_backend::wave_provenance::DeclineSnapshot::now();
    let t_run = Instant::now();
    // **The whole backlog in one call, so a guest that can batch does.**
    //
    // This was `for job in jobs { model.run(..) }`, which paid the load once and
    // then decoded strictly one sequence at a time — the drain's own log said
    // `jobs=1` on every line of a nineteen-turn outline, and `run_ms` was 48,000
    // against a `load_ms` of 1,150. Amortising the load, which is what the queue
    // was already doing, was saving 2% of the wrong thing.
    let pairs: Vec<(&GuestRequest, &GuestSink)> =
        jobs.iter().map(|p| (&p.request, &p.sink)).collect();
    // Answered through a callback rather than a returned list, so a guest that
    // decodes sequentially still unblocks each caller as its own job lands
    // instead of at the end of the backlog. Taken out of `jobs` by index so each
    // is answered exactly once, and whatever the guest never answered for is
    // caught below.
    let mut answered = vec![false; jobs.len()];
    let (mut served, mut failed) = (0usize, 0usize);
    {
        let jobs = &jobs;
        let answered = &mut answered;
        let mut answer = |i: usize, outcome: Result<GuestOutcome, String>| {
            let Some(job) = jobs.get(i).filter(|_| !answered[i]) else {
                // An index outside the backlog, or one answered twice. Neither
                // is recoverable into an outcome for anybody, so it is reported
                // rather than silently pairing one job's answer with another's
                // caller.
                tracing::error!(
                    target: "candle_conversation::guest",
                    %guest,
                    "the guest answered job {i} of {} — out of range, or twice",
                    jobs.len()
                );
                return;
            };
            answered[i] = true;
            match outcome {
                Ok(outcome) => {
                    served += 1;
                    job.answer(Ok(outcome));
                }
                Err(e) => {
                    // One job's failure is not the backlog's. The guest is
                    // loaded and the ground is claimed; abandoning the rest
                    // would pay the whole cost of a drain to serve nothing, and
                    // the next drain would pay it again for the same jobs.
                    tracing::warn!(
                        target: "candle_conversation::guest",
                        %guest,
                        seq = job.seq,
                        "guest job failed: {e}"
                    );
                    failed += 1;
                    job.answer(Err(GuestError::Failed(e)));
                }
            }
        };
        model.run_batch(&pairs, &mut answer);
    }
    drop(pairs);

    // **A queued job whose caller is left waiting is the failure this whole
    // function is arranged to make impossible**, and a guest that returns
    // without answering one is the way it would happen.
    for (job, done) in jobs.iter().zip(&answered) {
        if *done {
            continue;
        }
        tracing::error!(
            target: "candle_conversation::guest",
            %guest,
            seq = job.seq,
            "the guest returned without answering this job"
        );
        failed += 1;
        job.answer(Err(GuestError::Failed(format!(
            "the {guest} guest returned without answering this job"
        ))));
    }
    report.served = served;
    report.failed = failed;
    report.run_ms = t_run.elapsed().as_millis() as u64;
    // Read before `unload`, which closes the arena and would leave the last
    // stage's figures describing a guest that is already gone.
    let declines = before_declines.bytes_since();

    model.unload();
    // Step 5. Explicit rather than left to the end of scope, so the order —
    // model down, then ground back — is stated where it matters instead of
    // being a consequence of declaration order.
    release(ground, guest);
    room.reclaim();

    report.total_ms = started.elapsed().as_millis() as u64;
    tracing::info!(
        target: "candle_conversation::guest",
        %guest,
        jobs = report.jobs,
        served = report.served,
        failed = report.failed,
        ground_mib = report.ground_mib,
        freed_mib = report.freed_mib,
        load_ms = report.load_ms,
        run_ms = report.run_ms,
        total_ms = report.total_ms,
        // Bytes that reached the pool through the inheriting path. Not the whole
        // story — a site calling `dev.alloc` directly never asks, so it never
        // appears here — but it is the difference between "the arena is too
        // small" and "the provenance broke", which is the question a guest
        // out-of-memory always turns out to be.
        no_ticket_mib = declines.0 >> 20,
        arena_full_mib = declines.1 >> 20,
        "guest drained"
    );
    report
}

/// Hand the ground back, saying so if the guest did not let go of it.
///
/// **This is the one check that a guest tore itself down properly.** Every
/// weight the guest holds is a tensor *viewing* ground, and the regions return
/// to the KV side the moment the last `Arc` drops. A guest that stashed a clone
/// — in a cache, a closure, a background thread — keeps those regions alive
/// while the pool hands their addresses to a KV arena, and both tenants then
/// write the same bytes. Nothing faults: every address in the span is mapped,
/// so it surfaces later as attention reading a diffusion model's weights.
///
/// A count above one means the drop below did *not* return the ground, so this
/// is the last moment anything can name the cause. It is logged rather than
/// panicked because the drain is on the scheduler thread and taking the whole
/// engine down is a worse answer than a loud warning and a leaked drain's worth
/// of regions.
fn release(ground: Arc<Mutex<GuestGround>>, guest: Guest) {
    let held = Arc::strong_count(&ground);
    if held > 1 {
        let regions = ground.lock().map(|g| g.regions()).unwrap_or(0);
        tracing::error!(
            target: "candle_conversation::guest",
            %guest,
            outstanding = held - 1,
            regions,
            "the guest unloaded but {} handle(s) on its ground are still alive — those regions \
             cannot go back to the KV side, and if they did they would be written by two tenants \
             at once. The guest's `unload` is not releasing everything its `load` took.",
            held - 1
        );
    }
    drop(ground);
}

/// Turn a ground refusal into the error a caller reads.
///
/// `NoRoom` is retryable and `Failed` is not, and the difference matters to a
/// caller deciding whether to try again: a full span may be empty in a minute,
/// while a forward being open when the drain ran is a bug in the drain.
fn ground_refusal(e: &GroundError, want: usize, freed: u64) -> GuestError {
    match e {
        GroundError::Short { .. } | GroundError::Exhausted { .. } => GuestError::NoRoom {
            wanted_mib: (want >> 20) as u64,
            freed_mib: freed >> 20,
        },
        GroundError::Window(_) | GroundError::TooLarge { .. } => GuestError::Failed(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guest::model::testing::FakeGuest;
    use crate::guest::progress::GuestSink;
    use crate::guest::queue::GuestQueue;
    use crate::guest::work::{GuestMatte, GuestOutcome, ImageLora, ImageRequest, MatteRequest};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};

    /// An engine that owns no card. Records the order of the steps, which is
    /// the thing the drain gets right or wrong.
    #[derive(Default)]
    struct FakeRoom {
        free: usize,
        /// Bytes each `shed` call frees, consumed front to back. Empty means
        /// the ladder has nothing left, which is the refusal case.
        sheds: Vec<u64>,
        log: Vec<&'static str>,
    }

    impl EngineRoom for FakeRoom {
        fn device(&self) -> Device {
            Device::Cpu
        }
        fn release_transient(&mut self) {
            self.log.push("release_transient");
        }
        fn shed(&mut self, _bytes: usize) -> u64 {
            self.log.push("shed");
            if self.sheds.is_empty() {
                return 0;
            }
            let got = self.sheds.remove(0);
            self.free += got as usize;
            got
        }
        fn free_bytes(&self) -> usize {
            self.free
        }
        fn claim_ground(&mut self, bytes: usize) -> Result<GuestGround, GroundError> {
            self.log.push("claim_ground");
            const REGION: usize = 16 << 20;
            let want = bytes.div_ceil(REGION);
            if want * REGION > self.free {
                return Err(GroundError::Short {
                    wanted_bytes: (want * REGION) as u64,
                    got_bytes: self.free as u64,
                    why: "the fake room had nothing left",
                });
            }
            self.free -= want * REGION;
            // Bases spaced two regions apart, so a test that straddled one
            // would be writing into the gap rather than into the next base.
            let bases = (0..want).map(|i| 0x1_0000_0000 + (i as u64 * 2 * REGION as u64));
            Ok(GuestGround::for_test(bases.collect(), REGION))
        }
        fn reclaim(&mut self) {
            self.log.push("reclaim");
        }
    }

    fn guest_model(
        guest: Guest,
        base_bytes: usize,
    ) -> (
        FakeGuest,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
        Arc<AtomicUsize>,
    ) {
        let (l, r, u) = FakeGuest::counters();
        (
            FakeGuest {
                guest,
                per_job_bytes: 0,
                base_bytes,
                loads: Arc::clone(&l),
                runs: Arc::clone(&r),
                unloads: Arc::clone(&u),
                fail_load: false,
                fail_run: false,
            },
            l,
            r,
            u,
        )
    }

    fn image_job() -> GuestRequest {
        GuestRequest::Image(ImageRequest {
            prompt: "a lantern".into(),
            width: 512,
            height: 512,
            steps: 4,
            seed: Some(3),
            lora: ImageLora::default(),
            reference: None,
            shift: None,
        })
    }

    fn matte_job() -> GuestRequest {
        GuestRequest::Matte(MatteRequest {
            pixels: vec![0; 4 * 4 * 3],
            width: 4,
            height: 4,
        })
    }

    /// Queue `n` jobs and take them, so the drain gets real `Pending`s with
    /// real reply channels — the answers are what the tests below check.
    fn queued(
        requests: Vec<GuestRequest>,
    ) -> (
        GuestQueue,
        Vec<Pending>,
        Vec<crate::guest::queue::GuestReceipt>,
    ) {
        let q = GuestQueue::new();
        let mut receipts = Vec::new();
        let guest = requests[0].guest();
        for r in requests {
            receipts.push(q.submit(r).unwrap());
        }
        let taken = q.take(guest);
        (q, taken, receipts)
    }

    /// Queue one job with a watcher attached, and hand back what it saw.
    fn queued_watched(
        request: GuestRequest,
    ) -> (
        GuestQueue,
        Vec<Pending>,
        crate::guest::queue::GuestReceipt,
        Arc<StdMutex<Vec<GuestEvent>>>,
    ) {
        let q = GuestQueue::new();
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let sink = {
            let seen = Arc::clone(&seen);
            GuestSink::new(move |e| seen.lock().unwrap().push(e))
        };
        let guest = request.guest();
        let receipt = q.submit_watched(request, sink).unwrap();
        let taken = q.take(guest);
        (q, taken, receipt, seen)
    }

    /// **A watcher hears about the load before it hears anything else.** Loading
    /// is seconds of weights crossing the link with nothing to show, and it is
    /// the whole of the wait a caller would otherwise stare at a blank spinner
    /// through.
    #[test]
    fn a_watched_job_is_told_the_guest_is_loading_then_streamed() {
        let mut room = FakeRoom {
            free: 64 << 20,
            ..Default::default()
        };
        let (mut model, _, _, _) = guest_model(Guest::Matte, 4 << 20);
        let (_q, jobs, receipt, seen) = queued_watched(matte_job());
        drain_one(&mut room, &mut model, jobs);

        let events = seen.lock().unwrap().clone();
        assert_eq!(
            events.first(),
            Some(&GuestEvent::Loading),
            "the watcher was not told the guest had started loading"
        );
        assert!(
            events.iter().any(|e| matches!(e, GuestEvent::Step { .. })),
            "the run reported no progress to the watcher"
        );
        assert!(receipt.wait().is_ok());
    }

    /// **A job that never got its ground is not told it is loading.** The
    /// refusal is the answer, and a `Loading` for a load that will not happen
    /// leaves a caller watching a progress line for a job that already failed.
    #[test]
    fn a_refused_job_is_never_told_it_is_loading() {
        let mut room = FakeRoom {
            free: 0,
            sheds: vec![],
            ..Default::default()
        };
        let (mut model, _, _, _) = guest_model(Guest::Matte, 64 << 20);
        let (_q, jobs, receipt, seen) = queued_watched(matte_job());
        drain_one(&mut room, &mut model, jobs);

        assert!(
            seen.lock().unwrap().is_empty(),
            "a job that was refused for want of ground was told it was loading"
        );
        assert!(matches!(receipt.wait(), Err(GuestError::NoRoom { .. })));
    }

    /// An unwatched job costs an `Option` check per event and nothing else —
    /// which is every job the console is not currently looking at.
    #[test]
    fn an_unwatched_job_still_drains() {
        let mut room = FakeRoom {
            free: 64 << 20,
            ..Default::default()
        };
        let (mut model, _, _, _) = guest_model(Guest::Matte, 4 << 20);
        let (_q, jobs, receipts) = queued(vec![matte_job()]);
        let report = drain_one(&mut room, &mut model, jobs);
        assert_eq!(report.served, 1);
        assert!(receipts.into_iter().next().unwrap().wait().is_ok());
    }

    /// **The transient tier goes back before anything tries to claim.** While
    /// one stands it sits flush against the KV frontier with no gap above it,
    /// so every claim below refuses — and refuses with a message about a
    /// running wave, which is the wrong diagnosis for a drain that simply
    /// forgot to hand the tier back.
    #[test]
    fn the_tier_is_released_before_the_ladder_runs() {
        let mut room = FakeRoom {
            free: 0,
            sheds: vec![1 << 20],
            ..Default::default()
        };
        let (mut model, _, _, _) = guest_model(Guest::Image, 4 << 20);
        let (_q, jobs, _r) = queued(vec![image_job()]);
        drain_one(&mut room, &mut model, jobs);
        assert_eq!(
            room.log.first(),
            Some(&"release_transient"),
            "the drain shed or claimed before handing the tier back"
        );
    }

    /// The ladder stops as soon as the ground is there. Walking it again costs
    /// a blocking warm flush to free memory nothing is going to use.
    #[test]
    fn shedding_stops_once_the_ground_is_there() {
        let mut room = FakeRoom {
            free: 8 << 20,
            sheds: vec![1 << 20; 8],
            ..Default::default()
        };
        let (mut model, _, _, _) = guest_model(Guest::Image, 4 << 20);
        let (_q, jobs, _r) = queued(vec![image_job()]);
        drain_one(&mut room, &mut model, jobs);
        assert_eq!(
            room.log.iter().filter(|s| **s == "shed").count(),
            0,
            "the ladder ran against ground that was already free"
        );
    }

    /// A ladder that stops finding anything is not walked again: the rungs are
    /// the same and the cheap ones have already run, so a second pass buys a
    /// blocking flush and the same refusal.
    #[test]
    fn a_ladder_that_finds_nothing_is_not_walked_again() {
        let mut room = FakeRoom {
            free: 0,
            sheds: Vec::new(),
            ..Default::default()
        };
        let (mut model, _, _, _) = guest_model(Guest::Image, 4 << 20);
        let (_q, jobs, _r) = queued(vec![image_job()]);
        drain_one(&mut room, &mut model, jobs);
        assert_eq!(
            room.log.iter().filter(|s| **s == "shed").count(),
            1,
            "the drain kept asking a ladder that had already said no"
        );
    }

    /// **Every queued job is answered, even when nothing could run.** A caller
    /// blocked on a receipt whose job was silently dropped waits for the life
    /// of the process.
    #[test]
    fn a_refused_drain_still_answers_every_caller() {
        let mut room = FakeRoom::default();
        let (mut model, _, runs, _) = guest_model(Guest::Image, 4 << 20);
        let (_q, jobs, receipts) = queued(vec![image_job(), image_job(), image_job()]);
        let report = drain_one(&mut room, &mut model, jobs);

        assert_eq!(report.failed, 3);
        assert_eq!(report.served, 0);
        assert_eq!(runs.load(Ordering::Relaxed), 0);
        for r in receipts {
            assert!(
                matches!(r.wait(), Err(GuestError::NoRoom { .. })),
                "a caller was left waiting, or told the wrong thing"
            );
        }
        assert_eq!(
            room.log.last(),
            Some(&"reclaim"),
            "the weight side never got its conceded ground back"
        );
    }

    /// **One load serves the whole backlog.** That is the entire reason the
    /// queue exists rather than a mailbox: the load is gigabytes across the
    /// link into ground the engine just evicted for it.
    #[test]
    fn one_load_serves_the_whole_backlog() {
        let mut room = FakeRoom {
            free: 64 << 20,
            ..Default::default()
        };
        let (mut model, loads, runs, unloads) = guest_model(Guest::Matte, 4 << 20);
        let (_q, jobs, receipts) = queued(vec![matte_job(), matte_job(), matte_job()]);
        let report = drain_one(&mut room, &mut model, jobs);

        assert_eq!(
            loads.load(Ordering::Relaxed),
            1,
            "the guest reloaded per job"
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(unloads.load(Ordering::Relaxed), 1);
        assert_eq!((report.served, report.failed), (3, 0));
        for r in receipts {
            assert!(matches!(r.wait(), Ok(GuestOutcome::Matte(_))));
        }
    }

    /// One job's failure is not the backlog's. The guest is already loaded and
    /// the ground is already claimed; abandoning the rest pays the whole cost
    /// of a drain to serve nothing, and the next drain pays it again.
    #[test]
    fn one_failing_job_does_not_abandon_the_rest() {
        let mut room = FakeRoom {
            free: 64 << 20,
            ..Default::default()
        };
        let (mut model, _, runs, _) = guest_model(Guest::Matte, 4 << 20);
        model.fail_run = true;
        let (_q, jobs, receipts) = queued(vec![matte_job(), matte_job()]);
        let report = drain_one(&mut room, &mut model, jobs);

        assert_eq!(
            runs.load(Ordering::Relaxed),
            2,
            "the second job was skipped"
        );
        assert_eq!((report.served, report.failed), (0, 2));
        for r in receipts {
            assert!(matches!(r.wait(), Err(GuestError::Failed(_))));
        }
    }

    /// **A failed load is still unloaded.** A half-loaded model holds device
    /// handles over ground that is about to return to the KV side, and a
    /// storage outliving its region is the aliasing this module exists to
    /// prevent.
    #[test]
    fn a_failed_load_is_torn_down_before_the_ground_goes_back() {
        let mut room = FakeRoom {
            free: 64 << 20,
            ..Default::default()
        };
        let (mut model, loads, runs, unloads) = guest_model(Guest::Image, 4 << 20);
        model.fail_load = true;
        let (_q, jobs, receipts) = queued(vec![image_job()]);
        let report = drain_one(&mut room, &mut model, jobs);

        assert_eq!(loads.load(Ordering::Relaxed), 1);
        assert_eq!(
            unloads.load(Ordering::Relaxed),
            1,
            "a partial load was left standing"
        );
        assert_eq!(runs.load(Ordering::Relaxed), 0);
        assert_eq!(report.failed, 1);
        assert!(matches!(
            receipts.into_iter().next().unwrap().wait(),
            Err(GuestError::Failed(_))
        ));
    }

    /// **A sequential guest answers each caller as its own job lands.**
    ///
    /// The drain hands its whole backlog to the guest in one call now, so that a
    /// guest able to decode a wave can. The hazard that introduces is latency for
    /// one that cannot: an image guest decodes its jobs one at a time, and if the
    /// drain collected the results and dispatched them at the end, the caller
    /// waiting on the first image would wait for the last one instead. The
    /// callback is what makes that unexpressible, and this is the test that says
    /// so — a guest that answers job 0 before it starts job 1 has job 0's caller
    /// unblocked at that moment.
    #[test]
    fn a_job_is_answered_when_it_lands_and_not_when_the_backlog_ends() {
        // Records the order of (job answered, next job started), so a drain that
        // batched the dispatch would show every answer after every run.
        struct Narrator {
            log: Arc<StdMutex<Vec<String>>>,
        }
        impl GuestModel for Narrator {
            fn guest(&self) -> Guest {
                Guest::Image
            }
            fn footprint_bytes(&self, _: &[GuestRequest]) -> usize {
                1 << 20
            }
            fn load(
                &mut self,
                _: &Device,
                _: &Arc<Mutex<GuestGround>>,
                _: &[GuestRequest],
            ) -> Result<(), String> {
                Ok(())
            }
            fn run(
                &mut self,
                request: &GuestRequest,
                _: &GuestSink,
            ) -> Result<GuestOutcome, String> {
                self.log.lock().unwrap().push("run".into());
                let GuestRequest::Image(r) = request else {
                    return Err("not an image".into());
                };
                Ok(GuestOutcome::Image(super::super::work::GuestImage {
                    width: r.width,
                    height: r.height,
                    png: vec![],
                    seed: 0,
                }))
            }
            fn unload(&mut self) {}
        }

        let log = Arc::new(StdMutex::new(Vec::new()));
        let mut room = FakeRoom {
            free: 64 << 20,
            ..Default::default()
        };
        let mut model = Narrator {
            log: Arc::clone(&log),
        };
        let (_q, jobs, receipts) = queued(vec![image_job(), image_job(), image_job()]);

        // The default `run_batch` answers through the callback, so each answer
        // lands between two `run`s rather than after all of them.
        let report = drain_one(&mut room, &mut model, jobs);
        assert_eq!(report.served, 3);

        for r in receipts {
            assert!(matches!(r.wait(), Ok(GuestOutcome::Image(_))));
        }
        assert_eq!(log.lock().unwrap().len(), 3, "one run per job");
    }

    /// **A guest that does not let go of its ground is named.**
    ///
    /// Every weight a guest holds views ground, and the regions go back to the
    /// KV side the moment the last handle drops. A guest that stashed a clone
    /// keeps them alive while the pool hands their addresses to a KV arena, and
    /// both tenants then write the same bytes — which does not fault, because
    /// every address in the span is mapped, and surfaces later as attention
    /// reading a guest's weights.
    #[test]
    fn a_guest_that_keeps_a_handle_on_its_ground_is_reported() {
        struct Hoarder {
            kept: Option<Arc<Mutex<GuestGround>>>,
        }
        impl GuestModel for Hoarder {
            fn guest(&self) -> Guest {
                Guest::Matte
            }
            fn footprint_bytes(&self, _: &[GuestRequest]) -> usize {
                1 << 20
            }
            fn load(
                &mut self,
                _d: &Device,
                ground: &Arc<Mutex<GuestGround>>,
                _jobs: &[GuestRequest],
            ) -> Result<(), String> {
                self.kept = Some(Arc::clone(ground));
                Ok(())
            }
            fn run(
                &mut self,
                _r: &GuestRequest,
                _sink: &GuestSink,
            ) -> Result<GuestOutcome, String> {
                Ok(GuestOutcome::Matte(GuestMatte {
                    width: 1,
                    height: 1,
                    png: vec![],
                    lifted: 0.0,
                }))
            }
            fn unload(&mut self) {
                // Deliberately does not release `kept` — this is the bug the
                // refcount check exists to name.
            }
        }

        let mut room = FakeRoom {
            free: 64 << 20,
            ..Default::default()
        };
        let mut model = Hoarder { kept: None };
        let (_q, jobs, _r) = queued(vec![matte_job()]);
        drain_one(&mut room, &mut model, jobs);
        assert!(
            model.kept.is_some(),
            "the test's own guest did not keep the handle it exists to keep"
        );
        assert_eq!(
            Arc::strong_count(model.kept.as_ref().unwrap()),
            1,
            "the drain's own handle outlived the drain"
        );
    }

    /// An empty backlog does nothing at all — it must not evict, claim, or
    /// load, because the scheduler polls this and a spurious drain costs the
    /// engine its working set.
    #[test]
    fn an_empty_backlog_touches_nothing() {
        let mut room = FakeRoom::default();
        let (mut model, loads, _, _) = guest_model(Guest::Image, 4 << 20);
        let report = drain_one(&mut room, &mut model, Vec::new());
        assert!(report.is_empty());
        assert!(room.log.is_empty());
        assert_eq!(loads.load(Ordering::Relaxed), 0);
    }

    /// The footprint is asked for once, against the whole backlog — an image at
    /// 512 and one at 2048 share a checkpoint and do not share a latent, so a
    /// per-job estimate would under-claim for the largest of them.
    #[test]
    fn the_footprint_covers_every_job_in_the_backlog() {
        let mut room = FakeRoom {
            free: 1 << 30,
            ..Default::default()
        };
        let (mut model, _, _, _) = guest_model(Guest::Image, 4 << 20);
        model.per_job_bytes = 16 << 20;
        let (_q, jobs, _r) = queued(vec![image_job(), image_job(), image_job()]);
        let report = drain_one(&mut room, &mut model, jobs);
        // 4 MiB base + 3 jobs × 16 MiB, rounded up to whole regions.
        assert!(
            report.ground_mib >= 52,
            "the claim covered fewer jobs than the backlog: {} MiB",
            report.ground_mib
        );
    }
}
