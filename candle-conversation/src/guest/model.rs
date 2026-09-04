//! What a co-resident model has to be able to do.
//!
//! The trait is deliberately narrow: a guest is asked how much ground it needs,
//! loaded into exactly that ground, run against a list of jobs, and dropped.
//! It never sees the queue, the scheduler, or the engine's state — everything
//! about *when* it runs is [`super::drain`]'s business, and everything about
//! *whether there is room* is [`super::ground`]'s.
//!
//! # Why loading and running are separate
//!
//! Loading is the expensive half — gigabytes across the PCIe link into ground
//! the engine has just evicted for it — and the whole reason a queue exists is
//! to pay it once for a whole backlog. A trait with one `run(request)` method
//! would have no way to express that, and every implementation would end up
//! caching the load itself, differently.

use std::sync::{Arc, Mutex};

use candle::Device;

use super::ground::GuestGround;
use super::progress::GuestSink;
use super::work::{Guest, GuestOutcome, GuestRequest};

/// A model that borrows the GPU between the engine's waves.
///
/// Implementations are held un-loaded for the life of the daemon — a
/// [`GuestSpec`] naming a checkpoint costs nothing until a job arrives — and
/// loaded into [`GuestGround`] for the length of one drain.
pub trait GuestModel: Send {
    /// Which guest this serves.
    fn guest(&self) -> Guest;

    /// Device bytes the model needs to stand up, **including** the activations
    /// its largest queued job will produce.
    ///
    /// The jobs are handed in because the answer depends on them: an image at
    /// 512×512 and one at 2048×2048 share a checkpoint and do not share a
    /// latent. Over-estimating costs the engine an eviction it did not need;
    /// under-estimating is worse, because the shortfall is discovered after the
    /// eviction has already happened.
    fn footprint_bytes(&self, jobs: &[GuestRequest]) -> usize;

    /// Stand the model up in `ground`, for this backlog.
    ///
    /// `jobs` is the same backlog [`Self::footprint_bytes`] sized, and it is
    /// handed in for the same reason: **which weights** to place can depend on
    /// the jobs, not only how much room they need — the image guest loads
    /// whichever adapter-fused transformer the backlog asks for. A load that
    /// could not see the jobs would have to load something and let `run`
    /// discover the mismatch after the engine was already evicted.
    ///
    /// Every **weight** must come from `ground`. One that reaches the CUDA pool
    /// instead is memory the reservation was never sized for, competing with
    /// the engine for the same card — which on WDDM is not an error but a
    /// demotion of whichever side loses. Activations are the deliberate
    /// exception; see [`super::prose`]'s header for why.
    ///
    /// **Every handle the model keeps must be released by [`Self::unload`].**
    /// The ground is shared as an `Arc` precisely so that is checkable: the
    /// drain reads the refcount after unloading, and a model that stashed a
    /// clone is named in the log instead of quietly outliving the memory it
    /// views.
    fn load(
        &mut self,
        device: &Device,
        ground: &Arc<Mutex<GuestGround>>,
        jobs: &[GuestRequest],
    ) -> Result<(), String>;

    /// Serve one job. Called once per queued job, with the model already loaded.
    ///
    /// `sink` is where progress goes for a caller that is watching this job run
    /// — usually nobody, in which case emitting costs an `Option` check. It is
    /// called on the scheduler thread with normal inference blocked, so a guest
    /// must emit and carry on rather than wait for anything.
    fn run(&mut self, request: &GuestRequest, sink: &GuestSink) -> Result<GuestOutcome, String>;

    /// Tear the model down, before the ground is returned.
    ///
    /// Called on every path out of a drain, including a failed load — a guest
    /// that dropped its handles but left a `QStorage` alive over a region the
    /// KV side is about to reuse is exactly the aliasing this whole module is
    /// arranged to prevent.
    fn unload(&mut self);
}

/// A guest that is configured but has not been loaded.
///
/// The registry holds these, not live models: a deployment naming an image
/// checkpoint should cost nothing until somebody asks for an image.
#[derive(Clone)]
pub struct GuestSlot {
    pub guest: Guest,
    /// Built fresh per drain. A guest that failed to load must not leave a
    /// half-loaded model behind for the next one to inherit.
    ///
    /// `Arc`, not `Box`, so the registry is `Clone` — `ModelBuilder` is, and a
    /// builder that lost its guests when cloned would configure a deployment
    /// that silently has none.
    pub build: Arc<dyn Fn() -> Box<dyn GuestModel> + Send + Sync>,
}

impl std::fmt::Debug for GuestSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestSlot")
            .field("guest", &self.guest)
            .finish()
    }
}

/// The guests a deployment has configured.
#[derive(Clone, Debug, Default)]
pub struct GuestRegistry {
    slots: Vec<GuestSlot>,
}

impl GuestRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a guest, replacing any previous one for the same kind.
    ///
    /// Replacement rather than a second entry, because the kind is what the
    /// queue routes by: two image guests would mean a job's destination
    /// depended on iteration order, which is not a decision anyone made.
    pub fn register(
        &mut self,
        guest: Guest,
        build: impl Fn() -> Box<dyn GuestModel> + Send + Sync + 'static,
    ) {
        self.slots.retain(|s| s.guest != guest);
        self.slots.push(GuestSlot {
            guest,
            build: Arc::new(build),
        });
    }

    /// Build a fresh instance of `guest`, or `None` if none is configured.
    pub fn build(&self, guest: Guest) -> Option<Box<dyn GuestModel>> {
        self.slots
            .iter()
            .find(|s| s.guest == guest)
            .map(|s| (s.build)())
    }

    pub fn has(&self, guest: Guest) -> bool {
        self.slots.iter().any(|s| s.guest == guest)
    }

    pub fn configured(&self) -> Vec<Guest> {
        let mut out: Vec<Guest> = self.slots.iter().map(|s| s.guest).collect();
        out.sort();
        out
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::guest::progress::GuestEvent;
    use crate::guest::work::{GuestImage, GuestMatte, GuestRequest};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A guest that runs no kernels, for testing everything around one.
    ///
    /// The drain's whole job is *when* and *with what room* — which is testable
    /// without a card, and would not be if the only guests were real ones.
    pub struct FakeGuest {
        pub guest: Guest,
        pub per_job_bytes: usize,
        pub base_bytes: usize,
        pub loads: Arc<AtomicUsize>,
        pub runs: Arc<AtomicUsize>,
        pub unloads: Arc<AtomicUsize>,
        pub fail_load: bool,
        pub fail_run: bool,
    }

    impl FakeGuest {
        pub fn counters() -> (Arc<AtomicUsize>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
            (
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            )
        }
    }

    impl GuestModel for FakeGuest {
        fn guest(&self) -> Guest {
            self.guest
        }

        fn footprint_bytes(&self, jobs: &[GuestRequest]) -> usize {
            self.base_bytes + self.per_job_bytes * jobs.len()
        }

        fn load(
            &mut self,
            _device: &Device,
            _ground: &Arc<Mutex<GuestGround>>,
            _jobs: &[GuestRequest],
        ) -> Result<(), String> {
            self.loads.fetch_add(1, Ordering::Relaxed);
            if self.fail_load {
                return Err("the fake guest was asked to fail its load".into());
            }
            Ok(())
        }

        fn run(
            &mut self,
            request: &GuestRequest,
            sink: &GuestSink,
        ) -> Result<GuestOutcome, String> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            if self.fail_run {
                return Err("the fake guest was asked to fail its run".into());
            }
            // Emitted like a real guest, so the drain's own progress wiring is
            // exercised by the tests that run without a card.
            if let GuestRequest::Prose(r) = request {
                sink.emit(GuestEvent::Token(format!("<{}>", r.prompt)));
            }
            Ok(match request {
                GuestRequest::Image(r) => GuestOutcome::Image(GuestImage {
                    width: r.width,
                    height: r.height,
                    png: vec![0x89, b'P', b'N', b'G'],
                    seed: r.seed.unwrap_or(0),
                }),
                GuestRequest::Prose(r) => GuestOutcome::Prose {
                    text: format!("<{}>", r.prompt),
                    tokens: 3,
                    seed: r.seed.unwrap_or(0),
                },
                GuestRequest::Matte(r) => GuestOutcome::Matte(GuestMatte {
                    width: r.width,
                    height: r.height,
                    png: vec![0x89, b'P', b'N', b'G'],
                    lifted: 0.5,
                }),
            })
        }

        fn unload(&mut self) {
            self.unloads.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::FakeGuest;
    use super::*;

    fn fake(guest: Guest) -> impl Fn() -> Box<dyn GuestModel> + Send + Sync {
        move || {
            let (l, r, u) = FakeGuest::counters();
            Box::new(FakeGuest {
                guest,
                per_job_bytes: 0,
                base_bytes: 1 << 20,
                loads: l,
                runs: r,
                unloads: u,
                fail_load: false,
                fail_run: false,
            })
        }
    }

    #[test]
    fn an_unregistered_guest_builds_nothing() {
        let r = GuestRegistry::new();
        assert!(r.is_empty());
        assert!(!r.has(Guest::Image));
        assert!(r.build(Guest::Image).is_none());
    }

    #[test]
    fn a_registered_guest_builds_a_fresh_instance() {
        let mut r = GuestRegistry::new();
        r.register(Guest::Prose, fake(Guest::Prose));
        assert!(r.has(Guest::Prose));
        assert!(!r.has(Guest::Image));
        assert_eq!(r.build(Guest::Prose).unwrap().guest(), Guest::Prose);
        assert_eq!(r.configured(), vec![Guest::Prose]);
    }

    /// **A guest is built per drain, never reused.** A guest that failed
    /// half-way through a load would otherwise be inherited by the next drain,
    /// which would find a model that reports itself loaded and holds nothing.
    #[test]
    fn each_build_is_a_separate_model() {
        let mut r = GuestRegistry::new();
        r.register(Guest::Image, fake(Guest::Image));
        let a = r.build(Guest::Image).unwrap();
        let b = r.build(Guest::Image).unwrap();
        assert!(
            !std::ptr::eq(
                a.as_ref() as *const _ as *const u8,
                b.as_ref() as *const _ as *const u8
            ),
            "two drains would share one model's state"
        );
    }

    /// Registering the same kind twice replaces rather than appends: the kind
    /// is what the queue routes by, so two entries would make a job's
    /// destination depend on iteration order.
    #[test]
    fn registering_a_kind_twice_replaces_it() {
        let mut r = GuestRegistry::new();
        r.register(Guest::Image, fake(Guest::Image));
        r.register(Guest::Image, fake(Guest::Image));
        assert_eq!(r.configured(), vec![Guest::Image]);
    }
}
