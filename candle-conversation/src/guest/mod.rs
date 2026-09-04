//! Co-resident models that borrow the GPU between the engine's waves.
//!
//! An NPC world wants two things the conversation model does not do: pictures,
//! and prose in a voice it is not carrying. Both are models, both want the same
//! card, and neither is worth a second card or a second machine — they are
//! bursty, and the gaps between an engine's waves are where their work fits.
//!
//! # The shape
//!
//! A caller submits to a [`GuestQueue`] and blocks on a [`GuestReceipt`]. The
//! scheduler reads the queue's depth once per wave — one atomic load — and when
//! it is not zero, stops between forwards and drains it: evict, claim, load,
//! serve the whole backlog, unload, hand the ground back. Normal inference is
//! blocked for the length of that drain, which is not a compromise but the
//! point: the guest is standing in ground the KV side was using.
//!
//! ```text
//!   wave  wave  wave   ┌─────────── guest drain ───────────┐   wave  wave
//!   ────  ────  ────   │ evict │ claim │ load │ N jobs │ ⏎ │   ────  ────
//!                      └───────────────────────────────────┘
//!                        normal inference blocked throughout
//! ```
//!
//! # Why there is no restore step
//!
//! The eviction is the engine's ordinary KV relief ladder, driven to a target
//! rather than to a shortfall, and the ground is ordinary span regions. So the
//! way back is also ordinary: a warm turn elevates on the demand that needs it,
//! an expert pages in on the layer that routes to it, and the weight side takes
//! back its conceded ground at the end of the drain. There is nothing to
//! reverse, which is why there is no second mechanism here to get wrong.
//!
//! # Module layout
//!
//! | File | Concern |
//! |---|---|
//! | [`work`] | What is asked for and what comes back. Plain data. |
//! | [`checkpoint`] | Headers and tokenizers, parsed once per process, not per drain. |
//! | [`queue`] | The backlog, its atomic depth, and the reply channels. |
//! | [`ground`] | Span regions for a guest, and the placement arithmetic. |
//! | [`seed`] | Drawing and splitting the seed a job samples with. |
//! | [`model`] | The [`GuestModel`] trait and the registry of configured guests. |
//! | [`progress`] | Watching a job while it runs, a token at a time. |
//! | [`drain`] | The sequence: evict → claim → load → serve → hand back. |
//! | [`tiled`] | The autoencoder a tile at a time, so its peak is fixed. |
//! | [`varground`] | A `VarBuilder` that places into ground, so any candle model loads there. |
//! | [`prose`] | The Hermes-3 backend. |
//! | [`image`] | The Stable Diffusion backend. |

pub mod checkpoint;
pub mod drain;
pub mod ground;
pub mod groundgguf;
pub mod image;
pub mod matte;
pub mod model;
pub mod progress;
pub mod prose;
pub mod queue;
pub mod seed;
pub mod tiled;
pub mod varground;
pub mod work;

pub use drain::{drain_one, DrainReport, EngineRoom};
pub use ground::{GroundError, GuestGround, Placement};
pub use image::{ImageGuest, ImageSpec};
pub use matte::{Family as MatteFamily, MatteGuest, MatteSpec};
pub use model::{GuestModel, GuestRegistry};
pub use progress::{GuestEvent, GuestSink};
pub use prose::{LoadPhases, ProseGuest, ProseSpec};
pub use queue::{GuestQueue, GuestReceipt};
pub use seed::{resolve_seed, Seeded};
pub use varground::GroundVars;
pub use work::{
    Guest, GuestError, GuestImage, GuestMatte, GuestOutcome, GuestRequest, ImageLora,
    ImageReference, ImageRequest, MatteRequest, ProseRequest, DEFAULT_REFERENCE_HOLD,
    MAX_REFERENCE_HOLD, MAX_SHIFT, MIN_SHIFT,
};

use std::sync::Arc;

/// Everything a deployment configures about its guests, in one place.
///
/// Held by the engine and shared with the scheduler. A daemon with no guests
/// configured carries an empty registry and a queue nothing submits to, which
/// costs one atomic load per wave and nothing else.
#[derive(Debug)]
pub struct Guests {
    pub queue: Arc<GuestQueue>,
    pub registry: GuestRegistry,
}

impl Default for Guests {
    fn default() -> Self {
        Self::new()
    }
}

impl Guests {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(GuestQueue::new()),
            registry: GuestRegistry::new(),
        }
    }

    /// Whether the scheduler should stop and drain. One atomic load.
    pub fn has_work(&self) -> bool {
        self.queue.has_work()
    }

    /// The guest to load next: the one whose oldest job has waited longest.
    ///
    /// `None` when the queue is empty. A guest that is queued for but **not
    /// configured** is answered here rather than at the drain — otherwise the
    /// engine evicts its working set to load a model that does not exist, and
    /// the queue never empties because nothing ever takes those jobs.
    pub fn next_to_drain(&self) -> Option<Guest> {
        loop {
            let guest = self.queue.next_guest()?;
            if self.registry.has(guest) {
                return Some(guest);
            }
            for job in self.queue.take(guest) {
                job.answer(Err(GuestError::Unavailable(guest)));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::model::testing::FakeGuest;
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn prose_request() -> GuestRequest {
        GuestRequest::Prose(ProseRequest {
            system: String::new(),
            prompt: "the yard".into(),
            max_tokens: 16,
            temperature: None,
            seed: None,
            choices: None,
        })
    }

    fn image_request() -> GuestRequest {
        GuestRequest::Image(ImageRequest {
            prompt: "a lantern".into(),
            width: 512,
            height: 512,
            steps: 4,
            seed: None,
            lora: ImageLora::default(),
            reference: None,
            shift: None,
        })
    }

    fn register(g: &mut Guests, guest: Guest) {
        g.registry.register(guest, move || {
            Box::new(FakeGuest {
                guest,
                per_job_bytes: 0,
                base_bytes: 1 << 20,
                loads: Arc::new(AtomicUsize::new(0)),
                runs: Arc::new(AtomicUsize::new(0)),
                unloads: Arc::new(AtomicUsize::new(0)),
                fail_load: false,
                fail_run: false,
            })
        });
    }

    #[test]
    fn a_daemon_with_no_guests_never_drains() {
        let g = Guests::new();
        assert!(!g.has_work());
        assert_eq!(g.next_to_drain(), None);
    }

    /// **A job for an unconfigured guest is answered, not queued forever.**
    /// Left in the queue it would keep `has_work` true on every wave, so the
    /// scheduler would stop between every pair of forwards to drain a backlog
    /// nothing can take — and each of those stops would first evict the
    /// engine's working set for a model that does not exist.
    #[test]
    fn a_job_for_an_unconfigured_guest_is_refused_before_anything_is_evicted() {
        let g = Guests::new();
        let r = g.queue.submit(image_request()).unwrap();
        assert_eq!(
            g.next_to_drain(),
            None,
            "the scheduler was told to evict for a guest that is not configured"
        );
        assert!(!g.has_work(), "the unservable job stayed in the queue");
        assert_eq!(r.wait(), Err(GuestError::Unavailable(Guest::Image)));
    }

    /// An unconfigured guest's jobs are cleared without hiding a configured
    /// guest's behind them.
    #[test]
    fn clearing_an_unconfigured_guest_still_finds_the_configured_one() {
        let mut g = Guests::new();
        register(&mut g, Guest::Prose);
        let refused = g.queue.submit(image_request()).unwrap();
        g.queue.submit(prose_request()).unwrap();

        assert_eq!(g.next_to_drain(), Some(Guest::Prose));
        assert_eq!(refused.wait(), Err(GuestError::Unavailable(Guest::Image)));
        assert!(
            g.has_work(),
            "the prose job was cleared with the image ones"
        );
    }

    #[test]
    fn the_longest_waiting_configured_guest_is_next() {
        let mut g = Guests::new();
        register(&mut g, Guest::Prose);
        register(&mut g, Guest::Image);
        g.queue.submit(prose_request()).unwrap();
        g.queue.submit(image_request()).unwrap();
        assert_eq!(g.next_to_drain(), Some(Guest::Prose));
        g.queue.take(Guest::Prose);
        assert_eq!(g.next_to_drain(), Some(Guest::Image));
    }
}
