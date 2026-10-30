//! What a conversation opened for one job holds, let go however the job ends.
//!
//! A probe, a dream, a prose job and a memory rebuild each decode on a
//! conversation minted for that one request. Their decodes are awaited, so the
//! whole future can be dropped mid-turn — a console that stops polling, a
//! client that disconnects — and an explicit tail never runs on that path:
//! every abandoned request would strand a registered, path-less timeline and a
//! frame section nothing releases. These guards are the tail every path
//! shares.
//!
//! **A frame outlives the slot reading it; that order is the caller's to
//! keep.** Hold the [`Sequence`](crate::Sequence) ahead of its [`HeldFrame`]
//! in one struct, whose fields drop in declaration order, so the slot goes
//! first on every path. The tombstone may land on either side of the slot: a
//! decode still winding down seals onto a tombstoned timeline, whose records
//! nothing reads.

use std::sync::Arc;

use crate::projection::{SectionId, TimelineId};
use crate::{ConversationEngine, Result};

/// The three engine calls the guards make, on whatever keeps the engine alive
/// for them: the engine itself behind an `Arc`, or a daemon's state that owns
/// it.
pub trait Retires {
    fn tombstone(&self, timeline: TimelineId) -> Result<()>;
    fn take_frame(&self, prompt_text: &str) -> Result<SectionId>;
    fn release_frame(&self, section: SectionId);
}

impl Retires for ConversationEngine {
    fn tombstone(&self, timeline: TimelineId) -> Result<()> {
        self.tombstone_timeline(timeline)
    }

    fn take_frame(&self, prompt_text: &str) -> Result<SectionId> {
        self.transient_prompt_section(prompt_text)
    }

    fn release_frame(&self, section: SectionId) {
        self.release_prompt_section(section)
    }
}

impl<T: Retires + ?Sized> Retires for Arc<T> {
    fn tombstone(&self, timeline: TimelineId) -> Result<()> {
        (**self).tombstone(timeline)
    }

    fn take_frame(&self, prompt_text: &str) -> Result<SectionId> {
        (**self).take_frame(prompt_text)
    }

    fn release_frame(&self, section: SectionId) {
        (**self).release_frame(section)
    }
}

/// Tombstones its timeline on drop.
///
/// Unconditional, because these conversations are equally dead however the
/// request ended — finished, failed, or abandoned, nothing may ever surface
/// them. A tombstone under a still-winding-down decode is safe: the wind-down
/// seals onto a tombstoned timeline whose records nothing reads and the next
/// compaction drops.
pub struct Throwaway<R: Retires> {
    holder: R,
    timeline: TimelineId,
    /// What the conversation was for, for the warning when the tombstone
    /// itself fails.
    what: &'static str,
}

impl<R: Retires> Throwaway<R> {
    pub fn new(holder: R, timeline: TimelineId, what: &'static str) -> Self {
        Self {
            holder,
            timeline,
            what,
        }
    }
}

impl<R: Retires> Drop for Throwaway<R> {
    fn drop(&mut self) {
        if let Err(e) = self.holder.tombstone(self.timeline) {
            tracing::warn!(
                "{} conversation {} could not be retired: {e:?} — it stays selectable and \
                 nothing will ever read it",
                self.what,
                self.timeline
            );
        }
    }
}

/// A transient frame — [`ConversationEngine::transient_prompt_section`] —
/// released on drop, because a frame never released is a section never
/// retired.
pub struct HeldFrame<R: Retires> {
    holder: R,
    id: SectionId,
}

impl<R: Retires> HeldFrame<R> {
    /// The transient frame for `prompt_text`, held until this drops.
    pub fn take(holder: R, prompt_text: &str) -> Result<Self> {
        let id = holder.take_frame(prompt_text)?;
        Ok(Self { holder, id })
    }

    pub fn id(&self) -> SectionId {
        self.id
    }
}

impl<R: Retires> Drop for HeldFrame<R> {
    fn drop(&mut self) {
        self.holder.release_frame(self.id);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::ConversationError;

    /// An engine that records what the guards ask of it.
    #[derive(Clone, Default)]
    struct Fake {
        calls: Arc<Mutex<Vec<String>>>,
        refuse: bool,
    }

    impl Fake {
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }

        fn log(&self, call: String) {
            self.calls.lock().unwrap().push(call);
        }
    }

    impl Retires for Fake {
        fn tombstone(&self, timeline: TimelineId) -> Result<()> {
            self.log(format!("tombstone {}", timeline.raw()));
            if self.refuse {
                return Err(ConversationError::Other("refused".into()));
            }
            Ok(())
        }

        fn take_frame(&self, prompt_text: &str) -> Result<SectionId> {
            if self.refuse {
                return Err(ConversationError::Other("no frame".into()));
            }
            self.log(format!("take {prompt_text}"));
            Ok(SectionId::new(9_001))
        }

        fn release_frame(&self, section: SectionId) {
            self.log(format!("release {}", section.raw()));
        }
    }

    fn timeline() -> TimelineId {
        TimelineId::from_raw(41).unwrap()
    }

    #[test]
    fn a_throwaway_tombstones_its_timeline_once_on_drop() {
        let fake = Fake::default();
        let guard = Throwaway::new(fake.clone(), timeline(), "test");
        assert_eq!(fake.calls(), Vec::<String>::new(), "nothing until it drops");
        drop(guard);
        assert_eq!(fake.calls(), ["tombstone 41"]);
    }

    #[test]
    fn a_tombstone_that_fails_is_reported_not_raised() {
        let fake = Fake {
            refuse: true,
            ..Fake::default()
        };
        drop(Throwaway::new(fake.clone(), timeline(), "test"));
        assert_eq!(fake.calls(), ["tombstone 41"]);
    }

    /// **One take, one release**: a frame taken is released once, with its
    /// own id; one that could not be taken leaves nothing to release.
    #[test]
    fn a_held_frame_is_released_once_with_the_id_it_took() {
        let fake = Fake::default();
        let frame = HeldFrame::take(fake.clone(), "You keep summaries.").unwrap();
        assert_eq!(frame.id(), SectionId::new(9_001));
        drop(frame);
        assert_eq!(fake.calls(), ["take You keep summaries.", "release 9001"]);
        let refusing = Fake {
            refuse: true,
            ..Fake::default()
        };
        assert!(HeldFrame::take(refusing.clone(), "x").is_err());
        assert_eq!(refusing.calls(), Vec::<String>::new());
    }

    /// **Held in one struct, the frame goes last** — fields drop in
    /// declaration order, so a frame declared after the slot outlives it.
    #[test]
    fn a_job_holding_both_retires_the_timeline_then_releases_the_frame() {
        struct Job {
            _retired: Throwaway<Fake>,
            _frame: HeldFrame<Fake>,
        }
        let fake = Fake::default();
        let job = Job {
            _retired: Throwaway::new(fake.clone(), timeline(), "test"),
            _frame: HeldFrame::take(fake.clone(), "p").unwrap(),
        };
        drop(job);
        assert_eq!(fake.calls(), ["take p", "tombstone 41", "release 9001"]);
    }
}
