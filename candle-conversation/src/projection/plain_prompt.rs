//! Where a plain prompt's frame section is sealed.
//!
//! # Why the id has to come from the text
//!
//! A conversation opened from a raw prompt string gets a synthetic schema with
//! one section holding that string — its frame. When the conversation opens, a
//! section the substrate already holds is skipped **by id alone**: the ingest
//! tests the id before it tokenises anything, because for a schema's own
//! sections the id is all it needs. So the id a frame is sealed under decides
//! which text the conversation reads. Under a fixed id, the first prompt ever
//! sealed there is the prompt every later conversation reads, whatever it was
//! opened with — and nothing reports it, because the conversation opens and
//! decodes normally under the wrong frame.
//!
//! That is not hypothetical. Every plain-prompt conversation used to seal its
//! frame at one fixed id. A daemon's prose jobs all shared it, so a description
//! written after a name was written under the naming voice — "reply with the
//! name and nothing else" — and came back as a name.
//!
//! # How an id is chosen
//!
//! From the frame's own tokens, in a partition of the id space kept for frames,
//! and checked against what the substrate already holds. The tokens' hash picks
//! a starting slot and the probe walks forward from it. A slot is taken when it
//! is free, or when it already holds exactly these tokens — the same prompt
//! reopened, which reuses its sealed K/V instead of prefilling it again. A slot
//! holding anything else is passed over.
//!
//! **Claims cover the window before a section seals.** The substrate learns
//! about a section only once it has sealed, so two conversations opening
//! different prompts at the same moment would both find the same slot free.
//! [`PlainPromptFrames`] records every id it hands out, and a different prompt
//! probing a claimed slot passes over it even before the first has sealed.
//!
//! # Kept frames and transient ones
//!
//! A conversation that lives on — a character's, an ingest's — keeps its frame
//! for as long as the substrate does, and the same prompt reopened reuses it.
//! A conversation opened for one job and thrown away is different: a daemon's
//! prose carries a whole node of a life story in its system prompt, so nearly
//! every one is new, and a kept frame per prompt is one permanent section per
//! job — a ladder over one life seals hundreds, and none is ever read again.
//!
//! So a job's frame is **transient** ([`PlainPromptFrames::acquire`]): claimed
//! from a partition of its own, counted while conversations hold it, never
//! written to disk, and retired from the substrate when the last holder lets go
//! ([`PlainPromptFrames::release`]). A retired id is not handed out again until
//! the substrate has actually dropped it, so a job that arrives while a
//! retirement is in flight never opens on a section about to vanish from under
//! it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::SectionId;
use crate::persistence::content_hash::{hash_tokens, ContentHash};

/// One past the highest id the kept partition may use.
///
/// The top of the u32 range is taken: the [`super::Reserved`] band sits at the
/// very top, and reserved corpora allocate upward from `u32::MAX - 4096` (zend's
/// tool-calibration catalog). Frames live below both.
const CEILING: u32 = u32::MAX - 4096;

/// Slots in the kept partition. Each takes two ids: the frame, and the summary
/// framing section a synthetic schema declares at `frame + 1`.
const SLOTS: u32 = 1 << 23;

/// The lowest id in the kept partition.
///
/// Schemas allocate their own section ids upward from 1, and runtime additions
/// continue from their highest, so they grow toward this rather than from it —
/// and are billions of sections short of reaching it.
const FLOOR: u32 = CEILING - 2 * SLOTS;

/// Slots in the transient partition, directly below the kept one. Fewer: a
/// transient frame occupies its slot only while a job holds it.
const TRANSIENT_SLOTS: u32 = 1 << 20;

/// The lowest id a transient frame can take.
const TRANSIENT_FLOOR: u32 = FLOOR - 2 * TRANSIENT_SLOTS;

// The partitions' bounds, checked where they are declared: a change that walks
// them into the reserved band or down into schema-allocated ids fails the build.
const _: () = {
    assert!(
        CEILING < u32::MAX - super::Reserved::COUNT,
        "the partition reaches the reserved band"
    );
    assert!(
        TRANSIENT_FLOOR > 1 << 31,
        "the partitions reach into schema-allocated ids"
    );
};

/// The frame section ids handed out for plain prompts, and the claim on each.
#[derive(Debug, Default)]
pub struct PlainPromptFrames {
    /// Kept frames, claimed for the life of the process.
    kept: HashMap<SectionId, ContentHash>,
    /// Transient frames, with how many conversations hold each.
    held: HashMap<SectionId, (ContentHash, usize)>,
    /// Transient frames released by their last holder and on their way out of
    /// the substrate — not handed out again until it has dropped them.
    retiring: HashSet<SectionId>,
}

impl PlainPromptFrames {
    /// How far the probe walks from a prompt's starting slot before it gives up.
    ///
    /// A slot is passed over only when it holds a different prompt, and the
    /// partitions have hundreds of thousands of slots, so this bounds a walk
    /// through a pathological cluster rather than anything a deployment meets.
    pub const MAX_PROBES: u32 = 64;

    /// The kept frame id for a prompt that tokenises to `tokens`, claimed for
    /// the life of the process.
    ///
    /// `stored` reports what the substrate holds under an id — `None` when
    /// nothing is sealed there. Returns `None` only when every probe was held by
    /// another prompt.
    pub fn resolve(
        &mut self,
        tokens: &[u32],
        stored: impl Fn(SectionId) -> Option<Arc<Vec<u32>>>,
    ) -> Option<SectionId> {
        self.resolve_hashed(hash_tokens(tokens), tokens, stored)
    }

    /// [`Self::resolve`] with the hash supplied, so a test can place two prompts
    /// on the same starting slot.
    fn resolve_hashed(
        &mut self,
        hash: ContentHash,
        tokens: &[u32],
        stored: impl Fn(SectionId) -> Option<Arc<Vec<u32>>>,
    ) -> Option<SectionId> {
        for attempt in 0..Self::MAX_PROBES {
            let id = slot(FLOOR, SLOTS, hash, attempt);
            match self.kept.get(&id) {
                Some(claimed) if *claimed == hash => return Some(id),
                Some(_) => continue,
                None => {}
            }
            match stored(id) {
                Some(present) if present.as_slice() != tokens => continue,
                _ => {
                    self.kept.insert(id, hash);
                    return Some(id);
                }
            }
        }
        None
    }

    /// A transient frame id for a prompt that tokenises to `tokens`, held once
    /// more. Every `Some` must be matched by one [`Self::release`].
    ///
    /// Two jobs on the same prompt share one frame while both hold it. `stored`
    /// is as for [`Self::resolve`].
    pub fn acquire(
        &mut self,
        tokens: &[u32],
        stored: impl Fn(SectionId) -> Option<Arc<Vec<u32>>>,
    ) -> Option<SectionId> {
        self.acquire_hashed(hash_tokens(tokens), tokens, stored)
    }

    /// [`Self::acquire`] with the hash supplied, as for [`Self::resolve_hashed`].
    fn acquire_hashed(
        &mut self,
        hash: ContentHash,
        tokens: &[u32],
        stored: impl Fn(SectionId) -> Option<Arc<Vec<u32>>>,
    ) -> Option<SectionId> {
        for attempt in 0..Self::MAX_PROBES {
            let id = slot(TRANSIENT_FLOOR, TRANSIENT_SLOTS, hash, attempt);
            match self.held.get_mut(&id) {
                Some((claimed, holders)) if *claimed == hash => {
                    *holders += 1;
                    return Some(id);
                }
                Some(_) => continue,
                None => {}
            }
            let present = stored(id);
            if self.retiring.contains(&id) {
                // Released, and still in the substrate: its retirement has not
                // run yet, and a job opened on it now would lose it mid-decode.
                if present.is_some() {
                    continue;
                }
                self.retiring.remove(&id);
            }
            match present {
                Some(p) if p.as_slice() != tokens => continue,
                _ => {
                    self.held.insert(id, (hash, 1));
                    return Some(id);
                }
            }
        }
        None
    }

    /// Let go of one hold on a transient frame.
    ///
    /// `true` when that was the last hold: the section is now the caller's to
    /// retire, and its id is withheld until the substrate has dropped it.
    pub fn release(&mut self, id: SectionId) -> bool {
        let Some((_, holders)) = self.held.get_mut(&id) else {
            return false;
        };
        *holders -= 1;
        if *holders > 0 {
            return false;
        }
        self.held.remove(&id);
        self.retiring.insert(id);
        true
    }
}

/// The frame id of probe `attempt` for a prompt hashing to `hash`, in the
/// partition of `slots` slots starting at `floor`.
fn slot(floor: u32, slots: u32, hash: ContentHash, attempt: u32) -> SectionId {
    let s = (hash.lo.wrapping_add(u64::from(attempt)) % u64::from(slots)) as u32;
    SectionId::new(floor + 2 * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOTHING: fn(SectionId) -> Option<Arc<Vec<u32>>> = |_| None;

    fn at(lo: u64) -> ContentHash {
        ContentHash { lo, hi: 0 }
    }

    fn kept(hash: ContentHash, attempt: u32) -> SectionId {
        slot(FLOOR, SLOTS, hash, attempt)
    }

    fn transient(hash: ContentHash, attempt: u32) -> SectionId {
        slot(TRANSIENT_FLOOR, TRANSIENT_SLOTS, hash, attempt)
    }

    /// A free slot is taken, and it is the prompt's own starting slot.
    #[test]
    fn a_free_slot_is_taken() {
        let mut frames = PlainPromptFrames::default();
        let tokens = [11, 12, 13];
        let id = frames.resolve(&tokens, NOTHING).unwrap();
        assert_eq!(id, kept(hash_tokens(&tokens), 0));
    }

    /// **The same prompt reopens its own section** — in this process by its
    /// claim, and in the next one by what the substrate holds, so a restart
    /// reuses the sealed K/V instead of prefilling the frame again.
    #[test]
    fn the_same_prompt_reopens_its_own_section() {
        let tokens = vec![21, 22, 23];
        let mut frames = PlainPromptFrames::default();
        let first = frames.resolve(&tokens, NOTHING).unwrap();
        assert_eq!(frames.resolve(&tokens, NOTHING), Some(first));

        let sealed = Arc::new(tokens.clone());
        let mut restarted = PlainPromptFrames::default();
        let again = restarted
            .resolve(&tokens, |id| (id == first).then(|| Arc::clone(&sealed)))
            .unwrap();
        assert_eq!(again, first);
    }

    /// **The regression.** A slot the substrate holds for a different prompt is
    /// passed over. Taking it is what wrote every description under the naming
    /// voice that had sealed there first.
    #[test]
    fn a_slot_holding_another_prompt_is_passed_over() {
        let tokens = [31, 32, 33];
        let start = kept(hash_tokens(&tokens), 0);
        let naming_voice = Arc::new(vec![1, 2, 3]);
        let mut frames = PlainPromptFrames::default();
        let id = frames
            .resolve(&tokens, |id| {
                (id == start).then(|| Arc::clone(&naming_voice))
            })
            .unwrap();
        assert_ne!(id, start, "handed the slot another prompt had sealed");
        assert_eq!(id, kept(hash_tokens(&tokens), 1));
    }

    /// **A claim holds before the section seals.** Two prompts starting on one
    /// slot, opened back to back: the substrate knows neither yet, and the
    /// second must still not be handed the first's id.
    #[test]
    fn a_claim_holds_before_the_section_seals() {
        let (a, b) = (at(5), at(5 + u64::from(SLOTS)));
        assert_eq!(kept(a, 0), kept(b, 0), "the two must share a starting slot");

        let mut frames = PlainPromptFrames::default();
        let first = frames.resolve_hashed(a, &[1], NOTHING).unwrap();
        let second = frames.resolve_hashed(b, &[2], NOTHING).unwrap();
        assert_eq!(first, kept(a, 0));
        assert_eq!(second, kept(b, 1));
        // And the first prompt still reopens its own.
        assert_eq!(frames.resolve_hashed(a, &[1], NOTHING), Some(first));
    }

    /// The walk is bounded: a run of slots all held by other prompts ends in
    /// `None`, never in one of their ids.
    #[test]
    fn probing_is_bounded() {
        let other = Arc::new(vec![9]);
        let mut frames = PlainPromptFrames::default();
        assert_eq!(
            frames.resolve(&[41, 42], |_| Some(Arc::clone(&other))),
            None
        );
        assert_eq!(
            frames.acquire(&[41, 42], |_| Some(Arc::clone(&other))),
            None
        );
    }

    /// Every id a frame can take — and the summary framing section beside it —
    /// is inside its own partition, and the two partitions do not meet. Their
    /// bounds against the reserved band and schema-allocated ids are checked at
    /// compile time, beside the constants.
    #[test]
    fn the_partitions_are_disjoint_and_hold_their_summary_ids() {
        let top = u64::from(SLOTS) - 1;
        assert_eq!(kept(at(0), 0).raw(), FLOOR);
        assert!(kept(at(top), 0).raw() + 1 < CEILING);

        let transient_top = u64::from(TRANSIENT_SLOTS) - 1;
        assert_eq!(transient(at(0), 0).raw(), TRANSIENT_FLOOR);
        assert!(
            transient(at(transient_top), 0).raw() + 1 < FLOOR,
            "the transient partition reaches into the kept one"
        );

        // Frames take even offsets, so a frame never lands on another's summary.
        for lo in [0u64, 1, 2, 12_345, u64::MAX] {
            assert_eq!((kept(at(lo), 0).raw() - FLOOR) % 2, 0);
            assert_eq!((transient(at(lo), 0).raw() - TRANSIENT_FLOOR) % 2, 0);
        }
    }

    /// **A job's frame is never a kept one**, even for the same text: a kept
    /// frame outlives every job, and retiring it with one would take it from
    /// a conversation that is still living on it.
    #[test]
    fn a_transient_frame_is_never_a_kept_one() {
        let tokens = [51, 52];
        let mut frames = PlainPromptFrames::default();
        let k = frames.resolve(&tokens, NOTHING).unwrap();
        let t = frames.acquire(&tokens, NOTHING).unwrap();
        assert_ne!(k, t);
        assert!(!frames.release(k), "a kept frame is not released");
    }

    /// **Shared while held, retired by the last holder only.** Two jobs on
    /// the same prompt read one section; the first to finish must not take it
    /// from the second.
    #[test]
    fn a_shared_transient_frame_is_retired_by_its_last_holder_only() {
        let tokens = [61, 62, 63];
        let mut frames = PlainPromptFrames::default();
        let a = frames.acquire(&tokens, NOTHING).unwrap();
        let b = frames.acquire(&tokens, NOTHING).unwrap();
        assert_eq!(a, b, "one prompt, one frame");
        assert!(!frames.release(a), "the second job still holds it");
        assert!(frames.release(a), "the last hold retires it");
        assert!(!frames.release(a), "and there is nothing left to release");
    }

    /// **A retiring frame is not reused until the substrate drops it.** A job
    /// arriving while the retirement is in flight would open on a section that
    /// is about to disappear; it takes the next slot instead, and the original
    /// comes back into use once the substrate no longer holds it.
    #[test]
    fn a_retiring_frame_is_not_reused_until_the_substrate_drops_it() {
        let tokens = vec![71, 72];
        let mut frames = PlainPromptFrames::default();
        let first = frames.acquire(&tokens, NOTHING).unwrap();
        assert!(frames.release(first));

        let sealed = Arc::new(tokens.clone());
        let still_there = |id: SectionId| (id == first).then(|| Arc::clone(&sealed));
        let during = frames.acquire(&tokens, still_there).unwrap();
        assert_ne!(during, first, "handed a section mid-retirement");
        assert!(frames.release(during));

        let after = frames.acquire(&tokens, NOTHING).unwrap();
        assert_eq!(
            after, first,
            "the slot is free once the substrate dropped it"
        );
    }

    /// A transient slot the substrate holds for a different prompt is passed
    /// over, as a kept one is.
    #[test]
    fn a_transient_slot_holding_another_prompt_is_passed_over() {
        let tokens = [81, 82];
        let start = transient(hash_tokens(&tokens), 0);
        let other = Arc::new(vec![1, 2]);
        let mut frames = PlainPromptFrames::default();
        let id = frames
            .acquire(&tokens, |id| (id == start).then(|| Arc::clone(&other)))
            .unwrap();
        assert_eq!(id, transient(hash_tokens(&tokens), 1));
    }
}
