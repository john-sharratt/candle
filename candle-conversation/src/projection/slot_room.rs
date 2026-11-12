//! How much of a dialogue's slot its conversation may take
//! (`docs/hit_driven_projection.md` §4, §7.2).
//!
//! The slot is the model's trained window. Everything a projection places
//! fits inside it: the system prompt and the working set as they are emitted,
//! measured at each projection, and a reserve for what arrives before the next
//! one. The conversation — the share that flexes — takes what is left.

use super::project::{ProjectionSegment, SealedKind};
use crate::substrate::ContentResolver;
use crate::working_set::WorkingSetConfig;

/// Tokens `segments` place: sealed sections at their recorded counts, live
/// glue at its length. Turns are not counted here — the conversation's own
/// budget is what is being found.
pub(super) fn segment_tokens<R: ContentResolver>(
    segments: &[ProjectionSegment],
    resolver: &R,
) -> usize {
    segments
        .iter()
        .map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Section(s)) => resolver.section_token_count(s.id),
            ProjectionSegment::Generated { tokens, .. } => tokens.len(),
            ProjectionSegment::NewUserMessage { tokens } => tokens.len(),
            ProjectionSegment::Sealed(_) => 0,
        })
        .sum()
}

/// What of `ws`'s slot is left for the conversation once `placed` tokens —
/// the system prompt and the working set — and the reserve are taken.
pub(super) fn slot_room(ws: &WorkingSetConfig, placed: usize) -> usize {
    ws.slot_tokens
        .saturating_sub(ws.reserve_tokens)
        .saturating_sub(placed)
}

#[cfg(test)]
mod tests {
    use super::slot_room;
    use crate::working_set::WorkingSetConfig;
    use std::num::NonZeroUsize;

    fn ws(slot: usize, reserve: usize) -> WorkingSetConfig {
        WorkingSetConfig {
            budget_tokens: 64_000,
            folder_tokens: 10_000,
            guidance_tokens: 16_000,
            lock_tokens: 48_000,
            beta: 0.2,
            min_momentum: 100.0,
            max_file_tokens: 48_000,
            seeds: Vec::new(),
            release_on: Vec::new(),
            max_admits: NonZeroUsize::new(2).unwrap(),
            slot_tokens: slot,
            reserve_tokens: reserve,
        }
    }

    /// The conversation gets the slot less the reserve and what is placed;
    /// nothing when they take it all.
    #[test]
    fn the_conversation_takes_what_the_slot_leaves() {
        let c = ws(262_144, 16_384);
        assert_eq!(slot_room(&c, 40_000 + 120_000), 262_144 - 16_384 - 160_000);
        assert_eq!(slot_room(&c, 300_000), 0);
    }
}
