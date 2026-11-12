//! How many positions a projection's emitted segments take — what the system
//! prompt costs a dialogue's fill as emitted (`docs/hit_driven_projection.md`
//! §7.2).

use super::project::{ProjectionSegment, SealedKind};
use crate::substrate::ContentResolver;

/// Tokens `segments` place: sealed sections at their recorded counts, live
/// glue at its length. Turns are not counted here — they are what the fill
/// places in what is left.
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
