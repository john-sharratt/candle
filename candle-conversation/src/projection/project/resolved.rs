//! The output of a projection: the sections and turns it emitted, and the
//! segments it assembles into.
//!
//! A `ResolvedTurn` is identified by its timeline and index, and the timeline is
//! stamped once at projection, so consumers read the key rather than re-deriving
//! the conversation from the group. `Projection` holds the selector choices, the
//! belief for the next reprojection, the reason each turn was selected and the
//! fill report. Its iterators walk the sealed sections and turns in emission
//! order. A `ProjectionSegment` is either a sealed payload from the substrate or
//! a live-prefilled structural run.

use super::*;

// ── Output types ──────────────────────────────────────────────────────────────

/// One emitted system-prompt section reference. The caller resolves the
/// content from [`SectionId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSection {
    pub id: SectionId,
    /// Whether every selecting query attends it whole — its item's `pin:
    /// true`, marked as the section is emitted.
    pub pin: Pin,
}

impl ResolvedSection {
    /// Section `id`, not pinned.
    pub fn new(id: SectionId) -> Self {
        Self { id, pin: Pin::No }
    }
}

/// Mark every emitted system-prompt section with how `sp` pins it, held to
/// the schema's `pin_tokens`: every whole pin that fits, then a pinned
/// collection's members newest first (`within_budget`). A section the budget
/// leaves out is selected by score like any other.
pub(super) fn mark_pins<R: ContentResolver>(
    sp: &SystemPromptSchema,
    resolver: &R,
    segments: &mut [ProjectionSegment],
) {
    if sp.pins.is_empty() {
        return;
    }
    let mut marked: Vec<(usize, Pin, usize)> = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        if let ProjectionSegment::Sealed(SealedKind::Section(s)) = seg {
            let pin = sp.pins.pin_of(sp, s.id);
            if pin != Pin::No {
                marked.push((i, pin, resolver.section_token_count(s.id)));
            }
        }
    }
    let wanted: Vec<(Pin, usize)> = marked.iter().map(|&(_, p, t)| (p, t)).collect();
    let keep = within_budget(&wanted, sp.pins.tokens);
    for (&(i, pin, _), kept) in marked.iter().zip(keep) {
        if let ProjectionSegment::Sealed(SealedKind::Section(s)) = &mut segments[i] {
            s.pin = if kept { pin } else { Pin::No };
        }
    }
}

/// One emitted turn reference, fully self-describing via [`TurnId`].
///
/// Carries the resolved [`TimelineId`] — the *conversation* this turn belongs to
/// — stamped ONCE at projection where the target is unambiguous (see
/// [`ContentResolver::turn_timeline`]).  Downstream consumers read [`Self::key`]
/// directly instead of re-deriving the timeline from the group: a turn's
/// identity is `(timeline, index)`, and re-resolving `group → timeline` per
/// consumer is exactly what once let the reproject pick the wrong conversation's
/// timeline and drop a slot's whole history.  `None` only for mock resolvers in
/// tests (which never reach the apply path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedTurn {
    pub id: TurnId,
    pub timeline: Option<TimelineId>,
}

impl ResolvedTurn {
    #[inline]
    pub fn group(&self) -> GroupId {
        self.id.group_id
    }

    #[inline]
    pub fn index(&self) -> TurnIndex {
        self.id.index
    }

    /// The fully-resolved `(timeline, index)` identity — the only correct key for
    /// substrate turn lookups.  `None` for mock-resolver turns in tests.
    #[inline]
    pub fn key(&self) -> Option<TurnKey> {
        self.timeline.map(|timeline| TurnKey {
            timeline,
            index: self.id.index,
        })
    }
}

impl Projection {
    /// Iter over only the sealed-section payloads, in emission order.
    /// Convenience over walking [`Self::segments`] manually; skips
    /// generated runs and new-user-message segments.
    pub fn sealed_sections(&self) -> impl Iterator<Item = &ResolvedSection> + '_ {
        self.segments.iter().filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Section(s)) => Some(s),
            _ => None,
        })
    }

    /// Iter over only the sealed-turn payloads, in emission order.
    /// Yields `&ResolvedTurn` — the `Role` discriminator under
    /// `SealedKind::Turn` is dropped here so callers that only care
    /// about which `(group, index)` pair was selected don't have to
    /// destructure the part.  Callers that need the part too use
    /// [`Self::sealed_turn_segments`].
    pub fn sealed_turns(&self) -> impl Iterator<Item = &ResolvedTurn> + '_ {
        self.segments.iter().filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Turn(t, _)) => Some(t),
            _ => None,
        })
    }

    /// Iter over each sealed-turn segment as `(ResolvedTurn, Role)` so
    /// callers can distinguish the user half from the assistant half
    /// of the same turn.  Used by the projection assembler when
    /// injecting per-part residence bytes.
    pub fn sealed_turn_segments(&self) -> impl Iterator<Item = (&ResolvedTurn, crate::Role)> + '_ {
        self.segments.iter().filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Turn(t, role)) => Some((t, *role)),
            _ => None,
        })
    }
}
