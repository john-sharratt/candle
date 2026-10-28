//! Who a resident turn is held for: every conversation, or one.
//!
//! The reader sets it (`docs/gallery_residency.md` §2). A scan masked to its
//! target's own timeline holds a conversation-scoped layer's turns for that
//! conversation alone ([`Holder::masked_to`]); any other read holds what it
//! reads for every conversation. Holders only widen, so a turn is dropped for
//! its conversation going away only if nothing else ever read it.

use serde::Serialize;

use crate::projection::{GatherScope, TimelineId};

/// Who a resident turn is held for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Holder {
    /// Read across conversations — preloaded, and given up only by the keep
    /// trade.
    Shared,
    /// Read only by the conversation on this timeline — resident while it is
    /// loaded.
    Conversation(TimelineId),
}

/// A [`Holder`]'s kind, without the timeline — what a breakdown sums by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HolderKind {
    Shared,
    Conversation,
}

impl Holder {
    /// The holder a read masked to `timeline` gives the turns of a layer
    /// gathered `scope`: that conversation's alone when the layer is
    /// conversation-scoped, every conversation's when it is shared — a shared
    /// layer's turn read by its own conversation is still one every other
    /// conversation will read.
    pub fn masked_to(scope: GatherScope, timeline: TimelineId) -> Holder {
        match scope {
            GatherScope::Conversation => Holder::Conversation(timeline),
            GatherScope::Shared => Holder::Shared,
        }
    }

    /// The holder of a turn read by both `self` and `other`: the same
    /// conversation stays its own, anything else is shared.
    pub fn widen(self, other: Holder) -> Holder {
        match (self, other) {
            (Holder::Conversation(a), Holder::Conversation(b)) if a == b => self,
            _ => Holder::Shared,
        }
    }

    pub fn kind(self) -> HolderKind {
        match self {
            Holder::Shared => HolderKind::Shared,
            Holder::Conversation(_) => HolderKind::Conversation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(raw: u64) -> Holder {
        Holder::Conversation(TimelineId::from_raw(raw).expect("timeline id"))
    }

    /// A masked read holds a conversation-scoped layer's turn for that
    /// conversation, and a shared layer's for everyone.
    #[test]
    fn a_masked_read_holds_by_the_layers_scope() {
        let tl = TimelineId::from_raw(7).unwrap();
        assert_eq!(Holder::masked_to(GatherScope::Conversation, tl), conv(7));
        assert_eq!(Holder::masked_to(GatherScope::Shared, tl), Holder::Shared);
    }

    /// **Holders only widen.** A conversation's own reads keep it its own; a
    /// second conversation, or any shared read, makes it shared — in either
    /// order.
    #[test]
    fn a_holder_widens_and_never_narrows() {
        assert_eq!(conv(7).widen(conv(7)), conv(7));
        assert_eq!(conv(7).widen(conv(8)), Holder::Shared);
        assert_eq!(conv(7).widen(Holder::Shared), Holder::Shared);
        assert_eq!(Holder::Shared.widen(conv(7)), Holder::Shared);
        assert_eq!(Holder::Shared.widen(Holder::Shared), Holder::Shared);
    }
}
