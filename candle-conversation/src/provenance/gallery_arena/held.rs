//! Records who a resident turn is held for, and which of its tokens and fold
//! groups it holds, so one reader's hold never frees another reader's turn.
//!
//! A holder only widens, so a turn held for one conversation and then read by
//! another becomes shared rather than being freed under the first. A re-upload
//! keeps the union of what the old pages held and what the new read weighs. A re-seal
//! is a new turn to its readers, so the old holder and keep stand only while the
//! content is unchanged. A read that supplies no codebook to choose tokens holds
//! every token, so a turn is never treated as reduced by a keep it was not given.

use super::*;

/// What a turn's upload holds: who for, which fold groups, which tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Held {
    pub(super) holder: Holder,
    pub(super) groups: GroupSet,
    pub(super) keep: Keep,
}

impl Held {
    /// What an upload of the content `fingerprint`, read for `holder`
    /// weighing `groups` of the tokens `keep` keeps, holds over `old`, the
    /// entry it replaces: the wider holder, the union of the groups, and of the
    /// keeps while the content is unchanged — a re-seal is a new turn to its
    /// readers. A keep with no codebook (`has_book` false) to name its tokens
    /// holds them all, and says so: the turn is not held under a keep it was
    /// never reduced by.
    pub(super) fn widened(
        old: Option<&ResidentTurn>,
        fingerprint: u64,
        holder: Holder,
        groups: GroupSet,
        keep: Keep,
        has_book: bool,
    ) -> Self {
        let mut held = Held {
            holder,
            groups,
            keep,
        };
        if let Some(old) = old {
            held.holder = old.holder.widen(holder);
            held.groups = old.groups.union(groups);
            if old.fingerprint == fingerprint {
                held.keep = old.keep.union(keep);
            }
        }
        if !has_book {
            held.keep = Keep::All;
        }
        held
    }
}
