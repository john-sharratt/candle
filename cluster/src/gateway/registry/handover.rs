//! A sticky key's last release, and who may take its next turn.

use std::collections::BTreeSet;

use super::MemberView;

/// A sticky key let go ([`super::Registry::released`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Handover {
    pub(super) by: u64,
    pub(super) through: u64,
    /// The members that reported holding the key complete since this release
    /// ([`super::Registry::completed`]): its next turn may go to them whatever
    /// else of its last holder's they lack.
    pub(super) complete: BTreeSet<u64>,
}

impl Handover {
    /// Whether `m` may take the key's next turn: it holds the key complete, or
    /// every record of its last holder's through the release.
    pub(super) fn held_by(&self, m: &MemberView) -> bool {
        self.complete.contains(&m.node) || m.holds_through(self.by, self.through)
    }
}
