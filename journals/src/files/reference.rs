//! A reference (§2.5): a named link from one journal to another in the same
//! group, listed in the linking journal's `journal.yaml`. A path through one
//! is the target's path; it points at a journal, never at another reference.

use crate::path::Below;
use crate::slug::Slug;

/// The most references one journal holds.
pub const MAX_REFS: usize = 10;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ref {
    /// The name a path takes through it, beside the journal's children.
    pub name: Slug,
    /// The journal it leads to, below the group's master.
    pub to: Below,
    /// "Open this when…", as an index shows it.
    pub cue: String,
}
