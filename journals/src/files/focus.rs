//! A conversation's focus (§2.4): who it has worked out is speaking, and
//! the journals it is about, in its `journal.yaml`. Every call's frame starts
//! from it.

use crate::path::Below;
use crate::slug::Slug;

/// The most journals a conversation is about at once.
pub const MAX_ABOUT: usize = 3;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Focus {
    /// The people journal of whoever is talking, once the model knows.
    pub speaker: Option<Slug>,
    /// The journals the conversation is about, below the group's master.
    pub about: Vec<Below>,
}

impl Focus {
    pub fn is_empty(&self) -> bool {
        self.speaker.is_none() && self.about.is_empty()
    }
}
