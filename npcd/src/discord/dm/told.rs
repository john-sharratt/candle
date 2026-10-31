//! Who a bot has already answered instead of hearing (`docs/npcd_discord.md`
//! §3.2): an unlinked author is told once how to link, and a person who may
//! not talk with the character is told once that they may not. After that
//! their messages are dropped without an answer, so nobody can make a bot
//! talk on a loop.
//!
//! Kept in memory, per bot: a restart tells each of them once more.

use std::collections::HashSet;

/// The Discord users one bot has told.
#[derive(Debug, Default)]
pub struct Told {
    unlinked: HashSet<u64>,
    refused: HashSet<u64>,
}

impl Told {
    /// Whether unlinked `author` is to be told how to link: the first time.
    pub fn first_unlinked(&mut self, author: u64) -> bool {
        self.unlinked.insert(author)
    }

    /// Whether `author` is to be told the character will not talk with them:
    /// the first time.
    pub fn first_refused(&mut self, author: u64) -> bool {
        self.refused.insert(author)
    }

    /// `author` has just linked: what they were told before is history.
    pub fn forget(&mut self, author: u64) {
        self.unlinked.remove(&author);
        self.refused.remove(&author);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_is_told_once_until_they_link() {
        let mut told = Told::default();
        assert!(told.first_unlinked(7));
        assert!(!told.first_unlinked(7));
        assert!(told.first_unlinked(8));
        assert!(told.first_refused(7));
        assert!(!told.first_refused(7));
        told.forget(7);
        assert!(told.first_unlinked(7));
        assert!(told.first_refused(7));
    }
}
