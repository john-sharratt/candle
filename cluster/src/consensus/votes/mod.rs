//! An acceptor's durable state (§5.9): per Paxos key, the ballot it promised
//! and the ballot and value it accepted. A vote is durable before the frame
//! that reports it is sent. It is this member's own, so it is never
//! replicated.
//!
//! [`FileJournal`] is the member's `substrate/VOTES`: preallocated, written in
//! place, never grown, so a member can vote with its volume full (§7.7).
//! [`MemoryVotes`] holds the same state in memory, for simulations that model
//! a restart by keeping the store and dropping everything else.

mod file;
mod held_key;
mod memory;

pub use file::{FileJournal, DEFAULT_SIZE, HALF, MAX_KEY, MAX_VALUE};
pub use held_key::held_key;
pub use memory::MemoryVotes;

use std::io;

use crate::frame::{AcceptedValue, Ballot};

/// One key's vote.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Vote {
    pub promised: Ballot,
    pub accepted: Option<AcceptedValue>,
}

pub trait VoteStore: Send {
    /// Every vote held, as the store was last left.
    fn load(&mut self) -> io::Result<Vec<(Vec<u8>, Vote)>>;

    /// Record `vote` for `key`, durably, before returning.
    fn write(&mut self, key: &[u8], vote: &Vote) -> io::Result<()>;

    /// Forget `key`, freeing its place.
    fn free(&mut self, key: &[u8]) -> io::Result<()>;
}

/// A member's journal chosen at run time: the file on a daemon, memory in a
/// simulation.
impl VoteStore for Box<dyn VoteStore> {
    fn load(&mut self) -> io::Result<Vec<(Vec<u8>, Vote)>> {
        (**self).load()
    }

    fn write(&mut self, key: &[u8], vote: &Vote) -> io::Result<()> {
        (**self).write(key, vote)
    }

    fn free(&mut self, key: &[u8]) -> io::Result<()> {
        (**self).free(key)
    }
}
