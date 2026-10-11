//! Votes in memory: the same contract as the journal, with nothing on disk.

use std::io;
use std::sync::{Arc, Mutex};

use super::super::maps::Map;
use super::{Vote, VoteStore};

/// A store a simulation keeps across a member's restart: clones share it.
#[derive(Clone, Default)]
pub struct MemoryVotes {
    votes: Arc<Mutex<Map<Vec<u8>, Vote>>>,
}

impl MemoryVotes {
    pub fn new() -> Self {
        Self::default()
    }
}

impl VoteStore for MemoryVotes {
    fn load(&mut self) -> io::Result<Vec<(Vec<u8>, Vote)>> {
        Ok(self
            .votes
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    }

    fn write(&mut self, key: &[u8], vote: &Vote) -> io::Result<()> {
        self.votes
            .lock()
            .unwrap()
            .insert(key.to_vec(), vote.clone());
        Ok(())
    }

    fn free(&mut self, key: &[u8]) -> io::Result<()> {
        self.votes.lock().unwrap().remove(key);
        Ok(())
    }
}
