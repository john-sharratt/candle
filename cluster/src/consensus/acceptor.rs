//! A member as an acceptor (§5.9).
//!
//! - It **promises** a ballot above every ballot it has promised for the key,
//!   reporting what it has accepted, and refuses any other with the ballot it
//!   promised.
//! - It **accepts** a proposal at a ballot it has not promised against.
//! - A vote is written to the store before the reply that reports it.
//! - An accepted **lease** counts only while its `ttl` runs, from when this
//!   acceptor accepted it, on its own monotonic clock. A lease accepted before
//!   a restart counts for one full `ttl` from the restart: the clock began
//!   again, and that errs on the side of the old holder. A `RELEASE` drops it.
//! - A key whose decision the member has **learned** — written as a
//!   replicated record — no longer needs a slot: the acceptor answers for it
//!   from the decision, as an acceptance above every ballot.

use std::io;
use std::time::Duration;

use super::clock::Millis;
use super::maps::Map;
use super::value::Value;
use super::votes::{Vote, VoteStore};
use crate::frame::{AcceptedValue, Ballot};

/// A ballot above every ballot a proposer can hold: what a learned decision
/// answers with, so a proposer always adopts it.
pub const DECIDED: Ballot = Ballot {
    round: u64::MAX,
    node: u64::MAX,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrepareReply {
    Promise(Option<AcceptedValue>),
    Nack(Ballot),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProposeReply {
    Accepted,
    Nack(Ballot),
}

pub struct Acceptor<S: VoteStore> {
    store: S,
    votes: Map<Vec<u8>, Vote>,
    /// Per lease key, when its last lease ends on this process's clock, and
    /// its term: the slot is freed one term after the end.
    leases: Map<Vec<u8>, (Millis, Duration)>,
    decided: Map<Vec<u8>, Vec<u8>>,
}

impl<S: VoteStore> Acceptor<S> {
    /// Resume from `store`. `now` is the restart: every lease accepted before
    /// it counts from it.
    pub fn open(mut store: S, now: Millis) -> io::Result<Self> {
        let mut votes = Map::default();
        let mut leases = Map::default();
        for (key, vote) in store.load()? {
            if let Some(a) = &vote.accepted {
                if let Ok(Value::Lease { ttl, .. }) = Value::decode(&a.value) {
                    leases.insert(key.clone(), (now + ttl, ttl));
                }
            }
            votes.insert(key, vote);
        }
        Ok(Self {
            store,
            votes,
            leases,
            decided: Map::default(),
        })
    }

    /// The accepted value `key` reports now: a learned decision, or the
    /// accepted value — a lease only while its term runs.
    fn reported(&self, key: &[u8], now: Millis) -> Option<AcceptedValue> {
        if let Some(value) = self.decided.get(key) {
            return Some(AcceptedValue {
                ballot: DECIDED,
                value: value.clone(),
            });
        }
        let accepted = self.votes.get(key)?.accepted.as_ref()?;
        if let Some((end, _)) = self.leases.get(key) {
            if now >= *end {
                return None;
            }
        }
        Some(accepted.clone())
    }

    fn promised(&self, key: &[u8]) -> Ballot {
        if self.decided.contains_key(key) {
            return DECIDED;
        }
        self.votes.get(key).map(|v| v.promised).unwrap_or_default()
    }

    pub fn prepare(&mut self, key: &[u8], ballot: Ballot, now: Millis) -> io::Result<PrepareReply> {
        if self.decided.contains_key(key) {
            // Every proposer adopts a learned decision.
            return Ok(PrepareReply::Promise(self.reported(key, now)));
        }
        let promised = self.promised(key);
        if ballot <= promised {
            return Ok(PrepareReply::Nack(promised));
        }
        let mut vote = self.votes.get(key).cloned().unwrap_or_default();
        vote.promised = ballot;
        self.store.write(key, &vote)?;
        self.votes.insert(key.to_vec(), vote);
        Ok(PrepareReply::Promise(self.reported(key, now)))
    }

    pub fn propose(
        &mut self,
        key: &[u8],
        ballot: Ballot,
        value: &[u8],
        now: Millis,
    ) -> io::Result<ProposeReply> {
        if let Some(decided) = self.decided.get(key) {
            return Ok(if decided == value {
                ProposeReply::Accepted
            } else {
                ProposeReply::Nack(DECIDED)
            });
        }
        let promised = self.promised(key);
        if ballot < promised {
            return Ok(ProposeReply::Nack(promised));
        }
        let vote = Vote {
            promised: ballot,
            accepted: Some(AcceptedValue {
                ballot,
                value: value.to_vec(),
            }),
        };
        self.store.write(key, &vote)?;
        self.votes.insert(key.to_vec(), vote);
        match Value::decode(value) {
            Ok(Value::Lease { ttl, .. }) => {
                self.leases.insert(key.to_vec(), (now + ttl, ttl));
            }
            _ => {
                self.leases.remove(key);
            }
        }
        Ok(ProposeReply::Accepted)
    }

    /// The holder of the lease accepted at `ballot` handed it back at `now`.
    pub fn release(&mut self, key: &[u8], ballot: Ballot, now: Millis) -> io::Result<()> {
        let Some(vote) = self.votes.get(key) else {
            return Ok(());
        };
        if vote.accepted.as_ref().map(|a| a.ballot) != Some(ballot) {
            return Ok(());
        }
        let vote = Vote {
            promised: vote.promised,
            accepted: None,
        };
        self.store.write(key, &vote)?;
        self.votes.insert(key.to_vec(), vote);
        if let Some((end, _)) = self.leases.get_mut(key) {
            *end = now;
        }
        Ok(())
    }

    /// `key`'s decision is recorded in the substrate: answer from it, and
    /// free the slot.
    pub fn learn(&mut self, key: &[u8], value: Vec<u8>) -> io::Result<()> {
        self.decided.insert(key.to_vec(), value);
        if self.votes.remove(key).is_some() {
            self.store.free(key)?;
        }
        self.leases.remove(key);
        Ok(())
    }

    /// Free the slots of lease keys whose last lease ended at least one `ttl`
    /// ago, by this acceptor's clock.
    pub fn free_idle_leases(&mut self, now: Millis) -> io::Result<usize> {
        let idle: Vec<Vec<u8>> = self
            .leases
            .iter()
            .filter(|(_, (end, ttl))| now >= *end + *ttl)
            .map(|(key, _)| key.clone())
            .collect();
        for key in &idle {
            self.votes.remove(key);
            self.leases.remove(key);
            self.store.free(key)?;
        }
        Ok(idle.len())
    }

    /// The vote held for `key`, for tests and the status page.
    pub fn vote(&self, key: &[u8]) -> Option<&Vote> {
        self.votes.get(key)
    }

    /// Every key this acceptor holds a vote for.
    pub fn keys(&self) -> impl Iterator<Item = &Vec<u8>> {
        self.votes.keys()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::votes::MemoryVotes;

    fn b(round: u64, node: u64) -> Ballot {
        Ballot { round, node }
    }

    fn lease(holder: u64, ttl_s: u64) -> Vec<u8> {
        Value::Lease {
            holder,
            ttl: Duration::from_secs(ttl_s),
        }
        .encode()
    }

    fn at(s: u64) -> Millis {
        Millis(s * 1000)
    }

    /// An acceptor promises only above its highest promised ballot, NACKs the
    /// rest with that ballot, and reports what it accepted.
    #[test]
    fn promises_only_above_and_reports_its_acceptance() {
        let mut a = Acceptor::open(MemoryVotes::new(), at(0)).unwrap();
        assert_eq!(
            a.prepare(b"k", b(2, 1), at(0)).unwrap(),
            PrepareReply::Promise(None)
        );
        assert_eq!(
            a.prepare(b"k", b(2, 1), at(0)).unwrap(),
            PrepareReply::Nack(b(2, 1))
        );
        assert_eq!(
            a.prepare(b"k", b(1, 9), at(0)).unwrap(),
            PrepareReply::Nack(b(2, 1))
        );
        assert_eq!(
            a.propose(b"k", b(1, 9), b"\0x", at(0)).unwrap(),
            ProposeReply::Nack(b(2, 1))
        );
        assert_eq!(
            a.propose(b"k", b(2, 1), b"\0v", at(0)).unwrap(),
            ProposeReply::Accepted
        );
        let PrepareReply::Promise(Some(acc)) = a.prepare(b"k", b(3, 2), at(0)).unwrap() else {
            panic!()
        };
        assert_eq!((acc.ballot, acc.value), (b(2, 1), b"\0v".to_vec()));
    }

    /// Promises and acceptances are in the store before the reply, and a
    /// restarted acceptor resumes from them.
    #[test]
    fn votes_are_durable_across_a_restart() {
        let store = MemoryVotes::new();
        let mut a = Acceptor::open(store.clone(), at(0)).unwrap();
        a.prepare(b"k", b(5, 1), at(0)).unwrap();
        a.propose(b"k", b(5, 1), b"\0v", at(0)).unwrap();
        drop(a);
        let mut a = Acceptor::open(store, at(100)).unwrap();
        assert_eq!(
            a.prepare(b"k", b(4, 9), at(100)).unwrap(),
            PrepareReply::Nack(b(5, 1))
        );
        let PrepareReply::Promise(Some(acc)) = a.prepare(b"k", b(6, 9), at(100)).unwrap() else {
            panic!()
        };
        assert_eq!(acc.value, b"\0v");
    }

    /// An accepted lease is reported only while its ttl runs from acceptance;
    /// after a restart it counts for a full ttl from the restart.
    #[test]
    fn a_lease_counts_for_its_term_and_a_full_term_after_a_restart() {
        let store = MemoryVotes::new();
        let mut a = Acceptor::open(store.clone(), at(0)).unwrap();
        a.prepare(b"w", b(1, 1), at(10)).unwrap();
        a.propose(b"w", b(1, 1), &lease(1, 30), at(10)).unwrap();
        let PrepareReply::Promise(Some(_)) = a.prepare(b"w", b(2, 2), at(39)).unwrap() else {
            panic!("still running at 39 s")
        };
        assert_eq!(
            a.prepare(b"w", b(3, 2), at(40)).unwrap(),
            PrepareReply::Promise(None)
        );
        drop(a);

        let mut a = Acceptor::open(store.clone(), at(0)).unwrap();
        a.prepare(b"x", b(1, 1), at(0)).unwrap();
        a.propose(b"x", b(1, 1), &lease(1, 30), at(25)).unwrap();
        drop(a);
        // Restarted at 5 s on a fresh clock: the lease counts until 35 s.
        let mut a = Acceptor::open(store, at(5)).unwrap();
        let PrepareReply::Promise(Some(_)) = a.prepare(b"x", b(2, 2), at(34)).unwrap() else {
            panic!("counts a full ttl from the restart")
        };
        assert_eq!(
            a.prepare(b"x", b(3, 2), at(35)).unwrap(),
            PrepareReply::Promise(None)
        );
    }

    #[test]
    fn a_release_drops_the_lease_at_once() {
        let mut a = Acceptor::open(MemoryVotes::new(), at(0)).unwrap();
        a.prepare(b"w", b(1, 1), at(0)).unwrap();
        a.propose(b"w", b(1, 1), &lease(1, 30), at(0)).unwrap();
        a.release(b"w", b(9, 9), at(0)).unwrap();
        assert!(
            a.vote(b"w").unwrap().accepted.is_some(),
            "another ballot's release"
        );
        a.release(b"w", b(1, 1), at(0)).unwrap();
        assert_eq!(
            a.prepare(b"w", b(2, 2), at(1)).unwrap(),
            PrepareReply::Promise(None)
        );
    }

    /// A learned decision answers for its key above every ballot and frees the
    /// slot; a proposal of another value is refused.
    #[test]
    fn a_learned_decision_answers_for_its_key() {
        let store = MemoryVotes::new();
        let mut a = Acceptor::open(store.clone(), at(0)).unwrap();
        a.prepare(b"gen/3", b(1, 1), at(0)).unwrap();
        a.learn(b"gen/3", b"\0seven".to_vec()).unwrap();
        assert!(store.clone().load().unwrap().is_empty());
        let PrepareReply::Promise(Some(acc)) =
            a.prepare(b"gen/3", b(u64::MAX - 1, 1), at(0)).unwrap()
        else {
            panic!()
        };
        assert_eq!((acc.ballot, acc.value), (DECIDED, b"\0seven".to_vec()));
        assert_eq!(
            a.propose(b"gen/3", b(9, 9), b"\0eight", at(0)).unwrap(),
            ProposeReply::Nack(DECIDED)
        );
        assert_eq!(
            a.propose(b"gen/3", b(9, 9), b"\0seven", at(0)).unwrap(),
            ProposeReply::Accepted
        );
    }

    #[test]
    fn an_idle_lease_frees_its_slot_one_ttl_after_its_term() {
        let store = MemoryVotes::new();
        let mut a = Acceptor::open(store.clone(), at(0)).unwrap();
        a.prepare(b"w", b(1, 1), at(0)).unwrap();
        a.propose(b"w", b(1, 1), &lease(1, 30), at(0)).unwrap();
        a.prepare(b"gen/1", b(1, 1), at(0)).unwrap();
        a.propose(b"gen/1", b(1, 1), b"\0d", at(0)).unwrap();
        assert_eq!(a.free_idle_leases(at(59)).unwrap(), 0);
        assert_eq!(a.free_idle_leases(at(60)).unwrap(), 1);
        let left: Vec<Vec<u8>> = store
            .clone()
            .load()
            .unwrap()
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(
            left,
            vec![b"gen/1".to_vec()],
            "a decision is never freed by age"
        );
    }

    /// A released lease ends at its release: its slot frees one ttl later.
    #[test]
    fn a_released_lease_frees_its_slot_one_ttl_after_the_release() {
        let store = MemoryVotes::new();
        let mut a = Acceptor::open(store.clone(), at(0)).unwrap();
        a.prepare(b"w", b(1, 1), at(0)).unwrap();
        a.propose(b"w", b(1, 1), &lease(1, 30), at(0)).unwrap();
        a.release(b"w", b(1, 1), at(5)).unwrap();
        assert_eq!(a.free_idle_leases(at(34)).unwrap(), 0);
        assert_eq!(a.free_idle_leases(at(35)).unwrap(), 1);
        assert!(store.clone().load().unwrap().is_empty());
    }
}
