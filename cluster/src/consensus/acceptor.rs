//! A member as an acceptor (§5.9).
//!
//! - It **promises** a ballot above every ballot it has promised for the key,
//!   reporting what it has accepted, and refuses any other with the ballot it
//!   promised.
//! - It **accepts** a proposal at a ballot it has not promised against.
//! - A vote is written to the store before the reply that reports it.
//! - An accepted **lease** counts only while its `ttl` runs, from when this
//!   acceptor accepted it, on its own monotonic clock. The vote also records
//!   when that is by the wall clock, and a lease accepted before a restart
//!   counts for what is left of it by the wall clock at the restart, at most
//!   one `ttl` ([`super::rearm`], which states what that assumes). A `RELEASE`
//!   drops it.
//! - A key whose decision the member has **learned** — written as a
//!   replicated record — no longer needs a slot: the acceptor answers for it
//!   from the decision, as an acceptance above every ballot.
//! - Every key is held as [`held_key`] holds it, so a key longer than a
//!   journal slot is voted on under its digest rather than refused.

use std::io;
use std::time::Duration;

use super::armed::ArmedLease;
use super::clock::Millis;
use super::maps::Map;
use super::rearm::rearm;
use super::value::Value;
use super::votes::{held_key, Vote, VoteStore};
use super::wall::WallMillis;
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

/// A lease key's term on this process's clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Term {
    /// When its last lease ends.
    end: Millis,
    /// Its `ttl`.
    ttl: Duration,
    /// When its slot is freed: one `ttl` after the end.
    free_at: Millis,
}

impl Term {
    /// A lease accepted at `now` for `ttl`.
    fn accepted(now: Millis, ttl: Duration) -> Term {
        Term {
            end: now + ttl,
            ttl,
            free_at: now + ttl + ttl,
        }
    }
}

pub struct Acceptor<S: VoteStore> {
    store: S,
    votes: Map<Vec<u8>, Vote>,
    /// Per lease key, its last lease's term on this process's clock.
    leases: Map<Vec<u8>, Term>,
    decided: Map<Vec<u8>, Vec<u8>>,
}

impl<S: VoteStore> Acceptor<S> {
    /// Resume from `store`, reopened at `now` on this process's clock and at
    /// `wall` on the wall clock. Every lease accepted before it counts for
    /// what is left of the end its vote recorded ([`rearm`]).
    pub fn open(mut store: S, now: Millis, wall: WallMillis) -> io::Result<Self> {
        let mut votes = Map::default();
        let mut leases = Map::default();
        for (key, vote) in store.load()? {
            if let Some(a) = &vote.accepted {
                if let Ok(Value::Lease { ttl, .. }) = Value::decode(&a.value) {
                    let r = rearm(vote.lease_ends, ttl, wall);
                    leases.insert(
                        key.clone(),
                        Term {
                            end: now + r.left,
                            ttl,
                            free_at: now + r.free_in,
                        },
                    );
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
        if let Some(term) = self.leases.get(key) {
            if now >= term.end {
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
        let key = &*held_key(key);
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

    /// Accept `value` at `ballot`, at `now` on this process's clock and `wall`
    /// on the wall clock. A lease's term runs from here, and its vote records
    /// `wall` plus its `ttl` as its end, for a restart to count from.
    pub fn propose(
        &mut self,
        key: &[u8],
        ballot: Ballot,
        value: &[u8],
        now: Millis,
        wall: WallMillis,
    ) -> io::Result<ProposeReply> {
        let key = &*held_key(key);
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
        let ttl = match Value::decode(value) {
            Ok(Value::Lease { ttl, .. }) => Some(ttl),
            _ => None,
        };
        let vote = Vote {
            promised: ballot,
            accepted: Some(AcceptedValue {
                ballot,
                value: value.to_vec(),
            }),
            lease_ends: ttl.map_or(WallMillis(0), |ttl| wall + ttl),
        };
        self.store.write(key, &vote)?;
        self.votes.insert(key.to_vec(), vote);
        match ttl {
            Some(ttl) => {
                self.leases.insert(key.to_vec(), Term::accepted(now, ttl));
            }
            None => {
                self.leases.remove(key);
            }
        }
        Ok(ProposeReply::Accepted)
    }

    /// The holder of the lease accepted at `ballot` handed it back at `now`.
    pub fn release(&mut self, key: &[u8], ballot: Ballot, now: Millis) -> io::Result<()> {
        let key = &*held_key(key);
        let Some(vote) = self.votes.get(key) else {
            return Ok(());
        };
        if vote.accepted.as_ref().map(|a| a.ballot) != Some(ballot) {
            return Ok(());
        }
        let vote = Vote {
            promised: vote.promised,
            accepted: None,
            lease_ends: WallMillis(0),
        };
        self.store.write(key, &vote)?;
        self.votes.insert(key.to_vec(), vote);
        if let Some(term) = self.leases.get_mut(key) {
            term.end = now;
            term.free_at = now + term.ttl;
        }
        Ok(())
    }

    /// `key`'s decision is recorded in the substrate: answer from it, and
    /// free the slot.
    pub fn learn(&mut self, key: &[u8], value: Vec<u8>) -> io::Result<()> {
        let key = &*held_key(key);
        self.decided.insert(key.to_vec(), value);
        if self.votes.remove(key).is_some() {
            self.store.free(key)?;
        }
        self.leases.remove(key);
        Ok(())
    }

    /// Free the slots of lease keys whose last lease ended at least one `ttl`
    /// ago, by this acceptor's clock — a lease that ended before a restart by
    /// the end its vote recorded.
    pub fn free_idle_leases(&mut self, now: Millis) -> io::Result<usize> {
        let idle: Vec<Vec<u8>> = self
            .leases
            .iter()
            .filter(|(_, term)| now >= term.free_at)
            .map(|(key, _)| key.clone())
            .collect();
        for key in &idle {
            self.votes.remove(key);
            self.leases.remove(key);
            self.store.free(key)?;
        }
        Ok(idle.len())
    }

    /// Every lease this acceptor has accepted and still counts at `now`: its
    /// key as held, its holder, and how long its term has left on this
    /// acceptor's clock. In key order.
    pub fn armed_leases(&self, now: Millis) -> Vec<ArmedLease> {
        let mut armed: Vec<ArmedLease> = self
            .leases
            .iter()
            .filter(|(_, term)| now < term.end)
            .filter_map(|(key, term)| {
                let accepted = self.votes.get(key)?.accepted.as_ref()?;
                match Value::decode(&accepted.value) {
                    Ok(Value::Lease { holder, .. }) => Some(ArmedLease {
                        key: key.clone(),
                        holder,
                        left: term.end - now,
                    }),
                    _ => None,
                }
            })
            .collect();
        armed.sort();
        armed
    }

    /// The vote held for `key`, for tests and the status page.
    pub fn vote(&self, key: &[u8]) -> Option<&Vote> {
        self.votes.get(&*held_key(key))
    }

    /// Every key this acceptor holds a vote for, as [`held_key`] holds it.
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

    /// The wall clock `s` seconds after 2026-10-26T07:33:20Z.
    const T0: u64 = 1_793_000_000_000;

    fn wall(s: u64) -> WallMillis {
        WallMillis(T0 + s * 1000)
    }

    /// An acceptor whose clocks both start at `s`.
    fn opened(store: MemoryVotes, s: u64) -> Acceptor<MemoryVotes> {
        Acceptor::open(store, at(s), wall(s)).unwrap()
    }

    /// Whether a proposer other than the holder, preparing at `round`, is
    /// shown a lease — and so stops, busy.
    fn busy_for_another(a: &mut Acceptor<MemoryVotes>, round: u64, now: Millis) -> bool {
        match a.prepare(b"upgrade/zend", b(round, 9), now).unwrap() {
            PrepareReply::Promise(accepted) => accepted.is_some(),
            PrepareReply::Nack(n) => panic!("round {round} refused: {n:?}"),
        }
    }

    const MIN: u64 = 60;

    /// A millisecond before `t`.
    fn before(t: Millis) -> Millis {
        t - Duration::from_millis(1)
    }

    /// The term the only armed lease has left at `now`.
    fn left(a: &Acceptor<MemoryVotes>, now: Millis) -> Duration {
        a.armed_leases(now)[0].left
    }

    fn mins(m: u64) -> Duration {
        Duration::from_secs(m * MIN)
    }

    /// Node 4 takes `upgrade/zend` for 20 minutes at wall `T0`.
    fn upgrade_lease_taken() -> MemoryVotes {
        let store = MemoryVotes::new();
        let mut a = opened(store.clone(), 0);
        a.prepare(b"upgrade/zend", b(1, 4), at(0)).unwrap();
        let value = lease(4, 20 * MIN);
        a.propose(b"upgrade/zend", b(1, 4), &value, at(0), wall(0))
            .unwrap();
        assert_eq!(
            a.vote(b"upgrade/zend").unwrap().lease_ends,
            wall(20 * MIN),
            "the vote records the end by the wall clock"
        );
        store
    }

    /// An acceptor promises only above its highest promised ballot, NACKs the
    /// rest with that ballot, and reports what it accepted.
    #[test]
    fn promises_only_above_and_reports_its_acceptance() {
        let mut a = opened(MemoryVotes::new(), 0);
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
            a.propose(b"k", b(1, 9), b"\0x", at(0), wall(0)).unwrap(),
            ProposeReply::Nack(b(2, 1))
        );
        assert_eq!(
            a.propose(b"k", b(2, 1), b"\0v", at(0), wall(0)).unwrap(),
            ProposeReply::Accepted
        );
        assert_eq!(
            a.vote(b"k").unwrap().lease_ends,
            WallMillis(0),
            "a decision records no end"
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
        let mut a = opened(store.clone(), 0);
        a.prepare(b"k", b(5, 1), at(0)).unwrap();
        a.propose(b"k", b(5, 1), b"\0v", at(0), wall(0)).unwrap();
        drop(a);
        let mut a = opened(store, 100);
        assert_eq!(
            a.prepare(b"k", b(4, 9), at(100)).unwrap(),
            PrepareReply::Nack(b(5, 1))
        );
        let PrepareReply::Promise(Some(acc)) = a.prepare(b"k", b(6, 9), at(100)).unwrap() else {
            panic!()
        };
        assert_eq!(acc.value, b"\0v");
    }

    /// An accepted lease is reported only while its ttl runs from acceptance.
    #[test]
    fn a_lease_counts_for_its_term() {
        let mut a = opened(MemoryVotes::new(), 0);
        a.prepare(b"w", b(1, 1), at(10)).unwrap();
        a.propose(b"w", b(1, 1), &lease(1, 30), at(10), wall(10))
            .unwrap();
        let PrepareReply::Promise(Some(_)) = a.prepare(b"w", b(2, 2), at(39)).unwrap() else {
            panic!("still running at 39 s")
        };
        assert_eq!(
            a.prepare(b"w", b(3, 2), at(40)).unwrap(),
            PrepareReply::Promise(None)
        );
    }

    /// Reopened 5 minutes into a 20-minute lease, on a fresh process clock:
    /// armed for the 15 minutes left, then free.
    #[test]
    fn a_restart_inside_the_term_arms_what_is_left() {
        let store = upgrade_lease_taken();
        let mut a = Acceptor::open(store, Millis(0), wall(5 * MIN)).unwrap();
        assert_eq!(
            a.armed_leases(Millis(0)),
            [ArmedLease {
                key: b"upgrade/zend".to_vec(),
                holder: 4,
                left: Duration::from_secs(15 * MIN),
            }]
        );
        assert!(busy_for_another(&mut a, 2, before(at(15 * MIN))));
        assert!(!busy_for_another(&mut a, 3, at(15 * MIN)));
    }

    /// Reopened after the lease's end: nothing is armed, another proposer
    /// takes it at once, and its slot frees one ttl after the recorded end.
    #[test]
    fn a_restart_past_the_term_arms_nothing() {
        let store = upgrade_lease_taken();
        let mut a = Acceptor::open(store.clone(), Millis(0), wall(25 * MIN)).unwrap();
        assert!(a.armed_leases(Millis(0)).is_empty());
        assert!(!busy_for_another(&mut a, 2, Millis(0)));
        // The end was T0 + 20 min, so the slot frees at T0 + 40 min: 15 min on.
        assert_eq!(a.free_idle_leases(before(at(15 * MIN))).unwrap(), 0);
        assert_eq!(a.free_idle_leases(at(15 * MIN)).unwrap(), 1);
        assert!(store.clone().load().unwrap().is_empty());
    }

    /// Reopened with the wall clock earlier than the acceptance: it went back,
    /// so the lease counts for its full term from the reopen.
    #[test]
    fn a_restart_with_the_clock_gone_back_arms_the_full_term() {
        let store = upgrade_lease_taken();
        let mut a = Acceptor::open(store, Millis(0), WallMillis(T0 - 60_000)).unwrap();
        assert_eq!(left(&a, Millis(0)), mins(20));
        assert!(busy_for_another(&mut a, 2, before(at(20 * MIN))));
        assert!(!busy_for_another(&mut a, 3, at(20 * MIN)));
    }

    /// **Restarts do not extend a lease.** The gateway's tie-break acceptor
    /// restarts on every web deploy, and each restart re-armed `upgrade/zend`
    /// for a full 20 minutes from itself, so a lease nobody renewed stayed busy
    /// for as long as deploys kept coming (2026-10-31, 09:52 to ~10:02). Two
    /// reopens at T+5 and T+10 both end it at T+20.
    #[test]
    fn repeated_restarts_never_extend_the_lease_past_its_end() {
        let store = upgrade_lease_taken();
        let a = Acceptor::open(store.clone(), Millis(0), wall(5 * MIN)).unwrap();
        assert_eq!(left(&a, Millis(0)), mins(15));
        drop(a);
        let mut a = Acceptor::open(store.clone(), Millis(0), wall(10 * MIN)).unwrap();
        assert_eq!(left(&a, Millis(0)), mins(10), "T+10 + 10 min is still T+20");
        assert_eq!(
            store.clone().load().unwrap()[0].1.lease_ends,
            wall(20 * MIN),
            "a reopen leaves the recorded end alone"
        );
        assert!(busy_for_another(&mut a, 2, before(at(10 * MIN))));
        assert!(!busy_for_another(&mut a, 3, at(10 * MIN)));
    }

    /// A promise keeps the recorded end; a renewal moves it.
    #[test]
    fn a_renewal_records_its_new_end() {
        let store = upgrade_lease_taken();
        let mut a = opened(store.clone(), 5 * MIN);
        a.prepare(b"upgrade/zend", b(2, 4), at(5 * MIN)).unwrap();
        assert_eq!(a.vote(b"upgrade/zend").unwrap().lease_ends, wall(20 * MIN));
        a.propose(
            b"upgrade/zend",
            b(2, 4),
            &lease(4, 20 * MIN),
            at(5 * MIN),
            wall(5 * MIN),
        )
        .unwrap();
        let ends = store.clone().load().unwrap()[0].1.lease_ends;
        assert_eq!(ends, wall(25 * MIN));
        let a = Acceptor::open(store, Millis(0), wall(10 * MIN)).unwrap();
        assert_eq!(left(&a, Millis(0)), mins(15));
    }

    /// The armed leases name each key and holder with the term it has left; a
    /// decision, an expired lease and a released one are not among them.
    #[test]
    fn armed_leases_are_the_unexpired_ones_by_key() {
        let mut a = opened(MemoryVotes::new(), 0);
        a.propose(b"w1", b(1, 4), &lease(4, 30), at(10), wall(10))
            .unwrap();
        a.propose(b"w2", b(1, 7), &lease(7, 20), at(10), wall(10))
            .unwrap();
        a.propose(b"d", b(1, 7), b"\0x", at(10), wall(10)).unwrap();
        a.propose(b"w3", b(1, 9), &lease(9, 30), at(10), wall(10))
            .unwrap();
        a.release(b"w3", b(1, 9), at(12)).unwrap();
        let armed = |key: &[u8], holder, s| ArmedLease {
            key: key.to_vec(),
            holder,
            left: Duration::from_secs(s),
        };
        assert_eq!(
            a.armed_leases(at(15)),
            [armed(b"w1", 4, 25), armed(b"w2", 7, 15)]
        );
        assert_eq!(a.armed_leases(at(30)), [armed(b"w1", 4, 10)]);
        assert!(a.armed_leases(at(40)).is_empty());
    }

    #[test]
    fn a_release_drops_the_lease_at_once() {
        let mut a = opened(MemoryVotes::new(), 0);
        a.prepare(b"w", b(1, 1), at(0)).unwrap();
        a.propose(b"w", b(1, 1), &lease(1, 30), at(0), wall(0))
            .unwrap();
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
        let mut a = opened(store.clone(), 0);
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
            a.propose(b"gen/3", b(9, 9), b"\0eight", at(0), wall(0))
                .unwrap(),
            ProposeReply::Nack(DECIDED)
        );
        assert_eq!(
            a.propose(b"gen/3", b(9, 9), b"\0seven", at(0), wall(0))
                .unwrap(),
            ProposeReply::Accepted
        );
    }

    #[test]
    fn an_idle_lease_frees_its_slot_one_ttl_after_its_term() {
        let store = MemoryVotes::new();
        let mut a = opened(store.clone(), 0);
        a.prepare(b"w", b(1, 1), at(0)).unwrap();
        a.propose(b"w", b(1, 1), &lease(1, 30), at(0), wall(0))
            .unwrap();
        a.prepare(b"gen/1", b(1, 1), at(0)).unwrap();
        a.propose(b"gen/1", b(1, 1), b"\0d", at(0), wall(0))
            .unwrap();
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
        let mut a = opened(store.clone(), 0);
        a.prepare(b"w", b(1, 1), at(0)).unwrap();
        a.propose(b"w", b(1, 1), &lease(1, 30), at(0), wall(0))
            .unwrap();
        a.release(b"w", b(1, 1), at(5)).unwrap();
        assert_eq!(
            a.vote(b"w").unwrap().lease_ends,
            WallMillis(0),
            "a released lease records no end"
        );
        assert_eq!(a.free_idle_leases(at(34)).unwrap(), 0);
        assert_eq!(a.free_idle_leases(at(35)).unwrap(), 1);
        assert!(store.clone().load().unwrap().is_empty());
    }
}
