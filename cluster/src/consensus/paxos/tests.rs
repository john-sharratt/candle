//! The engine among a few members on an in-memory network.

use std::collections::VecDeque;
use std::time::Duration;

use super::*;
use crate::consensus::votes::MemoryVotes;

/// Members, the frames in flight, and virtual time.
struct Net {
    members: Vec<Paxos<MemoryVotes>>,
    stores: Vec<MemoryVotes>,
    /// Frames waiting, by recipient index.
    queue: VecDeque<(usize, Frame)>,
    now: Millis,
    /// Members whose frames are dropped, both ways.
    cut: Vec<bool>,
    /// Every frame is delivered twice, as through two clusters.
    doubled: bool,
    events: Vec<(usize, Out)>,
}

impl Net {
    fn new(n: usize, quorum: usize) -> Net {
        let stores: Vec<MemoryVotes> = (0..n).map(|_| MemoryVotes::new()).collect();
        let members = (0..n)
            .map(|i| {
                let mut p = Paxos::open(
                    Config::new(i as u64 + 1),
                    stores[i].clone(),
                    Millis(0),
                    i as u64,
                )
                .unwrap();
                p.set_quorum(quorum);
                p
            })
            .collect();
        Net {
            members,
            stores,
            queue: VecDeque::new(),
            now: Millis(0),
            cut: vec![false; n],
            doubled: false,
            events: Vec::new(),
        }
    }

    fn index(&self, node: u64) -> usize {
        node as usize - 1
    }

    fn route(&mut self, from: usize, outs: Vec<Out>) {
        for out in outs {
            match out {
                Out::Broadcast(frame) => {
                    for to in 0..self.members.len() {
                        if to != from {
                            self.send(from, to, frame.clone());
                        }
                    }
                }
                Out::Reply { to, frame } => {
                    let to = self.index(to);
                    self.send(from, to, frame);
                }
                other => self.events.push((from, other)),
            }
        }
    }

    fn send(&mut self, from: usize, to: usize, frame: Frame) {
        if self.cut[from] || self.cut[to] {
            return;
        }
        if self.doubled {
            self.queue.push_back((to, frame.clone()));
        }
        self.queue.push_back((to, frame));
    }

    /// Deliver until nothing is in flight.
    fn settle(&mut self) {
        while let Some((to, frame)) = self.queue.pop_front() {
            let outs = self.members[to].handle(&frame, self.now).unwrap();
            self.route(to, outs);
        }
    }

    /// Advance time in 100 ms steps, ticking everyone and settling.
    fn run_for(&mut self, d: Duration) {
        let end = self.now + d;
        while self.now < end {
            self.now = self.now + Duration::from_millis(100);
            for i in 0..self.members.len() {
                let outs = self.members[i].tick(self.now).unwrap();
                self.route(i, outs);
            }
            self.settle();
        }
    }

    fn act(
        &mut self,
        i: usize,
        f: impl FnOnce(&mut Paxos<MemoryVotes>, Millis) -> io::Result<Vec<Out>>,
    ) {
        let now = self.now;
        let outs = f(&mut self.members[i], now).unwrap();
        self.route(i, outs);
    }

    fn decided(&self, key: &[u8]) -> Vec<(usize, Vec<u8>)> {
        self.events
            .iter()
            .filter_map(|(i, e)| match e {
                Out::Decided { key: k, value } if k == key => Some((*i, value.clone())),
                _ => None,
            })
            .collect()
    }

    fn take_events(&mut self) -> Vec<(usize, Out)> {
        std::mem::take(&mut self.events)
    }
}

#[test]
fn a_lone_proposer_decides_its_value() {
    let mut net = Net::new(3, 2);
    net.act(0, |p, now| p.decide(b"gen/1", b"v1".to_vec(), now));
    net.settle();
    assert_eq!(net.decided(b"gen/1"), vec![(0, b"v1".to_vec())]);
}

/// Two proposers out-ballot each other, back off, and end with one decision.
#[test]
fn duelling_proposers_end_with_one_decision() {
    for seed in 0..20u64 {
        let mut net = Net::new(3, 2);
        for (i, m) in net.members.iter_mut().enumerate() {
            m.rng = StdRng::seed_from_u64(seed * 7 + i as u64);
        }
        net.act(0, |p, now| p.decide(b"k", b"a".to_vec(), now));
        net.act(1, |p, now| p.decide(b"k", b"b".to_vec(), now));
        net.settle();
        net.run_for(Duration::from_secs(20));
        let decided = net.decided(b"k");
        assert!(!decided.is_empty(), "seed {seed}: nothing decided");
        let first = &decided[0].1;
        assert!(
            decided.iter().all(|(_, v)| v == first),
            "seed {seed}: {decided:?}"
        );
    }
}

/// **Two members asking for a long lease together settle it in seconds.** The
/// backoff after a duel was drawn over the lease's whole term, so two members
/// asking for `upgrade/<app>` — a term of minutes — could both sit out for
/// minutes, and no member installed. The window is a decision's at most.
#[test]
fn a_duel_over_a_long_lease_settles_within_seconds() {
    let term = Duration::from_secs(10 * 60);
    for seed in 0..20u64 {
        let mut net = Net::new(3, 2);
        for (i, m) in net.members.iter_mut().enumerate() {
            m.rng = StdRng::seed_from_u64(seed * 7 + i as u64);
        }
        net.act(0, |p, now| p.acquire(b"upgrade/zend", term, now));
        net.act(1, |p, now| p.acquire(b"upgrade/zend", term, now));
        net.settle();
        net.run_for(Duration::from_secs(10));
        let held: Vec<usize> = net
            .events
            .iter()
            .filter_map(|(i, e)| matches!(e, Out::LeaseHeld { .. }).then_some(*i))
            .collect();
        assert!(!held.is_empty(), "seed {seed}: nobody holds the lease");
        assert!(
            held.iter().all(|i| *i == held[0]),
            "seed {seed}: two holders {held:?}"
        );
    }
}

/// A proposer proposes the highest-ballot accepted value it is shown.
#[test]
fn a_proposer_adopts_the_highest_accepted_value() {
    let mut net = Net::new(3, 2);
    // Member 1's proposal reaches only member 2 before member 1 is cut off.
    net.act(0, |p, now| p.decide(b"k", b"first".to_vec(), now));
    // Deliver the PREPAREs and their replies, then only the PROPOSE to member 2.
    let mut held = VecDeque::new();
    while let Some((to, frame)) = net.queue.pop_front() {
        if matches!(frame, Frame::Propose(_)) && to == 2 {
            held.push_back((to, frame));
            continue;
        }
        let outs = net.members[to].handle(&frame, net.now).unwrap();
        net.route(to, outs);
    }
    net.cut[0] = true;
    net.queue.clear();
    // Member 2 accepted "first"; member 3 never saw the proposal.
    assert!(net.members[1]
        .acceptor
        .vote(b"k")
        .unwrap()
        .accepted
        .is_some());
    net.take_events();
    net.act(2, |p, now| p.decide(b"k", b"second".to_vec(), now));
    net.settle();
    assert_eq!(net.decided(b"k"), vec![(2, b"first".to_vec())]);
    drop(held);
}

/// A reply delivered twice — through two clusters — is one vote.
#[test]
fn a_doubled_reply_counts_once() {
    let mut net = Net::new(5, 3);
    net.doubled = true;
    net.cut[3] = true;
    net.cut[4] = true;
    net.act(0, |p, now| p.decide(b"k", b"v".to_vec(), now));
    net.settle();
    // Members 1 and 2 answer, twice each: with member 1 itself that is three
    // distinct acceptors, a quorum.
    assert_eq!(net.decided(b"k").len(), 1);

    let mut net = Net::new(5, 3);
    net.doubled = true;
    net.cut[2] = true;
    net.cut[3] = true;
    net.cut[4] = true;
    net.act(0, |p, now| p.decide(b"k", b"v".to_vec(), now));
    net.settle();
    assert!(
        net.decided(b"k").is_empty(),
        "two acceptors, each heard twice, are not three"
    );
}

/// The status page sees which members promised the round in progress and,
/// when an acceptor refused it, the ballot that beat it.
#[test]
fn a_rounds_voters_and_the_ballot_that_beat_it_are_reported() {
    let mut net = Net::new(3, 3);
    net.cut[1] = true;
    net.cut[2] = true;
    net.act(0, |p, now| p.decide(b"k", b"v".to_vec(), now));
    net.settle();
    let s = net.members[0].status(b"k", net.now);
    assert_eq!(s.round.map(|(_, phase)| phase), Some("preparing"));
    assert_eq!(s.voters, [1], "its own acceptor's promise");
    assert_eq!(s.beaten_by, None);

    let higher = Ballot { round: 9, node: 2 };
    let nack = Frame::Nack(Nack {
        key: b"k".to_vec(),
        promised: higher,
        from: 2,
        reply_to: 1,
    });
    let now = net.now;
    net.members[0].handle(&nack, now).unwrap();
    let s = net.members[0].status(b"k", net.now);
    assert_eq!(s.round.map(|(_, phase)| phase), Some("waiting"));
    assert_eq!(s.beaten_by, Some(higher));
}

#[test]
fn a_member_alone_in_a_group_of_three_never_holds_a_lease() {
    let mut net = Net::new(3, 2);
    net.cut[1] = true;
    net.cut[2] = true;
    net.act(0, |p, now| {
        p.acquire(b"world/w", Duration::from_secs(10), now)
    });
    net.run_for(Duration::from_secs(60));
    assert!(!net
        .events
        .iter()
        .any(|(_, e)| matches!(e, Out::LeaseHeld { .. })));
    assert!(net.members[0].proposing(b"world/w"), "it keeps trying");
}

fn held(m: usize, until: u64) -> (usize, Out) {
    let key = b"w".to_vec();
    let until = Millis(until);
    (m, Out::LeaseHeld { key, until })
}

fn busy(m: usize, holder: u64) -> (usize, Out) {
    let key = b"w".to_vec();
    (m, Out::LeaseBusy { key, holder })
}

/// Another member's unexpired lease stops a proposer, which learns the
/// holder; the holder renews; once the term lapses another takes it.
#[test]
fn leases_are_held_renewed_refused_and_taken_over() {
    let ttl = Duration::from_secs(10);
    let mut net = Net::new(3, 2);
    net.act(0, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    assert_eq!(net.take_events(), vec![held(0, 8_000)]);

    net.act(1, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    assert_eq!(net.take_events(), vec![busy(1, 1)]);

    // Renewal at 5 s: the holder's own lease does not stop it.
    net.run_for(Duration::from_secs(5));
    net.act(0, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    assert_eq!(net.take_events(), vec![held(0, 13_000)]);

    // The holder's term ends 2 s before its acceptors': at 14 s it has let go,
    // and the acceptors still refuse another proposer.
    net.run_for(Duration::from_secs(9));
    assert!(!net.members[0].holds(b"w", net.now));
    let lost = (0, Out::LeaseLost { key: b"w".to_vec() });
    assert!(net.take_events().contains(&lost));
    net.act(1, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    assert_eq!(net.take_events(), vec![busy(1, 1)]);

    // At 15 s every acceptor's term has run out.
    net.run_for(Duration::from_secs(1));
    net.act(1, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    assert!(net.members[1].holds(b"w", net.now));
}

#[test]
fn a_released_lease_is_free_at_once() {
    let ttl = Duration::from_secs(30);
    let mut net = Net::new(3, 2);
    net.act(0, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    net.act(0, |p, now| p.release(b"w", 42, now));
    net.settle();
    assert!(!net.members[0].holds(b"w", net.now));
    net.take_events();
    net.act(1, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    assert!(net.members[1].holds(b"w", net.now));
}

/// A restarted acceptor keeps its votes, and counts an accepted lease for a
/// full term from the restart.
#[test]
fn a_restarted_member_keeps_its_votes() {
    let ttl = Duration::from_secs(10);
    let mut net = Net::new(3, 2);
    net.act(0, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    // Member 2 restarts 9 s later on a fresh clock reading.
    net.run_for(Duration::from_secs(9));
    let mut restarted = Paxos::open(Config::new(2), net.stores[1].clone(), net.now, 99).unwrap();
    restarted.set_quorum(2);
    net.members[1] = restarted;
    // Member 3 is cut off, so a quorum needs member 2, which still counts
    // member 1's lease for 10 s from its restart.
    net.cut[2] = true;
    net.run_for(Duration::from_secs(3));
    net.take_events();
    net.act(1, |p, now| p.acquire(b"w", ttl, now));
    net.settle();
    assert_eq!(net.take_events(), vec![busy(1, 1)]);
}

/// A learned decision answers every later round with the same value.
#[test]
fn a_learned_decision_is_what_every_later_round_decides() {
    let mut net = Net::new(3, 2);
    net.act(0, |p, now| p.decide(b"gen/7", b"x".to_vec(), now));
    net.settle();
    for m in &mut net.members {
        m.learn(b"gen/7", b"x").unwrap();
    }
    assert!(
        net.stores
            .iter()
            .all(|s| s.clone().load().unwrap().is_empty()),
        "slots freed"
    );
    net.take_events();
    net.act(2, |p, now| p.decide(b"gen/7", b"y".to_vec(), now));
    net.settle();
    assert_eq!(net.decided(b"gen/7"), vec![(2, b"x".to_vec())]);
}
