//! A member's Paxos engine (§5.9): its acceptor, its proposals, and the leases
//! it holds, driven by the frames it receives and readings of its monotonic
//! clock. It does no I/O but its vote store's: what it sends comes back as
//! [`Out`], for the owner to put on its sockets.
//!
//! **One round**, for one key: `PREPARE` to every member; with promises from a
//! quorum of distinct acceptors, `PROPOSE` the highest-ballot accepted value
//! among them, or its own; with `ACCEPTED` from a quorum, it is decided. The
//! member is an acceptor of its own rounds too. In a group of two, one
//! gateway promise flagged as the tie-break fills the empty seat (§5.9).
//!
//! **Leases add a term.** A promise carrying another member's unexpired lease
//! stops the proposer (`LeaseBusy`); one carrying its own does not, and that is
//! renewal. The holder counts its term from when it sent `PREPARE` and stops
//! 2 s before the end, so it stops before any acceptor would let another
//! proposer in.
//!
//! **Duelling proposers back off.** A `NACK`, or a phase that does not finish
//! within its timeout, waits a random time drawn uniformly from zero to the
//! key's `ttl`, capped at a decision's 2 s, then prepares again above every
//! ballot seen. One proposer then completes a round while the other waits, and the
//! other learns the result from its next promises.

use std::io;
use std::time::Duration;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::acceptor::{Acceptor, PrepareReply, ProposeReply};
use super::clock::Millis;
use super::dedup::Dedup;
use super::maps::{Map, Set};
use super::value::Value;
use super::votes::VoteStore;
use crate::frame::{
    Accepted, AcceptedValue, Ballot, Frame, Kind, Nack, Prepare, Promise, Propose, Release,
};

/// How much earlier than its acceptors a holder stops counting its lease.
pub const HOLDER_MARGIN: Duration = Duration::from_secs(2);

/// How many handled frames the duplicate filter remembers.
const DEDUP_CAP: usize = 1 << 16;

/// How often idle lease slots are looked for.
const FREE_EVERY: Duration = Duration::from_secs(1);

#[derive(Debug, Clone)]
pub struct Config {
    pub node: u64,
    /// A decision's backoff window.
    pub decision_ttl: Duration,
    /// How long one phase may wait for its quorum before backing off.
    pub round_timeout: Duration,
}

impl Config {
    pub fn new(node: u64) -> Self {
        Self {
            node,
            decision_ttl: Duration::from_secs(2),
            round_timeout: Duration::from_secs(2),
        }
    }
}

/// What the engine asks its owner to do, or tells it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Out {
    /// To every other member of the app, through every cluster.
    Broadcast(Frame),
    /// To one member, through the gateway's `reply_to` routing.
    Reply { to: u64, frame: Frame },
    /// `key` is decided: write it as a replicated record.
    Decided { key: Vec<u8>, value: Vec<u8> },
    /// This member holds `key`'s lease until `until` on its clock.
    LeaseHeld { key: Vec<u8>, until: Millis },
    /// Another member holds `key`'s lease: route to it.
    LeaseBusy { key: Vec<u8>, holder: u64 },
    /// This member's lease on `key` ran out unrenewed.
    LeaseLost { key: Vec<u8> },
}

/// One key as a member sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyStatus {
    /// A round in progress: its ballot and phase.
    pub round: Option<(Ballot, &'static str)>,
    /// The members that promised, or accepted, the round in progress.
    pub voters: Vec<u64>,
    /// The higher ballot that beat it, when a `NACK` did.
    pub beaten_by: Option<Ballot>,
    /// The lease this member holds, while it holds it.
    pub held_until: Option<Millis>,
    /// What its acceptor has promised and accepted.
    pub promised: Option<Ballot>,
    pub accepted: Option<Ballot>,
}

#[derive(Debug, Clone)]
enum Goal {
    Decide(Vec<u8>),
    Lease(Duration),
}

#[derive(Debug)]
enum Phase {
    Preparing {
        /// Promises from members.
        promises: Map<u64, Option<AcceptedValue>>,
        /// Promises from gateways voting as the tie-break (§5.9). At most one
        /// of them counts, and only in a group of two.
        tiebreak: Map<u64, Option<AcceptedValue>>,
    },
    Proposing {
        value: Vec<u8>,
        accepted: Set<u64>,
        tiebreak: Set<u64>,
    },
    Waiting {
        until: Millis,
    },
}

/// How many votes count toward `quorum`. A gateway's vote fills the one empty
/// seat of a group of two, and no other group: a quorum of two is also three
/// members' majority, and one member plus a gateway must not decide there
/// while the other two decide on their own.
fn seats(quorum: usize, tiebreak_counts: bool, members: usize, tiebreaks: usize) -> usize {
    let extra = usize::from(tiebreak_counts && quorum == 2 && members >= 1 && tiebreaks >= 1);
    members + extra
}

#[derive(Debug)]
struct Round {
    goal: Goal,
    ballot: Ballot,
    phase: Phase,
    /// When this round's `PREPARE` was sent: a lease's term counts from here.
    started: Millis,
    deadline: Millis,
    /// The higher ballot an acceptor's `NACK` named, when one beat this round.
    beaten_by: Option<Ballot>,
}

pub struct Paxos<S: VoteStore> {
    cfg: Config,
    acceptor: Acceptor<S>,
    quorum: usize,
    /// Count a gateway tie-break vote. Set only while the group size is two.
    tiebreak: bool,
    rounds: Map<Vec<u8>, Round>,
    /// The highest ballot round seen per key, so the next ballot goes above.
    highest: Map<Vec<u8>, u64>,
    held: Map<Vec<u8>, (Ballot, Millis)>,
    /// The earliest end among held leases: a tick looks for lapsed ones only
    /// once it has come.
    next_lapse: Option<Millis>,
    dedup: Dedup,
    rng: StdRng,
    freed_at: Millis,
}

impl<S: VoteStore> Paxos<S> {
    /// Resume from `store` at `now`; `seed` drives the backoff draws.
    pub fn open(cfg: Config, store: S, now: Millis, seed: u64) -> io::Result<Self> {
        Ok(Self {
            acceptor: Acceptor::open(store, now)?,
            cfg,
            quorum: 1,
            tiebreak: false,
            rounds: Map::default(),
            highest: Map::default(),
            held: Map::default(),
            next_lapse: None,
            dedup: Dedup::new(DEDUP_CAP),
            rng: StdRng::seed_from_u64(seed),
            freed_at: now,
        })
    }

    pub fn node(&self) -> u64 {
        self.cfg.node
    }

    /// The quorum, from `max_members` (§5.9).
    pub fn set_quorum(&mut self, quorum: usize) {
        self.quorum = quorum.max(1);
    }

    /// Whether a gateway's tie-break vote counts. A caller sets this from the
    /// group size: only a group of two, whose quorum is two, may be completed
    /// by the gateway (§5.9).
    pub fn allow_tiebreak(&mut self, allow: bool) {
        self.tiebreak = allow;
    }

    /// Decide one value for `key`: `value` if no other was accepted first.
    pub fn decide(&mut self, key: &[u8], value: Vec<u8>, now: Millis) -> io::Result<Vec<Out>> {
        self.start(key, Goal::Decide(value), now)
    }

    /// Take `key`'s lease for `ttl`, or renew it.
    pub fn acquire(&mut self, key: &[u8], ttl: Duration, now: Millis) -> io::Result<Vec<Out>> {
        self.start(key, Goal::Lease(ttl), now)
    }

    /// Hand `key`'s lease back; `through` is this member's own `oseq`.
    pub fn release(&mut self, key: &[u8], through: u64, now: Millis) -> io::Result<Vec<Out>> {
        self.rounds.remove(key);
        let Some((ballot, _)) = self.held.remove(key) else {
            return Ok(Vec::new());
        };
        self.acceptor.release(key, ballot, now)?;
        Ok(vec![Out::Broadcast(Frame::Release(Release {
            key: key.to_vec(),
            ballot,
            through,
        }))])
    }

    /// `key`'s decision is recorded in the substrate.
    pub fn learn(&mut self, key: &[u8], value: &[u8]) -> io::Result<()> {
        self.rounds.remove(key);
        self.acceptor
            .learn(key, Value::Decision(value.to_vec()).encode())
    }

    /// Whether this member holds `key`'s lease at `now`.
    pub fn holds(&self, key: &[u8], now: Millis) -> bool {
        self.held.get(key).is_some_and(|(_, until)| now < *until)
    }

    fn goal_ttl(&self, goal: &Goal) -> Duration {
        match goal {
            Goal::Decide(_) => self.cfg.decision_ttl,
            Goal::Lease(ttl) => *ttl,
        }
    }

    fn goal_value(&self, goal: &Goal) -> Vec<u8> {
        match goal {
            Goal::Decide(bytes) => Value::Decision(bytes.clone()).encode(),
            Goal::Lease(ttl) => Value::Lease {
                holder: self.cfg.node,
                ttl: *ttl,
            }
            .encode(),
        }
    }

    fn start(&mut self, key: &[u8], goal: Goal, now: Millis) -> io::Result<Vec<Out>> {
        let mut out = Vec::new();
        self.prepare(key, goal, now, &mut out)?;
        Ok(out)
    }

    fn prepare(
        &mut self,
        key: &[u8],
        goal: Goal,
        now: Millis,
        out: &mut Vec<Out>,
    ) -> io::Result<()> {
        let round = self.highest.get(key).copied().unwrap_or(0) + 1;
        self.highest.insert(key.to_vec(), round);
        let ballot = Ballot {
            round,
            node: self.cfg.node,
        };
        self.rounds.insert(
            key.to_vec(),
            Round {
                goal,
                ballot,
                phase: Phase::Preparing {
                    promises: Map::default(),
                    tiebreak: Map::default(),
                },
                started: now,
                deadline: now + self.cfg.round_timeout,
                beaten_by: None,
            },
        );
        out.push(Out::Broadcast(Frame::Prepare(Prepare {
            key: key.to_vec(),
            ballot,
        })));
        match self.acceptor.prepare(key, ballot, now)? {
            PrepareReply::Promise(accepted) => {
                self.on_promise(key, self.cfg.node, ballot, accepted, false, now, out)?
            }
            PrepareReply::Nack(promised) => self.on_nack(key, promised, now),
        }
        Ok(())
    }

    fn backoff(&mut self, key: &[u8], now: Millis) {
        let Some(round) = self.rounds.get(key) else {
            return;
        };
        // At most a decision's window, however long the lease: a duel is over
        // in a round or two, and the next prepare learns of any winner. Drawn
        // over an upgrade lease's term — minutes — two members asking for
        // `upgrade/<app>` together could both sit out for minutes.
        let window = self.goal_ttl(&round.goal).min(self.cfg.decision_ttl);
        let wait = Duration::from_millis(self.rng.random_range(0..=window.as_millis() as u64));
        let round = self.rounds.get_mut(key).expect("checked");
        round.phase = Phase::Waiting { until: now + wait };
    }

    /// A ballot of `round` was seen for `key`: the next one goes above it.
    fn saw(&mut self, key: &[u8], round: u64) {
        let round = round.min(u64::MAX - 1);
        match self.highest.get_mut(key) {
            Some(seen) => *seen = (*seen).max(round),
            None => {
                self.highest.insert(key.to_vec(), round);
            }
        }
    }

    fn on_nack(&mut self, key: &[u8], promised: Ballot, now: Millis) {
        self.saw(key, promised.round);
        // Only a promise above the round's ballot beats it. A NACK naming the
        // round's own ballot answers one of this member's earlier rounds,
        // delayed, from an acceptor that has since promised this one.
        let beaten = self
            .rounds
            .get(key)
            .is_some_and(|r| !matches!(r.phase, Phase::Waiting { .. }) && promised > r.ballot);
        if beaten {
            if let Some(r) = self.rounds.get_mut(key) {
                r.beaten_by = Some(promised);
            }
            self.backoff(key, now);
        }
    }

    fn on_promise(
        &mut self,
        key: &[u8],
        from: u64,
        ballot: Ballot,
        accepted: Option<AcceptedValue>,
        tiebreak: bool,
        now: Millis,
        out: &mut Vec<Out>,
    ) -> io::Result<()> {
        let quorum = self.quorum;
        let tiebreak_counts = self.tiebreak;
        let node = self.cfg.node;
        let Some(round) = self.rounds.get_mut(key) else {
            return Ok(());
        };
        if round.ballot != ballot {
            return Ok(());
        }
        let Phase::Preparing {
            promises,
            tiebreak: ties,
        } = &mut round.phase
        else {
            return Ok(());
        };
        if tiebreak {
            if !(tiebreak_counts && quorum == 2) {
                return Ok(());
            }
            ties.insert(from, accepted);
        } else {
            promises.insert(from, accepted);
        }
        if seats(quorum, tiebreak_counts, promises.len(), ties.len()) < quorum {
            return Ok(());
        }
        let mut promises = std::mem::take(promises);
        // One tie-break seat. The lowest gateway node is the voter (§5.9); if
        // another also answered, its accepted value still competes, so a
        // higher ballot already accepted is not dropped.
        for (from, accepted) in std::mem::take(ties) {
            promises.insert(from, accepted);
        }

        // Another member's unexpired lease stops a lease round.
        if let Goal::Lease(_) = round.goal {
            for acc in promises.values().flatten() {
                if let Ok(Value::Lease { holder, .. }) = Value::decode(&acc.value) {
                    if holder != node {
                        self.rounds.remove(key);
                        out.push(Out::LeaseBusy {
                            key: key.to_vec(),
                            holder,
                        });
                        return Ok(());
                    }
                }
            }
        }
        let highest = promises
            .values()
            .flatten()
            .max_by_key(|a| a.ballot)
            .cloned();
        let goal = round.goal.clone();
        let value = match (&goal, highest) {
            // Renewing: the reported lease is this member's own.
            (Goal::Lease(_), Some(acc))
                if matches!(Value::decode(&acc.value), Ok(Value::Lease { .. })) =>
            {
                self.goal_value(&goal)
            }
            (_, Some(acc)) => acc.value,
            (_, None) => self.goal_value(&goal),
        };
        let round = self.rounds.get_mut(key).expect("held above");
        round.phase = Phase::Proposing {
            value: value.clone(),
            accepted: Set::default(),
            tiebreak: Set::default(),
        };
        round.deadline = now + self.cfg.round_timeout;
        out.push(Out::Broadcast(Frame::Propose(Propose {
            key: key.to_vec(),
            ballot,
            value: value.clone(),
        })));
        match self.acceptor.propose(key, ballot, &value, now)? {
            ProposeReply::Accepted => self.on_accepted(key, node, ballot, false, out),
            ProposeReply::Nack(promised) => self.on_nack(key, promised, now),
        }
        Ok(())
    }

    fn on_accepted(
        &mut self,
        key: &[u8],
        from: u64,
        ballot: Ballot,
        tiebreak: bool,
        out: &mut Vec<Out>,
    ) {
        let quorum = self.quorum;
        let tiebreak_counts = self.tiebreak;
        let Some(round) = self.rounds.get_mut(key) else {
            return;
        };
        if round.ballot != ballot {
            return;
        }
        let Phase::Proposing {
            value,
            accepted,
            tiebreak: ties,
        } = &mut round.phase
        else {
            return;
        };
        if tiebreak {
            if !(tiebreak_counts && quorum == 2) {
                return;
            }
            ties.insert(from);
        } else {
            accepted.insert(from);
        }
        if seats(quorum, tiebreak_counts, accepted.len(), ties.len()) < quorum {
            return;
        }
        let value = std::mem::take(value);
        let started = round.started;
        self.rounds.remove(key);
        match Value::decode(&value) {
            Ok(Value::Decision(bytes)) => out.push(Out::Decided {
                key: key.to_vec(),
                value: bytes,
            }),
            Ok(Value::Lease { holder, ttl }) if holder == self.cfg.node => {
                let until = (started + ttl) - HOLDER_MARGIN;
                self.held.insert(key.to_vec(), (ballot, until));
                self.next_lapse = Some(self.next_lapse.map_or(until, |n| n.min(until)));
                out.push(Out::LeaseHeld {
                    key: key.to_vec(),
                    until,
                });
            }
            Ok(Value::Lease { holder, .. }) => out.push(Out::LeaseBusy {
                key: key.to_vec(),
                holder,
            }),
            Err(_) => {}
        }
    }

    /// A frame from another member.
    pub fn handle(&mut self, frame: &Frame, now: Millis) -> io::Result<Vec<Out>> {
        let node = self.cfg.node;
        let mut out = Vec::new();
        match frame {
            Frame::Prepare(p) => {
                if p.ballot.node == node
                    || !self
                        .dedup
                        .first(Kind::Prepare, &p.key, p.ballot, p.ballot.node)
                {
                    return Ok(out);
                }
                self.saw(&p.key, p.ballot.round);
                let reply = match self.acceptor.prepare(&p.key, p.ballot, now)? {
                    PrepareReply::Promise(accepted) => Frame::Promise(Promise {
                        key: p.key.clone(),
                        ballot: p.ballot,
                        accepted,
                        from: node,
                        reply_to: p.ballot.node,
                        tiebreak: false,
                    }),
                    PrepareReply::Nack(promised) => Frame::Nack(Nack {
                        key: p.key.clone(),
                        promised,
                        from: node,
                        reply_to: p.ballot.node,
                        tiebreak: false,
                    }),
                };
                out.push(Out::Reply {
                    to: p.ballot.node,
                    frame: reply,
                });
            }
            Frame::Propose(p) => {
                if p.ballot.node == node
                    || !self
                        .dedup
                        .first(Kind::Propose, &p.key, p.ballot, p.ballot.node)
                {
                    return Ok(out);
                }
                self.saw(&p.key, p.ballot.round);
                let reply = match self.acceptor.propose(&p.key, p.ballot, &p.value, now)? {
                    ProposeReply::Accepted => Frame::Accepted(Accepted {
                        key: p.key.clone(),
                        ballot: p.ballot,
                        from: node,
                        reply_to: p.ballot.node,
                        tiebreak: false,
                    }),
                    ProposeReply::Nack(promised) => Frame::Nack(Nack {
                        key: p.key.clone(),
                        promised,
                        from: node,
                        reply_to: p.ballot.node,
                        tiebreak: false,
                    }),
                };
                out.push(Out::Reply {
                    to: p.ballot.node,
                    frame: reply,
                });
            }
            Frame::Release(r)
                if r.ballot.node != node
                    && self
                        .dedup
                        .first(Kind::Release, &r.key, r.ballot, r.ballot.node) =>
            {
                self.acceptor.release(&r.key, r.ballot, now)?;
            }
            Frame::Promise(p)
                if p.reply_to == node
                    && self.dedup.first(Kind::Promise, &p.key, p.ballot, p.from) =>
            {
                let accepted = p.accepted.clone();
                self.on_promise(
                    &p.key, p.from, p.ballot, accepted, p.tiebreak, now, &mut out,
                )?;
            }
            Frame::Nack(n)
                if n.reply_to == node
                    && self.dedup.first(Kind::Nack, &n.key, n.promised, n.from) =>
            {
                if !(n.tiebreak && !(self.tiebreak && self.quorum == 2)) {
                    self.on_nack(&n.key, n.promised, now);
                }
            }
            Frame::Accepted(a)
                if a.reply_to == node
                    && self.dedup.first(Kind::Accepted, &a.key, a.ballot, a.from) =>
            {
                self.on_accepted(&a.key, a.from, a.ballot, a.tiebreak, &mut out);
            }
            _ => {}
        }
        Ok(out)
    }

    /// The clock moved: retry rounds whose backoff ended, back off from rounds
    /// that timed out, and let lapsed leases go.
    pub fn tick(&mut self, now: Millis) -> io::Result<Vec<Out>> {
        let mut out = Vec::new();
        let mut restart = Vec::new();
        let mut timed_out = Vec::new();
        for (key, round) in &self.rounds {
            match round.phase {
                Phase::Waiting { until } if now >= until => {
                    restart.push((key.clone(), round.goal.clone()))
                }
                Phase::Preparing { .. } | Phase::Proposing { .. } if now >= round.deadline => {
                    timed_out.push(key.clone())
                }
                _ => {}
            }
        }
        for key in timed_out {
            self.backoff(&key, now);
        }
        for (key, goal) in restart {
            self.prepare(&key, goal, now, &mut out)?;
        }
        if self.next_lapse.is_some_and(|n| now >= n) {
            let lapsed: Vec<Vec<u8>> = self
                .held
                .iter()
                .filter(|(_, (_, until))| now >= *until)
                .map(|(k, _)| k.clone())
                .collect();
            for key in lapsed {
                self.held.remove(&key);
                out.push(Out::LeaseLost { key });
            }
            self.next_lapse = self.held.values().map(|(_, until)| *until).min();
        }
        if now >= self.freed_at + FREE_EVERY {
            self.freed_at = now;
            self.acceptor.free_idle_leases(now)?;
        }
        Ok(out)
    }

    /// Whether a round for `key` is in progress.
    pub fn proposing(&self, key: &[u8]) -> bool {
        self.rounds.contains_key(key)
    }

    /// What this member knows of `key`, for the status page (§12.1).
    pub fn status(&self, key: &[u8], now: Millis) -> KeyStatus {
        let round = self.rounds.get(key);
        KeyStatus {
            round: round.map(|r| {
                let phase = match r.phase {
                    Phase::Preparing { .. } => "preparing",
                    Phase::Proposing { .. } => "proposing",
                    Phase::Waiting { .. } => "waiting",
                };
                (r.ballot, phase)
            }),
            voters: match round.map(|r| &r.phase) {
                Some(Phase::Preparing { promises, tiebreak }) => {
                    let mut voters: Vec<u64> = promises.keys().copied().collect();
                    voters.extend(tiebreak.keys().copied());
                    voters.sort_unstable();
                    voters
                }
                Some(Phase::Proposing {
                    accepted, tiebreak, ..
                }) => {
                    let mut voters: Vec<u64> = accepted.iter().copied().collect();
                    voters.extend(tiebreak.iter().copied());
                    voters.sort_unstable();
                    voters
                }
                _ => Vec::new(),
            },
            beaten_by: round.and_then(|r| r.beaten_by),
            held_until: self
                .held
                .get(key)
                .map(|(_, until)| *until)
                .filter(|until| now < *until),
            promised: self.acceptor.vote(key).map(|v| v.promised),
            accepted: self
                .acceptor
                .vote(key)
                .and_then(|v| v.accepted.as_ref().map(|a| a.ballot)),
        }
    }

    /// Every key this member knows of — a round in progress, a lease it holds,
    /// a vote its acceptor keeps — in order, for the status page (§12.1).
    pub fn keys(&self) -> Vec<Vec<u8>> {
        let mut keys: Vec<Vec<u8>> = self
            .rounds
            .keys()
            .chain(self.held.keys())
            .chain(self.acceptor.keys())
            .cloned()
            .collect();
        keys.sort();
        keys.dedup();
        keys
    }

    /// The leases this member holds at `now`.
    pub fn held(&self, now: Millis) -> Vec<Vec<u8>> {
        self.held
            .iter()
            .filter(|(_, (_, until))| now < *until)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Stop proposing for `key`: rendezvous names another member now (§9.5),
    /// and a proposer that kept asking would only duel with it. A lease this
    /// member holds is unaffected; it runs out or is released.
    pub fn abandon(&mut self, key: &[u8]) {
        self.rounds.remove(key);
    }
}

#[cfg(test)]
mod tests;
