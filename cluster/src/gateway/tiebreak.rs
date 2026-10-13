//! The gateway as a tie-break acceptor (§5.9).
//!
//! A group of two needs both members for a majority. This acceptor is the
//! other vote: a `PREPARE` or `PROPOSE` that arrives on a member's socket is
//! answered on that same socket, and the reply carries the tie-break flag so
//! a proposer counts it only for a group of two.
//!
//! One gateway votes. With several, each voting would let a partition pair
//! each member with a different gateway and decide twice. The voter is the
//! lowest node id. A gateway that has peers does not vote until it has seen
//! every configured peer's node, and it keeps a peer it has seen after the
//! link drops, so a higher id never votes while a lower one exists. A gateway
//! with no peers is the voter at once.
//!
//! The vote is this gateway's own, in its journal, never replicated. Keys are
//! scoped by app, so zend and npcd do not share a slot. The gateway writes no
//! decision record; the members do.

use std::collections::BTreeSet;
use std::io;
use std::time::Instant;

use crate::consensus::votes::VoteStore;
use crate::consensus::{Config, Millis, Out, Paxos};
use crate::frame::{Accepted, Frame, Nack, Prepare, Promise, Propose, Release};

/// How a frame's key is stored: `app`, a zero byte, then the Paxos key.
fn scoped(app: &str, key: &[u8]) -> io::Result<Vec<u8>> {
    use crate::consensus::votes::MAX_KEY;
    let mut out = Vec::with_capacity(app.len() + 1 + key.len());
    out.extend_from_slice(app.as_bytes());
    out.push(0);
    out.extend_from_slice(key);
    if out.is_empty() || out.len() > MAX_KEY {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("a tie-break key is 1 to {MAX_KEY} bytes, not {}", out.len()),
        ));
    }
    Ok(out)
}

fn scope(app: &str, frame: &Frame) -> io::Result<Option<Frame>> {
    Ok(Some(match frame {
        Frame::Prepare(p) => Frame::Prepare(Prepare {
            key: scoped(app, &p.key)?,
            ballot: p.ballot,
        }),
        Frame::Propose(p) => Frame::Propose(Propose {
            key: scoped(app, &p.key)?,
            ballot: p.ballot,
            value: p.value.clone(),
        }),
        Frame::Release(r) => Frame::Release(Release {
            key: scoped(app, &r.key)?,
            ballot: r.ballot,
            through: r.through,
        }),
        _ => return Ok(None),
    }))
}

/// Put the proposer's key back and mark the reply as this gateway's vote.
fn unscope(original: &[u8], frame: Frame) -> Frame {
    match frame {
        Frame::Promise(p) => Frame::Promise(Promise {
            key: original.to_vec(),
            tiebreak: true,
            ..p
        }),
        Frame::Nack(n) => Frame::Nack(Nack {
            key: original.to_vec(),
            tiebreak: true,
            ..n
        }),
        Frame::Accepted(a) => Frame::Accepted(Accepted {
            key: original.to_vec(),
            tiebreak: true,
            ..a
        }),
        other => other,
    }
}

fn key_of(frame: &Frame) -> Option<&[u8]> {
    match frame {
        Frame::Prepare(p) => Some(&p.key),
        Frame::Propose(p) => Some(&p.key),
        Frame::Release(r) => Some(&r.key),
        _ => None,
    }
}

pub(super) struct TieBreak {
    node: u64,
    /// Configured peers. Zero: this gateway is the only one, and it votes.
    peers: usize,
    /// Peer node ids seen on a link. Kept after the link drops.
    seen: BTreeSet<u64>,
    started: Instant,
    paxos: Paxos<Box<dyn VoteStore>>,
}

impl TieBreak {
    pub(super) fn open(node: u64, peers: usize, store: Box<dyn VoteStore>) -> io::Result<Self> {
        Ok(Self {
            node,
            peers,
            seen: BTreeSet::new(),
            started: Instant::now(),
            paxos: Paxos::open(Config::new(node), store, Millis(0), node)?,
        })
    }

    pub(super) fn note_peer(&mut self, node: u64) {
        if node != 0 && node != self.node {
            self.seen.insert(node);
        }
    }

    /// This gateway is the tie-break voter.
    pub(super) fn voting(&self) -> bool {
        if self.peers == 0 {
            return true;
        }
        if self.seen.len() < self.peers {
            return false;
        }
        self.seen.iter().all(|id| self.node < *id)
    }

    fn now(&self) -> Millis {
        Millis(self.started.elapsed().as_millis() as u64)
    }

    /// Answer `frame` from `app`'s member, on the socket it arrived on.
    /// `None` when this gateway is not the voter, or the frame is not a vote.
    pub(super) fn consider(&mut self, app: &str, frame: &Frame) -> io::Result<Option<Frame>> {
        if !self.voting() {
            return Ok(None);
        }
        let Some(original) = key_of(frame) else {
            return Ok(None);
        };
        let original = original.to_vec();
        let Some(scoped) = scope(app, frame)? else {
            return Ok(None);
        };
        let now = self.now();
        let outs = self.paxos.handle(&scoped, now)?;
        for out in outs {
            if let Out::Reply { frame, .. } = out {
                return Ok(Some(unscope(&original, frame)));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::votes::MemoryVotes;
    use crate::frame::Ballot;

    fn prepare(node: u64) -> Frame {
        Frame::Prepare(Prepare {
            key: b"gen/1".to_vec(),
            ballot: Ballot { round: 1, node },
        })
    }

    /// A sole gateway promises on the socket's round, flagged as the tie-break,
    /// and a second app's same key is a different slot.
    #[test]
    fn a_sole_gateway_promises_on_its_voting_socket() {
        let votes = MemoryVotes::new();
        let mut t = TieBreak::open(7, 0, Box::new(votes.clone())).unwrap();
        assert!(t.voting());
        let reply = t.consider("zend", &prepare(4)).unwrap().unwrap();
        match reply {
            Frame::Promise(p) => {
                assert!(p.tiebreak);
                assert_eq!(p.from, 7);
                assert_eq!(p.reply_to, 4);
                assert_eq!(p.key, b"gen/1");
                assert!(p.accepted.is_none());
            }
            other => panic!("a prepare is promised: {other:?}"),
        }
        // The promise is durable: a new engine on the same journal still holds it.
        let mut again = TieBreak::open(7, 0, Box::new(votes)).unwrap();
        let reply = again.consider("zend", &prepare(4)).unwrap().unwrap();
        match reply {
            Frame::Nack(n) => {
                assert!(n.tiebreak);
                assert_eq!(n.promised, Ballot { round: 1, node: 4 });
            }
            other => panic!("the promise survived: {other:?}"),
        }
        // npcd's gen/1 has not been promised.
        let reply = again.consider("npcd", &prepare(4)).unwrap().unwrap();
        assert!(matches!(reply, Frame::Promise(_)), "{reply:?}");
    }

    /// The lowest node votes, and only once every configured peer has been seen.
    #[test]
    fn the_lowest_gateway_is_the_voter() {
        let mut low = TieBreak::open(2, 1, Box::new(MemoryVotes::new())).unwrap();
        assert!(!low.voting(), "a peer not yet seen");
        low.note_peer(9);
        assert!(low.voting());
        low.note_peer(9);
        assert!(low.voting(), "a peer seen again changes nothing");

        let mut high = TieBreak::open(9, 1, Box::new(MemoryVotes::new())).unwrap();
        high.note_peer(2);
        assert!(!high.voting());
        assert!(high.consider("zend", &prepare(4)).unwrap().is_none());

        let mut waiting = TieBreak::open(1, 2, Box::new(MemoryVotes::new())).unwrap();
        waiting.note_peer(3);
        assert!(!waiting.voting(), "one peer of two");
        waiting.note_peer(4);
        assert!(waiting.voting());
    }
}
