//! Paxos frames handled once (§5.9): a frame reaches a member through every
//! cluster it has joined, so each member drops one it has already handled,
//! keyed `(kind, key, ballot, from)`. The memory is bounded: the oldest are
//! forgotten first, which only costs handling a very late duplicate again —
//! harmless, since every Paxos step is idempotent per ballot.
//!
//! An entry is kept as a 64-bit fingerprint of its key. Two different frames
//! sharing one — about one chance in 2^48 among the entries held — would drop
//! the second as a duplicate, which Paxos treats as any lost frame.

use std::collections::VecDeque;
use std::hash::{BuildHasher, Hash, Hasher};

use super::maps::{Fixed, Set};
use crate::frame::{Ballot, Kind};

pub struct Dedup {
    seen: Set<u64>,
    order: VecDeque<u64>,
    cap: usize,
}

fn fingerprint(kind: Kind, key: &[u8], ballot: Ballot, from: u64) -> u64 {
    let mut h = Fixed.build_hasher();
    kind.hash(&mut h);
    key.hash(&mut h);
    ballot.hash(&mut h);
    from.hash(&mut h);
    h.finish()
}

impl Dedup {
    pub fn new(cap: usize) -> Self {
        Self {
            seen: Set::with_capacity_and_hasher(cap, Default::default()),
            order: VecDeque::with_capacity(cap),
            cap,
        }
    }

    /// Whether this frame is new; it is remembered either way.
    pub fn first(&mut self, kind: Kind, key: &[u8], ballot: Ballot, from: u64) -> bool {
        let entry = fingerprint(kind, key, ballot, from);
        if !self.seen.insert(entry) {
            return false;
        }
        self.order.push_back(entry);
        if self.order.len() > self.cap {
            let old = self.order.pop_front().expect("over capacity");
            self.seen.remove(&old);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_through_two_clusters_is_handled_once() {
        let mut d = Dedup::new(2);
        let b = Ballot { round: 1, node: 2 };
        assert!(d.first(Kind::Promise, b"k", b, 3));
        assert!(!d.first(Kind::Promise, b"k", b, 3));
        assert!(d.first(Kind::Promise, b"k", b, 4), "another acceptor");
        assert!(d.first(Kind::Accepted, b"k", b, 3), "another kind");
        assert!(
            d.first(Kind::Promise, b"k", b, 3),
            "forgotten past capacity"
        );
    }
}
