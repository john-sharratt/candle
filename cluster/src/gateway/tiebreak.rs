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
//!
//! The gateway restarts on every web deploy. A lease it accepted counts after
//! a restart for what is left of the end its vote recorded, never a fresh
//! term (`consensus::rearm`): each deploy once re-armed a dead `upgrade/zend`
//! for a full 20 minutes, and members asking for it were refused for as long
//! as deploys kept coming (2026-10-31). The leases it counts are on its
//! status page, by app.

use std::collections::BTreeSet;
use std::io;
use std::time::Instant;

use super::status::TiebreakLease;
use crate::consensus::votes::VoteStore;
use crate::consensus::{Config, Millis, Out, Paxos, WallMillis};
use crate::frame::{Accepted, Frame, Nack, Prepare, Promise, Propose, Release};

/// How a frame's key is voted on: `app`, a zero byte, then the Paxos key.
/// Its length is the acceptor's to fit (`consensus::votes::held_key`): a key
/// the scope took past a journal slot was refused here, and no round for it
/// was ever decided.
fn scoped(app: &str, key: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(app.len() + 1 + key.len());
    out.extend_from_slice(app.as_bytes());
    out.push(0);
    out.extend_from_slice(key);
    out
}

fn scope(app: &str, frame: &Frame) -> Option<Frame> {
    Some(match frame {
        Frame::Prepare(p) => Frame::Prepare(Prepare {
            key: scoped(app, &p.key),
            ballot: p.ballot,
        }),
        Frame::Propose(p) => Frame::Propose(Propose {
            key: scoped(app, &p.key),
            ballot: p.ballot,
            value: p.value.clone(),
        }),
        Frame::Release(r) => Frame::Release(Release {
            key: scoped(app, &r.key),
            ballot: r.ballot,
            through: r.through,
        }),
        _ => return None,
    })
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

/// A held key as its app and the member's key: [`scoped`] undone. A key held
/// under its digest (`consensus::votes::held_key`) has no app to show.
fn unscoped(held: &[u8]) -> (String, String) {
    match held.iter().position(|b| *b == 0) {
        Some(at) => (
            String::from_utf8_lossy(&held[..at]).into_owned(),
            String::from_utf8_lossy(&held[at + 1..]).into_owned(),
        ),
        None => (String::new(), String::from_utf8_lossy(held).into_owned()),
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
    /// Resume from `store`, opened with the wall clock reading `wall`.
    pub(super) fn open(
        node: u64,
        peers: usize,
        store: Box<dyn VoteStore>,
        wall: WallMillis,
    ) -> io::Result<Self> {
        Ok(Self {
            node,
            peers,
            seen: BTreeSet::new(),
            started: Instant::now(),
            paxos: Paxos::open(Config::new(node), store, Millis(0), wall, node)?,
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

    /// This acceptor's monotonic clock: milliseconds since it opened.
    pub(super) fn now(&self) -> Millis {
        Millis(self.started.elapsed().as_millis() as u64)
    }

    /// Answer `frame` from `app`'s member, on the socket it arrived on, at
    /// `now` on [`Self::now`]'s clock and `wall` on the wall clock. `None`
    /// when this gateway is not the voter, or the frame is not a vote. Lease
    /// slots idle one `ttl` past their end are freed as it goes.
    pub(super) fn consider(
        &mut self,
        app: &str,
        frame: &Frame,
        now: Millis,
        wall: WallMillis,
    ) -> io::Result<Option<Frame>> {
        if !self.voting() {
            return Ok(None);
        }
        let Some(original) = key_of(frame) else {
            return Ok(None);
        };
        let original = original.to_vec();
        let Some(scoped) = scope(app, frame) else {
            return Ok(None);
        };
        let outs = self.paxos.handle(&scoped, now, wall)?;
        // The gateway proposes nothing, so a tick only frees idle slots.
        self.paxos.tick(now, wall)?;
        for out in outs {
            if let Out::Reply { frame, .. } = out {
                return Ok(Some(unscope(&original, frame)));
            }
        }
        Ok(None)
    }

    /// Every lease this acceptor counts as armed at `now`, by app and key, for
    /// the status page.
    pub(super) fn armed_leases(&self, now: Millis) -> Vec<TiebreakLease> {
        self.paxos
            .armed_leases(now)
            .iter()
            .map(|lease| {
                let (app, key) = unscoped(&lease.key);
                TiebreakLease::of(app, key, lease)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::consensus::votes::{FileJournal, MemoryVotes, MAX_KEY};
    use crate::consensus::Value;
    use crate::frame::Ballot;

    /// The wall clock when a test's gateway first opens: 2026-10-26.
    const T0: u64 = 1_793_000_000_000;
    const MIN: u64 = 60_000;

    fn opened(node: u64, peers: usize, store: Box<dyn VoteStore>) -> TieBreak {
        TieBreak::open(node, peers, store, WallMillis(T0)).unwrap()
    }

    /// `frame` from `app`'s member, at the start of both the gateway's clocks.
    fn ask(t: &mut TieBreak, app: &str, frame: &Frame) -> Option<Frame> {
        t.consider(app, frame, Millis(0), WallMillis(T0)).unwrap()
    }

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
        let mut t = opened(7, 0, Box::new(votes.clone()));
        assert!(t.voting());
        let reply = ask(&mut t, "zend", &prepare(4)).unwrap();
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
        let mut again = opened(7, 0, Box::new(votes));
        let reply = ask(&mut again, "zend", &prepare(4)).unwrap();
        match reply {
            Frame::Nack(n) => {
                assert!(n.tiebreak);
                assert_eq!(n.promised, Ballot { round: 1, node: 4 });
            }
            other => panic!("the promise survived: {other:?}"),
        }
        // npcd's gen/1 has not been promised.
        let reply = ask(&mut again, "npcd", &prepare(4)).unwrap();
        assert!(matches!(reply, Frame::Promise(_)), "{reply:?}");
    }

    /// **A key the scope takes past a journal slot is still voted on**: the
    /// measured 128-byte ingest key, 133 scoped, is promised on a file journal
    /// and the promise survives a restart, where it was refused outright and
    /// its round never decided (BlackWorld, 2026-10-27).
    #[test]
    fn a_long_key_is_promised_and_held_across_a_restart() {
        let key = b"ingest/v3:candle/docs/results/flash_next_single_session_rtx_pro_5000_72gb_2026-10-06.md@c209644b98224bfce329a662dd3ef740e03f41d6".to_vec();
        assert_eq!(key.len(), MAX_KEY);
        let prepare = Frame::Prepare(Prepare {
            key: key.clone(),
            ballot: Ballot { round: 3, node: 4 },
        });
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("gateway-votes");
        let open = || {
            let journal = FileJournal::open(&path, 64 * 1024).unwrap();
            opened(7, 0, Box::new(journal))
        };
        let mut t = open();
        match ask(&mut t, "zend", &prepare).unwrap() {
            Frame::Promise(p) => assert_eq!(p.key, key, "answered under the member's key"),
            other => panic!("a long key is promised: {other:?}"),
        }
        drop(t);
        match ask(&mut open(), "zend", &prepare).unwrap() {
            Frame::Nack(n) => assert_eq!(n.promised, Ballot { round: 3, node: 4 }),
            other => panic!("the promise survived the restart: {other:?}"),
        }
    }

    /// The lowest node votes, and only once every configured peer has been seen.
    #[test]
    fn the_lowest_gateway_is_the_voter() {
        let mut low = opened(2, 1, Box::new(MemoryVotes::new()));
        assert!(!low.voting(), "a peer not yet seen");
        low.note_peer(9);
        assert!(low.voting());
        low.note_peer(9);
        assert!(low.voting(), "a peer seen again changes nothing");

        let mut high = opened(9, 1, Box::new(MemoryVotes::new()));
        high.note_peer(2);
        assert!(!high.voting());
        assert!(ask(&mut high, "zend", &prepare(4)).is_none());

        let mut waiting = opened(1, 2, Box::new(MemoryVotes::new()));
        waiting.note_peer(3);
        assert!(!waiting.voting(), "one peer of two");
        waiting.note_peer(4);
        assert!(waiting.voting());
    }

    /// A millisecond before `t`.
    fn before(t: Millis) -> Millis {
        t - Duration::from_millis(1)
    }

    fn upgrade_prepare(round: u64, node: u64) -> Frame {
        Frame::Prepare(Prepare {
            key: b"upgrade/zend".to_vec(),
            ballot: Ballot { round, node },
        })
    }

    /// Whether member `node`'s prepare at `round`, at `now` on a gateway
    /// opened at `opened_at` on the wall clock, is shown another member's
    /// lease — the promise that stops it with `LeaseBusy`.
    fn shown_a_lease(t: &mut TieBreak, round: u64, node: u64, now: Millis, opened_at: u64) -> bool {
        let wall = WallMillis(opened_at + now.0);
        match t
            .consider("zend", &upgrade_prepare(round, node), now, wall)
            .unwrap()
        {
            Some(Frame::Promise(p)) => p.accepted.is_some(),
            other => panic!("round {round} is promised: {other:?}"),
        }
    }

    /// Member 4 takes zend's `upgrade/zend` for 20 minutes through a gateway
    /// opened at `T0`, on the journal at `path`.
    fn take_upgrade_lease(path: &Path) {
        let mut t = opened(7, 0, Box::new(FileJournal::open(path, 64 * 1024).unwrap()));
        assert!(!shown_a_lease(&mut t, 1, 4, Millis(0), T0));
        let lease = Value::Lease {
            holder: 4,
            ttl: Duration::from_secs(20 * 60),
        }
        .encode();
        let propose = Frame::Propose(Propose {
            key: b"upgrade/zend".to_vec(),
            ballot: Ballot { round: 1, node: 4 },
            value: lease,
        });
        assert!(matches!(
            ask(&mut t, "zend", &propose),
            Some(Frame::Accepted(_))
        ));
    }

    fn reopened(path: &Path, at: u64) -> TieBreak {
        let journal = FileJournal::open(path, 64 * 1024).unwrap();
        TieBreak::open(7, 0, Box::new(journal), WallMillis(at)).unwrap()
    }

    /// **A web deploy no longer revives an upgrade lease.** The gateway
    /// restarted at 09:42:09Z on 2026-10-31 and re-armed a dead `upgrade/zend`
    /// for a full 20 minutes from the restart, so a member asking for it was
    /// refused from 09:52 to ~10:02. Reopened 5 minutes into the term, the
    /// gateway counts the 15 minutes left; reopened again 5 minutes later, the
    /// 10 left — the end never moves past T0 + 20 min.
    #[test]
    fn a_restarted_gateway_counts_only_what_is_left_of_a_lease() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("gateway-votes");
        take_upgrade_lease(&path);

        let at5 = T0 + 5 * MIN;
        let t = reopened(&path, at5);
        assert_eq!(
            serde_json::to_value(t.armed_leases(Millis(0))).unwrap(),
            json!([{
                "app": "zend",
                "key": "upgrade/zend",
                "holder": "0000000000000004",
                "remaining_s": 900,
            }]),
            "the status page's tiebreak_leases"
        );
        drop(t);

        let at10 = T0 + 10 * MIN;
        let mut t = reopened(&path, at10);
        assert_eq!(t.armed_leases(Millis(0))[0].remaining_s, 10 * 60);
        let end = Millis(10 * MIN);
        assert!(shown_a_lease(&mut t, 2, 5, before(end), at10));
        assert!(!shown_a_lease(&mut t, 3, 5, end, at10), "free at T0+20");
        assert!(t.armed_leases(end).is_empty());
    }

    /// Reopened past the lease's end, nothing is armed and another member
    /// takes the lease at once; reopened with the wall clock gone back, the
    /// lease counts for its full term.
    #[test]
    fn a_gateway_reopened_late_or_with_its_clock_back() {
        let dir = tempfile::TempDir::new().unwrap();
        let late = dir.path().join("late");
        take_upgrade_lease(&late);
        let at25 = T0 + 25 * MIN;
        let mut t = reopened(&late, at25);
        assert!(t.armed_leases(Millis(0)).is_empty());
        assert!(!shown_a_lease(&mut t, 2, 5, Millis(0), at25));

        let back = dir.path().join("back");
        take_upgrade_lease(&back);
        let back_at = T0 - MIN;
        let mut t = reopened(&back, back_at);
        assert_eq!(t.armed_leases(Millis(0))[0].remaining_s, 20 * 60);
        let end = Millis(20 * MIN);
        assert!(shown_a_lease(&mut t, 2, 5, before(end), back_at));
        assert!(!shown_a_lease(&mut t, 3, 5, end, back_at));
    }

    /// A held key shows as its app and the member's key; a digested one has
    /// no app.
    #[test]
    fn a_held_key_shows_its_app_and_key() {
        assert_eq!(
            unscoped(b"zend\0upgrade/zend"),
            ("zend".to_string(), "upgrade/zend".to_string())
        );
        assert_eq!(
            unscoped(b"#sha256:abc"),
            (String::new(), "#sha256:abc".to_string())
        );
    }
}
