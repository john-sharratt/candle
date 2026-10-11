//! What a substrate stamps on the records it writes itself.
//!
//! The writer holds its node, the next number in its own sequence, its clock
//! and its current generation, and mints one [`RecordIdentity`] per record it
//! originates. At open it is primed from every record the walk meets, so a
//! restarted substrate continues its sequence where it left off and never
//! stamps a reading below one it has already seen.

use std::time::{SystemTime, UNIX_EPOCH};

use super::hlc::Hlc;
use super::record_identity::{RecordIdentity, GEN_DETACHED};

/// Milliseconds since the Unix epoch, by the wall clock.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The writer's identity state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Origination {
    node: u64,
    next_oseq: u64,
    clock: Hlc,
    gen: u64,
    /// While cut off from every cluster: the first `oseq` written detached,
    /// or the next one when nothing has been yet (§6.7).
    detached_from: Option<u64>,
}

impl Origination {
    /// A writer for `node` that has written nothing.
    pub fn new(node: u64) -> Origination {
        Origination {
            node,
            next_oseq: 1,
            clock: Hlc::new(),
            gen: 0,
            detached_from: None,
        }
    }

    /// The member is cut off: what it mints from here on is detached, under
    /// [`GEN_DETACHED`], until [`Self::attach`]. Starting from `oseq` when it
    /// was cut off already at that record (a reopen that found it so), or from
    /// the next record otherwise.
    pub fn detach(&mut self, from: Option<u64>) {
        if self.detached_from.is_none() {
            self.detached_from = Some(from.unwrap_or(self.next_oseq));
        }
    }

    /// The member rejoined under generation `gen`, which fixes everything it
    /// wrote detached; it mints in `gen` from here on.
    pub fn attach(&mut self, gen: u64) {
        self.detached_from = None;
        self.gen = self.gen.max(gen);
    }

    /// Whether the member is cut off.
    pub fn is_detached(&self) -> bool {
        self.detached_from.is_some()
    }

    /// The `oseq`s written detached since the member was cut off, when there
    /// are any.
    pub fn detached_range(&self) -> Option<(u64, u64)> {
        let from = self.detached_from?;
        (from < self.next_oseq).then_some((from, self.next_oseq - 1))
    }

    /// Take a record the substrate holds into account, whoever wrote it: its
    /// clock reading and its generation bound everything minted after, and one
    /// of this node's own moves the sequence past it.
    pub fn observe(&mut self, id: &RecordIdentity) {
        if id.is_none() {
            return;
        }
        if id.origin == self.node {
            self.next_oseq = self.next_oseq.max(id.oseq + 1);
        }
        self.clock.observe(id.hlc);
        if id.gen != GEN_DETACHED {
            self.gen = self.gen.max(id.gen);
        }
    }

    /// The identity of the next record this node writes, with the wall clock at
    /// `now_ms`, under claim or lease `epoch` (zero for none).
    pub fn mint(&mut self, now_ms: u64, epoch: u64) -> RecordIdentity {
        let oseq = self.next_oseq;
        self.next_oseq += 1;
        RecordIdentity {
            origin: self.node,
            oseq,
            gen: if self.is_detached() {
                GEN_DETACHED
            } else {
                self.gen
            },
            epoch,
            hlc: self.clock.tick(now_ms),
        }
    }

    /// This writer's node.
    pub fn node(&self) -> u64 {
        self.node
    }

    /// The `oseq` the next record will carry.
    pub fn next_oseq(&self) -> u64 {
        self.next_oseq
    }

    /// The greatest clock reading taken or observed.
    pub fn last_hlc(&self) -> u64 {
        self.clock.last()
    }

    /// The generation minted records carry.
    pub fn gen(&self) -> u64 {
        self.gen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_writer_numbers_from_one_with_rising_readings() {
        let mut w = Origination::new(7);
        let a = w.mint(1_000, 0);
        let b = w.mint(1_000, 0);
        assert_eq!((a.origin, a.oseq, b.oseq), (7, 1, 2));
        assert!(b.hlc > a.hlc);
        assert!(b.stamp() > a.stamp());
    }

    /// Primed from a walk, a writer continues its own sequence and passes every
    /// reading it saw — its own and other origins'.
    #[test]
    fn a_primed_writer_continues_past_what_it_saw() {
        let mut w = Origination::new(7);
        w.observe(&RecordIdentity {
            origin: 7,
            oseq: 41,
            gen: 2,
            epoch: 0,
            hlc: 9 << 16,
        });
        w.observe(&RecordIdentity {
            origin: 8,
            oseq: 500,
            gen: 3,
            epoch: 0,
            hlc: 20 << 16,
        });
        let next = w.mint(1, 0);
        assert_eq!(
            next.oseq, 42,
            "another origin's numbers do not move this one"
        );
        assert_eq!(next.gen, 3);
        assert!(next.hlc > 20 << 16);
    }

    /// A detached writer stamps the provisional generation, which sorts above
    /// everything it holds; its detached range is what it wrote since; once
    /// attached under a decided generation it stamps that one.
    #[test]
    fn a_detached_writer_stamps_the_provisional_generation() {
        let mut w = Origination::new(7);
        let joined = w.mint(1_000, 0);
        w.detach(None);
        assert_eq!(w.detached_range(), None, "nothing written detached yet");
        let a = w.mint(1_000, 0);
        let b = w.mint(1_000, 0);
        assert_eq!((a.gen, b.gen), (GEN_DETACHED, GEN_DETACHED));
        assert!(a.stamp() > joined.stamp());
        assert_eq!(w.detached_range(), Some((2, 3)));
        w.detach(None);
        assert_eq!(
            w.detached_range(),
            Some((2, 3)),
            "still from where it began"
        );
        w.attach(4);
        assert!(!w.is_detached());
        assert_eq!(w.mint(1_000, 0).gen, 4);
        assert_eq!(w.detached_range(), None);
    }

    /// A detached record does not raise the generation a joined writer stamps,
    /// and a record with no identity changes nothing.
    #[test]
    fn detached_and_unstamped_records_leave_the_generation() {
        let mut w = Origination::new(7);
        w.observe(&RecordIdentity {
            origin: 8,
            oseq: 1,
            gen: GEN_DETACHED,
            epoch: 0,
            hlc: 1,
        });
        w.observe(&RecordIdentity::NONE);
        assert_eq!(w.gen(), 0);
        assert_eq!(w.next_oseq(), 1);
    }
}
