//! The census's counters: per size class, live bytes and live count, and the
//! bytes ever allocated — split into shards so threads do not share a line.
//!
//! **A thread writes one shard, chosen once.** An allocation is counted on the
//! shard of the thread that makes it and a free on the shard of the thread that
//! frees, which need not be the same one. A shard's own figures can therefore
//! run negative — a shard whose thread frees what another allocated — and are
//! kept as wrapping `u64`s; only their **sum across shards** means anything, and
//! that sum is exact, because every add and every subtract lands somewhere.
//!
//! Every update is one `Relaxed` `fetch_add` or `fetch_sub`. A reading sums the
//! shards with `Relaxed` loads while threads go on counting, so it can see a
//! call that has updated its bytes and not yet its count; the error is bounded
//! by the calls in flight while it reads.
//!
//! A block that carries its scope ([`Census::on_alloc_scoped`]) is counted
//! twice: in its class, like every block, and in its scope's class.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use super::classes::{class_of, CLASSES};
use super::scope::SCOPES;

/// How many shards. More than the cores of any machine this runs on, so two
/// threads share one only when there are more threads than shards.
pub const SHARDS: usize = 64;

/// One shard. Aligned to 128 bytes — two cache lines, the unit adjacent-line
/// prefetch pulls in — so no two shards ever share one.
#[repr(align(128))]
pub struct Shard {
    bytes: [AtomicU64; CLASSES],
    count: [AtomicU64; CLASSES],
    allocated: AtomicU64,
    scoped_bytes: [[AtomicU64; CLASSES]; SCOPES],
    scoped_count: [[AtomicU64; CLASSES]; SCOPES],
}

impl Shard {
    const fn new() -> Shard {
        Shard {
            bytes: [const { AtomicU64::new(0) }; CLASSES],
            count: [const { AtomicU64::new(0) }; CLASSES],
            allocated: AtomicU64::new(0),
            scoped_bytes: [const { [const { AtomicU64::new(0) }; CLASSES] }; SCOPES],
            scoped_count: [const { [const { AtomicU64::new(0) }; CLASSES] }; SCOPES],
        }
    }
}

/// The whole census.
pub struct Census {
    shards: [Shard; SHARDS],
}

/// One size class's live figures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ClassCount {
    pub bytes: u64,
    pub count: u64,
}

/// The census summed: what is live, by class and by scope, and the bytes ever
/// allocated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counts {
    pub classes: [ClassCount; CLASSES],
    /// The blocks that carry their scope, by scope and class.
    pub scopes: [[ClassCount; CLASSES]; SCOPES],
    pub allocated_total: u64,
}

impl Counts {
    /// Live bytes across every class.
    pub fn live_bytes(&self) -> u64 {
        self.classes.iter().map(|c| c.bytes).sum()
    }

    /// Live allocations across every class.
    pub fn live_count(&self) -> u64 {
        self.classes.iter().map(|c| c.count).sum()
    }

    /// Bytes ever freed: everything allocated that is no longer live.
    pub fn freed_total(&self) -> u64 {
        self.allocated_total.saturating_sub(self.live_bytes())
    }

    /// Live bytes of the blocks scope `s` made, across every class.
    pub fn scope_bytes(&self, s: usize) -> u64 {
        self.scopes[s].iter().map(|c| c.bytes).sum()
    }
}

/// A wrapping sum read back as a live figure: clamped at zero, since a free
/// counted before its allocation is visible reads below it for an instant.
fn live(sum: u64) -> u64 {
    (sum as i64).max(0) as u64
}

impl Default for Census {
    fn default() -> Self {
        Census::new()
    }
}

impl Census {
    pub const fn new() -> Census {
        Census {
            shards: [const { Shard::new() }; SHARDS],
        }
    }

    /// Count a `size`-byte allocation on `shard`.
    #[inline(always)]
    pub fn on_alloc(&self, shard: usize, size: usize) {
        let s = &self.shards[shard % SHARDS];
        let k = class_of(size);
        s.bytes[k].fetch_add(size as u64, Relaxed);
        s.count[k].fetch_add(1, Relaxed);
        s.allocated.fetch_add(size as u64, Relaxed);
    }

    /// Count the free of a `size`-byte allocation on `shard`.
    #[inline(always)]
    pub fn on_free(&self, shard: usize, size: usize) {
        let s = &self.shards[shard % SHARDS];
        let k = class_of(size);
        s.bytes[k].fetch_sub(size as u64, Relaxed);
        s.count[k].fetch_sub(1, Relaxed);
    }

    /// Count a `size`-byte allocation made in scope `scope` on `shard`: in its
    /// class, and in its scope's.
    #[inline(always)]
    pub fn on_alloc_scoped(&self, shard: usize, size: usize, scope: u8) {
        self.on_alloc(shard, size);
        let s = &self.shards[shard % SHARDS];
        let (k, c) = (class_of(size), scope as usize & (SCOPES - 1));
        s.scoped_bytes[c][k].fetch_add(size as u64, Relaxed);
        s.scoped_count[c][k].fetch_add(1, Relaxed);
    }

    /// Count the free of a `size`-byte allocation made in scope `scope`.
    #[inline(always)]
    pub fn on_free_scoped(&self, shard: usize, size: usize, scope: u8) {
        self.on_free(shard, size);
        let s = &self.shards[shard % SHARDS];
        let (k, c) = (class_of(size), scope as usize & (SCOPES - 1));
        s.scoped_bytes[c][k].fetch_sub(size as u64, Relaxed);
        s.scoped_count[c][k].fetch_sub(1, Relaxed);
    }

    /// Sum every shard. Wrapping, then clamped at zero: a class can read below
    /// zero only for the instant a free is counted before its allocation's
    /// count is visible, and a negative live figure is never the answer.
    pub fn counts(&self) -> Counts {
        let mut bytes = [0u64; CLASSES];
        let mut count = [0u64; CLASSES];
        let mut scoped_bytes = [[0u64; CLASSES]; SCOPES];
        let mut scoped_count = [[0u64; CLASSES]; SCOPES];
        let mut allocated = 0u64;
        for s in &self.shards {
            for k in 0..CLASSES {
                bytes[k] = bytes[k].wrapping_add(s.bytes[k].load(Relaxed));
                count[k] = count[k].wrapping_add(s.count[k].load(Relaxed));
            }
            for c in 0..SCOPES {
                for k in 0..CLASSES {
                    scoped_bytes[c][k] =
                        scoped_bytes[c][k].wrapping_add(s.scoped_bytes[c][k].load(Relaxed));
                    scoped_count[c][k] =
                        scoped_count[c][k].wrapping_add(s.scoped_count[c][k].load(Relaxed));
                }
            }
            allocated = allocated.wrapping_add(s.allocated.load(Relaxed));
        }
        let mut classes = [ClassCount::default(); CLASSES];
        for k in 0..CLASSES {
            classes[k] = ClassCount {
                bytes: live(bytes[k]),
                count: live(count[k]),
            };
        }
        let mut scopes = [[ClassCount::default(); CLASSES]; SCOPES];
        for c in 0..SCOPES {
            for k in 0..CLASSES {
                scopes[c][k] = ClassCount {
                    bytes: live(scoped_bytes[c][k]),
                    count: live(scoped_count[c][k]),
                };
            }
        }
        Counts {
            classes,
            scopes,
            allocated_total: allocated,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    /// **An allocation is counted in its class, and its free takes it out
    /// again** — exact bytes, exact count.
    #[test]
    fn an_allocation_and_its_free_are_counted_exactly() {
        let c = Census::new();
        c.on_alloc(0, 30_000);
        let n = c.counts();
        assert_eq!(
            n.classes[15],
            ClassCount {
                bytes: 30_000,
                count: 1
            }
        );
        assert_eq!(
            (n.live_bytes(), n.live_count(), n.allocated_total),
            (30_000, 1, 30_000)
        );
        c.on_free(0, 30_000);
        let n = c.counts();
        assert_eq!(n.classes[15], ClassCount::default());
        assert_eq!(
            (n.live_bytes(), n.freed_total(), n.allocated_total),
            (0, 30_000, 30_000)
        );
    }

    /// **A free counted on another shard than its allocation still balances**:
    /// the shards differ, the sum is exact.
    #[test]
    fn a_free_on_another_shard_balances_in_the_sum() {
        let c = Census::new();
        c.on_alloc(3, 5_000);
        c.on_free(41, 5_000);
        assert_eq!(c.counts().live_bytes(), 0);
        assert_eq!(c.counts().live_count(), 0);
        assert_eq!(c.counts().allocated_total, 5_000);
    }

    /// **A scoped block is counted in its class and in its scope's**, and its
    /// free — on another shard — takes it out of both.
    #[test]
    fn a_scoped_block_is_counted_in_its_scope() {
        let c = Census::new();
        c.on_alloc_scoped(2, 100_000, 5);
        let n = c.counts();
        assert_eq!(
            n.scopes[5][17],
            ClassCount {
                bytes: 100_000,
                count: 1
            }
        );
        assert_eq!(n.classes[17].bytes, 100_000);
        assert_eq!(n.scope_bytes(5), 100_000);
        assert_eq!(n.scope_bytes(0), 0);
        c.on_free_scoped(9, 100_000, 5);
        let n = c.counts();
        assert_eq!(n.scopes[5][17], ClassCount::default());
        assert_eq!(n.live_bytes(), 0);
    }

    /// **Sixteen threads allocating and freeing on their own shards leave the
    /// census exact**: nothing live, everything allocated also freed.
    #[test]
    fn many_threads_leave_an_exact_census() {
        let c = Arc::new(Census::new());
        let threads: Vec<_> = (0..16)
            .map(|t| {
                let c = Arc::clone(&c);
                thread::spawn(move || {
                    for i in 0..10_000 {
                        let size = 16 + (i * 37 + t * 101) % 40_000;
                        c.on_alloc(t, size);
                        // Freed on the next thread's shard half the time.
                        c.on_free(if i % 2 == 0 { t } else { t + 1 }, size);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let n = c.counts();
        assert_eq!((n.live_bytes(), n.live_count()), (0, 0));
        assert_eq!(n.freed_total(), n.allocated_total);
        assert!(n.allocated_total > 0);
    }
}
