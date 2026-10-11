//! The member's substrate, as replication drives it (§6, §7). The `cluster`
//! crate never decodes a record: it moves records' bytes between members and
//! asks the store to apply, list and read them. Every call may block on the
//! store's disk and locks, so the replicator makes them off the async runtime.

use applied_set::AppliedSet;

use crate::frame::{Gone, RecordHeader};

/// The provisional generation of a record written cut off (§6.7).
pub const GEN_DETACHED: u64 = u64::MAX;

/// Told the `oseq` of each own record that may be sent.
pub type Tap = Box<dyn Fn(u64) + Send + Sync>;

/// What applying a batch of records came to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Applied {
    /// Per record, in the batch's order: whether it is applied now — written,
    /// or skipped as a duplicate, dominated, superseded or deferred. A record
    /// refused is not, and a catch-up fetches it again.
    pub taken: Vec<bool>,
    /// A turn arrived ranked below a turn this member holds: it must stop
    /// taking work and reload (§6.6).
    pub reload: bool,
}

/// The volume a store is on, as fault mode judges it (§7.7).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Disk {
    pub free: u64,
    pub total: u64,
    /// A write has failed for lack of space since the last test write
    /// succeeded.
    pub write_failed: bool,
}

/// The timeline tombstones a store holds, by how far each is from expiring
/// (§6.8).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tombstones {
    pub held: usize,
    /// Of those, the ones a complete compaction has run past.
    pub swept: usize,
    /// Expired tombstones kept as deleted-set entries.
    pub deleted: usize,
}

/// What a store holds of one origin's range, for a `HEADERS`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Listing {
    pub records: Vec<RecordHeader>,
    /// The `oseq`s in the range it knows no longer exist, each one that was an
    /// expired tombstone with its deleted-set entry.
    pub gone: Vec<Gone>,
}

pub trait Store: Send + Sync + 'static {
    /// This member's node, the origin of every record it writes.
    fn node(&self) -> u64;

    /// Every own `oseq` up to this one is written and may be sent: everything
    /// but what it wrote cut off and no rejoin has fixed yet.
    fn own_through(&self) -> u64;

    /// Call `tap` with the `oseq` of every record this member writes from now
    /// on that may be sent — at once, on the writing thread, so it must not
    /// block. A record written cut off is tapped when a rejoin fixes it.
    fn set_tap(&self, tap: Tap);

    /// Record `(origin, oseq)`'s bytes — header line and payload — from
    /// wherever maintenance has moved it, or `None` when it is not held here.
    fn read(&self, origin: u64, oseq: u64) -> Option<Vec<u8>>;

    /// Every `(origin, oseq)` applied here or known gone, this member's own
    /// through [`Self::own_through`] included.
    fn applied(&self) -> AppliedSet;

    /// Apply records other members wrote, as `(origin, oseq, bytes)`.
    fn apply(&self, records: &[(u64, u64, Vec<u8>)]) -> Applied;

    /// The records held in `origin`'s `from..=to`, a range the store covers
    /// completely, and the `oseq`s in it that are gone.
    fn list(&self, origin: u64, from: u64, to: u64) -> Listing;

    /// `oseq`s of `origin` a catch-up learned are gone; an expired
    /// tombstone's deleted-set entry is applied before anything else.
    fn gone(&self, origin: u64, gone: &[Gone]);

    /// A position in the store's write queue: every record handed to the
    /// writer before this call is at or below it.
    fn ticket(&self) -> u64;

    /// Block until every record at or below `ticket` is written (and tapped).
    fn wait_written(&self, ticket: u64);

    /// The member is cut off from every cluster: what it writes from here on
    /// carries [`GEN_DETACHED`] and is not tapped.
    fn detach(&self);

    /// Whether it holds records written cut off that no rejoin has fixed.
    fn detached(&self) -> bool;

    /// Fix what it wrote cut off to generation `gen`, decided by the group,
    /// and write under it from here on; those records are tapped. The store
    /// holds their values under the provisional generation still, so the
    /// caller reloads it.
    fn rejoin(&self, gen: u64);

    /// The greatest generation this member knows to be decided.
    fn generation(&self) -> u64;

    /// Write consensus decision `key = value` as a replicated record (§5.9).
    fn record_decision(&self, key: &[u8], value: &[u8]);

    /// The decision recorded for `key`, from any member.
    fn decision(&self, key: &[u8]) -> Option<Vec<u8>>;

    /// `max_members` as the store holds it (§5.9): at least one.
    fn max_members(&self) -> u32;

    /// Write a raise of `max_members` to `seen`.
    fn raise_max_members(&self, seen: u32);

    /// Write a set of `max_members` to `n`.
    fn set_max_members(&self, n: u32);

    /// Rebuild the in-memory state from the log — after a rejoin fixed a
    /// generation, or a turn arrived ranked below one held. Blocks until done.
    fn reload(&self);

    /// The volume it is on, measured now.
    fn disk(&self) -> Disk;

    /// A test append and commit; whether it succeeded. One that does clears
    /// the record of a failed write.
    fn test_write(&self) -> bool;

    /// Drop one fully-dead segment of its log, if it has one: freeing space
    /// with no scratch space needed. Whether one was dropped.
    fn drop_dead_segment(&self) -> bool;

    /// Records held back because this build does not decode their type: per
    /// type and origin, how many.
    fn deferred(&self) -> Vec<(String, u64, usize)>;

    /// Its timeline tombstones.
    fn tombstones(&self) -> Tombstones;
}
