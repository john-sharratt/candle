//! A substrate's side of replication (`docs/replicated_substrate.md` §6–§7):
//! what a member's replicator asks of the store, on the substrate and its
//! persistence together. The cluster's protocol is the host's; this is the
//! substrate it drives.
//!
//! What touches the log runs on the log's owner; answering a peer runs there
//! as background work, behind every call and write of the member's own.

use std::io;
use std::path::PathBuf;

use serde_json::Value;

use super::ids::TimelineId;
use super::resolver::Conversation;
use crate::persistence::identity::{now_ms, MintGeneration, VersionStamp, GEN_DETACHED};
use crate::persistence::joined;
use crate::persistence::listing::ListedRecord;
use crate::persistence::record::{decode_header, CustomObjectPayload};
use crate::persistence::replica::AppliedSet;
use crate::persistence::streams::StreamDecl;
use crate::persistence::{PersistenceError, Result as PersistenceResult, Tap};

/// The keys consensus decisions and `max_members` are kept under (§5.9).
pub const CLUSTER_PREFIX: &str = "cluster/";

/// What a member holds of one origin's range, for a catch-up's headers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeldRange {
    pub records: Vec<ListedRecord>,
    /// The `oseq`s in the range no record holds any more.
    pub gone: Vec<u64>,
}

/// A generation the group has decided, handed to the rejoin that fixes what
/// this member wrote cut off.
struct Decided(u64);

impl MintGeneration for Decided {
    fn mint_generation(&self, _known: u64) -> u64 {
        self.0
    }
}

impl Conversation {
    /// Tell `tap` the `oseq` of every record this substrate writes that may be
    /// sent (§6.3).
    pub fn set_replication_tap(&self, tap: Tap) {
        self.persistence.call(|p| p.set_tap(tap));
    }

    /// What this member holds of every origin, its own written records
    /// included (§7.2, §7.5).
    pub fn applied_set(&self) -> AppliedSet {
        self.persistence.call(|p| p.applied_set())
    }

    /// Every own `oseq` up to this one is written and may be sent, whatever
    /// gaps a purge left below it (§7.8) — and never past one still on its way
    /// to the log ([`crate::persistence::SubstratePersistence::own_sendable_through`]).
    pub fn own_through(&self) -> u64 {
        self.persistence.call(|p| p.own_sendable_through())
    }

    /// This member is joined and caught up now: an open after a long absence
    /// is judged from this moment (§7.8).
    pub fn mark_joined(&self) -> io::Result<()> {
        joined::write(&self.substrate_dir(), now_ms())
    }

    /// The `oseq` of `timeline`'s newest commit that `origin` wrote — what a
    /// release names as the point the conversation must be held through
    /// (`docs/replicated_substrate.md` §2.2). `None` when `origin` has
    /// committed none of its turns.
    pub fn last_commit_oseq(&self, timeline: super::ids::TimelineId, origin: u64) -> Option<u64> {
        let read = self.read();
        read.turn_streams_of(timeline.raw())
            .map(|(_, _, rt)| rt.ids.commit)
            .filter(|c| c.origin == origin && c.oseq > 0)
            .map(|c| c.oseq)
            .max()
    }

    /// Whether every turn of `timeline` that `origin` declared with K/V has its
    /// commit — the point a release of it may name (§2.2).
    ///
    /// **A turn seals before its commit is written.** The seal runs in the
    /// turn's own loop; the commit is the persistence writer's, after the
    /// turn's chunks. A release sent between the two named the commit before —
    /// the previous turn's — and a member holding everything through *that*
    /// was handed the conversation while the new turn's snapshot, written
    /// first, had already reached it: on WhiteWorld (2026-10-20 20:02:24) a
    /// resume found a snapshot one turn ahead of the history it recovered and
    /// went on with no recurrent memory at all. A turn declared with no K/V
    /// writes no commit, and is not waited for.
    pub fn own_turns_committed(&self, timeline: TimelineId, origin: u64) -> bool {
        let read = self.read();
        let committed = read
            .turn_streams_of(timeline.raw())
            .filter(|(_, _, rt)| rt.ids.decl.origin == origin)
            .filter(|(_, _, rt)| match &rt.decl {
                Some(StreamDecl::Turn(t)) => t.block_end > t.block_start,
                _ => false,
            })
            .all(|(_, _, rt)| rt.ids.commit.oseq > 0);
        committed
    }

    /// Whether this member holds `timeline` complete through `origin`'s
    /// commit `through`: that commit applied here, and every turn up to the
    /// one it committed held — placed, so complete, with no gap below it.
    ///
    /// **Judged by the conversation's own records, not the origin's lane.** A
    /// member behind on the releaser's lane as a whole — its KV and snapshots
    /// of other conversations still arriving — may hold this one entire, and
    /// then it may take it (§4.4). The commit names the extent; the turns
    /// below it are what must be here.
    pub fn holds_complete_through(
        &self,
        timeline: super::ids::TimelineId,
        origin: u64,
        through: u64,
    ) -> bool {
        let read = self.read();
        let Some(committed) = read
            .turn_streams_of(timeline.raw())
            .find(|(_, _, rt)| rt.ids.commit.origin == origin && rt.ids.commit.oseq == through)
            .map(|(idx, _, _)| idx)
        else {
            return false;
        };
        (0..=committed.0).all(|i| read.holds_turn(timeline, super::ids::TurnIndex(i)))
    }

    /// Record `(origin, oseq)` as it travels — its header line and payload,
    /// unpadded — from wherever maintenance moved it, or `None` when this
    /// member does not hold it.
    ///
    /// Read as background work, as [`Self::list_held`] lists: every caller is
    /// replication — this member's own sends and the records a peer's catch-up
    /// fetches — and a fetch reads one record off disk after another, so the
    /// member's own calls and writes go first.
    pub fn read_event(&self, origin: u64, oseq: u64) -> Option<Vec<u8>> {
        let read = self.persistence.call_behind(|p| p.read_event(origin, oseq));
        let mut bytes = match read {
            Ok(Some(bytes)) => bytes,
            Ok(None) => return None,
            Err(e) => {
                tracing::error!(
                    origin,
                    oseq,
                    "a record owed to another member is unreadable: {e}"
                );
                return None;
            }
        };
        let (header, head) = decode_header(&bytes).ok()?;
        bytes.truncate(head + header.payload_len as usize);
        Some(bytes)
    }

    /// [`Self::read_event`] for each of `oseqs`, in order, read off disk
    /// without the log's owner.
    ///
    /// A peer's fetch read one record per turn on the log, each waiting behind
    /// the scheduler's — answering ran at the pace of the log, not the disk
    /// (about 9 MB/s to a member catching up while `.5` ingested). Now one
    /// short background call, memory only, takes every record's location and
    /// whether any rebase exists; the records are read from their sealed
    /// segments without the owner (`SealedReader`), so the scheduler waits on
    /// no fetch's I/O. A record that read cannot vouch for — in the active
    /// segment, moved since its location was taken, or detached while a rebase
    /// exists — is read by the owner, as before.
    pub fn read_events(&self, origin: u64, oseqs: &[u64]) -> Vec<Option<Vec<u8>>> {
        let reader = self.persistence.sealed_reader();
        let (places, no_rebases) = self.persistence.call_behind(|p| {
            let places: Vec<_> = oseqs.iter().map(|oseq| p.locate(origin, *oseq)).collect();
            (places, p.no_rebases())
        });
        oseqs
            .iter()
            .zip(places)
            .map(|(&oseq, place)| {
                let read = place
                    .and_then(|(segment, offset)| reader.read_raw(segment, offset))
                    .and_then(|r| r.ok())
                    .and_then(|mut bytes| {
                        let (header, head) = decode_header(&bytes).ok()?;
                        let id = header.identity();
                        let vouched = (id.origin, id.oseq) == (origin, oseq)
                            && (no_rebases || id.gen != GEN_DETACHED);
                        vouched.then(|| {
                            bytes.truncate(head + header.payload_len as usize);
                            bytes
                        })
                    });
                match read {
                    Some(bytes) => Some(bytes),
                    // Never located, or not vouched for: the owner's read
                    // decides, as it did for every record before.
                    None if place.is_some() => self.read_event(origin, oseq),
                    None => None,
                }
            })
            .collect()
    }

    /// What this member holds of `origin`'s `from..=to`, a range it covers
    /// completely.
    ///
    /// Listed from memory (`persistence::listing`) in one short background
    /// call on the log's owner: answering a peer is background work, and the
    /// scheduler's writes go first.
    pub fn list_held(&self, origin: u64, from: u64, to: u64) -> HeldRange {
        let listing = self.persistence.call_behind(|p| p.list(origin, from, to));
        HeldRange {
            records: listing.records,
            gone: listing.gone,
        }
    }

    /// `oseq`s of `origin` a catch-up learned are gone (§7.2).
    pub fn record_gone(&self, origin: u64, gone: &[u64]) {
        self.persistence.call(|p| {
            for oseq in gone {
                p.mark_gone(origin, *oseq, *oseq);
            }
        });
    }

    /// A position in the log owner's write queue.
    pub fn write_ticket(&self) -> u64 {
        self.persistence.ticket()
    }

    /// Wait until every write at or below `ticket` is in the log.
    pub fn wait_written(&self, ticket: u64) {
        self.persistence.wait_written(ticket);
    }

    /// Cut off from every cluster: what this substrate writes from here on is
    /// detached (§6.7).
    pub fn detach(&self) {
        self.persistence.call(|p| p.detach());
    }

    /// Whether it holds records written cut off that no rejoin has fixed.
    pub fn holds_detached(&self) -> bool {
        self.minter.origination().detached_range().is_some()
    }

    /// Rejoined under generation `gen`, decided by the group: what was written
    /// cut off is fixed to it, after the queued writes have landed, and minted
    /// under it from here on. The caller reloads.
    pub fn rejoin_at(&self, gen: u64) -> PersistenceResult<()> {
        self.persistence
            .call_after_writes(|p| p.rejoin(&Decided(gen)))?;
        Ok(())
    }

    /// The directory the substrate's log is in — where its member keeps its
    /// vote journal (§5.9).
    pub fn substrate_dir(&self) -> PathBuf {
        self.persistence.dir().to_path_buf()
    }

    /// The generation this substrate writes under.
    pub fn generation(&self) -> u64 {
        self.minter.origination().gen()
    }

    /// Every live custom object under [`CLUSTER_PREFIX`], with the stamp of the
    /// record that holds it.
    pub fn cluster_objects(&self) -> Vec<(String, Value, VersionStamp)> {
        self.read()
            .custom_objects_under(CLUSTER_PREFIX)
            .into_iter()
            .map(|(obj, id)| (obj.key, obj.blob, id.stamp()))
            .collect()
    }

    /// Write `blob` under `key`, a key under [`CLUSTER_PREFIX`], as a
    /// replicated record.
    pub fn put_cluster_object(&self, key: &str, blob: Value) -> PersistenceResult<()> {
        debug_assert!(key.starts_with(CLUSTER_PREFIX));
        let obj = CustomObjectPayload::new(key, blob);
        let id = self.persistence.call(|p| {
            let id = p.mint();
            p.write_custom_object(&obj, id)?;
            p.commit()?;
            Ok::<_, PersistenceError>(id)
        })?;
        self.update(|s| s.apply_custom_object(obj.clone(), id));
        Ok(())
    }
}
