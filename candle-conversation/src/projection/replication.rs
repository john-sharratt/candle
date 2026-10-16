//! A substrate's side of replication (`docs/replicated_substrate.md` §6–§7):
//! what a member's replicator asks of the store, on the substrate and its
//! persistence together. The cluster's protocol is the host's; this is the
//! substrate it drives.
//!
//! Locks are taken in the order a reload takes them — persistence, then the
//! substrate — and never nested the other way.

use std::path::PathBuf;

use serde_json::Value;

use super::resolver::Conversation;
use crate::persistence::deleted::{DeletedEntry, DeletedPayload};
use crate::persistence::identity::{MintGeneration, VersionStamp};
use crate::persistence::listing::ListedRecord;
use crate::persistence::record::{decode_header, CustomObjectPayload};
use crate::persistence::replica::AppliedSet;
use crate::persistence::{Result as PersistenceResult, Tap};

/// The keys consensus decisions and `max_members` are kept under (§5.9).
pub const CLUSTER_PREFIX: &str = "cluster/";

/// What a member holds of one origin's range, for a catch-up's headers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeldRange {
    pub records: Vec<ListedRecord>,
    /// The `oseq`s in the range no record holds any more, each one that was an
    /// expired tombstone with the deleted-set entry it left (§6.8).
    pub gone: Vec<(u64, Option<DeletedEntry>)>,
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
        self.persistence.lock().unwrap().set_tap(tap);
    }

    /// What this member holds of every origin, its own written records
    /// included (§7.2, §7.5).
    pub fn applied_set(&self) -> AppliedSet {
        self.persistence.lock().unwrap().applied_set()
    }

    /// Record `(origin, oseq)` as it travels — its header line and payload,
    /// unpadded — from wherever maintenance moved it, or `None` when this
    /// member does not hold it.
    ///
    /// Read behind the waiters, as [`Self::list_held`] lists: every caller is
    /// replication — this member's own sends and the records a peer's catch-up
    /// fetches — and a fetch reads one record off disk after another, so
    /// foreground holds would keep the scheduler out for the whole fetch.
    pub fn read_event(&self, origin: u64, oseq: u64) -> Option<Vec<u8>> {
        let read = self
            .persistence
            .lock_behind_waiters()
            .unwrap()
            .read_event(origin, oseq);
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

    /// What this member holds of `origin`'s `from..=to`, a range it covers
    /// completely; an `oseq` gone with an expired tombstone carries its entry.
    ///
    /// Listed from memory (`persistence::listing`) under one short hold of the
    /// persistence lock, taken behind the waiters (`lock_behind_waiters`):
    /// answering a peer is background work, and the scheduler's writes go
    /// first.
    pub fn list_held(&self, origin: u64, from: u64, to: u64) -> HeldRange {
        let listing = self
            .persistence
            .lock_behind_waiters()
            .unwrap()
            .list(origin, from, to);
        let view = self.read();
        let in_range = |e: &DeletedEntry| e.origin() == origin && (from..=to).contains(&e.oseq());
        let entries: Vec<DeletedEntry> = view
            .deleted_entries()
            .map(DeletedEntry::Timeline)
            .chain(view.deleted_keys().map(DeletedEntry::Key))
            .filter(in_range)
            .collect();
        HeldRange {
            records: listing.records,
            gone: listing
                .gone
                .into_iter()
                .map(|oseq| (oseq, entries.iter().copied().find(|e| e.oseq() == oseq)))
                .collect(),
        }
    }

    /// `oseq`s of `origin` a catch-up learned are gone. A deleted-set entry
    /// among them is applied, and made durable, before anything else: the
    /// timeline it names is dead here from now on, and the key it names holds
    /// against every write below its tombstone (§6.8, §7.2).
    pub fn record_gone(
        &self,
        origin: u64,
        gone: &[(u64, Option<DeletedEntry>)],
    ) -> PersistenceResult<()> {
        let mut learned = DeletedPayload::default();
        for entry in gone.iter().filter_map(|(_, e)| *e) {
            match entry {
                DeletedEntry::Timeline(t) => learned.timelines.push(t),
                DeletedEntry::Key(k) => learned.keys.push(k),
            }
        }
        let mut p = self.persistence.lock().unwrap();
        if learned != DeletedPayload::default() {
            let deleted = {
                let mut view = self.write();
                view.apply_deleted(&learned);
                view.deleted_payload()
            };
            if let Some(deleted) = deleted {
                p.write_deleted(&deleted)?;
            }
        }
        for (oseq, _) in gone {
            p.mark_gone(origin, *oseq, *oseq);
        }
        Ok(())
    }

    /// A position in the substrate writer's queue.
    pub fn write_ticket(&self) -> u64 {
        self.writer.ticket()
    }

    /// Block until every write at or below `ticket` is in the log.
    pub fn wait_written(&self, ticket: u64) {
        self.writer.wait_written(ticket);
    }

    /// Cut off from every cluster: what this substrate writes from here on is
    /// detached (§6.7).
    pub fn detach(&self) {
        self.persistence.lock().unwrap().detach();
    }

    /// Whether it holds records written cut off that no rejoin has fixed.
    pub fn holds_detached(&self) -> bool {
        self.persistence
            .lock()
            .unwrap()
            .origination()
            .detached_range()
            .is_some()
    }

    /// Rejoined under generation `gen`, decided by the group: what was written
    /// cut off is fixed to it, after the writer has drained, and minted under
    /// it from here on. The caller reloads.
    pub fn rejoin_at(&self, gen: u64) -> PersistenceResult<()> {
        self.writer.wait_written(self.writer.ticket());
        self.persistence.lock().unwrap().rejoin(&Decided(gen))?;
        Ok(())
    }

    /// The directory the substrate's log is in — where its member keeps its
    /// vote journal (§5.9).
    pub fn substrate_dir(&self) -> PathBuf {
        self.persistence.lock().unwrap().dir().to_path_buf()
    }

    /// The generation this substrate writes under.
    pub fn generation(&self) -> u64 {
        self.persistence.lock().unwrap().origination().gen()
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
        let id = {
            let mut p = self.persistence.lock().unwrap();
            let id = p.mint();
            p.write_custom_object(&obj, id)?;
            p.commit()?;
            id
        };
        self.write().apply_custom_object(obj, id);
        Ok(())
    }
}
