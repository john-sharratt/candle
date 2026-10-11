//! Appending a conversation's file events, and reading them back.

use serde_json::Value;

use super::index::VfsIndex;
use super::payload::{VfsEventId, VfsEventPayload, VfsTombstonePayload};
use crate::persistence::manifest::RecordLoc;
use crate::persistence::record::RecordType;
use crate::persistence::{Result, SubstratePersistence};

/// One set of a conversation's changes to its files, written together:
/// events first, then tombstones — so a batch cut short leaves old events
/// beside new ones, never a tombstone with nothing written in its place.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VfsWrite {
    pub tombstones: Vec<VfsKill>,
    pub events: Vec<VfsAppend>,
}

impl VfsWrite {
    pub fn is_empty(&self) -> bool {
        self.tombstones.is_empty() && self.events.is_empty()
    }
}

/// Kill the events `kills` of `key` in `repo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VfsKill {
    pub repo: String,
    pub key: String,
    pub kills: Vec<VfsEventId>,
}

/// Append `body` to `key` in `repo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VfsAppend {
    pub repo: String,
    pub key: String,
    pub body: Value,
}

impl SubstratePersistence {
    /// Stage `write` for `timeline`. Every record takes this member's next
    /// sequence number for the timeline, events first; returns the names the
    /// events took, in order.
    ///
    /// Staged, not made durable: the persistence thread's group commit makes
    /// it so, with everything else staged beside it, so a tool round never
    /// waits on an `fsync`. A crash before that commit loses the batch's
    /// tail — which the events' order and replay are built to survive
    /// (`docs/zend_vfs_events.md` §5).
    ///
    /// A read-only handle refuses with
    /// [`PersistenceError::ReadOnly`](crate::persistence::PersistenceError::ReadOnly)
    /// and writes nothing.
    pub fn write_vfs(&mut self, timeline: u64, write: &VfsWrite) -> Result<Vec<VfsEventId>> {
        let ids = self.append_vfs_events(timeline, write)?;
        self.append_vfs_tombstones(timeline, write)?;
        Ok(ids)
    }

    /// This member's next file record for `timeline`.
    fn next_vfs_id(&self, timeline: u64) -> VfsEventId {
        let origin = self.node();
        VfsEventId {
            origin,
            seq: self.vfs_index.next_seq(timeline, origin),
        }
    }

    fn append_vfs_tombstones(&mut self, timeline: u64, write: &VfsWrite) -> Result<()> {
        for kill in &write.tombstones {
            let id = self.next_vfs_id(timeline);
            let payload = VfsTombstonePayload {
                timeline_id: timeline,
                origin: id.origin,
                seq: id.seq,
                repo: kill.repo.clone(),
                key: kill.key.clone(),
                kills: kill.kills.clone(),
            }
            .encode();
            let loc = self
                .append_record(
                    RecordType::VfsTombstone,
                    0,
                    timeline,
                    id.seq,
                    0,
                    0,
                    &payload,
                )?
                .loc(payload.len() as u64);
            for killed in self
                .vfs_index
                .record_tombstone(timeline, id, loc, &kill.kills)
            {
                self.accounting
                    .retire(RecordType::VfsEvent, timeline, killed);
            }
        }
        Ok(())
    }

    fn append_vfs_events(&mut self, timeline: u64, write: &VfsWrite) -> Result<Vec<VfsEventId>> {
        let mut ids = Vec::with_capacity(write.events.len());
        for event in &write.events {
            let id = self.next_vfs_id(timeline);
            let payload = VfsEventPayload {
                timeline_id: timeline,
                origin: id.origin,
                seq: id.seq,
                repo: event.repo.clone(),
                key: event.key.clone(),
                body: event.body.clone(),
            }
            .encode();
            let loc = self
                .append_record(RecordType::VfsEvent, 0, timeline, id.seq, 0, 0, &payload)?
                .loc(payload.len() as u64);
            self.vfs_index.record_event(timeline, id, loc);
            ids.push(id);
        }
        Ok(ids)
    }

    /// Every live event of `timeline`, read back, in `(repo, key)` order and
    /// within a key in the order of their records' version stamps — what a
    /// resume replays. One writer's events are in the order it wrote them, and
    /// two writers' interleave by stamp, as their turns do
    /// (`docs/replicated_substrate.md` §6.6).
    ///
    /// Records still staged are not in the file yet, so whatever is staged
    /// is committed first: a conversation built again moments after its last
    /// save reads that save too.
    pub fn vfs_events(&mut self, timeline: u64) -> Result<Vec<VfsEventPayload>> {
        self.commit_if_pending()?;
        let locs: Vec<RecordLoc> = self
            .vfs_index
            .timeline(timeline)
            .map(|tl| tl.events().values().copied().collect())
            .unwrap_or_default();
        let mut out = Vec::with_capacity(locs.len());
        for loc in locs {
            let record = self
                .segments
                .read_record_at(loc.segment, loc.offset, loc.record_size)?;
            let stamp = record.header.identity().stamp();
            out.push((stamp, VfsEventPayload::decode(&record.payload)?));
        }
        out.sort_by(|(sa, a), (sb, b)| {
            (&a.repo, &a.key, sa, a.id()).cmp(&(&b.repo, &b.key, sb, b.id()))
        });
        Ok(out.into_iter().map(|(_, event)| event).collect())
    }

    /// Where every live file event and tombstone is.
    pub fn vfs_index(&self) -> &VfsIndex {
        &self.vfs_index
    }
}
