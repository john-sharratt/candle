//! Appending a conversation's file events, and reading them back.

use serde_json::Value;

use super::index::VfsIndex;
use super::payload::{VfsEventPayload, VfsTombstonePayload};
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
    pub kills: Vec<u64>,
}

/// Append `body` to `key` in `repo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VfsAppend {
    pub repo: String,
    pub key: String,
    pub body: Value,
}

impl SubstratePersistence {
    /// Stage `write` for `timeline`. Every record takes the timeline's next
    /// sequence number, events first; returns the sequence numbers the events
    /// took, in order.
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
    pub fn write_vfs(&mut self, timeline: u64, write: &VfsWrite) -> Result<Vec<u64>> {
        let seqs = self.append_vfs_events(timeline, write)?;
        self.append_vfs_tombstones(timeline, write)?;
        Ok(seqs)
    }

    fn append_vfs_tombstones(&mut self, timeline: u64, write: &VfsWrite) -> Result<()> {
        for kill in &write.tombstones {
            let seq = self.vfs_index.next_seq(timeline);
            let payload = VfsTombstonePayload {
                timeline_id: timeline,
                seq,
                repo: kill.repo.clone(),
                key: kill.key.clone(),
                kills: kill.kills.clone(),
            }
            .encode();
            let (segment, offset, size) =
                self.append_record(RecordType::VfsTombstone, 0, timeline, seq, 0, 0, &payload)?;
            let loc = RecordLoc {
                segment,
                offset,
                payload_len: payload.len() as u64,
                record_size: size,
            };
            for killed in self
                .vfs_index
                .record_tombstone(timeline, seq, loc, &kill.kills)
            {
                self.accounting
                    .retire(RecordType::VfsEvent, timeline, killed);
            }
        }
        Ok(())
    }

    fn append_vfs_events(&mut self, timeline: u64, write: &VfsWrite) -> Result<Vec<u64>> {
        let mut seqs = Vec::with_capacity(write.events.len());
        for event in &write.events {
            let seq = self.vfs_index.next_seq(timeline);
            let payload = VfsEventPayload {
                timeline_id: timeline,
                seq,
                repo: event.repo.clone(),
                key: event.key.clone(),
                body: event.body.clone(),
            }
            .encode();
            let (segment, offset, size) =
                self.append_record(RecordType::VfsEvent, 0, timeline, seq, 0, 0, &payload)?;
            self.vfs_index.record_event(
                timeline,
                seq,
                RecordLoc {
                    segment,
                    offset,
                    payload_len: payload.len() as u64,
                    record_size: size,
                },
            );
            seqs.push(seq);
        }
        Ok(seqs)
    }

    /// Every live event of `timeline`, read back, in `(repo, key, seq)`
    /// order — what a resume replays.
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
            out.push(VfsEventPayload::decode(&record.payload)?);
        }
        out.sort_by(|a, b| (&a.repo, &a.key, a.seq).cmp(&(&b.repo, &b.key, b.seq)));
        Ok(out)
    }

    /// Where every live file event and tombstone is.
    pub fn vfs_index(&self) -> &VfsIndex {
        &self.vfs_index
    }
}
