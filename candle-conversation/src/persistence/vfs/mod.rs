//! A conversation's changes to its repositories' files, as events on its
//! timeline — `docs/zend_vfs_events.md`.
//!
//! | Module | Concern |
//! |---|---|
//! | [`payload`] | What an event and a tombstone carry |
//! | [`index`] | Where every live event and tombstone is |
//! | `writes` | Appending a batch, reading a conversation's events back |

pub mod index;
pub mod payload;
mod writes;

#[cfg(test)]
mod tests;

pub use index::{TimelineVfs, VfsIndex};
pub use payload::{VfsEventPayload, VfsTombstonePayload};
pub use writes::{VfsAppend, VfsKill, VfsWrite};

use std::num::NonZeroU64;

use super::accounting::RecordAccounting;
use super::manifest::RecordLoc;
use super::record::{RecordType, TombstonePayload};
use super::walker::WalkEntry;
use crate::projection::TimelineId;
use crate::substrate::Substrate;

/// The timelines whose file events and tombstones a rewrite of the log
/// carries: registered, not tombstoned, not distilled. Anything else is an
/// orphan — its conversation retired, distilled to its signatures, or never
/// there — and every rewrite leaves it behind. The one rule compaction,
/// maintenance and segment liveness all read, so what is carried and what is
/// counted live never disagree.
pub(super) fn carried<'a>(
    substrate: &Substrate,
    index: &'a VfsIndex,
) -> Vec<(u64, &'a TimelineVfs)> {
    index
        .timelines()
        .into_iter()
        .filter(|(raw, _)| {
            let Some(timeline) = NonZeroU64::new(*raw).map(TimelineId::new) else {
                return false;
            };
            substrate.timeline_entry(timeline).is_some()
                && !substrate.is_tombstoned(timeline)
                && !substrate.distilled_timelines().contains_key(&timeline)
        })
        .collect()
}

/// Mirror one walked record into the index, and count what it kills as dead:
/// an event is recorded unless something already killed it; a tombstone
/// kills the events it names; a timeline `Tombstone` takes all of its
/// timeline's. The load and compaction walks run this; the runtime appends
/// keep the index themselves.
pub(super) fn record_vfs_loc(
    index: &mut VfsIndex,
    accounting: &mut RecordAccounting,
    entry: &WalkEntry,
) {
    let h = &entry.record.header;
    let loc = RecordLoc {
        segment: entry.segment,
        offset: entry.offset,
        payload_len: h.payload_len,
        record_size: entry.size,
    };
    match h.record_type {
        RecordType::VfsEvent => {
            if !index.record_event(h.stream_id, h.chunk_index, loc) {
                accounting.retire(RecordType::VfsEvent, h.stream_id, h.chunk_index);
            }
        }
        RecordType::VfsTombstone => match VfsTombstonePayload::decode(&entry.record.payload) {
            Ok(p) => {
                for seq in index.record_tombstone(h.stream_id, h.chunk_index, loc, &p.kills) {
                    accounting.retire(RecordType::VfsEvent, h.stream_id, seq);
                }
            }
            Err(e) => tracing::warn!(
                timeline = h.stream_id,
                seq = h.chunk_index,
                "a file-event tombstone could not be read, so the events it killed read as \
                 live: {e}"
            ),
        },
        RecordType::Tombstone => {
            if let Ok(p) = TombstonePayload::decode(&entry.record.payload) {
                if p.turn_index.is_none() {
                    retire_timeline(index, accounting, p.timeline_id);
                }
            }
        }
        _ => {}
    }
}

/// `timeline` is tombstoned: its events and tombstones leave the index, and
/// their bytes count as dead.
pub(super) fn retire_timeline(
    index: &mut VfsIndex,
    accounting: &mut RecordAccounting,
    timeline: u64,
) {
    let (events, tombstones) = index.drop_timeline(timeline);
    for seq in events {
        accounting.retire(RecordType::VfsEvent, timeline, seq);
    }
    for seq in tombstones {
        accounting.retire(RecordType::VfsTombstone, timeline, seq);
    }
}
