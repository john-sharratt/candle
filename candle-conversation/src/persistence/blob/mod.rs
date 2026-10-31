//! Bytes named by their SHA-256, as records (`docs/replicated_substrate.md`
//! §10.3): an image a character's portrait names, kept once however many
//! name it, and replicated like any other record.
//!
//! **A blob can be retired.** Its tombstone is the custom object
//! [`BlobKey::tombstone_key`] marked ceased, which replicates, wins by stamp
//! and expires like every keyed tombstone (§6.8). It kills each record of the
//! blob it outranks — so the same bytes written again later live — and the
//! index forgets a blob it kills, which is what compaction, maintenance and
//! the liveness count all carry and count by.
//!
//! | Module | Concern |
//! |---|---|
//! | [`payload`] | What a blob record carries, its name, and its tombstone's key |
//! | [`index`] | Where every live blob is |
//! | `writes` | Appending a blob, reading one back, retiring one |

pub mod index;
pub mod payload;
mod writes;

pub(super) use writes::blob_bytes;

#[cfg(test)]
mod tests;

pub use index::BlobIndex;
pub use payload::{hex, parse_hex, BlobKey, BlobPayload};

use super::accounting::RecordAccounting;
use super::manifest::RecordLoc;
use super::record::{CustomObjectPayload, RecordType};
use super::walker::WalkEntry;
use crate::substrate::Substrate;

/// Mirror one walked record into the index: a blob record is held when it
/// outranks the one its hash holds ([`BlobIndex::record`]). The load,
/// compaction and remote-apply walks run this; the runtime appends keep the
/// index themselves.
pub(super) fn record_blob_loc(index: &mut BlobIndex, entry: &WalkEntry) {
    let h = &entry.record.header;
    if h.record_type != RecordType::Blob {
        return;
    }
    let loc = RecordLoc {
        segment: entry.segment,
        offset: entry.offset,
        payload_len: h.payload_len,
        record_size: entry.size,
    };
    index.record(
        BlobKey {
            hi: h.stream_id,
            lo: h.chunk_index,
        },
        loc,
        h.identity(),
    );
}

/// The blob `obj` retires, when it is a blob's tombstone.
pub fn retired_blob(obj: &CustomObjectPayload) -> Option<BlobKey> {
    obj.ceased
        .then(|| BlobKey::of_tombstone_key(&obj.key))
        .flatten()
}

/// The blob a walked record bears on: a blob record's own, or the blob a
/// blob tombstone retires.
pub(super) fn blob_of(entry: &WalkEntry) -> Option<BlobKey> {
    let h = &entry.record.header;
    match h.record_type {
        RecordType::Blob => Some(BlobKey {
            hi: h.stream_id,
            lo: h.chunk_index,
        }),
        RecordType::CustomObject => CustomObjectPayload::decode(&entry.record.payload)
            .ok()
            .and_then(|obj| retired_blob(&obj)),
        _ => None,
    }
}

/// Forget blob `key` when the tombstone standing on it outranks the record
/// the index holds, and count that record dead.
pub(super) fn forget_if_retired(
    index: &mut BlobIndex,
    accounting: &mut RecordAccounting,
    substrate: &Substrate,
    key: BlobKey,
) {
    let (Some(held), Some(tombstone)) = (index.identity(key), substrate.blob_tombstone(key)) else {
        return;
    };
    if tombstone.supersedes(&held) {
        index.forget(key);
        accounting.retire_blob(key);
    }
}

/// [`forget_if_retired`] for every blob a tombstone stands on — after an
/// open, whose walk met each blob and its tombstone in whatever order the log
/// holds them.
pub(super) fn forget_retired(
    index: &mut BlobIndex,
    accounting: &mut RecordAccounting,
    substrate: &Substrate,
) {
    for key in substrate.blob_tombstones() {
        forget_if_retired(index, accounting, substrate, key);
    }
}
