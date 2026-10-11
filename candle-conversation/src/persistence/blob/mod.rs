//! Bytes named by their SHA-256, as records (`docs/replicated_substrate.md`
//! §10.3): an image a character's portrait names, kept once however many
//! name it, and replicated like any other record.
//!
//! | Module | Concern |
//! |---|---|
//! | [`payload`] | What a blob record carries, and its name |
//! | [`index`] | Where every blob is |
//! | `writes` | Appending a blob, reading one back |

pub mod index;
pub mod payload;
mod writes;

#[cfg(test)]
mod tests;

pub use index::BlobIndex;
pub use payload::{hex, parse_hex, BlobKey, BlobPayload};

use super::manifest::RecordLoc;
use super::record::RecordType;
use super::walker::WalkEntry;

/// Mirror one walked record into the index: the first blob of a hash is
/// recorded, and a second of the same hash — the same bytes again — is left
/// out (the accounting counts it dead by its key). The load, compaction and
/// remote-apply walks run this; the runtime appends keep the index
/// themselves.
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
    );
}
