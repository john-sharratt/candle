//! The per-record header: its type, payload length, checksums and identity.
//!
//! Every field besides the type, length and CRC is optional on the wire, so older
//! writers decode cleanly and newer writers can add fields. The identity fields
//! name the writing node and the record's place in that node's sequence. A header
//! with no identity writes none of them, so older records stay byte-identical. A
//! snapshot record with no payload retracts the stream's snapshot.

use super::*;

/// The decoded record header — the wire fields we carry per record.
/// Defaults make every field (except `type` / `payload_len` / `crc`)
/// optional in the JSON, so older writers that omit a field decode
/// cleanly and newer writers can introduce optional fields without
/// breaking older readers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordHeader {
    #[serde(rename = "type")]
    pub record_type: RecordType,
    /// Unpadded payload byte length.
    pub payload_len: u64,
    /// CRC-32 over the payload bytes.
    pub crc: u32,
    /// `Chunk` only — the `KvFormat` tag; 0 for every other record type.
    #[serde(default)]
    pub format: u8,
    /// Owning stream, or 0 when not stream-scoped.
    #[serde(default)]
    pub stream_id: u64,
    /// Position in the stream's local chunk grid (`Chunk`), or
    /// `through_index` (`Commit`); 0 otherwise.
    #[serde(default)]
    pub chunk_index: u64,
    /// Token count of a `Chunk` (`32` sealed, `<32` partial tail).
    #[serde(default)]
    pub token_count: u64,
    /// The node of the substrate that wrote the record — its origin
    /// ([`RecordIdentity`], `docs/replicated_substrate.md` §6.2). Zero on a
    /// record written before identities, and on the header index.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub origin: u64,
    /// The record's place in its origin's sequence, from 1.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub oseq: u64,
    /// The generation its writer stood in (§6.7).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub gen: u64,
    /// The claim or lease epoch it was written under; zero when nothing governs
    /// it.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub epoch: u64,
    /// The hybrid logical clock reading it was written at.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub hlc: u64,
    /// [`ContentHash`] over the whole payload — what the `crc` does not cover:
    /// a `Chunk`'s metadata, and every other payload's content as one value
    /// two copies can be compared by.
    #[serde(default, skip_serializing_if = "content_hash_is_zero")]
    pub digest: ContentHash,
}

pub(super) fn is_zero(v: &u64) -> bool {
    *v == 0
}

pub(super) fn content_hash_is_zero(h: &ContentHash) -> bool {
    *h == ContentHash::ZERO
}

impl RecordHeader {
    /// Every field zero, of the never-written [`RecordType::Unknown`] — the
    /// base a header literal completes with `..RecordHeader::NONE`.
    pub const NONE: RecordHeader = RecordHeader {
        record_type: RecordType::Unknown,
        payload_len: 0,
        crc: 0,
        format: 0,
        stream_id: 0,
        chunk_index: 0,
        token_count: 0,
        origin: 0,
        oseq: 0,
        gen: 0,
        epoch: 0,
        hlc: 0,
        digest: ContentHash::ZERO,
    };

    /// The record's identity.
    pub fn identity(&self) -> RecordIdentity {
        RecordIdentity {
            origin: self.origin,
            oseq: self.oseq,
            gen: self.gen,
            epoch: self.epoch,
            hlc: self.hlc,
        }
    }

    /// This header carrying `id`.
    pub fn with_identity(mut self, id: RecordIdentity) -> RecordHeader {
        self.origin = id.origin;
        self.oseq = id.oseq;
        self.gen = id.gen;
        self.epoch = id.epoch;
        self.hlc = id.hlc;
        self
    }

    /// A `Snapshot` record with no payload: the stream's snapshot is retracted
    /// and nothing for it is live. See [`RecordType::Snapshot`].
    pub fn retracts_snapshot(&self) -> bool {
        self.record_type == RecordType::Snapshot && self.payload_len == 0
    }
}

/// A fully decoded record — header and owned payload bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub header: RecordHeader,
    pub payload: Vec<u8>,
}
