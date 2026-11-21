//! The item a compaction stages into the new log: a verbatim copy of a source
//! record, or a record encoded from memory.
//!
//! A `Raw` item is staged byte for byte from its source segment, with a header
//! synthesised from the index, so the record is neither decoded nor re-checked.
//! A `Synth` item carries a freshly encoded payload, or shares the substrate's
//! own bytes for large blobs, so the rewrite does not hold a second copy of
//! them. The write buffer is flushed at a fixed size, so the whole new log is
//! never resident at once. The singleton and housekeeping helpers build the
//! headers of the items they describe.

use super::*;

/// Bytes staged into the compacted log before an incremental flush to disk —
/// bounds the write buffer so a whole-store rewrite never holds the entire new
/// log in RAM at once.
pub(super) const COMPACT_FLUSH_BYTES: usize = 256 * 1024 * 1024;

/// One record destined for the compacted log. `Raw` records (`Chunk` / `Tokens`
/// / the singletons) are staged **verbatim** from their source segment — no
/// decode, no CRC re-verify, no re-encode — via coalesced stripe reads; their
/// header is synthesized from the substrate/manifest index (never read from
/// disk) for the header-index digest + singleton manifest. `Synth` records are
/// the resident metadata, encoded fresh from in-RAM state — or, for the blobs
/// the substrate already holds shared (index pages, wide-Q signatures,
/// projection events), the substrate's own bytes. Those run to a megabyte a
/// turn and the whole live list exists before the first byte is written, so a
/// copy here is a second image of every live turn's metadata held for the
/// length of the rewrite.
pub enum CompactItem {
    Raw {
        header: RecordHeader,
        segment: SegmentId,
        offset: u64,
        record_size: u64,
    },
    Synth {
        header: RecordHeader,
        payload: Arc<[u8]>,
    },
}

impl CompactItem {
    pub(super) fn raw(
        header: RecordHeader,
        segment: SegmentId,
        offset: u64,
        record_size: u64,
    ) -> CompactItem {
        CompactItem::Raw {
            header,
            segment,
            offset,
            record_size,
        }
    }

    pub(super) fn synth(header: RecordHeader, payload: impl Into<Arc<[u8]>>) -> CompactItem {
        CompactItem::Synth {
            header,
            payload: payload.into(),
        }
    }

    /// The record's header — synthesized for `Raw`, authored for `Synth`.
    pub fn header(&self) -> &RecordHeader {
        match self {
            CompactItem::Raw { header, .. } | CompactItem::Synth { header, .. } => header,
        }
    }

    /// A `Synth` item's payload, or `None` for a `Raw` (read-back) item.
    #[cfg(test)]
    pub fn synth_payload(&self) -> Option<&[u8]> {
        match self {
            CompactItem::Synth { payload, .. } => Some(&payload[..]),
            CompactItem::Raw { .. } => None,
        }
    }
}

/// Header for a singleton (`ModelKey` / `Template` / `Tokenizer`) staged
/// verbatim — `payload_len` from its manifest location, every other field zero.
pub(super) fn singleton_header(rt: RecordType, payload_len: u64) -> RecordHeader {
    RecordHeader {
        record_type: rt,
        format: 0,
        payload_len,
        crc: 0,
        stream_id: 0,
        chunk_index: 0,
        token_count: 0,
        ..RecordHeader::NONE
    }
}

/// `(offset, record_size)` of a `Raw` item — used by tests to read a chosen
/// source location back and confirm which record was selected.
#[cfg(test)]
pub(super) fn raw_loc(it: &CompactItem) -> (u64, u64) {
    match it {
        CompactItem::Raw {
            offset,
            record_size,
            ..
        } => (*offset, *record_size),
        CompactItem::Synth { .. } => unreachable!("raw_loc on a Synth item"),
    }
}

/// The records a member keeps of its own and never replicates, which a rewrite
/// carries whole: what it has applied of other origins and deferred — the
/// records it covers that a rewrite drops are known gone from then on — and the
/// generations fixed for what it wrote cut off (§6.7).
#[derive(Clone, Debug, Default)]
pub struct Housekeeping {
    pub replica: ReplicaPayload,
    pub rebases: RebaseMap,
}

/// A housekeeping record — no identity — holding `payload`.
pub(super) fn local_record(record_type: RecordType, payload: Vec<u8>) -> CompactItem {
    CompactItem::synth(
        RecordHeader {
            record_type,
            payload_len: payload.len() as u64,
            ..RecordHeader::NONE
        },
        payload,
    )
}
