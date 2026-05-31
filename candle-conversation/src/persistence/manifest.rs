//! The in-RAM manifest — the index a walk reconstructs.
//!
//! The manifest resolves the live state of the log under
//! **last-writer-wins**: ingesting records in append order, a later
//! record for the same key simply overwrites an earlier one. It
//! indexes the stream DAG, every chunk's location, and the latest
//! singleton records, and it serialises into a `Checkpoint` record so
//! a restart need not re-walk the whole log.
//!
//! All structured metadata payloads in this module are encoded as
//! UTF-8 JSON via serde, with `#[serde(default)]` on every field so
//! older writers can omit fields and newer writers can add new ones
//! without breaking either side. This is the field-level forward
//! compatibility lever; the type-level lever lives in `record.rs`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::log_file::LogSource;
use super::record::RecordType;
use super::streams::{StreamDecl, StreamId};
use super::walker::{self, WalkEntry, WalkOutcome};
use super::{PersistenceError, Result};

/// Location of a whole record in the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordLoc {
    pub offset: u64,
    pub payload_len: u64,
    /// Padded on-disk size of the record (header + payload + sector
    /// padding). Captured at write time and at walk-time; with the
    /// NDJSON header framing the padded size depends on the header's
    /// serialized length and can't be recomputed from `payload_len`
    /// alone. Always populated — load-bearing for the batched cold-read
    /// path (one read per stripe, sized exactly from the manifest).
    pub record_size: u64,
}

/// Location and shape of one chunk record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkLoc {
    pub offset: u64,
    pub payload_len: u64,
    /// Padded on-disk size of the record — see [`RecordLoc::record_size`].
    pub record_size: u64,
    pub token_count: u64,
    pub format: u8,
}

/// The indexed state of one stream.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StreamEntry {
    /// The decoded stream declaration, once a `StreamDecl` record is seen.
    #[serde(default)]
    pub decl: Option<StreamDecl>,
    /// Live chunk locations by local chunk index — last-writer-wins.
    /// Serialised as a `Vec<(idx, loc)>` so the JSON map-key constraint
    /// (keys are strings) doesn't force stringified u64s on disk.
    #[serde(default, with = "u64_btreemap_as_pairs")]
    pub chunks: BTreeMap<u64, ChunkLoc>,
    /// Latest `Tokens` record for the stream.
    #[serde(default)]
    pub tokens: Option<RecordLoc>,
    /// Latest `Signatures` record for the stream.
    #[serde(default)]
    pub signatures: Option<RecordLoc>,
    /// Highest chunk index the stream is durably committed through.
    #[serde(default)]
    pub committed_through: Option<u64>,
}

/// The whole-log index.
///
/// Serialised as the `Checkpoint` record payload, in JSON, so any
/// future field can be added or removed without breaking older
/// readers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// Latest `ModelSpec` record.
    #[serde(default)]
    pub model_spec: Option<RecordLoc>,
    /// Latest `Template` record.
    #[serde(default)]
    pub template: Option<RecordLoc>,
    /// Latest `Tokenizer` record.
    #[serde(default)]
    pub tokenizer: Option<RecordLoc>,
    /// Per-stream index, ordered by id for deterministic serialisation.
    /// Serialised as `Vec<(id, entry)>` so JSON map-key string
    /// stringification doesn't force StreamId-as-string on disk.
    #[serde(default, with = "stream_btreemap_as_pairs")]
    pub streams: BTreeMap<StreamId, StreamEntry>,
    /// Per-timeline conversation metadata — the daemon's client-side
    /// `conv_id` string paired with the sidebar title. Keyed by
    /// `timeline_id`, decoded inline. Written as `RecordType::Label`
    /// records, **last-write-wins** on replay so the conv_id can be
    /// established immediately at first submit and the title can be
    /// filled in later by the titler.
    #[serde(default, with = "u64_btreemap_as_pairs")]
    pub labels: BTreeMap<u64, ConvMeta>,
    /// Per-timeline lifecycle state. Keyed by `timeline_id`. Written
    /// as `RecordType::ConvState` records, **last-write-wins** on
    /// replay so toggling archive↔unarchive each appends a small
    /// record and the latest wins.
    #[serde(default, with = "u64_btreemap_as_pairs")]
    pub conv_states: BTreeMap<u64, ConvState>,
    /// Offset of the most recent `Checkpoint` record seen.
    #[serde(default)]
    pub last_checkpoint_offset: Option<u64>,
}

/// Per-timeline conversation metadata persisted in `RecordType::Label`.
/// `conv_id` is the client-supplied identifier (e.g. the frontend's
/// `Date.now()` string) used as the sidebar's stable id; `label` is
/// the human-readable title, possibly empty during the brief window
/// between first-submit and titler-completion.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvMeta {
    #[serde(default)]
    pub conv_id: String,
    #[serde(default)]
    pub label: String,
}

/// Per-timeline lifecycle flags persisted in `RecordType::ConvState`.
/// Today: just the `archived` flag (hide-from-sidebar without losing
/// the conversation). Future fields slot in alongside — serde's
/// `#[serde(default)]` covers both omit-on-write and ignore-on-read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvState {
    #[serde(default)]
    pub archived: bool,
}

/// Wire-format `Label` payload: `{timeline_id, conv_id, label}`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct LabelPayload {
    #[serde(default)]
    timeline_id: u64,
    #[serde(default)]
    conv_id: String,
    #[serde(default)]
    label: String,
}

/// Wire-format `ConvState` payload: `{timeline_id, archived}`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct ConvStatePayload {
    #[serde(default)]
    timeline_id: u64,
    #[serde(default)]
    archived: bool,
}

impl Manifest {
    /// An empty manifest.
    pub fn new() -> Manifest {
        Manifest::default()
    }

    /// Apply one walked record, last-writer-wins.
    pub fn ingest(&mut self, entry: &WalkEntry) -> Result<()> {
        let h = &entry.record.header;
        let loc = RecordLoc {
            offset: entry.offset,
            payload_len: h.payload_len,
            record_size: entry.size,
        };
        match h.record_type {
            RecordType::ModelSpec => self.model_spec = Some(loc),
            RecordType::Template => self.template = Some(loc),
            RecordType::Tokenizer => self.tokenizer = Some(loc),
            RecordType::StreamDecl => {
                let decl = StreamDecl::decode(&entry.record.payload)?;
                self.streams.entry(StreamId(h.stream_id)).or_default().decl = Some(decl);
            }
            RecordType::Chunk => {
                let e = self.streams.entry(StreamId(h.stream_id)).or_default();
                e.chunks.insert(
                    h.chunk_index,
                    ChunkLoc {
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                        token_count: h.token_count,
                        format: h.format,
                    },
                );
            }
            RecordType::Tokens => {
                self.streams
                    .entry(StreamId(h.stream_id))
                    .or_default()
                    .tokens = Some(loc);
            }
            RecordType::Signatures => {
                self.streams
                    .entry(StreamId(h.stream_id))
                    .or_default()
                    .signatures = Some(loc);
            }
            RecordType::Commit => {
                self.streams
                    .entry(StreamId(h.stream_id))
                    .or_default()
                    .committed_through = Some(h.chunk_index);
            }
            RecordType::Checkpoint => {
                self.last_checkpoint_offset = Some(entry.offset);
            }
            RecordType::Label => {
                let (timeline_id, meta) = decode_label_payload(&entry.record.payload)?;
                self.labels.insert(timeline_id, meta);
            }
            RecordType::ConvState => {
                let (timeline_id, state) = decode_conv_state_payload(&entry.record.payload)?;
                self.conv_states.insert(timeline_id, state);
            }
            RecordType::Unknown => {
                // Skipped by the walker before reaching here; the
                // arm is present for exhaustiveness.
            }
        }
        Ok(())
    }

    /// Build a manifest by walking the log from `start`.
    pub fn build_from_walk(src: &mut dyn LogSource, start: u64) -> Result<(Manifest, WalkOutcome)> {
        let mut manifest = Manifest::new();
        let mut err: Option<PersistenceError> = None;
        let outcome = walker::walk(src, start, |entry| {
            if err.is_none() {
                if let Err(e) = manifest.ingest(entry) {
                    err = Some(e);
                }
            }
        })?;
        if let Some(e) = err {
            return Err(e);
        }
        Ok((manifest, outcome))
    }

    /// Total live chunk count across all streams.
    pub fn live_chunk_count(&self) -> usize {
        self.streams.values().map(|s| s.chunks.len()).sum()
    }

    /// Serialise to the `Checkpoint` record payload bytes — UTF-8
    /// JSON. Forward-compatible: adding fields is a no-op for older
    /// readers (they ignore unknown keys), removing fields decodes
    /// to defaults.
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("Manifest JSON encoding is infallible")
    }

    /// Reconstruct from `Checkpoint` record payload bytes.
    pub fn decode(payload: &[u8]) -> Result<Manifest> {
        serde_json::from_slice(payload)
            .map_err(|e| PersistenceError::Corrupt(format!("Manifest JSON decode: {e}")))
    }
}

/// Encode a `Label` record's payload — JSON.
pub fn encode_label_payload(timeline_id: u64, conv_id: &str, label: &str) -> Vec<u8> {
    let p = LabelPayload {
        timeline_id,
        conv_id: conv_id.to_string(),
        label: label.to_string(),
    };
    serde_json::to_vec(&p).expect("Label payload JSON encoding is infallible")
}

/// Decode a `Label` record's payload — see [`encode_label_payload`].
/// Returns `(timeline_id, ConvMeta)`.
pub fn decode_label_payload(payload: &[u8]) -> Result<(u64, ConvMeta)> {
    let p: LabelPayload = serde_json::from_slice(payload)
        .map_err(|e| PersistenceError::Corrupt(format!("Label payload JSON decode: {e}")))?;
    Ok((
        p.timeline_id,
        ConvMeta {
            conv_id: p.conv_id,
            label: p.label,
        },
    ))
}

/// Encode a `ConvState` record's payload — JSON.
pub fn encode_conv_state_payload(timeline_id: u64, state: ConvState) -> Vec<u8> {
    let p = ConvStatePayload {
        timeline_id,
        archived: state.archived,
    };
    serde_json::to_vec(&p).expect("ConvState payload JSON encoding is infallible")
}

/// Decode a `ConvState` record's payload — see [`encode_conv_state_payload`].
/// Returns `(timeline_id, ConvState)`. Unknown JSON keys are ignored
/// so a future flag (`pinned`, `unread`, …) doesn't fail-load on an
/// older daemon — that's the field-level forward-compat property.
pub fn decode_conv_state_payload(payload: &[u8]) -> Result<(u64, ConvState)> {
    let p: ConvStatePayload = serde_json::from_slice(payload)
        .map_err(|e| PersistenceError::Corrupt(format!("ConvState payload JSON decode: {e}")))?;
    Ok((
        p.timeline_id,
        ConvState {
            archived: p.archived,
        },
    ))
}

/// Serde shim: serialise a `BTreeMap<u64, V>` as a `Vec<(u64, V)>`
/// pair list, since JSON object keys are strings. Round-trips
/// preserve ordering (BTreeMap iterates by key).
mod u64_btreemap_as_pairs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<V, S>(map: &BTreeMap<u64, V>, ser: S) -> Result<S::Ok, S::Error>
    where
        V: Serialize,
        S: Serializer,
    {
        let pairs: Vec<(u64, &V)> = map.iter().map(|(k, v)| (*k, v)).collect();
        pairs.serialize(ser)
    }

    pub fn deserialize<'de, V, D>(de: D) -> Result<BTreeMap<u64, V>, D::Error>
    where
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let pairs: Vec<(u64, V)> = Vec::deserialize(de)?;
        Ok(pairs.into_iter().collect())
    }
}

/// Same shape as [`u64_btreemap_as_pairs`] but for `StreamId` keys.
mod stream_btreemap_as_pairs {
    use super::StreamId;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<V, S>(map: &BTreeMap<StreamId, V>, ser: S) -> Result<S::Ok, S::Error>
    where
        V: Serialize,
        S: Serializer,
    {
        let pairs: Vec<(StreamId, &V)> = map.iter().map(|(k, v)| (*k, v)).collect();
        pairs.serialize(ser)
    }

    pub fn deserialize<'de, V, D>(de: D) -> Result<BTreeMap<StreamId, V>, D::Error>
    where
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let pairs: Vec<(StreamId, V)> = Vec::deserialize(de)?;
        Ok(pairs.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::log_file::{MemLog, SUPERBLOCK_SIZE};
    use crate::persistence::record::{encode_record, RecordHeader};
    use crate::persistence::streams::{ContentAddress, SectionDecl};

    fn record(rt: RecordType, stream_id: u64, chunk_index: u64, payload: &[u8]) -> Vec<u8> {
        encode_record(
            &RecordHeader {
                record_type: rt,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0, // overwritten by encode_record
                stream_id,
                chunk_index,
                token_count: if rt == RecordType::Chunk { 32 } else { 0 },
            },
            payload,
        )
    }

    #[test]
    fn manifest_indexes_a_walk() {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress::default(),
            debug_name: "sec".to_string(),
        });
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::ModelSpec, 0, 0, b"model-v1"));
        blob.extend_from_slice(&record(RecordType::StreamDecl, 7, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 0, b"chunk0"));
        blob.extend_from_slice(&record(RecordType::Chunk, 7, 1, b"chunk1"));
        blob.extend_from_slice(&record(RecordType::Commit, 7, 1, b""));
        let mut mem = MemLog::with_records(&blob);

        let (manifest, outcome) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert!(!outcome.torn);
        assert!(manifest.model_spec.is_some());
        let s = &manifest.streams[&StreamId(7)];
        assert_eq!(s.chunks.len(), 2);
        assert_eq!(s.committed_through, Some(1));
        assert_eq!(s.decl, Some(decl));
        assert_eq!(manifest.live_chunk_count(), 2);
    }

    #[test]
    fn last_writer_wins_supersedes_a_chunk() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"twenty-token-partial"));
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"sealed-final-version"));
        let mut mem = MemLog::with_records(&blob);

        let (manifest, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let loc = manifest.streams[&StreamId(1)].chunks[&0];
        // The winner is the second (later) record — at the higher offset.
        assert_eq!(loc.payload_len, "sealed-final-version".len() as u64);
        assert!(loc.offset > SUPERBLOCK_SIZE);
    }

    #[test]
    fn latest_singleton_wins() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::ModelSpec, 0, 0, b"v1"));
        blob.extend_from_slice(&record(RecordType::ModelSpec, 0, 0, b"v2-newer"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(
            manifest.model_spec.unwrap().payload_len,
            "v2-newer".len() as u64
        );
    }

    #[test]
    fn manifest_encode_decode_roundtrip() {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress::default(),
            debug_name: "roundtrip".to_string(),
        });
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Template, 0, 0, b"template-blob"));
        blob.extend_from_slice(&record(RecordType::StreamDecl, 3, 0, &decl.encode()));
        blob.extend_from_slice(&record(RecordType::Chunk, 3, 0, b"c0"));
        blob.extend_from_slice(&record(RecordType::Tokens, 3, 0, b"tok"));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();

        let encoded = manifest.encode();
        let decoded = Manifest::decode(&encoded).unwrap();
        assert_eq!(decoded, manifest);
    }

    #[test]
    fn empty_manifest_roundtrips() {
        let m = Manifest::new();
        assert_eq!(Manifest::decode(&m.encode()).unwrap(), m);
    }

    /// `ConvState` payload encodes / decodes correctly with the
    /// `archived` flag both set and clear.
    #[test]
    fn conv_state_payload_round_trip() {
        let archived = encode_conv_state_payload(42, ConvState { archived: true });
        let (tl, st) = decode_conv_state_payload(&archived).unwrap();
        assert_eq!(tl, 42);
        assert!(st.archived);

        let unarchived = encode_conv_state_payload(7, ConvState { archived: false });
        let (tl, st) = decode_conv_state_payload(&unarchived).unwrap();
        assert_eq!(tl, 7);
        assert!(!st.archived);
    }

    /// Multiple `ConvState` records for the same timeline collapse
    /// to the latest one in the manifest — last-writer-wins.
    #[test]
    fn conv_state_last_writer_wins_in_manifest() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::ConvState,
            0,
            0,
            &encode_conv_state_payload(99, ConvState { archived: true }),
        ));
        blob.extend_from_slice(&record(
            RecordType::ConvState,
            0,
            0,
            &encode_conv_state_payload(99, ConvState { archived: false }),
        ));
        blob.extend_from_slice(&record(
            RecordType::ConvState,
            0,
            0,
            &encode_conv_state_payload(99, ConvState { archived: true }),
        ));
        let mut mem = MemLog::with_records(&blob);
        let (manifest, _) = Manifest::build_from_walk(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert_eq!(manifest.conv_states.len(), 1);
        assert!(manifest.conv_states.get(&99).unwrap().archived);
    }

    /// ConvState entries survive a checkpoint encode/decode cycle.
    #[test]
    fn conv_state_survives_checkpoint_roundtrip() {
        let mut m = Manifest::new();
        m.conv_states.insert(1, ConvState { archived: true });
        m.conv_states.insert(2, ConvState { archived: false });
        let bytes = m.encode();
        let decoded = Manifest::decode(&bytes).unwrap();
        assert_eq!(decoded, m);
    }

    /// A future-version writer adds a JSON field we don't model. The
    /// older reader must ignore it and decode the rest normally —
    /// the field-level forward-compatibility property.
    #[test]
    fn conv_state_payload_ignores_unknown_fields() {
        let payload =
            br#"{"timeline_id":12,"archived":true,"pinned":true,"reminder":"someday"}"#.to_vec();
        let (tl, st) = decode_conv_state_payload(&payload).unwrap();
        assert_eq!(tl, 12);
        assert!(st.archived);
    }

    /// An older writer omitted a field (`archived` not present). The
    /// newer reader must use the default rather than fail.
    #[test]
    fn conv_state_payload_defaults_missing_fields() {
        let payload = br#"{"timeline_id":5}"#.to_vec();
        let (tl, st) = decode_conv_state_payload(&payload).unwrap();
        assert_eq!(tl, 5);
        assert!(!st.archived, "missing archived must default to false");
    }

    /// Same forward-compat properties for the Label payload.
    #[test]
    fn label_payload_ignores_unknown_fields() {
        let payload =
            br#"{"timeline_id":3,"conv_id":"abc","label":"My chat","future_field":"x"}"#.to_vec();
        let (tl, meta) = decode_label_payload(&payload).unwrap();
        assert_eq!(tl, 3);
        assert_eq!(meta.conv_id, "abc");
        assert_eq!(meta.label, "My chat");
    }

    #[test]
    fn label_payload_defaults_missing_fields() {
        // No conv_id, no label.
        let payload = br#"{"timeline_id":9}"#.to_vec();
        let (tl, meta) = decode_label_payload(&payload).unwrap();
        assert_eq!(tl, 9);
        assert!(meta.conv_id.is_empty());
        assert!(meta.label.is_empty());
    }

    /// The Manifest itself must tolerate added/removed top-level
    /// JSON fields — a future field doesn't fail-load, a missing
    /// field decodes to its `Default` value.
    #[test]
    fn manifest_ignores_unknown_top_level_fields() {
        let payload = br#"{"streams":[],"labels":[],"conv_states":[],"future_top_level":42}"#;
        let decoded = Manifest::decode(payload).unwrap();
        assert!(decoded.streams.is_empty());
    }
}
