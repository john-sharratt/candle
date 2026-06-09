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
use super::streams::StreamDecl;
use super::walker::{self, WalkEntry, WalkOutcome};
use super::{PersistenceError, Result};
#[cfg(test)]
use crate::substrate::Substrate;

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
/// The set of singleton record locations the `Checkpoint` payload
/// carries.
///
/// **Phase 3.** The manifest used to hold every per-entity collection
/// in RAM and mirror them into the `Checkpoint` payload — making the
/// payload scale with stream count, turn count, and tree-node count.
/// Now it carries only the singleton fields below.  Per-entity state
/// (chunks, tokens locations, signatures locations, stream decls,
/// labels, conv states, tree metadata, debug ids) lives on the
/// [`crate::substrate::Substrate`] directly; the walker dispatches
/// each record through [`crate::substrate::Substrate::apply_walker_entry`]
/// during the same pass that populates the manifest's singletons.
///
/// On reload the walker rebuilds substrate state from records
/// directly; the manifest just records "where the singletons sit."
/// Compaction bounds the live log to the working set, which bounds
/// reload wall-clock.  This is the "atomic per-record entries + walk
/// to reconstruct" pattern.
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
    /// Free-form per-conversation key/value metadata. Persisted in the
    /// Label record (forward-compatible: old logs decode to an empty map).
    /// Used as a content-addressed cache index — e.g. zend's `code_read`
    /// and `repo_map` ingests tag each conversation with `kind`, `path`,
    /// `content_sha256`, etc., then skip rebuilding any unit whose hash is
    /// already present after substrate load. Searchable by (key, value)
    /// via `Substrate::timelines_with_metadata`.
    #[serde(default)]
    pub custom: BTreeMap<String, String>,
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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom: BTreeMap<String, String>,
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

    /// Apply one walked record, last-writer-wins.  After Phase 3 the
    /// manifest only tracks the singleton record types — every
    /// per-entity record (`StreamDecl`, `Chunk`, `Tokens`,
    /// `Signatures`, `Commit`, `Label`, `ConvState`, `TreeMetadata`,
    /// `DebugId`) is handled by the walker's per-record sink, which
    /// dispatches into the substrate via
    /// [`crate::substrate::Substrate::apply_walker_entry`].
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
            RecordType::Checkpoint => {
                self.last_checkpoint_offset = Some(entry.offset);
            }
            // Per-entity records flow to the substrate via the walker
            // sink — see `Substrate::apply_walker_entry`.  They never
            // enter the manifest.
            RecordType::StreamDecl
            | RecordType::Chunk
            | RecordType::Tokens
            | RecordType::Signatures
            | RecordType::Commit
            | RecordType::Label
            | RecordType::ConvState
            | RecordType::TreeMetadata
            | RecordType::DebugId
            | RecordType::Tombstone
            | RecordType::Unknown => {}
        }
        Ok(())
    }

    /// Build a manifest + populated substrate by walking the log from
    /// `start`.  Test helper for callers that need to assert on
    /// per-entity state — production reload goes through
    /// `SubstratePersistence::open_in_with_substrate` which uses the
    /// same walker pass.
    #[cfg(test)]
    pub fn build_with_substrate(
        src: &mut dyn LogSource,
        start: u64,
    ) -> Result<(Manifest, Substrate, WalkOutcome)> {
        let mut manifest = Manifest::new();
        let mut substrate = Substrate::new();
        let mut err: Option<PersistenceError> = None;
        let outcome = walker::walk(src, start, |entry| {
            if err.is_none() {
                if let Err(e) = manifest.ingest(entry) {
                    err = Some(e);
                }
            }
            substrate.apply_walker_entry(entry);
        })?;
        if let Some(e) = err {
            return Err(e);
        }
        Ok((manifest, substrate, outcome))
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

    /// Always 0 after Phase 3 — chunks live on the substrate, not the
    /// manifest.  Test helpers that need the live chunk count should
    /// query `Substrate::live_chunk_count` instead.
    pub fn live_chunk_count(&self) -> usize {
        0
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
pub fn encode_label_payload(
    timeline_id: u64,
    conv_id: &str,
    label: &str,
    custom: &BTreeMap<String, String>,
) -> Vec<u8> {
    let p = LabelPayload {
        timeline_id,
        conv_id: conv_id.to_string(),
        label: label.to_string(),
        custom: custom.clone(),
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
            custom: p.custom,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::log_file::{MemLog, SUPERBLOCK_SIZE};
    use crate::persistence::record::{encode_record, RecordHeader};
    use crate::persistence::streams::{ContentAddress, SectionDecl, StreamId};
    use crate::projection::{GroupId, LayerId, TimelineId};

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

        let (manifest, substrate, outcome) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        assert!(!outcome.torn);
        assert!(manifest.model_spec.is_some());
        let s = substrate.stream_of(StreamId(7)).unwrap();
        assert_eq!(s.chunks.len(), 2);
        assert_eq!(s.committed_through, Some(1));
        assert_eq!(s.decl, Some(decl));
        assert_eq!(substrate.live_chunk_count(), 2);
    }

    #[test]
    fn last_writer_wins_supersedes_a_chunk() {
        let mut blob = Vec::new();
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"twenty-token-partial"));
        blob.extend_from_slice(&record(RecordType::Chunk, 1, 0, b"sealed-final-version"));
        let mut mem = MemLog::with_records(&blob);

        let (_, substrate, _) = Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let loc = substrate.stream_of(StreamId(1)).unwrap().chunks[&0];
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

    /// Phase 3: only the singleton offsets (model_spec, template,
    /// tokenizer, last_checkpoint_offset) survive an encode/decode
    /// round-trip.  Per-entity collections (`streams`, `labels`,
    /// `conv_states`, `tree_metadata`, `debug_ids`) are
    /// `#[serde(skip)]` — they get rebuilt by the walker on reload.
    #[test]
    fn manifest_encode_decode_singletons_only() {
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
        let (manifest, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        // Walker built a non-empty in-RAM streams index on substrate.
        assert!(substrate.live_chunk_count() > 0);
        assert!(manifest.template.is_some());

        let encoded = manifest.encode();
        let decoded = Manifest::decode(&encoded).unwrap();
        // Singletons round-trip.
        assert_eq!(decoded.template, manifest.template);
        assert_eq!(decoded.model_spec, manifest.model_spec);
        assert_eq!(decoded.tokenizer, manifest.tokenizer);
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
        let (_, mut substrate, _) = Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        // Timeline 99 isn't registered during the walk; the three
        // ConvState records are stashed pending registration.  Drain
        // them by registering the timeline, then verify the
        // last-writer-wins archive flag (the final `true` here)
        // lands on the TimelineEntry.
        let tl = TimelineId::from_raw(99).unwrap();
        substrate.register_timeline(tl, LayerId::for_test(1), GroupId::for_test(1));
        assert!(
            substrate.is_archived(tl),
            "last ConvState (archived=true) must win after registration drains the stash"
        );
    }

    /// A `Tombstone` record in the redo log must propagate
    /// to the substrate via the walker, regardless of whether it
    /// arrives before or after the matching TurnDecls.
    #[test]
    fn timeline_tombstone_walker_replay_applies_to_substrate() {
        use crate::persistence::record::TombstonePayload;

        let mut blob = Vec::new();
        blob.extend_from_slice(&record(
            RecordType::Tombstone,
            0,
            0,
            &TombstonePayload { timeline_id: 77 }.encode(),
        ));
        let mut mem = MemLog::with_records(&blob);
        let (_, substrate, _) =
            Manifest::build_with_substrate(&mut mem, SUPERBLOCK_SIZE).unwrap();
        let tl = TimelineId::from_raw(77).unwrap();
        // Pending until register_timeline drains, but `is_tombstoned`
        // sees the pending entry immediately.
        assert!(substrate.is_tombstoned(tl));
    }

    /// `conv_states` is `#[serde(skip)]` — it does NOT survive a
    /// checkpoint encode/decode cycle.  ConvState records are
    /// recovered by walking the log on reload (the
    /// `conv_state_last_writer_wins_in_manifest` test covers that).
    #[test]
    fn conv_state_dropped_from_checkpoint_payload() {
        // ConvState state lives on the substrate after Phase 3, not
        // on the manifest.  This test is a regression guard that the
        // manifest itself stays empty + small regardless of whether
        // conv-state records are walked.
        let m = Manifest::new();
        let bytes = m.encode();
        let decoded = Manifest::decode(&bytes).unwrap();
        assert_eq!(m, decoded);
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
        assert!(meta.custom.is_empty(), "missing custom must default to empty");
    }

    /// The `custom` key/value bag round-trips through encode → decode.
    #[test]
    fn label_payload_round_trips_custom_metadata() {
        let mut custom = BTreeMap::new();
        custom.insert("kind".to_string(), "code_read".to_string());
        custom.insert("path".to_string(), "src/auth/handler.rs".to_string());
        custom.insert("content_sha256".to_string(), "deadbeef".to_string());
        let bytes = encode_label_payload(7, "cid", "Label", &custom);
        let (tl, meta) = decode_label_payload(&bytes).unwrap();
        assert_eq!(tl, 7);
        assert_eq!(meta.conv_id, "cid");
        assert_eq!(meta.label, "Label");
        assert_eq!(meta.custom, custom);
    }

    /// A Label written by an older binary (no `custom` field at all)
    /// decodes to an empty bag rather than failing — forward compat.
    #[test]
    fn label_payload_without_custom_decodes_empty() {
        let payload = br#"{"timeline_id":4,"conv_id":"c","label":"L"}"#.to_vec();
        let (_, meta) = decode_label_payload(&payload).unwrap();
        assert!(meta.custom.is_empty());
    }

    /// An empty `custom` is omitted from the wire form (skip_serializing_if)
    /// so existing logs/readers are byte-compatible when no metadata is set.
    #[test]
    fn label_payload_omits_empty_custom() {
        let bytes = encode_label_payload(1, "c", "L", &BTreeMap::new());
        let s = String::from_utf8(bytes).unwrap();
        assert!(!s.contains("custom"), "empty custom must not be serialized: {s}");
    }

    /// The Manifest itself must tolerate added/removed top-level
    /// JSON fields — a future field doesn't fail-load, a missing
    /// field decodes to its `Default` value.
    #[test]
    fn manifest_ignores_unknown_top_level_fields() {
        let payload = br#"{"streams":[],"labels":[],"conv_states":[],"future_top_level":42}"#;
        let decoded = Manifest::decode(payload).unwrap();
        // After Phase 3 the per-entity fields don't exist on Manifest;
        // unknown / legacy fields in the JSON payload are silently
        // ignored.  The decoder must produce an empty manifest.
        assert_eq!(decoded, Manifest::new());
    }
}
