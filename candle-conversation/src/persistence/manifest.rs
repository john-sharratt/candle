//! The in-RAM manifest — the index a walk reconstructs (§5.5–§5.6 of
//! `docs/kv_tier_migration.md`).
//!
//! The manifest resolves the live state of the log under **last-writer-wins**:
//! ingesting records in append order, a later record for the same key simply
//! overwrites an earlier one. It indexes the stream DAG, every chunk's
//! location, and the latest singleton records, and it serialises into a
//! `Checkpoint` record so a restart need not re-walk the whole log.

use std::collections::BTreeMap;

use super::log_file::LogSource;
use super::record::{ByteReader, ByteWriter, RecordType};
use super::streams::{StreamDecl, StreamId};
use super::walker::{self, WalkEntry, WalkOutcome};
use super::{PersistenceError, Result};

/// Location of a whole record in the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordLoc {
    pub offset: u64,
    pub payload_len: u64,
}

/// Location and shape of one chunk record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkLoc {
    pub offset: u64,
    pub payload_len: u64,
    pub token_count: u64,
    pub format: u8,
}

/// The indexed state of one stream.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamEntry {
    /// The decoded stream declaration, once a `StreamDecl` record is seen.
    pub decl: Option<StreamDecl>,
    /// Live chunk locations by local chunk index — last-writer-wins.
    pub chunks: BTreeMap<u64, ChunkLoc>,
    /// Latest `Tokens` record for the stream.
    pub tokens: Option<RecordLoc>,
    /// Latest `Signatures` record for the stream.
    pub signatures: Option<RecordLoc>,
    /// Highest chunk index the stream is durably committed through.
    pub committed_through: Option<u64>,
}

/// The whole-log index.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Manifest {
    /// Latest `ModelSpec` record.
    pub model_spec: Option<RecordLoc>,
    /// Latest `Template` record.
    pub template: Option<RecordLoc>,
    /// Latest `Tokenizer` record (the model's `tokenizer.json` bytes).
    pub tokenizer: Option<RecordLoc>,
    /// Per-stream index, ordered by id for deterministic serialisation.
    pub streams: BTreeMap<StreamId, StreamEntry>,
    /// Per-timeline conversation metadata — the daemon's client-side
    /// `conv_id` string paired with the sidebar title. Keyed by
    /// `timeline_id`, decoded inline. Written as `RecordType::Label`
    /// records, **last-write-wins** on replay so the conv_id can be
    /// established immediately at first submit and the title can be
    /// filled in later by the titler.
    pub labels: BTreeMap<u64, ConvMeta>,
    /// Per-timeline lifecycle state (`archived` flag today; reserves
    /// room for more flags via a versioned 1-byte payload). Keyed by
    /// `timeline_id`. Written as `RecordType::ConvState` records,
    /// **last-write-wins** on replay so toggling archive↔unarchive
    /// each appends a small record and the latest wins.
    pub conv_states: BTreeMap<u64, ConvState>,
    /// Offset of the most recent `Checkpoint` record seen.
    pub last_checkpoint_offset: Option<u64>,
}

/// Per-timeline conversation metadata persisted in `RecordType::Label`.
/// `conv_id` is the client-supplied identifier (e.g. the frontend's
/// `Date.now()` string) used as the sidebar's stable id; `label` is the
/// human-readable title, possibly empty during the brief window between
/// first-submit and titler-completion.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConvMeta {
    pub conv_id: String,
    pub label: String,
}

/// Per-timeline lifecycle flags persisted in `RecordType::ConvState`.
/// Today: just the `archived` flag (hide-from-sidebar without losing
/// the conversation). Encoded as `{u8 version, u8 flags}` so future
/// flags (pinned, unread, …) can slot in without a new record type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConvState {
    pub archived: bool,
}

/// Wire-format version of the `ConvState` payload. Bumped if a
/// breaking change to the flag-byte layout becomes necessary.
pub const CONV_STATE_VERSION: u8 = 1;

const CONV_STATE_FLAG_ARCHIVED: u8 = 1 << 0;

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
                let meta = decode_label_payload(&entry.record.payload)?;
                // Last-write-wins: the conv_id is written at first
                // submit; the title may be filled in later by the
                // titler. Subsequent writes overwrite (e.g. a user
                // rename would land here).
                self.labels.insert(meta.0, meta.1);
            }
            RecordType::ConvState => {
                let (timeline_id, state) = decode_conv_state_payload(&entry.record.payload)?;
                // Last-write-wins: every archive / unarchive append
                // supersedes the previous state.
                self.conv_states.insert(timeline_id, state);
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

    /// Serialise to the `Checkpoint` record payload bytes.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = ByteWriter::new();
        put_opt_loc(&mut w, self.model_spec);
        put_opt_loc(&mut w, self.template);
        put_opt_loc(&mut w, self.tokenizer);
        match self.last_checkpoint_offset {
            Some(o) => {
                w.put_u8(1);
                w.put_u64(o);
            }
            None => w.put_u8(0),
        }
        w.put_u32(self.streams.len() as u32);
        for (id, entry) in &self.streams {
            w.put_u64(id.0);
            match &entry.decl {
                Some(d) => {
                    w.put_u8(1);
                    w.put_blob(&d.encode());
                }
                None => w.put_u8(0),
            }
            w.put_u32(entry.chunks.len() as u32);
            for (idx, loc) in &entry.chunks {
                w.put_u64(*idx);
                w.put_u64(loc.offset);
                w.put_u64(loc.payload_len);
                w.put_u64(loc.token_count);
                w.put_u8(loc.format);
            }
            put_opt_loc(&mut w, entry.tokens);
            put_opt_loc(&mut w, entry.signatures);
            match entry.committed_through {
                Some(c) => {
                    w.put_u8(1);
                    w.put_u64(c);
                }
                None => w.put_u8(0),
            }
        }
        w.put_u32(self.labels.len() as u32);
        for (timeline_id, meta) in &self.labels {
            w.put_u64(*timeline_id);
            w.put_str(&meta.conv_id);
            w.put_str(&meta.label);
        }
        w.put_u32(self.conv_states.len() as u32);
        for (timeline_id, state) in &self.conv_states {
            w.put_u64(*timeline_id);
            // Serialise the same way the on-disk record encodes —
            // version byte + flag byte — so the checkpoint is
            // forward-compatible with future flag bits.
            w.put_u8(CONV_STATE_VERSION);
            let mut flags: u8 = 0;
            if state.archived {
                flags |= CONV_STATE_FLAG_ARCHIVED;
            }
            w.put_u8(flags);
        }
        w.into_bytes()
    }

    /// Reconstruct from `Checkpoint` record payload bytes.
    pub fn decode(payload: &[u8]) -> Result<Manifest> {
        let mut r = ByteReader::new(payload);
        let model_spec = get_opt_loc(&mut r)?;
        let template = get_opt_loc(&mut r)?;
        let tokenizer = get_opt_loc(&mut r)?;
        let last_checkpoint_offset = if r.get_u8()? == 1 {
            Some(r.get_u64()?)
        } else {
            None
        };
        let n_streams = r.get_u32()? as usize;
        let mut streams = BTreeMap::new();
        for _ in 0..n_streams {
            let id = StreamId(r.get_u64()?);
            let decl = if r.get_u8()? == 1 {
                Some(StreamDecl::decode(r.get_blob()?)?)
            } else {
                None
            };
            let n_chunks = r.get_u32()? as usize;
            let mut chunks = BTreeMap::new();
            for _ in 0..n_chunks {
                let idx = r.get_u64()?;
                let loc = ChunkLoc {
                    offset: r.get_u64()?,
                    payload_len: r.get_u64()?,
                    token_count: r.get_u64()?,
                    format: r.get_u8()?,
                };
                chunks.insert(idx, loc);
            }
            let tokens = get_opt_loc(&mut r)?;
            let signatures = get_opt_loc(&mut r)?;
            let committed_through = if r.get_u8()? == 1 {
                Some(r.get_u64()?)
            } else {
                None
            };
            streams.insert(
                id,
                StreamEntry {
                    decl,
                    chunks,
                    tokens,
                    signatures,
                    committed_through,
                },
            );
        }
        let n_labels = r.get_u32()? as usize;
        let mut labels = BTreeMap::new();
        for _ in 0..n_labels {
            let timeline_id = r.get_u64()?;
            let conv_id = r.get_str()?;
            let label = r.get_str()?;
            labels.insert(timeline_id, ConvMeta { conv_id, label });
        }
        // ConvStates were added after the original checkpoint format —
        // tolerate a manifest payload that ends here (older
        // checkpoint, no ConvState entries) by treating EOF here as
        // "zero conv_states" rather than a corruption error.
        let mut conv_states = BTreeMap::new();
        if !r.is_done() {
            let n_states = r.get_u32()? as usize;
            for _ in 0..n_states {
                let timeline_id = r.get_u64()?;
                let version = r.get_u8()?;
                if version != CONV_STATE_VERSION {
                    return Err(PersistenceError::Corrupt(format!(
                        "manifest ConvState version {version}, expected {CONV_STATE_VERSION}"
                    )));
                }
                let flags = r.get_u8()?;
                conv_states.insert(
                    timeline_id,
                    ConvState {
                        archived: flags & CONV_STATE_FLAG_ARCHIVED != 0,
                    },
                );
            }
        }
        if !r.is_done() {
            return Err(PersistenceError::Corrupt(format!(
                "manifest payload has {} trailing bytes",
                r.remaining()
            )));
        }
        Ok(Manifest {
            model_spec,
            template,
            tokenizer,
            streams,
            labels,
            conv_states,
            last_checkpoint_offset,
        })
    }
}

/// Encode a `Label` record's payload — `{u64 timeline_id, str conv_id, str label}`.
pub fn encode_label_payload(timeline_id: u64, conv_id: &str, label: &str) -> Vec<u8> {
    let mut w = ByteWriter::new();
    w.put_u64(timeline_id);
    w.put_str(conv_id);
    w.put_str(label);
    w.into_bytes()
}

/// Decode a `Label` record's payload — see [`encode_label_payload`].
/// Returns `(timeline_id, ConvMeta)`.
pub fn decode_label_payload(payload: &[u8]) -> Result<(u64, ConvMeta)> {
    let mut r = ByteReader::new(payload);
    let timeline_id = r.get_u64()?;
    let conv_id = r.get_str()?;
    let label = r.get_str()?;
    if !r.is_done() {
        return Err(PersistenceError::Corrupt(format!(
            "Label payload has {} trailing bytes",
            r.remaining()
        )));
    }
    Ok((timeline_id, ConvMeta { conv_id, label }))
}

/// Encode a `ConvState` record's payload — `{u64 timeline_id, u8 version,
/// u8 flags}`. The flag byte is a bitfield (`CONV_STATE_FLAG_*`); the
/// version lets us evolve to wider flag fields without changing the
/// record type.
pub fn encode_conv_state_payload(timeline_id: u64, state: ConvState) -> Vec<u8> {
    let mut w = ByteWriter::new();
    w.put_u64(timeline_id);
    w.put_u8(CONV_STATE_VERSION);
    let mut flags: u8 = 0;
    if state.archived {
        flags |= CONV_STATE_FLAG_ARCHIVED;
    }
    w.put_u8(flags);
    w.into_bytes()
}

/// Decode a `ConvState` record's payload — see
/// [`encode_conv_state_payload`]. Returns `(timeline_id, ConvState)`.
/// Unknown flag bits in the byte are silently ignored so a future
/// pinned/unread/etc. bit doesn't fail-load on an older daemon.
pub fn decode_conv_state_payload(payload: &[u8]) -> Result<(u64, ConvState)> {
    let mut r = ByteReader::new(payload);
    let timeline_id = r.get_u64()?;
    let version = r.get_u8()?;
    if version != CONV_STATE_VERSION {
        return Err(PersistenceError::Corrupt(format!(
            "ConvState payload version {version}, expected {CONV_STATE_VERSION}"
        )));
    }
    let flags = r.get_u8()?;
    if !r.is_done() {
        return Err(PersistenceError::Corrupt(format!(
            "ConvState payload has {} trailing bytes",
            r.remaining()
        )));
    }
    let state = ConvState {
        archived: flags & CONV_STATE_FLAG_ARCHIVED != 0,
    };
    Ok((timeline_id, state))
}

fn put_opt_loc(w: &mut ByteWriter, loc: Option<RecordLoc>) {
    match loc {
        Some(l) => {
            w.put_u8(1);
            w.put_u64(l.offset);
            w.put_u64(l.payload_len);
        }
        None => w.put_u8(0),
    }
}

fn get_opt_loc(r: &mut ByteReader) -> Result<Option<RecordLoc>> {
    if r.get_u8()? == 1 {
        Ok(Some(RecordLoc {
            offset: r.get_u64()?,
            payload_len: r.get_u64()?,
        }))
    } else {
        Ok(None)
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

    /// `ConvState` payload encodes / decodes byte-identical with the
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
    /// to the latest one in the manifest — last-writer-wins,
    /// matching the contract Label records have.
    #[test]
    fn conv_state_last_writer_wins_in_manifest() {
        let mut blob = Vec::new();
        // Three writes on the same timeline: archive, unarchive,
        // archive again. Final state must be archived.
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

    /// ConvState entries survive a checkpoint encode/decode cycle —
    /// the manifest carries them through `encode` / `decode` along
    /// with labels.
    #[test]
    fn conv_state_survives_checkpoint_roundtrip() {
        let mut m = Manifest::new();
        m.conv_states.insert(1, ConvState { archived: true });
        m.conv_states.insert(2, ConvState { archived: false });
        let bytes = m.encode();
        let decoded = Manifest::decode(&bytes).unwrap();
        assert_eq!(decoded, m);
    }
}
