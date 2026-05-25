//! Substrate persistence layer — the three-tier KV-cache storage path.
//!
//! This is the generalized, mandatory persistence layer specified by
//! `docs/kv_tier_migration.md`: an append-only NVMe redo log of
//! content-addressed streams that forms a complete, self-contained
//! substrate image. It is not optional — a substrate is always backed by
//! its log.
//!
//! The module is built bottom-up, one concern per file (§13.3):
//!
//! - [`content_hash`] — the deterministic content-hash chain.
//! - [`record`] — the eight record types and the framing codec.
//! - [`streams`] — stream identity and the `StreamDecl` payload.
//! - [`log_file`] — the append-only redo-log file.
//! - [`walker`] — the skip-load record walk.
//! - [`manifest`] — the in-RAM last-writer-wins index.
//! - [`checkpoint`] — checkpoint serialisation and recovery.
//! - [`inherit`] — multi-log inheritance and the shared cache.
//!
//! [`SubstratePersistence`] is the public API tying them together.

pub mod checkpoint;
pub mod compaction;
pub mod content_hash;
pub mod inherit;
pub mod log_file;
pub mod manifest;
pub mod record;
pub mod resume;
pub mod streams;
pub mod transfer;
pub mod walker;
pub mod warm_pool;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use checkpoint::recover;
use inherit::InheritedSubstrate;
use log_file::{read_record_at, LogFile};
use manifest::Manifest;
use record::{encode_record, ChunkPayload, Record, RecordHeader, RecordType};
use streams::{ContentAddress, StreamDecl, StreamId, StreamKind, StreamRef};
use walker::WalkEntry;

/// Errors raised by the persistence layer.
#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("bad record magic: expected {expected:#010x}, found {found:#010x}")]
    BadMagic { expected: u32, found: u32 },

    #[error("record checksum mismatch: stored {header:#010x}, computed {computed:#010x}")]
    BadChecksum { header: u32, computed: u32 },

    #[error("truncated input: need {need} bytes, have {have}")]
    Truncated { need: usize, have: usize },

    #[error("unknown record type tag {0}")]
    UnknownRecordType(u8),

    #[error("unknown stream kind tag {0}")]
    UnknownStreamKind(u8),

    #[error("corrupt persistence data: {0}")]
    Corrupt(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Result type for the persistence layer.
pub type Result<T> = std::result::Result<T, PersistenceError>;

/// The name of the per-working-directory persistence subdirectory.
pub const SUBSTRATE_DIR: &str = ".substrate";

/// The name of the active redo-log file inside [`SUBSTRATE_DIR`].
pub const ACTIVE_LOG_NAME: &str = "substrate.log";

/// The persistence layer behind a substrate — owns the active redo log, the
/// inherited read-only logs, and the in-RAM manifest.
///
/// Persistence is mandatory: a substrate cannot exist without one.
pub struct SubstratePersistence {
    log: LogFile,
    manifest: Manifest,
    inherited: Vec<Arc<InheritedSubstrate>>,
    model_spec: Option<Vec<u8>>,
    template: Option<Vec<u8>>,
    /// SHA-256 of the on-disk `Tokenizer` record's bytes — kept (32 bytes)
    /// instead of the full ~11 MB so [`SubstratePersistence::set_tokenizer`]
    /// can decide whether to re-embed by comparing hashes.
    tokenizer_sha256: Option<[u8; 32]>,
    /// Filesystem path of the active log — the rename target of a
    /// compaction swap (§5.8).
    active_path: PathBuf,
}

/// SHA-256 of `bytes` — the tokenizer change-detection digest.
fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Dead-record ratio at which [`SubstratePersistence::should_compact`] fires
/// — half the log being dead weight is enough to justify a rewrite (§5.8).
pub const COMPACTION_DEAD_RATIO_THRESHOLD: f32 = 0.5;

impl SubstratePersistence {
    /// Open the persistence layer at `<cwd>/.substrate/substrate.log`,
    /// creating the directory and file if absent and recovering the
    /// manifest if present.
    pub fn open() -> Result<SubstratePersistence> {
        let cwd = std::env::current_dir()?;
        SubstratePersistence::open_in(&cwd)
    }

    /// Open the persistence layer at `<dir>/.substrate/substrate.log`.
    pub fn open_in(dir: &Path) -> Result<SubstratePersistence> {
        let active = ensure_active_path(dir)?;
        SubstratePersistence::from_paths(&active, &[])
    }

    /// Open over an ordered list of logs. The last entry is the active,
    /// writable log; every earlier entry is inherited and read-only,
    /// loaded through the shared cache (§13.5).
    pub fn open_concat(logs: &[PathBuf]) -> Result<SubstratePersistence> {
        let (active, inherited) = logs.split_last().ok_or_else(|| {
            PersistenceError::Corrupt("open_concat needs at least one log".into())
        })?;
        if let Some(parent) = active.parent() {
            std::fs::create_dir_all(parent)?;
        }
        SubstratePersistence::from_paths(active, inherited)
    }

    fn from_paths(active: &Path, inherited: &[PathBuf]) -> Result<SubstratePersistence> {
        let mut inherited_subs = Vec::with_capacity(inherited.len());
        for path in inherited {
            inherited_subs.push(InheritedSubstrate::load(path)?);
        }

        let (mut log, manifest) = open_or_create_active(active)?;
        let model_spec = manifest
            .model_spec
            .map(|loc| read_record_at(&mut log, loc.offset).map(|r| r.payload))
            .transpose()?;
        let template = manifest
            .template
            .map(|loc| read_record_at(&mut log, loc.offset).map(|r| r.payload))
            .transpose()?;
        let tokenizer_sha256 = manifest
            .tokenizer
            .map(|loc| read_record_at(&mut log, loc.offset).map(|r| sha256(&r.payload)))
            .transpose()?;

        Ok(SubstratePersistence {
            log,
            manifest,
            inherited: inherited_subs,
            model_spec,
            template,
            tokenizer_sha256,
            active_path: active.to_path_buf(),
        })
    }

    /// The active log's manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// The durable logical end of the active log.
    pub fn write_offset(&self) -> u64 {
        self.log.write_offset()
    }

    /// Number of inherited logs.
    pub fn inherited_count(&self) -> usize {
        self.inherited.len()
    }

    /// Append one record to the active log, updating the in-RAM manifest.
    /// Returns the file offset the record occupies.
    pub fn append_record(
        &mut self,
        record_type: RecordType,
        format: u8,
        stream_id: u64,
        chunk_index: u64,
        token_count: u64,
        payload: &[u8],
    ) -> Result<u64> {
        let header = RecordHeader {
            record_type,
            format,
            payload_len: payload.len() as u64,
            stream_id,
            chunk_index,
            token_count,
        };
        let bytes = encode_record(&header, payload);
        let offset = self.log.stage(&bytes);
        let entry = WalkEntry {
            offset,
            record: Record {
                header,
                payload: payload.to_vec(),
            },
            size: bytes.len() as u64,
        };
        self.manifest.ingest(&entry)?;
        Ok(offset)
    }

    /// Declare a stream — append its `StreamDecl` record. Returns the
    /// stream's derived id.
    pub fn declare_stream(&mut self, decl: &StreamDecl) -> Result<StreamId> {
        let id = decl.stream_id();
        self.append_record(RecordType::StreamDecl, 0, id.0, 0, 0, &decl.encode())?;
        Ok(id)
    }

    /// Append a stream's `Tokens` record.
    pub fn append_tokens(&mut self, stream_id: StreamId, tokens: &[u8]) -> Result<()> {
        self.append_record(RecordType::Tokens, 0, stream_id.0, 0, 0, tokens)?;
        Ok(())
    }

    /// Append a stream's `Signatures` record.
    pub fn append_signatures(&mut self, stream_id: StreamId, sigs: &[u8]) -> Result<()> {
        self.append_record(RecordType::Signatures, 0, stream_id.0, 0, 0, sigs)?;
        Ok(())
    }

    /// Append a `Commit` record marking `stream_id` durable through
    /// `through_index`.
    pub fn commit_stream(&mut self, stream_id: StreamId, through_index: u64) -> Result<()> {
        self.append_record(RecordType::Commit, 0, stream_id.0, through_index, 0, &[])?;
        Ok(())
    }

    /// Append a `Chunk` record — one sealed or partial KV chunk's bytes and
    /// quantization metadata. `token_count` is 32 for a sealed chunk, less
    /// for a partial tail; `format` is the `KvFormat` tag.
    pub fn write_chunk(
        &mut self,
        stream_id: StreamId,
        chunk_index: u64,
        token_count: u64,
        format: u8,
        payload: &ChunkPayload,
    ) -> Result<u64> {
        self.append_record(
            RecordType::Chunk,
            format,
            stream_id.0,
            chunk_index,
            token_count,
            &payload.encode(),
        )
    }

    /// Read one chunk's payload — from the active log, else any inherited
    /// log (§13.5). The chunk must be durable (committed); it is read from
    /// the file, not the un-flushed staging buffer.
    pub fn read_chunk(&mut self, stream_id: StreamId, chunk_index: u64) -> Result<ChunkPayload> {
        if let Some(loc) = self
            .manifest
            .streams
            .get(&stream_id)
            .and_then(|s| s.chunks.get(&chunk_index))
            .copied()
        {
            let record = read_record_at(&mut self.log, loc.offset)?;
            return ChunkPayload::decode(&record.payload);
        }
        for inherited in &self.inherited {
            if let Some(loc) = inherited
                .manifest()
                .streams
                .get(&stream_id)
                .and_then(|s| s.chunks.get(&chunk_index))
                .copied()
            {
                let record = inherited.read_record(loc.offset)?;
                return ChunkPayload::decode(&record.payload);
            }
        }
        Err(PersistenceError::Corrupt(format!(
            "no chunk record for stream {} index {chunk_index}",
            stream_id.0
        )))
    }

    /// Read every chunk of a stream, in chunk-index order — the cold-load
    /// disk read. Resolves across the active and inherited logs.
    pub fn read_stream_chunks(&mut self, stream_id: StreamId) -> Result<Vec<(u64, ChunkPayload)>> {
        let mut indices: Vec<u64> = Vec::new();
        if let Some(s) = self.manifest.streams.get(&stream_id) {
            indices.extend(s.chunks.keys().copied());
        }
        for inherited in &self.inherited {
            if let Some(s) = inherited.manifest().streams.get(&stream_id) {
                indices.extend(s.chunks.keys().copied());
            }
        }
        indices.sort_unstable();
        indices.dedup();
        let mut out = Vec::with_capacity(indices.len());
        for idx in indices {
            out.push((idx, self.read_chunk(stream_id, idx)?));
        }
        Ok(out)
    }

    /// Read a stream's latest `Tokens` record payload — from the active log,
    /// else any inherited log. `None` if the stream has no `Tokens` record.
    pub fn read_tokens(&mut self, stream_id: StreamId) -> Result<Option<Vec<u8>>> {
        if let Some(loc) = self.manifest.streams.get(&stream_id).and_then(|s| s.tokens) {
            let record = read_record_at(&mut self.log, loc.offset)?;
            return Ok(Some(record.payload));
        }
        for inherited in &self.inherited {
            if let Some(loc) = inherited
                .manifest()
                .streams
                .get(&stream_id)
                .and_then(|s| s.tokens)
            {
                let record = inherited.read_record(loc.offset)?;
                return Ok(Some(record.payload));
            }
        }
        Ok(None)
    }

    /// Read a stream's latest `Signatures` record payload — from the active
    /// log, else any inherited log. `None` if the stream has no `Signatures`.
    pub fn read_signatures(&mut self, stream_id: StreamId) -> Result<Option<Vec<u8>>> {
        if let Some(loc) = self
            .manifest
            .streams
            .get(&stream_id)
            .and_then(|s| s.signatures)
        {
            let record = read_record_at(&mut self.log, loc.offset)?;
            return Ok(Some(record.payload));
        }
        for inherited in &self.inherited {
            if let Some(loc) = inherited
                .manifest()
                .streams
                .get(&stream_id)
                .and_then(|s| s.signatures)
            {
                let record = inherited.read_record(loc.offset)?;
                return Ok(Some(record.payload));
            }
        }
        Ok(None)
    }

    /// Flush and `fsync` the active log — the group-commit durability point.
    pub fn commit(&mut self) -> Result<()> {
        self.log.commit()
    }

    /// Write a `Checkpoint` record snapshotting the manifest, commit it
    /// durably, and update the superblock's latest-checkpoint hint.
    pub fn checkpoint(&mut self) -> Result<()> {
        let payload = checkpoint::encode_checkpoint(&self.manifest);
        let offset = self.append_record(RecordType::Checkpoint, 0, 0, 0, 0, &payload)?;
        self.log.commit()?;
        self.log.set_latest_checkpoint(offset)?;
        Ok(())
    }

    /// Set the model spec — last-writer-wins. Appends a fresh `ModelSpec`
    /// record only when the bytes differ from the latest on file. Returns
    /// `true` if a record was written.
    pub fn set_model_spec(&mut self, spec: &[u8]) -> Result<bool> {
        if self.model_spec.as_deref() == Some(spec) {
            return Ok(false);
        }
        self.append_record(RecordType::ModelSpec, 0, 0, 0, 0, spec)?;
        self.model_spec = Some(spec.to_vec());
        Ok(true)
    }

    /// Set the projection template — last-writer-wins, like
    /// [`SubstratePersistence::set_model_spec`].
    pub fn set_template(&mut self, template: &[u8]) -> Result<bool> {
        if self.template.as_deref() == Some(template) {
            return Ok(false);
        }
        self.append_record(RecordType::Template, 0, 0, 0, 0, template)?;
        self.template = Some(template.to_vec());
        Ok(true)
    }

    /// Set the model's `tokenizer.json` bytes — last-writer-wins, like
    /// [`SubstratePersistence::set_model_spec`]. Appends only when the bytes
    /// differ from the latest on file, so the (large) tokenizer is written
    /// at most once per distinct model.
    pub fn set_tokenizer(&mut self, tokenizer: &[u8]) -> Result<bool> {
        let hash = sha256(tokenizer);
        if self.tokenizer_sha256 == Some(hash) {
            return Ok(false);
        }
        self.append_record(RecordType::Tokenizer, 0, 0, 0, 0, tokenizer)?;
        self.tokenizer_sha256 = Some(hash);
        Ok(true)
    }

    /// The latest model spec payload, if any.
    pub fn model_spec(&self) -> Option<&[u8]> {
        self.model_spec.as_deref()
    }

    /// The latest projection template payload, if any.
    pub fn template(&self) -> Option<&[u8]> {
        self.template.as_deref()
    }

    /// SHA-256 of the embedded tokenizer, if a `Tokenizer` record is present.
    pub fn tokenizer_sha256(&self) -> Option<[u8; 32]> {
        self.tokenizer_sha256
    }

    /// Resolve a content-addressed prompt section across the active log and
    /// every inherited log (§13.5). `Some` is a prefix-cache hit.
    pub fn lookup_section(&self, addr: ContentAddress) -> Option<StreamRef> {
        let id = content_hash::section_stream_id(addr);
        if self.has_stream(id) {
            Some(StreamRef {
                stream_id: id,
                kind: StreamKind::PromptSection,
            })
        } else {
            None
        }
    }

    /// Whether `stream_id` is present in the active log or any inherited log.
    pub fn has_stream(&self, stream_id: StreamId) -> bool {
        self.manifest.streams.contains_key(&stream_id)
            || self.inherited.iter().any(|i| i.has_stream(stream_id))
    }

    /// Whether the active log has accumulated enough dead weight to justify
    /// a compaction pass — the dead-record ratio (§5.8) crossing
    /// [`COMPACTION_DEAD_RATIO_THRESHOLD`]. Flushes pending writes first so
    /// the measurement reflects the durable log.
    pub fn should_compact(&mut self) -> Result<bool> {
        self.log.commit()?;
        let ratio = compaction::dead_record_ratio(&mut self.log, &self.manifest)?;
        Ok(ratio >= COMPACTION_DEAD_RATIO_THRESHOLD)
    }

    /// Compact the active log — the whole-file dead-record rewrite (§5.8).
    ///
    /// Quiesces in-flight writes, streams the live record set into a
    /// `substrate.log.compact` sibling, and atomically renames it over the
    /// active log. The in-RAM manifest and the `ModelSpec`/`Template` caches
    /// are rebuilt from the compacted file. Inherited logs are untouched —
    /// a child never compacts a base it only reads (§13.5).
    pub fn compact(&mut self) -> Result<()> {
        // 1. Quiesce — every staged write is now durable on disk.
        self.log.commit()?;

        // 2. Collect the live winners, in dependency order.
        let live = compaction::collect_live_records(&mut self.log, &self.manifest)?;

        // 3. Rewrite into a sibling file.
        let compact_path = compaction_path(&self.active_path);
        let (new_log, new_manifest) = compaction::write_compacted_log(&compact_path, &live)?;

        // 4. Swap: drop the old active handle, then rename the compacted
        //    file over it. Renaming an open file is sound — Rust opens with
        //    FILE_SHARE_DELETE, so the surviving handle follows the rename.
        let old = std::mem::replace(&mut self.log, new_log);
        drop(old);
        std::fs::rename(&compact_path, &self.active_path)?;

        // 5. Adopt the compacted manifest and refresh the metadata caches.
        self.manifest = new_manifest;
        self.model_spec = self
            .manifest
            .model_spec
            .map(|loc| read_record_at(&mut self.log, loc.offset).map(|r| r.payload))
            .transpose()?;
        self.template = self
            .manifest
            .template
            .map(|loc| read_record_at(&mut self.log, loc.offset).map(|r| r.payload))
            .transpose()?;
        self.tokenizer_sha256 = self
            .manifest
            .tokenizer
            .map(|loc| read_record_at(&mut self.log, loc.offset).map(|r| sha256(&r.payload)))
            .transpose()?;
        Ok(())
    }
}

/// The `substrate.log.compact` sibling of an active log path — compaction's
/// scratch file before the atomic rename-swap.
fn compaction_path(active: &Path) -> PathBuf {
    let mut name = active
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".compact");
    active.with_file_name(name)
}

/// Ensure `<dir>/.substrate/` exists and return the active log path inside it.
fn ensure_active_path(dir: &Path) -> Result<PathBuf> {
    let sub = dir.join(SUBSTRATE_DIR);
    std::fs::create_dir_all(&sub)?;
    Ok(sub.join(ACTIVE_LOG_NAME))
}

/// Open the active log, recovering and truncating a torn tail; or create it.
fn open_or_create_active(path: &Path) -> Result<(LogFile, Manifest)> {
    if path.exists() {
        let mut log = LogFile::open(path)?;
        let hint = log.superblock().latest_checkpoint_offset;
        let recovered = recover(&mut log, hint)?;
        if recovered.torn {
            log.truncate_to(recovered.tail_offset)?;
        }
        log.set_write_offset(recovered.tail_offset);
        Ok((log, recovered.manifest))
    } else {
        let log = LogFile::create(path)?;
        Ok((log, Manifest::new()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use streams::{SectionDecl, TurnDecl};

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("kvtier_sp_{tag}_{nanos}"));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn section(name: &str, prefix: u64) -> StreamDecl {
        StreamDecl::PromptSection(SectionDecl {
            address: ContentAddress {
                prefix_hash: content_hash::ContentHash { lo: prefix, hi: 0 },
                section_hash: content_hash::hash_bytes(name.as_bytes()),
            },
            debug_name: name.to_string(),
        })
    }

    #[test]
    fn open_creates_substrate_dir_and_log() {
        let dir = tmp_dir("open");
        {
            let sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.inherited_count(), 0);
        }
        assert!(dir.join(SUBSTRATE_DIR).join(ACTIVE_LOG_NAME).exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_spec_is_last_updated() {
        let dir = tmp_dir("modelspec");
        let mut sp = SubstratePersistence::open_in(&dir).unwrap();

        assert!(
            sp.set_model_spec(b"qwen3-30b-a3b").unwrap(),
            "first write happens"
        );
        let off1 = sp.manifest().model_spec.unwrap().offset;
        assert!(
            !sp.set_model_spec(b"qwen3-30b-a3b").unwrap(),
            "unchanged spec is a no-op"
        );
        assert_eq!(
            sp.manifest().model_spec.unwrap().offset,
            off1,
            "no-op appends no record"
        );
        assert!(
            sp.set_model_spec(b"qwen3-235b").unwrap(),
            "changed spec writes"
        );
        assert!(
            sp.manifest().model_spec.unwrap().offset > off1,
            "changed spec appends a fresh record"
        );
        assert_eq!(sp.model_spec(), Some(b"qwen3-235b".as_slice()));

        sp.commit().unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_spec_survives_reopen() {
        let dir = tmp_dir("ms_reopen");
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.set_model_spec(b"the-model").unwrap();
            sp.set_template(b"the-template").unwrap();
            sp.checkpoint().unwrap();
        }
        {
            let sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.model_spec(), Some(b"the-model".as_slice()));
            assert_eq!(sp.template(), Some(b"the-template".as_slice()));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streams_and_turns_recover_across_reopen() {
        let dir = tmp_dir("recover");
        let turn = StreamDecl::Turn(TurnDecl {
            timeline_id: 1,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 1,
            block_start: 0,
            block_end: 32,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            scores: streams::PerDepthScores::default(),
        });
        let turn_id;
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            turn_id = sp.declare_stream(&turn).unwrap();
            sp.append_tokens(turn_id, b"token-bytes").unwrap();
            sp.commit_stream(turn_id, 0).unwrap();
            sp.commit().unwrap();
        }
        {
            let sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(sp.has_stream(turn_id));
            let entry = &sp.manifest().streams[&turn_id];
            assert_eq!(entry.committed_through, Some(0));
            assert!(entry.tokens.is_some());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn open_concat_inherits_and_keeps_last_active() {
        let base_dir = tmp_dir("base");
        let child_dir = tmp_dir("child");

        // Build a base log with a section stream.
        let sec = section("shared_section", 0);
        let base_log = base_dir.join("base.log");
        {
            let mut base = SubstratePersistence::open_concat(&[base_log.clone()]).unwrap();
            base.declare_stream(&sec).unwrap();
            base.commit().unwrap();
        }

        // A child inherits the base; its active log is its own.
        let child_log = child_dir.join(SUBSTRATE_DIR).join(ACTIVE_LOG_NAME);
        let mut child =
            SubstratePersistence::open_concat(&[base_log.clone(), child_log.clone()]).unwrap();
        assert_eq!(child.inherited_count(), 1);

        // The section stream resolves through inheritance — a prefix-cache hit.
        let hit = child.lookup_section(match &sec {
            StreamDecl::PromptSection(s) => s.address,
            _ => unreachable!(),
        });
        assert!(hit.is_some(), "inherited section must resolve");

        // The child writes only to its own active log.
        let child_turn = StreamDecl::Turn(TurnDecl {
            timeline_id: 9,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 16,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: vec![sec.stream_id()],
            view: Vec::new(),
            scores: streams::PerDepthScores::default(),
        });
        child.declare_stream(&child_turn).unwrap();
        child.commit().unwrap();
        // The child's turn is in the child; the base is untouched.
        assert!(child
            .manifest()
            .streams
            .contains_key(&child_turn.stream_id()));
        assert!(!child.manifest().streams.contains_key(&sec.stream_id()));

        InheritedSubstrate::forget(&base_log);
        std::fs::remove_dir_all(&base_dir).ok();
        std::fs::remove_dir_all(&child_dir).ok();
    }

    #[test]
    fn unknown_section_is_a_cache_miss() {
        let dir = tmp_dir("miss");
        let sp = SubstratePersistence::open_in(&dir).unwrap();
        let absent = ContentAddress {
            prefix_hash: content_hash::ContentHash { lo: 12345, hi: 678 },
            section_hash: content_hash::ContentHash { lo: 9, hi: 9 },
        };
        assert!(sp.lookup_section(absent).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    fn chunk_payload(seed: u32) -> ChunkPayload {
        ChunkPayload {
            offset: seed as u16,
            k_format: 4,
            v_format: 5,
            k_pal: vec![seed as u8; 4],
            v_pal: vec![(seed + 1) as u8; 2],
            k_scale: vec![seed as f32, seed as f32 * 0.5],
            v_scale: vec![seed as f32 + 0.25],
            kv_bytes: (0..512u32).map(|i| ((i + seed * 13) % 256) as u8).collect(),
        }
    }

    #[test]
    fn chunk_write_read_roundtrip() {
        let dir = tmp_dir("chunk_rw");
        let sid = StreamId(777);
        let payload = chunk_payload(3);
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.write_chunk(sid, 0, 32, 4, &payload).unwrap();
            sp.commit().unwrap();
            assert_eq!(sp.read_chunk(sid, 0).unwrap(), payload);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chunks_survive_restart() {
        let dir = tmp_dir("chunk_restart");
        let sid = StreamId(42);
        let chunks: Vec<ChunkPayload> = (0..3).map(chunk_payload).collect();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            for (i, c) in chunks.iter().enumerate() {
                sp.write_chunk(sid, i as u64, 32, 4, c).unwrap();
            }
            sp.commit_stream(sid, 2).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            let read = sp.read_stream_chunks(sid).unwrap();
            assert_eq!(read.len(), 3);
            for (i, (idx, c)) in read.iter().enumerate() {
                assert_eq!(*idx, i as u64);
                assert_eq!(c, &chunks[i]);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compact_drops_dead_records_and_reopens_identical() {
        let dir = tmp_dir("compact");
        let sid = StreamId(55);
        let live_chunk = chunk_payload(7);
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            // Dead weight: superseded ModelSpecs and stale chunk-0 bodies.
            for s in ["qwen3-8b", "qwen3-14b", "qwen3-30b"] {
                sp.set_model_spec(s.as_bytes()).unwrap();
            }
            sp.set_model_spec(b"qwen3-235b-live").unwrap();
            sp.set_template(b"the-template").unwrap();
            for seed in [97, 98, 99] {
                sp.write_chunk(sid, 0, 32, 4, &chunk_payload(seed)).unwrap();
            }
            sp.write_chunk(sid, 0, 32, 4, &live_chunk).unwrap();
            sp.commit_stream(sid, 0).unwrap();
            sp.commit().unwrap();

            assert!(
                sp.should_compact().unwrap(),
                "a log half dead weight wants compaction"
            );
            sp.compact().unwrap();

            // Caches and the manifest survive the swap.
            assert_eq!(sp.model_spec(), Some(b"qwen3-235b-live".as_slice()));
            assert_eq!(sp.template(), Some(b"the-template".as_slice()));
            assert_eq!(sp.read_chunk(sid, 0).unwrap(), live_chunk);
            assert!(
                !sp.should_compact().unwrap(),
                "a freshly compacted log has no dead weight"
            );
        }
        // The compacted file recovers to the same live substrate on reopen.
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.model_spec(), Some(b"qwen3-235b-live".as_slice()));
            assert_eq!(sp.template(), Some(b"the-template".as_slice()));
            assert_eq!(sp.read_chunk(sid, 0).unwrap(), live_chunk);
        }
        assert!(
            !dir.join(SUBSTRATE_DIR)
                .join(format!("{ACTIVE_LOG_NAME}.compact"))
                .exists(),
            "the .compact scratch file is renamed away, not left behind"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn chunk_resolves_from_inherited_log() {
        let base_dir = tmp_dir("ck_base");
        let child_dir = tmp_dir("ck_child");
        let base_log = base_dir.join("base.log");
        let sid = StreamId(9000);
        let payload = chunk_payload(11);
        {
            let mut base = SubstratePersistence::open_concat(&[base_log.clone()]).unwrap();
            base.write_chunk(sid, 0, 32, 4, &payload).unwrap();
            base.commit().unwrap();
        }
        let child_log = child_dir.join(SUBSTRATE_DIR).join(ACTIVE_LOG_NAME);
        let mut child = SubstratePersistence::open_concat(&[base_log.clone(), child_log]).unwrap();
        // The chunk lives only in the inherited base — read_chunk resolves it.
        assert_eq!(child.read_chunk(sid, 0).unwrap(), payload);

        InheritedSubstrate::forget(&base_log);
        std::fs::remove_dir_all(&base_dir).ok();
        std::fs::remove_dir_all(&child_dir).ok();
    }
}
