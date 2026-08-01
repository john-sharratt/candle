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
//! - [`record`] — the record types and the framing codec.
//! - [`streams`] — stream identity and the `StreamDecl` payload.
//! - [`log_file`] — the append-only redo-log file.
//! - [`walker`] — the skip-load record walk.
//! - [`manifest`] — the in-RAM last-writer-wins singleton index.
//! - [`header_index`] — the batched record-digest chain (§5.6).
//! - [`recovery`] — chain-first recovery with a forward-walk fallback.
//! - [`accounting`] — O(1) live/dead byte accounting for compaction.
//! - [`inherit`] — multi-log inheritance and the shared cache.
//!
//! [`SubstratePersistence`] is the public API tying them together.

pub mod accounting;
pub mod chunk_plan;
pub mod cold_load;
pub mod compaction;
pub mod content_hash;
pub mod direct_io;
pub mod elevate;
pub mod header_index;
pub mod inherit;
pub mod log_file;
pub mod maintenance;
pub mod manifest;
pub mod pipeline;
pub mod record;
pub mod recovery;
pub mod resume;
pub mod segment;
pub mod segmented_log;
pub mod streams;
pub mod thread;
pub mod transfer;
pub mod walker;
pub mod writer;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use thiserror::Error;

use crate::substrate::Substrate;

use accounting::RecordAccounting;
use chunk_plan::ChunkedReadPlan;
use header_index::{encode_index_payload, IndexEntry, INDEX_FLUSH_ENTRIES};
use inherit::InheritedSubstrate;
use manifest::{ChunkLoc, Manifest, RecordLoc};
use record::{
    decode_record, encode_record, ChunkPayload, DebugIdPayload, Record, RecordHeader, RecordType,
    TreeMetadataPayload,
};
use segment::SegmentId;
use segmented_log::SegmentedLog;
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

/// The name of the per-working-directory persistence subdirectory. Holds the
/// segmented redo log (`seg-*.log` sealed, one `seg-*.active`) — see
/// [`segmented_log`].
pub const SUBSTRATE_DIR: &str = ".substrate";

/// The persistence layer behind a substrate — owns the active redo log, the
/// inherited read-only logs, and the in-RAM manifest.
///
/// Persistence is mandatory: a substrate cannot exist without one.
pub struct SubstratePersistence {
    /// The segmented redo log — the active append segment plus the sealed
    /// segment set under `.substrate/`. Replaces the single monolithic log:
    /// reads route to the segment holding each record by `(segment, offset)`.
    segments: SegmentedLog,
    manifest: Manifest,
    inherited: Vec<Arc<InheritedSubstrate>>,
    model_spec: Option<Vec<u8>>,
    template: Option<Vec<u8>>,
    /// SHA-256 of the `Tokenizer` record's payload — kept (32 bytes)
    /// instead of the full ~11 MB so [`SubstratePersistence::set_tokenizer`]
    /// can decide whether to re-embed by comparing hashes. The bytes
    /// themselves stay on disk and are read on demand via
    /// [`SubstratePersistence::read_tokenizer_bytes`].
    tokenizer_sha256: Option<[u8; 32]>,
    /// O(1) live/dead byte accounting over the active log — fed by
    /// every append and by the recovery walk, read by
    /// [`SubstratePersistence::should_compact`]. Reset and rebuilt when
    /// compaction rewrites the file.
    accounting: RecordAccounting,
    /// Digests of appended records not yet covered by a `HeaderIndex`
    /// record — seeded at open with the un-indexed tail recovery
    /// reports, flushed every [`INDEX_FLUSH_ENTRIES`] appends.
    pending_index: Vec<IndexEntry>,
    /// `(offset, padded_size)` of the newest `HeaderIndex` record —
    /// the link the next flush chains to and the superblock hint's
    /// source of truth. `None` starts a fresh chain.
    last_index: Option<(u64, u64)>,
    /// Data records replayed by the open-time recovery (chain digests +
    /// walked tail). Diagnostic: startup logging reports it alongside
    /// the open latency.
    recovered_records: usize,
    /// The last background-maintenance op applied — `(label, unix_secs)` —
    /// surfaced in the daemon status for the GUI's compaction indicator.
    /// `None` until the first op runs.
    last_maintenance: Option<(String, u64)>,
    /// The active-segment id captured at the last maintenance **resident-set
    /// re-emission** (the drop-safety net). Because segment ids are strictly
    /// append-ordered (a fresh id is always above every existing one), every
    /// metadata record that existed at that re-emission is duplicated into
    /// segments `>= resident_reemit_floor`. So a later maintenance op that
    /// targets only segments `< floor` cannot lose a unique metadata record and
    /// can SKIP re-emitting the whole resident set — which is what stops the
    /// re-emit→looks-dead→compact→re-emit churn. `None` forces a re-emit (no
    /// durable snapshot yet, or reset by a full `compact`).
    resident_reemit_floor: Option<SegmentId>,
    /// On-disk location of each stream's CURRENT per-stream metadata record —
    /// `StreamDecl` / `ProjectionEvents` / `WideQSig` / `Commit` — keyed by
    /// `(record_type, stream_id)`, last-writer-wins. The substrate index caches
    /// these payloads but not their locations, so without this map
    /// [`segment_liveness`](SubstratePersistence::segment_liveness) would count
    /// them as dead — making the segment that holds the live metadata perpetually
    /// look reclaimable and get "compacted" (re-emitted forward) on every
    /// maintenance pass. Counting them as live via this map is what actually
    /// halts that churn. Populated on every append and rebuilt on load / compact.
    metadata_locs: HashMap<(RecordType, u64), RecordLoc>,
}

/// Whether a record type is a per-stream metadata record whose current-copy
/// location is tracked in [`SubstratePersistence::metadata_locs`] so it counts
/// as live weight (see that field). These are the records the maintenance
/// resident-set re-emits and that carry no location in the substrate index.
fn is_tracked_metadata(rt: RecordType) -> bool {
    matches!(
        rt,
        RecordType::StreamDecl
            | RecordType::ProjectionEvents
            | RecordType::WideQSig
            | RecordType::Commit
    )
}

/// Populate `map` with a walked record's metadata location (LWW). The load /
/// compact walk uses this before the `SubstratePersistence` exists; the runtime
/// append path uses [`SubstratePersistence::track_metadata_loc`].
fn record_metadata_loc(map: &mut HashMap<(RecordType, u64), RecordLoc>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    if is_tracked_metadata(h.record_type) {
        map.insert(
            (h.record_type, h.stream_id),
            RecordLoc {
                segment: entry.segment,
                offset: entry.offset,
                payload_len: h.payload_len,
                record_size: entry.size,
            },
        );
    }
}

/// SHA-256 of `bytes` — the tokenizer change-detection digest.
fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Read-into-RAM/VRAM golden check (non-fatal). Recompute a chunk's golden over
/// the KV bytes just read off disk and warn on mismatch — flagging on-disk or
/// read-path corruption without failing the latency-sensitive load. The
/// restart/recovery read never reaches here: it reads only metadata records.
fn warn_on_chunk_golden_mismatch(header: &RecordHeader, payload: &ChunkPayload, chunk_idx: u64) {
    let recomputed = candle::fletcher::fletcher32(&payload.kv_bytes);
    if recomputed != header.crc {
        tracing::warn!(
            target: "candle_conversation::persistence::golden",
            chunk_idx,
            stored = header.crc,
            recomputed,
            "chunk golden mismatch on batched read into RAM — possible on-disk/read corruption"
        );
    }
}

/// Dead-byte ratio at which [`SubstratePersistence::should_compact`] fires
/// — half the log being dead weight is enough to justify a rewrite (§5.8).
pub const COMPACTION_DEAD_RATIO_THRESHOLD: f32 = 0.5;

/// Logs shorter than this never auto-compact — a rewrite of a small file
/// reclaims nothing worth the pause, regardless of its dead ratio.
pub const COMPACTION_MIN_LOG_BYTES: u64 = 64 * 1024 * 1024;

impl SubstratePersistence {
    /// Open the persistence layer at `<cwd>/.substrate/substrate.log`,
    /// creating the directory and file if absent and recovering the
    /// manifest if present.
    pub fn open() -> Result<SubstratePersistence> {
        let cwd = std::env::current_dir()?;
        SubstratePersistence::open_in(&cwd)
    }

    /// Open the persistence layer at `<dir>/.substrate/` (the segment set).
    pub fn open_in(dir: &Path) -> Result<SubstratePersistence> {
        Self::from_dir_with_sink(&dir.join(SUBSTRATE_DIR), &[], |_| {})
    }

    /// Open the persistence layer and drive every record through
    /// [`Substrate::apply_walker_entry`] in the same pass that builds
    /// the manifest's singleton offsets.  The substrate ends up fully
    /// populated (streams, labels, conv_states, tree metadata, debug
    /// ids) by the time this returns.
    ///
    /// Replaces the legacy two-phase `open_in` + `reconstruct →
    /// mirror_from_manifest` pattern with a single walker pass.  No
    /// per-entity state is mirrored from manifest to substrate
    /// afterwards because the manifest's per-entity fields are
    /// `#[serde(skip)]` and the substrate is the authoritative
    /// in-RAM index.
    pub fn open_in_with_substrate(
        dir: &Path,
        substrate: &mut Substrate,
    ) -> Result<SubstratePersistence> {
        Self::from_dir_with_sink(&dir.join(SUBSTRATE_DIR), &[], |entry| {
            substrate.apply_walker_entry(entry)
        })
    }

    /// Open over an ordered list of paths. The last entry is the active,
    /// writable **segment directory** (`.substrate/`); every earlier entry is
    /// an inherited read-only single-file log, loaded through the shared
    /// cache (§13.5).
    pub fn open_concat(logs: &[PathBuf]) -> Result<SubstratePersistence> {
        let (active_dir, inherited) = logs.split_last().ok_or_else(|| {
            PersistenceError::Corrupt("open_concat needs at least one log".into())
        })?;
        Self::from_dir_with_sink(active_dir, inherited, |_| {})
    }

    /// Open the segment set in `dir` (the `.substrate/` directory) with the
    /// listed inherited single-file logs, driving every recovered record
    /// through `sink` in the same pass that builds the manifest and the
    /// dead-weight accounting.
    fn from_dir_with_sink<F>(
        dir: &Path,
        inherited: &[PathBuf],
        mut sink: F,
    ) -> Result<SubstratePersistence>
    where
        F: FnMut(&walker::WalkEntry),
    {
        let mut inherited_subs = Vec::with_capacity(inherited.len());
        for path in inherited {
            inherited_subs.push(InheritedSubstrate::load(path)?);
        }

        // Recover every segment (ascending id, active last) — feeding the
        // dead-weight accounting and the caller's sink in the same pass that
        // builds the combined singleton manifest.
        let mut accounting = RecordAccounting::new();
        let mut metadata_locs: HashMap<(RecordType, u64), RecordLoc> = HashMap::new();
        let segmented_log::OpenedSegments {
            mut segments,
            manifest,
            last_index,
            tail_digests,
            recovered_records,
        } = SegmentedLog::open_with_sink(dir, |entry| {
            accounting.record(&entry.record.header, entry.size);
            record_metadata_loc(&mut metadata_locs, entry);
            sink(entry);
        })?;

        let model_spec = manifest
            .model_spec
            .map(|loc| {
                segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)
                    .map(|r| r.payload)
            })
            .transpose()?;
        let template = manifest
            .template
            .map(|loc| {
                segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)
                    .map(|r| r.payload)
            })
            .transpose()?;
        let tokenizer_sha256 = manifest
            .tokenizer
            .map(|loc| {
                let r = segments.read_record_at(loc.segment, loc.offset, loc.record_size)?;
                Ok::<_, PersistenceError>(sha256(&r.payload))
            })
            .transpose()?;

        let mut sp = SubstratePersistence {
            segments,
            manifest,
            inherited: inherited_subs,
            model_spec,
            tokenizer_sha256,
            template,
            accounting,
            pending_index: tail_digests,
            last_index,
            recovered_records,
            last_maintenance: None,
            resident_reemit_floor: None,
            metadata_locs,
        };
        // Self-heal a large un-indexed tail (a crash window, or a log
        // that predates the index chain entirely): flush it now so the
        // next open takes the chain path instead of re-walking it.
        if sp.pending_index.len() >= INDEX_FLUSH_ENTRIES {
            sp.flush_header_index()?;
        }
        Ok(sp)
    }

    /// The active log's manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// All inherited (read-only) substrates, in registration order.
    /// Each carries its own manifest.  Exposed so the substrate-side
    /// reload can mirror per-stream state across all inherited logs
    /// without each call site re-deriving the list.
    pub fn inherited_substrates(&self) -> &[Arc<InheritedSubstrate>] {
        &self.inherited
    }

    /// The durable logical end of the active segment.
    pub fn write_offset(&self) -> u64 {
        self.segments.write_offset()
    }

    /// The active (append target) segment's id — the segment fresh appends
    /// land in.
    pub fn active_segment(&self) -> SegmentId {
        self.segments.active_id()
    }

    /// The cache-bypassing read handles for the active segment — exposed
    /// for the pipelined cold-load reader pool.
    pub(super) fn active_direct_file(&self) -> &direct_io::DirectFile {
        self.segments.active_direct_file()
    }

    /// A freshly-opened direct-I/O handle set on sealed `segment`, owned by
    /// the caller for the duration of one cold-load. The GPU pipeline holds
    /// several of these at once (a turn spanning a seal), so they are opened
    /// directly rather than borrowed from the read pool.
    pub(super) fn open_sealed_direct(&self, segment: SegmentId) -> Result<direct_io::DirectFile> {
        self.segments.open_sealed_direct(segment)
    }

    /// The cache-bypassing read handles for the `i`-th inherited log.
    pub(super) fn inherited_direct_file(&self, i: usize) -> &direct_io::DirectFile {
        self.inherited[i].direct_file()
    }

    /// Number of inherited logs.
    pub fn inherited_count(&self) -> usize {
        self.inherited.len()
    }

    /// Append one record to the active segment, updating the in-RAM manifest.
    /// Returns the `(segment, offset, size)` the record occupies — callers
    /// that index the record (cold-load, chunk persistence) need the segment
    /// so the read routes back to the right file, plus the padded on-disk
    /// size for the bytes-on-disk footprint.
    pub fn append_record(
        &mut self,
        record_type: RecordType,
        format: u8,
        stream_id: u64,
        chunk_index: u64,
        token_count: u64,
        golden: u32,
        payload: &[u8],
    ) -> Result<(SegmentId, u64, u64)> {
        let header = RecordHeader {
            record_type,
            format,
            payload_len: payload.len() as u64,
            // `Chunk` records store the GPU-computed golden (Fletcher-32 over the
            // arena bytes, before the DtoH copy) here; `encode_record` keeps it.
            // Every other type ignores `golden` — `encode_record` fills `crc`
            // with crc32 over the payload.
            crc: golden,
            stream_id,
            chunk_index,
            token_count,
        };
        // Write-boundary validation: recompute the chunk's golden over the host
        // bytes we are about to write and compare against the golden the seal
        // computed on the GPU (before the DtoH copy). A mismatch means the copy
        // or host handling corrupted the KV bytes — a real byte-movement fault.
        // Log it (a rebuild regenerates the substrate; this proves the pipeline)
        // and still write what we have.
        if record_type == RecordType::Chunk && golden != 0 {
            match crate::persistence::record::recompute_chunk_golden(payload) {
                Ok(recomputed) if recomputed != golden => tracing::error!(
                    target: "candle_conversation::persistence::golden",
                    stream_id,
                    chunk_index,
                    golden,
                    recomputed,
                    "chunk golden mismatch before write — KV bytes changed between GPU seal and disk (DtoH/host corruption)"
                ),
                Ok(_) => {}
                Err(e) => tracing::error!(
                    target: "candle_conversation::persistence::golden",
                    stream_id,
                    chunk_index,
                    "chunk payload undecodable before write: {e}"
                ),
            }
        }
        let bytes = encode_record(&header, payload);
        let (segment, offset) = self.segments.stage(&bytes);
        let size = bytes.len() as u64;
        self.accounting.record(&header, size);
        self.track_metadata_loc(&header, segment, offset, size);
        let entry = WalkEntry {
            segment,
            offset,
            record: Record {
                header,
                payload: payload.to_vec(),
            },
            size,
        };
        self.manifest.ingest(&entry)?;
        // Digest every data record for the header-index chain. Index
        // records themselves are self-describing via their `prev` links
        // and are never digested.
        if header.record_type != RecordType::HeaderIndex {
            self.pending_index
                .push(IndexEntry::from_header(&header, offset, size));
            if self.pending_index.len() >= INDEX_FLUSH_ENTRIES {
                self.flush_header_index()?;
            }
            self.rotate_if_over_target()?;
        }
        Ok((segment, offset, size))
    }

    /// Append an **already-encoded** record verbatim to the active segment —
    /// the same accounting + header-index bookkeeping as [`Self::append_record`]
    /// but with **no** re-encode: the caller supplies the raw 4 KB-aligned
    /// record bytes (read verbatim from another segment) plus a header for the
    /// bookkeeping. Used by background maintenance to relocate `Chunk`/`Tokens`
    /// records with no decode/CRC-verify/re-encode round trip. Returns
    /// `(segment, offset, size)`. Not for singletons — those go through
    /// [`Self::append_record`] so `manifest.ingest` repoints them.
    pub fn append_raw_record(
        &mut self,
        header: &RecordHeader,
        raw: &[u8],
    ) -> Result<(SegmentId, u64, u64)> {
        let (segment, offset) = self.segments.stage(raw);
        let size = raw.len() as u64;
        self.accounting.record(header, size);
        // `Chunk` / `Tokens` are indexed on the substrate, not the manifest, so
        // no `manifest.ingest` — the caller repoints the substrate index.
        if header.record_type != RecordType::HeaderIndex {
            self.pending_index
                .push(IndexEntry::from_header(header, offset, size));
            if self.pending_index.len() >= INDEX_FLUSH_ENTRIES {
                self.flush_header_index()?;
            }
            self.rotate_if_over_target()?;
        }
        Ok((segment, offset, size))
    }

    /// Flush the accumulated digests as `HeaderIndex` record(s) chained
    /// to the previous index, commit them durably, and publish the new
    /// chain head in the superblock. The commit-before-publish order
    /// means the hint never points at un-flushed bytes; recovery's
    /// fallback covers the crash window between the two writes either
    /// way.
    fn flush_header_index(&mut self) -> Result<()> {
        if self.pending_index.is_empty() {
            return Ok(());
        }
        while !self.pending_index.is_empty() {
            let take = self.pending_index.len().min(INDEX_FLUSH_ENTRIES);
            let batch: Vec<IndexEntry> = self.pending_index.drain(..take).collect();
            let payload = encode_index_payload(self.last_index.unwrap_or((0, 0)), &batch);
            let (_seg, offset, size) =
                self.append_record(RecordType::HeaderIndex, 0, 0, 0, 0, 0, &payload)?;
            self.last_index = Some((offset, size));
        }
        self.segments.commit()?;
        if let Some(li) = self.last_index {
            self.segments.set_last_index(li)?;
        }
        Ok(())
    }

    /// The newest committed `HeaderIndex` chain head, if any — the
    /// record the next flush chains to and the superblock hint's value.
    /// Diagnostic accessor.
    pub fn last_index(&self) -> Option<(u64, u64)> {
        self.last_index
    }

    /// Number of appended records not yet covered by a `HeaderIndex`
    /// record. Diagnostic accessor.
    pub fn pending_index_len(&self) -> usize {
        self.pending_index.len()
    }

    /// Data records the open-time recovery replayed (chain digests +
    /// walked tail). Diagnostic accessor.
    pub fn recovered_record_count(&self) -> usize {
        self.recovered_records
    }

    /// The last background-maintenance op applied — `(label, unix_secs)` — or
    /// `None` if none has run this session. Surfaced in the daemon status.
    pub fn last_maintenance(&self) -> Option<(String, u64)> {
        self.last_maintenance.clone()
    }

    /// Declare a stream — append its `StreamDecl` record. Returns the
    /// stream's derived id.
    pub fn declare_stream(&mut self, decl: &StreamDecl) -> Result<StreamId> {
        let id = decl.stream_id();
        self.append_record(RecordType::StreamDecl, 0, id.0, 0, 0, 0, &decl.encode())?;
        Ok(id)
    }

    /// Record the on-disk location of a per-stream metadata record
    /// (`StreamDecl` / `ProjectionEvents` / `WideQSig` / `Commit`) so
    /// [`segment_liveness`](Self::segment_liveness) counts the current copy as
    /// live weight. Last-writer-wins per `(record_type, stream_id)`: a re-write
    /// (or maintenance re-emission) supersedes the prior location, which then
    /// reads as dead. A no-op for every other record type. Called from
    /// [`append_record`](Self::append_record) (runtime writes) and the
    /// load / compact walk so the map covers both.
    fn track_metadata_loc(&mut self, h: &RecordHeader, segment: SegmentId, offset: u64, size: u64) {
        if is_tracked_metadata(h.record_type) {
            self.metadata_locs.insert(
                (h.record_type, h.stream_id),
                RecordLoc {
                    segment,
                    offset,
                    payload_len: h.payload_len,
                    record_size: size,
                },
            );
        }
    }

    /// Append a stream's `Tokens` record.
    pub fn append_tokens(&mut self, stream_id: StreamId, tokens: &[u8]) -> Result<()> {
        self.append_record(RecordType::Tokens, 0, stream_id.0, 0, 0, 0, tokens)?;
        Ok(())
    }

    /// Append a turn's `ProjectionEvents` record (opaque JSON payload).
    pub fn append_projection_events(&mut self, stream_id: StreamId, payload: &[u8]) -> Result<()> {
        self.append_record(
            RecordType::ProjectionEvents,
            0,
            stream_id.0,
            0,
            0,
            0,
            payload,
        )?;
        Ok(())
    }

    /// Append a turn's `WideQSig` record (opaque wide-Q window payload), keyed by stream id.
    pub fn append_wide_q_sigs(&mut self, stream_id: StreamId, payload: &[u8]) -> Result<()> {
        self.append_record(RecordType::WideQSig, 0, stream_id.0, 0, 0, 0, payload)?;
        Ok(())
    }

    /// Append a `Commit` record marking `stream_id` durable through
    /// `through_index`.
    pub fn commit_stream(&mut self, stream_id: StreamId, through_index: u64) -> Result<()> {
        self.append_record(RecordType::Commit, 0, stream_id.0, through_index, 0, 0, &[])?;
        Ok(())
    }

    /// Write a conversation-metadata record for `timeline_id`. The record
    /// carries both the client-supplied `conv_id` string (used by the
    /// daemon as the sidebar id) and the human-readable `label`.
    ///
    /// Last-write-wins on replay: the conv_id is written at first-submit
    /// time with an empty label, then re-written with the title once the
    /// titler finishes. This call is a no-op when the manifest already
    /// holds the same `(conv_id, label)` tuple — cheap to invoke on
    /// every submit.
    pub fn write_conv_meta(
        &mut self,
        timeline_id: u64,
        conv_id: &str,
        label: &str,
        custom: &std::collections::BTreeMap<String, String>,
    ) -> Result<()> {
        // Idempotency was previously checked against `manifest.labels`;
        // after Phase 3 the substrate is the authority for live label
        // state.  Callers are expected to check against substrate
        // state before invoking this method (the Conversation-level
        // setters already do; daemon writes are infrequent enough
        // that an occasional duplicate log entry is negligible —
        // compaction collapses them).
        //
        // The Label record carries the *full* ConvMeta (conv_id + label +
        // custom), so each setter reads the sibling fields from the
        // substrate and passes them through — a partial write would drop
        // the others on reload/compaction.
        let payload = manifest::encode_label_payload(timeline_id, conv_id, label, custom);
        self.append_record(RecordType::Label, 0, 0, 0, 0, 0, &payload)?;
        Ok(())
    }

    /// Append a `ConvState` record for `timeline_id`.  Idempotency on
    /// the current value is now the caller's responsibility (see
    /// `write_conv_meta`).
    pub fn write_conv_state(&mut self, timeline_id: u64, state: manifest::ConvState) -> Result<()> {
        let payload = manifest::encode_conv_state_payload(timeline_id, state);
        self.append_record(RecordType::ConvState, 0, 0, 0, 0, 0, &payload)?;
        Ok(())
    }

    /// Append a `TreeMetadata` record for one `(timeline_id,
    /// turn_index)` summary-tree node.  Last-writer-wins on replay.
    /// Callers check idempotency against substrate state.
    pub fn write_tree_metadata(&mut self, payload: TreeMetadataPayload) -> Result<()> {
        let bytes = payload.encode();
        self.append_record(RecordType::TreeMetadata, 0, 0, 0, 0, 0, &bytes)?;
        Ok(())
    }

    /// Append a [`RecordType::Tombstone`] record marking
    /// `timeline_id` as logically deleted.  Walker replay applies it
    /// to the substrate; the compactor drops every record bound to
    /// the timeline on the next compaction pass.  Idempotent —
    /// duplicate tombstones replay identically. `reason` is a diagnostic
    /// note (e.g. the corrupt-reload detail) recorded in the payload; pass
    /// `None` for an ordinary deletion.
    pub fn write_tombstone(&mut self, timeline_id: u64, reason: Option<&str>) -> Result<()> {
        let payload = record::TombstonePayload {
            timeline_id,
            reason: reason.map(str::to_string),
        };
        let bytes = payload.encode();
        self.append_record(RecordType::Tombstone, 0, 0, 0, 0, 0, &bytes)?;
        Ok(())
    }

    /// Append a [`RecordType::TurnCoupling`] record joining `from_turn` to the
    /// tool response that follows it.
    ///
    /// Written before the response turn is submitted — the one window where the
    /// round-trip is certain — so replay can never observe the response turn
    /// without its coupling. Idempotent: duplicates replay into the same set.
    pub fn write_turn_coupling(&mut self, timeline_id: u64, from_turn: u32) -> Result<()> {
        let payload = record::TurnCouplingPayload {
            timeline_id,
            from_turn,
        };
        let bytes = payload.encode();
        self.append_record(RecordType::TurnCoupling, 0, 0, 0, 0, 0, &bytes)?;
        Ok(())
    }

    /// Append a [`RecordType::Distilled`] record marking `timeline_id` for
    /// distillation at `mode` — its turns shed content on the next compaction
    /// pass. Idempotent for a fixed mode; a later record upgrades the mode
    /// (last-writer-wins on replay).
    pub fn write_distill(&mut self, timeline_id: u64, mode: record::DistillMode) -> Result<()> {
        let payload = record::DistillPayload { timeline_id, mode };
        let bytes = payload.encode();
        self.append_record(RecordType::Distilled, 0, 0, 0, 0, 0, &bytes)?;
        Ok(())
    }

    /// Append a `DebugId` record for `timeline_id`.  Last-writer-wins
    /// on replay.  Callers check idempotency against substrate state.
    pub fn write_debug_id(&mut self, timeline_id: u64, debug_id: &str) -> Result<()> {
        let payload = DebugIdPayload {
            timeline_id,
            debug_id: debug_id.to_string(),
        };
        let bytes = payload.encode();
        self.append_record(RecordType::DebugId, 0, 0, 0, 0, 0, &bytes)?;
        Ok(())
    }

    /// Append a `Chunk` record — one sealed or partial KV chunk's bytes and
    /// quantization metadata. `token_count` is 32 for a sealed chunk, less
    /// for a partial tail; `format` is the `KvFormat` tag. Returns the
    /// file offset of the record.
    pub fn write_chunk(
        &mut self,
        stream_id: StreamId,
        chunk_index: u64,
        token_count: u64,
        format: u8,
        golden: Option<u32>,
        payload: &ChunkPayload,
    ) -> Result<u64> {
        // `Some(g)` is the seal path's GPU-computed golden (Fletcher-32 over the
        // arena bytes before the DtoH copy). `None` means no precomputed golden
        // is available, so take it host-side over these bytes now — the same
        // host-truth the CPU gather path uses.
        let golden = golden.unwrap_or_else(|| candle::fletcher::fletcher32(&payload.kv_bytes));
        let (_seg, offset, _) = self.append_record(
            RecordType::Chunk,
            format,
            stream_id.0,
            chunk_index,
            token_count,
            golden,
            &payload.encode(),
        )?;
        Ok(offset)
    }

    /// Read one chunk's payload — from the active log, else any inherited
    /// log (§13.5). The chunk must be durable (committed); it is read from
    /// the file, not the un-flushed staging buffer.
    pub fn read_chunk(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
        chunk_index: u64,
    ) -> Result<ChunkPayload> {
        if let Some(loc) = substrate
            .stream_of(stream_id)
            .and_then(|s| s.chunks.get(&chunk_index))
            .copied()
        {
            let record = self
                .segments
                .read_record_at(loc.segment, loc.offset, loc.record_size)?;
            return ChunkPayload::decode(&record.payload);
        }
        for inherited in &self.inherited {
            if let Some(loc) = inherited
                .substrate()
                .stream_of(stream_id)
                .and_then(|s| s.chunks.get(&chunk_index))
                .copied()
            {
                let record = inherited.read_record(loc.offset, loc.record_size)?;
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
    ///
    /// Allocates a scratch buffer internally; callers that load many turns
    /// in sequence (e.g. `elevate_to_hot`) should call
    /// [`Self::read_stream_chunks_batched`] directly with a reusable buffer.
    pub fn read_stream_chunks(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
    ) -> Result<Vec<(u64, ChunkPayload)>> {
        let mut buf: Vec<u8> = Vec::new();
        self.read_stream_chunks_batched(substrate, stream_id, &mut buf)
    }

    /// Read every chunk of a stream in a small number of stripe-coalesced
    /// I/Os — the cold-load hot path.
    ///
    /// For each chunk, the manifest already carries `offset` and the
    /// exact padded `record_size`. We sort chunks by file offset per
    /// source log (active + inherited), coalesce adjacent records into
    /// stripes, and issue **one read per stripe** into the caller's
    /// reusable scratch buffer. For a freshly-persisted turn the records
    /// are contiguous on disk → ~1 syscall covers all of them.
    ///
    /// Decoding (CRC verify + payload parse) happens in memory against
    /// slices of `buf`. The returned `ChunkPayload`s own their data;
    /// `buf` can be reused for the next stream.
    ///
    /// `buf` is resized as needed (grows monotonically across calls).
    pub fn read_stream_chunks_batched(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
        buf: &mut Vec<u8>,
    ) -> Result<Vec<(u64, ChunkPayload)>> {
        use std::collections::{BTreeMap, HashSet};

        #[derive(Copy, Clone)]
        struct ChunkSource {
            chunk_idx: u64,
            file_offset: u64,
            record_size: u64,
        }
        #[derive(Copy, Clone)]
        struct Stripe {
            file_offset: u64,
            len: usize,
        }

        fn build_stripes(chunks: &[ChunkSource]) -> Vec<Stripe> {
            let mut out: Vec<Stripe> = Vec::new();
            let mut it = chunks.iter();
            let Some(first) = it.next() else {
                return out;
            };
            let mut start = first.file_offset;
            let mut end = first.file_offset + first.record_size;
            for c in it {
                if c.file_offset == end {
                    end = c.file_offset + c.record_size;
                } else {
                    out.push(Stripe {
                        file_offset: start,
                        len: (end - start) as usize,
                    });
                    start = c.file_offset;
                    end = c.file_offset + c.record_size;
                }
            }
            out.push(Stripe {
                file_offset: start,
                len: (end - start) as usize,
            });
            out
        }

        // Collect from active first; remember which indices we've seen so
        // inherited logs don't shadow live entries. A live stream's chunks
        // may span several segments (sealed ones plus the active), so group
        // them by segment — stripe coalescing runs within one segment file.
        let mut seen: HashSet<u64> = HashSet::new();
        let mut active_by_seg: BTreeMap<SegmentId, Vec<ChunkSource>> = BTreeMap::new();
        if let Some(s) = substrate.stream_of(stream_id) {
            for (&chunk_idx, loc) in &s.chunks {
                seen.insert(chunk_idx);
                active_by_seg
                    .entry(loc.segment)
                    .or_default()
                    .push(ChunkSource {
                        chunk_idx,
                        file_offset: loc.offset,
                        record_size: loc.record_size,
                    });
            }
        }
        // Then each inherited log, filling in indices not present in active.
        let mut per_inh: Vec<Vec<ChunkSource>> =
            self.inherited.iter().map(|_| Vec::new()).collect();
        for (i, inh) in self.inherited.iter().enumerate() {
            if let Some(s) = inh.substrate().stream_of(stream_id) {
                for (&chunk_idx, loc) in &s.chunks {
                    if seen.insert(chunk_idx) {
                        per_inh[i].push(ChunkSource {
                            chunk_idx,
                            file_offset: loc.offset,
                            record_size: loc.record_size,
                        });
                    }
                }
            }
        }

        // Per-source: sort by file_offset, build stripes. The active source
        // is split per segment; each segment's stripes read from its own file.
        let mut active_seg_stripes: Vec<(SegmentId, Vec<ChunkSource>, Vec<Stripe>)> = Vec::new();
        for (seg, mut chunks) in active_by_seg {
            chunks.sort_unstable_by_key(|c| c.file_offset);
            let stripes = build_stripes(&chunks);
            active_seg_stripes.push((seg, chunks, stripes));
        }
        for v in &mut per_inh {
            v.sort_unstable_by_key(|c| c.file_offset);
        }
        let inh_stripes: Vec<Vec<Stripe>> = per_inh.iter().map(|v| build_stripes(v)).collect();

        // Size the scratch buffer to the exact sum of stripe spans.
        let total: usize = active_seg_stripes
            .iter()
            .flat_map(|(_, _, st)| st.iter())
            .map(|s| s.len)
            .sum::<usize>()
            + inh_stripes
                .iter()
                .flat_map(|v| v.iter())
                .map(|s| s.len)
                .sum::<usize>();
        if buf.len() < total {
            buf.resize(total, 0);
        }

        // Walk each source's stripes; read each stripe once, then decode
        // every chunk that falls within it.
        let mut out: Vec<(u64, ChunkPayload)> = Vec::with_capacity(
            active_seg_stripes
                .iter()
                .map(|(_, c, _)| c.len())
                .sum::<usize>()
                + per_inh.iter().map(|v| v.len()).sum::<usize>(),
        );
        let mut buf_cur: usize = 0;

        // Active source, one segment file at a time.
        for (seg, chunks, stripes) in &active_seg_stripes {
            let mut chunk_iter = chunks.iter().peekable();
            for stripe in stripes {
                let region = &mut buf[buf_cur..buf_cur + stripe.len];
                self.segments.read_into(*seg, stripe.file_offset, region)?;
                let stripe_end = stripe.file_offset + stripe.len as u64;
                while let Some(c) = chunk_iter.peek() {
                    if c.file_offset >= stripe_end {
                        break;
                    }
                    let within = (c.file_offset - stripe.file_offset) as usize;
                    let start = buf_cur + within;
                    let end = start + c.record_size as usize;
                    let (header, payload_bytes, _) = decode_record(&buf[start..end])?;
                    let payload = ChunkPayload::decode(payload_bytes)?;
                    warn_on_chunk_golden_mismatch(&header, &payload, c.chunk_idx);
                    out.push((c.chunk_idx, payload));
                    chunk_iter.next();
                }
                buf_cur += stripe.len;
            }
        }

        // Each inherited source.
        for (i, chunks) in per_inh.iter().enumerate() {
            let stripes = &inh_stripes[i];
            let mut chunk_iter = chunks.iter().peekable();
            for stripe in stripes {
                let region = &mut buf[buf_cur..buf_cur + stripe.len];
                self.inherited[i].read_into(stripe.file_offset, region)?;
                let stripe_end = stripe.file_offset + stripe.len as u64;
                while let Some(c) = chunk_iter.peek() {
                    if c.file_offset >= stripe_end {
                        break;
                    }
                    let within = (c.file_offset - stripe.file_offset) as usize;
                    let start = buf_cur + within;
                    let end = start + c.record_size as usize;
                    let (header, payload_bytes, _) = decode_record(&buf[start..end])?;
                    let payload = ChunkPayload::decode(payload_bytes)?;
                    warn_on_chunk_golden_mismatch(&header, &payload, c.chunk_idx);
                    out.push((c.chunk_idx, payload));
                    chunk_iter.next();
                }
                buf_cur += stripe.len;
            }
        }

        out.sort_unstable_by_key(|(idx, _)| *idx);
        Ok(out)
    }

    /// Build a chunked-read plan for `stream_id`, sized so each
    /// chunk's bytes fit within `buffer_size`. Used by the cold-load
    /// orchestrator to stream a turn through a fixed-size pinned
    /// scratch — see [`chunk_plan`] for the partition semantics.
    pub fn plan_chunked_read(
        &self,
        substrate: &Substrate,
        stream_id: StreamId,
        buffer_size: usize,
    ) -> ChunkedReadPlan {
        let active_chunks = substrate.stream_of(stream_id).map(|s| &s.chunks);
        let inherited_chunks: Vec<Option<&std::collections::BTreeMap<u64, ChunkLoc>>> = self
            .inherited
            .iter()
            .map(|i| i.substrate().stream_of(stream_id).map(|s| &s.chunks))
            .collect();
        chunk_plan::plan_chunked_read(
            self.segments.active_id(),
            active_chunks,
            &inherited_chunks,
            buffer_size,
        )
    }

    /// Read a stream's latest `Tokens` record payload — from the active log,
    /// else any inherited log. `None` if the stream has no `Tokens` record.
    pub fn read_tokens(
        &mut self,
        substrate: &Substrate,
        stream_id: StreamId,
    ) -> Result<Option<Vec<u8>>> {
        if let Some(loc) = substrate.stream_of(stream_id).and_then(|s| s.tokens) {
            let record = self
                .segments
                .read_record_at(loc.segment, loc.offset, loc.record_size)?;
            return Ok(Some(record.payload));
        }
        for inherited in &self.inherited {
            if let Some(loc) = inherited
                .substrate()
                .stream_of(stream_id)
                .and_then(|s| s.tokens)
            {
                let record = inherited.read_record(loc.offset, loc.record_size)?;
                return Ok(Some(record.payload));
            }
        }
        Ok(None)
    }

    /// Flush and `fsync` the active segment — the group-commit durability
    /// point. Seals + rotates the active if it has reached the size target.
    pub fn commit(&mut self) -> Result<()> {
        self.segments.commit()?;
        self.maybe_rotate_active()
    }

    /// Bytes staged but not yet flushed to the active segment. Returns 0 when
    /// there is nothing to write. The periodic flush task uses this to
    /// avoid pointless `fsync` calls on an idle workspace.
    pub fn pending_bytes(&self) -> usize {
        self.segments.pending_len()
    }

    /// Group-commit if (and only if) there are staged records. Returns
    /// `Ok(true)` when a flush+fsync actually happened, `Ok(false)` for the
    /// no-op idle path. Cheap to call on a tight timer.
    pub fn commit_if_pending(&mut self) -> Result<bool> {
        if self.segments.pending_len() == 0 {
            return Ok(false);
        }
        self.segments.commit()?;
        self.maybe_rotate_active()?;
        Ok(true)
    }

    /// Seal the active segment and mint a fresh one when it has reached the
    /// size target (§12). Runs only at a commit boundary — the active is
    /// durable and has no staged records — so a segment is never split
    /// mid-commit. Completes the sealed segment's `HeaderIndex` chain first
    /// (fast recovery), then resets the chain state for the fresh active.
    fn maybe_rotate_active(&mut self) -> Result<()> {
        if !self.segments.should_rotate() {
            return Ok(());
        }
        self.seal_active()
    }

    /// Seal + rotate the active segment when its durable **plus staged** bytes
    /// have reached the size target — a byte-based check on the append path, not
    /// only at external commit boundaries. A long append batch (migration
    /// relocation, large-file ingest) stages many records between commits, so
    /// `should_rotate` alone — which fires only after a group-commit — lets the
    /// active overshoot the target by the whole batch (observed: ~8 GB segments
    /// against the 4 GB target). Checking `write_offset + pending_len` on every
    /// data append bounds each segment to ~target + one record and, as a side
    /// benefit, caps the in-memory staging buffer at ~target. [`Self::seal_active`]
    /// flushes the index chain and commits the staged records durable first, so
    /// the sealed segment is well-formed and the batch continues in a fresh
    /// active. Cheap on the common path: two field reads; the seal fires at most
    /// once per segment fill. Never called for `HeaderIndex` appends (the seal
    /// path emits those), so it can't recurse.
    fn rotate_if_over_target(&mut self) -> Result<()> {
        let projected = self.segments.write_offset() + self.segments.pending_len() as u64;
        if projected >= self.segments.target_bytes() {
            self.seal_active()?;
        }
        Ok(())
    }

    /// Seal the active segment and mint a fresh one, unconditionally. Completes
    /// the sealing segment's `HeaderIndex` chain (so its records recover via the
    /// chain fast path, not a full walk), rotates, and resets the chain state
    /// for the fresh active. The caller must have committed first; this flushes
    /// the index but assumes no un-flushed data records.
    pub fn seal_active(&mut self) -> Result<()> {
        self.flush_header_index()?;
        self.segments.seal_and_rotate()?;
        // The fresh active starts its own `HeaderIndex` chain — the sealed
        // segment keeps the chain the flush above completed.
        self.last_index = None;
        self.pending_index.clear();
        Ok(())
    }

    /// Set the model spec — last-writer-wins. Appends a fresh `ModelSpec`
    /// record only when the bytes differ from the latest on file. Returns
    /// `true` if a record was written.
    pub fn set_model_spec(&mut self, spec: &[u8]) -> Result<bool> {
        if self.model_spec.as_deref() == Some(spec) {
            return Ok(false);
        }
        self.append_record(RecordType::ModelSpec, 0, 0, 0, 0, 0, spec)?;
        self.model_spec = Some(spec.to_vec());
        Ok(true)
    }

    /// Set the projection template — last-writer-wins, like
    /// [`SubstratePersistence::set_model_spec`].
    pub fn set_template(&mut self, template: &[u8]) -> Result<bool> {
        if self.template.as_deref() == Some(template) {
            return Ok(false);
        }
        self.append_record(RecordType::Template, 0, 0, 0, 0, 0, template)?;
        self.template = Some(template.to_vec());
        Ok(true)
    }

    /// Set the model's `tokenizer.json` bytes — last-writer-wins, like
    /// [`SubstratePersistence::set_model_spec`]. Appends only when the
    /// bytes differ from the latest on file (compared by SHA-256).
    ///
    /// The full bytes are the `Tokenizer` record's payload, so the log is
    /// a self-contained substrate image (§5.7) — no companion files.
    /// Recovery never reads the payload (the walk skips it and the digest
    /// chain carries only its header), so the ~11 MB record costs one
    /// on-demand read at open to compute the change-detection hash and
    /// nothing else.
    pub fn set_tokenizer(&mut self, tokenizer: &[u8]) -> Result<bool> {
        let hash = sha256(tokenizer);
        if self.tokenizer_sha256 == Some(hash) {
            return Ok(false);
        }
        self.append_record(RecordType::Tokenizer, 0, 0, 0, 0, 0, tokenizer)?;
        self.tokenizer_sha256 = Some(hash);
        Ok(true)
    }

    /// Read the tokenizer bytes embedded in the active log's `Tokenizer`
    /// record. `Ok(None)` when the log has no such record yet (no model
    /// has ever called `set_tokenizer` against this substrate). The read
    /// CRC-verifies the payload, so bit rot surfaces as `BadChecksum`.
    pub fn read_tokenizer_bytes(&mut self) -> Result<Option<Vec<u8>>> {
        let Some(loc) = self.manifest.tokenizer else {
            return Ok(None);
        };
        let r = self
            .segments
            .read_record_at(loc.segment, loc.offset, loc.record_size)?;
        Ok(Some(r.payload))
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
    pub fn lookup_section(&self, substrate: &Substrate, addr: ContentAddress) -> Option<StreamRef> {
        let id = content_hash::section_stream_id(addr);
        if self.has_stream(substrate, id) {
            Some(StreamRef {
                stream_id: id,
                kind: StreamKind::PromptSection,
            })
        } else {
            None
        }
    }

    /// Whether `stream_id` is present in the active substrate or any
    /// inherited log.  The active source is read from `substrate.streams`
    /// (the in-RAM index built by the reload walk); inherited logs still
    /// carry their own manifest.
    pub fn has_stream(&self, substrate: &Substrate, stream_id: StreamId) -> bool {
        substrate.has_stream(stream_id) || self.inherited.iter().any(|i| i.has_stream(stream_id))
    }

    /// Whether the active log has accumulated enough dead weight to justify
    /// a compaction pass — the dead-byte ratio (§5.8) crossing
    /// [`COMPACTION_DEAD_RATIO_THRESHOLD`] on a log at least
    /// [`COMPACTION_MIN_LOG_BYTES`] long.
    ///
    /// Cheap enough to poll every persistence-thread pass: the dead-byte
    /// counter is maintained O(1) per append (see [`accounting`]), and the
    /// tombstoned-stream sum is a walk of the in-RAM stream index — no
    /// disk I/O.
    pub fn should_compact(&self, substrate: &Substrate) -> bool {
        let total = self.segments.total_record_bytes().unwrap_or(0);
        total >= COMPACTION_MIN_LOG_BYTES
            && self.dead_ratio(substrate) >= COMPACTION_DEAD_RATIO_THRESHOLD
    }

    /// Fraction of the store's record bytes that are dead weight, computed from
    /// the SAME per-segment liveness the background maintenance uses: `dead =
    /// Σ(segment_total − segment_live)` over the whole log. This is a real
    /// fraction in `[0, 1]` — it can never exceed 100%, and it drops as
    /// maintenance reclaims dead segments.
    ///
    /// It deliberately does NOT use the incremental `accounting.dead_bytes()`
    /// counter: that counter is a whole-log cumulative total reset only by the
    /// global `compact()` (a load-time / manual path), so under the daemon's
    /// per-segment maintenance it climbs without bound — every relocation append
    /// supersedes the moved-from copy and adds to it — and the ratio sails past
    /// 100% into meaningless territory. Liveness-derived is scope-consistent
    /// (numerator and denominator span the same segments) and self-correcting.
    pub fn dead_ratio(&self, substrate: &Substrate) -> f32 {
        let total = self.segments.total_record_bytes().unwrap_or(0);
        if total == 0 {
            return 0.0;
        }
        // `segment_liveness` sums the read-back records (chunks/tokens) the index
        // still references; everything else in the segments (superseded records,
        // tombstoned-timeline records, per-stream metadata) is dead weight. `live`
        // can't exceed `total` (each live record's bytes are in a counted
        // segment), so `dead` is non-negative; clamp guards inherited-log streams
        // whose bytes aren't in this log's `total`.
        let live: u64 = self.segment_liveness(substrate).values().sum();
        let dead = total.saturating_sub(live);
        (dead as f32 / total as f32).clamp(0.0, 1.0)
    }

    /// Compact the store — reclaim dead weight by rewriting only the **live**
    /// records into a fresh single segment, then dropping every prior segment.
    ///
    /// Quiesces in-flight writes, collects the live winners across every
    /// segment (routing each read to the segment holding it), writes them to
    /// a compacted scratch file, and adopts it as the sole active segment
    /// ([`SegmentedLog::adopt_compacted`]). The in-RAM manifest, substrate
    /// index, and metadata caches are rebuilt from the compacted segment.
    /// Inherited logs are untouched — a child never compacts a base it only
    /// reads (§13.5).
    pub fn compact(
        &mut self,
        substrate: &mut Substrate,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> Result<()> {
        // Coarse phase progress (5 phases) for the loading screen.
        let report = |phase: usize| {
            if let Some(p) = progress {
                p(phase, 5);
            }
        };
        report(0);

        // 1. Quiesce — every staged write is now durable on disk.
        self.segments.commit()?;
        report(1);

        // 2. Collect the live winners, in dependency order — sourced from
        //    substrate state (per-entity) + manifest singletons. Each record
        //    is read from the segment that physically holds it.
        let dir = self.segments.dir().to_path_buf();
        // Planning only — no disk reads; each read-back record carries its
        // source location, read back coalesced + staged verbatim in step 3.
        let live = compaction::collect_live_records(&self.manifest, substrate);
        report(2);

        // 3. Write the live set into a compacted scratch file (a fresh index
        //    chain interleaved), reading read-back records off the source
        //    segments in coalesced stripes, then adopt it as the sole active
        //    segment, dropping every prior segment.
        let scratch = dir.join(".compact");
        let (new_log, new_manifest) = {
            let segments = &mut self.segments;
            compaction::write_compacted_log(&scratch, &live, &mut |seg, off, dest| {
                segments.read_into(seg, off, dest)
            })?
        };
        report(3);
        self.segments.adopt_compacted(new_log, &scratch)?;
        report(4);

        // 4. Adopt the compacted manifest and refresh the metadata caches.
        //    `write_compacted_log` stamped every singleton with `FIRST_SEGMENT`,
        //    but `adopt_compacted` names the compacted segment with a fresh id
        //    (crash safety) and wrote ALL records into that one segment — so
        //    re-stamp the singletons at the actual active id before reading them.
        self.manifest = new_manifest;
        let adopted = self.segments.active_id();
        for loc in [
            &mut self.manifest.model_spec,
            &mut self.manifest.template,
            &mut self.manifest.tokenizer,
        ] {
            if let Some(loc) = loc {
                loc.segment = adopted;
            }
        }
        self.model_spec = self
            .manifest
            .model_spec
            .map(|loc| {
                self.segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)
                    .map(|r| r.payload)
            })
            .transpose()?;
        self.template = self
            .manifest
            .template
            .map(|loc| {
                self.segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)
                    .map(|r| r.payload)
            })
            .transpose()?;
        self.tokenizer_sha256 = self
            .manifest
            .tokenizer
            .map(|loc| {
                let r = self
                    .segments
                    .read_record_at(loc.segment, loc.offset, loc.record_size)?;
                Ok::<_, PersistenceError>(sha256(&r.payload))
            })
            .transpose()?;
        // 5. The substrate's stream / timeline state still holds offsets into
        //    the OLD segments. Reset the walker-built collections and replay
        //    the compacted segment to rebuild them with the new offsets.
        //    Per-turn KV residence and timeline registrations survive (not
        //    walker-built). The dead-weight accounting restarts from the fresh
        //    segment in the same pass (a just-compacted store is all-live).
        substrate.clear_walker_state();
        self.accounting.reset();
        // Compaction rewrote every live record (metadata included) into the fresh
        // single segment, so the previous drop-safety floor is meaningless. Clear
        // it — the next maintenance op re-establishes a floor with a fresh re-emit.
        self.resident_reemit_floor = None;
        // Metadata locations point at the OLD segments; rebuild them from the
        // freshly-compacted segment in the same recovery pass (mirrors the load
        // walk in `from_dir_with_sink`).
        self.metadata_locs.clear();
        let accounting = &mut self.accounting;
        let metadata_locs = &mut self.metadata_locs;
        let (last_index, tail_digests) = self.segments.recover_active_with_sink(|entry| {
            accounting.record(&entry.record.header, entry.size);
            record_metadata_loc(metadata_locs, entry);
            substrate.apply_walker_entry(entry);
        })?;
        // The compacted segment carries a fresh index chain; chain the next
        // flush onto it.
        self.last_index = last_index;
        self.pending_index = tail_digests;
        // 6. Every residence's cold tier still references the OLD offsets.
        //    Re-point them at the rebuilt stream index so a mid-session
        //    cold→hot elevation reads the right bytes.
        substrate.refresh_cold_refs();
        report(5);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use log_file::{LogFile, SUPERBLOCK_SIZE};
    use segment::FIRST_SEGMENT;
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
        // A fresh store mints the first active segment.
        assert!(dir.join(SUBSTRATE_DIR).join("seg-0000000001.log").exists());
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
    fn tokenizer_embedded_once_by_hash() {
        let dir = tmp_dir("tok_hash");
        let v1 = vec![1u8; 4096];
        let v2 = vec![2u8; 4096];
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(sp.set_tokenizer(&v1).unwrap(), "first tokenizer is written");
            assert!(
                !sp.set_tokenizer(&v1).unwrap(),
                "identical bytes are a hash-match no-op"
            );
            assert!(
                sp.set_tokenizer(&v2).unwrap(),
                "changed bytes are re-embedded"
            );
            assert_eq!(sp.tokenizer_sha256(), Some(sha256(&v2)));
            sp.commit().unwrap();
        }
        {
            // The hash is recovered from disk, so an unchanged tokenizer on a
            // fresh open does not re-embed.
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.tokenizer_sha256(), Some(sha256(&v2)));
            assert!(
                !sp.set_tokenizer(&v2).unwrap(),
                "reopened log recognises the unchanged tokenizer by hash"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The tokenizer bytes are the `Tokenizer` record's payload — the
    /// log is a self-contained substrate image with no companion files.
    /// (Payload bit-rot coverage lives with `read_record_at`, which
    /// CRC-verifies every record it returns.)
    #[test]
    fn tokenizer_bytes_live_in_the_log() {
        let dir = tmp_dir("tok_embed");
        let bytes = (0..8192u32).map(|i| (i % 251) as u8).collect::<Vec<u8>>();
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(sp.set_tokenizer(&bytes).unwrap());
            sp.commit().unwrap();
            let tok_loc = sp.manifest().tokenizer.expect("manifest has tokenizer loc");
            assert_eq!(
                tok_loc.payload_len,
                bytes.len() as u64,
                "Tokenizer record payload must be the full tokenizer bytes"
            );
            assert_eq!(sp.read_tokenizer_bytes().unwrap(), Some(bytes.clone()));
        }
        // Bytes survive a reopen, recovered from the record on demand.
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.tokenizer_sha256(), Some(sha256(&bytes)));
            assert_eq!(sp.read_tokenizer_bytes().unwrap(), Some(bytes.clone()));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn model_spec_survives_reopen() {
        let dir = tmp_dir("ms_reopen");
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.set_model_spec(b"the-model").unwrap();
            sp.set_template(b"the-template").unwrap();
            sp.commit().unwrap();
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
            segments: Vec::new(),
            tags: Vec::new(),
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
            let mut substrate = Substrate::new();
            let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert!(sp.has_stream(&substrate, turn_id));
            let entry = substrate.stream_of(turn_id).unwrap();
            assert_eq!(entry.committed_through, Some(0));
            assert!(entry.tokens.is_some());
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    // Fork inheritance (`open_concat` over a single-file base) predates the
    // segmented store: an inherited base is a read-only single-file log, but a
    // substrate is now written as a segment directory, so a base can no longer
    // be produced as one file. Inheriting a read-only segment *set* is a fork
    // redesign tracked separately; ignored until then.
    #[test]
    #[ignore = "fork inheritance over a segment set is a separate redesign"]
    fn open_concat_inherits_and_keeps_last_active() {
        let base_dir = tmp_dir("base");
        let child_dir = tmp_dir("child");

        // Build a base log with a section stream.
        let sec = section("shared_section", 0);
        let base_log = base_dir.join("base.log");
        {
            let mut base =
                SubstratePersistence::open_concat(std::slice::from_ref(&base_log)).unwrap();
            base.declare_stream(&sec).unwrap();
            base.commit().unwrap();
        }

        // A child inherits the base; its active segment set is its own.
        let child_log = child_dir.join(SUBSTRATE_DIR);
        let mut child =
            SubstratePersistence::open_concat(&[base_log.clone(), child_log.clone()]).unwrap();
        assert_eq!(child.inherited_count(), 1);

        // The section stream resolves through inheritance — a prefix-cache hit.
        // The active substrate has no streams; the resolve falls through to
        // the inherited substrate's stream index.
        let substrate = Substrate::new();
        let hit = child.lookup_section(
            &substrate,
            match &sec {
                StreamDecl::PromptSection(s) => s.address,
                _ => unreachable!(),
            },
        );
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
            segments: Vec::new(),
            tags: Vec::new(),
        });
        child.declare_stream(&child_turn).unwrap();
        child.commit().unwrap();
        // Drop child + re-open via the substrate-aware path so the
        // walker rebuilds substrate.streams; then assert that the
        // child has its own turn and the inherited base section.
        drop(child);
        let mut child_substrate = Substrate::new();
        let child =
            SubstratePersistence::open_concat(&[base_log.clone(), child_log.clone()]).unwrap();
        // open_concat populates the active substrate via the same
        // walker pass that builds the manifest.  We approximate it
        // here by walking the active log explicitly into substrate.
        // (open_concat doesn't have a *_with_substrate variant; the
        // production path uses open_in_with_substrate.)
        {
            let mut log = LogFile::open(&child_log).unwrap();
            let hint = log.superblock().last_index;
            recovery::recover_with_sink(&mut log, FIRST_SEGMENT, hint, |e| {
                child_substrate.apply_walker_entry(e)
            })
            .unwrap();
        }
        assert!(child_substrate.has_stream(child_turn.stream_id()));
        assert!(child.inherited_substrates()[0]
            .substrate()
            .has_stream(sec.stream_id()));

        InheritedSubstrate::forget(&base_log);
        std::fs::remove_dir_all(&base_dir).ok();
        std::fs::remove_dir_all(&child_dir).ok();
    }

    #[test]
    fn unknown_section_is_a_cache_miss() {
        let dir = tmp_dir("miss");
        let sp = SubstratePersistence::open_in(&dir).unwrap();
        let substrate = Substrate::new();
        let absent = ContentAddress {
            prefix_hash: content_hash::ContentHash { lo: 12345, hi: 678 },
            section_hash: content_hash::ContentHash { lo: 9, hi: 9 },
        };
        assert!(sp.lookup_section(&substrate, absent).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    fn chunk_payload(seed: u32) -> ChunkPayload {
        ChunkPayload {
            offset: seed as u16,
            k_formats: vec![4, 4, 4, 4],
            v_formats: vec![5, 5, 5, 5],
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
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.write_chunk(sid, 0, 32, 4, None, &payload).unwrap();
            sp.commit().unwrap();
            // After write, repoen so the walker rebuilds substrate.streams.
            drop(sp);
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), payload);
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
                sp.write_chunk(sid, i as u64, 32, 4, None, c).unwrap();
            }
            sp.commit_stream(sid, 2).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let read = sp.read_stream_chunks(&substrate, sid).unwrap();
            assert_eq!(read.len(), 3);
            for (i, (idx, c)) in read.iter().enumerate() {
                assert_eq!(*idx, i as u64);
                assert_eq!(c, &chunks[i]);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The write- and read-side golden checks are non-blocking. A chunk stored
    /// with a deliberately WRONG golden still writes (`append_record` logs an
    /// error but proceeds) and still reads back (`read_stream_chunks_batched`'s
    /// check warns, not errors) — so a bad byte-movement pipeline is flagged in
    /// the log without ever aborting a write or a load. (The mismatch detection
    /// itself is unit-tested in `record::tests::recompute_chunk_golden_*`.)
    #[test]
    fn wrong_golden_is_nonfatal_on_write_and_read() {
        let dir = tmp_dir("golden_nonfatal");
        let sid = StreamId(555);
        let payload = chunk_payload(9);
        // Sanity: the bogus golden really doesn't match the KV bytes.
        assert_ne!(
            candle::fletcher::fletcher32(&payload.kv_bytes),
            0xBAD0_C0DE,
            "test's bogus golden must differ from the true one"
        );
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            // Some(wrong) forces the stored golden to mismatch the bytes.
            sp.write_chunk(sid, 0, 32, 4, Some(0xBAD0_C0DE), &payload)
                .unwrap();
            sp.commit_stream(sid, 0).unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // Read is non-fatal despite the golden mismatch — payload comes back.
            let read = sp.read_stream_chunks(&substrate, sid).unwrap();
            assert_eq!(read.len(), 1);
            assert_eq!(read[0].1, payload);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// End-to-end index chain: appending past the flush threshold
    /// writes `HeaderIndex` records and publishes the superblock hint;
    /// a reopen recovers through the chain to identical state, seeds
    /// the accumulator with the un-indexed tail, and reads records at
    /// the recovered locations.
    #[test]
    fn header_index_chain_survives_reopen() {
        let dir = tmp_dir("index_chain");
        let sid = StreamId(4040);
        let n = INDEX_FLUSH_ENTRIES + 7; // one flush + an un-indexed tail
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(sp.last_index().is_none(), "fresh log has no chain");
            for i in 0..n {
                sp.write_chunk(sid, i as u64, 32, 4, None, &chunk_payload(i as u32))
                    .unwrap();
            }
            assert!(
                sp.last_index().is_some(),
                "crossing the threshold flushes an index record"
            );
            assert_eq!(sp.pending_index_len(), 7);
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(substrate.live_chunk_count(), n);
            assert!(
                sp.last_index().is_some(),
                "reopen recovers through the published chain"
            );
            assert_eq!(
                sp.pending_index_len(),
                7,
                "the un-indexed tail seeds the accumulator"
            );
            // The recovered chunk locations are exact — read one from
            // each side of the index flush boundary.
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), chunk_payload(0));
            assert_eq!(
                sp.read_chunk(&substrate, sid, (n - 1) as u64).unwrap(),
                chunk_payload((n - 1) as u32)
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A long append batch (many records staged before any external commit)
    /// must roll segments **on the append path**, bounding each to ~target —
    /// not overshoot by the whole batch. Regression: with only commit-boundary
    /// rotation the active grew to ~2× the target (observed 8 GB segments
    /// against the 4 GB target), because `maybe_rotate_active` runs after a
    /// group-commit and a whole migration/ingest batch stages between commits.
    #[test]
    fn append_batch_rotates_by_bytes_not_only_at_commit() {
        let dir = tmp_dir("append_rotate");
        let sid = StreamId(9091);
        // Tiny target so a modest batch crosses it many times.
        let target = 128 * 1024u64;
        let mut sp = SubstratePersistence::open_in(&dir).unwrap();
        sp.segments.set_target_bytes_for_test(target);

        // Write a batch far larger than the target with NO intervening commit.
        let n = 2000u32;
        for i in 0..n {
            sp.write_chunk(sid, i as u64, 32, 4, None, &chunk_payload(i))
                .unwrap();
        }
        // Rotation already fired on the append path — before we ever commit.
        // (Pre-fix: the whole batch stayed in one active until the end-commit,
        // giving ~1 segment; the fix seals a fresh one every ~target bytes.)
        let mid_batch_segments = sp.segment_count();
        assert!(
            mid_batch_segments > 4,
            "append-path rotation should have sealed several segments mid-batch, got {mid_batch_segments}"
        );
        sp.commit().unwrap();

        // Every sealed segment's on-disk size is bounded to ~target (+ one
        // record of slack): the append-path check rolls before overshoot, and
        // seal truncates the over-allocated tail so file size == record bytes.
        for &id in sp.segments.sealed_ids() {
            let bytes = sp.segments.sealed_record_bytes(id).unwrap();
            assert!(
                bytes <= target + 64 * 1024,
                "sealed segment {id:?} = {bytes} B exceeds target {target} B + slack"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The self-heal path: a log whose chain is unusable (garbage hint)
    /// falls back to the full walk, and — because the walk reported every
    /// record as an un-indexed digest — the open immediately back-fills a
    /// fresh chain, so the *next* open takes the chain path again. A
    /// pre-chain log pays the slow walk exactly once.
    #[test]
    fn broken_chain_backfills_at_open_and_heals() {
        let dir = tmp_dir("index_heal");
        let sid = StreamId(5050);
        let n = INDEX_FLUSH_ENTRIES + 3;
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            for i in 0..n {
                sp.write_chunk(sid, i as u64, 32, 4, None, &chunk_payload(i as u32))
                    .unwrap();
            }
            sp.commit().unwrap();
        }
        // Sabotage the hint: point it at a data record (wrong type), the
        // shape an out-of-band corruption or a retired-format superblock
        // produces.
        {
            let active = dir.join(SUBSTRATE_DIR).join("seg-0000000001.log");
            let mut log = LogFile::open(&active).unwrap();
            log.set_last_index((SUPERBLOCK_SIZE, 4096)).unwrap();
        }
        // Open 1: fallback walk, then back-fill — the whole log is
        // pending, which exceeds the threshold, so a fresh chain is
        // flushed before the open returns.
        {
            let mut substrate = Substrate::new();
            let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(substrate.live_chunk_count(), n);
            assert!(
                sp.last_index().is_some(),
                "the open back-fills a fresh chain"
            );
            assert_eq!(sp.pending_index_len(), 0, "everything is indexed now");
        }
        // Open 2: the back-filled chain is taken and state is identical.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(substrate.live_chunk_count(), n);
            assert!(sp.last_index().is_some());
            assert_eq!(sp.pending_index_len(), 0);
            assert_eq!(
                sp.read_chunk(&substrate, sid, (n - 1) as u64).unwrap(),
                chunk_payload((n - 1) as u32)
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compact_drops_dead_records_and_reopens_identical() {
        let dir = tmp_dir("compact");
        let sid = StreamId(55);
        let live_chunk = chunk_payload(7);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            // Dead weight: superseded ModelSpecs and stale chunk-0 bodies.
            for s in ["qwen3-8b", "qwen3-14b", "qwen3-30b"] {
                sp.set_model_spec(s.as_bytes()).unwrap();
            }
            sp.set_model_spec(b"qwen3-235b-live").unwrap();
            sp.set_template(b"the-template").unwrap();
            for seed in [97, 98, 99] {
                sp.write_chunk(sid, 0, 32, 4, None, &chunk_payload(seed))
                    .unwrap();
            }
            sp.write_chunk(sid, 0, 32, 4, None, &live_chunk).unwrap();
            sp.commit_stream(sid, 0).unwrap();
            sp.commit().unwrap();
            // Re-walk so substrate.streams sees the freshly-written
            // chunks (writes don't auto-apply to substrate).
            drop(sp);
            substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();

            assert!(
                sp.dead_ratio(&substrate) >= COMPACTION_DEAD_RATIO_THRESHOLD,
                "a log half dead weight wants compaction, ratio = {}",
                sp.dead_ratio(&substrate)
            );
            assert!(
                !sp.should_compact(&substrate),
                "a small log never auto-compacts regardless of its ratio"
            );
            sp.compact(&mut substrate, None).unwrap();

            // Caches and the manifest survive the swap.
            assert_eq!(sp.model_spec(), Some(b"qwen3-235b-live".as_slice()));
            assert_eq!(sp.template(), Some(b"the-template".as_slice()));
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), live_chunk);
            // `dead_ratio` is liveness-derived (consistent with the maintenance
            // trigger), and `segment_liveness` intentionally counts only the
            // read-back records (chunks/tokens + manifest singletons) as live, not
            // the per-stream metadata (`StreamDecl`/`Commit`/…). So a freshly
            // compacted log reads slightly non-zero — the metadata-liveness
            // blindspot — but well under the compaction threshold (no reclaimable
            // chunk/token weight remains). In production, where KV chunks dominate,
            // this residual is negligible.
            assert!(
                sp.dead_ratio(&substrate) < COMPACTION_DEAD_RATIO_THRESHOLD,
                "a freshly compacted log has no reclaimable dead weight, ratio = {}",
                sp.dead_ratio(&substrate)
            );
            assert!(!sp.should_compact(&substrate));
        }
        // The compacted file recovers to the same live substrate on reopen.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert_eq!(sp.model_spec(), Some(b"qwen3-235b-live".as_slice()));
            assert_eq!(sp.template(), Some(b"the-template".as_slice()));
            assert_eq!(sp.read_chunk(&substrate, sid, 0).unwrap(), live_chunk);
        }
        assert!(
            !dir.join(SUBSTRATE_DIR).join(".compact").exists(),
            "the compaction scratch file is renamed away, not left behind"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[ignore = "fork inheritance over a segment set is a separate redesign"]
    fn chunk_resolves_from_inherited_log() {
        let base_dir = tmp_dir("ck_base");
        let child_dir = tmp_dir("ck_child");
        let base_log = base_dir.join("base.log");
        let sid = StreamId(9000);
        let payload = chunk_payload(11);
        {
            let mut base =
                SubstratePersistence::open_concat(std::slice::from_ref(&base_log)).unwrap();
            base.write_chunk(sid, 0, 32, 4, None, &payload).unwrap();
            base.commit().unwrap();
        }
        let child_log = child_dir.join(SUBSTRATE_DIR);
        let mut child = SubstratePersistence::open_concat(&[base_log.clone(), child_log]).unwrap();
        // The chunk lives only in the inherited base — read_chunk
        // resolves it via the inherited substrate's stream index.
        let substrate = Substrate::new();
        assert_eq!(child.read_chunk(&substrate, sid, 0).unwrap(), payload);

        InheritedSubstrate::forget(&base_log);
        std::fs::remove_dir_all(&base_dir).ok();
        std::fs::remove_dir_all(&child_dir).ok();
    }
}
