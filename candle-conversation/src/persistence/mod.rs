//! Substrate persistence layer — the three-tier KV-cache storage path.
//!
//! This is the generalized, mandatory persistence layer specified by
//! `docs/archived/kv_tier_migration.md`: an append-only NVMe redo log of
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
//! - [`survival`] — how each record type outlives a rewrite (exhaustive).
//! - [`chunk_window_integrity`] — the generic per-layer chunk window-geometry
//!   comparison, shared by every caller that needs to ask "do these layers
//!   describe the same K/V windows".
//! - [`inherit`] — multi-log inheritance and the shared cache.
//!
//! [`SubstratePersistence`] is the public API tying them together.

pub mod accounting;
pub mod adoption;
#[cfg(test)]
mod adoption_tests;
pub mod appended;
pub mod blob;
pub mod chunk_plan;
pub mod chunk_window_integrity;
pub mod cold_load;
pub mod cold_source;
pub mod compaction;
pub mod content_hash;
#[cfg(test)]
mod convergence_tests;
mod dead_ratio;
pub mod deleted;
pub mod elevate;
#[cfg(test)]
mod expiry_tests;
pub mod flush;
pub mod header_index;
mod held_groups;
mod held_types;
pub mod identity;
pub mod inherit;
pub mod integrity;
pub mod joined;
#[cfg(test)]
mod joined_tests;
mod le_reader;
pub mod listing;
pub mod liveness_audit;
pub mod log_file;
pub mod maintenance;
pub mod manifest;
mod model_binding;
pub mod open_progress;
pub mod owner;
pub mod payload_parts;
pub mod pipeline;
pub mod rebase;
#[cfg(test)]
mod rebase_tests;
pub mod record;
pub mod recovery;
pub mod relocation_watch;
pub mod remote;
pub mod replica;
#[cfg(test)]
mod replica_tests;
pub mod resume;
pub mod rewritten;
pub mod sealed_reader;
pub mod segment;
pub mod segmented_log;
pub mod space;
pub mod stamped_locs;
pub mod streams;
mod stripes;
pub mod survival;
pub mod thread;
pub mod tombstone_expiry;
pub mod transfer;
pub mod triggers;
mod vanished_segment;
pub mod vfs;
pub mod walker;
pub mod writer;
pub mod writer_lock;
pub mod written_snapshots;

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::Arc;

use dead_ratio::dead_ratio_of;
use std::time::Duration;

use thiserror::Error;

use crate::projection::TimelineId;
use crate::substrate::pcoll::POrdMap;
use crate::substrate::{file_store_key, Substrate, SubstrateLock, TurnMark};

use accounting::RecordAccounting;
use adoption::Adoption;
use appended::Appended;
use blob::{retired_blob, BlobIndex, BlobKey};
use chunk_plan::ChunkedReadPlan;
use compaction::Housekeeping;
use content_hash::snapshot_stream_id;
use header_index::{encode_index_payload, IndexEntry, INDEX_FLUSH_ENTRIES};
use identity::{
    load_or_mint, now_ms, read_node, MarkerIds, MintGeneration, Minter, NodeSource, Origination,
    OwnSections, RecordIdentity, Stamped, VersionStamp, GEN_DETACHED,
};
use inherit::InheritedSubstrate;
use joined::Purge;
use liveness_audit::{keyed_by_payload, resident_key, ResidentKey};
use manifest::{ChunkLoc, Manifest, RecordLoc};
pub use model_binding::{Admit, BoundSubstrate, ModelBinding};
use open_progress::MibMeter;
pub use open_progress::{OpenProgress, OpenProgressHook, OpenStage};
pub use owner::PersistenceOwner;
use rebase::{RebaseEntry, RebaseMap};
use record::verify_record_crc;
use record::{
    decode_header, decode_record, encode_record, encode_record_written, padded_record_len,
    ChunkPayload, CustomObjectPayload, DebugIdPayload, NpcPayload, Record, RecordHeader,
    RecordType, SectionGeneration, SectionTombstonePayload, TombstonePayload, TreeMetadataPayload,
    HEADER_PROBE_SIZE,
};
use relocation_watch::RelocationWatch;
use replica::{AppliedSet, DeferredRecord, Located, LocationIndex, OriginApplied, ReplicaPayload};
use rewritten::Rewritten;
use sealed_reader::SealedReader;
use segment::SegmentId;
use segmented_log::{store_mib, SegmentedLog};
use stamped_locs::StampedLocs;
use streams::{ContentAddress, StreamDecl, StreamId, StreamKind, StreamRef};
use survival::RecordCensus;
use tombstone_expiry::{TombstoneExpiry, DEFAULT_TOMBSTONE_RETENTION};
use vanished_segment::retry_vanished;
use vfs::{VfsEventId, VfsEventPayload, VfsIndex, VfsWrite};
use walker::WalkEntry;
use writer_lock::WriterLock;

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

    /// A write refused because the substrate was opened read-only
    /// ([`SharedSubstrate::open_in_read_only`]). Nothing was staged.
    #[error("the substrate is open read-only; this operation would write to it")]
    ReadOnly,

    /// The substrate was written by another model
    /// ([`SubstratePersistence::bind_model`]). Nothing was read or written.
    #[error(
        "the substrate at {} belongs to {recorded}, not {running}. Everything \
         model-derived in it — K/V, recurrent state, token ids — was computed by \
         that model and is wrong under any other. Start on a new substrate, or move \
         this one aside.",
        dir.display()
    )]
    WrongModel {
        dir: PathBuf,
        recorded: String,
        running: String,
    },

    /// Another writable handle holds the substrate
    /// ([`writer_lock::WriterLock`]). Nothing was opened.
    #[error(
        "the substrate at {} is already open for writing — by another daemon on this \
         workspace, or another handle in this process. A substrate takes one writer: a \
         second would lose records at the next compaction.",
        dir.display()
    )]
    Locked { dir: PathBuf },
}

/// Result type for the persistence layer.
pub type Result<T> = std::result::Result<T, PersistenceError>;

/// The name of the per-working-directory persistence subdirectory. Holds the
/// segmented redo log (`seg-*.log` sealed, one `seg-*.active`) — see
/// [`segmented_log`]. It is visible, not a dot-directory: it sits in a
/// workspace folder beside the repositories it serves, not inside one of them.
/// Every path to it is built from this constant, never from a literal.
pub const SUBSTRATE_DIR: &str = "substrate";

/// A plain character record, for the tests in this module and in
/// [`maintenance`] that need one to survive something.
#[cfg(test)]
pub(crate) fn test_npc(id: u64, name: &str) -> NpcPayload {
    NpcPayload {
        npc_id: id,
        owner_id: "u1".to_string(),
        revision: 1,
        created_ms: 1_000,
        updated_ms: 1_000,
        state: "active".to_string(),
        name: name.to_string(),
        world_id: "battle-cities".to_string(),
        personality_id: "maker".to_string(),
        hidden: false,
        heartbeat_ms: 60_000,
        salience_gate: 0.5,
        tags: Vec::new(),
        persona_description: String::new(),
        persona_origin: "authored".to_string(),
        portrait_image_id: None,
        portrait_origin: None,
        at: None,
        mood: None,
        beliefs: Vec::new(),
        relationships: Vec::new(),
        agency: Vec::new(),
        modulation: record::Modulation::default(),
    }
}

/// Every entry under `dir`, recursively, as `(relative path, length,
/// SHA-256)` sorted by path — the before/after the read-only tests hold a
/// store to. A directory appears as its path with a trailing `/`, length 0 and
/// a zero digest, so a created directory shows up as well as a created file.
#[cfg(test)]
pub(crate) fn dir_fingerprint(dir: &Path) -> Vec<(String, u64, [u8; 32])> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, u64, [u8; 32])>) {
        for entry in std::fs::read_dir(dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            let rel = path
                .strip_prefix(root)
                .expect("entry under the root")
                .to_string_lossy()
                .replace('\\', "/");
            if path.is_dir() {
                out.push((format!("{rel}/"), 0, [0u8; 32]));
                walk(root, &path, out);
            } else {
                let bytes = std::fs::read(&path).expect("read file");
                out.push((rel, bytes.len() as u64, sha256(&bytes)));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

/// A substrate the host process opened, for the engine to adopt rather than
/// open a second time.
///
/// **One writable handle per `substrate/`.** A second [`SubstratePersistence`]
/// over the same directory would be a second append cursor *and* a second
/// [`SubstratePersistence::npc_locs`] view. The two writers would choose offsets
/// neither told the other about, and — because
/// [`compaction::collect_live_records`] can only carry forward what the
/// compacting handle's own walk recorded — a compaction would drop every
/// header-keyed record the other handle appended after it opened. A writable
/// open therefore holds [`writer_lock::WriterLock`], and a second one is refused.
///
/// That is not theoretical: `npcd` opened one handle for its character registry
/// and let the engine open another for conversation turns, and every character
/// created while the daemon ran vanished at the next restart. The ones that
/// predated both opens survived, which made the loss look intermittent instead
/// of mechanical. See
/// [`tests::a_second_writable_handle_is_refused`].
///
/// So a host that writes records of its own — `npcd` writes
/// [`RecordType::Npc`] — opens the substrate **once** and hands this pair over,
/// instead of naming a directory and letting the engine open its own. Both
/// sides then append through the same cursor and compact against the same view.
#[derive(Clone)]
pub struct SharedSubstrate {
    pub substrate: Arc<SubstrateLock>,
    pub persistence: Arc<PersistenceOwner>,
}

/// Names the directory rather than the contents. A substrate's interesting
/// state is megabytes of index and arena bookkeeping, and this appears inside
/// the `Debug` for a whole engine config — the useful fact at that scale is
/// *which* log it is.
impl std::fmt::Debug for SharedSubstrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SharedSubstrate")
            .field(&self.persistence.dir())
            .finish()
    }
}

impl SharedSubstrate {
    /// Wrap a freshly-opened pair: the substrate's versioned root, and the log
    /// handed to a thread of its own ([`PersistenceOwner`]). The caller does the
    /// walk — typically through
    /// [`SubstratePersistence::open_in_with_substrate_and_sink`], so its own
    /// record classes are collected in the same pass.
    pub fn new(substrate: Substrate, persistence: SubstratePersistence) -> Self {
        let substrate = Arc::new(SubstrateLock::new(substrate));
        let persistence = PersistenceOwner::spawn(persistence, Arc::clone(&substrate));
        Self {
            substrate,
            persistence,
        }
    }

    /// Open the substrate under `dir` for `admit`'s model and share it. The
    /// one-liner for a host with no records of its own to collect on the walk.
    pub fn open_in(dir: &Path, admit: Admit<'_>) -> Result<Self> {
        Self::open_in_reporting(dir, admit, &|_, _, _| {})
    }

    /// [`Self::open_in`], telling `progress` how the open goes — a store written
    /// before identities is rewritten whole before it returns.
    pub fn open_in_reporting(
        dir: &Path,
        admit: Admit<'_>,
        progress: OpenProgress<'_>,
    ) -> Result<Self> {
        let mut substrate = Substrate::new();
        let persistence = SubstratePersistence::open_adopting(
            &dir.join(SUBSTRATE_DIR),
            &mut substrate,
            admit,
            |_| {},
            progress,
        )?;
        Ok(Self::new(substrate, persistence))
    }

    /// Open the substrate under `dir` **read-only** and share it.
    ///
    /// For a tool that reads a workspace another process — the daemon — may be
    /// writing to at the same time. The store must already exist, and nothing
    /// under `dir/substrate/` is created, renamed, deleted, truncated, grown or
    /// written for as long as the pair lives. Everything in RAM behaves as it
    /// does on a writable substrate; only the durable side is absent — see
    /// [`SubstratePersistence::open_in_with_substrate_read_only`].
    pub fn open_in_read_only(dir: &Path) -> Result<Self> {
        let mut substrate = Substrate::new();
        let persistence =
            SubstratePersistence::open_in_with_substrate_read_only(dir, &mut substrate)?;
        Ok(Self::new(substrate, persistence))
    }

    /// Store a keyed [`CustomObjectPayload`] — durably and in RAM in one call.
    ///
    /// A later write with the same key supersedes this one (the greater stamp
    /// wins): the header carries the key's hash as its `stream_id`, and the
    /// in-RAM [`Substrate`] holds only the winner, under the identity its
    /// record carries, so compaction re-emits it. This is the write half of the
    /// durable host state (e.g. npcd's command-table open flag);
    /// [`Self::custom_object`] reads it back. A blob's tombstone
    /// ([`BlobKey::tombstone_key`], ceased) retires the blob here at once.
    pub fn put_custom_object(&self, obj: CustomObjectPayload) -> Result<()> {
        let retires = retired_blob(&obj);
        let id = self.persistence.call(|p| {
            let id = p.mint();
            p.write_custom_object(&obj, id)?;
            p.commit()?;
            Ok::<_, PersistenceError>(id)
        })?;
        self.substrate
            .update(|s| s.apply_custom_object(obj.clone(), id));
        self.forget_retired_blobs(retires);
        Ok(())
    }

    /// Forget each of `keys` whose tombstone now outranks the blob held —
    /// after the tombstones are durable and applied.
    fn forget_retired_blobs(&self, keys: impl IntoIterator<Item = BlobKey>) {
        let keys: Vec<BlobKey> = keys.into_iter().collect();
        if keys.is_empty() {
            return;
        }
        let s = self.substrate.read();
        self.persistence.call(|p| {
            for key in keys {
                p.forget_retired_blob(&s, key);
            }
        });
    }

    /// Retire the blob `sha256` names: write its tombstone
    /// ([`BlobKey::tombstone_key`], ceased), which replicates like any keyed
    /// one. It reads as absent at once, and its bytes go at the next
    /// compaction or maintenance pass over them (`docs/replicated_substrate.md`
    /// §10.3).
    pub fn retire_blob(&self, sha256: &[u8; 32]) -> Result<()> {
        self.put_custom_object(CustomObjectPayload::ceased(
            BlobKey::of(sha256).tombstone_key(),
        ))
    }

    /// Read the live [`CustomObjectPayload`] for `key`, or `None` if none was
    /// ever written, or its winning record is its tombstone
    /// ([`CustomObjectPayload::ceased`]).
    pub fn custom_object(&self, key: &str) -> Option<CustomObjectPayload> {
        self.substrate.read().custom_object(key).cloned()
    }

    /// Store `objs` — values and tombstones alike — under the lease epoch
    /// `epoch`, in one commit (`docs/replicated_substrate.md` §9.1): a world's
    /// authority writes every entity that changed in a beat at once, each
    /// stamped with its lease's epoch so no earlier holder's write outranks it.
    pub fn put_custom_objects_under(
        &self,
        objs: Vec<CustomObjectPayload>,
        epoch: u64,
    ) -> Result<()> {
        if objs.is_empty() {
            return Ok(());
        }
        let ids = self.persistence.call(|p| {
            let mut ids = Vec::with_capacity(objs.len());
            for obj in &objs {
                let id = p.mint_under(epoch);
                p.write_custom_object(obj, id)?;
                ids.push(id);
            }
            p.commit()?;
            Ok::<_, PersistenceError>(ids)
        })?;
        let retires: Vec<BlobKey> = objs.iter().filter_map(retired_blob).collect();
        self.substrate.update(|s| {
            for (obj, &id) in objs.iter().zip(&ids) {
                s.apply_custom_object(obj.clone(), id);
            }
        });
        self.forget_retired_blobs(retires);
        Ok(())
    }

    /// Every live custom object whose key starts with `prefix`, in key order,
    /// with the stamp of the record that holds it.
    pub fn custom_objects_under(&self, prefix: &str) -> Vec<(CustomObjectPayload, VersionStamp)> {
        self.substrate
            .read()
            .custom_objects_under(prefix)
            .into_iter()
            .map(|(obj, id)| (obj, id.stamp()))
            .collect()
    }

    /// How many live custom objects' keys start with `prefix`.
    pub fn custom_objects_count_under(&self, prefix: &str) -> usize {
        self.substrate.read().custom_objects_count_under(prefix)
    }

    /// The live custom object under `prefix` whose key sorts last.
    pub fn custom_object_last_under(&self, prefix: &str) -> Option<CustomObjectPayload> {
        self.substrate.read().custom_object_last_under(prefix)
    }

    /// A count that moves whenever a custom object changes: read a prefix
    /// again only when it has.
    pub fn custom_objects_version(&self) -> u64 {
        self.substrate.read().custom_objects_version()
    }

    /// Write `write` as `timeline`'s file events, then its tombstones, in
    /// one commit; the ids the events took (`docs/zend_vfs_events.md`).
    pub fn write_vfs(&self, timeline: u64, write: &VfsWrite) -> Result<Vec<VfsEventId>> {
        self.persistence.call(|p| {
            let ids = p.write_vfs(timeline, write)?;
            p.commit()?;
            Ok(ids)
        })
    }

    /// Every live file event of `timeline`, in `(repo, key)` order and within
    /// a key in stamp order.
    pub fn vfs_events(&self, timeline: u64) -> Result<Vec<VfsEventPayload>> {
        self.persistence.call(|p| p.vfs_events(timeline))
    }

    /// Every timeline holding file events or tombstones, in id order.
    pub fn vfs_timelines(&self) -> Vec<u64> {
        self.persistence.call(|p| {
            p.vfs_index()
                .timelines()
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
    }

    /// Declare `timeline` a file store, so every rewrite of the log carries its
    /// events though no conversation is registered on it
    /// ([`Substrate::is_file_store`]) — once: a store already declared writes
    /// nothing.
    pub fn declare_file_store(&self, timeline: TimelineId) -> Result<()> {
        if self.substrate.read().is_file_store(timeline) {
            return Ok(());
        }
        self.put_custom_object(CustomObjectPayload::new(
            file_store_key(timeline),
            serde_json::Value::Null,
        ))
    }

    /// Keep `bytes` as a blob, durably, unless one of their hash is already
    /// held; their SHA-256 either way (`docs/replicated_substrate.md` §10.3).
    pub fn put_blob(&self, bytes: Vec<u8>) -> Result<[u8; 32]> {
        self.persistence.call(|p| {
            let sha256 = p.write_blob(bytes)?;
            p.commit_if_pending()?;
            Ok(sha256)
        })
    }

    /// The bytes named `sha256`, or `None` when no blob of that hash is held.
    pub fn blob(&self, sha256: &[u8; 32]) -> Result<Option<Vec<u8>>> {
        self.persistence.call(|p| p.blob(sha256))
    }

    /// Whether a blob of `sha256` is held, without reading it: a pack of
    /// 64 MiB is known to have arrived without being read
    /// (`docs/substrate_git.md` §4.1).
    pub fn has_blob(&self, sha256: &[u8; 32]) -> bool {
        self.persistence.call(|p| p.has_blob(sha256))
    }
}

/// The persistence layer behind a substrate — owns the active redo log, the
/// inherited read-only logs, and the in-RAM manifest.
///
/// Persistence is mandatory: a substrate cannot exist without one.
///
/// **Exactly one of these may exist per `substrate/` directory in a process.**
/// See [`SharedSubstrate`] for what a second one costs and how a host that
/// needs its own writes avoids opening it.
pub struct SubstratePersistence {
    /// The segmented redo log — the active append segment plus the sealed
    /// segment set under `substrate/`. Replaces the single monolithic log:
    /// reads route to the segment holding each record by `(segment, offset)`.
    segments: SegmentedLog,
    manifest: Manifest,
    inherited: Vec<Arc<InheritedSubstrate>>,
    /// The latest `ModelKey` record's payload — the model this substrate is
    /// bound to ([`SubstratePersistence::bind_model`]).
    model_record: Option<Vec<u8>>,
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
    /// The dead ratio the last liveness scan found, as `f32` bits — NaN before
    /// the first ([`Self::scanned_dead_ratio`]).
    scanned_dead_ratio: AtomicU32,
    /// The writes made since the running maintenance op was planned that its
    /// relocation must step around — see [`RelocationWatch`]. `Some` from the
    /// plan until the op's sources are unlinked (or the op fails), so it also
    /// marks an op in flight: a handle runs one at a time.
    relocation_watch: Option<RelocationWatch>,
    /// On-disk location of each stream's CURRENT per-stream metadata record —
    /// `StreamDecl` / `ProjectionEvents` / `WideQSig` / `Commit` — keyed by
    /// [`MetadataKey`], the greatest stamp. The substrate index caches
    /// these payloads but not their locations, so without this map
    /// [`segment_liveness`](SubstratePersistence::segment_liveness) would count
    /// them as dead — making the segment that holds the live metadata perpetually
    /// look reclaimable and get "compacted" (re-emitted forward) on every
    /// maintenance pass. Counting them as live via this map is what actually
    /// halts that churn. Populated on every append and rebuilt on load / compact.
    metadata_locs: StampedLocs<MetadataKey>,
    /// On-disk location of the CURRENT copy of each resident record keyed by
    /// its payload rather than its header — `Label`, `ConvState`,
    /// `TreeMetadata`, `TurnCoupling`, `DebugId`, `Tombstone`, `Distilled`,
    /// `SectionTombstone` — keyed by [`ResidentKey`], the greatest stamp.
    ///
    /// The twin of `metadata_locs` for the types it cannot key. Maintenance
    /// re-emits all of them on every op, and with no location to count they
    /// read dead wherever they landed: every re-emission left its copies dead
    /// weight in the segment it wrote, so that segment looked reclaimable and
    /// the next op re-emitted them again. Populated on every append and rebuilt
    /// on load / compact, like `metadata_locs`.
    resident_locs: StampedLocs<ResidentKey>,
    /// On-disk location of each conversation's CURRENT recurrent-state
    /// `Snapshot` record, keyed by snapshot stream id — the greatest stamp, the
    /// persistence-side twin of the substrate's `recurrent_snapshots` index.
    /// Snapshot records are superseded by stamp, and the seal thread's
    /// writer can append a newer tail between a maintenance plan and its
    /// execute — so the execute phase (which runs WITHOUT the substrate lock)
    /// consults this map to skip relocating a planned snapshot that is no
    /// longer the stream's live tail. A timeline tombstone removes its entry
    /// (a dead conversation has no live tail). Populated on every append
    /// (both encode and verbatim paths) and rebuilt on load / compact.
    snapshot_locs: StampedLocs<u64>,

    /// On-disk location of each character's CURRENT [`RecordType::Npc`] record,
    /// keyed by `npc_id` (which the header carries as its `stream_id`) — the
    /// greatest stamp.
    ///
    /// This map is what keeps characters alive. NPC payloads do not live in the
    /// substrate's RAM — the substrate holds no opinion about a character, and
    /// the registry that does is in another process's memory — so compaction
    /// cannot re-synthesise them the way it re-synthesises a `Label`. Instead
    /// the winner is relocated verbatim, exactly as `Snapshot` is, and this map
    /// says which record that is. Without it, `collect_live_records` would omit
    /// every NPC and the first compaction would delete the entire cast.
    npc_locs: StampedLocs<u64>,

    /// Where every conversation's live file events and tombstones are —
    /// `docs/zend_vfs_events.md`. Their bodies belong to the daemon above and
    /// are never in RAM here, so like `npc_locs` this map is what keeps them:
    /// compaction and maintenance carry what it names, and a resume reads it.
    vfs_index: VfsIndex,

    /// Where every blob is (`blob`). Their bytes are never in RAM here: this
    /// map is what keeps them, as `vfs_index` keeps the file events.
    blob_index: BlobIndex,

    /// Per-type record counts as the store stood when it opened — the baseline
    /// a rewrite's carry-forward tally is reported against, so a class that
    /// stops being carried is named at the moment it happens rather than
    /// inferred later from a missing feature. Set once at open and never
    /// updated: it is a *baseline*, and re-basing it after a rewrite would erase
    /// the very comparison it exists for.
    open_census: RecordCensus,
    /// Every record type the substrate has held since it opened, as bit `tag`:
    /// the open walk's census, and each type appended since — written here or
    /// applied from another member. What a build replacing this one must
    /// decode, or it would not see them (`docs/replicated_substrate.md` §3.9).
    held_types: u32,
    /// The exclusive lock a writable handle holds on its directory
    /// ([`WriterLock`]) until it drops or [`Self::release_writer`] hands the
    /// directory on; `None` on a read-only handle.
    writer_lock: Option<WriterLock>,
    /// What this substrate stamps on the records it originates — its node, the
    /// next `oseq`, its clock and its generation — primed at open from every
    /// record the walk met (`docs/replicated_substrate.md` §6.2). Shared with
    /// the runtime write paths, which mint as they change the substrate.
    minter: Minter,
    /// How the node came to be this handle's: kept from the directory, minted
    /// fresh, or re-minted because the directory was copied (§3.7). `None` on a
    /// read-only handle, which mints nothing.
    node_source: Option<NodeSource>,
    /// The identity of each marker record a rewrite writes back — see
    /// [`MarkerIds`]. Built by the open walk and kept by every append.
    marker_ids: MarkerIds,
    /// How long a timeline tombstone is kept before a compaction may shrink it
    /// to its deleted-set entry (`--tombstone-retention`, §6.8).
    tombstone_retention: Duration,
    /// What this member has applied of every other origin, and the records it
    /// deferred for a type this build does not decode (§6.2, §6.5). Its own
    /// records are covered by its own `oseq`s ([`Self::applied_through`]).
    /// Rebuilt at open from the last `Replica` record and every record the walk
    /// meets.
    replica: ReplicaPayload,
    /// The replica changed since a `Replica` record last said what it holds:
    /// the next commit writes one.
    replica_dirty: bool,
    /// Where every record with an identity is, kept current through every
    /// append, relocation and rewrite (§6.2).
    locations: LocationIndex,
    /// The generations fixed for what this node wrote cut off (§6.7), applied
    /// to its own records wherever it reads them back.
    rebases: RebaseMap,
    /// The records the open met that were written before identities, by type:
    /// any at all, and the store is to be adopted ([`adoption`]).
    unstamped: RecordCensus,
    /// Told the `oseq` of every record this node writes that may be sent, once
    /// each, as it is appended — the replication tap (§6.3).
    tap: Option<Tap>,
    /// This node's own records written and sendable: everything it holds but
    /// what it wrote cut off and no rejoin has fixed. A rewrite or relocation
    /// of a record already here is not tapped again.
    own_written: OriginApplied,
    /// The top of `own_written` when the log was opened. A gap at or below it
    /// is one a purge left; one above it is a record this process minted that
    /// has not reached the log yet ([`Self::own_sendable_through`]).
    open_top: u64,
    /// `oseq`s this process minted whose append failed: never coming, so no
    /// hold on [`Self::own_sendable_through`].
    own_unwritten: OriginApplied,
    /// A write failed for lack of space (§7.7) — see [`space`].
    out_of_space: bool,
}

/// Told the `oseq` of each record this node writes that may be sent.
pub type Tap = Box<dyn Fn(u64) + Send + Sync>;

/// What [`SubstratePersistence::metadata_locs`] keys a record by: its type, the
/// header's `stream_id`, and the writer's origin for a stream's own records —
/// two members' turn *N* share a stream id and are two streams
/// (`docs/replicated_substrate.md` §6.6). A `CustomObject` carries its key's
/// hash in `stream_id`, and any writer's record of a key competes for it, so
/// its writer is zero.
pub(crate) type MetadataKey = (RecordType, u64, u64);

/// The [`MetadataKey`] of a record with header `h`.
fn metadata_key(h: &RecordHeader) -> MetadataKey {
    metadata_key_parts(h.record_type, h.stream_id, h.origin)
}

/// The [`MetadataKey`] of a record of type `rt` on `stream_id`, written by
/// `origin`.
fn metadata_key_parts(rt: RecordType, stream_id: u64, origin: u64) -> MetadataKey {
    let writer = match rt {
        RecordType::CustomObject
        | RecordType::Deleted
        | RecordType::Replica
        | RecordType::Rebase => 0,
        _ => origin,
    };
    (rt, stream_id, writer)
}

/// Whether a record type is a per-stream metadata record whose current-copy
/// location is tracked in [`SubstratePersistence::metadata_locs`] so it counts
/// as live weight (see that field). These are the records the maintenance
/// resident-set re-emits and that carry no location in the substrate index.
fn is_tracked_metadata(rt: RecordType) -> bool {
    // `Snapshot` is deliberately NOT here: its payload never lives in RAM, so
    // the maintenance resident-set (which re-encodes from RAM) cannot carry
    // it. It is tracked through the substrate index and relocated verbatim
    // like `Tokens` — see `segment_liveness` and `gather_relocations`.
    matches!(
        rt,
        RecordType::StreamDecl
            | RecordType::ProjectionEvents
            | RecordType::WideQSig
            | RecordType::TurnIndexPage
            | RecordType::Commit
            | RecordType::CustomObject
            | RecordType::Deleted
            | RecordType::Replica
            | RecordType::Rebase
    )
}

/// The location of a walked record.
fn walked_loc(entry: &walker::WalkEntry) -> RecordLoc {
    RecordLoc {
        segment: entry.segment,
        offset: entry.offset,
        payload_len: entry.record.header.payload_len,
        record_size: entry.size,
    }
}

/// Offer a walked payload-keyed resident record's location to `map` — the
/// greatest stamp per key holds it. The load / compact walk uses this; the
/// runtime append path uses [`SubstratePersistence::track_resident_loc`].
fn record_resident_loc(map: &mut StampedLocs<ResidentKey>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    if !keyed_by_payload(h.record_type) {
        return;
    }
    let id = h.identity();
    if let Some(key) = resident_key(h.record_type, h.stream_id, &entry.record.payload, &id) {
        map.offer(key, Some(walked_loc(entry)), id);
    }
}

/// Offer a walked record's metadata location to `map` — the greatest stamp per
/// key holds it. The load / compact walk uses this before the
/// `SubstratePersistence` exists; the runtime append path uses
/// [`SubstratePersistence::track_metadata_loc`].
fn record_metadata_loc(map: &mut StampedLocs<MetadataKey>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    if is_tracked_metadata(h.record_type) {
        map.offer(metadata_key(h), Some(walked_loc(entry)), h.identity());
    }
}

/// `entry` with its identity fixed: a record this node wrote cut off carries
/// the generation a rejoin fixed for it ([`RebaseMap`]); any other record is
/// itself.
fn fix_walked<'a>(entry: &'a WalkEntry, rebases: &RebaseMap) -> Cow<'a, WalkEntry> {
    let id = entry.record.header.identity();
    let fixed = rebases.fix(id);
    if fixed == id {
        return Cow::Borrowed(entry);
    }
    let mut owned = entry.clone();
    owned.record.header = owned.record.header.with_identity(fixed);
    Cow::Owned(owned)
}

/// Take one walked record into the replica and the location index: every
/// record with an identity is located, another origin's is applied, and a
/// `Replica` record's sets join what is held. Both only grow, so the order the
/// walk meets them in does not matter.
///
/// A `Replica` record's entry for `node` itself is not the replica's: it is
/// this node's own written set as the record saw it, its gaps what a purge
/// dropped (§7.8). The last record walked is the latest written, so it alone
/// says which gaps still stand — `own_kept` takes its entry, or `None` when it
/// has none.
fn observe_replica(
    replica: &mut ReplicaPayload,
    locations: &mut LocationIndex,
    own_kept: &mut Option<OriginApplied>,
    node: u64,
    entry: &walker::WalkEntry,
) {
    let h = &entry.record.header;
    let id = h.identity();
    if !id.is_none() {
        locations.insert(
            h,
            Located {
                segment: entry.segment,
                offset: entry.offset,
            },
        );
        if id.origin != node {
            replica.applied.insert(id.origin, id.oseq);
        }
    }
    if h.record_type == RecordType::Replica {
        if let Ok(mut held) = ReplicaPayload::decode(&entry.record.payload) {
            *own_kept = held.applied.remove(node);
            replica.applied.merge(&held.applied);
            replica.deferred.extend(held.deferred);
        }
    }
}

/// Take off the deferred set every record whose type this build decodes: it is
/// wanted again, so the next catch-up fetches it (§6.5). Returns whether any
/// was.
fn release_known_deferred(replica: &mut ReplicaPayload) -> bool {
    let before = replica.deferred.len();
    replica
        .deferred
        .retain(|d| RecordType::from_name(&d.record_type) == RecordType::Unknown);
    replica.deferred.len() != before
}

/// Offer one walked record to the NPC-location map — the greatest stamp per
/// character holds it. Keyed by the header's `stream_id`, which for an `Npc`
/// record is the `npc_id` — so the walk never has to decode a payload to know
/// which character it belongs to.
///
/// A tombstoned character keeps its entry: `state: "tombstoned"` is a
/// superseding record like any other, and the id must stay taken so the acts
/// that already name it still resolve. Dropping it here would let a later
/// character reuse the id.
fn record_npc_loc(map: &mut StampedLocs<u64>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    if h.record_type == RecordType::Npc {
        map.offer(h.stream_id, Some(walked_loc(entry)), h.identity());
    }
}

/// Mirror one walked record into the snapshot-location map: a `Snapshot`
/// record offers its location and a retraction withdraws it, the greatest
/// stamp per stream holding; a timeline-scoped `Tombstone` removes the
/// timeline's entry, so a snapshot record replayed before its timeline's
/// tombstone never reads as live. The load / compact walk uses this before
/// the `SubstratePersistence` exists; the runtime append path uses
/// [`SubstratePersistence::track_snapshot_loc`] plus the removal in
/// [`SubstratePersistence::write_tombstone`].
fn record_snapshot_loc(map: &mut StampedLocs<u64>, entry: &walker::WalkEntry) {
    let h = &entry.record.header;
    match h.record_type {
        RecordType::Snapshot if h.retracts_snapshot() => {
            map.offer(h.stream_id, None, h.identity());
        }
        RecordType::Snapshot | RecordType::BranchCheckpoint => {
            map.offer(h.stream_id, Some(walked_loc(entry)), h.identity());
        }
        RecordType::Tombstone => {
            if let Ok(p) = TombstonePayload::decode(&entry.record.payload) {
                if p.turn_index.is_none() {
                    map.remove(&snapshot_stream_id(p.timeline_id).0);
                }
            }
        }
        // A `Distilled` marker retires the tail for the same reason a tombstone
        // does: the recurrent state IS content, derived from a token stream that
        // distillation has shed, and useless without the K/V it was computed
        // against. A distilled conversation is never resumed — that is what
        // distillation means — so the state describes a resume that cannot
        // happen.
        //
        // Dropping it HERE, at the map, rather than at each rewrite is what
        // makes the two rewrite paths agree without either having to remember.
        // They did not: `collect_live_records` gated on the distilled set while
        // incremental maintenance relocated the same tails forward on every
        // pass, forever, on the reasoning that "the substrate map only ever
        // holds live conversations' tails" — true of tombstone, which removes
        // the entry, and false of distill, which did not. Measured on a
        // production store: 766 distilled timelines carrying ~116 MiB of state
        // each, ~89 GB of a 120 GB log, none of it resumable.
        RecordType::Distilled => {
            if let Ok(p) = record::DistillPayload::decode(&entry.record.payload) {
                map.remove(&snapshot_stream_id(p.timeline_id).0);
            }
        }
        _ => {}
    }
}

/// The first eight bytes of a digest, hex — enough to name which tokenizer a
/// substrate is bound to in an error a person has to act on.
fn hex16(digest: &[u8; 32]) -> String {
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of `bytes` — the tokenizer identity digest.
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

/// Staged bytes at which an append writes the group-commit buffer through to
/// the active segment. Writing through is not a commit — nothing is synced
/// until the next [`SubstratePersistence::commit`] — it only stops the buffer
/// from holding everything between commits. Commits are paced by
/// [`INDEX_FLUSH_ENTRIES`] records, and maintenance relocates chunk records of
/// ~165 KB each, so an unbounded buffer reached 2.1 GB inside one compaction
/// and aborted the daemon on the allocation.
pub const STAGE_FLUSH_BYTES: usize = 64 * 1024 * 1024;

impl SubstratePersistence {
    /// Open the persistence layer at `<cwd>/substrate/substrate.log`,
    /// creating the directory and file if absent and recovering the
    /// manifest if present.
    pub fn open() -> Result<SubstratePersistence> {
        let cwd = std::env::current_dir()?;
        SubstratePersistence::open_in(&cwd)
    }

    /// Open the persistence layer at `<dir>/substrate/` (the segment set).
    pub fn open_in(dir: &Path) -> Result<SubstratePersistence> {
        Self::from_dir_with_sink(&dir.join(SUBSTRATE_DIR), &[], false, Purge::Nothing, |_| {})
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
    ///
    /// A store written before identities is adopted here, once ([`adoption`]).
    /// Opened for no model ([`Admit::Unbound`]): a host that runs one binds
    /// through [`SharedSubstrate::open_in`].
    pub fn open_in_with_substrate(
        dir: &Path,
        substrate: &mut Substrate,
    ) -> Result<SubstratePersistence> {
        Self::open_adopting(
            &dir.join(SUBSTRATE_DIR),
            substrate,
            Admit::Unbound,
            |_| {},
            &|_, _, _| {},
        )
    }

    /// The writable open with a substrate: walk the store into `substrate`
    /// and `sink`, refuse it when `admit`'s model cannot open it, and when it
    /// was written before identities, adopt it and open it again into a fresh
    /// substrate. The refusal comes before the adoption, so a refused store is
    /// left as it was found. `sink` sees the walk once: the adoption changes
    /// record headers and the payloads that name a writer — turn tombstones and
    /// file events — and leaves every other payload as it was.
    ///
    /// `progress` is told the stage it is in and how far through it the open
    /// is: MiB of log walked against the store's size for a walk, records
    /// rewritten of the live set for the adoption ([`OpenStage::unit`]).
    ///
    /// A member away too long ([`joined`]) walks without what it can no longer
    /// trust — `sink` never sees it — and compacts it away before it returns.
    fn open_adopting<F>(
        dir: &Path,
        substrate: &mut Substrate,
        admit: Admit<'_>,
        mut sink: F,
        progress: OpenProgress<'_>,
    ) -> Result<Self>
    where
        F: FnMut(&WalkEntry),
    {
        let last_joined = joined::read(dir)?;
        let purge = Purge::after(last_joined, now_ms());
        let total = store_mib(dir);
        progress(OpenStage::Reading, 0, total);
        let mut meter = MibMeter::default();
        let mut sp = Self::from_dir_with_sink(dir, &[], false, purge, |entry| {
            substrate.apply_walker_entry(entry);
            sink(entry);
            if let Some(mib) = meter.add(entry.size) {
                progress(OpenStage::Reading, mib.min(total), total);
            }
        })?;
        progress(OpenStage::Reading, total, total);
        // Before anything below rewrites the log, so a rewrite does not carry
        // a blob its tombstone retired.
        sp.forget_retired_blobs(substrate);
        sp.admit(admit)?;
        if sp.predates_identities() {
            sp.adopt(substrate, &|done, total| {
                progress(OpenStage::Stamping, done, total)
            })?;
            drop(sp);
            *substrate = Substrate::new();
            let total = store_mib(dir);
            progress(OpenStage::Rereading, 0, total);
            let mut meter = MibMeter::default();
            sp = Self::from_dir_with_sink(dir, &[], false, Purge::Nothing, |entry| {
                substrate.apply_walker_entry(entry);
                if let Some(mib) = meter.add(entry.size) {
                    progress(OpenStage::Rereading, mib.min(total), total);
                }
            })?;
            progress(OpenStage::Rereading, total, total);
            sp.forget_retired_blobs(substrate);
        } else if purge.drops_any() {
            sp.compact(
                substrate,
                Some(&|done, total| progress(OpenStage::Purging, done, total)),
            )?;
        }
        // Judged now: a store never judged starts its clock here, and one just
        // purged starts it again rather than purging at every open until the
        // replicator next marks it joined.
        if last_joined.is_none() || purge.drops_any() {
            joined::write(dir, now_ms())?;
        }
        Ok(sp)
    }

    /// Whether the open met records written before identities.
    fn predates_identities(&self) -> bool {
        self.unstamped.total() > 0
    }

    /// As [`Self::open_in_with_substrate`], but **read-only**: the segment set
    /// is opened with [`SegmentedLog::open_read_only_with_sink`], so nothing
    /// under `<dir>/substrate/` is ever created, renamed, deleted, truncated,
    /// grown or written through this handle, and the store must already exist.
    ///
    /// Every durable operation keeps its signature and answers without
    /// touching disk: the appends refuse with [`PersistenceError::ReadOnly`]
    /// and stage nothing; `commit`, `commit_if_pending` and `seal_active`
    /// succeed with nothing to do; `set_template` reports that nothing was
    /// written; `bind_model` and `set_tokenizer` still refuse a model or
    /// vocabulary other than the recorded one but never record one;
    /// maintenance plans no op and compaction refuses.
    pub fn open_in_with_substrate_read_only(
        dir: &Path,
        substrate: &mut Substrate,
    ) -> Result<SubstratePersistence> {
        // Beside a live writer a segment can be compacted away mid-open; the
        // open starts over from an empty substrate (`vanished_segment`).
        let dir = dir.join(SUBSTRATE_DIR);
        let substrate = RefCell::new(substrate);
        let mut sp = retry_vanished(
            || {
                Self::from_dir_with_sink(&dir, &[], true, Purge::Nothing, |entry| {
                    substrate.borrow_mut().apply_walker_entry(entry)
                })
            },
            || **substrate.borrow_mut() = Substrate::new(),
        )?;
        sp.forget_retired_blobs(&substrate.borrow());
        Ok(sp)
    }

    /// As [`Self::open_in_with_substrate`], but the caller also sees every
    /// record as it is walked.
    ///
    /// For record classes the substrate does not interpret — [`RecordType::Npc`]
    /// is the one that exists today — this is how their owner rebuilds its
    /// index without a second pass over the log. Reading them back through
    /// [`Self::npc_locs`] afterwards would work too, but costs one seek per
    /// character where this costs none: the bytes are already in hand.
    ///
    /// The sink sees records in log order, so last-writer-wins is simply the
    /// last value the sink is handed for a key. A store `admit`'s model cannot
    /// open is refused; one written before identities is adopted here, once,
    /// after the sink has seen it ([`Self::open_adopting`]).
    pub fn open_in_with_substrate_and_sink<F>(
        dir: &Path,
        substrate: &mut Substrate,
        admit: Admit<'_>,
        sink: F,
        progress: OpenProgress<'_>,
    ) -> Result<SubstratePersistence>
    where
        F: FnMut(&WalkEntry),
    {
        Self::open_adopting(&dir.join(SUBSTRATE_DIR), substrate, admit, sink, progress)
    }

    /// Open over an ordered list of paths. The last entry is the active,
    /// writable **segment directory** (`substrate/`); every earlier entry is
    /// an inherited read-only single-file log, loaded through the shared
    /// cache (§13.5).
    pub fn open_concat(logs: &[PathBuf]) -> Result<SubstratePersistence> {
        let (active_dir, inherited) = logs.split_last().ok_or_else(|| {
            PersistenceError::Corrupt("open_concat needs at least one log".into())
        })?;
        Self::from_dir_with_sink(active_dir, inherited, false, Purge::Nothing, |_| {})
    }

    /// Open the segment set in `dir` (the `substrate/` directory) with the
    /// listed inherited single-file logs, driving every recovered record
    /// through `sink` in the same pass that builds the manifest and the
    /// dead-weight accounting. `read_only` opens the segment set with
    /// [`SegmentedLog::open_read_only_with_sink`] instead of
    /// [`SegmentedLog::open_with_sink`].
    ///
    /// A record `purge` drops is walked past as if it were not there — but for
    /// the clock and the `oseq` counter, which it still advances, so this node
    /// never mints an identity it issued before. The replica then forgets every
    /// other origin, and this node's own dropped `oseq`s are gaps in what it
    /// has written, for a catch-up to fill ([`joined`]).
    fn from_dir_with_sink<F>(
        dir: &Path,
        inherited: &[PathBuf],
        read_only: bool,
        purge: Purge,
        mut sink: F,
    ) -> Result<SubstratePersistence>
    where
        F: FnMut(&walker::WalkEntry),
    {
        // One writer per substrate: taken before a byte is read, so a refused
        // open costs nothing. A read-only handle takes none.
        let writer_lock = if read_only {
            None
        } else {
            Some(WriterLock::acquire(dir)?)
        };
        // The node, once the directory is ours: a read-only handle takes the
        // one the directory names, if any, and mints nothing.
        let (node, node_source) = if read_only {
            (read_node(dir)?.unwrap_or(0), None)
        } else {
            let (node, source) = load_or_mint(dir)?;
            if let NodeSource::Reminted { previous } = source {
                tracing::warn!(
                    target: "candle_conversation::persistence::identity",
                    previous = format!("{previous:016x}"),
                    node = format!("{node:016x}"),
                    dir = %dir.display(),
                    "this substrate was minted in another directory or on another machine — \
                     it writes under a fresh node from here on, so it can never issue a \
                     record its original also issues"
                );
            }
            (node, Some(source))
        };
        let mut origination = Origination::new(node);
        let mut own_sections = OwnSections::default();
        let mut replica = ReplicaPayload::default();
        let mut locations = LocationIndex::default();
        let mut marker_ids = MarkerIds::new();
        let mut inherited_subs = Vec::with_capacity(inherited.len());
        for path in inherited {
            inherited_subs.push(InheritedSubstrate::load(path)?);
        }

        // Recover every segment (ascending id, active last) — feeding the
        // dead-weight accounting and the caller's sink in the same pass that
        // builds the combined singleton manifest.
        let mut accounting = RecordAccounting::new();
        let mut metadata_locs = StampedLocs::new();
        let mut resident_locs = StampedLocs::new();
        let mut snapshot_locs = StampedLocs::new();
        let mut npc_locs = StampedLocs::new();
        let mut vfs_index = VfsIndex::new();
        let mut blob_index = BlobIndex::new();
        // What the store actually holds, by type. One array increment per record
        // in a walk that already visits every record — see `RecordCensus`.
        let mut census = RecordCensus::new();
        // What this node wrote cut off carries the provisional generation until
        // a rejoin fixed it, and the `Rebase` record that says so is written
        // after the records it fixes. So those records are held back until the
        // walk has read every `Rebase`, then applied with the generation fixed
        // for them — nothing reaches the substrate, or any index, unfixed.
        let rebases = RefCell::new(RebaseMap::default());
        let held: RefCell<Vec<WalkEntry>> = RefCell::new(Vec::new());
        let draining = Cell::new(false);
        let mut unstamped = RecordCensus::new();
        let mut own_kept = None;
        let mut dropped_own = OriginApplied::default();
        let mut dropped_foreign = 0u64;
        let mut walk = |entry: &WalkEntry| {
            let h = &entry.record.header;
            if adoption::predates_identities(h) {
                unstamped.record(h.record_type);
            }
            if h.record_type == RecordType::Rebase {
                if let Ok(map) = RebaseMap::decode(&entry.record.payload) {
                    rebases.borrow_mut().merge(&map);
                }
            }
            if h.gen == GEN_DETACHED && h.origin == node && !draining.get() {
                held.borrow_mut().push(entry.clone());
                return;
            }
            let fixed = fix_walked(entry, &rebases.borrow());
            let entry = fixed.as_ref();
            let h = &entry.record.header;
            let id = h.identity();
            origination.observe(&id);
            if purge.drops(node, &id) {
                if id.origin == node {
                    dropped_own.insert(id.oseq);
                } else {
                    dropped_foreign += 1;
                }
                return;
            }
            own_sections.observe(node, h, &entry.record.payload);
            observe_replica(&mut replica, &mut locations, &mut own_kept, node, entry);
            marker_ids.record(
                h.record_type,
                h.stream_id,
                &entry.record.payload,
                h.identity(),
            );
            accounting.record(&entry.record.header, &entry.record.payload, entry.size);
            census.record(entry.record.header.record_type);
            record_metadata_loc(&mut metadata_locs, entry);
            record_resident_loc(&mut resident_locs, entry);
            record_snapshot_loc(&mut snapshot_locs, entry);
            record_npc_loc(&mut npc_locs, entry);
            vfs::record_vfs_loc(&mut vfs_index, &mut accounting, entry);
            blob::record_blob_loc(&mut blob_index, entry);
            sink(entry);
        };
        let opened = if read_only {
            SegmentedLog::open_read_only_with_sink(dir, &mut walk)?
        } else {
            SegmentedLog::open_with_sink(dir, &mut walk)?
        };
        draining.set(true);
        let held = held.take();
        // A detached record no `Rebase` covers was written since this node was
        // last cut off and has not rejoined: it is cut off still, from the
        // first such record.
        let still_detached = held
            .iter()
            .map(|e| rebases.borrow().fix(e.record.header.identity()))
            .filter(|id| id.gen == GEN_DETACHED)
            .map(|id| id.oseq)
            .min();
        for entry in &held {
            walk(entry);
        }
        let rebases = rebases.into_inner();
        if still_detached.is_some() {
            origination.detach(still_detached);
        }
        // Every own record the store holds may be sent, but what it wrote cut
        // off and no rejoin fixed (§6.3, §6.7), and what a purge dropped and no
        // catch-up has returned yet — the gaps the replica's own entry kept,
        // and the ones this open just made (§7.8).
        let mut own_missing = dropped_own;
        if let Some(kept) = own_kept {
            for (start, end) in kept.gaps() {
                own_missing.insert_range(start, end);
            }
        }
        if purge.drops_any() {
            replica = ReplicaPayload::default();
            tracing::warn!(
                target: "candle_conversation::persistence::joined",
                node = format!("{node:016x}"),
                dir = %dir.display(),
                ?purge,
                dropped_foreign,
                own_missing = own_missing.ranges().iter().map(|(s, e)| e - s + 1).sum::<u64>(),
                "this member was away longer than a tombstone may be missed — dropping what \
                 it can no longer trust; a catch-up refetches what the cluster still holds"
            );
        }
        let own_top = still_detached.unwrap_or(origination.next_oseq()) - 1;
        let own_written = OriginApplied::from_top_and_gaps(
            own_top,
            own_missing
                .ranges()
                .into_iter()
                .filter(|(start, _)| *start <= own_top)
                .map(|(start, end)| (start, end.min(own_top))),
        );
        let segmented_log::OpenedSegments {
            mut segments,
            manifest,
            last_index,
            tail_digests,
            recovered_records,
        } = opened;
        // The census at open is the "before" of every before/after: compare two
        // restarts and a class that a rewrite dropped in between reads as a
        // count that went to zero.
        tracing::info!(
            target: "candle_conversation::persistence::census",
            total = census.total(),
            "substrate census at open: {}",
            census.summary()
        );

        let model_record = manifest
            .model_key
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
            model_record,
            tokenizer_sha256,
            template,
            accounting,
            pending_index: tail_digests,
            last_index,
            recovered_records,
            last_maintenance: None,
            scanned_dead_ratio: AtomicU32::new(f32::NAN.to_bits()),
            relocation_watch: None,
            metadata_locs,
            resident_locs,
            snapshot_locs,
            npc_locs,
            vfs_index,
            blob_index,
            held_types: census.held_mask(),
            open_census: census,
            writer_lock,
            minter: Minter::with_sections(origination, own_sections.live()),
            node_source,
            marker_ids,
            tombstone_retention: DEFAULT_TOMBSTONE_RETENTION,
            replica_dirty: (release_known_deferred(&mut replica) || purge.drops_any())
                && !read_only,
            replica,
            locations,
            rebases,
            unstamped,
            tap: None,
            open_top: own_written.top(),
            own_unwritten: OriginApplied::default(),
            own_written,
            out_of_space: false,
        };
        // Self-heal a large un-indexed tail (a crash window, or a log
        // that predates the index chain entirely): flush it now so the
        // next open takes the chain path instead of re-walking it. A
        // read-only handle leaves the tail as it found it — the next
        // writable open heals it.
        if !read_only && sp.pending_index.len() >= INDEX_FLUSH_ENTRIES {
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

    /// This substrate's node — the origin of every record it writes (§3.7).
    /// Zero on a read-only handle over a directory that never had one.
    pub fn node(&self) -> u64 {
        self.minter.node()
    }

    /// How this handle came by its node; `None` on a read-only handle.
    pub fn node_source(&self) -> Option<NodeSource> {
        self.node_source
    }

    /// The writer's identity state as it stands: the next `oseq`, the clock,
    /// the generation.
    pub fn origination(&self) -> Origination {
        self.minter.origination()
    }

    /// The shared handle the runtime write paths mint through, so a value set
    /// in memory holds from the start the identity its record will carry.
    pub fn minter(&self) -> Minter {
        self.minter.clone()
    }

    /// The identity of the next record this substrate originates.
    pub fn mint(&self) -> RecordIdentity {
        self.minter.mint(0)
    }

    /// The identity of the next record this substrate originates under the
    /// claim or lease epoch `epoch` (§6.2, §9.2).
    pub fn mint_under(&self, epoch: u64) -> RecordIdentity {
        self.minter.mint(epoch)
    }

    /// The identity of the next record of type `rt` this substrate writes
    /// under stream `stream_id` — carrying the section's generation when the
    /// stream is a section this node is writing ([`Minter::mint_for`]).
    pub fn mint_for(&self, rt: RecordType, stream_id: StreamId) -> RecordIdentity {
        self.minter.mint_for(rt, stream_id.0)
    }

    /// The identity of each marker record a rewrite writes back.
    pub fn marker_ids(&self) -> &MarkerIds {
        &self.marker_ids
    }

    /// How long a timeline tombstone is kept before a compaction may shrink it
    /// to its deleted-set entry — the member's `--tombstone-retention`. Only
    /// when its bytes are reclaimed depends on it; the timeline stays dead.
    pub fn set_tombstone_retention(&mut self, retention: Duration) {
        self.tombstone_retention = retention;
    }

    /// The logical end of what is written to the active segment — synced only
    /// once committed.
    pub fn write_offset(&self) -> u64 {
        self.segments.write_offset()
    }

    /// The active (append target) segment's id — the segment fresh appends
    /// land in.
    pub fn active_segment(&self) -> SegmentId {
        self.segments.active_id()
    }

    /// Whether this handle refuses writes — opened read-only
    /// ([`Self::open_in_with_substrate_read_only`]), or released
    /// ([`Self::release_writer`]).
    pub fn is_read_only(&self) -> bool {
        self.segments.is_read_only()
    }

    /// Commit what is staged, then stop writing and give up the directory's
    /// [`WriterLock`], so the next writer can open it while this handle is
    /// still alive. From here on the handle behaves as a read-only one: every
    /// write refuses, and reads go on. Idempotent; a read-only handle has
    /// nothing to release.
    ///
    /// An engine's `shutdown` ends here. The engine can outlive it — another
    /// thread still holding it — and its successor opens the same workspace in
    /// the meantime; nothing the shut-down engine does may reach the log the
    /// successor writes.
    pub fn release_writer(&mut self) -> Result<()> {
        if self.is_read_only() {
            return Ok(());
        }
        self.commit()?;
        self.segments.close_for_writing();
        self.writer_lock = None;
        Ok(())
    }

    /// Number of inherited logs.
    pub fn inherited_count(&self) -> usize {
        self.inherited.len()
    }

    /// Append one record to the active segment, updating the in-RAM manifest.
    /// Returns where it landed and the identity it carries — callers that
    /// index the record (cold-load, chunk persistence) need the segment so the
    /// read routes back to the right file, the padded on-disk size for the
    /// bytes-on-disk footprint, and the identity the substrate keeps beside
    /// the value.
    ///
    /// A read-only handle refuses with [`PersistenceError::ReadOnly`] and
    /// stages nothing.
    ///
    /// **The record is this substrate's own**: it is stamped with a fresh
    /// identity — this node as origin, the next `oseq`, the current generation
    /// and a clock reading (`docs/replicated_substrate.md` §6.2) — except a
    /// `HeaderIndex`, which no origin writes. A copy of an existing record goes
    /// through [`Self::append_carried`] instead and keeps the identity it has.
    pub fn append_record(
        &mut self,
        record_type: RecordType,
        format: u8,
        stream_id: u64,
        chunk_index: u64,
        token_count: u64,
        golden: u32,
        payload: &[u8],
    ) -> Result<Appended> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        let identity = if record_type == RecordType::HeaderIndex {
            RecordIdentity::NONE
        } else {
            self.minter.mint_for(record_type, stream_id)
        };
        // Write-boundary validation of this member's own seal: recompute the
        // chunk's golden over the host bytes about to be written and compare
        // it with the golden the seal computed on the GPU, before the DtoH
        // copy. A mismatch means the copy or host handling corrupted the KV
        // bytes — a real byte-movement fault. Log it (a rebuild regenerates
        // the substrate; this proves the pipeline) and still write what we
        // have. A record carried with its identity already decided — another
        // member's, checked by its digest on arrival, or a rewrite of bytes
        // this store already holds — has no copy of its own to prove, and the
        // golden is checked again on the device when the chunk is loaded.
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
        self.append_carried(
            record_type,
            format,
            stream_id,
            chunk_index,
            token_count,
            golden,
            payload,
            identity,
        )
    }

    /// [`Self::append_record`] for a record whose identity is already decided:
    /// a resident value re-encoded from RAM by a rewrite, a singleton carried
    /// forward, or a runtime write whose identity was minted when the value
    /// changed in memory ([`Self::minter`]). The record keeps `identity`, so a
    /// copy is recognised everywhere as the record it copies and ranks where
    /// that record did.
    pub fn append_carried(
        &mut self,
        record_type: RecordType,
        format: u8,
        stream_id: u64,
        chunk_index: u64,
        token_count: u64,
        golden: u32,
        payload: &[u8],
        identity: RecordIdentity,
    ) -> Result<Appended> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
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
            ..RecordHeader::NONE
        }
        .with_identity(identity);
        let (written, bytes) = encode_record_written(&header, payload);
        let (segment, offset) = self.segments.stage(&bytes);
        let size = bytes.len() as u64;
        self.account_appended(&written, payload, segment, offset, size)?;
        Ok(Appended {
            segment,
            offset,
            size,
            identity,
        })
    }

    /// Append a recurrent-state record — a [`RecordType::Snapshot`] or a
    /// [`RecordType::BranchCheckpoint`] — from its payload's parts, written
    /// straight to the active segment rather than staged. The bookkeeping is
    /// [`Self::append_record`]'s.
    ///
    /// Only these two types come through here: they are the multi-megabyte
    /// records, and none of the append hooks reads their payload, so the parts
    /// never need joining. Any other type is refused.
    fn append_recurrent_parts(
        &mut self,
        record_type: RecordType,
        stream_id: u64,
        parts: &[&[u8]],
    ) -> Result<Appended> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        assert!(
            matches!(
                record_type,
                RecordType::Snapshot | RecordType::BranchCheckpoint
            ),
            "append_recurrent_parts writes recurrent-state records only, not {record_type:?}"
        );
        let header = RecordHeader {
            record_type,
            format: 0,
            payload_len: parts.iter().map(|p| p.len() as u64).sum(),
            crc: 0,
            stream_id,
            chunk_index: 0,
            token_count: 0,
            ..RecordHeader::NONE
        }
        .with_identity(self.minter.mint(0));
        let head = record::record_head(&header, parts);
        let appended = self.segments.append_parts(&head.line, parts, head.pad);
        let (segment, offset) = self.watch_space(appended)?;
        let size = (head.line.len() + header.payload_len as usize + head.pad) as u64;
        // The hooks key these types by header alone, so the payload they are
        // handed is never read.
        self.account_appended(&head.header, &[], segment, offset, size)?;
        Ok(Appended {
            segment,
            offset,
            size,
            identity: header.identity(),
        })
    }

    /// Everything an append records once its bytes are placed: the relocation
    /// watch, dead-byte accounting, the singleton and snapshot locations, the
    /// manifest, the header-index chain and segment rotation.
    fn account_appended(
        &mut self,
        header: &RecordHeader,
        payload: &[u8],
        segment: SegmentId,
        offset: u64,
        size: u64,
    ) -> Result<()> {
        let header = *header;
        if let Some(watch) = self.relocation_watch.as_mut() {
            watch.record(&header, payload);
        }
        self.held_types |= 1u32
            .checked_shl(header.record_type.tag() as u32)
            .unwrap_or(0);
        self.track_identity(&header, segment, offset);
        self.accounting.record(&header, payload, size);
        self.track_metadata_loc(&header, segment, offset, size);
        self.track_resident_loc(&header, payload, segment, offset, size);
        self.marker_ids.record(
            header.record_type,
            header.stream_id,
            payload,
            header.identity(),
        );
        self.track_snapshot_loc(&header, segment, offset, size);
        self.retire_snapshot_on_distill(&header, payload);
        self.track_npc_loc(&header, segment, offset, size);
        // Header + location only. This used to build a `WalkEntry`, whose
        // `Record` owns its payload — so every appended record cloned its whole
        // payload for a call that reads nothing but the header, then dropped it.
        self.manifest
            .ingest_located(&header, segment, offset, size)?;
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
        // Last, once the record is accounted and indexed: a write that fails
        // leaves it staged and known, for the next commit to carry.
        self.write_through_if_over_stage()
    }

    /// Append an **already-encoded** record verbatim to the active segment —
    /// the same accounting + header-index bookkeeping as [`Self::append_record`]
    /// but with **no** re-encode: the caller supplies the raw 4 KB-aligned
    /// record bytes, read verbatim from another segment. Used by background
    /// maintenance to relocate `Chunk`/`Tokens` records with no
    /// decode/CRC-verify/re-encode round trip. Returns `(segment, offset,
    /// size)`. Not for singletons — those go through [`Self::append_carried`]
    /// so `manifest.ingest` repoints them. A read-only handle refuses with
    /// [`PersistenceError::ReadOnly`].
    ///
    /// The bookkeeping reads the header out of `raw` — one JSON line — so the
    /// copy is indexed with the identity and digest it was written with, which
    /// no location map holds.
    pub fn append_raw_record(&mut self, raw: &[u8]) -> Result<Appended> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        let (header, _) = decode_header(raw)?;
        let header = &header;
        let (segment, offset) = self.segments.stage(raw);
        let size = raw.len() as u64;
        // The copy keeps the record's identity, so it is now where the record
        // is.
        self.track_identity(header, segment, offset);
        // Maintenance relocates verbatim only records the accounting keys by
        // header alone — never a section tombstone, which it re-emits.
        self.accounting.record(header, &[], size);
        // A relocated `Snapshot` becomes the stream's current on-disk tail —
        // the persistence-side map must follow it, or the next maintenance
        // pass would mis-read the relocated (live) copy as superseded and skip
        // carrying it out of a segment about to be dropped.
        self.track_snapshot_loc(header, segment, offset, size);
        // A relocated `Npc` record is that character's new on-disk home, for
        // exactly the same reason.
        self.track_npc_loc(header, segment, offset, size);
        // A relocated QSA index page likewise: its copy is where the record is,
        // and the segment liveness counts it from `metadata_locs`.
        if header.record_type == RecordType::TurnIndexPage {
            self.metadata_locs.offer(
                metadata_key(header),
                Some(Self::placed_loc(header, segment, offset, size)),
                header.identity(),
            );
        }
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
        // Last, as in `append_record`.
        self.write_through_if_over_stage()?;
        Ok(Appended {
            segment,
            offset,
            size,
            identity: header.identity(),
        })
    }

    /// Flush the accumulated digests as `HeaderIndex` record(s) chained
    /// to the previous index, commit them durably, and publish the new
    /// chain head in the superblock. The commit-before-publish order
    /// means the hint never points at un-flushed bytes; recovery's
    /// fallback covers the crash window between the two writes either
    /// way. A read-only handle flushes nothing: its un-indexed tail is the one
    /// it found on disk, and the next writable open indexes it.
    fn flush_header_index(&mut self) -> Result<()> {
        if self.pending_index.is_empty() || self.is_read_only() {
            return Ok(());
        }
        while !self.pending_index.is_empty() {
            let take = self.pending_index.len().min(INDEX_FLUSH_ENTRIES);
            let batch: Vec<IndexEntry> = self.pending_index.drain(..take).collect();
            let payload = encode_index_payload(self.last_index.unwrap_or((0, 0)), &batch);
            let at = self.append_record(RecordType::HeaderIndex, 0, 0, 0, 0, 0, &payload)?;
            self.last_index = Some((at.offset, at.size));
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
    /// stream's derived id and the identity the record was written with. A
    /// section's declaration starts its generation, which every record this
    /// node writes for the section from here on carries.
    pub fn declare_stream(&mut self, decl: &StreamDecl) -> Result<Stamped<StreamId>> {
        let stream = decl.stream_id();
        let at =
            self.append_record(RecordType::StreamDecl, 0, stream.0, 0, 0, 0, &decl.encode())?;
        if matches!(decl, StreamDecl::PromptSection(_)) {
            self.minter.start_section(stream.0, at.identity.oseq);
        }
        Ok(Stamped {
            value: stream,
            id: at.identity,
        })
    }

    /// The location of a record just placed at `(segment, offset)`.
    fn placed_loc(h: &RecordHeader, segment: SegmentId, offset: u64, size: u64) -> RecordLoc {
        RecordLoc {
            segment,
            offset,
            payload_len: h.payload_len,
            record_size: size,
        }
    }

    /// The runtime twin of [`observe_replica`]: a record just placed at
    /// `(segment, offset)` is where its identity is now; another origin's
    /// record is applied; and one of this node's own, written for the first
    /// time and not cut off, is tapped (§6.3).
    fn track_identity(&mut self, h: &RecordHeader, segment: SegmentId, offset: u64) {
        let id = h.identity();
        if id.is_none() {
            return;
        }
        self.locations.insert(h, Located { segment, offset });
        if id.origin != self.node() {
            if self.replica.applied.insert(id.origin, id.oseq) {
                self.replica_dirty = true;
            }
        } else if id.gen != GEN_DETACHED && self.own_written.insert(id.oseq) {
            if let Some(tap) = &self.tap {
                tap(id.oseq);
            }
        }
    }

    /// Tell `tap` the `oseq` of every record this node writes from now on that
    /// may be sent, as it is appended — and of what it wrote cut off, when a
    /// rejoin fixes it (§6.3, §6.7). It is called on the log's owner thread,
    /// so it must not block.
    pub fn set_tap(&mut self, tap: Tap) {
        self.tap = Some(tap);
    }

    /// Every own `oseq` written and sendable: what this node can say it holds
    /// of itself.
    pub fn own_written(&self) -> &OriginApplied {
        &self.own_written
    }

    /// Every own `oseq` up to this one is written, or known gone: what this
    /// member may tell another it holds of itself (§7.8, R10).
    ///
    /// Not [`OriginApplied::top`]. A record is stamped when its value changes
    /// and appended when the log's owner reaches its job, so this node's
    /// `oseq`s reach the log out of order, and the top can stand above a record
    /// still queued — a peer told it held the top would take that record for
    /// gone. A gap a purge left at open is gone for good and holds nothing back;
    /// a gap above what the log held at open is a record still on its way, and
    /// what may be told stops below it. A record whose append failed is never
    /// coming ([`Self::mark_unwritten`]), so it does not hold the count back.
    pub fn own_sendable_through(&self) -> u64 {
        let written = &self.own_written;
        written
            .gaps()
            .into_iter()
            .filter(|&(start, _)| start > self.open_top)
            .find(|&(start, end)| !self.own_unwritten.contains_range(start, end))
            .map_or(written.top(), |(start, _)| start - 1)
    }

    /// `id`, minted by this node, will never be written: its append failed.
    /// It holds nothing back ([`Self::own_sendable_through`]).
    pub fn mark_unwritten(&mut self, id: RecordIdentity) {
        if id.origin == self.node() && id.oseq > 0 {
            self.own_unwritten.insert(id.oseq);
        }
    }

    /// What this member holds of every origin, its own written records
    /// included — what a catch-up starts from and a restart offers (§7.2,
    /// §7.5).
    pub fn applied_set(&self) -> AppliedSet {
        let mut set = self.replica.applied.clone();
        if self.own_written.top() > 0 {
            set.set_origin(self.node(), self.own_written.clone());
        }
        set
    }

    /// Every `oseq` of `origin` up to this one is applied here or known gone —
    /// where a catch-up of `origin` starts (§7.2). This node's own records are
    /// all its own `oseq`s so far.
    pub fn applied_through(&self, origin: u64) -> u64 {
        if origin == self.node() {
            self.minter.origination().next_oseq() - 1
        } else {
            self.replica.applied.through(origin)
        }
    }

    /// Whether `(origin, oseq)` is applied here or known gone. Of this node's
    /// own, every `oseq` minted so far but a gap a purge left (§7.8): what it
    /// wrote past its written set's top is held, cut off or not.
    pub fn is_applied(&self, origin: u64, oseq: u64) -> bool {
        if origin == self.node() {
            oseq != 0
                && oseq < self.minter.origination().next_oseq()
                && (oseq > self.own_written.top() || self.own_written.contains(oseq))
        } else {
            self.replica.applied.contains(origin, oseq)
        }
    }

    /// Record `start..=end` of `origin` as gone — `oseq`s a catch-up learned
    /// no member holds any more (§7.2), this node's own among them when a
    /// purge dropped them (§7.8). The next commit persists it.
    pub fn mark_gone(&mut self, origin: u64, start: u64, end: u64) {
        let added = if origin == self.node() {
            self.own_written
                .insert_range(start, end.min(self.own_written.top()))
        } else {
            self.replica.applied.insert_range(origin, start, end)
        };
        if added {
            self.replica_dirty = true;
        }
    }

    /// Defer a record this build does not decode: it is neither written nor
    /// applied, and a catch-up does not ask for it again until a build that
    /// knows its type opens the substrate (§6.5). The next commit persists it.
    pub fn defer(&mut self, deferred: DeferredRecord) {
        if self.replica.deferred.insert(deferred) {
            self.replica_dirty = true;
        }
    }

    /// Whether `(origin, oseq)` is deferred.
    pub fn is_deferred(&self, origin: u64, oseq: u64) -> bool {
        self.replica
            .deferred
            .iter()
            .any(|d| d.origin == origin && d.oseq == oseq)
    }

    /// What this member has applied of other origins, and what it deferred.
    pub fn replica(&self) -> &ReplicaPayload {
        &self.replica
    }

    /// Where `(origin, oseq)` is, if this member holds it.
    pub fn location_of(&self, origin: u64, oseq: u64) -> Option<Located> {
        self.locations.get(origin, oseq)
    }

    /// The bytes of record `(origin, oseq)` — header line, payload and
    /// padding — from wherever maintenance has moved it, or `None` when this
    /// member does not hold it. What a member sends for an event it owes, or
    /// to answer a fetch (§6.2, §6.3, §7.2). A record this node wrote cut off
    /// comes back with the generation its rejoin fixed (§6.7).
    pub fn read_event(&mut self, origin: u64, oseq: u64) -> Result<Option<Vec<u8>>> {
        let Some(at) = self.locations.get(origin, oseq) else {
            return Ok(None);
        };
        self.flush_staged()?;
        let mut probe = vec![0u8; HEADER_PROBE_SIZE];
        self.segments.read_into(at.segment, at.offset, &mut probe)?;
        let (header, header_bytes) = decode_header(&probe)?;
        let id = header.identity();
        if (id.origin, id.oseq) != (origin, oseq) {
            return Err(PersistenceError::Corrupt(format!(
                "the location index names {:x}/{oseq} at {:?}+{} but the record there is \
                 {:x}/{}",
                origin, at.segment, at.offset, id.origin, id.oseq
            )));
        }
        let mut bytes = vec![0u8; padded_record_len(header_bytes - 1, header.payload_len)];
        self.segments.read_into(at.segment, at.offset, &mut bytes)?;
        let fixed = self.rebases.fix(id);
        if fixed == id {
            return Ok(Some(bytes));
        }
        let (header, payload, _) = decode_record(&bytes)?;
        Ok(Some(encode_record(&header.with_identity(fixed), payload)))
    }

    /// Write what is staged to its segment, so a read by location finds it —
    /// a record appended a moment ago is in the staging buffer, not the file.
    /// Not a commit: nothing is made durable.
    fn flush_staged(&mut self) -> Result<()> {
        if self.segments.pending_len() > 0 {
            self.segments.flush()?;
        }
        Ok(())
    }

    /// [`Self::read_event`], decoded and checked against its CRC.
    pub fn read_by_identity(&mut self, origin: u64, oseq: u64) -> Result<Option<Record>> {
        let Some(bytes) = self.read_event(origin, oseq)? else {
            return Ok(None);
        };
        let (header, payload, _) = decode_record(&bytes)?;
        verify_record_crc(&header, payload)?;
        Ok(Some(Record {
            header,
            payload: payload.to_vec(),
        }))
    }

    /// This member is cut off from every cluster: what it writes from here on
    /// is detached, under the provisional generation, until [`Self::rejoin`]
    /// (§6.7).
    pub fn detach(&self) {
        self.minter.detach(None);
    }

    /// Whether this member is cut off — detached since [`Self::detach`], or
    /// found so at open: it holds records it wrote detached that no rejoin
    /// fixed.
    pub fn is_detached(&self) -> bool {
        self.minter.origination().is_detached()
    }

    /// The generations fixed for what this node wrote cut off.
    pub fn rebases(&self) -> &RebaseMap {
        &self.rebases
    }

    /// This member rejoined (§6.7). When it wrote anything while cut off, a
    /// generation is decided for it through `generations`, fixed for every
    /// record it wrote detached by a `Rebase` record made durable here, and
    /// minted from here on; the entry is returned. When it wrote nothing it
    /// has nothing to place, takes no generation, and returns `None`.
    ///
    /// A substrate built from this store holds those records' values under the
    /// provisional generation still: the caller reloads it, as every rejoin
    /// does (§6.6). The fixed records are tapped, so the caller lets the
    /// substrate writer drain first: a record minted cut off and still queued
    /// would be tapped before it could be read.
    pub fn rejoin(&mut self, generations: &dyn MintGeneration) -> Result<Option<RebaseEntry>> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        let origination = self.minter.origination();
        let Some((from, through)) = origination.detached_range() else {
            self.minter.attach(origination.gen());
            return Ok(None);
        };
        let entry = RebaseEntry {
            origin: self.node(),
            from,
            through,
            gen: generations.mint_generation(origination.gen()),
        };
        self.rebases.add(entry);
        let payload = self.rebases.encode();
        self.append_carried(
            RecordType::Rebase,
            0,
            0,
            0,
            0,
            0,
            &payload,
            RecordIdentity::NONE,
        )?;
        self.commit()?;
        self.minter.attach(entry.gen);
        // Fixed, they may be sent now (§6.7 step 3). The caller has let the
        // writer drain first, so each of them is in the log to be read.
        self.own_written.insert_range(from, through);
        if let Some(tap) = &self.tap {
            for oseq in from..=through {
                tap(oseq);
            }
        }
        Ok(Some(entry))
    }

    /// The replica as its record holds it: every other origin's applied set,
    /// and this node's own written set, whose gaps are what a purge dropped
    /// and no catch-up has returned (§7.8). The walk takes the own entry from
    /// the latest record alone ([`observe_replica`]), so once the store holds
    /// any replica record every record carries it: one without would leave an
    /// older record's gaps standing whenever maintenance reclaimed the newer
    /// first. A node that has never applied another origin, holds no replica
    /// record and has no gaps writes none.
    fn replica_record(&self) -> ReplicaPayload {
        let mut record = self.replica.clone();
        let holds_one = self.held_types & (1u32 << RecordType::Replica.tag()) != 0;
        let needed = holds_one || !self.replica.is_empty() || !self.own_written.gaps().is_empty();
        if self.own_written.top() > 0 && needed {
            record
                .applied
                .set_origin(self.node(), self.own_written.clone());
        }
        record
    }

    /// Write the replica when it changed since it was last written — at a
    /// commit, before the sync that makes it durable. Local housekeeping: it
    /// carries no identity and is never replicated.
    fn write_replica_if_changed(&mut self) -> Result<()> {
        if !self.replica_dirty {
            return Ok(());
        }
        let payload = self.replica_record().encode();
        self.append_carried(
            RecordType::Replica,
            0,
            0,
            0,
            0,
            0,
            &payload,
            RecordIdentity::NONE,
        )?;
        self.replica_dirty = false;
        Ok(())
    }

    /// The runtime twin of [`record_resident_loc`]: offer where a just-appended
    /// payload-keyed resident record landed — the greatest stamp per key holds.
    fn track_resident_loc(
        &mut self,
        h: &RecordHeader,
        payload: &[u8],
        segment: SegmentId,
        offset: u64,
        size: u64,
    ) {
        if !keyed_by_payload(h.record_type) {
            return;
        }
        let id = h.identity();
        if let Some(key) = resident_key(h.record_type, h.stream_id, payload, &id) {
            self.resident_locs
                .offer(key, Some(Self::placed_loc(h, segment, offset, size)), id);
        }
    }

    /// Offer the on-disk location of a per-stream metadata record
    /// (`StreamDecl` / `ProjectionEvents` / `WideQSig` / `Commit`) so
    /// [`segment_liveness`](Self::segment_liveness) counts the current copy as
    /// live weight. The greatest stamp per [`MetadataKey`] holds: a
    /// re-write supersedes the prior location, which then reads as dead, and a
    /// maintenance re-emission keeps the identity and moves it. A no-op for
    /// every other record type. Called from
    /// [`append_record`](Self::append_record) (runtime writes) and the
    /// load / compact walk so the map covers both.
    fn track_metadata_loc(&mut self, h: &RecordHeader, segment: SegmentId, offset: u64, size: u64) {
        if is_tracked_metadata(h.record_type) {
            self.metadata_locs.offer(
                metadata_key(h),
                Some(Self::placed_loc(h, segment, offset, size)),
                h.identity(),
            );
        }
    }

    /// Offer the on-disk location of a `Snapshot` record to
    /// [`snapshot_locs`](Self::snapshot_locs) — a retraction withdraws it, and
    /// the greatest stamp per snapshot stream holds, so the map always names
    /// the stream's current tail. A no-op for every other record type. Called
    /// from both append paths ([`append_record`](Self::append_record) and
    /// [`append_raw_record`](Self::append_raw_record)) so runtime snapshot
    /// writes AND maintenance relocations keep the map current; the load /
    /// compact walk uses the free-function mirror [`record_snapshot_loc`].
    fn track_snapshot_loc(&mut self, h: &RecordHeader, segment: SegmentId, offset: u64, size: u64) {
        if h.retracts_snapshot() {
            self.snapshot_locs.offer(h.stream_id, None, h.identity());
        } else if matches!(
            h.record_type,
            RecordType::Snapshot | RecordType::BranchCheckpoint
        ) {
            self.snapshot_locs.offer(
                h.stream_id,
                Some(Self::placed_loc(h, segment, offset, size)),
                h.identity(),
            );
        }
    }

    /// Retire a timeline's recurrent-state tail when its `Distilled` marker is
    /// appended — the runtime twin of the `Distilled` arm in
    /// [`record_snapshot_loc`], and the exact counterpart of the removal
    /// [`Self::write_tombstone`] performs.
    ///
    /// Distillation sheds a timeline's content and keeps its provenance
    /// signatures. The recurrent state is content: it is derived from the token
    /// stream and is meaningless without the K/V it was computed against, so a
    /// distilled conversation's snapshot describes a resume that can never
    /// happen. Retiring the tail here means it stops being live at the *source*,
    /// which is what lets both rewrite paths do the right thing without either
    /// having to consult the distilled set.
    ///
    /// Keyed off the payload, not the header: a `Distilled` record carries its
    /// `timeline_id` in the payload and leaves `stream_id` zero, so unlike
    /// `Snapshot` / `Npc` the identity cannot be read from the header alone.
    fn retire_snapshot_on_distill(&mut self, h: &RecordHeader, payload: &[u8]) {
        if h.record_type != RecordType::Distilled {
            return;
        }
        if let Ok(p) = record::DistillPayload::decode(payload) {
            self.snapshot_locs
                .remove(&snapshot_stream_id(p.timeline_id).0);
        }
    }

    /// The runtime-append twin of [`record_npc_loc`].
    fn track_npc_loc(&mut self, h: &RecordHeader, segment: SegmentId, offset: u64, size: u64) {
        if h.record_type == RecordType::Npc {
            self.npc_locs.offer(
                h.stream_id,
                Some(Self::placed_loc(h, segment, offset, size)),
                h.identity(),
            );
        }
    }

    /// Append one character's durable state.
    ///
    /// The `npc_id` goes in the header's `stream_id`, which is what makes this
    /// supersede the character's previous record mechanically: the accounting
    /// credits the old copy as dead the moment this one lands, and
    /// [`Self::npc_locs`] now points here. There is no delete record — an edit
    /// supersedes, and a deletion is an edit that sets `state: "tombstoned"`.
    ///
    /// Writes are expected to be rare: the payload is durable configuration
    /// only, so a character that is merely *running* never writes at all.
    pub fn write_npc(&mut self, npc: &NpcPayload) -> Result<RecordIdentity> {
        let bytes = npc.encode();
        let at = self.append_record(RecordType::Npc, 0, npc.npc_id, 0, 0, 0, &bytes)?;
        Ok(at.identity)
    }

    /// Append custom-object record `id`. The header's `stream_id` is the key's
    /// stable hash, so of two writes with the same key the greater stamp is the
    /// object on replay. The caller applies it to the in-RAM [`Substrate`]
    /// under the same identity, so it re-emits on compaction — see
    /// [`SharedSubstrate::put_custom_object`].
    pub fn write_custom_object(
        &mut self,
        obj: &CustomObjectPayload,
        id: RecordIdentity,
    ) -> Result<()> {
        let bytes = obj.encode();
        self.append_carried(
            RecordType::CustomObject,
            0,
            obj.stream_id(),
            0,
            0,
            0,
            &bytes,
            id,
        )?;
        Ok(())
    }

    /// Every character's current record location, keyed by `npc_id`.
    ///
    /// The `substrate/` directory this store's segments live in.
    pub fn dir(&self) -> &Path {
        self.segments.dir()
    }

    /// Locations rather than payloads, because the payloads are not held here —
    /// the registry that owns them lives in the daemon. Compaction and segment
    /// liveness both read this to know which bytes are still worth keeping.
    pub fn npc_locs(&self) -> &StampedLocs<u64> {
        &self.npc_locs
    }

    /// Read back one character's current record.
    ///
    /// The load path does not use this — it collects payloads from the recovery
    /// walk's sink in one sweep, which is a great deal cheaper than a seek per
    /// character. This exists for the one-off: re-reading a single character
    /// after a write, or a diagnostic.
    pub fn read_npc(&mut self, npc_id: u64) -> Result<Option<NpcPayload>> {
        let Some(loc) = self.npc_locs.get(&npc_id).copied() else {
            return Ok(None);
        };
        let record = self
            .segments
            .read_record_at(loc.segment, loc.offset, loc.record_size)?;
        Ok(Some(NpcPayload::decode(&record.payload)?))
    }

    /// Append a stream's `Tokens` record. The caller MUST fold its location and
    /// identity into the substrate index (`apply_tokens_loc`) — otherwise the
    /// in-RAM index has no record of the tokens, and a subsequent segment
    /// compaction / relocation can't carry them forward and reclaims them (the
    /// KV path already registers its chunk locations; tokens must too, or a
    /// live turn silently loses its text).
    pub fn append_tokens(&mut self, stream_id: StreamId, tokens: &[u8]) -> Result<Appended> {
        self.append_record(RecordType::Tokens, 0, stream_id.0, 0, 0, 0, tokens)
    }

    /// Append a turn's `ProjectionEvents` record `id` (opaque JSON payload).
    pub fn append_projection_events(
        &mut self,
        stream_id: StreamId,
        payload: &[u8],
        id: RecordIdentity,
    ) -> Result<()> {
        self.append_carried(
            RecordType::ProjectionEvents,
            0,
            stream_id.0,
            0,
            0,
            0,
            payload,
            id,
        )?;
        Ok(())
    }

    /// Append a turn's `WideQSig` record `id` (opaque wide-Q window payload),
    /// keyed by stream id.
    pub fn append_wide_q_sigs(
        &mut self,
        stream_id: StreamId,
        payload: &[u8],
        id: RecordIdentity,
    ) -> Result<()> {
        self.append_carried(RecordType::WideQSig, 0, stream_id.0, 0, 0, 0, payload, id)?;
        Ok(())
    }

    /// Append a turn's QSA index page, record `id` (opaque payload), keyed by
    /// stream id.
    ///
    /// The counterpart of [`Self::append_wide_q_sigs`]: both carry a blob that
    /// belongs to a turn and that no amount of re-reading the turn's K/V can
    /// reconstruct, because it was computed from hidden states.
    ///
    /// Returns the record's location, which the writer settles the substrate's
    /// held copy to ([`crate::substrate::Substrate::settle_index_page`]): a
    /// page is held only until it is durable, then read from the log.
    pub fn append_turn_index_page(
        &mut self,
        stream_id: StreamId,
        payload: &[u8],
        id: RecordIdentity,
    ) -> Result<RecordLoc> {
        let at = self.append_carried(
            RecordType::TurnIndexPage,
            0,
            stream_id.0,
            0,
            0,
            0,
            payload,
            id,
        )?;
        Ok(at.loc(payload.len() as u64))
    }

    /// Append a conversation's recurrent-state snapshot under its snapshot
    /// stream id. Single tail: the accounting layer credits the previous
    /// snapshot as dead the moment this one lands, and `snapshot_locs` keeps
    /// exactly this copy alive through segment maintenance. Returns the
    /// record's location for the caller's in-RAM index.
    ///
    /// The payload comes as its parts ([`record::SnapshotPayload::parts`]) and is
    /// written from them, so the state reaches the file without being copied
    /// into one buffer first.
    pub fn write_snapshot(
        &mut self,
        stream_id: StreamId,
        payload: &[&[u8]],
    ) -> Result<Stamped<RecordLoc>> {
        self.write_recurrent_record(RecordType::Snapshot, stream_id, payload)
    }

    /// Append a snapshot retraction for `stream_id`: an empty `Snapshot`
    /// record, after which the stream has no live snapshot on this run or any
    /// reload ([`RecordHeader::retracts_snapshot`]). The writer appends one
    /// when a seal's snapshot fails to land, because the previous snapshot
    /// would otherwise stay the tail while the turn's records land behind it.
    /// Returns the retraction's identity, which outranks the tail it withdraws.
    pub fn retract_snapshot(&mut self, stream_id: StreamId) -> Result<RecordIdentity> {
        let at = self.append_record(RecordType::Snapshot, 0, stream_id.0, 0, 0, 0, &[])?;
        Ok(at.identity)
    }

    /// Append a prompt branch's recurrent checkpoint — the same single-tail
    /// append under a `BranchCheckpoint` record, from its parts
    /// ([`record::BranchCheckpointPayload::parts`]).
    pub fn write_branch_checkpoint(
        &mut self,
        stream_id: StreamId,
        payload: &[&[u8]],
    ) -> Result<Stamped<RecordLoc>> {
        self.write_recurrent_record(RecordType::BranchCheckpoint, stream_id, payload)
    }

    fn write_recurrent_record(
        &mut self,
        record_type: RecordType,
        stream_id: StreamId,
        payload: &[&[u8]],
    ) -> Result<Stamped<RecordLoc>> {
        let at = self.append_recurrent_parts(record_type, stream_id.0, payload)?;
        Ok(Stamped {
            value: at.loc(payload.iter().map(|p| p.len() as u64).sum()),
            id: at.identity,
        })
    }

    /// Append a `Commit` record marking `stream_id` durable through
    /// `through_index`. Returns the identity it was written with.
    pub fn commit_stream(
        &mut self,
        stream_id: StreamId,
        through_index: u64,
    ) -> Result<RecordIdentity> {
        let at =
            self.append_record(RecordType::Commit, 0, stream_id.0, through_index, 0, 0, &[])?;
        Ok(at.identity)
    }

    /// Append `TreeMetadata` record `id` for one `(timeline_id, turn_index)`
    /// summary-tree node — the greatest stamp per node on replay. Callers
    /// check idempotency against substrate state.
    pub fn write_tree_metadata(
        &mut self,
        payload: TreeMetadataPayload,
        id: RecordIdentity,
    ) -> Result<()> {
        let bytes = payload.encode();
        self.append_carried(RecordType::TreeMetadata, 0, 0, 0, 0, 0, &bytes, id)?;
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
        let id = self.minter.mint(0);
        self.append_timeline_tombstone(timeline_id, reason, id)
    }

    /// [`Self::write_tombstone`] for a **repair**: this member's judgement
    /// about its own bytes — a turn it finds incomplete or unreadable at
    /// reload — not a deletion. Housekeeping, written with no identity, so it
    /// is never sent to another member, which may hold the conversation whole
    /// (§6.5).
    pub fn write_repair_tombstone(&mut self, timeline_id: u64, reason: &str) -> Result<()> {
        self.append_timeline_tombstone(timeline_id, Some(reason), RecordIdentity::NONE)
    }

    fn append_timeline_tombstone(
        &mut self,
        timeline_id: u64,
        reason: Option<&str>,
        id: RecordIdentity,
    ) -> Result<()> {
        let payload = TombstonePayload {
            timeline_id,
            turn_index: None,
            turn_writer: 0,
            reason: reason.map(str::to_string),
        };
        let bytes = payload.encode();
        self.append_carried(RecordType::Tombstone, 0, 0, 0, 0, 0, &bytes, id)?;
        // A dead conversation has no live snapshot tail: drop its entry from
        // the persistence-side map so a maintenance execute planned before
        // this tombstone skips relocating the (now dead) snapshot record —
        // the same removal the substrate index performs in `apply_tombstone`,
        // and the same one `record_snapshot_loc` replays on reload.
        self.snapshot_locs
            .remove(&snapshot_stream_id(timeline_id).0);
        // Its file events are orphans now: out of the index, so no rewrite
        // carries them, and their bytes dead.
        vfs::retire_timeline(&mut self.vfs_index, &mut self.accounting, timeline_id);
        Ok(())
    }

    /// Append a **turn-scoped** tombstone — kill only the turn `mark` names,
    /// leaving the rest of the timeline live. Used by the per-layer `drop_turn`
    /// corrupt-turn policy so one unrecoverable turn (e.g. a partial write) does
    /// not drop its whole conversation the way a timeline tombstone would, and
    /// by retention. The mark names the turn as its writer wrote it
    /// ([`Substrate::tombstone_turn`] returns it).
    pub fn write_turn_tombstone(&mut self, mark: TurnMark, reason: Option<&str>) -> Result<()> {
        let id = self.minter.mint(0);
        self.append_turn_tombstone(mark, reason, id)
    }

    /// [`Self::write_turn_tombstone`] for a **repair** — see
    /// [`Self::write_repair_tombstone`].
    pub fn write_repair_turn_tombstone(&mut self, mark: TurnMark, reason: &str) -> Result<()> {
        self.append_turn_tombstone(mark, Some(reason), RecordIdentity::NONE)
    }

    fn append_turn_tombstone(
        &mut self,
        mark: TurnMark,
        reason: Option<&str>,
        id: RecordIdentity,
    ) -> Result<()> {
        let payload = TombstonePayload {
            timeline_id: mark.timeline.raw(),
            turn_index: Some(mark.turn_index),
            turn_writer: mark.writer,
            reason: reason.map(str::to_string),
        };
        let bytes = payload.encode();
        self.append_carried(RecordType::Tombstone, 0, 0, 0, 0, 0, &bytes, id)?;
        Ok(())
    }

    /// Append a [`RecordType::SectionTombstone`] record marking the
    /// content-addressed section stream `stream_id` as logically deleted.
    /// Walker replay applies it via [`Substrate::apply_section_tombstone`],
    /// which reads as "not persisted" on the next ingest triage, so the
    /// section re-prefills fresh instead of restoring the tombstoned chunks.
    /// Idempotent — duplicate tombstones for the same stream replay into the
    /// same set. `generation` is the one it ends
    /// ([`Substrate::tombstone_section`] returns it); `None` ends the
    /// unstamped stream. `reason` is a diagnostic note (the confirmed
    /// divergence detail); pass `None` when there is nothing more to say than
    /// "gone". A generation this node was writing stops being stamped on its
    /// records.
    pub fn write_section_tombstone(
        &mut self,
        stream_id: u64,
        generation: Option<SectionGeneration>,
        reason: Option<&str>,
    ) -> Result<()> {
        let payload = SectionTombstonePayload {
            generation,
            reason: reason.map(str::to_string),
        };
        let bytes = payload.encode();
        self.append_record(RecordType::SectionTombstone, 0, stream_id, 0, 0, 0, &bytes)?;
        if let Some(g) = generation {
            if g.writer == self.node() {
                self.minter.end_section(stream_id, g.oseq);
            }
        }
        Ok(())
    }

    /// Append a [`RecordType::TurnCoupling`] record joining `from_turn` to the
    /// tool response that follows it.
    ///
    /// Written before the response turn is submitted — the one window where the
    /// round-trip is certain — so replay can never observe the response turn
    /// without its coupling. Idempotent: duplicates replay into the same set.
    pub fn write_turn_coupling(
        &mut self,
        timeline_id: u64,
        from_turn: u32,
        id: RecordIdentity,
    ) -> Result<()> {
        let payload = record::TurnCouplingPayload {
            timeline_id,
            from_turn,
        };
        let bytes = payload.encode();
        self.append_carried(RecordType::TurnCoupling, 0, 0, 0, 0, 0, &bytes, id)?;
        Ok(())
    }

    /// Append `DebugId` record `id` for `timeline_id` — the greatest stamp per
    /// timeline on replay. Callers check idempotency against substrate state.
    pub fn write_debug_id(
        &mut self,
        timeline_id: u64,
        debug_id: &str,
        id: RecordIdentity,
    ) -> Result<()> {
        let payload = DebugIdPayload {
            timeline_id,
            debug_id: debug_id.to_string(),
        };
        let bytes = payload.encode();
        self.append_carried(RecordType::DebugId, 0, 0, 0, 0, 0, &bytes, id)?;
        Ok(())
    }

    /// Append a `Chunk` record — one sealed or partial KV chunk's bytes and
    /// quantization metadata. `token_count` is 32 for a sealed chunk, less
    /// for a partial tail; `format` is the `KvFormat` tag. Returns where the
    /// record landed and its identity.
    pub fn write_chunk(
        &mut self,
        stream_id: StreamId,
        chunk_index: u64,
        token_count: u64,
        format: u8,
        golden: Option<u32>,
        payload: &ChunkPayload,
    ) -> Result<Appended> {
        // `Some(g)` is the seal path's GPU-computed golden (Fletcher-32 over the
        // arena bytes before the DtoH copy). `None` means no precomputed golden
        // is available, so take it host-side over these bytes now — the same
        // host-truth the CPU gather path uses.
        let golden = golden.unwrap_or_else(|| candle::fletcher::fletcher32(&payload.kv_bytes));
        self.append_record(
            RecordType::Chunk,
            format,
            stream_id.0,
            chunk_index,
            token_count,
            golden,
            &payload.encode(),
        )
    }

    /// Read one chunk's payload — from the active log, else any inherited
    /// log (§13.5). The chunk must be written to the file — committed, or
    /// written through by a long append; it is read from the file, not the
    /// staging buffer.
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
        let inherited_chunks: Vec<Option<&POrdMap<u64, ChunkLoc>>> = self
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

    /// A reader for this store's sealed segments that needs no lock on it —
    /// taken once, beside the handle, so a read that finds its record sealed
    /// never waits on this persistence's mutex. See [`sealed_reader`].
    pub fn sealed_reader(&self) -> SealedReader {
        self.segments.sealed_reader()
    }

    /// Where `(origin, oseq)` lies, from memory — for a read made without
    /// this persistence's lock (`SealedReader::read_raw`). `None` when it is
    /// not held here.
    pub fn locate(&self, origin: u64, oseq: u64) -> Option<(SegmentId, u64)> {
        self.locations
            .get(origin, oseq)
            .map(|at| (at.segment, at.offset))
    }

    /// Whether a record read off disk is already as it travels: no rebase
    /// can apply to a detached identity (`read_event`'s fix-up).
    pub fn no_rebases(&self) -> bool {
        self.rebases.is_empty()
    }

    /// Read one record's payload back by location — the snapshot-restore
    /// read (the caller got `loc` from the substrate's snapshot index).
    pub fn read_record_payload(&mut self, loc: &RecordLoc) -> Result<Vec<u8>> {
        let record = self
            .segments
            .read_record_at(loc.segment, loc.offset, loc.record_size)?;
        Ok(record.payload)
    }

    /// Flush and `fsync` the active segment — the group-commit durability
    /// point. Seals + rotates the active if it has reached the size target.
    /// A no-op on a read-only handle, which never has anything staged.
    pub fn commit(&mut self) -> Result<()> {
        if self.is_read_only() {
            return Ok(());
        }
        self.write_replica_if_changed()?;
        let committed = self.segments.commit();
        self.watch_space(committed)?;
        self.maybe_rotate_active()
    }

    /// Bytes staged but not yet flushed to the active segment. Returns 0 when
    /// there is nothing staged — which, once a large append has flushed part
    /// way, is not the same as nothing to commit ([`Self::commit_if_pending`]).
    pub fn pending_bytes(&self) -> usize {
        self.segments.pending_len()
    }

    /// Write the staged records through to the active segment once they reach
    /// [`STAGE_FLUSH_BYTES`], leaving the sync to the next commit.
    fn write_through_if_over_stage(&mut self) -> Result<()> {
        if self.segments.pending_len() >= STAGE_FLUSH_BYTES {
            let flushed = self.segments.flush();
            self.watch_space(flushed)?;
        }
        Ok(())
    }

    /// Group-commit if (and only if) there is something to make durable —
    /// records staged, or written through by a large append and not yet
    /// synced. Returns `Ok(true)` when a flush+fsync actually happened,
    /// `Ok(false)` for the no-op idle path. Cheap to call on a tight timer.
    pub fn commit_if_pending(&mut self) -> Result<bool> {
        if self.is_read_only() || (!self.segments.needs_commit() && !self.replica_dirty) {
            return Ok(false);
        }
        self.write_replica_if_changed()?;
        let committed = self.segments.commit();
        self.watch_space(committed)?;
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
    ///
    /// A no-op on a read-only handle: the segment set stays exactly as the
    /// process that owns it left it.
    pub fn seal_active(&mut self) -> Result<()> {
        if self.is_read_only() {
            return Ok(());
        }
        self.flush_header_index()?;
        // A write-through may have left bytes on the segment that no commit
        // has synced yet; a sealed segment's handle is dropped, so sync now.
        if self.segments.needs_commit() {
            self.segments.commit()?;
        }
        self.segments.seal_and_rotate()?;
        // The fresh active starts its own `HeaderIndex` chain — the sealed
        // segment keeps the chain the flush above completed.
        self.last_index = None;
        self.pending_index.clear();
        Ok(())
    }

    /// Set the projection template — last-writer-wins, and never written
    /// through a read-only handle.
    pub fn set_template(&mut self, template: &[u8]) -> Result<bool> {
        if self.template.as_deref() == Some(template) || self.is_read_only() {
            return Ok(false);
        }
        self.append_record(RecordType::Template, 0, 0, 0, 0, 0, template)?;
        self.template = Some(template.to_vec());
        Ok(true)
    }

    /// Bind the substrate to a model's `tokenizer.json` bytes.
    ///
    /// Writes once, on a substrate that has no tokenizer yet, and thereafter
    /// **refuses** anything that does not hash identically: the tokenizer is
    /// the vocabulary every turn already in this log was sealed under. Token
    /// ids are only meaningful against it — accept a second one and every
    /// stored turn silently means something else, while the manifest points at
    /// the new record and reports success.
    ///
    /// [`SubstratePersistence::bind_model`] refuses a different vocabulary
    /// first, as part of the model's key; this is the embedded copy's own
    /// guard, so the bytes the log carries can never disagree with the key.
    ///
    /// The full bytes are the `Tokenizer` record's payload, so the log is
    /// a self-contained substrate image (§5.7) — no companion files.
    /// Recovery never reads the payload (the walk skips it and the digest
    /// chain carries only its header), so the ~11 MB record costs one
    /// on-demand read at open to compute the identity hash and nothing else.
    pub fn set_tokenizer(&mut self, tokenizer: &[u8]) -> Result<bool> {
        let hash = sha256(tokenizer);
        if let Some(existing) = self.tokenizer_sha256 {
            if existing == hash {
                return Ok(false);
            }
            return Err(PersistenceError::Corrupt(format!(
                "this substrate is bound to a different tokenizer (recorded {}, offered {}). \
                 Every turn in the log was sealed under the recorded vocabulary, so adopting \
                 another would silently change what all of them say. Point this model at a \
                 fresh substrate, or wipe this one.",
                hex16(&existing),
                hex16(&hash),
            )));
        }
        // A read-only handle refuses a mismatch like any other (above) but
        // never binds a store that has no tokenizer recorded.
        if self.is_read_only() {
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
        let live: u64 = self.segment_liveness(substrate).values().sum();
        self.note_dead_ratio(total, live)
    }

    /// Record the dead ratio of `total` record bytes of which `live` are live,
    /// and return it — what [`Self::scanned_dead_ratio`] answers until the next
    /// liveness scan, this one's or maintenance's.
    pub(crate) fn note_dead_ratio(&self, total: u64, live: u64) -> f32 {
        let ratio = dead_ratio_of(total, live);
        self.scanned_dead_ratio
            .store(ratio.to_bits(), AtomicOrdering::Relaxed);
        ratio
    }

    /// The dead ratio the last liveness scan found — maintenance's plan or a
    /// [`Self::dead_ratio`] — without a scan of its own; `None` before the first.
    ///
    /// For a viewer: the scan walks the whole substrate under its read lock,
    /// up to 0.86 s on BlackWorld (2026-10-29), and the substrate page asked
    /// for it on every refresh — a hold every writer, the scheduler's among
    /// them, waited out. Maintenance scans every pass anyway.
    pub fn scanned_dead_ratio(&self) -> Option<f32> {
        let ratio = f32::from_bits(self.scanned_dead_ratio.load(AtomicOrdering::Relaxed));
        (!ratio.is_nan()).then_some(ratio)
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
    /// reads (§13.5). A read-only handle refuses with
    /// [`PersistenceError::ReadOnly`].
    ///
    /// `progress` is told the records written of the live set — `(0, 0)` while
    /// that set is still being collected.
    ///
    /// For a substrate no other thread edits; [`Self::compact_shared`] is the
    /// same rewrite against a root other threads edit while it runs.
    pub fn compact(
        &mut self,
        substrate: &mut Substrate,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> Result<()> {
        let rewritten = self.rewrite(substrate, progress, None)?;
        rewritten.apply_to(substrate);
        self.unlink_rewritten(&rewritten)
    }

    /// [`Self::compact`] against a shared root: the rewrite reads one version,
    /// and its re-apply is a run of ordinary edits ([`Rewritten::batches`]), so
    /// every edit another thread publishes meanwhile survives it. The old
    /// segments are unlinked once the last edit is published.
    pub fn compact_shared(
        &mut self,
        substrate: &SubstrateLock,
        progress: Option<&(dyn Fn(usize, usize) + Sync)>,
    ) -> Result<()> {
        let read = substrate.read();
        let progress = progress.map(|p| p as &dyn Fn(usize, usize));
        let rewritten = self.rewrite(&read, progress, None)?;
        drop(read);
        for batch in rewritten.batches() {
            substrate.update(|s| {
                for entry in batch {
                    s.apply_walker_entry(entry);
                }
            });
        }
        substrate.update(|s| rewritten.settle(s));
        self.unlink_rewritten(&rewritten)
    }

    /// Unlink the segments `rewritten` replaced: the substrate names none of
    /// them any more.
    fn unlink_rewritten(&mut self, rewritten: &Rewritten) -> Result<()> {
        self.segments.unlink_replaced(rewritten.old_segments())
    }

    /// Adopt a store written before identities (§6.2, [`adoption`]): one
    /// compaction stamps every record that carries none as this node's own.
    /// Run by the first writable open of such a store, which then opens it
    /// again; a no-op on a store that needs none. `progress` counts the
    /// records rewritten, as [`Self::compact`]'s does.
    fn adopt(&mut self, substrate: &mut Substrate, progress: &dyn Fn(usize, usize)) -> Result<()> {
        if !self.predates_identities() {
            return Ok(());
        }
        tracing::info!(
            target: "candle_conversation::persistence::identity",
            node = format!("{:016x}", self.node()),
            dir = %self.dir().display(),
            unstamped = %self.unstamped.summary(),
            "this store was written before records carried an identity — rewriting it once, \
             with every record stamped as this node's own"
        );
        let mut adoption = Adoption::new(self.minter.clone());
        let rewritten = self.rewrite(substrate, Some(progress), Some(&mut adoption))?;
        // The caller opens the adopted store again into a fresh substrate, so
        // this one is not re-applied.
        self.unlink_rewritten(&rewritten)?;
        self.unstamped = RecordCensus::new();
        Ok(())
    }

    /// The log rewrite behind [`Self::compact`], stamping as it writes when
    /// `adoption` is given. Reads `substrate` and changes nothing in it: what
    /// the substrate has to do about the rewrite comes back as a [`Rewritten`].
    fn rewrite(
        &mut self,
        substrate: &Substrate,
        progress: Option<&dyn Fn(usize, usize)>,
        adoption: Option<&mut Adoption>,
    ) -> Result<Rewritten> {
        if self.is_read_only() {
            return Err(PersistenceError::ReadOnly);
        }
        // Records written of the live set, for the loading screen: `(0, 0)`
        // until the set is collected.
        let report = |done: usize, total: usize| {
            if let Some(p) = progress {
                p(done, total);
            }
        };
        report(0, 0);

        // 1. Quiesce — every staged write is now durable on disk.
        self.segments.commit()?;

        // 2. Collect the live winners, in dependency order — sourced from
        //    substrate state (per-entity) + manifest singletons. Each record
        //    is read from the segment that physically holds it.
        let dir = self.segments.dir().to_path_buf();
        // Planning only — no disk reads; each read-back record carries its
        // source location, read back coalesced + staged verbatim in step 3.
        let expiry = TombstoneExpiry {
            now_ms: now_ms(),
            retention: self.tombstone_retention,
        };
        // What this compaction drops: the substrate forgets it once the
        // rewritten log is walked back in, or maintenance would carry it again.
        let expired = expiry.expired(substrate, &self.marker_ids);
        let live = compaction::collect_live_records(
            &self.manifest,
            substrate,
            &self.npc_locs,
            &self.vfs_index,
            &self.blob_index,
            &self.marker_ids,
            self.node(),
            expiry,
            &Housekeeping {
                replica: self.replica_record(),
                rebases: self.rebases.clone(),
            },
        );
        // What this compaction is carrying forward, by type, against what the
        // store held when it opened. A class present at open and absent here is
        // being deleted — the failure this whole module exists to make visible,
        // and one that raises nothing on its own. One pass over a Vec already in
        // RAM, once per compaction.
        {
            let mut carried = RecordCensus::new();
            for item in &live {
                carried.record(item.header().record_type);
            }
            let lost = carried.types_lost_against(&self.open_census);
            if lost.is_empty() {
                tracing::info!(
                    target: "candle_conversation::persistence::census",
                    total = carried.total(),
                    "compaction carrying forward: {}",
                    carried.summary()
                );
            } else {
                // `warn`, and naming the classes: every type the store held has
                // a carrier in `survival.rs`, so a type going to zero here is
                // either a genuine emptying (a tombstoned timeline's last turn)
                // or a rewrite that forgot one. Both are worth a line that says
                // which.
                tracing::warn!(
                    target: "candle_conversation::persistence::census",
                    total = carried.total(),
                    dropped = ?lost,
                    "compaction is carrying NO records of {} type(s) the store held at open \
                     — verify each is a legitimate emptying and not a missing carrier \
                     (persistence/survival.rs): {}",
                    lost.len(),
                    carried.summary()
                );
            }
        }
        let total = live.len();
        report(0, total);

        // 3. Write the live set into a compacted scratch file (a fresh index
        //    chain interleaved), reading read-back records off the source
        //    segments in coalesced stripes, then adopt it as the sole active
        //    segment, dropping every prior segment.
        let scratch = dir.join(".compact");
        let (new_log, new_manifest) = {
            let segments = &mut self.segments;
            compaction::write_compacted_log(
                &scratch,
                &live,
                &mut |seg, off, dest| segments.read_into(seg, off, dest),
                adoption,
                &mut |done| report(done, total),
            )?
        };
        let old_segments = self.segments.adopt_compacted(new_log, &scratch)?;

        // 4. Adopt the compacted manifest and refresh the metadata caches.
        //    `write_compacted_log` stamped every singleton with `FIRST_SEGMENT`,
        //    but `adopt_compacted` names the compacted segment with a fresh id
        //    (crash safety) and wrote ALL records into that one segment — so
        //    re-stamp the singletons at the actual active id before reading them.
        self.manifest = new_manifest;
        let adopted = self.segments.active_id();
        for loc in [
            &mut self.manifest.model_key,
            &mut self.manifest.template,
            &mut self.manifest.tokenizer,
        ]
        .into_iter()
        .flatten()
        {
            loc.segment = adopted;
        }
        self.model_record = self
            .manifest
            .model_key
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
        // 5. The substrate still holds offsets into the OLD segments. The walk
        //    of the compacted segment below is kept for the caller to re-apply
        //    ([`Rewritten`]), which moves each value to its new offset. The
        //    dead-weight accounting restarts from the fresh segment in the same
        //    pass (a just-compacted store is all-live).
        self.accounting.reset();
        // Metadata and snapshot locations point at the OLD segments; rebuild
        // them from the freshly-compacted segment in the same recovery pass
        // (mirrors the load walk in `from_dir_with_sink`).
        self.metadata_locs.clear();
        self.resident_locs.clear();
        self.snapshot_locs.clear();
        self.npc_locs.clear();
        self.vfs_index.clear();
        self.blob_index.clear();
        // The marker identities are rebuilt the same way: every carried marker
        // kept its own, so the rebuilt table holds what the old one did for
        // every marker still live, and nothing for one the compaction dropped.
        self.marker_ids = MarkerIds::new();
        // Every record moved, so every location is rebuilt; the applied set
        // only grows, and the rewrite carried it whole.
        self.locations.clear();
        let node = self.node();
        let accounting = &mut self.accounting;
        let metadata_locs = &mut self.metadata_locs;
        let resident_locs = &mut self.resident_locs;
        let snapshot_locs = &mut self.snapshot_locs;
        let npc_locs = &mut self.npc_locs;
        let vfs_index = &mut self.vfs_index;
        let blob_index = &mut self.blob_index;
        let marker_ids = &mut self.marker_ids;
        let replica = &mut self.replica;
        let locations = &mut self.locations;
        let rebases = &self.rebases;
        // The carried record's own entry is `own_written` as it stands, which
        // this handle already holds.
        let mut own_kept = None;
        let mut walked: Vec<WalkEntry> = Vec::with_capacity(total);
        let (last_index, tail_digests) = self.segments.recover_active_with_sink(|entry| {
            let fixed = fix_walked(entry, rebases);
            let entry = fixed.as_ref();
            observe_replica(replica, locations, &mut own_kept, node, entry);
            let h = &entry.record.header;
            marker_ids.record(
                h.record_type,
                h.stream_id,
                &entry.record.payload,
                h.identity(),
            );
            accounting.record(&entry.record.header, &entry.record.payload, entry.size);
            record_metadata_loc(metadata_locs, entry);
            record_resident_loc(resident_locs, entry);
            record_snapshot_loc(snapshot_locs, entry);
            record_npc_loc(npc_locs, entry);
            vfs::record_vfs_loc(vfs_index, accounting, entry);
            blob::record_blob_loc(blob_index, entry);
            walked.push(Rewritten::keep(entry));
        })?;
        // The compacted segment carries a fresh index chain; chain the next
        // flush onto it.
        self.last_index = last_index;
        self.pending_index = tail_digests;
        report(total, total);
        Ok(Rewritten::new(walked, old_segments, substrate, expired))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::identity::{ModelKey, RecordedModel};
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

    /// A model key described as `file`, whose weights are `weights` repeated —
    /// so two keys with the same `weights` are one model, whatever `file` says.
    fn model(file: &str, weights: u8) -> ModelKey {
        ModelKey {
            arch: "Qwen3".into(),
            repo: "test/repo".into(),
            file: file.into(),
            weights: [weights; 32],
            tokenizer: [0; 32],
            rope: "file-stated min=none".into(),
        }
    }

    fn recorded(sp: &SubstratePersistence) -> Option<RecordedModel> {
        sp.recorded_model().map(|r| r.unwrap())
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

    /// **The staging buffer never holds more than the write-through bound plus
    /// one record**, and bytes written through still owe a commit: 70 one-MiB
    /// appends with no commit between them leave under 65 MiB staged, the rest
    /// on the segment, and `commit_if_pending` syncs once and then idles.
    #[test]
    fn a_long_append_run_writes_through_and_still_owes_a_commit() {
        let dir = tmp_dir("write_through");
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        sp.commit().unwrap();
        let start = sp.write_offset();
        let payload = vec![7u8; 1024 * 1024];
        let mut record = 0;
        for chunk in 0..70u64 {
            let at = sp
                .append_record(RecordType::Tokens, 0, 9, chunk, 0, 0, &payload)
                .unwrap();
            record = at.size as usize;
            assert!(sp.pending_bytes() < STAGE_FLUSH_BYTES + record);
        }
        assert!(sp.write_offset() > start);
        assert!(sp.pending_bytes() < STAGE_FLUSH_BYTES);
        assert!(sp.commit_if_pending().unwrap());
        assert_eq!(sp.pending_bytes(), 0);
        assert_eq!(sp.write_offset(), start + 70 * record as u64);
        assert!(!sp.commit_if_pending().unwrap());
        drop(sp);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **Records written through and never committed reopen cleanly**: a
    /// handle dropped mid-run — a crash, as far as the log can tell — leaves
    /// the written-through records on the segment and loses only the staged
    /// tail, and the log opens on a record boundary with a whole record count.
    #[test]
    fn records_written_through_but_not_committed_reopen_cleanly() {
        let dir = tmp_dir("write_through_reopen");
        let (start, record) = {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.commit().unwrap();
            let start = sp.write_offset();
            let payload = vec![7u8; 1024 * 1024];
            let mut record = 0;
            for chunk in 0..70u64 {
                let at = sp
                    .append_record(RecordType::Tokens, 0, 9, chunk, 0, 0, &payload)
                    .unwrap();
                record = at.size;
            }
            assert!(
                sp.write_offset() > start,
                "part of the run was written through"
            );
            (start, record)
        };
        let mut substrate = Substrate::new();
        let reopened = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let kept = reopened.write_offset() - start;
        assert_eq!(kept % record, 0, "the log ends on a record boundary");
        assert!(
            kept >= STAGE_FLUSH_BYTES as u64 && kept < 70 * record,
            "kept {kept} bytes"
        );
        drop(reopened);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A read-only handle refuses every write and changes no byte.** Built
    /// over a store with a sealed segment, a section, a model spec and a bound
    /// tokenizer: the appends refuse without staging, the commit paths succeed
    /// with nothing to do, the singletons report nothing written and keep the
    /// recorded values, a different tokenizer is still refused, maintenance
    /// plans nothing even when forced, compaction refuses — and every file is
    /// byte-identical after the handle drops.
    #[test]
    fn a_read_only_handle_refuses_writes_and_changes_nothing() {
        let dir = tmp_dir("read_only");
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&section("sys", 1)).unwrap();
            sp.bind_model(&model("spec-a", 1)).unwrap();
            sp.set_tokenizer(b"tokenizer-a").unwrap();
            sp.commit().unwrap();
            sp.seal_active().unwrap();
        }
        let before = dir_fingerprint(&dir);

        let mut substrate = Substrate::new();
        let mut sp =
            SubstratePersistence::open_in_with_substrate_read_only(&dir, &mut substrate).unwrap();
        assert!(sp.is_read_only());
        let offset = sp.write_offset();
        assert!(matches!(
            sp.append_record(RecordType::Tokens, 0, 9, 0, 0, 0, b"refused"),
            Err(PersistenceError::ReadOnly)
        ));
        assert!(matches!(
            sp.write_tombstone(9, None),
            Err(PersistenceError::ReadOnly)
        ));
        assert_eq!(sp.pending_bytes(), 0);
        assert_eq!(sp.write_offset(), offset);
        sp.commit().unwrap();
        assert!(!sp.commit_if_pending().unwrap());
        sp.seal_active().unwrap();
        assert!(matches!(
            sp.bind_model(&model("spec-b", 2)),
            Err(PersistenceError::WrongModel { .. })
        ));
        // The same model under another description would be rewritten on a
        // writable handle; here nothing is.
        assert_eq!(
            sp.bind_model(&model("spec-a-renamed", 1)).unwrap(),
            ModelBinding::Redescribed
        );
        assert_eq!(recorded(&sp), Some(RecordedModel::Key(model("spec-a", 1))));
        assert!(!sp.set_template(b"template").unwrap());
        assert_eq!(sp.template(), None);
        assert!(sp.set_tokenizer(b"tokenizer-b").is_err());
        assert!(!sp.set_tokenizer(b"tokenizer-a").unwrap());
        assert!(sp.plan_maintenance(&substrate, true).unwrap().is_none());
        assert!(matches!(
            sp.compact(&mut substrate, None),
            Err(PersistenceError::ReadOnly)
        ));
        assert!(sp.has_stream(&substrate, section("sys", 1).stream_id()));
        drop(sp);
        assert_eq!(dir_fingerprint(&dir), before);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A released writer hands the directory on while it is still alive.**
    /// What it committed is on disk, it refuses every write afterwards, and the
    /// next writable open is admitted at once rather than at its drop.
    #[test]
    fn a_released_writer_admits_the_next_and_refuses_writes() {
        let dir = tmp_dir("release_writer");
        let mut substrate = Substrate::new();
        let mut first = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        first.write_npc(&test_npc(1, "Perrin")).unwrap();
        let mut second_substrate = Substrate::new();
        assert!(matches!(
            SubstratePersistence::open_in_with_substrate(&dir, &mut second_substrate),
            Err(PersistenceError::Locked { .. })
        ));

        first.release_writer().unwrap();
        first.release_writer().unwrap();
        assert!(first.is_read_only());
        assert!(matches!(
            first.append_record(RecordType::Tokens, 0, 9, 0, 0, 0, b"refused"),
            Err(PersistenceError::ReadOnly)
        ));
        assert!(matches!(
            first.compact(&mut substrate, None),
            Err(PersistenceError::ReadOnly)
        ));
        assert_eq!(first.pending_bytes(), 0);

        let mut second =
            SubstratePersistence::open_in_with_substrate(&dir, &mut second_substrate).unwrap();
        second.write_npc(&test_npc(2, "Wyneth")).unwrap();
        second.commit().unwrap();
        drop(first);
        drop(second);
        assert_eq!(
            cast_on_disk(&dir),
            vec![1, 2],
            "the record committed at release and the successor's both survive"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A read-only open never makes a store: a workspace without one, and a
    /// workspace that does not exist, are errors and stay as they were.
    #[test]
    fn a_read_only_open_of_a_workspace_without_a_store_creates_nothing() {
        let dir = tmp_dir("read_only_missing");
        assert!(SharedSubstrate::open_in_read_only(&dir).is_err());
        assert!(dir_fingerprint(&dir).is_empty());
        let missing = dir.join("absent");
        assert!(SharedSubstrate::open_in_read_only(&missing).is_err());
        assert!(!missing.exists());
        std::fs::remove_dir_all(&dir).ok();
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

    /// Every character record in the store, by id, as a fresh open sees them.
    fn cast_on_disk(dir: &Path) -> Vec<u64> {
        let mut seen = Vec::new();
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate_and_sink(
            dir,
            &mut substrate,
            Admit::Unbound,
            |entry| {
                if entry.record.header.record_type == RecordType::Npc {
                    seen.push(entry.record.header.stream_id);
                }
            },
            &|_, _, _| {},
        )
        .unwrap();
        seen.sort_unstable();
        seen.dedup();
        seen
    }

    /// **Two writable handles on one substrate lose the records neither told
    /// the other about.**
    ///
    /// This is the `npcd` daemon's shape reproduced exactly. The character
    /// registry opens the substrate to rebuild its cast; the engine opens the
    /// *same* directory a moment later for conversation turns. Both hold a
    /// writable handle, and each built its [`SubstratePersistence::npc_locs`]
    /// from the records present at *its own* open.
    ///
    /// Nothing tells one handle about the other's appends. So a character
    /// created while the daemon runs is written through the registry's handle
    /// and is invisible to the engine's — and [`compaction::collect_live_records`]
    /// reads only the `npc_locs` of the handle doing the compacting. The
    /// character is not carried forward, and the next restart has never heard of
    /// it.
    ///
    /// Characters that predate *both* opens are in both views and survive, which
    /// is why the loss presented as intermittent rather than total: it is
    /// precisely the runtime-created ones that go.
    ///
    /// This is the failure [`SharedSubstrate`] and [`writer_lock::WriterLock`]
    /// exist to make unrepresentable — asserted in the shape the daemon actually
    /// failed in: the second writable open is now refused, and the character
    /// created through the one writer survives that writer's compaction.
    #[test]
    fn a_second_writable_handle_is_refused() {
        let dir = tmp_dir("two_owners");

        // A character that predates both opens — the control.
        {
            let mut registry = SubstratePersistence::open_in(&dir).unwrap();
            registry.write_npc(&test_npc(1, "Perrin")).unwrap();
            registry.commit().unwrap();
        }

        // Boot, in the daemon's old order: the registry opens, then the engine
        // tries to open the same directory.
        let mut registry_substrate = Substrate::new();
        let mut registry =
            SubstratePersistence::open_in_with_substrate(&dir, &mut registry_substrate).unwrap();
        let mut engine_substrate = Substrate::new();
        assert!(matches!(
            SubstratePersistence::open_in_with_substrate(&dir, &mut engine_substrate),
            Err(PersistenceError::Locked { .. })
        ));
        // A read-only handle — a tool reading the workspace — still opens.
        let mut reader_substrate = Substrate::new();
        assert!(SubstratePersistence::open_in_with_substrate_read_only(
            &dir,
            &mut reader_substrate
        )
        .is_ok());

        // Runtime: a character is created through the one writer, which then
        // runs a routine compaction.
        registry.write_npc(&test_npc(2, "Wyneth")).unwrap();
        registry.commit().unwrap();
        registry.compact(&mut registry_substrate, None).unwrap();
        drop(registry);

        assert_eq!(
            cast_on_disk(&dir),
            vec![1, 2],
            "with one writer, the character created at runtime survives its compaction"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A tool round-trip's coupling survives compaction.**
    ///
    /// [`RecordType::TurnCoupling`] is a resident payload-keyed record, exactly
    /// like `Label` / `DebugId` / `TreeMetadata`: the accounting excludes it from
    /// supersession and the walker replays it into
    /// [`crate::substrate::Substrate::couplings_of`], which means compaction has
    /// to re-synthesise it from RAM. Every one of its siblings is re-emitted by
    /// [`compaction::collect_live_records`]; this one was not, so each
    /// compaction silently unjoined every tool call from its response and
    /// [`crate::summary_tree::exchange`] began grouping the two halves of a
    /// round-trip as separate exchanges.
    ///
    /// For `npcd` that is a character's act and the world's answer to it coming
    /// apart in the summary tree.
    #[test]
    fn a_turn_coupling_survives_compaction() {
        let dir = tmp_dir("coupling");
        let turn = |idx: u32, seq: u32| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: 1,
                turn_index: idx,
                turn_id_day: 0,
                turn_id_seq: seq,
                role: 1,
                block_start: 0,
                block_end: 32,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };

        // A tool call at turn 0 and its response at turn 1, joined.
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            for (idx, seq) in [(0u32, 1u32), (1, 2)] {
                let id = sp.declare_stream(&turn(idx, seq)).unwrap().value;
                sp.append_tokens(id, b"token-bytes").unwrap();
                sp.commit_stream(id, 0).unwrap();
            }
            sp.write_turn_coupling(1, 0, sp.mint()).unwrap();
            sp.commit().unwrap();
        }

        let timeline = TimelineId::from_raw(1).unwrap();
        {
            let mut substrate = Substrate::new();
            let sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            assert!(
                substrate.couplings_of(timeline).contains(&0),
                "the coupling is durable before any compaction"
            );
            drop(sp);
        }

        // Compact, then reopen: the coupling must still be there.
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.compact(&mut substrate, None).unwrap();
        }
        let mut substrate = Substrate::new();
        SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        assert!(
            substrate.couplings_of(timeline).contains(&0),
            "compaction must carry the coupling forward — without it a tool call \
             and its response stop being one exchange"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A whole-store compaction carries a located index page by location,
    /// byte for byte.** Recovered pages are left in the log
    /// ([`crate::substrate::IndexPage::Stored`]); the rewrite copies the record
    /// verbatim, and the page read after it — and after a reload — is exactly
    /// the bytes the seal wrote. A page dropped by a rewrite would leave the
    /// turn borrowable and unindexable.
    #[test]
    fn a_located_index_page_survives_compaction_byte_for_byte() {
        let dir = tmp_dir("index_page_compact");
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 7,
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
        let page: Vec<u8> = (0..9_000u32)
            .map(|i| (i.wrapping_mul(131) % 253) as u8)
            .collect();
        let sid = {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            let sid = sp.declare_stream(&decl).unwrap().value;
            sp.append_tokens(sid, b"token-bytes").unwrap();
            sp.commit_stream(sid, 0).unwrap();
            sp.append_turn_index_page(sid, &page, sp.mint_for(RecordType::TurnIndexPage, sid))
                .unwrap();
            sp.commit().unwrap();
            sid
        };
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            let held = substrate.index_page_of(sid).expect("the page recovers");
            assert!(held.stored().is_some(), "recovered, the page is located");
            assert_eq!(held.read_with(&mut sp), page);
            sp.compact(&mut substrate, None).unwrap();
            let after = substrate
                .index_page_of(sid)
                .expect("the page survives compaction");
            assert_eq!(
                after.read_with(&mut sp),
                page,
                "the seal's bytes, after compaction"
            );
        }
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        let reloaded = substrate
            .index_page_of(sid)
            .expect("the page survives a reload");
        assert_eq!(
            reloaded.read_with(&mut sp),
            page,
            "the seal's bytes, after a reload"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The fix: one handle, shared, and the compaction carries the whole cast.
    ///
    /// Identical to the test above except that the engine *adopts* the
    /// registry's [`SharedSubstrate`] instead of opening the directory again.
    /// One append cursor, one `npc_locs` — so the character created at runtime
    /// is in the view the compactor reads.
    #[test]
    fn one_shared_handle_carries_a_runtime_created_record_through_compaction() {
        let dir = tmp_dir("one_owner");

        {
            let mut registry = SubstratePersistence::open_in(&dir).unwrap();
            registry.write_npc(&test_npc(1, "Perrin")).unwrap();
            registry.commit().unwrap();
        }

        // Boot: the registry opens once, and the engine takes that same handle.
        let shared = SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap();
        let engine = shared.clone();

        // Runtime: a character is created, through the registry's side.
        shared.persistence.call(|p| {
            p.write_npc(&test_npc(2, "Wyneth")).unwrap();
            p.commit().unwrap();
        });

        // The engine compacts — the same owner, the same view.
        let substrate = &engine.substrate;
        engine
            .persistence
            .call(|p| p.compact_shared(substrate, None))
            .unwrap();
        drop(shared);
        drop(engine);

        assert_eq!(
            cast_on_disk(&dir),
            vec![1, 2],
            "a shared handle compacts against a view that includes every \
             character, whenever it was created"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// **A custom object is durable, last-writer-wins, and survives compaction.**
    ///
    /// The whole point of the type: `npcd` writes the command-table flag once and
    /// it comes back after a reboot, keeps the newest value when re-written, and
    /// is not lost when the log is rewritten (the silent-drop failure mode
    /// `survival.rs` documents). Exercises the `SharedSubstrate` write/read pair
    /// across an open → re-write → compact → reopen cycle.
    #[test]
    fn a_custom_object_is_durable_lww_and_survives_compaction() {
        let dir = tmp_dir("custom_object");

        // Write once, read it straight back from the same handle.
        {
            let shared = SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap();
            let mut obj =
                CustomObjectPayload::new("command-table", serde_json::json!({ "open": true }));
            obj.metadata
                .insert("world".to_string(), "vault".to_string());
            shared.put_custom_object(obj).unwrap();

            let back = shared.custom_object("command-table").unwrap();
            assert_eq!(back.blob, serde_json::json!({ "open": true }));
            assert_eq!(
                back.metadata.get("world").map(String::as_str),
                Some("vault")
            );
            assert!(
                shared.custom_object("never-written").is_none(),
                "an unknown key has no object"
            );
        }

        // A fresh open replays the log — the object is durable.
        {
            let shared = SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap();
            let back = shared.custom_object("command-table").unwrap();
            assert_eq!(back.blob, serde_json::json!({ "open": true }));

            // Re-write the same key: last-writer-wins.
            shared
                .put_custom_object(CustomObjectPayload::new(
                    "command-table",
                    serde_json::json!({ "open": false }),
                ))
                .unwrap();
            assert_eq!(
                shared.custom_object("command-table").unwrap().blob,
                serde_json::json!({ "open": false }),
                "the newest write wins in RAM"
            );

            // Compact against this view, then confirm the winner survived.
            let substrate = &shared.substrate;
            shared
                .persistence
                .call(|p| p.compact_shared(substrate, None))
                .unwrap();
        }

        // Reopen after compaction: only the last value is present, and the
        // superseded first write did not resurrect.
        {
            let shared = SharedSubstrate::open_in(&dir, Admit::Unbound).unwrap();
            let back = shared.custom_object("command-table").unwrap();
            assert_eq!(
                back.blob,
                serde_json::json!({ "open": false }),
                "the last-written value survives compaction and reload"
            );
            assert!(
                back.metadata.is_empty(),
                "the second write carried no metadata, so the winner has none"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A substrate takes the first model's key, opens again under it without
    /// writing, rewrites the record when the same model is described anew,
    /// and refuses another model — on the handle that bound it and on a reopen.
    #[test]
    fn a_substrate_is_bound_to_the_first_model_that_writes_it() {
        let dir = tmp_dir("modelkey");
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(
                sp.bind_model(&model("qwen3-30b-a3b", 1)).unwrap(),
                ModelBinding::New
            );
            let off1 = sp.manifest().model_key.unwrap().offset;
            assert_eq!(
                sp.bind_model(&model("qwen3-30b-a3b", 1)).unwrap(),
                ModelBinding::Same
            );
            assert_eq!(
                sp.manifest().model_key.unwrap().offset,
                off1,
                "the same key appends no record"
            );
            assert_eq!(
                sp.bind_model(&model("qwen3-30b-a3b-mirror", 1)).unwrap(),
                ModelBinding::Redescribed
            );
            assert!(sp.manifest().model_key.unwrap().offset > off1);
            let err = sp.bind_model(&model("qwen3-235b", 2)).unwrap_err();
            let msg = err.to_string();
            let recorded_model = model("qwen3-30b-a3b-mirror", 1).describe();
            let running_model = model("qwen3-235b", 2).describe();
            assert!(
                recorded_model.starts_with("qwen3-30b-a3b-mirror (arch Qwen3, weights 01010101…")
                    && running_model.starts_with("qwen3-235b (arch Qwen3, weights 02020202…"),
                "{recorded_model} / {running_model}"
            );
            assert!(
                msg.contains(&format!(
                    "belongs to {recorded_model}, not {running_model}."
                )),
                "the refusal names both models: {msg}"
            );
            sp.commit().unwrap();
        }
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("qwen3-30b-a3b-mirror", 1)))
            );
            assert!(matches!(
                sp.bind_model(&model("qwen3-235b", 2)),
                Err(PersistenceError::WrongModel { .. })
            ));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A substrate written before keys records the spec text. A model that
    /// matches it is adopted — the key is written and decides from then on —
    /// and one that does not is refused.
    #[test]
    fn an_older_substrate_adopts_the_key_of_the_model_its_spec_names() {
        let spec = b"candle-conversation model-spec v1\n\
            arch=Qwen3\n\
            chat_format=ChatML\n\
            model_repo=test/repo\n\
            model_filename=qwen3-8b.gguf\n\
            tokenizer_repo=Qwen/Qwen3-8B\n\
            max_seq_len=40960\n";
        let dir = tmp_dir("modelkey_legacy");
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.append_record(RecordType::ModelKey, 0, 0, 0, 0, 0, spec)
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert!(matches!(
                sp.bind_model(&model("qwen3-14b.gguf", 1)),
                Err(PersistenceError::WrongModel { .. })
            ));
            assert_eq!(
                sp.bind_model(&model("qwen3-8b.gguf", 1)).unwrap(),
                ModelBinding::Adopted
            );
            sp.commit().unwrap();
        }
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("qwen3-8b.gguf", 1)))
            );
            // Adopted, the key decides: other weights under the same name are
            // another model.
            assert!(matches!(
                sp.bind_model(&model("qwen3-8b.gguf", 2)),
                Err(PersistenceError::WrongModel { .. })
            ));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A substrate binds to one tokenizer, and a second one is refused.
    ///
    /// This used to assert that "changed bytes are re-embedded", which is the
    /// behaviour that let a model swap rewrite the vocabulary out from under
    /// every turn already sealed in the log — the ids stay, their meaning
    /// changes, and nothing reports it. The binding is the invariant now.
    #[test]
    fn tokenizer_binds_the_substrate_and_a_second_one_is_refused() {
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
            let err = sp
                .set_tokenizer(&v2)
                .expect_err("a different tokenizer must be refused, not appended");
            let msg = err.to_string();
            assert!(
                msg.contains("bound to a different tokenizer"),
                "the refusal must name the cause: {msg}"
            );
            assert_eq!(
                sp.tokenizer_sha256(),
                Some(sha256(&v1)),
                "a refused binding leaves the recorded tokenizer untouched"
            );
            sp.commit().unwrap();
        }
        {
            // The hash is recovered from disk, so the binding survives a
            // reopen — including the refusal.
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(sp.tokenizer_sha256(), Some(sha256(&v1)));
            assert!(
                !sp.set_tokenizer(&v1).unwrap(),
                "reopened log recognises the unchanged tokenizer by hash"
            );
            assert!(
                sp.set_tokenizer(&v2).is_err(),
                "the binding is still enforced after a reopen"
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
    fn model_key_survives_reopen() {
        let dir = tmp_dir("ms_reopen");
        {
            let mut sp = SubstratePersistence::open_in(&dir).unwrap();
            sp.bind_model(&model("the-model", 1)).unwrap();
            sp.set_template(b"the-template").unwrap();
            sp.commit().unwrap();
        }
        {
            let sp = SubstratePersistence::open_in(&dir).unwrap();
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("the-model", 1)))
            );
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
            turn_id = sp.declare_stream(&turn).unwrap().value;
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

    /// **Full compaction carries a live run longer than one read stripe
    /// whole**: seventy one-MiB chunks back to back, read back byte for byte
    /// after the rewrite and after a reload.
    #[test]
    fn compact_carries_a_live_run_past_one_stripe() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("compact_stripes");
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 56,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = decl.stream_id();
        let big = |index: u64| ChunkPayload {
            kv_bytes: (0..1024 * 1024u64)
                .map(|i| ((i + index * 17) % 251) as u8)
                .collect(),
            ..chunk_payload(index as u32)
        };
        const LIVE: u64 = 70;
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            sp.write_chunk(sid, LIVE, 32, 4, None, &chunk_payload(1))
                .unwrap(); // superseded below
            for i in 0..LIVE {
                sp.write_chunk(sid, i, 32, 4, None, &big(i)).unwrap();
            }
            sp.write_chunk(sid, LIVE, 32, 4, None, &chunk_payload(2))
                .unwrap();
            sp.commit().unwrap();
        }
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.compact(&mut substrate, None).unwrap();
            for i in 0..LIVE {
                assert_eq!(
                    sp.read_chunk(&substrate, sid, i).unwrap(),
                    big(i),
                    "chunk {i}"
                );
            }
        }
        let mut substrate = Substrate::new();
        let mut sp = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
        for i in 0..LIVE {
            assert_eq!(
                sp.read_chunk(&substrate, sid, i).unwrap(),
                big(i),
                "chunk {i}"
            );
        }
        assert_eq!(
            sp.read_chunk(&substrate, sid, LIVE).unwrap(),
            chunk_payload(2)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn compact_drops_dead_records_and_reopens_identical() {
        use crate::persistence::streams::{StreamDecl, TurnDecl};
        let dir = tmp_dir("compact");
        // The chunk stream needs a decl, else it is an orphan whose chunks are
        // (correctly) reclaimed — here we are testing dead-record dropping, so the
        // stream must be legitimately live.
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: 55,
            turn_index: 0,
            turn_id_day: 0,
            turn_id_seq: 1,
            role: 2,
            block_start: 0,
            block_end: 1,
            layer_id: 1,
            group_id: 1,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: Vec::new(),
            tags: Vec::new(),
        });
        let sid = decl.stream_id();
        let live_chunk = chunk_payload(7);
        {
            let mut substrate = Substrate::new();
            let mut sp =
                SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            sp.declare_stream(&decl).unwrap();
            // Dead weight: superseded model records (one model, redescribed)
            // and stale chunk-0 bodies.
            for s in ["qwen3-8b", "qwen3-14b", "qwen3-30b"] {
                sp.bind_model(&model(s, 1)).unwrap();
            }
            sp.bind_model(&model("qwen3-235b-live", 1)).unwrap();
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
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("qwen3-235b-live", 1)))
            );
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
            assert_eq!(
                recorded(&sp),
                Some(RecordedModel::Key(model("qwen3-235b-live", 1)))
            );
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
