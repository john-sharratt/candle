//! The substrate's persistence layer: an append-only, segmented redo log that is the complete, durable image of a substrate.
//!
//! A substrate is always backed by its log, and there is no in-memory-only mode. Records carry an identity of origin, sequence and stamp, so the store can replicate, compact and recover from its own bytes. Recovery takes the header-index chain where it is intact and walks the log where it is not, and reads route to whichever segment holds each record. The module is split by concern, one file for each part.
//!
//! - `content_hash`: the deterministic content-hash chain.
//! - `record`: the record types and the framing codec.
//! - `streams`: stream identity and the `StreamDecl` payload.
//! - `log_file`: the append-only redo-log file.
//! - `walker`: the skip-load record walk.
//! - `manifest`: the in-RAM last-writer-wins singleton index.
//! - `header_index`: the batched record-digest chain.
//! - `recovery`: chain-first recovery with a forward-walk fallback.
//! - `accounting`: O(1) live and dead byte accounting for compaction.
//! - `survival`: how each record type outlives a rewrite.
//! - `chunk_window_integrity`: the per-layer chunk window-geometry comparison.
//! - `inherit`: multi-log inheritance and the shared cache.
//! - `opening`: opening the layer and recovering it from disk.
//! - `appends`: appending a record and keeping the manifest in step.
//! - `records`: appending each kind of record a substrate writes.
//! - `reads`: reading records back by identity or location.
//! - `binding`: binding the substrate to its model and tokenizer.
//! - `chunks`: KV chunk records written and read.
//! - `durability`: staging, write-through, group commit and sealing.
//! - `minting`: minting the identity of each originated record.
//! - `tracking`: where each kind of record currently lives.
//! - `replication`: which origin records this member holds.
//! - `compacting`: reclaiming dead weight by rewriting live records.
//! - `shared`: one writable handle shared between a host and its engine.
//! - `adoption`, `appended`, `applied_mix`, `chunk_plan`, `cold_load`, `cold_source`, `dead_ratio`, `deleted`, `elevate`, `flush`, `gauge`, `held_groups`, `held_types`, `index_q8_migration`, `integrity`, `joined`, `le_reader`, `listing`, `live_indexes`, `liveness_audit`, `log_facts`, `model_binding`, `open_progress`, `own_hold`, `owner`, `payload_parts`, `pipeline`, `rebase`, `relocation_watch`, `replication_load`, `resume`, `retired`, `rewritten`, `sealed_reader`, `segment`, `segmented_log`, `space`, `stamped_locs`, `stranded`, `stripes`, `thread`, `tombstone_expiry`, `transfer`, `triggers`, `vanished_segment`, `warm_asks`, `warm_ledger`, `writer`, `writer_lock`, `written_snapshots`: the remaining modules of the layer, each named for its own concern.
//! - Tests: `adoption_tests`, `convergence_tests`, `expiry_tests`, `joined_tests`, `rebase_tests`, `replica_tests`, `tests_chunks`, `tests_compaction`, `tests_durability`, `tests_identity`, `tests_records`, `tests_sharing`, `tests_support`.

pub mod accounting;
pub mod adoption;
#[cfg(test)]
mod adoption_tests;
pub mod appended;
pub mod applied_mix;
pub mod blob;
pub mod cast;
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
pub mod gauge;
pub mod header_index;
mod held_groups;
mod held_types;
pub mod identity;
pub mod index_q8_migration;
pub mod inherit;
pub mod integrity;
pub mod joined;
#[cfg(test)]
mod joined_tests;
pub mod kept_objects;
mod le_reader;
pub mod listing;
pub mod live_indexes;
pub mod liveness_audit;
mod log_facts;
pub mod log_file;
pub mod maintenance;
pub mod manifest;
mod model_binding;
pub mod open_progress;
mod own_hold;
pub mod owner;
pub mod payload_parts;
pub mod pipeline;
pub mod rebase;
#[cfg(test)]
mod rebase_tests;
pub mod record;
pub mod record_reader;
pub mod recovery;
#[cfg(test)]
mod regeneration_tests;
pub mod relocation_watch;
pub mod remote;
pub mod replica;
#[cfg(test)]
mod replica_tests;
pub mod replication_load;
pub mod resume;
pub mod retired;
pub mod rewritten;
pub mod sealed_reader;
pub mod segment;
pub mod segmented_log;
pub mod space;
pub mod stamped_locs;
pub mod stranded;
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
pub mod warm_asks;
pub mod warm_ledger;
pub mod writer;
pub mod writer_lock;
pub mod written_snapshots;

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
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
use applied_mix::AppliedMix;
use blob::{blob_bytes, retired_blob, BlobIndex, BlobKey};
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
use kept_objects::ImportReport;
use liveness_audit::{keyed_by_payload, resident_key, ResidentKey};
pub use log_facts::LogFacts;
use manifest::{ChunkLoc, Manifest, RecordLoc};
pub use model_binding::{Admit, BoundSubstrate, ModelBinding};
use open_progress::MibMeter;
pub use open_progress::{OpenProgress, OpenProgressHook, OpenStage};
use own_hold::OwnHold;
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
use rewritten::{Rewritten, Untaken};
use sealed_reader::SealedReader;
use segment::SegmentId;
use segmented_log::{store_mib, SegmentedLog};
use stamped_locs::StampedLocs;
use stranded::Stranded;
use streams::{ContentAddress, StreamDecl, StreamId, StreamKind, StreamRef};
use survival::RecordCensus;
use tombstone_expiry::{TombstoneExpiry, DEFAULT_TOMBSTONE_RETENTION};
use vanished_segment::retry_vanished;
use vfs::{vfs_events_of, VfsEventId, VfsEventPayload, VfsIndex, VfsWrite};
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

    /// A compaction's re-apply did not take back every chunk it carried, so
    /// it settled nothing and kept the segments it replaced
    /// (`persistence::rewritten`).
    #[error(
        "the compaction's re-apply did not take back {chunks} chunk(s) of {streams} stream(s): \
         the replaced segments are kept and nothing was forgotten"
    )]
    RewriteUntaken { streams: usize, chunks: usize },

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
    /// The writes made since the running maintenance op was planned that its
    /// relocation must step around — see [`RelocationWatch`]. `Some` from the
    /// plan until the op's sources are unlinked (or the op fails), so it also
    /// marks an op in flight: a handle runs one at a time.
    relocation_watch: Option<RelocationWatch>,
    /// Records the substrate index named inside a segment an op was about to
    /// drop, which kept that segment linked ([`stranded`]). Choosing an op
    /// leaves such a segment alone while the index still names the record
    /// there. In-memory; a restart rebuilds the index from the log.
    stranded: Stranded,
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
    /// Published for any thread to read without a call on the owner: every
    /// record type the substrate has held since it opened, as bit `tag` — the
    /// open walk's census, and each type appended since, written here or
    /// applied from another member, which a build replacing this one must
    /// decode or it would not see them (`docs/replicated_substrate.md` §3.9);
    /// a write failed for lack of space (§7.7, see [`space`]); and the dead
    /// ratio the last liveness scan found.
    facts: LogFacts,
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
    /// What this store has applied from other members since it opened, by
    /// record type — for the memory census ([`gauge`]).
    applied_mix: AppliedMix,
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
    /// How long the same gap has capped [`Self::own_sendable_through`].
    own_hold: OwnHold,
    /// `oseq`s this process minted whose append failed: never coming, so no
    /// hold on [`Self::own_sendable_through`].
    own_unwritten: OriginApplied,
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

mod appends;
mod binding;
mod chunks;
mod compacting;
mod durability;
mod minting;
mod opening;
mod reads;
mod records;
mod replication;
mod shared;
#[cfg(test)]
mod tests_chunks;
#[cfg(test)]
mod tests_compaction;
#[cfg(test)]
mod tests_durability;
#[cfg(test)]
mod tests_identity;
#[cfg(test)]
mod tests_records;
#[cfg(test)]
mod tests_sharing;
#[cfg(test)]
mod tests_support;
mod tracking;

use binding::*;
use replication::*;
pub use shared::*;
#[cfg(test)]
pub(crate) use tests_support::*;
