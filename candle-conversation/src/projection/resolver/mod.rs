//! The workspace-shared conversation handle, the substrate it reads and edits, and
//! the target-aware read over that substrate.
//!
//! Conversations in one workspace share a single lock-free substrate. Projection
//! reads one version without waiting, a seal publishes a new version, and durable
//! writes go to the log's owner so a seal never waits on the disk. Reset does not
//! clear the shared store, since other conversations would lose their history.
//!
//! - `belief_collections`: scores section collections against their galleries.
//! - `belief_gallery`: the gallery each collection votes with, and its preload.
//! - `belief_groups`: scores a layer's belief-driven turn groups.
//! - `belief_scoring`: which scopes a scoring pass teaches, and the belief scans.
//! - `conv_state`: archived flag, branch choices, debug id and sidebar record.
//! - `handle`: creating and opening the conversation handle.
//! - `lifecycle`: tombstones, releases and storage-shedding marks.
//! - `log_read`: reads of index pages and stored records from the redo log.
//! - `log_reconstruct`: rebuilding the substrate from the redo log at start.
//! - `maintenance`: group commit, shutdown drain and segment maintenance.
//! - `metadata`: label, conv id and custom metadata records.
//! - `recurrent_state`: recurrent snapshots and branch checkpoints in the log.
//! - `sections`: section declaration, triage and load counts.
//! - `targeted_read`: what a projection may select for one target.
//! - `turn_log`: recording turns, declarations and summary nodes.
//! - `turn_storage`: key-value state across device, RAM and the log.
//! - `warm_collection`: warm-up of section collection backgrounds.
//! - `warm_ingest`: warm-up of ingest file and cluster backgrounds.
//! - `warm_replay`: replay of recent dialogue turns at load.
//! - `warm_replay_progress`: how far that replay has been scored.
//! - `tests_belief`: belief rules, scope segmentation and policy parsing.
//! - `tests_persistence`: directory cleanup, reopen and read-only behaviour.
//! - `tests_recurrent_snapshot`: snapshot reads while compaction holds the owner.
//! - `tests_turns`: lineage, metadata visibility and adopted index pages.
//! - `tests_warm`: which groups and collections warm-up may teach.

use candle::engine_events::note;
use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::iter::repeat_n;
use std::ops::Range;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::{ArcSwap, ArcSwapOption};

mod maintenance_op;
pub use maintenance_op::MaintenanceOp;
use maintenance_op::Pass;

use super::adaptive::{ScanCases, ScanNorm};
use super::belief_files::{assemble_file_scans, cut_excluded, scan_file_scans, FileScan};
use super::belief_scan_split::GROUP_SCAN_SPLIT;
use super::fill_report::{Standing, Unpooled};
use super::group_assembly::{
    held, keep, Assemblies, AssemblyKey, AssemblyStamp, GroupAssembly, Whose,
};
use super::ids::{
    process_timeline_allocator, GroupId, LayerId, SectionId, TimelineAllocator, TimelineId,
    TurnIndex, TurnKey,
};
use super::owned_sections::OwnedSection;
use super::project::ProjectionTarget;
use super::scan_census;
use super::scan_plans::ScanPlans;
use super::scan_stats::{scan_stats, GroupScan};
use super::schema::{
    CorruptTurnPolicy, GatherScope, GroupSchema, LayerSchema, Recency, Schema, SectionCollection,
    SystemPromptItem, SystemPromptSchema,
};
use super::warm_pool;
use super::working_set_observe::{working_set_config, working_set_fresh};
use super::working_set_pick::{SlotBoundary, WorkingSetMembers};
use crate::cancel::CancelToken;
use crate::conversation::section_end_state;
use crate::error::ConversationError;
use crate::index_pages::StoredPages;
use crate::normalization::{kind_z_band, path_kind, ChildKey, NormalizationCache, ScopeKey};
use crate::persistence::chunk_window_integrity::{first_divergent_chunk, WindowDivergence};
use crate::persistence::content_hash::{
    branch_checkpoint_stream_id, section_end_state_stream_id, snapshot_stream_id, turn_stream_id,
    ContentHash,
};
use crate::persistence::identity::{Minter, RecordIdentity, GEN_DETACHED};
use crate::persistence::integrity::{classify_turn, TurnIntegrity};
use crate::persistence::manifest::{self, ConvState, RecordLoc};
use crate::persistence::record::{
    BranchCheckpointPayload, DistillMode, DistillPayload, RecordType, SectionGeneration,
    SnapshotPayload, TreeMetadataPayload,
};
use crate::persistence::remote::{turn_is_complete, PendingCensus, RemoteBatch};
use crate::persistence::resume::{
    read_persisted_section_windows, RecoveredTurnMeta, TurnChunkGrid,
};
use crate::persistence::sealed_reader::{SealedReader, SEALED_READ_ATTEMPTS};
use crate::persistence::streams::{ContentAddress, SectionDecl, StreamDecl, StreamId, TurnDecl};
use crate::persistence::vfs::{VfsEventId, VfsEventPayload, VfsWrite};
use crate::persistence::writer::WriteJob;
use crate::persistence::{
    PersistenceOwner, Result as PersistenceResult, SharedSubstrate, SubstratePersistence,
};
use crate::projection::adaptive::attention_mass;
use crate::provenance::gallery_arena::{
    scan_split, BeliefArena, GroupSet, Holder, Keep, PagedSegment, PagedWindow,
};
use crate::provenance::heads_per_group;
use crate::provenance::{
    decode_wide_sigs_for_scoring, score_slots_fused, score_slots_grouped, score_slots_weighted,
    FusionMode, GalleryArena, SigWindow, WideQSig,
};
use crate::scheduler::exported_state::SharedState;
use crate::scheduler::profile;
use crate::substrate::pcoll::POrdMap;
use crate::substrate::{
    ContentResolver, IndexPage, ProjectionScores, ResidenceIndex, ResidenceSlab, RestoredTurn,
    StoredSequence, Substrate, SubstrateLock, SubstrateRead, TreeNodeMeta, TurnPartWrite,
    TurnStream,
};
use crate::summary_tree::exchange::{exchanges, over_normals};
use crate::summary_tree::{SelectionDiagnostics, SelectionOrigin, TurnKind};
use crate::tenant_rank::TenantRank;
use crate::token_buffer::TokenBuffer;
use crate::turn::Role;
use crate::turn_layout::{TurnLayout, TurnSegment};
use crate::working_set::marks::is_dialogue;
use candle_nn::kv_cache::{SealedSequence, Sweep};

// ── Conversation ──────────────────────────────────────────────────────────────

/// Workspace-shared handle to the per-turn record store.
///
/// Multiple conversations in the same workspace clone this handle; they all
/// see (and write into) the same underlying [`Substrate`], which holds no lock
/// (`docs/substrate_lock_free.md`):
///
/// - **Read** — projection reads one version of the substrate for the
///   duration of a single `project()` call via [`Conversation::read`], never
///   waiting. Returns a [`SubstrateRead`] that implements [`ContentResolver`].
/// - **Edit** — at seal time, each conversation edits its new turn into the
///   shared store ([`Conversation::update`]), publishing a new version.
/// - **Durable writes** — go to the log's owner thread ([`PersistenceOwner`]):
///   queued and forgotten, or called and waited for alone.
/// - **Reset** — does *not* clear the shared store (other conversations
///   would lose their history).  Sequence-level reset only drops local
///   KV state via the scheduler.
#[derive(Clone)]
pub struct Conversation {
    pub(super) inner: Arc<SubstrateLock>,
    /// The process's one timeline allocator ([`process_timeline_allocator`]),
    /// so a timeline id minted on one substrate is never minted on another
    /// attached to the same engine.
    allocator: Arc<TimelineAllocator>,
    /// This substrate's ingest-cancel latch — see [`crate::cancel`]. Its own,
    /// so a shutdown of one tenant's ingest leaves another's running.
    cancel: CancelToken,
    /// The rank of the tenant this substrate belongs to — where its sequences
    /// stand against another tenant's on the same scheduler
    /// ([`crate::tenant_rank`]).
    tenant_rank: TenantRank,
    /// The mandatory persistence layer — every turn is recorded into its
    /// redo log, by the thread
    /// that owns it. Every redo-log append is queued here
    /// ([`PersistenceOwner::enqueue`]), so a seal on the inference thread never
    /// waits on the disk. Shared across clones; the last drop drains, commits
    /// and joins the owner.
    pub(super) persistence: Arc<PersistenceOwner>,
    /// The substrate's identity minter, shared with `persistence`. A write
    /// that changes the substrate in memory and hands its record to the owner
    /// mints here first, so the value and its record carry one identity
    /// (`docs/replicated_substrate.md` §6.2).
    pub(super) minter: Minter,
    /// Runtime, in-memory score normalization (per-scope backgrounds). NOT
    /// persisted — rebuilt from the substrate's existing turns on first use, then
    /// evolved as new turns seal. Shared across clones of this handle so learning
    /// pools over all sessions. See `docs/provenance_score_normalization.md`.
    normalization: Arc<NormalizationCache>,
    /// Whether the dialogue warm-up replay has been handed out since it was last
    /// armed — once per process, and again after each
    /// [`Self::reset_normalization_warm`]. See [`Self::take_warm_replay`].
    normalization_warm: Arc<AtomicBool>,
    /// How far the replay [`Self::take_warm_replay`] handed out has been
    /// scored — the `/v1/status` warm-up bar.
    warm_replay_progress: Arc<WarmReplayProgress>,
    /// The ingest files whose backgrounds are warmed, and the warm-up running.
    ingest_warm: Arc<IngestWarmState>,
    /// The `/v1/status` maintenance indicator: `(segment_count, last_op,
    /// running)`. `running` is `true` only while a maintenance op's relocation
    /// is in flight, so the sidebar can show a live spinner (vs. a settled ✓
    /// for the last completed op). Refreshed by the persistence thread each
    /// pass and read with one atomic load, so the status endpoint never waits
    /// on the log's owner.
    maintenance: Arc<ArcSwap<MaintenanceStatus>>,
    /// Reads durable records without the owner — the recurrent-snapshot read
    /// at every hybrid-model admission, a cold load. Taken once at
    /// construction; see [`SealedReader`].
    sealed: SealedReader,
    /// The most recently read prompt-branch checkpoint, held decoded and keyed
    /// by the content prefix it belongs to.
    ///
    /// Every conversation opened on a branch installs that branch's recurrent
    /// state, and the durable record is large — ~63 MiB for a 40-layer hybrid.
    /// Without this, each open re-read the whole record off the log and decoded
    /// it again under the persistence mutex, which is the same mutex the writer
    /// needs: measured at 144 ms per conversation across 817 opens of one
    /// branch (~51 GB of reads) during the calibration phase, all of it on the
    /// calling thread with the GPU idle behind it.
    ///
    /// One entry is the right size. The key is a content hash, so a branch
    /// change misses and refills exactly once, and the phases that open many
    /// conversations all open them on a single shared prompt branch.
    branch_checkpoint: Arc<ArcSwapOption<(ContentHash, Arc<BranchCheckpointPayload>)>>,
    /// Each collection's assembled belief gallery, with the substrate's
    /// [`Substrate::gallery_epoch`] it was assembled at.
    ///
    /// Assembling walks every stream in the substrate to find the collection's
    /// members, and it ran on every reprojection: measured at ~194 ms for the
    /// tool catalog on a 10M-token substrate, against a 4 ms scan. Keyed on the
    /// epoch, it reruns only when a stream, blob or tombstone actually changed —
    /// about once per sealed turn.
    gallery_cache: Arc<papaya::HashMap<GalleryKey, Arc<(u64, BeliefGallery)>>>,
    /// Each belief turn group's assembled scan, per target, reused while the
    /// substrate it was gathered from stands (`group_assembly`).
    group_assemblies: Arc<Assemblies>,
    /// How this workspace's prompt sections came to be resident since it was
    /// opened: restored from the redo log, or prefilled. A restart on an
    /// unchanged prompt restores every one; a prefill there is a section the
    /// log failed to bring back, paid as a forward pass on every boot. See
    /// [`Self::section_loads`].
    section_loads: Arc<SectionLoadCounters>,
    /// The persistence handle was opened read-only
    /// ([`SharedSubstrate::open_in_read_only`]): every in-RAM change is made as
    /// usual, and every synchronous durable write is skipped and reported as a
    /// success. Nothing that was never written is given a disk location — a
    /// turn or chunk stays RAM-resident. Read once from the handle at
    /// construction; a handle's mode never changes.
    read_only: bool,
    /// The throwaway directory an [`Self::ephemeral`] conversation's log lives
    /// in, removed when the last clone of that conversation drops.
    ///
    /// `None` for every conversation opened on a real workspace — those own a
    /// directory somebody chose and must never delete it.
    ///
    /// **Without this the temp directory outlived the process.** An ephemeral
    /// conversation is a throwaway log, and `LogFile` grows in 64 MiB extents,
    /// so the first record written costs 64 MiB on disk — reclaimed only when a
    /// segment *seals*, which a throwaway never does. Every ephemeral
    /// conversation therefore left 64 MiB behind permanently: summarisation
    /// makes them, tests make them, and 727 of them had accumulated in `%TEMP%`
    /// on this machine, ~46 GB, alongside the same leak from the test helpers.
    /// Held behind the same `Arc` as everything else here so a clone does not
    /// delete the directory the original is still writing to.
    ///
    /// **Declared last on purpose.** Rust drops fields in declaration order, so
    /// `persistence` above goes first — draining, committing and joining the
    /// log's owner thread, which closes every file — and only then does this
    /// remove the directory. Windows refuses to delete a directory holding an
    /// open file, so a guard that ran before the owner released its handles
    /// would silently fail and leave the 64 MiB behind, which is the bug it
    /// exists to fix.
    ///
    /// Never read, and that is the point: the whole of its behaviour is in
    /// `Drop`. Holding it is the feature.
    #[allow(dead_code)]
    ephemeral_dir: Option<Arc<TempDirGuard>>,
}

/// The maintenance indicator: `(segment_count, last_op, running)` — see the
/// `maintenance` field of [`Conversation`].
pub type MaintenanceStatus = (usize, Option<(String, u64)>, bool);

/// Removes a directory when the last holder drops it.
///
/// Deliberately best-effort: a failure to remove a *temp* directory is not
/// worth failing a drop over, and on Windows a file another process still has
/// open will refuse. The alternative to best-effort is a panic in `Drop`, which
/// aborts.
#[derive(Debug)]
pub struct TempDirGuard {
    path: std::path::PathBuf,
}

impl TempDirGuard {
    pub fn new(path: std::path::PathBuf) -> Self {
        Self { path }
    }

    /// The directory this guard will remove.
    ///
    /// Only the tests that assert the removal actually happens have any reason
    /// to ask — production code holds the guard and never looks inside it.
    #[cfg(test)]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.path) {
            // `NotFound` is the ordinary outcome when a test removed it itself.
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::debug!(
                    "temp directory {:?} not removed: {e} — it will be left on disk",
                    self.path
                );
            }
        }
    }
}

impl Default for Conversation {
    /// An ephemeral conversation — see [`Conversation::ephemeral`].
    fn default() -> Self {
        Self::ephemeral()
    }
}

mod belief_collections;
mod belief_files_kinds;
mod belief_gallery;
mod belief_groups;
mod belief_scoring;
mod handle;
mod ingest_warm_state;
pub use ingest_warm_state::IngestWarmState;
mod warm_collection;
mod warm_ingest;
mod warm_replay;
mod warm_replay_progress;
#[cfg(test)]
use warm_ingest::*;
use warm_replay_progress::WarmReplayProgress;
mod conv_state;
mod lifecycle;
mod log_read;
mod log_reconstruct;
mod maintenance;
mod metadata;
mod recurrent_state;
mod sections;
mod targeted_read;
#[cfg(test)]
mod tests_belief;
#[cfg(test)]
mod tests_persistence;
#[cfg(test)]
mod tests_recurrent_snapshot;
#[cfg(test)]
mod tests_turns;
#[cfg(test)]
mod tests_warm;
mod turn_log;
mod turn_storage;

pub use belief_gallery::*;
pub use belief_scoring::*;
pub(crate) use handle::*;
pub use sections::*;
pub use targeted_read::*;
pub use warm_collection::*;
pub use warm_replay::*;
