//! The per-session store that holds every turn, section and timeline the
//! projection reads, and where each turn's KV currently lives.
//!
//! A `Substrate` is one version of a persistent collection, so cloning it is
//! cheap and an edit copies only what it changes, while the KV residences live in
//! a slab that every version shares. The redo log is replayed into it at startup,
//! and every later write goes through the same stamped apply, so members that
//! hold the same records hold the same state. The projection reads it through the
//! `ContentResolver` interface, which yields counts and scores rather than
//! content. Eviction and demotion are bounded by the working sets that live slots
//! hold.
//!
//! - `apply_records`: folds replayed records into state, the greatest stamp wins.
//! - `entries`: per-turn, per-section and per-timeline records and their residence handles.
//! - `eviction`: picks hot and warm copies to drop, never a live working set's.
//! - `live_state`: tool couplings, reasoning windows, summary forest and pending queue.
//! - `loading`: rebuilds state from the redo log on startup.
//! - `marking_dead`: applies tombstones and releases their KV now.
//! - `metadata`: labels, archive state, branches, custom metadata and calibrations.
//! - `registry`: timelines, groups, scopes, fork lineage and working sets.
//! - `resolver`: the `ContentResolver` interface the projection queries.
//! - `scored`: transient per-projection scores and the views that carry them.
//! - `sections`: prompt sections, their index pages and tier state.
//! - `tiers`: moves residences between VRAM, host RAM and disk.
//! - `transient_owned`: ephemeral and owned timelines and sections.
//! - `turn_access`: turn content, layout and K/V reads for the projection.
//! - `tests_cold_refs`: cold references through compaction and reload.
//! - `tests_copies_and_distill`: copy, distill and demote rules against live working sets.
//! - `tests_eviction`: warm purge and hot eviction under a budget.
//! - `tests_labels_tombstones`: label merge and tombstones in any replay order.
//! - `tests_listing`: tag filters, epochs, archive flags and prefix listings.
//! - `tests_timeline_meta`: lineage, base repair and turn resolution.
//! - `tests_transient_owned`: ephemeral and owned section release.
//! - `tests_working_set`: working sets, scopes and inherited chains.

pub mod calibration_record;
mod compacted;
#[cfg(test)]
mod custom_object_tests;
mod deleted_set;
mod dominance;
mod ended_sections;
mod faces;
mod file_gallery;
pub mod gauge;
mod group_epoch;
mod label_records;
mod lock;
#[cfg(test)]
mod order_independence_tests;
pub mod pcoll;
mod pending;
mod replicated_state;
pub mod residence;
mod stamps;
mod stream_index;
mod supersession;
mod tag_index;
mod tombstones;
mod turn_placement;

use std::sync::OnceLock;

use ahash::{AHashMap, AHashSet};
use candle_nn::kv_cache::{rewrite_sealed, FormatCensus, QuantFormat, SealedSequence, Sweep};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::conversation::{
    is_shared_stream_name, window_sealed_tokens, SHARED_SECTION_NAME_PREFIX,
};
use crate::index_pages::{self, StoredPages};
use crate::normalization::Phase;
use crate::persistence::blob::{retired_blob, BlobKey};
use crate::persistence::content_hash::{
    section_end_state_stream_id, snapshot_stream_id, turn_stream_id,
};
use crate::persistence::deleted::DeletedPayload;
use crate::persistence::identity::{RecordIdentity, VersionStamp};
use crate::persistence::manifest::{
    decode_conv_state_payload, decode_label_payload, ChunkLoc, ConvMeta, ConvState, RecordLoc,
};
use crate::persistence::record::{
    CustomObjectPayload, DebugIdPayload, DistillMode, DistillPayload, RecordType,
    SectionGeneration, SectionTombstonePayload, TombstonePayload, TreeMetadataPayload,
    TurnCouplingPayload,
};
use crate::persistence::streams::{StreamDecl, StreamId, TurnDecl};
use crate::persistence::tombstone_expiry::Expired;
use crate::persistence::walker::WalkEntry;
use crate::projection::scan_stats::GroupScan;
use crate::projection::{
    decode_events, CorruptTurnPolicy, GroupId, HitsConfig, LayerId, OwnedSection, OwnedSections,
    ProjectionEvent, ProjectionTarget, SectionId, SlotBoundary, Standing, SystemItem,
    TimelineAllocator, TimelineId, TurnIndex, TurnKey, Unpooled, WorkingSetMembers,
    WorkingSetShare,
};
use crate::provenance::chunk_exclusion::records::{from_json, group_of_key, CutRecord};
use crate::provenance::chunk_exclusion::ChunkExclusion;
use crate::provenance::SigWindow;
use crate::scheduler::profile;
use crate::summary_tree::exchange::Couplings;
use crate::summary_tree::exchange::{continued_exchange, exchanges, over_normals};
use crate::summary_tree::{
    select_dense, Node, NodeId, RecencyConfig, SelectionDiagnostics, SelectionOrigin, SummaryTree,
    TurnKind, MERGE_FANOUT,
};
use crate::token_buffer::TokenBuffer;
use crate::turn_layout::{phase_span_of, ThinkingLength, TurnLayout, TurnSegment};
use crate::working_set::{Candidate, PoolSize, TypeShares, WorkingSet};
use crate::ConversationError;
use calibration_record::{standing, CalRecord, CalRecords, CalSnapshot};
use deleted_set::DeletedSet;
use ended_sections::EndedSections;
pub use file_gallery::FileGallery;
use file_gallery::{entry_mut, GalleryMemo};
use label_records::label_records;
pub use lock::{SubstrateLock, SubstrateReadGuard};
use pcoll::{arc_mut, ord_arc_mut, PMap, POrdMap, POrdSet, PSet, PVec};
pub use pending::PendingGauge;
use pending::{PendingChild, PendingChildren};
pub use residence::{HotKv, Residence, ResidenceGauge, ResidenceSlab};
use residence::{COLD_PENDING, EVICT_WHEN_COLD, NO_COLD_PERSIST, PENDING_QUANTIZE, SPLICE_SOURCE};
use stamps::{take, take_keyed};
pub use stamps::{StreamIds, TimelineIds};
use stream_index::{Declared, StreamIndex};
pub use stream_index::{StreamKey, TurnStream};
use tag_index::TagIndex;
pub use tombstones::TurnMark;
pub use turn_placement::RestoredTurn;

// ── Substrate ─────────────────────────────────────────────────────────────────

/// What the belief scans read from a turn's projection-events blob, decoded
/// from its JSON once: the seams of its self-referencing sub-windows, sorted
/// and deduped, and per section collection the section its latest projection
/// that chose one in that collection selected.
#[derive(Debug, Default)]
pub struct EventsDigest {
    seams: Arc<Vec<usize>>,
    selected: HashMap<String, String>,
}

/// The digest of a turn's projection events, in their recorded order.
fn digest_events(events: &[ProjectionEvent]) -> EventsDigest {
    let mut seams: Vec<usize> = events
        .iter()
        .filter(|e| e.self_reference)
        .map(|e| e.start_token as usize)
        .collect();
    seams.sort_unstable();
    seams.dedup();
    let mut selected = HashMap::new();
    for event in events {
        for item in &event.selection.system {
            if let SystemItem::Collection { name, sections, .. } = item {
                if let Some(s) = sections.iter().find(|s| s.selected) {
                    selected.insert(name.clone(), s.name.clone());
                }
            }
        }
    }
    EventsDigest {
        seams: Arc::new(seams),
        selected,
    }
}

/// Per-session turn / section store and the single source of truth for KV
/// residence across hot (VRAM), warm (RAM), and cold (disk) tiers.
///
/// The substrate is the workspace-shared directory of *what* turns exist —
/// timelines, indices, role/text/tokens/sigs — plus the per-turn
/// [`Residence`] that records *where* each turn's KV bytes live right now.
/// Sections are held in their own table.
///
/// # Versions and residences
///
/// A `Substrate` value is one **version**: every field but the residence slab
/// is a persistent collection, so cloning one is O(1) and an edit copies only
/// what it changes. [`SubstrateLock`] publishes versions — a reader holds one
/// for as long as it reads, a writer edits a clone of the latest and swaps it
/// in (`docs/substrate_lock_free.md` §3).
///
/// The residences are not versioned. [`ResidenceSlab`] is shared by every
/// version; each [`TurnEntryData`] and [`SectionEntryData`] carries the
/// [`ResidenceIndex`] its KV lives at, allocated before the commit that names
/// it and never moved, and every tier transition is an atomic step on that
/// residence alone (§5).
#[derive(Debug, Default, Clone)]
pub struct Substrate {
    /// Every residence's KV across hot, warm and cold — shared by every
    /// version of the substrate.
    residences: Arc<ResidenceSlab>,
    /// The trained window of the engine this substrate is attached to, when
    /// its attention selects: the slot a dialogue's fill holds
    /// (`docs/hit_driven_projection.md` §4). Set by the engine at attach.
    trained_window: Option<usize>,

    /// Per-timeline directory — projection target, sidebar label,
    /// conv_id, and per-turn metadata (token counts, sig entries,
    /// role, text, token ids). Does **not** hold KV bytes; those live
    /// in [`Self::residences`] addressed by [`TurnEntryData::residence`].
    timelines: PMap<TimelineId, Arc<TimelineEntry>>,
    /// The creation rank given to a conversation named by a record that
    /// carries no identity — one written before identities existed — counted
    /// in the order the walk meets them, which for such a log is the order they
    /// were written. A conversation named by a stamped record ranks by that
    /// record's clock reading instead ([`TimelineEntry::order`]), which every
    /// member reads the same way and which sorts after every counted rank.
    conv_order_counter: u64,
    /// Monotonic counter stamped onto a [`TimelineEntry::active`] each time the
    /// conversation is used ([`Self::touch_conversation`]). Unlike `order`, the
    /// rank is persisted in the `ConvState` record, so replay restores it
    /// exactly whatever order compaction re-emits those records in; replay
    /// raises the counter to the highest rank it sees.
    conv_active_counter: u64,

    /// Inverse index: every timeline registered against a given group.
    /// Maintained in lockstep with [`Self::timelines`].
    timelines_by_group: PMap<GroupId, PVec<TimelineId>>,

    /// Per-timeline KV-compression override (set from the conversation's
    /// `SequenceConfig` at creation). Copied onto each turn residence at
    /// alloc time so the persistence thread can quantize different
    /// conversations at different levels. Absent ⇒ engine-wide turn policy.
    timeline_compression: PMap<TimelineId, ConvCompression>,

    /// Pinned section entries (system prompts, tool catalogs). Sections
    /// do not pass through LRU eviction — once ingested they stay hot
    /// for the session.
    sections: PMap<SectionId, SectionEntryData>,

    /// Running per-timeline sum of turn token counts — the O(1) corpus-size
    /// counter behind `total_token_count`. Maintained on turn append/extend
    /// (turns are append-only, so this only grows until `reset`), keeping the
    /// "materialized / N tokens" denominator off the O(corpus) hot path.
    timeline_token_totals: PMap<TimelineId, usize>,
    /// Running global sum of ingested section token counts (the shared
    /// workspace corpus). Maintained on section install (overwrite-aware).
    section_token_total: usize,

    /// Per dialogue, the already-ingested content it carries ahead of its own
    /// turns — seeds, fast-path locks and provenance
    /// (`docs/zend_working_set.md`). In-memory only: the locks are rebuilt
    /// from the marks on the dialogue's own turns and the momentum from the
    /// belief scan.
    working_sets: PMap<TimelineId, Arc<WorkingSet>>,

    /// Each calibrated conversation's calibrations, by the timeline each
    /// names under [`Self::CALIBRATES_KEY`] — kept as that metadata is applied,
    /// so a scan reads a unit's calibrations without walking every timeline.
    /// Tombstoned calibrations stay listed and are filtered at the read.
    calibrations_of: PMap<TimelineId, PVec<TimelineId>>,
    /// Every timeline by the content key it was read under
    /// ([`Self::CONTENT_KEY`]) — kept as that metadata is applied, so a
    /// parent named by key ([`Self::UNIT_PARENT`]) resolves without walking
    /// every timeline. Tombstoned and archived ones stay listed and are
    /// filtered at the read.
    units_by_key: PMap<String, PVec<TimelineId>>,
    /// Each calibrated conversation's calibration signatures as one unit, shared
    /// by every version ([`calibration_record`]).
    calibration_records: Arc<CalRecords>,
    /// Bumped whenever a stored signature is replaced or the streams cleared —
    /// the one change to a calibration turn its conversation's length does not
    /// show.
    sig_rewrites: u64,

    /// Groups whose conversations are offered to a projection only as its
    /// target's scope names them ([`Self::set_retrieval_scope`]). Ingested
    /// content keyed by what it shows — a file as one branch holds it — is
    /// only right for a conversation whose own files hold the same, so such a
    /// group offers nothing a target was not given. In-memory only, marked
    /// once at engine setup.
    scoped_groups: PSet<GroupId>,

    /// Which share of a dialogue's working-set budget each working-set group's
    /// conversations draw on — the schema's `selection: { kind: working_set,
    /// share }`. A conversation in no such group is never a working-set
    /// member. In-memory only, set once at engine setup.
    working_set_shares: PMap<GroupId, WorkingSetShare>,

    /// The types of document a working set caps, and the metadata key of a
    /// conversation's subject they are matched against
    /// (`docs/zend_working_set.md` §4.10). In-memory only, set once at engine
    /// setup; until then every candidate is of no type.
    working_set_types: Option<Arc<(TypeShares, String)>>,

    /// How large each dialogue's pool is on the card this substrate serves
    /// (`docs/hit_driven_projection.md` §7.2 step 1). Set by the scheduler from
    /// the card; the smallest card's until then.
    pool_size: PoolSize,

    /// Each drawing group and the group whose conversations it draws from —
    /// the schema's `draws: { from }` (`docs/guidance_layer.md` §1). A drawing
    /// group holds no conversation of its own: its candidates are its source's,
    /// as its retrieval scope names them, and a unit its scope names is placed
    /// through it rather than through the source. In-memory only, set once at
    /// engine setup.
    drawing_groups: PMap<GroupId, GroupId>,

    /// Per target, per scoped group, the conversations that group may offer
    /// it ([`Self::scoped_timelines_for_group`]). In-memory only and derived:
    /// the embedder sets it before each turn from the target's own files.
    retrieval_scopes: PMap<TimelineId, PMap<GroupId, Arc<HashSet<TimelineId>>>>,

    /// A digest of each scope in [`Self::retrieval_scopes`], taken when it is
    /// set: two targets whose scopes of a group name the same conversations
    /// share its assembled scan, and their probes ride one launch
    /// ([`Self::scope_digest`]). Each target's scope is its own set, so the
    /// sets themselves cannot be told equal without walking them.
    scope_digests: PMap<TimelineId, PMap<GroupId, u64>>,

    /// Per-stream in-RAM index of where each chunk / tokens record /
    /// committed-through watermark sits on disk, kept per writer and read
    /// through the logical view [`StreamIndex`] gives (§6.6).
    /// Built by replaying the redo log on startup (and updated on
    /// every fresh append).  Cold-load and seal-time persistence read
    /// this directly; the manifest holds only the workspace
    /// singletons.  It only ever sits in memory — reload rebuilds it
    /// from record headers.
    streams: StreamIndex,

    /// The streams of `streams` whose declaration is a turn: those in a gather
    /// scope — carrying a tag that is not a working-set mark — by tag, and the
    /// dialogue — carrying none — as a set, kept in step with every declaration
    /// installed and every stream removed.
    ///
    /// A tagged collection's gallery is drawn from the turns carrying its tags
    /// alone: the tool catalog's few thousand beside a substrate of every turn
    /// and chunk stream ingest has written. Walking all of `streams` to find
    /// them was ~115 ms, and walking every gather scope's turns, ~75 ms, both
    /// paid on the first reprojection after every seal anywhere in the
    /// workspace. An untagged collection's is drawn from the dialogue alone,
    /// and walking every turn to find it was ~440 ms of each seal scan. See
    /// [`TagIndex`].
    tagged_turns: TagIndex,

    /// Bumped whenever anything a collection's belief gallery is assembled from
    /// changes — a stream declared, a signature or projection-event blob
    /// written, a turn or timeline tombstoned, the streams cleared. The gallery
    /// walks every stream to find its members; keyed on this, it is assembled
    /// once per change instead of on every reprojection (see
    /// [`Self::gallery_epoch`]).
    gallery_epoch: u64,

    /// Each belief group's content epoch (`group_epoch`): moved by every
    /// change to what the group's scan is assembled from — a turn of a
    /// conversation it reads, a calibration of one of its files, its scoping,
    /// a target's scope of it, its chunk exclusion, what it draws from, a
    /// conversation's metadata or archived flag. An assembled group scan is
    /// stamped with it, so a turn sealed in one conversation reassembles the
    /// groups that read it and no other.
    group_epochs: PMap<GroupId, u64>,

    /// Moved by every change that moves [`Self::gallery_epoch`] and names no
    /// timeline: every group's content epoch moves with it.
    all_groups_epoch: u64,

    /// The epoch every file gallery held on a timeline entry was built under
    /// (`file_gallery`): moved by a change some gallery reads that names no
    /// timeline, so each is built again when next read.
    gallery_memo_epoch: u64,

    /// Bumped whenever a turn or a timeline is tombstoned — written here or
    /// replayed from the log, a replicated record included. The scheduler's
    /// gallery sweep looks for dead turns only when it has moved
    /// (`docs/gallery_residency.md` §3); see [`Self::tombstone_epoch`].
    tombstone_epoch: u64,

    /// Reverse index: stable resume keys (`debug_id`) → `TimelineId`.
    /// Populated by [`Self::set_debug_id`] and the cold-load reader.
    /// Provides O(1) lookup for the test-harness `find_or_create`
    /// pattern.
    timeline_by_debug_id: PMap<String, TimelineId>,

    /// Records about a timeline that is not registered yet — its label, state,
    /// summary nodes, debug id and couplings — held with their identities until
    /// the turn declaration that registers it, then applied. Zend writes a
    /// conversation's `Label` before its first turn, and records from another
    /// member arrive in any order; see [`PendingChildren`].
    pending: PendingChildren,
    /// A declaration ranked below turns this member holds moved one of them
    /// to another index, so the timeline's indices changed under the hot path
    /// and the member must reload (§6.6). Read and cleared by
    /// [`Self::take_reload_due`].
    reload_due: bool,
    /// The record behind each snapshot stream's tail, retracted or not — a
    /// retraction is a record and outranks the snapshot it withdraws.
    snapshot_ids: PMap<StreamId, RecordIdentity>,
    /// The record behind each prompt branch's checkpoint.
    branch_checkpoint_ids: PMap<StreamId, RecordIdentity>,
    /// The record behind each custom object, by its key's hash.
    custom_object_ids: PMap<u64, RecordIdentity>,

    /// Timelines flagged by [`RecordType::Tombstone`] as logically
    /// deleted.  [`Self::active_timelines_for_group`] filters them
    /// out so projection never surfaces their turns; the compactor
    /// drops the on-disk records during the next compaction pass.
    /// Entries may name timelines that are not yet registered
    /// (walker can apply a tombstone before its target
    /// `StreamDecl::Turn`) — registration just observes them as
    /// already tombstoned, which is the correct behaviour.
    ///
    /// A timeline whose tombstone expired stays here: its entry in
    /// [`Self::deleted`] dominates exactly as the tombstone did.
    tombstoned_timelines: PSet<TimelineId>,
    /// The deleted set: every timeline whose tombstone expired, and the
    /// tombstones a complete compaction has run past (§6.8). Rebuilt from the
    /// `Deleted` record each walk; like the tombstones it extends, never
    /// cleared.
    deleted: DeletedSet,
    /// The live recurrent-state snapshot per snapshot stream
    /// (`snapshot_stream_id(timeline)` — the key is computed by the reader at
    /// resume, never inverted from the id). Location only: the payload stays
    /// on disk until a resume materializes it, exactly like `Tokens`.
    /// Last-writer-wins by walk/append order — the single tail.
    recurrent_snapshots: PMap<StreamId, RecordLoc>,
    /// The snapshot streams of every timeline tombstoned or distilled: their
    /// tail is gone, and a snapshot for one that arrives later — replicated
    /// late, or met after the marker in a walk — is refused rather than
    /// installed again (§6.5). The tombstone and distill markers that fill it
    /// outlive every rewrite, so like them it is never cleared.
    retired_snapshots: PSet<StreamId>,
    /// The live recurrent checkpoint per **prompt branch**
    /// (`branch_checkpoint_stream_id(content prefix)`). Same single-tail rule
    /// and same location-only storage as `recurrent_snapshots`, kept separate
    /// because the two are reclaimed by opposite rules and the compactor cannot
    /// tell them apart from a location.
    ///
    /// A conversation's snapshot is **durable state**: losing it loses history
    /// nothing can recompute, so it lives until its timeline is tombstoned or
    /// distilled. A branch checkpoint is a **cache**: it is a pure function of a
    /// prompt that is still on disk, so losing it costs one prefill. Nothing
    /// supersedes one when the prompt changes — the key is the old content —
    /// and no timeline tombstone names it, so without a bound they accumulate
    /// one orphan per prompt edit at ~63 MiB each.
    branch_checkpoints: PMap<StreamId, RecordLoc>,
    /// Timelines whose KV **no reader will ever need from disk**, so the
    /// persistence thread never writes it to cold. Residences are flagged
    /// [`residence::NO_COLD_PERSIST`]; see
    /// [`Self::mark_timeline_transient`] for the two users and why each
    /// qualifies. In-memory only — the flag is re-derived on load, never a
    /// redo-log marker.
    transient_timelines: PSet<TimelineId>,
    /// Timelines that keep their KV but are **never resumed**, so they keep no
    /// recurrent snapshot — see [`Self::mark_timeline_unresumed`]. In-memory
    /// only: what made the mark durable is the empty snapshot written with it.
    unresumed_timelines: PSet<TimelineId>,
    /// Timelines whose turns are **splice sources** — code_read scope forks,
    /// whose sealed turns `adopt_turn` re-records onto the file timeline by
    /// reference. Residences are flagged [`residence::SPLICE_SOURCE`];
    /// see [`Self::mark_timeline_splice_source`]. In-memory only: a fork is
    /// tombstoned at its splice and never reloaded.
    splice_source_timelines: PSet<TimelineId>,
    /// Sections whose KV **no reader will ever need from disk** — the frame of
    /// a conversation opened for one job and thrown away. Their residences are
    /// flagged [`residence::NO_COLD_PERSIST`] at install, the
    /// scheduler's seal writes neither their stream declaration nor their
    /// tokens, and [`Self::retire_section`] removes them outright. See
    /// [`Self::mark_section_transient`]. In-memory only.
    transient_sections: PSet<SectionId>,
    /// Sections each conversation added at runtime, and the id counter they
    /// draw from. A timeline tombstone releases what its timeline owns.
    /// The registry is in-memory only: the owner re-submits after a restart and
    /// the stream is restored by its salted content address. What survives a
    /// restart is each stream's declaration, which names its owner — see
    /// [`Self::persisted_owned_streams`].
    owned_sections: OwnedSections,
    /// Section generations ended by a [`RecordType::SectionTombstone`] — the
    /// section counterpart of [`Self::tombstoned_timelines`]. Set by the
    /// reactive repair that confirms a section's persisted per-layer chunks
    /// disagree across layers (`candle-conversation/src/scheduler/mod.rs`'s
    /// `snapshot_sequence_per_layer` failure hook), by a rebuild, and by an
    /// owner releasing its section. [`Self::section_exists`] and
    /// `Conversation::section_stream_is_persisted` both read
    /// [`Self::is_section_tombstoned`] so a tombstoned section is neither
    /// "already present" nor "restorable" on the next ingest triage — the
    /// request falls through to a fresh prefill instead of reloading the
    /// corrupted chunks, in this same process as well as after a restart.
    ended_sections: EndedSections,
    /// Individual turns flagged dead by a **turn-scoped**
    /// [`RecordType::Tombstone`] (`turn_index = Some`), leaving the rest of their
    /// timeline live, each named as its writer wrote it ([`TurnMark`]).
    /// Written by the per-layer `drop_turn` corrupt-turn policy and by retention.
    /// Consumers: the reload restores a flagged turn as an EMPTY PLACEHOLDER
    /// (occupying its index so later turns keep their persisted stream ids),
    /// the belief galleries and group-default tag resolution skip it, and
    /// compaction sheds its bulk records while keeping its `StreamDecl` +
    /// re-emitting the tombstone.
    tombstoned_turns: PSet<TurnMark>,
    /// Per-layer policy for an unrecoverable turn on reload
    /// ([`CorruptTurnPolicy`]), resolved from the projection schema at engine
    /// setup. Absent → the default [`CorruptTurnPolicy::DropConversation`].
    layer_corrupt_turn: PMap<LayerId, CorruptTurnPolicy>,
    /// Timelines marked for distillation, each with the [`DistillMode`] degree
    /// its turns shed to at compaction. Same replay-order-independence as
    /// tombstones.
    distilled_timelines: PMap<TimelineId, DistillMode>,

    /// Generic keyed objects ([`RecordType::CustomObject`]), keyed by the stable
    /// hash of their string key so a re-write with the same key supersedes the
    /// old one (last-writer-wins). Held in RAM and re-emitted on compaction, the
    /// same survival the other resident metadata gets.
    custom_objects: PMap<u64, Arc<CustomObjectPayload>>,
    /// Every key [`Self::custom_objects`] holds, in key order, to its hash: a
    /// prefix is read as a range of it ([`Self::custom_objects_under`]), never
    /// by walking every object.
    custom_object_keys: POrdMap<String, u64>,
    /// Moves whenever a custom object is taken or dropped
    /// ([`Self::custom_objects_version`]).
    custom_objects_version: u64,

    /// Layers toggled out of projection assembly by name — a diagnostic kill
    /// switch flipped live and forgotten on restart (see
    /// [`crate::projection::layer_toggle`]). This substrate's own, so a toggle in
    /// one tenant never reaches another tenant's layer of the same name. Never
    /// persisted.
    disabled_layers: PSet<String>,

    /// Layers whose conversations are **append-only ingest trunks** (code_reading,
    /// repo_map) rather than interactive dialogue. A projection whose *target* is
    /// on such a layer is scored/selected **self-local**: every belief group is
    /// masked to the target's own timeline, not enumerated across all files (see
    /// `TargetedRead::group_turns` / `score_belief_groups`). This keeps an ingest
    /// scope-summary grounded in its own scope + system prompt — the multi-timeline
    /// belief scan is a *dialogue-retrieval* feature and has no place generating a
    /// scope's own summary, where cross-file turns derail it off-topic/off-language.
    /// Transient (rebuilt from the ingest setup each session), never redo-logged.
    /// Marked via [`Self::mark_layer_append_only`]; NOT keyed on `summarize`, which
    /// `--disable-summariser` also clears on dialogue.
    append_only_layers: PSet<LayerId>,

    /// What a `forked_from` of [`Self::SEED_PARENT`] names on this member: the
    /// priming chain's end, as this boot built it. Member-local and transient
    /// like [`Self::append_only_layers`] — set by the daemon once the chain is
    /// primed ([`Self::set_seed`]), never redo-logged.
    seed: Option<TimelineId>,

    /// Groups of LOWER layers an append-only layer's projections still score
    /// and select across conversations — the layers it compounds on. An ingest
    /// unit that analyses a file draws the analyses below it of OTHER files by
    /// provenance, while its own layer stays self-local: a sibling unit of the
    /// same layer is never retrieved into it. A layer absent here is wholly
    /// self-local, as every append-only layer was before. Transient, set from
    /// the ingest setup each session ([`Self::set_ingest_sees`]).
    ingest_sees: PMap<LayerId, Arc<HashSet<GroupId>>>,

    /// Per belief group, the gallery chunks cut from its live scans — those
    /// that win votes on every file's questions
    /// (`crate::provenance::chunk_exclusion`). Installed from the group's
    /// replicated cut record as it is applied ([`Self::apply_custom_object`]),
    /// on replay and replication alike.
    chunk_exclusion: PMap<GroupId, Arc<ChunkExclusion>>,
}

// ── Tier residence ────────────────────────────────────────────────────────────

/// How many persistence passes a hot residence may go unattended before the
/// idle demote sheds it.
///
/// **An epoch is one persistence pass**, not one wave and not one projection
/// publish. The pass wakes on every seal/flush trigger and otherwise on its own
/// 5 s tick, so the unit tracks real work with a wall-clock floor: at minimum
/// this is ~80 s of dormancy, and it shortens under load — which is the correct
/// direction, since load is exactly when the VRAM is wanted back.
///
/// This is **hygiene, not relief** — the ladder in
/// `docs/vram_governor_design.md` §8 still handles pressure — so the two failure
/// directions are not symmetric: too generous merely reclaims later, while too
/// aggressive sheds a conversation's KV between its own turns and charges a
/// warm→hot lift on every one of them.
///
/// Sized against the tick, not guessed: an earlier value of 256 was chosen
/// reasoning about waves while the clock actually ran on publishes, which made
/// the window unreachable and the demote silently inert.
pub const IDLE_DEMOTE_GRACE_EPOCHS: u64 = 16;

/// How many of a request's newest rounds keep their reasoning in its next
/// projection ([`Substrate::reasoning_turns`]).
///
/// The model's template keeps every round's reasoning since the user's message;
/// a cap keeps a long tool loop from carrying all of it. Measured 2026-11-18
/// (conversation `yenbot-employee-lifecycle/anonymous-demo-mode`): with one
/// round kept, a 27-round loop re-derived its plan from scratch every round —
/// 300 to 6,000 tokens of thinking each — and saved the same note six times.
/// Eight rounds hold a plan across a typical read-then-edit stretch at about
/// 2K tokens of thinking a round, ~16K in all, inside the dialogue's share.
pub const REASONING_ROUNDS: usize = 8;

/// [`Substrate::scope_digest`] of a target with no scope of a scoped group.
pub const EMPTY_SCOPE_DIGEST: u64 = 0;

/// A digest of the conversations `allowed` names, order-free: the same set
/// gives the same digest whichever target it was built for. An empty set is
/// [`EMPTY_SCOPE_DIGEST`], as no scope at all is: both offer nothing.
fn scope_digest_of(allowed: &HashSet<TimelineId>) -> u64 {
    if allowed.is_empty() {
        return EMPTY_SCOPE_DIGEST;
    }
    let mut ids: Vec<u64> = allowed.iter().map(|t| t.raw()).collect();
    ids.sort_unstable();
    let mut h = DefaultHasher::new();
    ids.hash(&mut h);
    h.finish().max(1)
}

/// The scheduler's publisher id for [`ResidenceSlab::publish_pins`].
const SCHEDULER_PINS: u64 = 0;

/// Epochs elapsed since `last_used`, saturating.
///
/// Pure and separate so the staleness rule is testable without a substrate, and
/// so the wrap case is stated once: the counter is `u64` at roughly one tick per
/// wave, so it does not wrap in any real run — but a `last_used` *ahead* of
/// `epoch` (a stamp published concurrently with a scan reading an older load)
/// must read as "just used", never as a huge age that would evict live KV.
pub fn idle_for(last_used: u64, epoch: u64) -> u64 {
    epoch.saturating_sub(last_used)
}

/// The custom object that declares `timeline` a file store
/// ([`Substrate::is_file_store`]).
pub fn file_store_key(timeline: TimelineId) -> String {
    format!("file-store/{}", timeline.raw())
}

/// Index into [`Substrate::residence`]. Strongly typed so it can't be
/// confused with [`TurnIndex`] or any other `usize`.
///
/// A residence slot is allocated when a turn or section is first
/// appended and stays at that index for the lifetime of the substrate
/// (until [`Substrate::reset`]). The hot- and warm-tier LRU lists store
/// `ResidenceIndex` values as their node identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidenceIndex(pub usize);

/// Per-conversation KV-compression override, resolved from the owning
/// timeline's [`SequenceConfig`] at residence-allocation time and read by
/// the persistence thread's hot→warm quantize pass. `None` on a residence
/// means "use the engine-wide turn policy"; `Some` applies the overrides
/// below. Used to compress utility layers (e.g. `code_reading`) harder than
/// live dialogue, or to pin dialogue turns to a fixed near-lossless format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvCompression {
    /// Skip the hot→warm quantize pass entirely for this conversation's turns,
    /// persisting their K/V in the native R16/F16 form (no adaptive
    /// compression). Used to capture lossless tool-call exemplars for the
    /// provenance work. Overrides every other field here when set.
    pub lossless: bool,
    /// Adaptive compression level override. `Some` replaces the engine-wide
    /// turn level; `None` keeps it (used when only the forced formats below
    /// are set).
    pub level: Option<u8>,
    /// Drop the global K-format override (Q4_KS) so K is adaptively quantized
    /// per-block like V.
    pub disable_k_override: bool,
    /// Force every K block of this conversation's turns to a single uniform
    /// quant format, bypassing adaptive per-block selection (and the global
    /// Q4_KS K override). `None` keeps the engine-wide K behaviour.
    pub force_k: Option<QuantFormat>,
    /// V counterpart to [`Self::force_k`].
    pub force_v: Option<QuantFormat>,
}

/// One layer's KV sequence as it lives in the redo log. Mirrors
/// [`SealedSequence`] (the VRAM/RAM form) but each chunk carries a
/// redo-log reference instead of GPU arena GIDs.
///
/// Promoting a `StoredSequence` back into VRAM goes through the cold-
/// load path ([`crate::persistence::transfer::load_stream`]): read each
/// chunk's bytes at its `log_offset`, decode the [`ChunkPayload`],
/// allocate arena slots, scatter the bytes, and produce a fresh
/// [`SealedSequence`].
///
/// [`ChunkPayload`]: crate::persistence::record::ChunkPayload
#[derive(Debug, Clone)]
pub struct StoredSequence {
    /// Ordered per-chunk references into the redo log, in token
    /// order. Length equals the number of sealed chunks this layer
    /// contributed when the turn was persisted.
    pub chunks: Vec<StoredChunk>,
    /// Sum of `chunks[i].token_count`. Cached so the cold-load path
    /// can size its pinned-staging buffer without re-walking chunks.
    pub token_count: usize,
}

/// One chunk's pointer into the redo log. The redo log holds a
/// [`crate::persistence::record::ChunkPayload`] encoded inside a
/// record of length `record_len` starting at `log_offset`; decoding
/// it yields the quantization metadata + arena bytes needed to
/// rebuild a [`candle_nn::kv_cache::chunked::SealedChunk`] in VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredChunk {
    /// Byte offset of the chunk's full record in the redo log.
    pub log_offset: u64,
    /// Length of the chunk's record on disk (read with
    /// `read_record_at(log_offset)` and then decode the payload).
    pub record_len: u64,
    /// Valid token count within this chunk (≤ `CHUNK_SIZE`).
    pub token_count: u16,
}

// ── Promotion plan (elevate_to_hot work shape) ────────────────────────────────

/// What kind of substrate entity owns a residence — needed because
/// section installs skip the hot LRU (pinned) while turn installs
/// push to the LRU front.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromotionItemKind {
    Section(SectionId),
    Turn(TurnKey),
}

/// One item already in VRAM-resident hot tier — no work needed.
/// Kept in [`PromotionPlan::already_hot`] so the report can count it.
///
/// Used to plan: "the fast path of [`super::persistence::elevate::elevate_to_hot`]
/// when most callers' items are already cached."
pub type AlreadyHotEntry = PromotionItemKind;

/// One item that needs warm → hot promotion. The warm
/// `Vec<SealedSequence>` is the residence's own, shared at snapshot time —
/// one refcount, never a per-chunk copy — so the caller can hand it to
/// the warm lift ([`candle_nn::kv_cache::lift_warm_to_gpu`])
/// without holding the substrate read lock, and a plan built only to price a
/// lift costs nothing per chunk.
#[derive(Debug)]
pub struct WarmToHotEntry {
    pub kind: PromotionItemKind,
    pub residence: ResidenceIndex,
    pub warm: Arc<Vec<SealedSequence>>,
}

/// One item that needs cold (disk) → hot promotion. The cold
/// `Vec<StoredSequence>` is shared for the same reason as
/// [`WarmToHotEntry::warm`]; the [`StreamId`] is the redo-log handle
/// the caller passes to `recover_turn_chunks` / `load_to_hot`.
#[derive(Debug, Clone)]
pub struct ColdToHotEntry {
    pub kind: PromotionItemKind,
    pub residence: ResidenceIndex,
    pub cold: Arc<Vec<StoredSequence>>,
    pub stream_id: StreamId,
}

/// Classification of a batch of items by what `elevate_to_hot` must
/// do for each. Produced by [`Substrate::snapshot_promotion_state`]
/// under a single read lock; consumed by the elevation orchestrator
/// (one round of CUDA work + one substrate write lock).
#[derive(Debug, Default)]
pub struct PromotionPlan {
    pub already_hot: Vec<AlreadyHotEntry>,
    pub warm_to_hot: Vec<WarmToHotEntry>,
    pub cold_to_hot: Vec<ColdToHotEntry>,
    /// Items the substrate has no record of at all — neither a turn
    /// entry in `timelines.<tl>.turns` nor a section entry in
    /// `sections`.  This is the genuinely-problematic bucket: the
    /// projection plan referenced an id that the substrate cannot
    /// resolve, which is either a stale projection or a substrate
    /// invariant violation.  `elevate_to_hot` logs a `WARN` per
    /// item in this bucket.
    pub missing: Vec<PromotionItemKind>,
    /// Items the substrate DOES have a record of, but whose
    /// residence has no hot/warm/cold tier installed — so there's
    /// nothing to elevate.  The two cases that produce this:
    ///
    ///  1. Ghost summary turns appended by the
    ///     `substrate-summariser` thread via
    ///     `record_summary_turn` → `append_with_blocks(0..0)`.
    ///     These declare a node in the per-timeline summary tree
    ///     and carry no K/V, by design.
    ///  2. A turn whose tier state was rolled back (a
    ///     `clear_turn_sealed` call between projection and elevate).
    ///
    /// `elevate_to_hot` silently skips these items.  Distinct from
    /// `missing` so the WARN doesn't fire for legitimately-tier-less
    /// turns.
    pub tier_less: Vec<PromotionItemKind>,
}

/// A full rehydration from cold (disk-only) state.
///
/// The elevation orchestrator produces one of these per turn that was
/// pulled out of the redo log: the cold load writes the bytes into VRAM,
/// and the residence lands **hot, backed by its cold record** — the state
/// every freshly sealed turn is in. The persistence thread's hot→warm pass
/// gives it a warm copy the way it gives every hot-only residence one, once
/// no live working set pins it.
///
/// Recalled dual-tier, the warm copy was a second crossing of the link on
/// the projection's critical path — the bytes gathered back off the GPU that
/// the cold load had just read into host memory — and it cost more than the
/// load itself: 227 ms against 136 ms for a 320 MB recall, every
/// reprojection of a dialogue opened after a restart.
///
/// Conceptually a *recall*: reaching back into durable storage and
/// reactivating a turn into the working set. Pairs with [`WarmLift`] (the
/// faster RAM-cached hop) and is consumed alongside it by
/// [`Substrate::install_promoted`].
#[derive(Debug)]
pub struct ColdRecall {
    pub kind: PromotionItemKind,
    pub residence: ResidenceIndex,
    pub hot: Vec<SealedSequence>,
}

/// A fast promotion from warm (RAM-cached) state into hot.
///
/// The elevation orchestrator produces one of these per turn already
/// resident in the warm tier: the warm lift (`lift_warm_to_gpu`)
/// scatters the warm payload into fresh VRAM arena chunks. Only hot
/// is installed — the warm copy stays in place, so the residence stays
/// dual-tier.
///
/// Conceptually a *lift*: a short PCIe hop from the RAM cache to
/// VRAM, paying nothing on NVMe. Pairs with [`ColdRecall`].
#[derive(Debug)]
pub struct WarmLift {
    pub kind: PromotionItemKind,
    pub residence: ResidenceIndex,
    pub hot: Vec<SealedSequence>,
}

/// Per-call summary from [`Substrate::evict_hot_except`].
/// `count` is the number of residences evicted, `bytes` is the total
/// VRAM freed (sum of each residence's cached `byte_size`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EvictionReport {
    pub count: usize,
    pub bytes: u64,
}

/// Hot VRAM bytes by why [`Substrate::evict_hot_to_free`] would or would not
/// take them. A refusal for room that freed nothing says which of these held
/// the ground; without it "live" is one figure with no owner.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct HotCensus {
    /// Sections: pinned, never on the hot LRU.
    pub sections: u64,
    /// In a live slot's working set (the keep list).
    pub attended: u64,
    /// In some dialogue's pinned working set.
    pub pinned: u64,
    /// No warm copy yet, so dropping the hot one would lose it.
    pub unwarmed: u64,
    /// A splice source.
    pub splice: u64,
    /// What an eviction may take.
    pub evictable: u64,
}

impl std::ops::AddAssign for HotCensus {
    fn add_assign(&mut self, o: Self) {
        self.sections += o.sections;
        self.attended += o.attended;
        self.pinned += o.pinned;
        self.unwarmed += o.unwarmed;
        self.splice += o.splice;
        self.evictable += o.evictable;
    }
}

/// Which tiers a residence currently occupies. Returned by
/// [`Substrate::turn_tier_state`] / [`Substrate::section_tier_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierState {
    pub hot: bool,
    pub warm: bool,
    pub cold: bool,
}

/// Per-call summary from [`Substrate::purge_warm_until_headroom`].
/// `count` warm-tier residences had their warm copy dropped; `bytes`
/// is the sum of their `byte_size` (RAM freed).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PurgeReport {
    pub count: usize,
    pub bytes: u64,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

mod apply_records;
mod entries;
mod eviction;
mod live_state;
mod loading;
mod marking_dead;
mod metadata;
mod registry;
mod resolver;
mod scored;
mod sections;
#[cfg(test)]
mod tests_cold_refs;
#[cfg(test)]
mod tests_copies_and_distill;
#[cfg(test)]
mod tests_eviction;
#[cfg(test)]
mod tests_labels_tombstones;
#[cfg(test)]
mod tests_listing;
#[cfg(test)]
mod tests_timeline_meta;
#[cfg(test)]
mod tests_transient_owned;
#[cfg(test)]
mod tests_working_set;
mod tiers;
mod transient_owned;
mod turn_access;

pub use entries::*;
pub use resolver::*;
pub use scored::*;
