//! Redo-log record codec: the on-disk framing and the typed payloads of every record.
//!
//! Each record is a JSON header line, then its payload, then zero padding to a 4 KB
//! sector. Older readers skip record types they do not know and ignore fields they
//! do not model. A torn write shows up as a header that fails to parse, a payload
//! that fails its CRC, or a file that ends early. `RecordType` lists the wire tags,
//! and it marks which records never leave the member that wrote them, and which
//! ones only free space.
//!
//! - `framing`: sector framing of a record.
//! - `header`: the typed record header.
//! - `decode`: header and payload parsing.
//! - `crc`: the CRC-32 primitives.
//! - `bytes`: the little-endian reader and writer.
//! - `chunk`: the KV chunk payload.
//! - `snapshot`: the conversation snapshot payload.
//! - `branch`: the prompt branch checkpoint payload.
//! - `metadata`: tree metadata, debug id and custom object payloads.
//! - `npc`: the character record.
//! - `tombstone`: timeline and section tombstone payloads.
//! - `distill`: distillation and turn coupling payloads.
//! - `tests_chunk`: chunk golden and chunk payload tests.
//! - `tests_framing`: framing, compatibility and corruption tests.
//! - `tests_metadata`: coupling and metadata payload encodings.
//! - `tests_npc`: character record byte layout tests.
//! - `tests_primitives`: CRC and byte primitive tests.
//! - `tests_snapshot`: snapshot and checkpoint payload tests.
//! - `tests_tombstone`: tombstone payload byte tests.

use std::collections::BTreeMap;

use candle_transformers::models::delta_net::{host_rows, ExportedLayerState};

use serde::{Deserialize, Serialize};
use serde_json::{from_value, Value};
use sha2::{Digest, Sha256};

use super::content_hash::{ContentHash, ContentHasher};
use super::identity::RecordIdentity;
use super::payload_parts::PayloadParts;
use super::{PersistenceError, Result};

/// The redo-log record types.
///
/// `Unknown` is the sentinel for tags this version doesn't recognise —
/// produced by `#[serde(other)]` on the deserialize side. The walker
/// skips records that decode to `Unknown` instead of erroring, which is
/// the forward-compatibility property the on-disk format buys us.
///
/// `Unknown` must never be written: [`encode_record`] panics if it sees
/// one (it's a programming error to construct a header with `Unknown`).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordType {
    /// The model the substrate is bound to — a `ModelKey`, or on a substrate
    /// written before keys the spec text that preceded it
    /// (`crate::models::identity::RecordedModel`). A workspace singleton.
    ///
    /// The wire name is the one the spec record was written under, so an older
    /// log's record decodes as this type and is checked, not skipped.
    #[serde(rename = "model_spec")]
    ModelKey = 1,
    Template = 2,
    StreamDecl = 3,
    Chunk = 4,
    Tokens = 5,
    Commit = 7,
    // Tag 8 is retired (it was the manifest checkpoint snapshot —
    // recovery is a pure walk now, and old logs' checkpoint records
    // decode as `Unknown` and are skipped by the walker).
    /// The model's `tokenizer.json` digest — a workspace singleton.
    Tokenizer = 9,
    /// Per-timeline conversation display label + conv_id (sidebar
    /// title). Last-writer-wins on replay.
    Label = 10,
    /// Per-timeline lifecycle state (archived flag, future flags…).
    /// Last-writer-wins on replay.
    ConvState = 11,
    /// Summary-tree node metadata for one turn — kind / children /
    /// height / dirty.  JSON payload — see
    /// [`TreeMetadataPayload`].  Reload constructs the per-timeline
    /// summary tree by replaying these records in order (the latest
    /// record per `(timeline, turn_index)` wins).  Last-writer-wins.
    TreeMetadata = 12,
    /// Per-timeline substrate-side resume key (`debug_id`).  JSON
    /// payload `{ debug_id: String }`.  Last-writer-wins.  Allows
    /// `find_or_create(debug_id)` to recover a workspace timeline
    /// after restart.
    DebugId = 13,
    /// Per-timeline tombstone — marks every turn, section, and
    /// stream record bound to the named timeline as logically
    /// deleted.  JSON payload [`TombstonePayload`].  Replay applies
    /// the tombstone so the old turns vanish from the substrate's
    /// view; the compactor physically drops the matching records
    /// on the next compaction pass.
    Tombstone = 14,
    /// Per-turn projection-event timeline — the materialized-context
    /// composition + decode throughput the GUI draws as timeline dots.
    /// JSON payload: a `Vec<ProjectionEvent>` for one turn, keyed by the
    /// turn's `stream_id`. Last-writer-wins on replay.
    ProjectionEvents = 15,
    /// Per-timeline distillation marker — the timeline's turns keep their
    /// `StreamDecl` + `WideQSig` (the belief gallery reads only the sig) but shed
    /// their content (`Tokens` + KV `Chunk`s) on the next compaction pass. Used to
    /// collapse the calibration corpus to sig-only. JSON payload
    /// [`DistillPayload`]. Idempotent — duplicate markers replay identically.
    ///
    /// The marker **survives** the compaction that acts on it, re-emitted with
    /// its mode from the substrate's live state ([`Survival::Resident`]). It has
    /// to: the marker is what exempts a distilled turn from the `MissingKv`
    /// integrity verdict, and a distilled turn has legitimately shed its content
    /// — so a consumed marker would leave the next reload condemning the very
    /// corpus the distillation created. See `docs/tool_provenance_distillation.md`.
    ///
    /// [`Survival::Resident`]: super::survival::Survival::Resident
    Distilled = 16,
    /// A turn's wide per-token `sign(Q)` window (all heads, all layers) — the decode→decode
    /// (`Q·Q`) consensus substrate. Opaque payload encoded by `provenance::wide_sig`, keyed
    /// by the turn's stream id, last-writer-wins (each (re)projection overwrites the window).
    WideQSig = 17,
    /// A batch of header digests for the records appended since the
    /// previous `HeaderIndex`, plus a link to that previous index — the
    /// backward chain recovery follows instead of probing every record
    /// header (§5.6). Binary payload — see
    /// [`header_index`](super::header_index). Derived data: the
    /// compactor drops every copy and the writer regenerates the chain
    /// in the new file.
    HeaderIndex = 18,
    /// Couples a turn to the tool response that follows it — the two halves of
    /// one tool round-trip. JSON payload [`TurnCouplingPayload`]. Idempotent;
    /// replay collects them into a per-timeline set.
    ///
    /// Written by the caller (not the seal) in the one window where the fact is
    /// certain: after the tools have actually returned output, and *before* the
    /// response turn is submitted. So a coupling exists iff the round-trip really
    /// happened — capture mode and malformed calls emit none — and it is always
    /// durable before the turn it points at exists, which is what lets the
    /// summariser group an exchange without ever racing its own record.
    TurnCoupling = 19,
    /// A recurrent-state snapshot — the Gated DeltaNet matrices and conv tails
    /// of every recurrent layer (hybrid Qwen3.5/3.6/3.8 models;
    /// `docs/qwen35_qwen38_models.md` §5).
    ///
    /// **Two kinds, told apart by stream id, never by inspection.** A
    /// *conversation's* snapshot ([`SnapshotPayload`]) is taken at a turn seal
    /// and keyed by `snapshot_stream_id(timeline)`. A *prompt branch's*
    /// checkpoint ([`BranchCheckpointPayload`]) is computed at build time for
    /// one selector assignment and keyed by `branch_checkpoint_stream_id(prefix)`.
    /// Everything below — the accounting supersede key, the recovery walk's
    /// location map, the compactor's carry-forward — reads `header.stream_id`
    /// and never the payload, so both kinds ride the identical single-tail
    /// machinery. A reader computes the stream id before asking, so it always
    /// knows which shape it will get; the payloads carry distinct leading magic
    /// so a future mistake is an error rather than a plausible state.
    ///
    /// **Single tail per conversation**: keyed in the *header* by a synthetic
    /// per-timeline stream id, so the newest snapshot supersedes every
    /// previous one mechanically — [`super::accounting`] credits the old copy
    /// as dead bytes and `snapshot_locs` keeps exactly the tail alive through
    /// segment maintenance. No explicit tombstone is written for supersede;
    /// the last-writer-wins accounting *is* the tombstone. Binary payload
    /// [`SnapshotPayload`]; the header CRC covers it like any metadata record.
    ///
    /// **An empty payload is a retraction** ([`RecordHeader::retracts_snapshot`]):
    /// the stream has no live snapshot from here on. Written when a seal's
    /// snapshot append fails, so the previous turn's snapshot — one turn
    /// behind the K/V that is about to land — cannot read as the tail.
    Snapshot = 20,
    /// A **prompt branch's** recurrent checkpoint — the state after a whole
    /// system prompt for one selector assignment, keyed by the branch's content
    /// prefix ([`BranchCheckpointPayload`]).
    ///
    /// Structurally identical to [`Self::Snapshot`] — same single-tail rule,
    /// same accounting, same relocation — and a separate type for one reason:
    /// **a branch checkpoint is a cache and a conversation snapshot is not.**
    /// Losing a snapshot loses history that cannot be recomputed. Losing a
    /// checkpoint costs one prefill, because the prompt that produced it is
    /// still on disk. Compaction has to treat them differently, and it reads
    /// the header rather than the payload, so the difference has to live in the
    /// type.
    ///
    /// That difference is load-bearing: checkpoints are keyed by content, so
    /// nothing supersedes one when the prompt changes and no timeline tombstone
    /// ever names it. Without a bound they accumulate one orphan per prompt
    /// edit, ~63 MiB each, carried forward verbatim by every compaction.
    BranchCheckpoint = 21,
    /// One NPC's durable identity and configuration — the `npcd` product's
    /// characters, stored here so a character outlives the process that made
    /// it. JSON payload [`NpcPayload`].
    ///
    /// **Single tail per character**, by the same rule as [`Self::Snapshot`]:
    /// the `npc_id` goes in the *header's* `stream_id`, so the newest record
    /// for an id supersedes every earlier one mechanically — the accounting
    /// credits the old copy as dead the moment the new one lands. No explicit
    /// tombstone is written for an edit; supersession is the tombstone.
    /// Deletion is a state transition (`state: "tombstoned"`), which is still
    /// just another superseding record.
    ///
    /// Header-keyed, so compaction carries the live tail forward verbatim from
    /// wherever it physically lives (`npc_locs` → a `Raw` item) rather than
    /// re-encoding it. It has to: unlike `Label` or `ConvState`, an NPC's
    /// payload is owned by the daemon's registry and is not in this process's
    /// RAM at all. See [`super::compaction`].
    Npc = 22,
    /// A turn's **QSA index page** — the compressed index rows covering exactly
    /// that turn's own tokens. Opaque payload, keyed by the turn's stream id,
    /// last-writer-wins, structurally a sibling of [`Self::WideQSig`].
    ///
    /// **Why this has to be on disk.** A projection borrows a sealed turn's K/V
    /// rather than recomputing it, and the index cannot be borrowed the same
    /// way: its keys come from hidden states the borrowing slot never computed.
    /// The page is what carries them across. Held only in the daemon's RAM, it
    /// survives exactly as long as the process — so after a restart every
    /// cold-loaded turn hands a slot its keys and no index of them, the slot's
    /// caches sit permanently short by that turn's width, and the QSA select
    /// refuses the moment the conversation grows past the identity threshold.
    /// Under the threshold nothing reads the index, so the loss is silent until
    /// it is fatal, and it is fatal only for conversations long enough to
    /// matter.
    TurnIndexPage = 23,
    /// A content-addressed section stream flagged logically deleted — the
    /// section counterpart of the whole-timeline [`Self::Tombstone`]. JSON
    /// payload [`SectionTombstonePayload`], keyed by the section's `StreamId`
    /// in the **header's** `stream_id` field (not re-encoded in the payload,
    /// the same convention [`Self::WideQSig`] and [`Self::TurnIndexPage`]
    /// use). Replay marks the stream dead; the section-ingest triage then
    /// reads it as neither present nor persisted, so the next request
    /// re-prefills the section instead of restoring the tombstoned chunks.
    ///
    /// Written when the reactive per-layer window-divergence repair confirms
    /// a section's own persisted chunks disagree across layers — a corrupted
    /// cold-tier section is never legitimately re-read, so nothing needs the
    /// old records once this lands. The compactor drops them on the next
    /// compaction pass; this marker survives it (it is what makes the drop
    /// permanent across reloads).
    SectionTombstone = 24,
    /// One change a conversation made to a repository's files — a child of
    /// the conversation's timeline. JSON payload
    /// [`super::vfs::VfsEventPayload`], whose `body` is its owner's and never
    /// read here. Keyed in the **header**: the timeline in `stream_id`, the
    /// event's sequence number in `chunk_index`, so the log opens without
    /// reading a single body.
    ///
    /// Dies two ways: explicitly, by a [`Self::VfsTombstone`] naming its
    /// sequence number, or implicitly, with its timeline — a tombstoned
    /// conversation's events are orphans that no rewrite carries. Relocated
    /// verbatim by location ([`super::vfs::VfsIndex`]). See
    /// `docs/zend_vfs_events.md`.
    VfsEvent = 25,
    /// Kills a set of a timeline's [`Self::VfsEvent`]s — every event of one
    /// path, up to the last. JSON payload [`super::vfs::VfsTombstonePayload`],
    /// keyed in the header like the events, by timeline and its own sequence
    /// number. Relocated while its timeline lives, so it stays after every
    /// copy of the events it killed.
    VfsTombstone = 26,
    /// A generic keyed object: a string `key`, a metadata bag, and a JSON blob.
    /// Last-writer-wins by `key` (a later record with the same key supersedes the
    /// older), keyed on the header's `stream_id` = a stable hash of the key. The
    /// substrate holds the live set in RAM and re-emits it on compaction, so it
    /// survives like the other resident metadata. See [`CustomObjectPayload`].
    CustomObject = 27,
    /// Bytes named by their SHA-256 — an image, say
    /// (`docs/replicated_substrate.md` §10.3). Payload [`BlobPayload`]: the
    /// hash, then the bytes. Keyed in the **header** by the hash's first 128
    /// bits — `stream_id` its first eight bytes, `chunk_index` the next eight —
    /// so the log opens without hashing a body, and the full hash is checked
    /// on every read. Content-addressed, so one hash is one value: a second
    /// record of a hash already held is a duplicate, dead on arrival. Never
    /// superseded and never resident; relocated verbatim by location
    /// ([`super::blob::BlobIndex`]).
    Blob = 29,
    /// What this member has applied of every other origin, and the records it
    /// deferred because it does not decode their type
    /// (`docs/replicated_substrate.md` §6.2, §6.5). Payload
    /// [`super::replica::ReplicaPayload`]. Local housekeeping — written at a
    /// commit in which it changed and carried by every rewrite, never
    /// replicated, with no identity.
    Replica = 28,
    /// The generations fixed for what this member wrote while cut off
    /// (`docs/replicated_substrate.md` §6.7). Payload
    /// [`super::rebase::RebaseMap`]. Local housekeeping — rewritten whole when
    /// a rejoin adds an entry and carried by every rewrite, never replicated,
    /// with no identity.
    Rebase = 30,
    /// A member's **deleted set** (`docs/replicated_substrate.md` §6.8): every
    /// timeline whose tombstone expired, shrunk to its entry, and the
    /// tombstones a complete compaction has run past. Payload
    /// [`super::deleted::DeletedPayload`]. Local housekeeping — one record,
    /// rewritten whole by each compaction and carried by maintenance, never
    /// replicated, with no identity.
    Deleted = 31,
    /// Catch-all for record-type tags this version doesn't recognise.
    /// Records that deserialize as `Unknown` are skipped by the walker.
    #[serde(other)]
    Unknown,
}

impl RecordType {
    /// The numeric wire tag used inside `HeaderIndex` digests (the
    /// enum's explicit discriminant). The JSON header keeps its
    /// snake_case string form; this compact form exists only for the
    /// fixed-width digest entries.
    pub fn tag(self) -> u8 {
        self as u8
    }

    /// Whether records of this type never leave the member that wrote them
    /// (`docs/replicated_substrate.md` §6.1): the header index, the model
    /// binding, the projection schema, the branch-checkpoint cache, and the
    /// member's own housekeeping.
    pub fn is_local_only(self) -> bool {
        matches!(
            self,
            RecordType::HeaderIndex
                | RecordType::ModelKey
                | RecordType::Tokenizer
                | RecordType::Template
                | RecordType::BranchCheckpoint
                | RecordType::Replica
                | RecordType::Rebase
                | RecordType::Deleted
        )
    }

    /// Whether a record of this type, applied, only makes other records dead:
    /// a tombstone of a timeline, of a section or of a file. The records it
    /// names are reclaimed by maintenance, so it is the one kind a member
    /// whose disk is full still applies (`docs/replicated_substrate.md` §7.7).
    pub fn frees_space(self) -> bool {
        matches!(
            self,
            RecordType::Tombstone | RecordType::SectionTombstone | RecordType::VfsTombstone
        )
    }

    /// The type a header names `name` — `"chunk"`, `"stream_decl"` — or
    /// [`RecordType::Unknown`] for a name this build does not know, by the
    /// header's own rule.
    pub fn from_name(name: &str) -> RecordType {
        from_value(Value::String(name.to_string())).unwrap_or(RecordType::Unknown)
    }

    /// Inverse of [`RecordType::tag`]. Unrecognised tags decode to
    /// [`RecordType::Unknown`] — the same forward-compatibility rule as
    /// the JSON header's `#[serde(other)]`.
    pub fn from_tag(tag: u8) -> RecordType {
        match tag {
            1 => RecordType::ModelKey,
            2 => RecordType::Template,
            3 => RecordType::StreamDecl,
            4 => RecordType::Chunk,
            5 => RecordType::Tokens,
            7 => RecordType::Commit,
            9 => RecordType::Tokenizer,
            10 => RecordType::Label,
            11 => RecordType::ConvState,
            12 => RecordType::TreeMetadata,
            13 => RecordType::DebugId,
            14 => RecordType::Tombstone,
            15 => RecordType::ProjectionEvents,
            16 => RecordType::Distilled,
            17 => RecordType::WideQSig,
            18 => RecordType::HeaderIndex,
            19 => RecordType::TurnCoupling,
            20 => RecordType::Snapshot,
            21 => RecordType::BranchCheckpoint,
            22 => RecordType::Npc,
            23 => RecordType::TurnIndexPage,
            24 => RecordType::SectionTombstone,
            25 => RecordType::VfsEvent,
            26 => RecordType::VfsTombstone,
            27 => RecordType::CustomObject,
            28 => RecordType::Replica,
            29 => RecordType::Blob,
            30 => RecordType::Rebase,
            31 => RecordType::Deleted,
            _ => RecordType::Unknown,
        }
    }
}

mod branch;
mod bytes;
mod chunk;
mod chunk_head;
mod crc;
mod decode;
mod distill;
mod framing;
mod header;
mod metadata;
mod npc;
mod snapshot;
#[cfg(test)]
mod tests_chunk;
#[cfg(test)]
mod tests_framing;
#[cfg(test)]
mod tests_metadata;
#[cfg(test)]
mod tests_npc;
#[cfg(test)]
mod tests_primitives;
#[cfg(test)]
mod tests_snapshot;
#[cfg(test)]
mod tests_tombstone;
mod tombstone;

pub use branch::*;
pub use bytes::*;
pub use chunk::*;
pub use chunk_head::*;
pub use crc::*;
pub use decode::*;
pub use distill::*;
pub use framing::*;
pub use header::*;
pub use metadata::*;
pub use npc::*;
pub use snapshot::*;
pub use tombstone::*;
