//! The substrate's queued redo-log writes (`docs/archived/kv_tier_migration.md`,
//! `docs/substrate_lock_free.md` §9.1).
//!
//! A write that does not need its result back is queued to the log's owner
//! ([`super::owner::PersistenceOwner::enqueue`]) and the caller goes on. Two
//! producers feed it:
//!
//! - **Stream-decl mirrors and the seal's metadata**, from the inference
//!   thread's seal path. The enqueuer applies each value in memory first and
//!   hands only the durable append here, so the inference thread never waits on
//!   the disk.
//! - **Warm→cold KV writes** (`WriteJob::KvCold`), from the persistence thread.
//!   The GPU gather stays there; the append and the `install_cold` bookkeeping
//!   run here, AFTER the bytes are appended, so a residence's cold copy only
//!   exists once the data is durable — `purge_warm` never drops the last copy
//!   of an un-written turn (the RAM-unload invariant).
//!
//! Every job is appended and then indexed in the substrate before the owner
//! takes the next one, so a record is indexed before its segment can seal.
//! Crash safety is unchanged: the reconstruct walker is order-independent and
//! per-turn fault-isolated (see `reconstruct_from_log`), so a lost-on-crash
//! write only ever yields an ABSENT turn, never corruption; the owner's
//! bounded queue caps that window.

use std::sync::Arc;

use super::identity::{RecordIdentity, Stamped};
use super::manifest::RecordLoc;
use super::record::{BranchCheckpointPayload, RecordType, SectionGeneration, SnapshotPayload};
use super::resume::{self, TurnChunkGrid};
use super::written_snapshots::{WrittenSnapshot, WrittenSnapshots};
use super::{PersistenceError, Result, SubstratePersistence};
use crate::persistence::streams::StreamId;
use crate::substrate::{ResidenceIndex, SubstrateLock, TurnMark};

/// A queued redo-log write.
///
/// A job whose value the enqueuer has already applied in memory carries the
/// `identity` it was applied under — minted when the value changed — and the
/// record is written with it, so the value and its record never disagree on
/// which write they are. A job whose value lands in the substrate only after
/// the append (`Tokens`, the recurrent records, KV-cold chunks) is stamped by
/// the append, and the owner applies it under that stamp.
pub(crate) enum WriteJob {
    /// A stream-decl (turn / section) metadata mirror. `stream_id` is the pure
    /// `decl.stream_id().0`, `payload` is `decl.encode()`; already applied
    /// in-memory by the enqueuer.
    StreamDecl {
        stream_id: u64,
        payload: Vec<u8>,
        identity: RecordIdentity,
    },
    /// A turn/section `Tokens` record. Disk-only — the in-memory token buffer
    /// is already set at record time, so this is purely the durable copy for
    /// reload.
    Tokens {
        stream_id: StreamId,
        token_ids: Vec<u32>,
    },
    /// A turn's wide-Q provenance signature (`WideQSig` record). The in-RAM
    /// blob is mirrored synchronously by the enqueuer (the same-session
    /// provenance scan needs it now); this appends only the durable copy.
    WideQSigs {
        stream_id: StreamId,
        payload: Vec<u8>,
        identity: RecordIdentity,
    },
    /// A turn's QSA index page (`TurnIndexPage` record). Mirrored in RAM by the
    /// enqueuer — this turn's own projections read it immediately — so this
    /// appends only the durable copy the next process start depends on.
    TurnIndexPage {
        stream_id: StreamId,
        payload: Vec<u8>,
        identity: RecordIdentity,
    },
    /// A warm→cold KV migration. The GPU gather already produced `grid`; the
    /// owner appends its chunks, folds their locations into the substrate
    /// index, marks the stream durable-through, then `install_cold`s — all
    /// AFTER the append, so a cold copy is only ever installed on durable data.
    KvCold {
        residence: ResidenceIndex,
        stream_id: StreamId,
        grid: TurnChunkGrid,
    },
    /// A conversation's recurrent-state snapshot at a turn seal (`Snapshot`
    /// record — the Gated DeltaNet matrices + conv tails). The newest
    /// supersedes all previous for the stream (single tail); the owner
    /// registers the fresh location in the substrate index after the append,
    /// `Tokens`-style.
    ///
    /// **Carried decoded and encoded on the owner's thread.** The encode is a
    /// ~65 MB copy per turn on a recurrent model, and its enqueuer is the
    /// scheduler's loop thread: encoding there put that copy between the decode
    /// steps of every turn still running. The queue is FIFO, so the record still
    /// lands before the turn's `Tokens`, exactly as the seal orders them.
    Snapshot {
        stream_id: StreamId,
        payload: SnapshotPayload,
    },
    /// A prompt branch's recurrent checkpoint (`BranchCheckpoint` record).
    /// Same single-tail append as [`Self::Snapshot`] and the same byte-cap
    /// class; a separate job because the record type is what tells compaction
    /// this one is a cache it may reclaim. Shared with the substrate's in-RAM
    /// memo of the same checkpoint, and written from its parts, so neither the
    /// queue nor the append copies the state.
    BranchCheckpoint {
        stream_id: StreamId,
        payload: Arc<BranchCheckpointPayload>,
    },
    /// A turn's projection-event trajectory (`ProjectionEvents` record,
    /// last-writer-wins per stream). The in-RAM blob is mirrored synchronously
    /// by the enqueuer; this appends only the durable copy.
    ProjectionEvents {
        stream_id: StreamId,
        payload: Vec<u8>,
        identity: RecordIdentity,
    },
    /// A pre-encoded conversation-metadata record — `Label` (the fields of a
    /// ConvMeta the write sets: conv_id, label, custom keys) or `Distilled`.
    /// Both are small and need no index registration after the append, so one
    /// job kind carries either. Encoded by the enqueuer.
    ConvMeta {
        record: RecordType,
        payload: Vec<u8>,
        identity: RecordIdentity,
    },
    /// A conversation's whole state (`ConvState` record), keyed in the header
    /// by its timeline so the greatest stamp supersedes the rest. Encoded by
    /// the enqueuer, like [`WriteJob::ConvMeta`].
    ConvState {
        timeline: u64,
        payload: Vec<u8>,
        identity: RecordIdentity,
    },
    /// A timeline tombstone. The enqueuer has already marked the timeline in
    /// RAM; the owner appends the record and drops the persistence-side
    /// snapshot and file index.
    TimelineTombstone {
        timeline_id: u64,
        reason: Option<String>,
    },
    /// A turn tombstone. Same split as [`Self::TimelineTombstone`]: the
    /// substrate mark is synchronous, the append is the owner's.
    TurnTombstone {
        mark: TurnMark,
        reason: Option<String>,
    },
    /// A section tombstone, including the generation the persistence minter
    /// stops stamping. Applied in RAM by the enqueuer first.
    SectionTombstone {
        stream_id: u64,
        generation: Option<SectionGeneration>,
        reason: Option<String>,
    },
    /// An identity this node minted for a write it then did not make — a
    /// read-modify-write that found nothing to change, or found its read
    /// overtaken. Marked never-coming in queue order, so it holds back no
    /// record written after it ([`SubstratePersistence::own_sendable_through`]).
    Abandoned { identity: RecordIdentity },
}

impl WriteJob {
    pub(super) fn byte_size(&self) -> u64 {
        match self {
            WriteJob::StreamDecl { payload, .. } => payload.len() as u64,
            WriteJob::Tokens { token_ids, .. } => (token_ids.len() * 4) as u64,
            WriteJob::WideQSigs { payload, .. } => payload.len() as u64,
            WriteJob::TurnIndexPage { payload, .. } => payload.len() as u64,
            WriteJob::ProjectionEvents { payload, .. } => payload.len() as u64,
            WriteJob::ConvMeta { payload, .. } => payload.len() as u64,
            WriteJob::ConvState { payload, .. } => payload.len() as u64,
            WriteJob::TimelineTombstone { reason, .. }
            | WriteJob::TurnTombstone { reason, .. }
            | WriteJob::SectionTombstone { reason, .. } => {
                reason.as_ref().map_or(0, |r| r.len()) as u64
            }
            WriteJob::Snapshot { payload, .. } => payload.encoded_len() as u64,
            WriteJob::BranchCheckpoint { payload, .. } => payload.parts().byte_len() as u64,
            WriteJob::KvCold { grid, .. } => grid.bytes() as u64,
            WriteJob::Abandoned { .. } => 0,
        }
    }

    /// Whether this job counts against the BYTE cap. Only the large payloads do —
    /// KV-cold grids and recurrent snapshots and checkpoints; the small metadata
    /// records (stream-decl / tokens / sigs) are gated by the EVENT cap alone, so
    /// a bulk backlog can never block the scheduler's latency-critical seal
    /// metadata behind the byte cap.
    pub(super) fn is_bulk(&self) -> bool {
        matches!(
            self,
            WriteJob::KvCold { .. } | WriteJob::Snapshot { .. } | WriteJob::BranchCheckpoint { .. }
        )
    }
}

/// Append one job, then index what it appended in the substrate. Run by the
/// log's owner, which holds `p` alone.
pub(super) fn process(
    job: WriteJob,
    substrate: &Arc<SubstrateLock>,
    p: &mut SubstratePersistence,
    snapshots: &WrittenSnapshots,
) {
    match job {
        WriteJob::StreamDecl {
            stream_id,
            payload,
            identity,
        } => {
            if let Err(e) = p.append_carried(
                RecordType::StreamDecl,
                0,
                stream_id,
                0,
                0,
                0,
                &payload,
                identity,
            ) {
                p.mark_unwritten(identity);
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    "stream-decl append failed: {e}"
                );
            }
        }
        WriteJob::Tokens {
            stream_id,
            token_ids,
        } => {
            // Fault injection (test-helpers): a stream armed via
            // [`fault::drop_next_tokens`] loses this record — the deterministic
            // stand-in for a crash between the snapshot append and the tokens
            // append, which is the one tear the §4.1 write ordering permits.
            // Nothing else about the write path changes; the record is simply
            // never appended, exactly as a crash would leave it.
            #[cfg(any(test, feature = "test-helpers"))]
            if fault::take_drop_tokens(stream_id) {
                tracing::warn!(
                    target: "candle_conversation::persistence::writer",
                    stream_id = stream_id.0,
                    "FAULT INJECTION: dropping Tokens record"
                );
                return;
            }
            // Register the record's location in the substrate index after the
            // append. Without the `apply_tokens_loc`, maintenance/compaction
            // can't see the tokens and reclaim them (see `persist_tokens_only`).
            match resume::persist_tokens_only(p, stream_id, &token_ids) {
                Ok(loc) => substrate.update(|s| s.apply_tokens_loc(stream_id, loc.value, loc.id)),
                Err(e) => tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    stream_id = stream_id.0,
                    "tokens append failed: {e}"
                ),
            }
        }
        WriteJob::WideQSigs {
            stream_id,
            payload,
            identity,
        } => {
            if let Err(e) = p.append_wide_q_sigs(stream_id, &payload, identity) {
                p.mark_unwritten(identity);
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    stream_id = stream_id.0,
                    "wide-Q sigs append failed: {e}"
                );
            }
        }
        WriteJob::TurnIndexPage {
            stream_id,
            payload,
            identity,
        } => {
            // Settle the substrate's held copy to the record's location after
            // the append. From here the page is read from the log when a
            // projection takes it, not held for the life of the process.
            match p.append_turn_index_page(stream_id, &payload, identity) {
                Ok(loc) => substrate.update(|s| s.settle_index_page(stream_id, loc, identity)),
                Err(e) => {
                    p.mark_unwritten(identity);
                    tracing::error!(
                        target: "candle_conversation::persistence::writer",
                        stream_id = stream_id.0,
                        "index page append failed: {e} — this turn or section will be \
                         borrowable but unindexable after a restart"
                    )
                }
            }
        }
        WriteJob::ProjectionEvents {
            stream_id,
            payload,
            identity,
        } => {
            if let Err(e) = p.append_projection_events(stream_id, &payload, identity) {
                p.mark_unwritten(identity);
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    stream_id = stream_id.0,
                    "projection events append failed: {e}"
                );
            }
        }
        WriteJob::ConvMeta {
            record,
            payload,
            identity,
        } => {
            if let Err(e) = p.append_carried(record, 0, 0, 0, 0, 0, &payload, identity) {
                p.mark_unwritten(identity);
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    ?record,
                    "conversation metadata append failed: {e}"
                );
            }
        }
        WriteJob::ConvState {
            timeline,
            payload,
            identity,
        } => {
            if let Err(e) = p.append_carried(
                RecordType::ConvState,
                0,
                timeline,
                0,
                0,
                0,
                &payload,
                identity,
            ) {
                p.mark_unwritten(identity);
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    timeline,
                    "conversation state append failed: {e}"
                );
            }
        }
        WriteJob::TimelineTombstone {
            timeline_id,
            reason,
        } => {
            if let Err(e) = p.write_tombstone(timeline_id, reason.as_deref()) {
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    timeline_id,
                    "timeline tombstone append failed: {e}"
                );
            }
        }
        WriteJob::TurnTombstone { mark, reason } => {
            if let Err(e) = p.write_turn_tombstone(mark, reason.as_deref()) {
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    timeline = mark.timeline.raw(),
                    turn = mark.turn_index,
                    "turn tombstone append failed: {e}"
                );
            }
        }
        WriteJob::SectionTombstone {
            stream_id,
            generation,
            reason,
        } => {
            if let Err(e) = p.write_section_tombstone(stream_id, generation, reason.as_deref()) {
                tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    stream_id,
                    "section tombstone append failed: {e}"
                );
            }
        }
        WriteJob::Snapshot { stream_id, payload } => {
            // Register the fresh location after the append, the `Tokens`
            // discipline. The old location is simply overwritten in the index;
            // its bytes were already credited dead by the accounting layer
            // inside the append. Written from the payload's parts: the state
            // blobs go from the conversation's own buffers to the file with no
            // intermediate copy.
            match append_snapshot(p, stream_id, &payload) {
                Ok(loc) => {
                    let tail =
                        substrate.update(|s| s.apply_snapshot_loc(stream_id, loc.value, loc.id));
                    // Reported only when it became the stream's tail — what a
                    // read of the timeline's snapshot then serves. One the index
                    // declined (the stream retired, or a newer record held)
                    // stands in for nothing, and a retired tail carries no state.
                    if tail && !payload.is_retired() {
                        snapshots.push(WrittenSnapshot {
                            timeline: payload.timeline_id,
                            turn_index: payload.turn_index,
                            loc: loc.value,
                        });
                    }
                }
                Err(e) => retract_failed_snapshot(substrate, p, stream_id, &e),
            }
        }
        WriteJob::BranchCheckpoint { stream_id, payload } => {
            // Same append-then-register discipline as `Snapshot`, into the
            // branch index, and written from the parts the same way.
            let parts = payload.parts();
            match p.write_branch_checkpoint(stream_id, &parts.slices()) {
                Ok(loc) => substrate
                    .update(|s| s.apply_branch_checkpoint_loc(stream_id, loc.value, loc.id)),
                Err(e) => tracing::error!(
                    target: "candle_conversation::persistence::writer",
                    stream_id = stream_id.0,
                    "branch checkpoint append failed: {e}"
                ),
            }
        }
        WriteJob::KvCold {
            residence,
            stream_id,
            grid,
        } => {
            if let Err(e) = write_kv_cold(substrate, p, residence, stream_id, &grid) {
                tracing::warn!(
                    target: "candle_conversation::persistence::writer",
                    residence = residence.0,
                    "warm→cold write failed: {e}"
                );
            }
        }
        WriteJob::Abandoned { identity } => p.mark_unwritten(identity),
    }
}

/// Append a turn's gathered KV chunks, mark the stream durable-through, fold
/// their locations into the substrate index, then `install_cold` — so a
/// residence's cold copy is set only after the bytes are appended.
fn write_kv_cold(
    substrate: &Arc<SubstrateLock>,
    p: &mut SubstratePersistence,
    residence: ResidenceIndex,
    stream_id: StreamId,
    grid: &TurnChunkGrid,
) -> Result<()> {
    let (stored, locs) = resume::persist_turn_chunks_capture(p, stream_id, grid)?;
    if stored.is_empty() {
        return Ok(());
    }
    // Mark durable through the last-written chunk BEFORE the cold install, so
    // a crash after a resident tier is dropped still knows the chunks landed.
    let through = (stored.iter().map(|s| s.chunks.len()).sum::<usize>().max(1) - 1) as u64;
    let commit = p.commit_stream(stream_id, through)?;
    substrate.update(|view| {
        for &(flat, loc) in &locs {
            view.apply_chunk_loc(stream_id, flat, loc.value, loc.id);
        }
        // The index holds the commit as well as the log: maintenance rewrites
        // a `Commit` from `committed_through`, so one only the log knows is
        // lost with the first rewrite of its segment.
        view.apply_commit_through(stream_id, through, commit);
    });
    substrate.read().install_cold(residence, stored);
    Ok(())
}

/// Append a snapshot from its parts.
fn append_snapshot(
    p: &mut SubstratePersistence,
    stream_id: StreamId,
    payload: &SnapshotPayload,
) -> Result<Stamped<RecordLoc>> {
    // Fault injection (test-helpers): a stream armed via
    // [`fault::fail_next_snapshot`] has this append fail, as a full or failing
    // disk would fail it.
    #[cfg(any(test, feature = "test-helpers"))]
    if fault::take_fail_snapshot(stream_id) {
        return Err(PersistenceError::Corrupt(
            "fault injection: the snapshot append failed".into(),
        ));
    }
    // Fault injection (test-helpers): a stream armed via
    // [`fault::garble_next_snapshot`] lands bytes no decoder accepts in place of
    // this payload — a record that is present, checksummed and undecodable.
    #[cfg(any(test, feature = "test-helpers"))]
    if fault::take_garble_snapshot(stream_id) {
        return p.write_snapshot(stream_id, &[fault::GARBLED_SNAPSHOT]);
    }
    let parts = payload.parts();
    p.write_snapshot(stream_id, &parts.slices())
}

/// A seal's snapshot did not land. The stream's previous snapshot is a turn
/// behind the `Tokens` record queued after this one, and a resume would install
/// it under K/V it never saw — silent staleness, which the seal's write order
/// exists to rule out. So the stream is retracted: dropped from the in-RAM
/// index now, and durably by an empty `Snapshot` record, after which a resume
/// recomputes instead.
fn retract_failed_snapshot(
    substrate: &Arc<SubstrateLock>,
    p: &mut SubstratePersistence,
    stream_id: StreamId,
    error: &PersistenceError,
) {
    let retracted = p.retract_snapshot(stream_id);
    // Dropped from the in-RAM index whether or not the retraction landed: the
    // tail is stale either way. When it landed, its identity also refuses any
    // older snapshot that reaches the substrate later.
    substrate.update(|sub| match &retracted {
        Ok(id) => sub.retract_snapshot(stream_id, *id),
        Err(_) => sub.drop_snapshot_loc(stream_id),
    });
    match retracted {
        Ok(_) => tracing::error!(
            target: "candle_conversation::persistence::writer",
            stream_id = stream_id.0,
            "snapshot append failed: {error} — the stream's snapshot is retracted, \
             so a resume recomputes its recurrent state"
        ),
        Err(retract) => tracing::error!(
            target: "candle_conversation::persistence::writer",
            stream_id = stream_id.0,
            "snapshot append failed: {error}, and so did its retraction: {retract} — \
             the previous snapshot is still the durable tail, and a reload before \
             the next successful seal resumes one turn behind its K/V"
        ),
    }
}

/// Deterministic fault injection for the durable write path — the enablement
/// the torn-seal oracles (catalogue D2/D3) are blocked on. Not a behaviour
/// switch: production code never arms a fault; a test arms a specific stream
/// and the owner drops exactly that one record, reproducing the one tear the
/// §4.1 write ordering permits (snapshot landed, tokens lost) without needing
/// to crash the process at a precise instant.
#[cfg(any(test, feature = "test-helpers"))]
pub mod fault {
    use crossbeam::queue::SegQueue;

    use crate::persistence::streams::StreamId;

    /// The streams armed for one fault each. Lock-free: armed by a test,
    /// consumed by the owner thread.
    pub struct Armed(SegQueue<StreamId>);

    impl Armed {
        const fn new() -> Self {
            Armed(SegQueue::new())
        }

        fn arm(&self, stream: StreamId) {
            self.0.push(stream);
        }

        /// Consume an armed fault for `stream`, if any, leaving every other.
        fn take(&self, stream: StreamId) -> bool {
            let mut found = false;
            for _ in 0..self.0.len() {
                let Some(s) = self.0.pop() else { break };
                if !found && s == stream {
                    found = true;
                } else {
                    self.0.push(s);
                }
            }
            found
        }
    }

    static DROP_TOKENS: Armed = Armed::new();

    /// Arm: the next `Tokens` record for `stream` is silently dropped instead
    /// of appended. One-shot per call — the arm is consumed when it fires.
    pub fn drop_next_tokens(stream: StreamId) {
        DROP_TOKENS.arm(stream);
    }

    /// Consume an armed drop for `stream`, if any.
    pub(super) fn take_drop_tokens(stream: StreamId) -> bool {
        DROP_TOKENS.take(stream)
    }

    static FAIL_SNAPSHOT: Armed = Armed::new();

    /// Arm: the next `Snapshot` append for `stream` fails, as a full or failing
    /// disk would fail it. One-shot per call, like [`drop_next_tokens`].
    pub fn fail_next_snapshot(stream: StreamId) {
        FAIL_SNAPSHOT.arm(stream);
    }

    /// Consume an armed snapshot failure for `stream`, if any.
    pub(super) fn take_fail_snapshot(stream: StreamId) -> bool {
        FAIL_SNAPSHOT.take(stream)
    }

    static GARBLE_SNAPSHOT: Armed = Armed::new();

    /// What a garbled snapshot lands: too short for any version word and field
    /// that follows it, so every decode refuses it.
    pub const GARBLED_SNAPSHOT: &[u8] = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01];

    /// Arm: the next `Snapshot` append for `stream` lands [`GARBLED_SNAPSHOT`]
    /// instead of its payload. One-shot per call, like [`drop_next_tokens`].
    pub fn garble_next_snapshot(stream: StreamId) {
        GARBLE_SNAPSHOT.arm(stream);
    }

    /// Consume an armed garble for `stream`, if any.
    pub(super) fn take_garble_snapshot(stream: StreamId) -> bool {
        GARBLE_SNAPSHOT.take(stream)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{fault, WriteJob};
    use crate::persistence::identity::RecordIdentity;
    use crate::persistence::owner::PersistenceOwner;
    use crate::persistence::record::{RecordType, SnapshotPayload};
    use crate::persistence::streams::StreamId;
    use crate::persistence::{dir_fingerprint, SubstratePersistence};
    use crate::substrate::{Substrate, SubstrateLock};

    fn snapshot_at(turn_index: u32) -> SnapshotPayload {
        SnapshotPayload {
            timeline_id: 7,
            turn_index,
            schedule_hash: 0,
            layers: Vec::new(),
            aux: vec![turn_index as u8; 64],
        }
    }

    fn written(owner: &PersistenceOwner) {
        owner.wait_written(owner.ticket());
    }

    /// **A snapshot that fails to land retracts the one before it.** That one
    /// describes the previous turn, and the turn's `Tokens` record is queued
    /// right behind the failed append: left as the tail, a reload would
    /// install a state one turn behind its K/V. The retraction removes it from
    /// the in-RAM index at once and from every later reload, and leaves other
    /// conversations' snapshots alone; the stream's next snapshot is live again.
    #[test]
    fn a_failed_snapshot_retracts_the_previous_tail_now_and_on_reload() {
        let dir = tempfile::tempdir().unwrap();
        let failed = StreamId(0xF00D_0077);
        let other = StreamId(0xF00D_0078);
        let mut substrate = Substrate::new();
        let p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let substrate = Arc::new(SubstrateLock::new(substrate));
        let owner = PersistenceOwner::spawn(p, substrate.clone());

        for stream_id in [failed, other] {
            owner.enqueue(WriteJob::Snapshot {
                stream_id,
                payload: snapshot_at(0),
            });
        }
        // Both turn-0 snapshots land before the fault is armed, so it can only
        // fire on the turn-1 append.
        written(&owner);
        assert!(substrate.read().recurrent_snapshot_loc(failed).is_some());
        fault::fail_next_snapshot(failed);
        owner.enqueue(WriteJob::Snapshot {
            stream_id: failed,
            payload: snapshot_at(1),
        });
        written(&owner);

        let kept = substrate.read().recurrent_snapshot_loc(other);
        assert!(kept.is_some());
        assert_eq!(substrate.read().recurrent_snapshot_loc(failed), None);
        owner.shutdown();
        drop(owner);

        let mut reloaded = Substrate::new();
        let p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut reloaded).unwrap();
        assert_eq!(reloaded.recurrent_snapshot_loc(failed), None);
        assert_eq!(reloaded.recurrent_snapshot_loc(other), kept);

        // The next seal's snapshot is the stream's tail again, on reload too.
        let substrate = Arc::new(SubstrateLock::new(reloaded));
        let owner = PersistenceOwner::spawn(p, substrate.clone());
        owner.enqueue(WriteJob::Snapshot {
            stream_id: failed,
            payload: snapshot_at(2),
        });
        owner.shutdown();
        drop(owner);
        let fresh = substrate.read().recurrent_snapshot_loc(failed);
        assert!(fresh.is_some());
        let mut reloaded = Substrate::new();
        let _p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut reloaded).unwrap();
        assert_eq!(reloaded.recurrent_snapshot_loc(failed), fresh);
    }

    /// **A retired tail supersedes the state before it, now and on reload.** A
    /// timeline that is never resumed retires its snapshot by writing an empty
    /// one under the same stream: the index then names the empty record, so the
    /// whole state behind it is dead to maintenance, and a reload agrees.
    #[test]
    fn a_retired_snapshot_supersedes_the_state_before_it_now_and_on_reload() {
        let dir = tempfile::tempdir().unwrap();
        let stream_id = StreamId(0xF00D_0090);
        let mut substrate = Substrate::new();
        let p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut substrate).unwrap();
        let substrate = Arc::new(SubstrateLock::new(substrate));
        let owner = PersistenceOwner::spawn(p, substrate.clone());
        let mut full = snapshot_at(0);
        full.aux = vec![9u8; 1 << 16];
        owner.enqueue(WriteJob::Snapshot {
            stream_id,
            payload: full,
        });
        owner.enqueue(WriteJob::Snapshot {
            stream_id,
            payload: SnapshotPayload::retired(7),
        });
        written(&owner);

        let tail = substrate
            .read()
            .recurrent_snapshot_loc(stream_id)
            .expect("the retired tail is indexed");
        assert!(tail.payload_len < 4096, "the tail is the empty record");
        let bytes = owner.call(|p| p.read_record_payload(&tail)).unwrap();
        assert!(SnapshotPayload::decode(&bytes).unwrap().is_retired());
        owner.shutdown();
        drop(owner);

        let mut reloaded = Substrate::new();
        let _p = SubstratePersistence::open_in_with_substrate(dir.path(), &mut reloaded).unwrap();
        assert_eq!(reloaded.recurrent_snapshot_loc(stream_id), Some(tail));
    }

    /// **A read-only owner drops every write and touches no file.** Metadata,
    /// tokens and a bulk snapshot are all dropped before they take a queue
    /// slot, no location is registered for a record that was never written,
    /// shutdown returns without a commit, and the store is byte-identical.
    #[test]
    fn a_read_only_owner_drops_every_write_and_touches_no_file() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("kvtier_writer_ro_{nanos}"));
        {
            let mut substrate = Substrate::new();
            let mut p = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate).unwrap();
            p.write_debug_id(1, "seed", p.mint()).unwrap();
            p.commit().unwrap();
        }
        let before = dir_fingerprint(&dir);

        let mut substrate = Substrate::new();
        let p =
            SubstratePersistence::open_in_with_substrate_read_only(&dir, &mut substrate).unwrap();
        let substrate = Arc::new(SubstrateLock::new(substrate));
        let owner = PersistenceOwner::spawn(p, substrate.clone());
        let stream = StreamId(42);
        owner.enqueue(WriteJob::StreamDecl {
            stream_id: stream.0,
            payload: vec![1, 2, 3],
            identity: RecordIdentity::NONE,
        });
        owner.enqueue(WriteJob::Tokens {
            stream_id: stream,
            token_ids: vec![7, 8, 9],
        });
        owner.enqueue(WriteJob::ConvMeta {
            record: RecordType::Label,
            payload: vec![4, 5],
            identity: RecordIdentity::NONE,
        });
        owner.enqueue(WriteJob::Snapshot {
            stream_id: stream,
            payload: SnapshotPayload {
                timeline_id: 7,
                turn_index: 0,
                schedule_hash: 0,
                layers: Vec::new(),
                aux: vec![0u8; 4096],
            },
        });
        assert_eq!(owner.ticket(), 0, "nothing took a queue slot");
        assert!(substrate
            .read()
            .stream_of(stream)
            .and_then(|s| s.tokens)
            .is_none());
        assert_eq!(owner.call(|p| p.pending_bytes()), 0);
        owner.shutdown();
        drop(owner);
        drop(substrate);
        assert_eq!(dir_fingerprint(&dir), before);
        std::fs::remove_dir_all(&dir).ok();
    }
}
