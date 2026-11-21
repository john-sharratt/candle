//! Snapshots the substrate's decoded state into the records a sweep re-emits.
//!
//! These records live decoded in RAM, so they can be rebuilt rather than read
//! back, and a drop must never remove the only copy of one. Every record class
//! belongs to exactly one of this set or the relocation set, since a class in
//! neither is deleted silently. The gates skip orphans and the bulk of retired
//! turns, but keep tombstone markers, so retirement survives a sweep. Shared
//! payloads are referenced rather than copied.

use super::*;

/// A resident record snapshotted from the substrate for re-emission — carries
/// no on-disk location (the substrate holds its decoded state), so it is
/// rebuilt from live state rather than read back, under `id`: the identity of
/// the record holding the value when the snapshot was taken, which the copy
/// keeps.
///
/// The payload shares the substrate's own bytes where the substrate holds them
/// shared (index pages, wide-Q signatures, projection events). Those are the
/// bulk of the set — up to a megabyte a turn — and a plan lives for the whole
/// op, so an owned copy held a second image of every live turn's metadata in
/// RAM for minutes at a time.
pub(in super::super) struct Resident {
    pub(in super::super) rt: RecordType,
    pub(in super::super) stream_id: u64,
    pub(in super::super) chunk_index: u64,
    pub(in super::super) payload: Arc<[u8]>,
    pub(in super::super) id: RecordIdentity,
}

/// Snapshot every **resident** record from the substrate's live state — the
/// drop-safety net (re-emitted so no dropped segment holds the only copy of
/// live metadata), including a `Tombstone` marker per tombstoned timeline and a
/// `Distilled` marker per distilled timeline. Pure read; no I/O.
///
/// **Resident means "the substrate holds its decoded state".** A record class
/// that does not — `Npc`, whose payload belongs to the daemon above, and the
/// read-back `Chunk` / `Tokens` / `Snapshot` / `BranchCheckpoint` bulk — cannot
/// appear here and must be *relocated* instead ([`SubstratePersistence::gather_relocations`]
/// and the `npc_relocs` worklist). Every class must be in exactly one of those
/// two sets: a class in neither is deleted by the first op that touches its
/// segment, silently. `Npc` and `TurnCoupling` were each in neither.
///
/// Each record carries the identity of the record holding its value: a value's
/// from the substrate, which keeps it beside the value, and a marker's from
/// `markers`.
pub(in super::super) fn gather_resident_set(
    substrate: &Substrate,
    markers: &MarkerIds,
    node: u64,
) -> Vec<Resident> {
    let tombstoned: HashSet<u64> = substrate
        .tombstoned_timelines()
        .iter()
        .map(|t| t.raw())
        .collect();
    let distilled: HashMap<u64, DistillMode> = substrate
        .distilled_timelines()
        .iter()
        .map(|(t, m)| (t.raw(), *m))
        .collect();

    let mut out: Vec<Resident> = Vec::new();
    for (key, entry) in substrate.all_physical_streams() {
        let stream_id = key.stream_id;
        let shed = classify(substrate, key, entry, &tombstoned, &distilled);
        // Tombstoned AND undistilled goes; tombstoned-but-distilled is the
        // provenance corpus and is retained by its mode. See the same gate in
        // `compaction::collect_live_records`.
        if shed.dropped_wholesale() {
            continue;
        }
        // Orphan (no decl, and none coming): re-emit nothing — its StreamDecl
        // is gone, so its sig/projection records are unreachable dead weight.
        // Re-emitting them would keep the orphan's metadata immortal on the
        // incremental path. (The `if let Some(decl)` below already skips the
        // StreamDecl; this also skips the sig/projection re-emit.) Mirrors the
        // other orphan gates.
        if entry.is_orphan(node) {
            continue;
        }
        // A retired turn keeps only its `StreamDecl` placeholder; its sig and
        // projection records are shed with the rest of its bulk, exactly as
        // `collect_live_records` sheds them.
        let keep_sig = shed.keep_sig();
        if let Some(decl) = &entry.decl {
            out.push(Resident {
                rt: RecordType::StreamDecl,
                stream_id: stream_id.0,
                chunk_index: 0,
                payload: decl.encode().into(),
                id: entry.ids.decl,
            });
        }
        if keep_sig {
            if let Some(p) = &entry.projection_events {
                out.push(Resident {
                    rt: RecordType::ProjectionEvents,
                    stream_id: stream_id.0,
                    chunk_index: 0,
                    payload: Arc::clone(p),
                    id: entry.ids.projection_events,
                });
            }
            if let Some(p) = &entry.wide_q_sigs {
                out.push(Resident {
                    rt: RecordType::WideQSig,
                    stream_id: stream_id.0,
                    chunk_index: 0,
                    payload: Arc::clone(p),
                    id: entry.ids.wide_q_sigs,
                });
            }
            // The index page rides with the turn's K/V: a turn that survives
            // relocation with its chunks but loses its page comes back
            // borrowable and unindexable. Only a page this process holds is
            // re-emitted from RAM; one left in the log is relocated, by
            // location, like the turn's tokens
            // ([`SubstratePersistence::gather_index_page_relocs`]).
            if let Some(IndexPage::Resident(p)) = &entry.index_page {
                out.push(Resident {
                    rt: RecordType::TurnIndexPage,
                    stream_id: stream_id.0,
                    chunk_index: 0,
                    payload: p.shared_bytes(),
                    id: entry.ids.index_page,
                });
            }
        }
        if let Some(through) = entry.committed_through {
            out.push(Resident {
                rt: RecordType::Commit,
                stream_id: stream_id.0,
                chunk_index: through,
                payload: Arc::from([]),
                id: entry.ids.commit,
            });
        }
    }
    for (tl, meta, id) in substrate.live_label_records() {
        // A tombstoned timeline that is ALSO distilled is the provenance corpus
        // (calibration exemplars: archived, distilled, then tombstoned out of the
        // live gather while their signatures keep answering the belief scan).
        // Its marker/metadata must survive, or the next reload sees a
        // content-declaring turn with no distill exemption and condemns it.
        // Only an undistilled tombstone is a retired conversation.
        if tombstoned.contains(&tl) && !distilled.contains_key(&tl) {
            continue;
        }
        out.push(Resident {
            rt: RecordType::Label,
            stream_id: 0,
            chunk_index: 0,
            payload: encode_label_payload(tl, &meta.conv_id, &meta.label, &meta.custom).into(),
            id,
        });
    }
    // A snapshot retraction carries no payload and so has no location to
    // relocate from, but it is the stream's value: re-emitted, so a drop never
    // takes the record that refuses an older snapshot arriving later.
    for (stream_id, id) in substrate.retracted_snapshots() {
        out.push(Resident {
            rt: RecordType::Snapshot,
            stream_id: stream_id.0,
            chunk_index: 0,
            payload: Arc::from([]),
            id,
        });
    }
    // Records about a timeline not registered yet — durable like any other,
    // and carried so a drop never takes the only copy.
    for (tl, rt, payload, id) in substrate.pending_records() {
        if tombstoned.contains(&tl) {
            continue;
        }
        let stream_id = if rt == RecordType::ConvState { tl } else { 0 };
        out.push(Resident {
            rt,
            stream_id,
            chunk_index: 0,
            payload: payload.into(),
            id,
        });
    }
    // The whole conversation state, keyed in the header by timeline — see
    // `ConvState`. Same retirement rule as the labels above.
    for (tl, state, id) in substrate.live_conv_states() {
        if tombstoned.contains(&tl) && !distilled.contains_key(&tl) {
            continue;
        }
        out.push(Resident {
            rt: RecordType::ConvState,
            stream_id: tl,
            chunk_index: 0,
            payload: encode_conv_state_payload(tl, &state).into(),
            id,
        });
    }
    for (p, id) in substrate.live_tree_metadata_payloads() {
        if tombstoned.contains(&p.timeline_id) {
            continue;
        }
        out.push(Resident {
            rt: RecordType::TreeMetadata,
            stream_id: 0,
            chunk_index: 0,
            payload: p.encode().into(),
            id,
        });
    }
    for (tl, debug_id, id) in substrate.live_debug_ids() {
        if tombstoned.contains(&tl) {
            continue;
        }
        out.push(Resident {
            rt: RecordType::DebugId,
            stream_id: 0,
            chunk_index: 0,
            payload: DebugIdPayload {
                timeline_id: tl,
                debug_id,
            }
            .encode()
            .into(),
            id,
        });
    }
    // A marker's identity is the persistence side's, not the substrate's.
    let marker = |rt: RecordType, stream_id: u64, payload: Vec<u8>| Resident {
        rt,
        stream_id,
        chunk_index: 0,
        id: markers.identity_for(rt, stream_id, &payload),
        payload: payload.into(),
    };
    // Re-emitted for tombstoned timelines too: the marker is what exempts a
    // distilled turn from the integrity verdict, and a distilled turn has
    // legitimately shed its tokens/KV. Losing it condemns the provenance corpus
    // as "corrupt" on the next reload.
    for (&tl, &mode) in &distilled {
        out.push(marker(
            RecordType::Distilled,
            0,
            DistillPayload {
                timeline_id: tl,
                mode,
            }
            .encode(),
        ));
    }
    // A tombstone a compaction expired is no longer held dead, so it is not
    // carried (§6.8).
    for t in substrate.tombstoned_timelines() {
        out.push(marker(
            RecordType::Tombstone,
            0,
            TombstonePayload {
                timeline_id: t.raw(),
                turn_index: None,
                turn_writer: 0,
                reason: None,
            }
            .encode(),
        ));
    }
    if let Some(deleted) = substrate.deleted_payload() {
        out.push(Resident {
            rt: RecordType::Deleted,
            stream_id: 0,
            chunk_index: 0,
            payload: deleted.encode().into(),
            id: RecordIdentity::NONE,
        });
    }
    // **And the turn-scoped ones, or retirement does not survive a sweep.**
    //
    // This loop re-emitted only whole-timeline tombstones. A turn tombstone —
    // what `retention` writes when a conversation's tail passes `keep_turns`,
    // and what the `drop_turn` corrupt-turn policy writes when it condemns one —
    // lived in whatever segment it was appended to, and when maintenance
    // relocated that segment's live records it was not among them. The old
    // segment was then dropped and the tombstone went with it.
    //
    // Losing one does not merely forget a deletion. The `is_tomb` gate at the
    // top of this function reads `tombstoned_timelines`, so a turn-scoped
    // tombstone leaves its stream fully live and the turn's decl, tokens and KV
    // are all carried forward: retirement undid itself on a timer. The turn was
    // marked dead, the mark was swept away, and the next reload rebuilt
    // `tombstoned_turns` from a log that no longer mentioned it. Every retired
    // turn came back to life — with the corruption it was condemned for, where
    // that was the reason — its KV counted live again, and the segment holding
    // it pinned for good.
    //
    // Measured on a live daemon: 37,705 turn streams on disk, 70 tombstones
    // between them (only the ones written since the last sweep), and a 132 GB
    // floor that compaction ran against for days without ever reclaiming.
    // `compaction::collect_live_records` always carried these forward; this is
    // the incremental path catching up with it.
    let mut marks: Vec<_> = substrate.tombstoned_turns().iter().copied().collect();
    marks.sort_unstable();
    for mark in marks {
        // A turn of a timeline that is going wholesale needs no marker of its
        // own — the same gate `collect_live_records` applies.
        if tombstoned.contains(&mark.timeline.raw()) {
            continue;
        }
        out.push(marker(
            RecordType::Tombstone,
            0,
            TombstonePayload {
                timeline_id: mark.timeline.raw(),
                turn_index: Some(mark.turn_index),
                turn_writer: mark.writer,
                reason: None,
            }
            .encode(),
        ));
    }
    // Tombstoned section streams — the section counterpart of the two
    // `Tombstone` loops above, and the incremental path's half of the same
    // pair `compaction::collect_live_records` carries. Losing this on a
    // relocation sweep is the exact bug this file's own history warns
    // about (`Npc`, `TurnCoupling`): the corrupted section's `Chunk`
    // records would still be shed here as ordinary dead weight, but with
    // the marker gone the next reload would find no tombstone, treat the
    // section as merely absent, and the next ingest would re-persist and
    // eventually re-corrupt the same content address. One per ended
    // generation, naming it; one per unstamped stream ended before
    // generations.
    let ended = substrate
        .ended_section_generations()
        .map(|(sid, g)| (sid, Some(g)))
        .chain(substrate.ended_unstamped_sections().map(|sid| (sid, None)));
    for (stream_id, generation) in ended {
        out.push(marker(
            RecordType::SectionTombstone,
            stream_id.0,
            SectionTombstonePayload {
                generation,
                reason: None,
            }
            .encode(),
        ));
    }
    // Tool-round-trip couplings — see the twin loop in
    // `compaction::collect_live_records`. Payload-keyed, so no supersession
    // accounting protects them and no location map relocates them; re-emitting
    // from the substrate's live set is the only thing that carries them.
    for (timeline_id, from_turn, id) in substrate.live_couplings() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        out.push(Resident {
            rt: RecordType::TurnCoupling,
            stream_id: 0,
            chunk_index: 0,
            payload: TurnCouplingPayload {
                timeline_id,
                from_turn,
            }
            .encode()
            .into(),
            id,
        });
    }
    // Host-defined keyed state — the npcd command-table flag and anything else
    // stored through `SharedSubstrate::put_custom_object`. Resident, so the same
    // re-emit contract as the couplings above: the winner for each key is held
    // in `Substrate::custom_objects` and carried forward from RAM. The key's
    // hash rides in the header `stream_id` so supersession accounting is
    // mechanical (see the twin block in `compaction::collect_live_records`).
    for (obj, id) in substrate.held_custom_objects() {
        out.push(Resident {
            rt: RecordType::CustomObject,
            stream_id: obj.stream_id(),
            chunk_index: 0,
            payload: obj.encode().into(),
            id,
        });
    }
    out
}

impl LiveIndexes {
    /// The whole resident set an op re-emits: the substrate's
    /// ([`gather_resident_set`]) and the log's own — the `Replica` record,
    /// when the member has written or applied anything or deferred anything,
    /// and the `Rebase` record, when it has rejoined after writing cut off.
    pub(in super::super) fn resident_set(&self, substrate: &Substrate) -> Vec<Resident> {
        let mut out = gather_resident_set(substrate, &self.marker_ids, self.node());
        let replica = self.replica_record();
        if !replica.is_empty() {
            out.push(Resident {
                rt: RecordType::Replica,
                stream_id: 0,
                chunk_index: 0,
                payload: replica.encode().into(),
                id: RecordIdentity::NONE,
            });
        }
        if !self.rebases.is_empty() {
            out.push(Resident {
                rt: RecordType::Rebase,
                stream_id: 0,
                chunk_index: 0,
                payload: self.rebases.encode().into(),
                id: RecordIdentity::NONE,
            });
        }
        out
    }
}
