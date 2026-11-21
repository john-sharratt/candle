//! Plans the live set of a compaction in dependency order, without reading any
//! bytes from disk.
//!
//! Every live record becomes an item. Records that sit in the log are carried
//! verbatim by their location, and resident metadata is encoded afresh from the
//! substrate's memory. Every record class must appear here or be reclaimed on
//! purpose, since an unhandled class disappears at the next compaction without
//! a warning. Tombstones are carried until they expire. A distilled timeline
//! sheds its recurrent state and its KV but keeps its signatures and projection
//! events. A stream with no declaration is an orphan and is dropped. Characters
//! are carried verbatim, because their payloads live in the daemon, not here.

use super::*;

/// Collect every **live** record, in dependency order, as [`CompactItem`]s —
/// planning only, **no disk reads**. `ModelKey` / `Template` / `Tokenizer` /
/// `Chunk` / `Tokens` become `Raw` items carrying their source `(segment,
/// offset, record_size)` and a header synthesized from the substrate/manifest
/// index; every resident record (`StreamDecl`, `Commit`, `Label`, `ConvState`,
/// `ProjectionEvents`, `WideQSig`, `TreeMetadata`, `TurnCoupling`, `DebugId`,
/// `Distilled`) is a `Synth` item carrying its freshly-encoded payload.
/// [`write_compacted_log`] reads the `Raw` bytes back coalesced and stages them
/// verbatim.
///
/// **Every record class must appear here or be deliberately reclaimed.** There
/// is no fallback that carries an unhandled type forward and nothing warns when
/// one is missing — the record simply stops existing at the next compaction.
/// `Npc` and `TurnCoupling` were each lost exactly that way. The exhaustive
/// match in [`super::accounting::RecordAccounting::record`] is the checklist:
/// a header-keyed type needs a `Raw` carry-forward from its location map, and a
/// payload-keyed one needs a `Synth` re-emit from substrate RAM.
///
/// A tombstone is carried until `expiry` drops it (§6.8): this compaction runs
/// past every tombstone it carries, and says so in the `Deleted` record it
/// writes. The member's own [`Housekeeping`] is written whole.
#[allow(clippy::too_many_arguments)]
pub fn collect_live_records(
    manifest: &Manifest,
    substrate: &Substrate,
    npc_locs: &StampedLocs<u64>,
    vfs_index: &VfsIndex,
    blob_index: &BlobIndex,
    marker_ids: &MarkerIds,
    node: u64,
    expiry: TombstoneExpiry,
    housekeeping: &Housekeeping,
) -> Vec<CompactItem> {
    let mut out: Vec<CompactItem> = Vec::new();

    if !housekeeping.replica.is_empty() {
        out.push(local_record(
            RecordType::Replica,
            housekeeping.replica.encode(),
        ));
    }
    if !housekeeping.rebases.is_empty() {
        out.push(local_record(
            RecordType::Rebase,
            housekeeping.rebases.encode(),
        ));
    }

    // A conversation's file events, staged verbatim: their bodies belong to
    // the daemon above, so like characters they are carried by location.
    // Only the live events of live timelines — a retired, distilled or
    // unregistered conversation's are orphans and stay behind. A tombstone
    // stays behind too when the events it killed were all this member's: they
    // are not carried, so nothing is left for it to kill. One that killed
    // another member's event is carried, because that event can still arrive
    // and must still die (`VfsIndex`, §6.5).
    for (timeline, tl) in carried(substrate, vfs_index) {
        let events = tl
            .events()
            .iter()
            .map(|(id, loc)| (RecordType::VfsEvent, id, loc));
        let tombstones = tl
            .kept_tombstones(node)
            .map(|(id, loc)| (RecordType::VfsTombstone, id, loc));
        for (record_type, id, loc) in events.chain(tombstones) {
            out.push(CompactItem::raw(
                RecordHeader {
                    record_type,
                    format: 0,
                    payload_len: loc.payload_len,
                    crc: 0,
                    stream_id: timeline,
                    chunk_index: id.seq,
                    token_count: 0,
                    origin: id.origin,
                    ..RecordHeader::NONE
                },
                loc.segment,
                loc.offset,
                loc.record_size,
            ));
        }
    }

    // Blobs, staged verbatim by location: their bytes are never in RAM. The
    // index holds each hash's greatest-stamp record and nothing a tombstone
    // retired, so that is what is carried; the tombstone itself is a custom
    // object, carried below until it expires.
    for (key, loc) in blob_index.blobs() {
        out.push(CompactItem::raw(
            RecordHeader {
                record_type: RecordType::Blob,
                format: 0,
                payload_len: loc.payload_len,
                crc: 0,
                stream_id: key.hi,
                chunk_index: key.lo,
                token_count: 0,
                ..RecordHeader::NONE
            },
            loc.segment,
            loc.offset,
            loc.record_size,
        ));
    }

    // Characters, staged verbatim from wherever they physically live.
    //
    // `Raw`, not `Synth`, and this is the one record class where that is forced
    // rather than chosen: every other payload-keyed record on this page is
    // re-encoded from the substrate's RAM, but an NPC's payload is not in this
    // process's RAM at all — the registry that owns it lives in the daemon
    // above. So the winner is carried across byte-for-byte, exactly as
    // `Snapshot` is.
    //
    // Omitting this loop does not fail a test or log a warning. It deletes
    // every character in the store, at the next compaction, silently.
    //
    // Tombstoned characters are carried too: a deleted character's id must stay
    // taken, because the acts it already committed still name it.
    let mut npcs: Vec<(&u64, &RecordLoc)> = npc_locs.iter().collect();
    // Deterministic order — a compaction that shuffles the cast produces a
    // different file from the same inputs, which makes the output impossible to
    // diff between runs.
    npcs.sort_unstable_by_key(|(id, _)| **id);
    for (npc_id, loc) in npcs {
        out.push(CompactItem::raw(
            RecordHeader {
                record_type: RecordType::Npc,
                format: 0,
                payload_len: loc.payload_len,
                crc: 0,
                stream_id: *npc_id,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            },
            loc.segment,
            loc.offset,
            loc.record_size,
        ));
    }

    // Singletons — staged verbatim from wherever they physically live.
    for (rt, loc) in [
        (RecordType::ModelKey, manifest.model_key),
        (RecordType::Template, manifest.template),
        (RecordType::Tokenizer, manifest.tokenizer),
    ] {
        if let Some(loc) = loc {
            out.push(CompactItem::raw(
                singleton_header(rt, loc.payload_len),
                loc.segment,
                loc.offset,
                loc.record_size,
            ));
        }
    }

    // Tombstoned timelines drop out of the compacted log entirely
    // — their records are physically gone, not merely hidden.  This
    // is what reclaims disk after a refresh cycle replaces a
    // timeline.
    let tombstoned: std::collections::HashSet<u64> = substrate
        .tombstoned_timelines()
        .iter()
        .map(|t| t.raw())
        .collect();
    let distilled: HashMap<u64, DistillMode> = substrate
        .distilled_timelines()
        .iter()
        .map(|(t, m)| (t.raw(), *m))
        .collect();
    // An ended section generation needs no gate here: its `StreamDecl` /
    // `Chunk` / `Tokens` records are the corrupted or stale bytes the tombstone
    // exists to keep unread, and the tombstone already took them out of the
    // index (`Substrate::end_section`), so the walk below never meets them.

    // Recurrent-state snapshots: one live tail per conversation, staged
    // verbatim (`Raw`, the `Tokens` shape — the payload is a multi-MB state
    // blob nothing holds in RAM).
    //
    // **Distilled timelines shed theirs.** Distillation keeps the provenance
    // signatures and drops the content; the recurrent state IS content —
    // derived from the token stream, and useless without the K/V that was shed
    // alongside it. Without this gate a distilled timeline sheds megabytes of
    // K/V and carries ~63 MiB of state forward on every compaction pass,
    // forever, for a conversation that is never resumed. That a distilled
    // conversation becomes unresumable is correct: distillation is for
    // calibration exemplars.
    //
    // Tombstoned timelines are already handled on apply (the tombstone removes
    // the map entry), but a timeline that is tombstoned AND distilled
    // deliberately escapes the wholesale drop below, so the same set is checked
    // here rather than assumed.
    let snapshot_drop: std::collections::HashSet<u64> = distilled
        .keys()
        .chain(tombstoned.iter())
        .map(|&t| snapshot_stream_id(t).0)
        .collect();
    for (stream_id, loc) in substrate.recurrent_snapshot_entries() {
        if snapshot_drop.contains(&stream_id.0) {
            continue;
        }
        out.push(CompactItem::raw(
            RecordHeader {
                record_type: RecordType::Snapshot,
                format: 0,
                payload_len: loc.payload_len,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            },
            loc.segment,
            loc.offset,
            loc.record_size,
        ));
    }
    // A stream whose current record is a retraction keeps it: the retraction
    // is the value, and it is what refuses an older snapshot arriving later.
    for (stream_id, id) in substrate.retracted_snapshots() {
        if snapshot_drop.contains(&stream_id.0) {
            continue;
        }
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Snapshot,
                format: 0,
                payload_len: 0,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            Vec::new(),
        ));
    }

    // Prompt branch checkpoints, **capped**.
    //
    // The only record here that is a cache rather than durable state: a
    // checkpoint is a pure function of a prompt that is still on disk, so
    // dropping one costs a single prefill on next use, and keeping one costs
    // ~63 MiB in every future segment. Nothing else reclaims them — they are
    // keyed by content, so a prompt edit supersedes nothing and orphans the old
    // branch, and no timeline tombstone names them. Without this the log grows
    // by one orphan per prompt edit, forever.
    //
    // Newest-first by write position, which is the closest thing to
    // recently-used the compactor can see. A workspace exercises few branches at
    // a time, so the cap is generous relative to real use and still bounds the
    // worst case (the live schema's tree has 200) at a fixed size.
    let kept = branch_checkpoints_to_keep(substrate);
    for (stream_id, loc) in kept {
        out.push(CompactItem::raw(
            RecordHeader {
                record_type: RecordType::BranchCheckpoint,
                format: 0,
                payload_len: loc.payload_len,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            },
            loc.segment,
            loc.offset,
            loc.record_size,
        ));
    }

    // Per-stream live records — sourced from the substrate's in-RAM
    // stream index, the authoritative source. Every writer's stream is
    // carried: two members' turn N share a stream id and are both kept.
    //
    // In a fixed order — each timeline's turns in turn order, then every
    // other stream by id — so two compactions of one store write the same
    // file, and an adoption (`super::adoption`), which stamps records in the
    // order they are written, stamps a timeline's declarations rising with
    // the index they were written at: every turn keeps its position.
    let mut streams: Vec<_> = substrate.all_physical_streams().collect();
    streams.sort_by_key(|(key, entry)| match &entry.decl {
        Some(StreamDecl::Turn(t)) => (0, t.timeline_id, t.turn_index, *key),
        _ => (1, 0, 0, *key),
    });
    for (key, entry) in streams {
        let stream_id = key.stream_id;
        // Distillation degree for this turn's timeline (None = not distilled).
        // Both modes drop the KV chunks (the bulk of the on-disk cost);
        // ProvenanceOnly also drops tokens (keep sig + projections — the belief
        // gallery reads only the sig), TextOnly also drops sig + projections
        // (keep tokens — a plain read-only text record). Tombstoned timelines
        // drop entirely, as before.
        // An ORPHAN stream — no StreamDecl, and none coming — cannot have its
        // turn/section reconstructed on reload (reload rebuilds turns from
        // their decl), so every record it still owns — chunks, tokens, sig,
        // projections — is unreachable dead weight. Drop the whole stream so
        // compaction reclaims the orphaned KV instead of copying it forward on
        // every pass forever. Without this gate a decl lost in a prior
        // generation leaves its (huge) KV chunks immortal on disk. A stream
        // another origin wrote is not an orphan for want of a decl: its decl
        // is still on its way (`StreamRuntime::is_orphan`).
        if entry.is_orphan(node) {
            continue;
        }
        let mut turn_dead = false;
        let distill: Option<DistillMode> = if let Some(StreamDecl::Turn(t)) = &entry.decl {
            // A tombstoned timeline's records go wholesale — UNLESS it is
            // also distilled. That pair is the provenance corpus: calibration
            // exemplars are archived, distilled `ProvenanceOnly`, and then
            // tombstoned so they leave the live gather, while their
            // `WideQSig`s and projection events keep answering the belief
            // scan. Dropping them here erased the very records the tombstone
            // exists to preserve, and — because the `Distilled` marker went
            // with them — the next reload saw a content-declaring turn with
            // no KV and no distill exemption, i.e. `TurnIntegrity::MissingKv`.
            // Integrity repair then tombstoned it "for corruption", the
            // calibration resume filter could no longer see it, and the case
            // regenerated: ~196 exemplars rebuilt per boot, forever.
            //
            // So distilled timelines fall through to the per-mode retention
            // below whether or not they are tombstoned; only an undistilled
            // tombstone is dropped outright.
            if tombstoned.contains(&t.timeline_id) && !distilled.contains_key(&t.timeline_id) {
                continue;
            }
            // A turn-scoped tombstone on a LIVE timeline sheds the turn's
            // bulk records (chunks/tokens/sig/projections) but keeps its
            // StreamDecl: the reload restores the turn as an empty
            // placeholder from the decl, preserving the timeline's turn
            // indexing (a fully-dropped decl would renumber every later
            // turn away from its persisted stream ids).
            turn_dead = substrate.is_turn_stream_retired(key, t);
            distilled.get(&t.timeline_id).copied()
        } else {
            None
        };
        let keep_chunks = !turn_dead && distill.is_none();
        let keep_tokens = !turn_dead && distill != Some(DistillMode::ProvenanceOnly);
        let keep_sig = !turn_dead && distill != Some(DistillMode::TextOnly);
        if let Some(decl) = &entry.decl {
            let payload = decl.encode();
            out.push(CompactItem::synth(
                RecordHeader {
                    record_type: RecordType::StreamDecl,
                    format: 0,
                    payload_len: payload.len() as u64,
                    crc: 0,
                    stream_id: stream_id.0,
                    chunk_index: 0,
                    token_count: 0,
                    ..RecordHeader::NONE
                }
                .with_identity(entry.ids.decl),
                payload,
            ));
        }
        if keep_chunks {
            for (&idx, loc) in &entry.chunks {
                out.push(CompactItem::raw(
                    RecordHeader {
                        record_type: RecordType::Chunk,
                        format: loc.format,
                        payload_len: loc.payload_len,
                        crc: 0,
                        stream_id: stream_id.0,
                        chunk_index: idx,
                        token_count: loc.token_count,
                        ..RecordHeader::NONE
                    },
                    loc.segment,
                    loc.offset,
                    loc.record_size,
                ));
            }
        }
        if keep_tokens {
            if let Some(loc) = entry.tokens {
                out.push(CompactItem::raw(
                    RecordHeader {
                        record_type: RecordType::Tokens,
                        format: 0,
                        payload_len: loc.payload_len,
                        crc: 0,
                        stream_id: stream_id.0,
                        chunk_index: 0,
                        token_count: 0,
                        ..RecordHeader::NONE
                    },
                    loc.segment,
                    loc.offset,
                    loc.record_size,
                ));
            }
        }
        // Per-turn projection-event timeline — re-emitted from the resident
        // bytes rather than read back from disk, so the GUI dots survive a
        // compaction pass. Dropped for TextOnly (`keep_sig` false).
        if keep_sig {
            if let Some(payload) = &entry.projection_events {
                out.push(CompactItem::synth(
                    RecordHeader {
                        record_type: RecordType::ProjectionEvents,
                        format: 0,
                        payload_len: payload.len() as u64,
                        crc: 0,
                        stream_id: stream_id.0,
                        chunk_index: 0,
                        token_count: 0,
                        ..RecordHeader::NONE
                    }
                    .with_identity(entry.ids.projection_events),
                    payload.clone(),
                ));
            }
        }
        // Per-turn wide-Q signature window — same resident-bytes re-emit so the
        // decode→decode consensus substrate survives a compaction pass. Dropped
        // for TextOnly (`keep_sig` false).
        if keep_sig {
            if let Some(payload) = &entry.wide_q_sigs {
                out.push(CompactItem::synth(
                    RecordHeader {
                        record_type: RecordType::WideQSig,
                        format: 0,
                        payload_len: payload.len() as u64,
                        crc: 0,
                        stream_id: stream_id.0,
                        chunk_index: 0,
                        token_count: 0,
                        ..RecordHeader::NONE
                    }
                    .with_identity(entry.ids.wide_q_sigs),
                    payload.clone(),
                ));
            }
        }
        // The turn's QSA index page, re-emitted with its K/V for the same
        // reason: a compaction that carries a turn's chunks forward but drops
        // its page leaves a turn that can be borrowed and cannot be indexed.
        // Unlike the signature this is not sig-gated — it belongs to the K/V,
        // and a `TextOnly` turn that keeps its chunks keeps its page.
        // A page held in RAM is written from its bytes; one left in the log is
        // carried by location, as the turn's K/V is.
        let page_header = |payload_len: u64| {
            RecordHeader {
                record_type: RecordType::TurnIndexPage,
                format: 0,
                payload_len,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(entry.ids.index_page)
        };
        match &entry.index_page {
            Some(IndexPage::Resident(payload)) => out.push(CompactItem::synth(
                page_header(payload.bytes().len() as u64),
                payload.shared_bytes(),
            )),
            Some(IndexPage::Stored(loc)) => out.push(CompactItem::raw(
                page_header(loc.payload_len),
                loc.segment,
                loc.offset,
                loc.record_size,
            )),
            None => {}
        }
        if let Some(through) = entry.committed_through {
            out.push(CompactItem::synth(
                RecordHeader {
                    record_type: RecordType::Commit,
                    format: 0,
                    payload_len: 0,
                    crc: 0,
                    stream_id: stream_id.0,
                    chunk_index: through,
                    token_count: 0,
                    ..RecordHeader::NONE
                }
                .with_identity(entry.ids.commit),
                Vec::new(),
            ));
        }
    }
    // Timeline tombstones are kept although their records vanish wholesale
    // above: a record replicated late — by a member that was cut off — must
    // still find the tombstone and die. Each stays until it is old and this is
    // not the first complete compaction to run past it, when it is dropped and
    // leaves nothing behind (a member away longer drops what it did not author
    // before it syncs, `docs/replicated_substrate.md` §7.8); this compaction
    // has run past every one it carries, and the `Deleted` record it writes
    // says so.
    let expired = expiry.expired(substrate, marker_ids);
    let mut dead: Vec<TimelineId> = substrate
        .tombstoned_timelines()
        .iter()
        .copied()
        .filter(|tl| !expired.timelines.contains(tl))
        .collect();
    dead.sort_unstable();
    let mut swept: Vec<u64> = Vec::new();
    for tl in dead {
        let payload = TombstonePayload {
            timeline_id: tl.raw(),
            turn_index: None,
            turn_writer: 0,
            reason: None,
        }
        .encode();
        let id = marker_ids.identity_for(RecordType::Tombstone, 0, &payload);
        swept.push(tl.raw());
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Tombstone,
                payload_len: payload.len() as u64,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // A keyed value's tombstone expires the same way: once a complete
    // compaction has run past it and it is old, it is dropped. The custom
    // objects below skip what expires here.
    let mut swept_keys: Vec<u64> = substrate
        .held_custom_objects()
        .into_iter()
        .filter(|(obj, _)| obj.ceased && !expired.keys.contains(&obj.stream_id()))
        .map(|(obj, _)| obj.stream_id())
        .collect();
    swept_keys.sort_unstable();
    if !swept.is_empty() || !swept_keys.is_empty() {
        let payload = DeletedPayload { swept, swept_keys }.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Deleted,
                payload_len: payload.len() as u64,
                ..RecordHeader::NONE
            },
            payload,
        ));
    }
    // Turn-scoped tombstones on LIVE timelines are re-emitted: the compacted
    // log keeps the dead turn's StreamDecl (index preservation) but sheds its
    // bulk records, and the reload's placeholder logic keys off this tombstone
    // to restore the turn as an empty hole instead of re-hitting the (now
    // absent) content.
    let mut marks: Vec<_> = substrate.tombstoned_turns().iter().copied().collect();
    marks.sort_unstable();
    for mark in marks {
        if tombstoned.contains(&mark.timeline.raw()) {
            continue;
        }
        let payload = TombstonePayload {
            timeline_id: mark.timeline.raw(),
            turn_index: Some(mark.turn_index),
            turn_writer: mark.writer,
            reason: None,
        }
        .encode();
        let id = marker_ids.identity_for(RecordType::Tombstone, 0, &payload);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Tombstone,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Tombstoned section streams — re-emitted the same way turn-scoped
    // tombstones are: the marker itself is what makes the drop permanent
    // across reloads, so it must survive the rewrite that drops the
    // section's now-dead `Chunk` / `Tokens` / `StreamDecl` records (those
    // simply never appear in the live set above, since the tombstone took them
    // out of the index). Losing this marker would resurrect the corrupted
    // chunks on the next boot that restores from an older segment still
    // holding them. One per ended generation, naming it; one per unstamped
    // stream a tombstone written before generations ended.
    let ended = substrate
        .ended_section_generations()
        .map(|(sid, g)| (sid, Some(g)))
        .chain(substrate.ended_unstamped_sections().map(|sid| (sid, None)));
    for (stream_id, generation) in ended {
        let payload = SectionTombstonePayload {
            generation,
            reason: None,
        }
        .encode();
        let id = marker_ids.identity_for(RecordType::SectionTombstone, stream_id.0, &payload);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::SectionTombstone,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: stream_id.0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-timeline Label / ConvState records — synthesised from
    // the substrate's live state, one `Label` per record that holds a field.
    // Tombstoned timelines are skipped so retired conversations don't leave
    // dangling sidebar entries on disk.
    for (timeline_id, meta, id) in substrate.live_label_records() {
        // A tombstoned timeline that is ALSO distilled is the provenance corpus
        // (calibration exemplars: archived, distilled, then tombstoned out of the
        // live gather while their signatures keep answering the belief scan).
        // Its marker/metadata must survive, or the next reload sees a
        // content-declaring turn with no distill exemption and condemns it.
        // Only an undistilled tombstone is a retired conversation.
        if tombstoned.contains(&timeline_id) && !distilled.contains_key(&timeline_id) {
            continue;
        }
        let payload = super::super::manifest::encode_label_payload(
            timeline_id,
            &meta.conv_id,
            &meta.label,
            &meta.custom,
        );
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Label,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-timeline conversation state — the whole state, archived flag and
    // branches alike, keyed in the header by timeline like every other
    // `ConvState` record. Same retirement rule as the labels above.
    for (timeline_id, state, id) in substrate.live_conv_states() {
        if tombstoned.contains(&timeline_id) && !distilled.contains_key(&timeline_id) {
            continue;
        }
        let payload = encode_conv_state_payload(timeline_id, &state);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::ConvState,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: timeline_id,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-(timeline, turn) summary-tree metadata — emit one record
    // per live tree node directly from substrate state.
    for (payload, id) in substrate.live_tree_metadata_payloads() {
        if tombstoned.contains(&payload.timeline_id) {
            continue;
        }
        let bytes = payload.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::TreeMetadata,
                format: 0,
                payload_len: bytes.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            bytes,
        ));
    }
    // Records about a timeline not registered yet, held until its first turn
    // declaration arrives. They are durable state like any other: dropping one
    // here would lose it for good, since its `oseq` was already applied.
    for (timeline_id, rt, payload, id) in substrate.pending_records() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let stream_id = if rt == RecordType::ConvState {
            timeline_id
        } else {
            0
        };
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: rt,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-(timeline, turn) tool round-trip couplings.
    //
    // Resident and payload-keyed, like `Label` and `DebugId` — nothing
    // supersedes an old copy and `accounting` never credits one as dead, so the
    // only way one reaches the compacted log is a re-emit from RAM. Dropping it
    // deletes the only statement that a call turn and its response are one
    // exchange: `summary_tree::exchange` groups an exchange as the maximal run
    // of coupled turns, so every compaction began splitting the two halves of a
    // round-trip apart. Nothing failed — a timeline with no couplings is a
    // legitimate state (nothing ever called a tool there) and reads as intact,
    // which is exactly why this went unnoticed.
    for (timeline_id, from_turn, id) in substrate.live_couplings() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let payload = TurnCouplingPayload {
            timeline_id,
            from_turn,
        }
        .encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::TurnCoupling,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Per-timeline debug_id.
    for (timeline_id, debug_id, id) in substrate.live_debug_ids() {
        if tombstoned.contains(&timeline_id) {
            continue;
        }
        let payload = DebugIdPayload {
            timeline_id,
            debug_id,
        };
        let bytes = payload.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::DebugId,
                format: 0,
                payload_len: bytes.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            bytes,
        ));
    }
    // Generic keyed custom objects — re-emitted from the held in-RAM set so
    // the winning object outlives the compaction pass, keyed on the same
    // stable key-hash `stream_id` it was written under; a tombstone among
    // them is carried until it expired above.
    for (obj, id) in substrate.held_custom_objects() {
        if expired.keys.contains(&obj.stream_id()) {
            continue;
        }
        let bytes = obj.encode();
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::CustomObject,
                format: 0,
                payload_len: bytes.len() as u64,
                crc: 0,
                stream_id: obj.stream_id(),
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            bytes,
        ));
    }
    // Per-distilled-timeline marker — re-emitted (with mode) so a distilled
    // timeline stays distilled across the compaction it just shed through.
    // Without this the marker is lost and, e.g., a text-only archived
    // conversation would read back as un-distilled.
    //
    // Re-emitted for TOMBSTONED timelines too. The marker is what exempts a
    // distilled turn from the `MissingKv` integrity verdict (`classify_turn`),
    // and a distilled turn has legitimately shed its KV — so losing the marker
    // condemns the very corpus the distillation created. The provenance corpus
    // is tombstoned by design (out of the live gather, still answering the
    // belief scan by signature), so "tombstoned" cannot mean "forget it was
    // distilled".
    for (&timeline_id, &mode) in &distilled {
        let payload = DistillPayload { timeline_id, mode }.encode();
        let id = marker_ids.identity_for(RecordType::Distilled, 0, &payload);
        out.push(CompactItem::synth(
            RecordHeader {
                record_type: RecordType::Distilled,
                format: 0,
                payload_len: payload.len() as u64,
                crc: 0,
                stream_id: 0,
                chunk_index: 0,
                token_count: 0,
                ..RecordHeader::NONE
            }
            .with_identity(id),
            payload,
        ));
    }
    // Every re-encoded record above kept the identity of the record that holds
    // its value now — the same origin and `oseq`, so every member still
    // recognises it, and the same stamp, so it still ranks where it did. A
    // read-back record keeps its own: `write_compacted_log` indexes it from its
    // copied header.
    out
}
