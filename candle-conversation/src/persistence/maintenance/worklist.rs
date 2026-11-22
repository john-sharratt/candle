//! Lists the records each op must relocate, read from the store's location maps.
//!
//! Read-back records are found through the index. Characters, file events,
//! blobs and index pages are found through the store's own location maps, since
//! the substrate does not hold their bytes. The same rules decide what counts
//! toward a segment's liveness, so a record carried forward is always counted
//! live. Orphans and retired turns are excluded, and the lists are sorted so an
//! identical state plans identically.

use super::*;

impl LiveIndexes {
    /// Character records physically inside a target segment.
    ///
    /// Read from this store's own `npc_locs` rather than the substrate, which
    /// has never held a character: their payloads belong to the daemon above.
    /// That is also why they are relocated rather than re-emitted — there is
    /// nothing in RAM here to re-encode them from.
    ///
    /// `npc_locs` is last-writer-wins and read under the persistence lock, so an
    /// entry naming a target segment is that character's live record by
    /// definition — no supersession filter is needed the way `Snapshot` needs
    /// one. Sorted by id so a pass over identical state produces an identical
    /// file.
    pub(in super::super) fn npc_relocations(
        &self,
        targets: &[SegmentId],
    ) -> Vec<(StreamId, RecordLoc)> {
        let mut out: Vec<(StreamId, RecordLoc)> = self
            .npc_locs
            .iter()
            .filter(|(_, loc)| targets.contains(&loc.segment))
            .map(|(&id, &loc)| (StreamId(id), loc))
            .collect();
        // Deterministic, so the same store plans the same work twice running.
        out.sort_unstable_by_key(|(sid, _)| sid.0);
        out
    }

    /// The file events and tombstones of live conversations physically inside
    /// a target segment — by the one rule compaction and liveness read
    /// ([`vfs_carried`]). In timeline and sequence order, so the same store plans
    /// the same work twice running.
    pub(in super::super) fn vfs_relocations(
        &self,
        substrate: &Substrate,
        targets: &[SegmentId],
    ) -> Vec<VfsReloc> {
        let mut out = Vec::new();
        for (timeline, tl) in vfs_carried(substrate, &self.vfs_index) {
            for (rt, at) in [
                (RecordType::VfsEvent, tl.events()),
                (RecordType::VfsTombstone, tl.tombstones()),
            ] {
                for (&id, &loc) in at {
                    if targets.contains(&loc.segment) {
                        out.push(VfsReloc {
                            rt,
                            timeline,
                            id,
                            loc,
                        });
                    }
                }
            }
        }
        out
    }

    /// The blobs physically inside a target segment, in key order.
    pub(in super::super) fn blob_relocations(
        &self,
        targets: &[SegmentId],
    ) -> Vec<(BlobKey, RecordLoc)> {
        self.blob_index
            .blobs()
            .filter(|(_, loc)| targets.contains(&loc.segment))
            .collect()
    }

    /// Gather the relocation worklist for `targets` — the live read-back records
    /// (`Chunk` / `Tokens` / singletons) physically in those segments, with the
    /// same distill/tombstone filter as [`Self::segment_liveness`]. Read-only.
    #[allow(clippy::type_complexity)]
    pub(in super::super) fn gather_relocations(
        &self,
        substrate: &Substrate,
        targets: &[SegmentId],
    ) -> (
        Vec<(StreamKey, u64, ChunkLoc)>,
        Vec<(StreamKey, RecordLoc)>,
        Vec<(StreamId, RecordLoc)>,
        Vec<(StreamId, RecordLoc)>,
        Vec<(RecordType, RecordLoc)>,
    ) {
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
        let in_target = |seg: SegmentId| targets.contains(&seg);
        let node = self.node();

        let mut chunks: Vec<(StreamKey, u64, ChunkLoc)> = Vec::new();
        let mut tokens: Vec<(StreamKey, RecordLoc)> = Vec::new();
        for (key, entry) in substrate.all_physical_streams() {
            let shed = classify(substrate, key, entry, &tombstoned, &distilled);
            // Same rule as the re-emit path: a tombstoned-but-DISTILLED timeline
            // is the provenance corpus, so it is relocated by its mode (the
            // per-mode gates below already withhold its chunks/tokens) rather
            // than abandoned. Only an undistilled tombstone is skipped outright.
            // A tombstoned section is never distilled, so it always takes this
            // branch — its corrupted chunks are never relocated forward.
            if shed.dropped_wholesale() {
                continue;
            }
            // Orphan (no decl, and none coming): its records are unreachable on
            // reload, so DON'T relocate them — segment_liveness excludes them
            // from live weight, so relocating (instead of dropping) would move
            // the orphan KV forward every compact while the dead-ratio never
            // improves: perpetual churn and the bloat is never reclaimed.
            // Dropping the source segment reclaims it. Mirrors the
            // collect_live_records / segment_liveness gate.
            if entry.is_orphan(node) {
                continue;
            }
            // **A retired turn is not relocated.** Moving it forward would carry
            // the bulk into the new segment while the dead ratio never improves
            // — perpetual churn, and the space never reclaimed. That is exactly
            // the reasoning the orphan gate above states, for the same reason.
            if shed.keep_chunks() {
                for (&idx, loc) in &entry.chunks {
                    if in_target(loc.segment) {
                        chunks.push((key, idx, *loc));
                    }
                }
            }
            if shed.keep_tokens() {
                if let Some(loc) = entry.tokens {
                    if in_target(loc.segment) {
                        tokens.push((key, loc));
                    }
                }
            }
        }
        // Live snapshot tails physically inside a target segment. No
        // tombstone/distill gate: the substrate map only ever holds live
        // conversations' tails. Execute re-validates each entry against the
        // persistence-side `snapshot_locs` right before relocating, so a tail
        // superseded (or tombstoned) after this plan is skipped, not copied.
        let mut snapshots: Vec<(StreamId, RecordLoc)> = Vec::new();
        for (sid, loc) in substrate.recurrent_snapshot_entries() {
            if in_target(loc.segment) {
                snapshots.push((sid, loc));
            }
        }
        // Branch checkpoints relocate too — maintenance moves records out of a
        // segment it is about to drop, and a checkpoint left behind would leave
        // the index pointing at bytes that are gone. The compaction *cap* is a
        // separate concern: this pass preserves, `collect_live_records` bounds.
        let mut branch_checkpoints: Vec<(StreamId, RecordLoc)> = Vec::new();
        for (sid, loc) in substrate.branch_checkpoint_entries() {
            if in_target(loc.segment) {
                branch_checkpoints.push((sid, loc));
            }
        }
        let mut singletons: Vec<(RecordType, RecordLoc)> = Vec::new();
        for (rt, loc) in [
            (RecordType::ModelKey, self.manifest.model_key),
            (RecordType::Template, self.manifest.template),
            (RecordType::Tokenizer, self.manifest.tokenizer),
        ] {
            if let Some(loc) = loc {
                if in_target(loc.segment) {
                    singletons.push((rt, loc));
                }
            }
        }
        (chunks, tokens, snapshots, branch_checkpoints, singletons)
    }

    /// Located per-stream records — QSA index pages, signature windows and
    /// projection events left in the log ([`IndexPage::Stored`],
    /// [`Located::Stored`]) — physically inside a target segment: relocated by
    /// location, as a turn's tokens are, since their bytes are not in RAM to
    /// re-emit. The resident set's gates: nothing of a stream dropped wholesale
    /// or an orphan, and these only where the stream keeps its signature
    /// records.
    pub(in super::super) fn gather_located_relocs(
        &self,
        substrate: &Substrate,
        targets: &[SegmentId],
    ) -> Vec<(StreamKey, RecordType, RecordLoc)> {
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
        let node = self.node();
        let mut out = Vec::new();
        for (key, entry) in substrate.all_physical_streams() {
            let stored = [
                (
                    RecordType::TurnIndexPage,
                    entry.index_page.as_ref().and_then(IndexPage::stored),
                ),
                (
                    RecordType::WideQSig,
                    entry.wide_q_sigs.as_ref().and_then(Located::stored),
                ),
                (
                    RecordType::ProjectionEvents,
                    entry.projection_events.as_ref().and_then(Located::stored),
                ),
            ];
            let in_target: Vec<(RecordType, RecordLoc)> = stored
                .into_iter()
                .filter_map(|(rt, loc)| loc.map(|l| (rt, l)))
                .filter(|(_, l)| targets.contains(&l.segment))
                .collect();
            if in_target.is_empty() {
                continue;
            }
            let shed = classify(substrate, key, entry, &tombstoned, &distilled);
            if shed.dropped_wholesale() || entry.is_orphan(node) || !shed.keep_sig() {
                continue;
            }
            out.extend(in_target.into_iter().map(|(rt, l)| (key, rt, l)));
        }
        out
    }

    /// Live record bytes per segment — every record
    /// [`counted_live_records`](Self::counted_live_records) counts: the
    /// read-back records the substrate index still references and the current
    /// copy of each carried resident record, with the same distill/tombstone
    /// filter [`super::compaction::collect_live_records`] applies.
    ///
    /// Summed as the records are visited rather than gathered first: a store
    /// of millions of live records spent most of the plan's substrate read
    /// hold — 520 ms of 1.1 s on BlackWorld (2026-10-27), every 15 s —
    /// building the list only to add it up.
    pub fn segment_liveness(&self, substrate: &Substrate) -> HashMap<SegmentId, u64> {
        let mut live: HashMap<SegmentId, u64> = HashMap::new();
        self.visit_counted_live_records(substrate, |r| {
            *live.entry(r.segment).or_default() += r.size;
        });
        live
    }

    /// The current per-stream metadata records maintenance carries forward —
    /// what a segment's liveness counts, and so exactly what a drop must not
    /// lose. `carried` maps each carried stream to whether its signature
    /// records (`ProjectionEvents` / `WideQSig` / `TurnIndexPage`) go with it;
    /// the decl and commit always do.
    ///
    /// `CustomObject` is always carried. It is keyed by a content hash of its
    /// key, not by a stream, so it has no `StreamDecl` lifecycle for `carried`
    /// to consult — it is host-defined resident state like the cast in
    /// `npc_locs`. Gating it on `carried` (which never holds a key hash) would
    /// count its segment dead and let a drop take its only current copy. The
    /// `Deleted`, `Replica` and `Rebase` records are always carried for the
    /// same reason: each belongs to no stream, and is the only record of what
    /// it holds.
    pub(super) fn carried_metadata<'a>(
        &'a self,
        carried: &'a CarriedStreams,
    ) -> impl Iterator<Item = (RecordType, &'a RecordLoc)> + 'a {
        self.metadata_locs
            .iter()
            .filter_map(move |(&(rt, stream_id, writer), loc)| {
                if matches!(
                    rt,
                    RecordType::CustomObject
                        | RecordType::Deleted
                        | RecordType::Replica
                        | RecordType::Rebase
                ) {
                    return Some((rt, loc));
                }
                let keep_sig = *carried.get(&(stream_id, writer))?;
                let is_signature = matches!(
                    rt,
                    RecordType::ProjectionEvents | RecordType::WideQSig | RecordType::TurnIndexPage
                );
                (!is_signature || keep_sig).then_some((rt, loc))
            })
    }

    /// Every record [`segment_liveness`](Self::segment_liveness) counts live,
    /// with its location and type, plus the carry map it was counted by. The one
    /// source of the count: liveness sums it, and the liveness audit compares it
    /// record by record against what maintenance carries.
    pub(in super::super) fn counted_live_records(
        &self,
        substrate: &Substrate,
    ) -> (Vec<CountedRecord>, CarriedStreams) {
        let mut live: Vec<CountedRecord> = Vec::new();
        let carried = self.visit_counted_live_records(substrate, |r| live.push(r));
        (live, carried)
    }

    /// [`Self::counted_live_records`], each record handed to `visit` as it is
    /// counted; answers the carry map.
    pub(super) fn visit_counted_live_records(
        &self,
        substrate: &Substrate,
        mut visit: impl FnMut(CountedRecord),
    ) -> CarriedStreams {
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

        for (rt, loc) in [
            (RecordType::ModelKey, self.manifest.model_key),
            (RecordType::Template, self.manifest.template),
            (RecordType::Tokenizer, self.manifest.tokenizer),
        ] {
            if let Some(loc) = loc {
                visit(CountedRecord::at(rt, &loc));
            }
        }
        // Streams maintenance carries forward, mapped to whether their signature
        // records (`ProjectionEvents` / `WideQSig` / `TurnIndexPage`) go with
        // them — the exact rule `gather_resident_set` and `gather_relocations`
        // carry by. Every record counted live here must be one maintenance
        // carries, and every record it carries must be counted live: a record
        // carried but counted dead makes the segment it lands in look
        // reclaimable, so the next pass carries it again, forever.
        //
        // That is not hypothetical. A tombstoned-but-distilled timeline — the
        // calibration corpus, some thousands of turns — is carried by its
        // distill mode, signatures included, and was counted wholly dead here.
        // Every pass re-emitted the corpus's signatures into a fresh segment,
        // which then read ≥10% dead and was compacted a minute later: ~4 GB
        // rewritten every ~75 s, indefinitely.
        //
        // A stream missing from the map carries nothing: a timeline dropped
        // wholesale (tombstoned and undistilled), or an orphan with no
        // `StreamDecl` and none coming (its decl was lost in a prior
        // generation, so its turn can never be rebuilt on reload — mirroring
        // the `collect_live_records` orphan gate — and the segments holding
        // only its records become reclaimable instead of pinned forever).
        let mut carried: CarriedStreams = HashMap::new();
        let node = self.node();
        for (key, entry) in substrate.all_physical_streams() {
            let shed = classify(substrate, key, entry, &tombstoned, &distilled);
            if shed.dropped_wholesale() || entry.is_orphan(node) {
                continue;
            }
            // A turn-dead stream keeps its `StreamDecl` — the compactor emits it
            // as a placeholder so the timeline's turn indexing survives — so it
            // stays a carried stream here and only its bulk stops counting.
            carried.insert((key.stream_id.0, key.writer), shed.keep_sig());
            if shed.keep_chunks() {
                for loc in entry.chunks.values() {
                    visit(CountedRecord {
                        record_type: RecordType::Chunk,
                        segment: loc.segment,
                        offset: loc.offset,
                        size: loc.record_size,
                    });
                }
            }
            if shed.keep_tokens() {
                if let Some(loc) = entry.tokens {
                    visit(CountedRecord::at(RecordType::Tokens, &loc));
                }
            }
        }
        // Per-stream metadata records (`StreamDecl` / `ProjectionEvents` /
        // `WideQSig` / `TurnIndexPage` / `Commit`) carry no location in the
        // substrate index, so they are counted from the persistence-side
        // `metadata_locs` map instead — last-writer-wins, so only the CURRENT
        // copy of each is here; superseded copies are absent and correctly read
        // as dead. Counted by the carry rule above: the decl and commit always,
        // the signature records only when the stream keeps its signature.
        for (rt, loc) in self.carried_metadata(&carried) {
            visit(CountedRecord::at(rt, loc));
        }
        // The payload-keyed resident records, current copy each — counted by
        // exactly the gates `gather_resident_set` re-emits them by, so what is
        // counted live is what an op carries. A tombstoned, undistilled
        // timeline's label and state are not re-emitted; neither are a
        // tombstoned timeline's summary nodes, couplings or debug id, or its
        // turn tombstones (the timeline's own tombstone covers them).
        // A `Label` record is live while the substrate holds a field it won
        // (or it is the record that first named the conversation) — exactly the
        // records `live_label_records` re-emits. One that won nothing, and a
        // timeline whose metadata is all default, re-emit nothing, so they are
        // dead.
        let label_records: HashSet<(u64, u64, u64)> = substrate
            .live_label_records()
            .into_iter()
            .map(|(tl, _, id)| (tl, id.origin, id.oseq))
            .collect();
        // A section tombstone is live while the substrate holds the generation
        // it ended — keyed as `resident_key` keys it, `(section, oseq,
        // writer)`, and `(section, 0, 0)` for one written before generations.
        let ended_sections: HashSet<(u64, u64, u64)> = substrate
            .ended_section_generations()
            .map(|(sid, g)| (sid.0, g.oseq, g.writer))
            .chain(
                substrate
                    .ended_unstamped_sections()
                    .map(|sid| (sid.0, 0, 0)),
            )
            .collect();
        for (key, loc) in self.resident_locs.iter() {
            let tl = key.id;
            let gone = tombstoned.contains(&tl);
            let kept = !gone || distilled.contains_key(&tl);
            let carried_record = match key.record_type {
                RecordType::Label => kept && label_records.contains(&(tl, key.origin, key.sub)),
                RecordType::ConvState => kept,
                RecordType::TreeMetadata | RecordType::TurnCoupling | RecordType::DebugId => !gone,
                RecordType::Tombstone => key.sub == 0 || !gone,
                RecordType::Distilled => distilled.contains_key(&tl),
                RecordType::SectionTombstone => {
                    ended_sections.contains(&(key.id, key.sub, key.origin))
                }
                _ => false,
            };
            if carried_record {
                visit(CountedRecord::at(key.record_type, loc));
            }
        }
        // Recurrent-state snapshots: one live tail per conversation, tracked in
        // the substrate index (the `Tokens` shape — no `StreamDecl` of their
        // own, so the live_streams gate does not apply). The map holds only
        // live conversations' tails — a timeline tombstone removes its entry —
        // so every entry counts.
        for (_sid, loc) in substrate.recurrent_snapshot_entries() {
            visit(CountedRecord::at(RecordType::Snapshot, &loc));
        }
        // Prompt-branch checkpoints, by the same rule: one live record per
        // branch, tracked in the substrate index, and carried forward by
        // `gather_relocations`. Each is a whole recurrent state — ~112 MiB on
        // Flash-Next — so leaving them uncounted read every segment holding one
        // as that much dead, and carried them forward again on every pass.
        for (_sid, loc) in substrate.branch_checkpoint_entries() {
            visit(CountedRecord::at(RecordType::BranchCheckpoint, &loc));
        }
        // Characters. Tracked persistence-side rather than on the substrate
        // index — the substrate holds no opinion about an NPC — and the map is
        // last-writer-wins, so only each character's current record is here and
        // every superseded one correctly reads as dead. Counting these is what
        // stops the segment holding the live cast from looking reclaimable:
        // without it the dead ratio is overstated, maintenance re-emits the
        // whole cast forward on every pass, and the log churns.
        for loc in self.npc_locs.values() {
            visit(CountedRecord::at(RecordType::Npc, loc));
        }
        // A live conversation's file events and tombstones — exactly what
        // `vfs_relocations` carries, so a segment holding only a retired
        // conversation's, or only events a tombstone killed, reads as dead.
        for (_, tl) in vfs_carried(substrate, &self.vfs_index) {
            for loc in tl.events().values() {
                visit(CountedRecord::at(RecordType::VfsEvent, loc));
            }
            for loc in tl.tombstones().values() {
                visit(CountedRecord::at(RecordType::VfsTombstone, loc));
            }
        }
        // Blobs — exactly what `blob_relocations` carries: a retired one is
        // not in the index, so its segment reads as that much dead.
        for (_, loc) in self.blob_index.blobs() {
            visit(CountedRecord::at(RecordType::Blob, &loc));
        }
        carried
    }
}

/// One record [`SubstratePersistence::counted_live_records`] counts live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountedRecord {
    pub record_type: RecordType,
    pub segment: SegmentId,
    pub offset: u64,
    /// Padded on-disk size.
    pub size: u64,
}

impl CountedRecord {
    pub(super) fn at(record_type: RecordType, loc: &RecordLoc) -> Self {
        Self {
            record_type,
            segment: loc.segment,
            offset: loc.offset,
            size: loc.record_size,
        }
    }
}
