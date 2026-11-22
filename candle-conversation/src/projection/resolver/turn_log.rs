//! Records turns, their declarations and summary nodes into the substrate and the
//! redo log, in one step.
//!
//! A turn and its declaration land in one edit, so a reader sees the turn with its
//! tags and layout, or neither. An adopted scope turn is re-recorded onto the file
//! timeline by reference to the same chunks, with its layout kept so its seams stay
//! live glue. A gap is held at any position a lost declaration leaves, so later
//! turns keep the stream ids they are keyed by.

use super::*;

impl Conversation {
    /// A turn's role as its `TurnDecl` carries it.
    pub(super) fn decl_role(role: Role) -> u8 {
        match role {
            Role::System => 0,
            Role::User => 1,
            Role::Assistant => 2,
        }
    }

    /// Atomically append a turn to the substrate.
    ///
    /// `write` carries the turn's text, token IDs, block range, and
    /// GPU-resident sealed K/V snapshot.  `migrate_to_cpu` is called
    /// to move the bytes to the warm (CPU) tier before storing.
    pub fn record_turn(
        &self,
        timeline: TimelineId,
        role: Role,
        write: TurnPartWrite,
        migrate_to_cpu: impl FnMut(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
    ) -> candle::Result<TurnIndex> {
        let block_start = write.block_start;
        let block_end = write.block_end;
        // Capture the segment layout and gather-scope tags before the write
        // moves into the substrate — the redo-log `TurnDecl` carries them
        // verbatim so reload can reconstruct `TurnPart::layout` (per-half text +
        // spans + `/no_think`) without re-tokenising, and re-tag the turn for the
        // provenance gallery's `tags:` scoping.
        let segments = write.layout.segments.clone();
        let tags = write.tags.clone();
        // Migrate BEFORE the edit: it moves this turn's K/V off the device,
        // and an edit may run more than once.
        let sealed_cpu = Substrate::migrate_sealed(&write, migrate_to_cpu)?;
        let identity = self.minter.mint(0);
        let role = Self::decl_role(role);
        // **The turn and its declaration land in one edit** (R1): a reader sees
        // the turn with its declaration, tags and layout, or neither.
        let (idx, decl) = self.update(|s| {
            let idx = s.append_migrated(timeline, write.clone(), sealed_cpu.clone())?;
            let decl = Self::turn_decl(
                s,
                timeline,
                idx,
                role,
                (block_start, block_end),
                &segments,
                &tags,
            );
            s.apply_stream_decl(decl.stream_id(), decl.clone(), identity);
            Ok::<_, candle::Error>((idx, decl))
        })?;
        self.persistence.enqueue(WriteJob::StreamDecl {
            stream_id: decl.stream_id().0,
            payload: decl.encode(),
            identity,
        });
        Ok(idx)
    }

    /// The declaration of turn `idx` of `timeline` — the turn's structure as
    /// the redo log records it, so reload can reconstruct `TurnPart::layout`
    /// (per-half text + spans + `/no_think`) without re-tokenising, and re-tag
    /// the turn for the provenance gallery's `tags:` scoping.
    pub(super) fn turn_decl(
        s: &Substrate,
        timeline: TimelineId,
        idx: TurnIndex,
        role: u8,
        (block_start, block_end): (u64, u64),
        segments: &[TurnSegment],
        tags: &[String],
    ) -> StreamDecl {
        let (layer_id, group_id) = s
            .timeline_target(timeline)
            .map(|(l, g)| (l.raw(), g.raw()))
            .unwrap_or((0, 0));
        StreamDecl::Turn(TurnDecl {
            timeline_id: timeline.raw(),
            turn_index: idx.0,
            turn_id_day: 0,
            turn_id_seq: idx.0 + 1,
            role,
            block_start,
            block_end,
            layer_id,
            group_id,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            segments: segments.to_vec(),
            tags: tags.to_vec(),
        })
    }

    /// Ordered-merge splice for parallel code_read: re-record a turn from
    /// `from_tl` (a per-scope fork timeline, sealed by the proven two-turn
    /// round-trip) onto `to_tl` (the file timeline) at the next index, fully
    /// persisted. The forked turn's K/V is referenced by the SAME
    /// content-addressed chunks (no copy — the sealed sequences clone their
    /// `ChunkGid` RAII handles, so the chunks outlive the dropped fork), and its
    /// layout + block range are preserved verbatim so the inter-turn seams stay
    /// LIVE GLUE (the correctness property code_read requires — a baked seam goes
    /// stale on re-injection). `tags` are supplied by the caller (the scope's
    /// `["code", <path>]`). The source turn must be HOT, which its fork
    /// guarantees by being a splice source ([`Self::mark_timeline_splice_source`]):
    /// no eviction path drops a splice source's hot copy before its tombstone.
    ///
    /// Persists four records for the adopted turn: the `TurnDecl` (via
    /// [`Self::record_turn`]), the token ids, the wide-Q provenance sigs
    /// (re-keyed onto the new stream so a provenance scan of the file timeline
    /// admits the turn), and the QSA index page (re-keyed so a projection that
    /// borrows the turn's K/V can also select it).
    pub fn adopt_turn(
        &self,
        from_tl: TimelineId,
        from_idx: TurnIndex,
        to_tl: TimelineId,
        role: Role,
        tags: Vec<String>,
    ) -> candle::Result<TurnIndex> {
        use crate::persistence::content_hash::turn_stream_id;
        let (layout, token_ids, block_range, sigs_blob) = {
            let r = self.read();
            let layout = r
                .turn_layout(from_tl, from_idx)
                .ok_or_else(|| candle::Error::Msg(format!("adopt_turn: {from_idx:?} missing")))?;
            let token_ids = r.token_ids_of(from_tl, from_idx);
            let block_range = r
                .turn_block_range(from_tl, from_idx)
                .ok_or_else(|| candle::Error::Msg("adopt_turn: no block range".into()))?;
            let sigs_blob = r
                .wide_q_sigs_blob(from_tl, from_idx)
                .map_err(|e| candle::Error::Msg(format!("adopt_turn: signature window: {e}")))?
                .map(|b| b.to_vec());
            (layout, token_ids, block_range, sigs_blob)
        };
        // The turn's index page after the lock: one left in the log is read
        // through the persistence path. Unreadable is an error, not a turn
        // adopted without its index rows.
        let index_page = self
            .turn_index_page(from_tl, from_idx)
            .map_err(|e| candle::Error::Msg(format!("adopt_turn: {e}")))?
            .map(|p| p.bytes().to_vec());
        let sealed = self
            .read()
            .turn_sealed_of(from_tl, from_idx)
            .ok_or_else(|| {
                candle::Error::Msg(
                    "adopt_turn: source K/V not hot — its timeline is not a splice \
                     source, or was already tombstoned"
                        .into(),
                )
            })?;
        let token_count = token_ids.len();
        let write = TurnPartWrite {
            layout,
            token_ids: TokenBuffer::from(token_ids.clone()),
            token_count,
            block_start: block_range.0,
            block_end: block_range.1,
            sealed_gpu: Some(sealed),
            tags,
        };
        let idx = self.record_turn(to_tl, role, write, |seqs| Ok(seqs.to_vec()))?;
        // Self-evict this code_read scope turn: it's never re-attended until query
        // time, and the file timeline is NOT in the scheduler's `ingest_timelines`,
        // so the gentle demote never covers it. Flag it now — while still hot,
        // before the hot→warm migrate — so the migrate drops its Q-format hot the
        // instant warm lands (`install_warm_and_evict_hot`) instead of leaving a
        // resident Q copy piling up on the file timeline (the quant-KV climb). This
        // is targeted to the code_read splice path (unlike a `summarize`-based
        // marker, which `--disable-summariser` would spuriously set on dialogue).
        // The source fork's turns are freed by `tombstone_timeline` at splice.
        self.read().mark_turn_evict_when_cold(to_tl, idx);
        let stream_id = turn_stream_id(to_tl.raw(), idx.0);
        // Queued to the log's owner, so the parallel code_read splice never
        // waits on the disk.
        self.enqueue_tokens(stream_id, token_ids.clone());
        if let Some(blob) = sigs_blob {
            self.enqueue_wide_q_sigs(stream_id, blob);
        }
        // The QSA index page is keyed by stream exactly like the sigs, and the
        // fork is tombstoned right after the splice — a page left under the
        // fork's stream is unreachable, and every projection that borrows this
        // turn then holds K/V its attention layers cannot select. A source that
        // sealed no page adopts none; the projection says so when it injects.
        if let Some(page) = index_page {
            self.enqueue_index_page(stream_id, page);
        }
        Ok(idx)
    }

    /// Declare a stream in the persistence redo log AND mirror the decl into
    /// the LIVE substrate. The reload walker only installs decls at startup,
    /// but live readers — the tag-scoped belief gallery reading a turn's tags,
    /// section-by-name lookups — consult the in-memory decls, so a stream
    /// declared in the current session (every calibration turn or section on
    /// a fresh substrate) must be visible to them without a restart.
    pub(super) fn declare_and_mirror(&self, decl: StreamDecl) -> StreamId {
        // The stream id is a PURE function of the decl (`turn_stream_id` /
        // `section_stream_id`), so compute it locally, apply the decl in-memory
        // synchronously (projection needs it this instant), and hand ONLY the
        // durable redo-log append to the log's owner. Reconstruct is
        // order-independent + per-turn fault-isolated, so the queued append is
        // crash-safe: worst case a decl lost on crash yields an absent turn,
        // never corruption.
        let stream_id = decl.stream_id();
        let payload = decl.encode();
        let identity = self.minter.mint(0);
        // A section's declaration starts its generation, which every record
        // this node writes for the section from here on carries.
        if matches!(decl, StreamDecl::PromptSection(_)) {
            self.minter.start_section(stream_id.0, identity.oseq);
        }
        self.update(|s| s.apply_stream_decl(stream_id, decl.clone(), identity));
        self.persistence.enqueue(WriteJob::StreamDecl {
            stream_id: stream_id.0,
            payload,
            identity,
        });
        stream_id
    }

    /// Union of the gather-scope tags on `children`'s TurnDecls, dedup'd
    /// with child order preserved. Empty when every child is untagged
    /// (dialogue turns), so summaries of untagged content stay in the
    /// untagged partition. Used by the summariser to stamp a summary node
    /// with the tags of the turns it compresses — a code_read leaf inherits
    /// its scan turn's `["code", <path>]`, a summary-of-summaries the union
    /// of its children's.
    pub fn union_turn_tags(&self, timeline: TimelineId, children: &[TurnIndex]) -> Vec<String> {
        use crate::persistence::content_hash::turn_stream_id;
        let read = self.inner.read();
        let mut tags: Vec<String> = Vec::new();
        for c in children {
            let sid = turn_stream_id(timeline.raw(), c.0);
            let Some(entry) = read.stream_of(sid) else {
                continue;
            };
            let Some(StreamDecl::Turn(d)) = &entry.decl else {
                continue;
            };
            for t in &d.tags {
                if !tags.contains(t) {
                    tags.push(t.clone());
                }
            }
        }
        tags
    }

    /// Append a summariser-allocated turn (SoT leaf or SoS internal)
    /// and persist its declaration to the redo log.
    ///
    /// The summariser allocates these turns to back tree nodes; they
    /// carry no KV chunks (`block_range = 0..0`) and `token_count` is
    /// the placeholder for the summary text.  Without persistence, on
    /// reopen the walker would replay [`TreeMetadata`] records for
    /// these indices but find no matching [`TurnDecl`], leaving
    /// orphan `tree_meta` entries that the score-density selector
    /// would then pick and elevate would fail to lift.
    ///
    /// Drops the auto-pending entry that [`append_with_blocks`] pushed
    /// — summary turns are not Normal and shouldn't loop back through
    /// the pending queue.
    pub fn record_summary_turn(
        &self,
        timeline: TimelineId,
        token_count: usize,
    ) -> candle::Result<TurnIndex> {
        let identity = self.minter.mint(0);
        // The turn and its declaration in one edit, as `record_turn`'s.
        let (idx, decl) = self.update(|s| {
            let idx = s.append_with_blocks(timeline, token_count, 0, 0);
            s.pop_pending_summary(timeline);
            let decl = Self::turn_decl(s, timeline, idx, 2, (0, 0), &[], &[]);
            s.apply_stream_decl(decl.stream_id(), decl.clone(), identity);
            (idx, decl)
        });
        self.persistence.enqueue(WriteJob::StreamDecl {
            stream_id: decl.stream_id().0,
            payload: decl.encode(),
            identity,
        });
        Ok(idx)
    }

    /// Restore a gap at every position of `decl`'s timeline below
    /// `decl.turn_index` that no turn holds yet. `restore_turn` assigns
    /// indices densely, so without it a turn above a lost decl would land one
    /// index low, away from the streams its index resolves to; a turn that
    /// arrives for the position later takes the gap in place.
    pub(super) fn fill_turn_gap(&self, decl: &TurnDecl) {
        let Some(timeline) = TimelineId::from_raw(decl.timeline_id) else {
            return;
        };
        if self.read().turn_count(timeline) >= decl.turn_index {
            return;
        }
        self.update(|view| {
            if let (Some(layer), Some(group)) = (
                LayerId::from_raw(decl.layer_id),
                GroupId::from_raw(decl.group_id),
            ) {
                view.register_timeline(timeline, layer, group);
            }
            while view.turn_count(timeline) < decl.turn_index {
                view.restore_gap_turn(timeline);
            }
        });
    }

    /// Restore `turn` at the position `decl` names in `timeline`, registering
    /// the timeline first — one edit, as a reload reads the turn.
    pub(super) fn restore_turn_at(
        &self,
        timeline: TimelineId,
        decl: &TurnDecl,
        turn: RestoredTurn,
    ) {
        self.update(|view| {
            if let (Some(layer), Some(group)) = (
                LayerId::from_raw(decl.layer_id),
                GroupId::from_raw(decl.group_id),
            ) {
                view.register_timeline(timeline, layer, group);
            }
            view.reload_turn(timeline, TurnIndex(decl.turn_index), turn.clone());
        });
    }

    /// Occupy the position `decl` names with an empty placeholder, so the
    /// timeline's later turns keep their persisted indices: `restore_turn`
    /// assigns densely, and a bare skip renumbers them away from their stream
    /// ids. Never selected (no content), never materialisable (no cold refs).
    pub(super) fn restore_placeholder(&self, timeline: TimelineId, decl: &TurnDecl) {
        self.restore_turn_at(
            timeline,
            decl,
            RestoredTurn {
                layout: TurnLayout::new(Vec::new()),
                token_ids: TokenBuffer::default(),
                token_count: 0,
                cold: None,
                block_start: decl.block_start,
                block_end: decl.block_end,
            },
        );
    }

    /// Whether `decl`'s turn, at the position its `turn_index` names, is
    /// another member's. A turn of this member's own, or one written before
    /// identities, is not.
    pub(super) fn is_remote_turn(&self, decl: &TurnDecl) -> bool {
        let Some(timeline) = TimelineId::from_raw(decl.timeline_id) else {
            return false;
        };
        let writer = self.read().turn_mark(timeline, decl.turn_index).writer;
        writer != 0 && writer != self.minter.node()
    }

    /// Whether every record `decl`'s turn calls for is here (§6.5).
    pub(super) fn is_turn_complete(&self, decl: &TurnDecl, n_layers: usize) -> bool {
        let read = self.read();
        let distilled = TimelineId::from_raw(decl.timeline_id)
            .is_some_and(|timeline| read.distill_mode(timeline).is_some());
        read.stream_of(turn_stream_id(decl.timeline_id, decl.turn_index))
            .is_some_and(|stream| turn_is_complete(decl, stream, n_layers, distilled))
    }

    /// Hold `decl`'s position with a gap, registering its timeline first.
    pub(super) fn restore_pending_turn(&self, decl: &TurnDecl) {
        let Some(timeline) = TimelineId::from_raw(decl.timeline_id) else {
            return;
        };
        self.update(|view| {
            if let (Some(layer), Some(group)) = (
                LayerId::from_raw(decl.layer_id),
                GroupId::from_raw(decl.group_id),
            ) {
                view.register_timeline(timeline, layer, group);
            }
            // Its position, as a gap, unless something holds it already: a
            // runtime reload finds the gap a live adoption left there.
            while view.turn_count(timeline) <= decl.turn_index {
                view.restore_gap_turn(timeline);
            }
        });
    }
}
