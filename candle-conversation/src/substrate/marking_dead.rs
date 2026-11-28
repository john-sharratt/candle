//! Applies the log's tombstones to the in-RAM state, so retired timelines, turns
//! and sections drop out of every read and their resident KV is released now.
//!
//! A tombstone takes effect the same way whether it arrives live or on replay. A
//! turn-scoped one empties the turn in place, as a reload does, so a retired tail
//! cannot be selected for the rest of the process. A wholesale tombstone
//! subsumes the turn tombstones it covers, which keeps the set the same in either
//! replay order. A section tombstone ends one generation of its stream, and a
//! later declaration of the same content starts a new one. A custom object stays
//! marked as tombstoned until a complete compaction has run past it.

use super::*;

impl Substrate {
    /// When `obj` is a group's exclusion record, install its cut for the
    /// group's live scans — on every member, as it applies the record
    /// (`docs/incremental_chunk_exclusion.md` §4.7). A tombstone, or a record
    /// this build cannot read, leaves the group with no cut.
    pub(super) fn install_chunk_exclusion(&mut self, obj: &CustomObjectPayload) {
        let Some(group) = group_of_key(&obj.key).and_then(GroupId::from_raw) else {
            return;
        };
        let cut = (!obj.ceased)
            .then(|| from_json(&obj.blob))
            .flatten()
            .and_then(|b| CutRecord::decode(&b))
            .map_or_else(ChunkExclusion::default, |r| {
                ChunkExclusion::of_ranges(r.cut)
            });
        self.chunk_exclusion.insert(group, Arc::new(cut));
        self.group_content_changed(group);
    }

    /// The record behind the custom object keyed `key_hash`.
    pub fn custom_object_id(&self, key_hash: u64) -> RecordIdentity {
        self.custom_object_ids
            .get(&key_hash)
            .copied()
            .unwrap_or_default()
    }

    /// The live custom object stored under `key`, if any: none for a key whose
    /// winning record is its tombstone.
    pub fn custom_object(&self, key: &str) -> Option<&CustomObjectPayload> {
        self.custom_objects
            .get(&CustomObjectPayload::stream_id_for(key))
            .map(|obj| &**obj)
            .filter(|obj| !obj.ceased)
    }

    /// The identity of the tombstone standing on blob `key`, when its key's
    /// winning record is one ([`BlobKey::tombstone_key`]): every record of the
    /// blob it outranks is dead (`docs/replicated_substrate.md` §6.8).
    pub fn blob_tombstone(&self, key: BlobKey) -> Option<RecordIdentity> {
        let hash = CustomObjectPayload::stream_id_for(&key.tombstone_key());
        self.custom_objects
            .get(&hash)
            .filter(|obj| obj.ceased)
            .map(|_| self.custom_object_id(hash))
    }

    /// Every blob a tombstone stands on.
    pub fn blob_tombstones(&self) -> Vec<BlobKey> {
        self.custom_objects
            .values()
            .filter_map(|obj| retired_blob(obj))
            .collect()
    }

    /// Every live custom object whose key starts with `prefix`, with the
    /// record behind it, in key order.
    pub fn custom_objects_under(&self, prefix: &str) -> Vec<(CustomObjectPayload, RecordIdentity)> {
        self.live_under(prefix)
            .map(|(hash, obj)| (obj.clone(), self.custom_object_id(hash)))
            .collect()
    }

    /// The live custom objects under `prefix`, in key order, as their key
    /// hash and the object, without copying any: a range of the key index.
    pub(super) fn live_under<'a>(
        &'a self,
        prefix: &'a str,
    ) -> impl Iterator<Item = (u64, &'a CustomObjectPayload)> + 'a {
        self.custom_object_keys
            .range(prefix.to_string()..)
            .take_while(move |(k, _)| k.starts_with(prefix))
            .filter_map(|(_, hash)| {
                let obj = self.custom_objects.get(hash)?;
                (!obj.ceased).then_some((*hash, &**obj))
            })
    }

    /// How many live custom objects' keys start with `prefix`.
    pub fn custom_objects_count_under(&self, prefix: &str) -> usize {
        self.live_under(prefix).count()
    }

    /// How many live custom objects under `prefix` have a key `keep` accepts:
    /// keys alone, nothing copied or decoded.
    pub fn custom_objects_count_where(&self, prefix: &str, keep: impl Fn(&str) -> bool) -> usize {
        self.custom_object_keys
            .range(prefix.to_string()..)
            .take_while(|(k, _)| k.starts_with(prefix))
            .filter(|(k, hash)| {
                keep(k)
                    && self
                        .custom_objects
                        .get(*hash)
                        .is_some_and(|obj| !obj.ceased)
            })
            .count()
    }

    /// The live custom object under `prefix` whose key sorts last: only it is
    /// copied.
    pub fn custom_object_last_under(&self, prefix: &str) -> Option<CustomObjectPayload> {
        self.live_under(prefix).last().map(|(_, obj)| obj.clone())
    }

    /// A count that moves whenever a custom object is taken: what a reader of
    /// a prefix compares to know whether to read it again.
    pub fn custom_objects_version(&self) -> u64 {
        self.custom_objects_version
    }

    /// Every custom object this substrate holds, tombstones included, with the
    /// record behind it — the set the compaction pass re-emits so they
    /// survive it.
    pub fn held_custom_objects(&self) -> Vec<(CustomObjectPayload, RecordIdentity)> {
        self.custom_objects
            .iter()
            .map(|(key, obj)| ((**obj).clone(), self.custom_object_id(*key)))
            .collect()
    }

    /// Whether a complete compaction has run past the tombstone on `key_hash`.
    pub fn is_key_tombstone_swept(&self, key_hash: u64) -> bool {
        self.deleted.is_key_swept(key_hash)
    }

    /// Apply a decoded [`TombstonePayload`].  Works whether or not
    /// the timeline is currently registered — registration just
    /// observes the tombstone bit when it later drains the same set.
    pub fn apply_tombstone(&mut self, payload: &TombstonePayload) {
        let Some(timeline) = TimelineId::from_raw(payload.timeline_id) else {
            return;
        };
        // Surface the reason on replay so a reload shows WHY a timeline (or turn)
        // was dropped (e.g. a corrupt-partial turn), not just that it vanished.
        if let Some(reason) = &payload.reason {
            tracing::debug!(
                timeline_id = payload.timeline_id,
                turn_index = payload.turn_index,
                reason = %reason,
                "replaying tombstone with recorded reason",
            );
        }
        // `turn_index = Some` is a turn-scoped tombstone (the `drop_turn` policy):
        // kill only that turn, leaving the rest of the timeline live. `None` drops
        // the whole timeline.
        match payload.turn_index {
            Some(turn) => {
                // Not if the whole timeline is already dead: that tombstone
                // says strictly more, and re-adding the turn here would make
                // the in-RAM set depend on the order the log happened to be
                // replayed in.
                if !self.tombstoned_timelines.contains(&timeline) {
                    self.tombstoned_turns.insert(TurnMark {
                        timeline,
                        turn_index: turn,
                        writer: payload.turn_writer,
                    });
                    self.gallery_epoch += 1;
                    self.tombstone_epoch += 1;
                    self.gallery_changed(timeline);
                    self.timeline_content_changed(timeline);
                }
            }
            None => self.mark_timeline_dead(timeline),
        }
    }

    /// Replay a timeline's death from its tombstone.
    pub(super) fn mark_timeline_dead(&mut self, timeline: TimelineId) {
        self.tombstoned_timelines.insert(timeline);
        self.gallery_epoch += 1;
        self.tombstone_epoch += 1;
        self.timeline_content_changed(timeline);
        // The same subsumption the live path applies — see
        // [`Self::tombstone_timeline`]. Replay order is arbitrary, so a turn
        // tombstone read after its timeline's must not resurrect an entry the
        // timeline tombstone already covers; `retain` here plus the guard in
        // `apply_tombstone`'s turn arm keeps both orders equal.
        self.tombstoned_turns.retain(|m| m.timeline != timeline);
        // A dead conversation's recurrent-state snapshot dies with it: dropping
        // the index entry here means compaction (which walks this map) never
        // carries the record forward. A snapshot of a turn-tombstoned tail is
        // handled at restore instead — the payload's turn_index is validated
        // against the recovered timeline before any state is scattered.
        self.retire_snapshot_stream(timeline);
    }

    /// Forget the tombstones a compaction expired (§6.8): their timelines are
    /// no longer held dead, and their ceased keyed values no longer held, so
    /// nothing carries them again. Every record they named is already gone.
    pub fn forget_expired_tombstones(&mut self, expired: &Expired) {
        for tl in &expired.timelines {
            self.tombstoned_timelines.remove(tl);
            // A throwaway timeline's mark outlives its tombstone only so a
            // decode still winding down seals nothing to disk; an expired
            // tombstone is long past any wind-down, so the mark goes with it.
            self.transient_timelines.remove(tl);
        }
        for key in &expired.keys {
            let ceased = self
                .custom_objects
                .get(key)
                .filter(|obj| obj.ceased)
                .map(|obj| obj.key.clone());
            if let Some(name) = ceased {
                self.custom_object_keys.remove(&name);
                self.custom_objects.remove(key);
                self.custom_object_ids.remove(key);
                self.custom_objects_version += 1;
            }
        }
    }

    /// Apply a `Deleted` record: the tombstones it names as swept have had a
    /// complete compaction run past them (§6.8).
    pub fn apply_deleted(&mut self, payload: &DeletedPayload) {
        self.deleted.apply(payload);
    }

    /// Whether a complete compaction has run past `timeline`'s tombstone.
    pub fn is_tombstone_swept(&self, timeline: TimelineId) -> bool {
        self.deleted.is_swept(timeline.raw())
    }

    /// The swept set as its record holds it — what maintenance carries — or
    /// `None` when it is empty.
    pub fn deleted_payload(&self) -> Option<DeletedPayload> {
        self.deleted.payload()
    }

    /// Mark a single `(timeline, turn)` tombstoned — the turn-scoped companion
    /// to [`Self::tombstone_timeline`]: retention retiring a conversation's
    /// tail, and the `drop_turn` corrupt-turn policy at reload. The rest of the
    /// timeline stays live.
    ///
    /// **The turn is emptied in place, now, exactly as a reload leaves it.** A
    /// reload restores a tombstoned turn as a zero-token placeholder at its own
    /// index — never selected, because it has no content, and never
    /// materialisable — and the live projection has no other way to tell a
    /// retired turn from a kept one. Marking the set alone left the turn
    /// selectable for the rest of the process: a retention sweep changed
    /// nothing a projection could see until the next restart, and a replay that
    /// retired one run's answer handed that answer to the next run as history.
    /// The index stays occupied so every later turn keeps its stream ids, and
    /// the turn's hot K/V is released as a retired timeline's is. A turn the
    /// timeline does not hold yet (the reload path marks before it restores) is
    /// only marked.
    ///
    /// Returns the turn as its tombstone record names it ([`TurnMark`]).
    pub fn tombstone_turn(&mut self, timeline: TimelineId, turn_index: u32) -> TurnMark {
        let mark = self.turn_mark(timeline, turn_index);
        // **Not onto a timeline that is already wholly dead.** Its tombstone
        // says strictly more, and [`Self::tombstone_timeline`] has just dropped
        // every turn mark it covers — so inserting one here would put back an
        // entry nothing will ever read and leave this set disagreeing with the
        // one a replay of the same log rebuilds. The replay path guards the
        // same way for the same reason; both orders have to land in one place.
        //
        // Reachable: a character deleted mid-decode is tombstoned by
        // `retire_npc` while the tick that is still running goes on to retire
        // that conversation's tail.
        if self.tombstoned_timelines.contains(&timeline) {
            return mark;
        }
        self.tombstoned_turns.insert(mark);
        self.gallery_epoch += 1;
        self.tombstone_epoch += 1;
        self.timeline_content_changed(timeline);
        let (residence, retired_tokens) = match self.turn_mut(timeline, TurnIndex(turn_index)) {
            Some(entry) => {
                let retired_tokens = entry.content.token_count;
                entry.content.layout = TurnLayout::new(Vec::new());
                entry.content.token_ids = TokenBuffer::default();
                entry.content.token_count = 0;
                (entry.content.residence, retired_tokens)
            }
            None => return mark,
        };
        // The running total counts what the timeline's turns pin; a retired turn
        // pins nothing, exactly as a reload restores it with none.
        if let Some(total) = self.timeline_token_totals.get_mut(&timeline) {
            *total = total.saturating_sub(retired_tokens);
        }
        self.release_dead_residence(residence);
        mark
    }

    /// The turn at `turn_index` of `timeline` as a turn tombstone names it: the
    /// index its writer gave it and the writer, read off the declaration at
    /// that position. A position with no declaration is named by itself, as a
    /// turn with no identity is.
    pub fn turn_mark(&self, timeline: TimelineId, turn_index: u32) -> TurnMark {
        let at = turn_stream_id(timeline.raw(), turn_index);
        self.streams
            .resolve(at)
            .and_then(|key| self.streams.turn(key))
            .map_or(
                TurnMark {
                    timeline,
                    turn_index,
                    writer: 0,
                },
                |turn| TurnMark {
                    timeline,
                    turn_index: turn.decl.turn_index,
                    writer: turn.key.writer,
                },
            )
    }

    /// Whether the declared turn on `key`'s stream was retired by a turn
    /// tombstone — the question a rewrite asks of every writer's stream.
    pub fn is_turn_stream_retired(&self, key: StreamKey, decl: &TurnDecl) -> bool {
        TimelineId::from_raw(decl.timeline_id).is_some_and(|timeline| {
            self.tombstoned_turns.contains(&TurnMark {
                timeline,
                turn_index: decl.turn_index,
                writer: key.writer,
            })
        })
    }

    /// Drop a retired residence's hot copy now rather than at compaction.
    ///
    /// Its chunks survive for any other holder: a code_read scope fork's turns
    /// are spliced onto the file timeline (which clones the chunk handles) right
    /// before the fork is tombstoned, so dropping the fork's hot only releases
    /// its redundant reference. For genuinely retired content the K/V is dead
    /// anyway. Without this, the orphaned hot copies — which nothing else
    /// evicts, their owner being tombstoned and never demoted — accumulate on
    /// the card through a bulk ingest (the `quant_live` climb).
    ///
    /// A kill ([`ResidenceSlab::kill`]): the residence is marked dead, so a
    /// migrate or a lift that built a copy before the tombstone and installs it
    /// after takes nothing — the install sees the mark, whichever order the two
    /// ran in. A transient (scratch-fork) residence never gets a cold copy, so
    /// its warm RAM is dead the moment it's tombstoned and goes too; a normal
    /// tombstone leaves warm for any holder / cold-land reclaim. Idempotent, so
    /// a version edit that is retried kills it again harmlessly.
    pub(super) fn release_dead_residence(&self, r: ResidenceIndex) {
        self.residences.kill(r);
    }

    /// Whether the turn at `turn_index` of `timeline` was dropped by a
    /// turn-scoped tombstone.
    pub fn is_turn_tombstoned(&self, timeline: TimelineId, turn_index: u32) -> bool {
        self.tombstoned_turns
            .contains(&self.turn_mark(timeline, turn_index))
    }

    /// Record a layer's [`CorruptTurnPolicy`] (from the projection schema),
    /// consulted at reload when one of the layer's turns is unrecoverable.
    pub fn set_layer_corrupt_turn_policy(&mut self, layer: LayerId, policy: CorruptTurnPolicy) {
        self.layer_corrupt_turn.insert(layer, policy);
    }

    /// The layer's corrupt-turn policy, or the default
    /// [`CorruptTurnPolicy::DropConversation`] if none was registered.
    pub fn corrupt_turn_policy_for(&self, layer: LayerId) -> CorruptTurnPolicy {
        self.layer_corrupt_turn
            .get(&layer)
            .copied()
            .unwrap_or_default()
    }

    /// Mark `timeline` as tombstoned in-RAM.  Callers writing the
    /// matching `Tombstone` record to the redo log invoke this to
    /// keep the live projection / resolver in sync — without it the
    /// deletion would only take effect on the next reload.
    pub fn tombstone_timeline(&mut self, timeline: TimelineId) {
        self.tombstoned_timelines.insert(timeline);
        self.gallery_epoch += 1;
        self.tombstone_epoch += 1;
        self.timeline_content_changed(timeline);
        // **A wholesale tombstone subsumes every turn-scoped one it covers.**
        //
        // `tombstoned_turns` only ever grew, so a conversation that retired a
        // long tail and was then retired itself left one entry per turn behind
        // for the life of the process — invisible while nothing read the set,
        // and no longer so now that maintenance re-emits it: those entries
        // would be walked on every sweep to be skipped, forever.
        //
        // Dropping them here is safe precisely because the timeline tombstone
        // says more than they do: its records go wholesale, so nothing needs to
        // know which of its turns were retired first.
        self.tombstoned_turns.retain(|m| m.timeline != timeline);
        // Its recurrent tail dies with it, as a replay of the tombstone has it.
        self.retire_snapshot_stream(timeline);
        self.splice_source_timelines.remove(&timeline);
        // A tombstoned conversation is never a projection's target again.
        // Its own assemblies go with it: it reprojects nothing again, and the
        // groups offering it moved with its tombstone above.
        self.retrieval_scopes.remove(&timeline);
        self.scope_digests.remove(&timeline);
        // Nor carries anything, nor is carried: a pinned item whose turns are
        // gone would keep emitting a promise nothing can deliver.
        self.working_sets.remove(&timeline);
        for (_, ws) in self.working_sets.iter_mut() {
            Arc::make_mut(ws).remove(timeline);
        }
        self.forget_files_set(timeline);
        // A tombstoned timeline's KV is dead — release its resident VRAM now.
        let residences: Vec<ResidenceIndex> = match self.timelines.get(&timeline) {
            Some(entry) => entry.turns.values().map(|t| t.content.residence).collect(),
            None => return,
        };
        for r in residences {
            self.release_dead_residence(r);
        }
    }

    /// Mark the section stream `stream_id` as tombstoned in-RAM — the
    /// section counterpart of [`Self::tombstone_timeline`]. Callers writing
    /// the matching [`RecordType::SectionTombstone`] record to the redo log
    /// invoke this to keep the live ingest triage in sync — without it the
    /// section would only read as gone on the next reload, and a request in
    /// this same process would restore the corrupted chunks again.
    ///
    /// Releases any hot/warm VRAM a currently-registered section pointing at
    /// this stream holds, and ends the generation the stream resolves to — see
    /// [`Self::end_section`]. A tombstoned section is never read again in this
    /// generation, so nothing later depends on keeping its resident bytes
    /// around — unlike an ordinary LRU eviction, this is not something a future
    /// projection will need back.
    ///
    /// Returns the generation ended, for the tombstone record to name.
    pub fn tombstone_section(&mut self, stream_id: StreamId) -> Option<SectionGeneration> {
        let generation = self.section_generation(stream_id);
        self.end_section(stream_id, generation);
        let residences: Vec<ResidenceIndex> = self
            .sections
            .values()
            .map(|e| e.residence)
            .filter(|&r| self.residences.get(r).stream_id == stream_id)
            .collect();
        for r in residences {
            self.release_dead_residence(r);
        }
        generation
    }

    /// The generation section `stream_id` resolves to: its writer and the
    /// `oseq` of the declaration that started it. `None` for a stream with no
    /// section declaration, or an unstamped one.
    pub fn section_generation(&self, stream_id: StreamId) -> Option<SectionGeneration> {
        let key = self.streams.resolve(stream_id)?;
        let rt = self.streams.physical(key)?;
        match &rt.decl {
            Some(StreamDecl::PromptSection(_)) if !rt.ids.decl.is_none() => {
                Some(SectionGeneration {
                    writer: key.writer,
                    oseq: rt.ids.decl.oseq,
                })
            }
            _ => None,
        }
    }

    /// End the persisted generation of `stream_id` for a **rebuild**, leaving
    /// every resident copy serving.
    ///
    /// [`Self::tombstone_section`] is the corruption path: those bytes must
    /// never be read again, so it releases them. A rebuild retires sound bytes
    /// that are merely stale — compressed under thresholds since changed — and
    /// the conversations reading them now should keep reading them until the
    /// replacement exists. So this only ends the generation the stream
    /// resolves to ([`Self::end_section`]): nothing reloads it, the compactor
    /// stops carrying it, and the next triage that asks for the section
    /// prefills it fresh — at the next start, or earlier if this process asks
    /// first.
    ///
    /// Returns the generation ended, for the tombstone record to name.
    pub fn retire_section_generation(&mut self, stream_id: StreamId) -> Option<SectionGeneration> {
        let generation = self.section_generation(stream_id);
        self.end_section(stream_id, generation);
        generation
    }

    /// Whether section `stream_id` reads as tombstoned: a generation of it has
    /// ended and no section declaration of it stands.
    pub fn is_section_tombstoned(&self, stream_id: StreamId) -> bool {
        self.ended_sections.any(stream_id)
            && !matches!(
                self.streams.get(stream_id).and_then(|s| s.decl.as_ref()),
                Some(StreamDecl::PromptSection(_))
            )
    }

    /// Every section generation a tombstone ended — what a rewrite re-emits a
    /// [`RecordType::SectionTombstone`] for, the same role
    /// [`Self::tombstoned_timelines`] plays for timeline tombstones.
    pub fn ended_section_generations(
        &self,
    ) -> impl Iterator<Item = (StreamId, SectionGeneration)> + '_ {
        self.ended_sections.generations()
    }

    /// Every unstamped section stream a tombstone written before generations
    /// ended, and no unstamped declaration started again since.
    pub fn ended_unstamped_sections(&self) -> impl Iterator<Item = StreamId> + '_ {
        self.ended_sections.unstamped()
    }

    /// Apply a decoded [`SectionTombstonePayload`] read back from the redo
    /// log — the generation it names ends the way the live
    /// [`Self::tombstone_section`] ends it; replay runs before anything is
    /// restored, so there is no VRAM to release.
    pub fn apply_section_tombstone(
        &mut self,
        stream_id: StreamId,
        payload: &SectionTombstonePayload,
    ) {
        if let Some(reason) = &payload.reason {
            tracing::debug!(
                stream_id = stream_id.0,
                reason = %reason,
                "replaying section tombstone with recorded reason",
            );
        }
        self.end_section(stream_id, payload.generation);
    }
}
