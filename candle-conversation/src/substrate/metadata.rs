//! Answers the conversation-level questions the sidebar, the caches and the
//! calibration lookups ask: labels, archive state, branches, custom metadata and
//! calibrations.
//!
//! Each field of a conversation's metadata keeps the value of the greatest record
//! that set it, so a rename and a tag written at once both land. A tombstoned
//! conversation is never listed or counted as a cache hit, because it no longer
//! serves retrieval. The intact-K/V check counts archived and distilled
//! conversations as intact, since their K/V was released on purpose, and it reads
//! only a conversation's own turns, because damage in an ancestor is repaired
//! once for every descendant. A calibration names the conversation it calibrates,
//! and its signatures vote for that conversation.

use super::*;

impl Substrate {
    /// The identity of the record that set `key` of `timeline`'s `custom`
    /// metadata, when one has.
    pub fn custom_field_id(&self, timeline: TimelineId, key: &str) -> Option<RecordIdentity> {
        self.timelines.get(&timeline)?.ids.custom.get(key).copied()
    }

    /// Hold `key` of `timeline`'s `custom` metadata under generation `gen`, the
    /// one a rejoin fixed for the cut-off record that set it, as the reload that
    /// follows the rejoin will. Until then its provisional generation outranks
    /// everything this member mints.
    pub fn settle_custom_field(&mut self, timeline: TimelineId, key: &str, gen: u64) {
        let Some(entry) = entry_mut(&mut self.timelines, &timeline) else {
            return;
        };
        if let Some(held) = entry.ids.custom.get(key).copied() {
            entry
                .ids
                .custom
                .insert(key.to_string(), RecordIdentity { gen, ..held });
        }
    }

    /// Metadata key a **calibration** conversation carries: the raw timeline
    /// id of the conversation it calibrates. Its turns are questions that
    /// conversation's content answers, each prefilled alone, and their
    /// signatures vote for that conversation wherever it is scored
    /// (`belief_files::assemble_file_scans`) — the gallery entries a live
    /// question resembles, where the conversation's own are its content.
    pub const CALIBRATES_KEY: &'static str = "calibrates";

    /// The conversation `timeline` calibrates, when it is a calibration.
    pub fn calibrates(&self, timeline: TimelineId) -> Option<TimelineId> {
        self.timelines
            .get(&timeline)?
            .custom
            .get(Self::CALIBRATES_KEY)?
            .parse()
            .ok()
            .and_then(TimelineId::from_raw)
    }

    /// `unit`'s live calibrations, in timeline order. Archived and distilled
    /// ones count: a calibration ends archived and provenance-only by design.
    pub fn calibrations_of(&self, unit: TimelineId) -> impl Iterator<Item = TimelineId> + '_ {
        self.calibrations_of
            .get(&unit)
            .into_iter()
            .flatten()
            .copied()
            .filter(|tl| !self.tombstoned_timelines.contains(tl))
    }

    /// The calibration records as they stand, for one scan to read without a
    /// lock ([`calibration_record`]).
    pub fn calibration_snapshot(&self) -> CalSnapshot {
        self.calibration_records.snapshot()
    }

    /// Keep the records a scan built, in one swap.
    pub fn publish_calibration_records(&self, fresh: Vec<(TimelineId, Arc<CalRecord>)>) {
        self.calibration_records.publish(fresh);
    }

    /// `unit`'s calibration signatures as one unit, or `None` for a conversation
    /// with no live calibration: `snapshot`'s record while it stands, otherwise
    /// one built now and added to `fresh` for the caller to publish
    /// ([`calibration_record`]).
    pub fn calibration_record(
        &self,
        unit: TimelineId,
        snapshot: &CalSnapshot,
        fresh: &mut Vec<(TimelineId, Arc<CalRecord>)>,
    ) -> Option<Arc<CalRecord>> {
        let live = || self.calibrations_of(unit).map(|c| (c, self.turn_count(c)));
        live().next()?;
        let generation = (self.sig_rewrites, self.tombstone_epoch);
        if let Some(record) = standing(snapshot, unit, live(), generation) {
            return Some(record);
        }
        let stamp: Vec<(TimelineId, u32)> = live().collect();
        let (mut sids, mut sigs) = (Vec::new(), Vec::new());
        let mut complete = true;
        for &(cal, turns) in &stamp {
            for i in 0..turns {
                let sid = turn_stream_id(cal.raw(), i);
                match self.wide_sig_window(sid) {
                    Some(window) => {
                        sids.push(sid);
                        sigs.push(window);
                    }
                    None => complete = false,
                }
            }
        }
        let record = Arc::new(CalRecord::new(stamp, generation, complete, sids, sigs));
        fresh.push((unit, record.clone()));
        Some(record)
    }

    /// Every live calibration, keyed by the conversation it calibrates —
    /// [`Self::calibrations_of`] for every unit that has one.
    pub fn calibrations(&self) -> HashMap<TimelineId, Vec<TimelineId>> {
        self.calibrations_of
            .keys()
            .filter_map(|&unit| {
                let live: Vec<TimelineId> = self.calibrations_of(unit).collect();
                (!live.is_empty()).then_some((unit, live))
            })
            .collect()
    }

    /// All **live** (non-tombstoned) timelines whose `custom` metadata
    /// contains `key == value` (exact match). The content-addressed cache
    /// lookup used by utility ingests to skip re-building units already
    /// present after load. Tombstoned timelines are excluded — a dead
    /// conversation must never count as a cache hit (it no longer serves
    /// retrieval), or its file would silently vanish from the layer.
    /// Like [`Self::timelines_with_metadata`], but also admits a tombstoned
    /// timeline when it carries a distillation mode.
    ///
    /// For the calibration resume filter only. A calibration exemplar's designed
    /// end state is archived + distilled(`ProvenanceOnly`) + tombstoned: out of
    /// the live gather, still answering the belief scan by signature. The
    /// live-only lookup cannot see that state, so the resume filter read every
    /// such exemplar as "never ran" and regenerated it — ~196 per boot.
    ///
    /// The distill requirement is the whole point: an ORDINARY tombstone still
    /// means gone, and must stay invisible to provenance and to these checks.
    /// Only the distill flag distinguishes "retired by design, sigs retained"
    /// from "retired, finished".
    pub fn timelines_with_metadata_including_distilled(
        &self,
        key: &str,
        value: &str,
    ) -> Vec<TimelineId> {
        self.timelines
            .iter()
            .filter(|(tid, tl)| {
                let live = !self.tombstoned_timelines.contains(tid);
                let distilled_tombstone = !live && self.distill_mode(**tid).is_some();
                (live || distilled_tombstone)
                    && tl.custom.get(key).map(|v| v.as_str()) == Some(value)
            })
            .map(|(tid, _)| *tid)
            .collect()
    }

    pub fn timelines_with_metadata(&self, key: &str, value: &str) -> Vec<TimelineId> {
        self.timelines
            .iter()
            .filter(|(tid, tl)| {
                !self.tombstoned_timelines.contains(tid)
                    && tl.custom.get(key).map(|v| v.as_str()) == Some(value)
            })
            .map(|(tid, _)| *tid)
            .collect()
    }

    /// The live timelines whose `custom[key]` is each of `values`, in one pass:
    /// [`Self::timelines_with_metadata`] for many values at once. A value no
    /// live timeline carries is absent from the map.
    pub fn timelines_with_metadata_values(
        &self,
        key: &str,
        values: &BTreeSet<String>,
    ) -> BTreeMap<String, Vec<TimelineId>> {
        let mut out: BTreeMap<String, Vec<TimelineId>> = BTreeMap::new();
        for (tid, tl) in self.timelines.iter() {
            if self.tombstoned_timelines.contains(tid) {
                continue;
            }
            if let Some(v) = tl.custom.get(key).filter(|v| values.contains(*v)) {
                out.entry(v.clone()).or_default().push(*tid);
            }
        }
        out
    }

    /// Whether a live timeline whose own K/V is intact carries `custom[key] ==
    /// value` — [`Self::metadata_values_for_key`]'s membership for one value,
    /// stopping at the first match.
    pub fn has_live_intact_metadata(&self, key: &str, value: &str) -> bool {
        self.timelines.iter().any(|(tid, tl)| {
            tl.custom.get(key).map(|v| v.as_str()) == Some(value)
                && !self.tombstoned_timelines.contains(tid)
                && self.timeline_kv_intact(*tid)
        })
    }

    /// The set of distinct `custom[key]` values across all **live**
    /// (non-tombstoned) timelines whose own K/V is intact. A one-pass snapshot
    /// so callers can probe membership in O(1) instead of an O(timelines) scan
    /// per probe (e.g. the code_read resume cache over thousands of files).
    ///
    /// **Intactness is part of the answer, because this set is a resume cache.**
    /// Its whole purpose is "has this content already been ingested, so the work
    /// can be skipped" — and a conversation that lost its K/V has not been
    /// usefully ingested, however complete its metadata looks. Keyed on the
    /// source hash alone, such a unit matched, was skipped, and was therefore
    /// never repaired: the hash describes the *input*, so it keeps matching
    /// forever while the stored answer stays a hole. Nothing else noticed
    /// either — `validate` only checks the chunk grid of turns that *have* one,
    /// and the crashed-partial sweep looks for a missing hash, which this unit
    /// has. Measured: `ARCHITECTURE.md`, a priming-chain anchor in every
    /// conversation's ancestry, sat with 104 blocks of a turn's K/V absent for
    /// days across many boots.
    ///
    /// A hash survives here if *any* intact timeline carries it, so this only
    /// drops a value when every conversation holding it is damaged — which is
    /// exactly when the unit needs re-ingesting.
    pub fn metadata_values_for_key(&self, key: &str) -> std::collections::HashSet<String> {
        self.timelines
            .iter()
            .filter(|(tid, _)| !self.tombstoned_timelines.contains(tid))
            .filter(|(tid, _)| self.timeline_kv_intact(**tid))
            .filter_map(|(_, tl)| tl.custom.get(key).cloned())
            .collect()
    }

    /// Whether every turn **this timeline owns** still holds its K/V in some
    /// tier (hot, warm or cold).
    ///
    /// **Own turns only — never the inherited chain.** A fork's ancestors are
    /// resolved live at projection time ([`Self::inherited_chain`]), so a hole
    /// in an ancestor is that ancestor's to repair and is fixed for every
    /// descendant at once when it is. Walking the chain here would instead mark
    /// every descendant damaged: each `code_read` file hangs off the priming
    /// chain's end, so one damaged anchor would condemn the entire corpus and
    /// re-ingest the whole workspace.
    ///
    /// **K/V that was deliberately reclaimed is not damage**, so an ARCHIVED or
    /// DISTILLED timeline always reads as intact. A tool-calibration exemplar
    /// ends archived + distilled(`ProvenanceOnly`): [`Self::release_distilled_kv`]
    /// drops its chunks on purpose, because its `WideQSig`s are all the belief
    /// scan still needs from it. Counting that as damage would be wrong twice
    /// over — it describes 876 of this store's 1,467 conversations, and a
    /// re-ingest is not the repair for content nobody ingests. Tombstoned
    /// timelines never reach here; [`Self::metadata_values_for_key`] drops them
    /// first.
    ///
    /// A turn spanning no blocks holds no K/V by construction and is skipped —
    /// it is not damage either. An unknown timeline reads as NOT intact, which
    /// fails toward doing the work rather than trusting a conversation that is
    /// not there.
    pub fn timeline_kv_intact(&self, timeline: TimelineId) -> bool {
        let Some(entry) = self.timelines.get(&timeline) else {
            return false;
        };
        if entry.archived || self.is_distilled(timeline) {
            return true;
        }
        entry.turns.values().all(|turn| {
            if turn.block_range.1 <= turn.block_range.0 {
                return true;
            }
            let slot = self.residences.get(turn.content.residence);
            slot.has_hot() || slot.has_warm() || slot.has_cold()
        })
    }

    /// All **live** (non-tombstoned) timelines that carry `key` in their
    /// `custom` metadata, paired with that key's value. Drives utility
    /// ingest reconciliation (e.g. tombstone every `code_read`
    /// conversation whose `path` is no longer on disk).
    pub fn timelines_with_metadata_key(&self, key: &str) -> Vec<(TimelineId, String)> {
        self.timelines
            .iter()
            .filter(|(tid, _)| !self.tombstoned_timelines.contains(tid))
            .filter_map(|(tid, tl)| tl.custom.get(key).map(|v| (*tid, v.clone())))
            .collect()
    }

    /// Whether `timeline` has been archived by the user. Untouched
    /// timelines default to `false`. Returns `false` for unknown
    /// timelines (matches "not archived" since the conversation
    /// doesn't exist as far as the sidebar is concerned).
    pub fn is_archived(&self, timeline: TimelineId) -> bool {
        self.timelines.get(&timeline).is_some_and(|e| e.archived)
    }

    /// Set a timeline's archived flag. No-op when the timeline isn't
    /// registered. Returns `true` when the flag actually changed —
    /// the daemon uses this to skip the persistence write when the
    /// caller is just re-asserting the current state.
    pub fn set_archived(&mut self, timeline: TimelineId, archived: bool) -> bool {
        let Some(entry) = entry_mut(&mut self.timelines, &timeline) else {
            return false;
        };
        if entry.archived == archived {
            return false;
        }
        entry.archived = archived;
        // An archived conversation takes no more turns: no scope of its is
        // read again. Either way the groups it is in offer a different set.
        if archived {
            self.retrieval_scopes.remove(&timeline);
            self.scope_digests.remove(&timeline);
        }
        self.timeline_content_changed(timeline);
        true
    }

    /// `timeline`'s whole conversation state — what its `ConvState` record
    /// carries — or `None` for an unregistered timeline.
    pub fn conv_state(&self, timeline: TimelineId) -> Option<ConvState> {
        self.timelines.get(&timeline).map(|e| ConvState {
            archived: e.archived,
            branches: e
                .branches
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            active: e.active,
        })
    }

    /// Mark `timeline` as used now: stamp it with the next last-use rank, so
    /// it outranks every conversation used before it. Returns `false` for an
    /// unregistered timeline, which has no state to write.
    pub fn touch_conversation(&mut self, timeline: TimelineId) -> bool {
        let Some(entry) = entry_mut(&mut self.timelines, &timeline) else {
            return false;
        };
        self.conv_active_counter += 1;
        entry.active = self.conv_active_counter;
        true
    }

    /// Set the branch `timeline` works on in `repo`. No-op when the timeline
    /// isn't registered. Returns `true` when the branch actually changed, so
    /// the caller can skip the persistence write when nothing did.
    pub fn set_branch(&mut self, timeline: TimelineId, repo: &str, branch: &str) -> bool {
        let Some(entry) = entry_mut(&mut self.timelines, &timeline) else {
            return false;
        };
        if entry.branches.get(repo).map(String::as_str) == Some(branch) {
            return false;
        }
        entry.branches.insert(repo.to_string(), branch.to_string());
        true
    }

    /// Record that `timeline`'s state, just changed in place by
    /// [`Self::set_archived`], [`Self::touch_conversation`] or
    /// [`Self::set_branch`], is carried by `ConvState` record `id` — minted
    /// for the change, so it outranks every state the timeline held.
    pub fn stamp_conv_state(&mut self, timeline: TimelineId, id: RecordIdentity) {
        if let Some(entry) = entry_mut(&mut self.timelines, &timeline) {
            take(&mut entry.ids.state, id);
        }
    }

    /// Every live timeline's conversation state that differs from an
    /// untouched one, as `(timeline_id, state, record)` in timeline order —
    /// what compaction and maintenance re-emit as `ConvState` records.
    pub fn live_conv_states(&self) -> Vec<(u64, ConvState, RecordIdentity)> {
        let mut out: Vec<(u64, ConvState, RecordIdentity)> = self
            .timelines
            .iter()
            .filter_map(|(tl, entry)| {
                let state = self.conv_state(*tl)?;
                (!state.is_default()).then_some((tl.raw(), state, entry.ids.state))
            })
            .collect();
        out.sort_by_key(|(tl, ..)| *tl);
        out
    }

    /// Every **live** timeline that has a `conv_id` recorded, paired
    /// with `(conv_id, label, archived)`. Drives the daemon's sidebar:
    /// `label` is empty during the brief window between first-submit
    /// and titler-completion, `archived` is the lifecycle filter the
    /// sidebar applies before rendering.
    ///
    /// **Tombstoned timelines are excluded**, like every other listing here
    /// ([`Self::live_timeline_ids`], [`Self::conversations_with_conv_id_prefix`]).
    /// This used to return them, on the reasoning that excluding them was each
    /// caller's job — which made `DELETE /v1/conversations/{id}` a no-op from
    /// the only place a user can see: the row came back on the next sidebar
    /// hydrate, for the life of the log, because a tombstone marks a timeline
    /// for reclamation rather than removing its registry entry. Archived is the
    /// flag a caller filters on; deleted is not a flag, and nothing outside
    /// maintenance has any use for a deleted conversation.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String, bool, u64)> {
        self.timelines
            .iter()
            .filter(|(tl, _)| !self.tombstoned_timelines.contains(tl))
            .filter_map(|(tl, entry)| {
                let conv_id = entry.conv_id.clone()?;
                let label = entry.label.clone().unwrap_or_default();
                Some((*tl, conv_id, label, entry.archived, entry.order))
            })
            .collect()
    }

    /// Live timelines whose `conv_id` starts with `prefix`, as
    /// `(timeline, conv_id)`.
    ///
    /// [`Self::known_conversations`] answers the same question by materialising
    /// every live conversation the workspace holds and cloning two strings for
    /// each. A caller that wants one character's handful out of that pays for
    /// the whole registry, and the registry only grows: retirement bounds the
    /// live set, not the entries.
    ///
    /// Filtering inside the read lock allocates for the matches alone.
    pub fn conversations_with_conv_id_prefix(&self, prefix: &str) -> Vec<(TimelineId, String)> {
        self.timelines
            .iter()
            .filter(|(tl, _)| !self.tombstoned_timelines.contains(tl))
            .filter_map(|(tl, entry)| {
                let conv_id = entry.conv_id.as_ref()?;
                conv_id.starts_with(prefix).then(|| (*tl, conv_id.clone()))
            })
            .collect()
    }

    /// Number of pinned sections — used by the substrate-reload summary log.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Number of registered timelines — every timeline that has at least
    /// one TurnDecl (or was registered via `register_timeline` even
    /// without turns).
    pub fn timeline_count(&self) -> usize {
        self.timelines.len()
    }

    /// Number of timelines that have a `conv_id` set — i.e. the size of
    /// the daemon's sidebar list.
    pub fn conversation_count(&self) -> usize {
        self.timelines
            .values()
            .filter(|e| e.conv_id.is_some())
            .count()
    }
}
