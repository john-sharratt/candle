//! Warms the normalization backgrounds of an ingest's files and clusters from their
//! own persisted turns.
//!
//! Ingest layers are append-only and scored self-locally, so each file's exchanges
//! teach every other exchange's background. A file of one exchange is not warmed,
//! since its self-match would sit at the ceiling. The warm-up runs once ingest has
//! finished, never beside an ingest writer, because its scan holds the substrate
//! read lock long enough to starve that writer. The GPU work runs on the thread that
//! owns the arena.

use super::*;

/// Self-probes per ingested timeline (file / cluster) when warming the append-only
/// belief groups' backgrounds from their own turns (see
/// [`Conversation::warm_ingest_normalization`]). Each probe is about one of the
/// timeline's exchanges and teaches every other exchange's background, so this
/// bounds the one-time warm cost over a large corpus.
pub(super) const WARM_INGEST_PROBES_PER_TIMELINE: usize = 8;

/// Whether a group's backgrounds can be learned by self-match.
///
/// Scanned (belief-driven or a working-set group), because only a scanned
/// group normalizes at all; and additive, because a gated-fusion group learns
/// its backgrounds from traffic alone — the gate already handles the promiscuous
/// domination this warm-up was built to fix. Config-keyed, not axis-keyed: any
/// additive-fusion group warms.
///
/// And exchange-scored: a file-scored group's files are cases of its pool
/// scope, whose backgrounds live traffic teaches; a file's self-match would
/// teach that scope nothing but the file's own ceiling.
pub(super) fn is_warmable(group: &GroupSchema) -> bool {
    group.is_scanned()
        && group.policy.scan.fusion == FusionMode::Additive
        && group.policy.scan.cases == ScanCases::Exchange
}

/// `group` of `layer`, if the schema declares it there and it [`is_warmable`].
pub(super) fn warmable(
    schema: &Schema,
    layer: LayerId,
    group: GroupId,
) -> Option<(&LayerSchema, &GroupSchema)> {
    let layer = schema.layers.iter().find(|l| l.id == layer)?;
    let group = layer.groups.iter().find(|g| g.id == group)?;
    // A layer out of retrieval teaches no backgrounds: they would be
    // denominators for candidates it can never return. See
    // `LayerSchema::gathered`, which `warm_ingest_normalization` honours too.
    (layer.gathered && is_warmable(group)).then_some((layer, group))
}

/// `group` of `layer` for the bulk ingest warm-up: [`is_warmable`], but NOT
/// gated on `gathered`. An append-only layer's own conversations score
/// self-local whether or not other conversations draw from it, so its levels
/// have a reader either way — see
/// [`Conversation::ingest_warm_work`].
pub(super) fn ingest_group(
    schema: &Schema,
    layer: LayerId,
    group: GroupId,
) -> Option<&GroupSchema> {
    let layer = schema.layers.iter().find(|l| l.id == layer)?;
    let group = layer.groups.iter().find(|g| g.id == group)?;
    is_warmable(group).then_some(group)
}

/// How `layer`'s turns are gathered — [`GatherScope::Shared`] for a layer the
/// schema does not declare, whose turns nothing marks as one conversation's.
pub(super) fn layer_scope(schema: &Schema, layer: LayerId) -> GatherScope {
    schema
        .layers
        .iter()
        .find(|l| l.id == layer)
        .map_or(GatherScope::Shared, |l| l.gather_scope)
}

impl Conversation {
    /// Warm the append-only INGEST groups' per-file / per-cluster backgrounds from
    /// their OWN persisted turns (self-match). Call this EXPLICITLY after an ingest
    /// pass or reconcile has finished — NEVER lazily during a live query. The scan
    /// holds the substrate read lock across a `score_belief_groups` call per
    /// (file × probe); run CONCURRENTLY with an ingest writer it starves that writer
    /// on the unfair reader/writer lock and freezes the scheduler (no forwards,
    /// backlog stuck). Run once the ingest is done the corpus is stable, there is no
    /// writer to starve, and concurrent live queries (also readers) share the lock
    /// fine.
    ///
    /// Each ingest layer is append-only, so `score_belief_groups` runs self_local: a
    /// turn scores against its own timeline's exchanges, teaching every other
    /// exchange's background under the SAME `scope = turn_group(group,
    /// file_timeline)` the live cross-file query reads back. Without this every
    /// ingested file stays COLD, read against nothing but the probe's own scores.
    pub fn warm_ingest_normalization(&self, schema: &Schema) {
        warm_pool::run(|| self.warm_ingest_normalization_on_pool(schema))
    }

    /// [`Self::warm_ingest_normalization`]'s body, run on the warm pool. It runs
    /// after `ready`, in the reconcile thread, while conversations are being
    /// served — the case the warm pool exists for.
    pub(super) fn warm_ingest_normalization_on_pool(&self, schema: &Schema) {
        let t_warm = Instant::now();
        let mut warmed_timelines = 0usize;
        for (layer, group, timelines) in self.ingest_warm_work(schema) {
            let Some(group) = ingest_group(schema, layer, group) else {
                continue;
            };
            warmed_timelines +=
                self.warm_timelines(group, layer_scope(schema, layer), &timelines, None);
        }
        tracing::info!(
            timelines = warmed_timelines,
            elapsed_ms = t_warm.elapsed().as_millis() as u64,
            "normalization warm-up: learned per-file backgrounds for ingest-layer timelines \
             (0 ⇒ no append-only ingest layer marked — belief levels stay cold)"
        );
    }

    /// What the ingest warm-up has to do: every warmable group of every
    /// append-only ingest layer, with its active timelines, in schema order.
    ///
    /// Split from the scoring so the caller can hand the timelines to whichever
    /// thread holds the GPU gallery arena, a slice at a time
    /// ([`Self::warm_ingest_timelines`]).
    pub fn ingest_warm_work(&self, schema: &Schema) -> Vec<(LayerId, GroupId, Vec<TimelineId>)> {
        let mut work = Vec::new();
        for layer in &schema.layers {
            // This function only ever warms an append-only ingest layer — the
            // loop body below is unreached otherwise, so this is the real
            // filter; a leftover `!layer.gathered` check used to sit ahead of
            // it and is deliberately gone (see below).
            let is_ingest = self.inner.read().is_append_only_layer(layer.id);
            if !is_ingest {
                continue;
            }
            // `gathered = false` promises "no OTHER conversation draws from
            // this layer" — it does NOT mean nothing reads it at all. An
            // append-only layer's own conversations score self-local
            // (`score_belief_groups` exempts a layer scoring its own target
            // regardless of `gathered`, for the same reason this warm still
            // has a reader to serve). A `!layer.gathered` skip here used to
            // run ahead of the `is_ingest` check above, so it silently
            // matched exactly the ingest layers this loop exists to warm —
            // the moment one was taken out of cross-layer gather (`zend`'s
            // provenance-injection exclusion for `code_reading`/`repo_map`),
            // its own turns went cold forever: no later pass ever warmed
            // them, so a conversation several turns into its own history
            // could no longer retrieve its own earlier turns.
            for group in layer.groups.iter().filter(|g| is_warmable(g)) {
                let timelines: Vec<TimelineId> = self
                    .inner
                    .read()
                    .active_timelines_for_group(group.id)
                    .collect();
                work.push((layer.id, group.id, timelines));
            }
        }
        work
    }

    /// Warm `timelines` of `group` — a slice of [`Self::ingest_warm_work`] —
    /// scoring on `arena`, the GPU gallery arena. Returns how many were warmed.
    /// Called on the thread that owns the arena; each file costs one batched
    /// launch for all of its self-probes, where the host path costs a
    /// probe × window scan per probe (270 s over 1,344 files on eight cores).
    pub fn warm_ingest_timelines(
        &self,
        schema: &Schema,
        layer: LayerId,
        group: GroupId,
        timelines: &[TimelineId],
        arena: &GalleryArena,
    ) -> usize {
        let Some(group) = ingest_group(schema, layer, group) else {
            return 0;
        };
        self.warm_timelines(group, layer_scope(schema, layer), timelines, Some(arena))
    }

    /// Warm one belief group's per-timeline backgrounds by self-match —
    /// [`Self::warm_ingest_normalization`] for a single group, whatever its
    /// layer.
    ///
    /// For a group nothing else teaches: a tag-scoped turn group learns only
    /// from probes inside its scope, and ordinary dialogue is untagged, so its
    /// backgrounds would stay empty for the life of the process — every probe
    /// read against nothing but itself. Only `group` is scored and taught (see
    /// [`alone`]).
    /// Returns how many timelines were warmed; `0` when the group is not in
    /// the schema or not [`is_warmable`].
    pub fn warm_group_normalization(&self, schema: &Schema, group: GroupId) -> usize {
        warm_pool::run(|| self.warm_group_normalization_on_pool(schema, group))
    }

    /// [`Self::warm_group_normalization`]'s body, run on the warm pool.
    pub(super) fn warm_group_normalization_on_pool(
        &self,
        schema: &Schema,
        group: GroupId,
    ) -> usize {
        let Some(layer) = schema
            .layers
            .iter()
            .find(|l| l.groups.iter().any(|g| g.id == group))
        else {
            return 0;
        };
        let Some((_, group)) = warmable(schema, layer.id, group) else {
            return 0;
        };
        let timelines: Vec<TimelineId> = self
            .inner
            .read()
            .active_timelines_for_group(group.id)
            .collect();
        self.warm_timelines(group, layer.gather_scope, &timelines, None)
    }

    /// Warm one timeline's backgrounds by self-match — a conversation written
    /// onto a belief group after the group was warmed, which would otherwise be
    /// the one member scored on a different scale from the rest. `false` when
    /// the timeline is unknown or its group is not [`is_warmable`].
    pub fn warm_timeline_normalization(&self, schema: &Schema, timeline: TimelineId) -> bool {
        warm_pool::run(|| self.warm_timeline_normalization_on_pool(schema, timeline))
    }

    /// [`Self::warm_timeline_normalization`]'s body, run on the warm pool.
    pub(super) fn warm_timeline_normalization_on_pool(
        &self,
        schema: &Schema,
        timeline: TimelineId,
    ) -> bool {
        let Some((layer, group)) = self.timeline_target(timeline) else {
            return false;
        };
        let Some((layer, group)) = warmable(schema, layer, group) else {
            return false;
        };
        self.warm_timelines(group, layer.gather_scope, &[timeline], None);
        true
    }

    /// Warm each of `timelines` of `group` — a layer gathered `scope` — on
    /// `arena` when given. Returns how many were asked for.
    pub(super) fn warm_timelines(
        &self,
        group: &GroupSchema,
        scope: GatherScope,
        timelines: &[TimelineId],
        arena: Option<&GalleryArena>,
    ) -> usize {
        for &tl in timelines {
            // A shutdown ends the warm-up here too, not one timeline at a time.
            if self.ingest_cancelled() {
                break;
            }
            self.warm_timeline(group, tl, arena, Holder::masked_to(scope, tl));
        }
        timelines.len()
    }

    /// Self-match one timeline of `group`: a handful of its own turns probe it,
    /// teaching its exchanges' backgrounds under the SAME `scope =
    /// turn_group(group, timeline)` a live query reads back.
    ///
    /// The observation is the live seal scan's: each probe's raw per-exchange
    /// scores against the file's own exchanges. The exchange the probe is a turn
    /// of is its answer, so it is left out twice: out of the gallery the probe
    /// is scored against, so it cannot take the votes the others' backgrounds
    /// are made of, and out of the fold, since its score is the hit. The probes
    /// of one exchange are scored in one call — one launch on `arena`.
    ///
    /// **A file of one exchange is not warmed.** It competes in its group's
    /// pool (`belief_files::competitions`), where a document probed by its own
    /// signature agrees with itself on every bit and leaves the next document
    /// far behind: the self-match ceiling `docs/provenance_score_normalization.md`
    /// §2 rejects, which put every live hit at 1–2% of the band. Its background
    /// is learned from query traffic, as §3 has it.
    ///
    /// What it reads on `arena` is held for `holder`.
    pub(super) fn warm_timeline(
        &self,
        group: &GroupSchema,
        tl: TimelineId,
        arena: Option<&GalleryArena>,
        holder: Holder,
    ) {
        // The file's own turn signatures and its candidates, under one
        // short-lived read lock, released before any scoring.
        let (sigs, files) = {
            let _g = profile::span("warm:assemble");
            let sub = self.inner.read();
            let count = sub.turn_count(tl);
            // A handful of self-probes: each scores against ALL the file's
            // exchanges and teaches every one but its own, so this caps the warm
            // cost. Only those are decoded.
            let sigs: Vec<(u64, u32, Vec<WideQSig>)> = (0..count)
                .filter_map(|i| {
                    let sid = turn_stream_id(tl.raw(), i);
                    sub.wide_sig_window(sid).map(|w| (sid.0, i, w))
                })
                .take(WARM_INGEST_PROBES_PER_TIMELINE)
                .map(|(source, i, w)| (source, i, w.decode()))
                .collect();
            let (files, _) = assemble_file_scans(&sub, vec![tl]);
            (sigs, files)
        };
        let Some(file) = files.first().filter(|f| f.n_slots > 1) else {
            return;
        };
        let probes: Vec<(u64, u32, &[WideQSig])> = sigs
            .iter()
            .filter(|(_, _, sig)| !sig.is_empty())
            .map(|(source, i, sig)| (*source, *i, sig.as_slice()))
            .collect();
        if probes.is_empty() || self.ingest_cancelled() {
            return;
        }
        let child_of =
            |slot: usize| ChildKey::turn(file.arc_turn[file.ex_ranges[slot].start].0 as u64);
        // The exchange a turn belongs to: the probe's own, its answer.
        let exchange_of = |turn: u32| {
            file.ex_ranges
                .iter()
                .position(|r| r.clone().any(|ai| file.arc_turn[ai].0 == turn))
        };
        // **Each probe is scored against the file WITHOUT its own exchange.** A
        // probe is one of the exchange's own turns, so in a gallery that holds
        // it every token's best match is itself: the exchange takes every vote,
        // and every other exchange's background sample is exactly 0. Eight
        // zeros are a background, so a live hit then reads past the whole of it
        // — the flat, high band the tool collection showed for the same reason
        // ([`Self::fold_warm_probes`]). One scan per exchange probed.
        let scan_span = profile::span("warm:scan");
        let rounds = leave_out_rounds(probes.iter().map(|(_, turn, _)| exchange_of(*turn)));
        let mut scanned: Vec<Vec<(Vec<f32>, Vec<f32>)>> = vec![Vec::new(); probes.len()];
        for (own, at) in &rounds {
            let without: Vec<FileScan> = match own {
                Some(slot) => vec![file.without_exchange(*slot)],
                None => files.clone(),
            };
            let probe_refs: Vec<&[WideQSig]> = at.iter().map(|&p| probes[p].2).collect();
            let out = scan_file_scans(
                &without,
                group,
                &probe_refs,
                arena.map(|a| a as &dyn BeliefArena),
                holder,
                &ScanPlans::default(),
            );
            for (&p, per_file) in at.iter().zip(out) {
                scanned[p] = per_file;
            }
        }
        scan_span.end();
        let _g = profile::span("warm:fold");
        let scope = ScopeKey::turn_group(group.id.raw() as u64, tl.raw());
        let cache = &*self.normalization;
        for ((source, turn, probe), per_file) in probes.iter().zip(&scanned) {
            // Self-match: the probe is this group's own turn, so it is inside
            // the scope by construction.
            let observe = Observe::Yes {
                tags: &group.policy.tags,
                source: *source,
            };
            if !observe.teaches(&group.policy.tags) {
                continue;
            }
            let Some((fused, _)) = per_file.first() else {
                continue;
            };
            let raw_pairs: Vec<(ChildKey, f32)> = (0..file.n_slots)
                .map(|slot| (child_of(slot), fused.get(slot).copied().unwrap_or(0.0)))
                .collect();
            let answer = exchange_of(*turn).map(child_of);
            cache.observe(&scope, *source, &raw_pairs, answer.as_ref(), probe.len());
        }
    }
}
