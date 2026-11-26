//! Drives the background ingest of the branch and file layers: sealing and
//! calibration.
//!
//! Ingest seals each repository's files and folders into the knowledge base.
//! A pass seals units without calibrating them, and the sweep that follows
//! calibrates what was sealed and what an earlier pass left, so a file is
//! sealed once rather than held back for its whole calibration.

use super::*;

impl InferenceState {
    /// Build a [`RefreshContext`] bound to this state's engine,
    /// projection schema, and dialect config.  Cheap (the schema is
    /// `Arc`-backed) and avoids duplicating the construction in
    /// both refresh paths.
    pub(super) fn refresh_ctx(&self) -> RefreshContext<'_> {
        RefreshContext {
            engine: &self.engine,
            proj_builder: self.refresh_builder.clone(),
            config: self.refresh_config.clone(),
            formatted_prompt: &self.formatted_prompt,
            think_triggers: turn_triggers(self, ThinkMode::Off, ToolMode::Restricted),
            // Ingest reads; it runs no job, so it belongs to no lineage.
            tool_ctx: self.tool_host.context_for(
                ToolMode::Restricted,
                &self.tool_host.conversation_files(),
                "",
            ),
            retrieval: &self.retrieval_scope,
            think_for: Some({
                let steering = self.think_steering.clone();
                // A unit decodes under the grammar of the mode whose catalog
                // holds its tools (`LayerSpec::tool_mode`); its grants stay
                // ingest's, widened only as its layer's tools need.
                let stencils = self.tool_stencils.clone();
                Arc::new(move |mode: ThinkMode, tools: ToolMode| {
                    let stencil = &stencils[tools.level() as usize];
                    match &steering {
                        Some(ts) => ts.registry_for(stencil, mode),
                        None => Arc::clone(stencil),
                    }
                })
            }),
            dialogue: self.refresh_builder.id_for_group("primary_conversation"),
        }
    }

    /// Every branch layer this boot ingests, as the branch ingest takes them.
    pub(super) fn layer_passes(&self) -> anyhow::Result<Vec<LayerPass<'_>>> {
        self.branch_layers
            .iter()
            .map(|(il, scope)| {
                Ok(LayerPass {
                    name: &il.name,
                    mode: il.mode,
                    scope: scope.clone(),
                    base: self.ingest_base(&il.name)?,
                    spec: il.spec.as_deref(),
                    sees: self
                        .layer_sees
                        .get(&il.name)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                })
            })
            .collect()
    }

    /// Calibrate every branch layer that declares a `calibrate:` block
    /// (`crate::calibrate`): each ready unit without a calibration of its
    /// current version writes its questions, and they are prefilled as its
    /// exemplars. Run right after the branch pass, which reads a file layer's
    /// units without calibrating them, and never ahead of it: this sweep is
    /// where they are calibrated, together with whatever an earlier pass or a
    /// restart left.
    pub(super) fn calibrate_layers(&self) {
        // Every sealed calibration, of any layer, declaring one or not.
        mark_sealed(&self.engine);
        for (il, _) in &self.branch_layers {
            if crate::ingest_cancel::cancelled() {
                return;
            }
            let Some(spec) = il.calibrate.as_deref() else {
                continue;
            };
            let base = match self.ingest_base(&il.name) {
                Ok(base) => base,
                Err(e) => {
                    tracing::error!(
                        target: "zend::calibrate",
                        layer = %il.name,
                        "the layer declares a calibration but cannot be calibrated: {e:#}",
                    );
                    continue;
                }
            };
            calibrate_layer(
                &self.engine,
                &CalibrateLayer { spec, base },
                &CalibrateFrame {
                    system_prompt: &self.formatted_prompt,
                    config: &self.refresh_config,
                    thinking: turn_triggers(self, ThinkMode::Balanced, ToolMode::None),
                },
            );
        }
    }

    /// Fold the chunk exclusions ([`Self::fold_chunk_exclusions`]) on a
    /// thread of their own, so the next pass starts without waiting for them —
    /// and not again while a fold is running: the pass after it folds what was
    /// written meanwhile.
    ///
    /// **Not at the end of the pass.** A pass ending with the scans reads
    /// nothing while they run; on its own thread a fold's launches are small
    /// jobs the scheduler runs between waves.
    pub(crate) fn fold_chunk_exclusions_behind(self: &Arc<Self>) {
        if self.branch_layers.is_empty() || self.learning_exclusions.swap(true, Ordering::SeqCst) {
            return;
        }
        let state = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name("zend-exclusions".into())
            .spawn(move || {
                state.fold_chunk_exclusions();
                state.learning_exclusions.store(false, Ordering::SeqCst);
            });
        if let Err(e) = spawned {
            self.learning_exclusions.store(false, Ordering::SeqCst);
            tracing::warn!(
                target: "zend::calibrate",
                "the chunk exclusions were not folded (no thread): {e}"
            );
        }
    }

    /// Fold what changed into the chunk exclusion of every branch layer that
    /// declares a `calibrate:` block, under the layer's exclusion lease: one
    /// member in the cluster folds a group, and every member installs the cut
    /// its records replicate (`docs/incremental_chunk_exclusion.md`). The
    /// learner keeps its state between passes, so a fold scans only what was
    /// appended since the last; after a restart it resumes from the records.
    pub(super) fn fold_chunk_exclusions(&self) {
        for (il, _) in &self.branch_layers {
            if crate::ingest_cancel::cancelled() {
                return;
            }
            if il.calibrate.is_none() {
                continue;
            }
            let Some(group) = self.refresh_builder.id_for_group(&il.group) else {
                continue;
            };
            let _lease = match ingest_lease::share_exclusion(&il.group) {
                ingest_lease::Share::Mine(lease) => lease,
                ingest_lease::Share::Theirs => {
                    // Another member folds it: what this one held would only
                    // pin signatures the substrate has let go, and goes stale
                    // the moment the other folds.
                    self.exclusion_learners
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .remove(&group);
                    continue;
                }
            };
            // Each competition in the fold groups its live scans weigh.
            let Some(policy) = self
                .refresh_builder
                .schema()
                .layers
                .iter()
                .flat_map(|l| &l.groups)
                .find(|g| g.id == group)
                .map(|g| &g.policy)
            else {
                continue;
            };
            let content = (
                GroupSet::weighed(&policy.layer_weights, FOLD_GROUPS),
                policy.scan.gallery_keep,
            );
            let calibration = (
                GroupSet::weighed(&policy.calibration_layer_weights, FOLD_GROUPS),
                policy.scan.calibration_gallery_keep,
            );
            let scanner = self.engine.group_scanner();
            let t_fold = Instant::now();
            let mut learners = self
                .exclusion_learners
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let learner = learners
                .entry(group)
                .or_insert_with(|| ExclusionLearner::new(group));
            match scanner.fold_chunk_exclusion(learner, content, calibration) {
                Ok(Ok(r)) => tracing::info!(
                    target: "zend::calibrate",
                    layer = %il.name,
                    loaded = r.loaded,
                    sample = r.sample,
                    appended = r.counts.appended,
                    removed = r.counts.removed,
                    entered = r.counts.entered,
                    left = r.counts.left,
                    rescanned = r.counts.rescanned,
                    caught_up = r.counts.caught_up,
                    dirty = r.dirty.0,
                    entries = r.dirty.1,
                    content_tokens = r.n[0],
                    calibration_tokens = r.n[1],
                    chunks = r.content.chunks,
                    votes = r.content.votes,
                    cut = r.content.cut.len(),
                    weakest_cut_votes = ?r.content.weakest_cut_votes,
                    calibration_chunks = r.calibration.chunks,
                    calibration_votes = r.calibration.votes,
                    calibration_cut = r.calibration.cut.len(),
                    calibration_weakest_cut_votes = ?r.calibration.weakest_cut_votes,
                    written = r.written,
                    tombstoned = r.tombstoned,
                    elapsed_ms = t_fold.elapsed().as_millis() as u64,
                    "the layer's anomalous-chunk exclusion folded",
                ),
                Ok(Err(e)) => tracing::warn!(
                    target: "zend::calibrate",
                    layer = %il.name,
                    "folding the layer's chunk exclusion failed; the next fold resumes from its records: {e:#}",
                ),
                Err(refused) => tracing::warn!(
                    target: "zend::calibrate",
                    layer = %il.name,
                    "the chunk-exclusion fold was refused: {refused:?}",
                ),
            }
        }
    }

    /// The prefilled template a `Folders`/`Files` ingest layer's units fork
    /// their own conversation from — see [`InferenceState::ingest_bases`].
    /// `Err` only for a layer name that isn't one of those two modes (a
    /// caller bug: `ingest_bases` is built for every such layer at boot).
    pub(super) fn ingest_base(&self, layer_name: &str) -> anyhow::Result<&Sequence> {
        self.ingest_bases
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("no ingest base built for layer '{layer_name}'"))
    }

    /// One pass of the background ingest worker (`crate::ingest_worker`) —
    /// at startup, and after every fetch that moved a branch on origin.
    ///
    /// The branch layers (`repo_map`, `code_reading`) are planned against
    /// every record branch together (`crate::branch_ingest`): what no branch
    /// holds is tombstoned, what nothing holds yet is ingested, each unit at
    /// the commit it was found on. A raw layer in the registry re-reads its
    /// folder's ChatML records; on a content-hash change it mints a fresh
    /// timeline, re-prefills, tombstones the old, and swaps it in.
    ///
    /// Stale-better-than-missing holds throughout: an old generation stays
    /// alive until its replacement commits. Returns `Ok(true)` if any layer
    /// changed.
    pub(crate) fn refresh_ingest_layers(&self) -> anyhow::Result<bool> {
        if self.ingest_layers.is_empty() {
            return Ok(false);
        }
        // Throwaway progress sink (a refresh, not the model-load lifecycle) —
        // silent so a pass doesn't log "load step started Loading model".
        let progress = Arc::new(LoadProgress::silent());
        let mut any = false;
        if !self.branch_layers.is_empty() {
            let passes = self.layer_passes()?;
            let ctx = self.refresh_ctx();
            any |= self
                .branch_ingest
                .pass(&ctx, &self.workspace, &passes, &progress)?;
            // **Read first, calibrate after — and only after.** The branch
            // pass reads files without calibrating them; the sweep that
            // follows calibrates what it committed and whatever an earlier
            // pass or a restart left uncalibrated. A sweep ahead of the pass
            // as well held every file back for its whole length: after the
            // 2026-11-08 restarts, each boot spent 15–30 minutes calibrating
            // the same 160 units, and with a deploy every ~20 minutes no file
            // was read again at all.
            self.calibrate_layers();
        }
        for il in &self.ingest_layers {
            // Cooperative shutdown: stop the background reconcile between layers so
            // a Ctrl-C during a post-`ready` re-ingest stops promptly (the per-file
            // worker pool also polls the same cancel). `shutdown` then drains the
            // engine via the ready path.
            if crate::ingest_cancel::cancelled() {
                break;
            }
            if il.mode != IngestMode::Raw {
                continue;
            }
            let snapshot = {
                let convs = self.ingest_convs.lock().unwrap();
                convs
                    .get(&il.name)
                    .map(|c| (c.sequence.timeline_id(), c.state.clone()))
            };
            let Some((old_timeline, prior_state)) = snapshot else {
                continue;
            };
            let ctx = self.refresh_ctx();
            let outcome = crate::raw_read::refresh_raw(
                &ctx,
                &self.workspace.root().join(&il.folder),
                &prior_state,
                old_timeline,
                &progress,
                &il.name,
                &il.group,
            )?;
            if let crate::raw_read::RefreshOutcome::Replaced { sequence, state } = outcome {
                self.ingest_convs
                    .lock()
                    .unwrap()
                    .insert(il.name.clone(), IngestConv { sequence, state });
                any = true;
                tracing::info!(layer = %il.name, "raw ingest layer refreshed");
            }
        }
        Ok(any)
    }
}
