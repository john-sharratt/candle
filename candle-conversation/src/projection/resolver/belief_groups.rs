//! Scores a layer's belief-driven turn groups against their own turns, one group
//! at a time.
//!
//! A turn's retrieval target is the turn itself, so each candidate is its own slot
//! and is scored against its stored window. Substrate reads are confined to the
//! gathering phase and dropped before the GPU scan, so a long scan never blocks
//! writers. The scan covers every conversation in a group, not only the first
//! registered one. A layer is never filtered out of its own target's scan by the
//! cross-layer retrieval flag, or an append-only ingest layer could not see its own
//! earlier turns.

use super::*;

impl Conversation {
    /// Score every belief-driven **turn group** in `layer` against its own turns
    /// (self-match), folding the fresh per-turn scores into `scores`.
    ///
    /// The turn-group analogue of [`Self::score_belief_collections`]: where a
    /// collection scans a tag-scoped gallery of *other* turns mapped to section
    /// slots, a turn group's retrieval target IS the turn itself, so each
    /// candidate turn is its own slot (identity map) and the probe scores against
    /// each turn's stored `WideQSig` window directly. A `Sequence` (recency) group
    /// is skipped — it is never scanned; a working-set group is scanned, since
    /// its momentum is built from these scores. The group's timeline is resolved the
    /// same way the projection will (`resolve_turn_timeline(Some(target), …)`) and
    /// its turns enumerated `0..turn_count(timeline)` exactly as selection does,
    /// so the `(timeline, index)` keys line up with what selection reads back.
    pub fn score_belief_groups(
        &self,
        layer: &LayerSchema,
        target: ProjectionTarget,
        probe: &[WideQSig],
        probe_q: Option<&[WideQSig]>,
        scores: &mut ProjectionScores,
        observe: Observe<'_>,
        arena: Option<&dyn BeliefArena>,
    ) -> Vec<(GroupId, Vec<(TurnKey, f32)>)> {
        let mut per_group: Vec<(GroupId, Vec<(TurnKey, f32)>)> = Vec::new();
        if probe.is_empty() {
            return per_group;
        }
        // Taken out of retrieval for this boot (`zend --disable-layer`). Its
        // turns are still in the substrate with their signatures intact — this
        // declines to score them, so they cannot be selected, which is the whole
        // of what the flag promises. Re-enabling restores them immediately; the
        // flag is a read-time filter, not a deletion.
        //
        // Exempt when `layer` IS the projection's own target — mirroring the
        // assembly loop's `layer_is_target` exemption in `project.rs` ("the
        // target layer is never skipped — that would leave the projection with
        // nothing to emit"). `gathered = false` promises "no OTHER conversation
        // draws from this layer"; it was never meant to promise "this layer's
        // own conversations can't see their own history" too. An append-only
        // ingest layer (`code_reading`, `repo_map`) scores self-local here —
        // candidates scoped to `target.timeline` alone, never another
        // conversation's — so honouring `!gathered` unconditionally silently
        // zeroed every such layer's OWN turn-group candidates the moment it was
        // taken out of cross-layer gather, with no way back in: a later turn
        // in the very conversation that owns this layer could never again see
        // turns from earlier in itself. Measured: a `code_reading` per-file
        // conversation's closing turn, several `file_read` rounds deep, lost
        // every earlier round in the same conversation and could not tell it
        // had already read the whole file.
        if !layer.gathered && layer.id != target.layer {
            return per_group;
        }
        // The substrate read guard is scoped to Phase A, never held across Phase
        // B's GPU scan.
        //
        // A `RwLockReadGuard` drops at end of scope, so binding it here would
        // hold it across `scan_weighted` — blocking EVERY substrate writer for
        // the scan's whole duration, most importantly the persistence thread's
        // install-warm (the same starvation the wave loop's livelock guard
        // documents). Nothing in Phase B needs it: `wide_sig_window` and
        // `decoded_seams` hand back shared buffers and `couplings_of` returns by value,
        // so the scan borrows owned data, not the substrate. The borrow checker
        // is what keeps this honest — if Phase B ever grows a substrate access,
        // the `drop(sub)` below stops compiling rather than silently re-extending
        // the hold.
        //
        // Self-local when the target is an append-only ingest layer: mask EVERY
        // belief group to the target timeline, so an ingest scope-summary scores
        // only its own scope (mirror of the `TargetedRead::group_turns` selection
        // mask). Dialogue targets are never append-only, so cross-file scoring
        // there is untouched.
        let self_local = self.inner.read().is_append_only_layer(target.layer);
        for group in &layer.groups {
            if !group.is_scanned() {
                continue;
            }
            // Per-group phase split. `groups_us` in the caller covers a whole
            // layer's worth of this loop, which is one level too coarse to
            // attribute a slow scan: assembling the candidate set (host, walks
            // every turn of every timeline in the group) and scoring it (one
            // batched GPU launch) are different costs with different fixes, and
            // a scan that has grown to tens of seconds looks identical from
            // outside whichever one it is. Timed per group so the group that
            // grew is named, not just the layer.
            let t_select = Instant::now();
            let sub = self.inner.read();
            // Score EVERY conversation in the group, not just the first. A belief
            // group like `code_reading` declares one timeline per file; scoring
            // only the first-registered timeline left every other file at score 0
            // — structurally unreachable, no matter how well it matched the query.
            // A belief group is never the projection target (the target is the
            // Sequence dialogue group, skipped above), but mirror the target mask
            // anyway so the invariant holds if that ever changes.
            // A scoped group offers only what the target's own scope names —
            // `Substrate::scoped_timelines_for_group` — so an out-of-scope
            // conversation is never scanned, and never teaches the group's
            // backgrounds on this target's behalf.
            // A layer the target compounds on is scored across conversations
            // like a dialogue's, its own layer still self-local.
            let self_local = self_local && sub.is_self_local_group(target.layer, group.id);
            let masked = self_local || group.id == target.group;
            // Who the turns this scan reads are held for: a masked read of a
            // conversation-scoped layer is that conversation's alone, anything
            // else every conversation's (`docs/gallery_residency.md` §2).
            let holder = if masked {
                Holder::masked_to(layer.gather_scope, target.timeline)
            } else {
                Holder::Shared
            };
            // The group's files as this target reads them: kept from an earlier
            // scan while nothing they were gathered from has moved
            // (`group_assembly`), gathered afresh otherwise.
            let by_kind = group.policy.scan.normalize == ScanNorm::KindZ;
            // A masked scan, and a compounded one that leaves the target's
            // lineage out, are the target's own; any other is every target's
            // whose scope of the group names the same conversations.
            let whose = if masked || sub.is_append_only_layer(target.layer) {
                Whose::Target(target.timeline)
            } else {
                Whose::Scope(sub.scope_digest(group.id, target.timeline))
            };
            let key = AssemblyKey {
                group: group.id,
                whose,
            };
            let stamp = AssemblyStamp::of(&sub, key, target.group);
            let (mut scope_us, mut tags_us, mut exclude_us, mut kinds_us) = (0, 0, 0, 0);
            let select_us = t_select.elapsed().as_micros() as u64;
            let t_assemble = Instant::now();
            let assembly = match held(
                &self.group_assemblies,
                key,
                stamp,
                &group.policy.tags,
                by_kind,
            ) {
                Some(kept) => {
                    GROUP_SCAN_SPLIT.set({
                        let mut s = GROUP_SCAN_SPLIT.get();
                        s.assemblies_reused += 1;
                        s
                    });
                    kept
                }
                None => {
                    let t_scope = Instant::now();
                    let mut timelines: Vec<TimelineId> = if masked {
                        vec![target.timeline]
                    } else if sub.is_append_only_layer(target.layer) {
                        // A compounded group never offers the target's own
                        // lineage: it is the target's opening already
                        // (`compounding_lineage`).
                        let lineage: HashSet<TimelineId> =
                            sub.inherited_chain(target.timeline).into_iter().collect();
                        sub.scoped_timelines_for_group(group.id, target.timeline)
                            .filter(|tl| !lineage.contains(tl))
                            .collect()
                    } else {
                        sub.scoped_timelines_for_group(group.id, target.timeline)
                            .collect()
                    };
                    scope_us = t_scope.elapsed().as_micros() as u64;
                    let t_tags = Instant::now();
                    // A tagged group reads only the conversations carrying its
                    // tags — the same scope projection applies to its
                    // candidates, applied here so an out-of-scope conversation
                    // is neither scanned nor allowed to teach the group's
                    // backgrounds. See `Builder::set_group_tags`. Never the
                    // target's own group, which is its own timeline already —
                    // the same exemption projection makes.
                    if !group.policy.tags.is_empty() && group.id != target.group {
                        let tags = &group.policy.tags;
                        timelines.retain(|&tl| {
                            (0..sub.turn_count(tl))
                                .any(|i| sub.turn_has_any_tag(tl, TurnIndex(i), tags))
                        });
                    }
                    tags_us = t_tags.elapsed().as_micros() as u64;
                    // ── Phase A: assemble every file's exchanges + gallery
                    // windows. All files in the group are built first so the
                    // whole group can be scored in ONE batched GPU launch
                    // (per-file z), or file-by-file on the CPU fallback — the two
                    // paths are numerically equivalent up to fast-math ULP (same
                    // ranking; see `examples/gpu_belief_parity.rs`).
                    let n_timelines = timelines.len();
                    let (mut files, turns_walked) = assemble_file_scans(&sub, timelines);
                    let t_exclude = Instant::now();
                    cut_excluded(&mut files, &sub.chunk_exclusion(group.id));
                    exclude_us = t_exclude.elapsed().as_micros() as u64;
                    let t_kinds = Instant::now();
                    // Each file's kind, its path's extension, when the group
                    // reads its files against their own kind — one metadata
                    // lookup a file.
                    let kinds: Option<Vec<String>> = by_kind.then(|| {
                        files
                            .iter()
                            .map(|f| {
                                sub.custom_of(f.timeline)
                                    .and_then(|m| m.get(Substrate::PATH_KEY))
                                    .map_or(String::new(), |p| path_kind(p).to_string())
                            })
                            .collect()
                    });
                    kinds_us = t_kinds.elapsed().as_micros() as u64;
                    let built = Arc::new(GroupAssembly {
                        stamp,
                        tags: group.policy.tags.clone(),
                        kinds,
                        files,
                        n_timelines,
                        turns_walked,
                        plans: ScanPlans::default(),
                    });
                    keep(&self.group_assemblies, key, Arc::clone(&built));
                    built
                }
            };
            let n_timelines = assembly.n_timelines;
            let turns_walked = assembly.turns_walked;
            let files = &assembly.files;
            let kinds = &assembly.kinds;
            let t_lineage = Instant::now();
            // The target's lineage, when the group leaves it out: the priming
            // chain the target opens on already.
            let lineage: HashSet<TimelineId> = if group.policy.scan.exclude_lineage {
                sub.inherited_chain(target.timeline)
                    .into_iter()
                    .filter(|&tl| tl != target.timeline)
                    .collect()
            } else {
                HashSet::new()
            };
            let lineage_us = t_lineage.elapsed().as_micros() as u64;
            let t_release = Instant::now();
            // End of Phase A — release the substrate before the scan. Every
            // `FileScan` owns its data (`arcs_kept` holds `Arc`s), so nothing
            // below borrows the guard.
            drop(sub);
            let release_us = t_release.elapsed().as_micros() as u64;
            let assemble_us = t_assemble.elapsed().as_micros() as u64;
            let n_windows: usize = files.iter().map(|f| f.windows.len()).sum();
            if files.is_empty() {
                continue;
            }
            let t_scan = Instant::now();

            // ── Phase B: score every file (`belief_files::scan_file_scans`). Per-
            // layer-group vote weights come from the group's `policy.layer_weights`
            // (empty ⇒ uniform — repo_map peaks on L46 (§83), configured in the
            // schema YAML). Per probe, per file: `(fused, mass_base)`, the tail and
            // question windows riding one launch. ──
            let plans = &assembly.plans;
            let scan_files = |probes: &[&[WideQSig]]| {
                scan_file_scans(files, group, probes, arena, holder, plans)
            };
            // Concept F: the pinned question window scans as a second probe,
            // max-fused with the tail per exchange AFTER normalization.
            let question = match probe_q {
                Some(q) if group.policy.scan.question_pin && !q.is_empty() => Some(q),
                _ => None,
            };
            let mut scanned = match question {
                Some(q) => scan_files(&[probe, q]),
                None => scan_files(&[probe]),
            };
            let question_scan = question.and_then(|_| scanned.pop());
            let q_base_per_file: Option<Vec<Vec<f32>>> = question_scan
                .as_ref()
                .map(|per_file| per_file.iter().map(|(_, base)| base.clone()).collect());
            let fresh_q_per_file: Option<Vec<Vec<f32>>> = question_scan
                .map(|per_file| per_file.into_iter().map(|(fused, _)| fused).collect());
            let tail_scan = scanned.pop().unwrap_or_default();
            let fresh_per_file: Vec<Vec<f32>> =
                tail_scan.iter().map(|(fused, _)| fused.clone()).collect();
            let mass_base_per_file: Vec<Vec<f32>> =
                tail_scan.into_iter().map(|(_, base)| base).collect();

            // Phase B ends here. Split from Phase C because the boundary scan is
            // the only one that also observes into the normalization levels
            // under a shared lock, so scoring time and lock time are reported
            // apart.
            let score_us = t_scan.elapsed().as_micros() as u64;
            let t_normalize = Instant::now();
            // ── Phase C: per-file normalize → Q-fuse → locality → stamp ──────────
            // Read the raw scores against each EXCHANGE's learned background so
            // selection compares candidates on a common band, not a shared
            // absolute scale (docs/provenance_score_normalization.md). Scope =
            // group@timeline — a re-scan mints a new timeline, hence a fresh scope,
            // resetting learning for the regenerated clusters; child = the exchange's
            // head-turn index (stable across scans). Concept F max-fuses the
            // normalized question scan; Concept C drags neighbors afterwards.
            let mut cands: Vec<(TurnKey, f32)> = Vec::new();
            // Concept B: mass rides on the RAW fused scores (pre-normalization)
            // — measured (results doc §25): the normalized band compresses
            // genuine concentration away, while the raw top-1-share² formula
            // keeps the code-vs-history contrast.
            let mut group_raw: Vec<f32> = Vec::new();
            // As on the collection path above: the tail and question scans are
            // sums over probes of very different lengths, so each is normalized
            // against the reference on its own length. Learning fires only on
            // the once-per-turn seal scan, not on every reprojection, and folds
            // the TAIL scan's raw scores (the stored whole-turn probe).
            let t_tail = probe.len();
            let t_q = probe_q.map_or(1, |q| q.len());
            // `belief_files::competitions` pools exactly the files of one
            // exchange — or, scored by file, every file.
            let in_pool =
                |f: &FileScan| f.n_slots == 1 || group.policy.scan.cases == ScanCases::File;
            // The pooled documents are one competition, so one scope, normalized
            // in one call — a document too new for a background of its own is
            // read against the others — and folded together (`observe` folds a
            // source into a scope once, so per-document folds would teach only
            // the first).
            let pool_scope = ScopeKey::pool(group.id.raw() as u64);
            let pool_pairs = |per_file: &[Vec<f32>]| -> Vec<(ChildKey, f32)> {
                files
                    .iter()
                    .zip(per_file)
                    .filter(|(f, _)| in_pool(f))
                    .flat_map(|(f, scores)| {
                        (0..f.n_slots).map(move |slot| {
                            (
                                ChildKey::timeline(f.timeline.raw()),
                                scores.get(slot).copied().unwrap_or(0.0),
                            )
                        })
                    })
                    .collect()
            };
            let pool_raw = pool_pairs(&fresh_per_file);
            let band = |out: Vec<(ChildKey, f32)>| out.into_iter().map(|(_, s)| s);
            // Under `kind_z` every file is pooled (`cases: file`), its one
            // score standing for each of its exchanges.
            let by_kind = |per_file: &[Vec<f32>], kinds: &[String]| -> Vec<f32> {
                let raw: Vec<f32> = per_file
                    .iter()
                    .map(|s| s.first().copied().unwrap_or(0.0))
                    .collect();
                files
                    .iter()
                    .zip(kind_z_band(&raw, kinds))
                    .flat_map(|(f, b)| repeat_n(b, f.n_slots))
                    .collect()
            };
            let pool_band = |per_file: &[Vec<f32>], pairs: &[(ChildKey, f32)], t: usize| {
                match &kinds {
                    Some(kinds) => by_kind(per_file, kinds),
                    None => band(self.normalization.normalize(&pool_scope, pairs, t)).collect(),
                }
                .into_iter()
            };
            let mut pool_normed = pool_band(&fresh_per_file, &pool_raw, t_tail);
            let mut pool_normed_q = fresh_q_per_file
                .as_ref()
                .map(|per_file| pool_band(per_file, &pool_pairs(per_file), t_q));
            for (fi, (f, fresh)) in files.iter().zip(&fresh_per_file).enumerate() {
                let timeline = f.timeline;
                let (normed, normed_q): (Vec<f32>, Option<Vec<f32>>) = if in_pool(f) {
                    (
                        pool_normed.by_ref().take(f.n_slots).collect(),
                        pool_normed_q
                            .as_mut()
                            .map(|q| q.by_ref().take(f.n_slots).collect()),
                    )
                } else {
                    let scope = ScopeKey::turn_group(group.id.raw() as u64, timeline.raw());
                    let pairs = |scores: &[f32]| -> Vec<(ChildKey, f32)> {
                        (0..f.n_slots)
                            .map(|slot| {
                                (
                                    ChildKey::turn(f.arc_turn[f.ex_ranges[slot].start].0 as u64),
                                    scores.get(slot).copied().unwrap_or(0.0),
                                )
                            })
                            .collect()
                    };
                    let raw_pairs = pairs(fresh);
                    let cache = &*self.normalization;
                    let normed = band(cache.normalize(&scope, &raw_pairs, t_tail)).collect();
                    let normed_q = fresh_q_per_file.as_ref().map(|per_file| {
                        band(cache.normalize(&scope, &pairs(&per_file[fi]), t_q)).collect()
                    });
                    // A turn group's retrieval target IS the turn, so its gather
                    // scope is the group's own; `tags` on the group route the
                    // same way collections are routed above. A live turn is
                    // about no known exchange, so it teaches them all.
                    if let (true, Some(source)) =
                        (observe.teaches(&group.policy.tags), observe.source())
                    {
                        cache.observe(&scope, source, &raw_pairs, None, t_tail);
                    }
                    (normed, normed_q)
                };
                let mut fused: Vec<f32> = (0..f.n_slots)
                    .map(|slot| {
                        let t = normed.get(slot).copied().unwrap_or(0.0);
                        let q = normed_q
                            .as_ref()
                            .and_then(|nq| nq.get(slot).copied())
                            .unwrap_or(0.0);
                        t.max(q)
                    })
                    .collect();
                // A link of the target's lineage competed in the scan and is not
                // offered: the target opens on it already.
                if lineage.contains(&timeline) {
                    group_raw.extend_from_slice(&mass_base_per_file[fi]);
                    continue;
                }
                // Concept C: a hit drags its timeline neighbors into contention
                // (exchange granularity, max-not-sum, radius grows with score).
                let boosted_flags: Vec<bool> = match &group.locality {
                    Some(loc) => {
                        let (dragged, flags) = loc.apply(&fused);
                        fused = dragged;
                        flags
                    }
                    None => vec![false; f.n_slots],
                };
                // Stamp EVERY member turn of an exchange with its (shared) normalized
                // score, so provenance selecting either half brings in the whole
                // round-trip — never half a tool call.
                for (slot, r) in f.ex_ranges.iter().enumerate() {
                    let sc = fused.get(slot).copied().unwrap_or(0.0);
                    for ai in r.clone() {
                        let idx = f.arc_turn[ai];
                        scores.set_turn(timeline, idx, sc);
                        if boosted_flags[slot] {
                            scores.mark_locality_boost(timeline, idx);
                        }
                        cands.push((TurnKey::new(timeline, idx), sc));
                    }
                }
                group_raw.extend_from_slice(&mass_base_per_file[fi]);
            }
            // A kind-read pool keeps no background: there is nothing to teach.
            if let (false, true, Some(source)) = (
                pool_raw.is_empty() || kinds.is_some(),
                observe.teaches(&group.policy.tags),
                observe.source(),
            ) {
                self.normalization
                    .observe(&pool_scope, source, &pool_raw, None, t_tail);
            }
            if cands.is_empty() {
                continue;
            }
            // Concept B: the group's attention mass over every candidate
            // exchange's raw fused score.
            scores.set_group_mass(
                group.id,
                attention_mass(
                    &group_raw,
                    0.0,
                    f32::MAX,
                    group.policy.scan.mass_top_k,
                    group.policy.scan.mass_rho,
                ),
            );
            if let Some(tail) = scan_stats(&group_raw) {
                let q_raw: Option<Vec<f32>> = q_base_per_file
                    .as_ref()
                    .map(|per_file| per_file.iter().flatten().copied().collect());
                let question = q_raw.as_deref().and_then(scan_stats);
                scores.set_group_scan(group.id, GroupScan { tail, question });
            }
            // Where this group's scan time went, and what it walked to spend it.
            // `assemble_us` is host work proportional to the group's turns;
            // `score_us` is the batched launch (plus the CPU fallback when the
            // arena declines); `normalize_us` is Phase C, which takes the
            // normalization lock and — on the seal scan only — observes into the
            // backgrounds. The counts say whether a phase grew because the corpus
            // did.
            //
            // `trace!`, because this is per group and there are ~17 of them per
            // scan: it was 1,262 lines of a 4,315-line daemon log, and the
            // `belief scan phase split` engine event above already carries the
            // per-scan totals that answer "did a phase grow". Raise the target to trace
            // when the question is which *group* grew.
            let normalize_us = t_normalize.elapsed().as_micros() as u64;
            tracing::trace!(
                target: "candle_conversation::provenance",
                layer = %layer.name,
                group = group.id.raw(),
                timelines = n_timelines,
                turns = turns_walked,
                windows = n_windows,
                seal = observe.source().is_some(),
                select_us,
                assemble_us,
                score_us,
                normalize_us,
                "belief group scan"
            );
            GROUP_SCAN_SPLIT.set({
                let mut s = GROUP_SCAN_SPLIT.get();
                s.scanned += 1;
                s.windows += n_windows as u64;
                s.turns += turns_walked as u64;
                s.select_us += select_us;
                s.assemble_us += assemble_us;
                s.score_us += score_us;
                s.normalize_us += normalize_us;
                s.scope_us += scope_us;
                s.tags_us += tags_us;
                s.exclude_us += exclude_us;
                s.kinds_us += kinds_us;
                s.lineage_us += lineage_us;
                s.release_us += release_us;
                s
            });
            per_group.push((group.id, cands));
        }
        per_group
    }
}
