//! Scores a conversation's belief-driven section collections against their
//! tag-scoped galleries.
//!
//! Each collection member is read against its own background, and a warm-up pass
//! skips collections it cannot teach, since the probe's tags fall outside the
//! scope it would learn. The loop returns the fused scores and an ungated mass,
//! because the gate that removes a scan's spikes also removes the concentration
//! signal the attention mass is keyed on. An empty gallery is the steady state of
//! an uncalibrated workspace, so its warning is logged once per collection and
//! repeats drop to trace.

use super::*;

impl Conversation {
    /// Score every belief-driven collection in the shared system prompt against
    /// its tag-scoped gallery, using `probe` as the query window, into per-section
    /// [`ProjectionScores`].
    ///
    /// Shared by the two probes that drive selection: the scheduler's live
    /// reproject scan (probe = live wide-Q gather over the decode window) and
    /// the post-turn projection event (probe = the finished turn's stored
    /// signature). A collection with an empty gallery or no sections contributes
    /// nothing — its sections read `0.0`.
    ///
    /// With an `arena`, each collection scans as ONE global-z segment on the GPU
    /// gallery arena (numerically equivalent to the CPU `score_slots_weighted`
    /// up to fast-math ULP — same ranking); collection windows are the same turn
    /// streams the turn-group scan touches, so their arena residency is shared.
    /// Without one (seal-time learning, non-CUDA), the CPU path runs unchanged.
    ///
    /// Concepts of `docs/provenance_adaptive_projection.md` applied here:
    /// **G** — the collection policy's `fusion` mode drives the scan
    /// (`content_gated` = full score gated by a one-hot gate-group scan, so the
    /// GPU arena serves both scans); **F** — with `question_pin` and a
    /// `probe_q`, the pinned question window scans separately and the
    /// per-section score is the max of the two normalized scans; **A** — every
    /// collection reads its members against their backgrounds
    /// (`ScopeKey::Collection`), `observe` teaching them only on the
    /// once-per-turn seal scan; **B** — the collection's attention mass rides
    /// on the returned scores.
    pub fn score_belief_collections(
        &self,
        sp: &SystemPromptSchema,
        probe: &[WideQSig],
        probe_q: Option<&[WideQSig]>,
        observe: Observe<'_>,
        arena: Option<&dyn BeliefArena>,
    ) -> ProjectionScores {
        let mut scores = ProjectionScores::new();
        if probe.is_empty() {
            return scores;
        }
        for item in &sp.items {
            let SystemPromptItem::Collection(coll) = item else {
                continue;
            };
            let n = coll.sections.len();
            if n == 0 {
                continue;
            }
            // **A warm-up pass skips the collections it cannot teach.**
            //
            // The dialogue replay (`score_warm_probe`) hands the
            // WHOLE system prompt to this loop, so every collection is scanned —
            // and `teaches` then discards the observation for any collection the
            // probe's tags are not inside. The caller drops the returned scores
            // (`let _ = …`), so that scan produces nothing at all: on the tool
            // corpus it is up to 512 probes against ~3,000 exemplars, at load.
            //
            // Only a teaching pass may be skipped this way. `Observe::No` is the
            // LIVE reprojection, whose scores are the whole point — `teaches` is
            // false for every collection there, so gating on `teaches` alone
            // would silently disable tool selection.
            if matches!(observe, Observe::Yes { .. }) && !observe.teaches(&coll.policy.tags) {
                continue;
            }
            let members: Vec<String> = coll.sections.iter().map(|s| s.name.clone()).collect();
            let t_gallery = Instant::now();
            let (windows, slots, sids) =
                self.belief_gallery(&coll.name, &coll.policy.tags, &members);
            let gallery_us = t_gallery.elapsed().as_micros() as u64;
            if windows.is_empty() {
                // The belief loop has nothing to score against, so selection falls
                // back to declaration order (the first tool in the catalog) and
                // pins there. This is THE signature of "wrong tool every turn".
                // An empty gallery is also the persistent steady state of an
                // uncalibrated workspace — and during the calibration load-phase
                // the scan runs on every one of hundreds of per-tool example
                // decodes while the gallery is still being built, so the repeat
                // fires en masse. WARN once per collection (the actionable
                // signal), then demote repeats to TRACE so they don't bury the log
                // during calibration; steady-state behaviour is unchanged.
                static EMPTY_GALLERY_WARNED: OnceLock<papaya::HashSet<String>> = OnceLock::new();
                let first = EMPTY_GALLERY_WARNED
                    .get_or_init(papaya::HashSet::new)
                    .pin()
                    .insert(coll.name.clone());
                if first {
                    tracing::warn!(
                        target: "candle_conversation::belief",
                        collection = %coll.name,
                        tags = ?coll.policy.tags,
                        sections = n,
                        probe_windows = probe.len(),
                        "belief gallery EMPTY — no tag-scoped gallery turns; tool selection \
                         falls back to catalog order (repeats logged at trace)"
                    );
                } else {
                    tracing::trace!(
                        target: "candle_conversation::belief",
                        collection = %coll.name,
                        "belief gallery still empty"
                    );
                }
                continue;
            }
            // Per-layer-group weights from the collection's `policy.layer_weights`
            // (empty ⇒ uniform — the tool default). Configured in the schema YAML.
            // GPU: the collection is ONE global-z segment over the resident arena
            // (windows keyed by their turn stream ids — shared with the group
            // scan's residency). CPU fallback preserves today's behaviour. The
            // fusion mode's scan law (`fused_scan`) is identical on both
            // backends: `content_gated` runs the additive scan twice (full +
            // one-hot gate group) and gates; consensus modes are CPU-only.
            // Returns `(fused, mass_base)`: the policy-fused per-section scores
            // and the UNGATED additive sum the Concept B mass keys on (the gate
            // deliberately removes the concentrated spike mass must see —
            // results doc §25).
            // The host path reads decoded windows: decoded on its first use in
            // this pass, shared by both probes, and dropped with the pass.
            let host_windows: OnceCell<Vec<Vec<WideQSig>>> = OnceCell::new();
            let scan_probe = |p: &[WideQSig]| -> (Vec<f32>, Vec<f32>) {
                let ranged: Vec<(usize, Range<usize>)> = (0..windows.len())
                    .map(|i| (i, 0..windows[i].len()))
                    .collect();
                if ranged.is_empty() {
                    return (vec![0.0; n], vec![0.0; n]);
                }
                let gpu_fresh: Option<(Vec<f32>, Vec<f32>)> = arena.and_then(|arena| {
                    let make_segment = || PagedSegment {
                        windows: ranged
                            .iter()
                            .map(|(i, r)| PagedWindow {
                                sid: sids[*i],
                                fingerprint: windows[*i].fingerprint(),
                                turn: (&windows[*i]).into(),
                                start: r.start,
                                end: r.end,
                                case: slots[*i],
                            })
                            .collect(),
                        n_cases: n,
                        // A collection's gallery is drawn from every
                        // conversation's turns.
                        holder: Holder::Shared,
                        groups: GroupSet::weighed(&coll.policy.layer_weights, arena.n_groups()),
                        keep: coll.policy.scan.gallery_keep,
                        law: coll.policy.scan.law,
                    };
                    // Every probe token's vote in each fold group, as the needle
                    // gate meets them: which member each token's best match
                    // named, and how strongly. What a wrong pick was built from.
                    if tracing::enabled!(target: "candle_conversation::belief::tokens", tracing::Level::TRACE)
                    {
                        if let Ok(votes) = arena.scan_votes(&[make_segment()], &[p]) {
                            let rows: Vec<String> = (0..p.len())
                                .map(|t| {
                                    let groups: Vec<String> = (0..votes.n_groups)
                                        .map(|g| match votes.at(t, g, 0) {
                                            Some((c, v)) => format!(
                                                "{}:{v:.0}",
                                                coll.sections.get(c).map_or("?", |s| s.name.as_str())
                                            ),
                                            None => "-".to_string(),
                                        })
                                        .collect();
                                    format!("t{t}[{}]", groups.join(" "))
                                })
                                .collect();
                            tracing::trace!(
                                target: "candle_conversation::belief::tokens",
                                collection = %coll.name,
                                tokens = p.len(),
                                votes = %rows.join(" "),
                                "probe token votes"
                            );
                        }
                    }
                    let scan_w = |w: &[f32]| -> Option<Vec<f32>> {
                        match arena.scan_weighted(&[make_segment()], &[p], w) {
                            Ok(mut out) => Some(out.pop().unwrap_or_default()),
                            Err(e) => {
                                tracing::warn!(
                                    target: "candle_conversation::provenance",
                                    "GPU collection scan unavailable, using CPU: {e}"
                                );
                                None
                            }
                        }
                    };
                    match coll.policy.scan.fusion {
                        FusionMode::Additive => {
                            scan_w(&coll.policy.layer_weights).map(|v| (v.clone(), v))
                        }
                        mode => {
                            // Non-additive modes: per-group one-hot scans (each
                            // IS that group's needle-gated tally), weighted,
                            // then fused per the mode — the same law as the CPU
                            // `score_slots_fused`. The ungated group sum rides
                            // along as the mass base.
                            // Groups are `n_heads / heads_per_group(n_heads)` —
                            // i.e. FOLD_GROUPS whenever the signature is
                            // well-formed. Dividing by the LOCKED
                            // `PROV_HEADS_PER_LAYER` instead reads 1 group for
                            // the hybrid's 6-head signatures (3 groups x 2 kv
                            // heads), collapsing the three-group late fusion to
                            // a single scan and making every `layer_weights`
                            // entry past the first address a group that is
                            // never scanned separately — confident, wrong
                            // rankings with no error.
                            let n_groups = p.first().map(|s| {
                                let n = s.n_heads as usize;
                                match heads_per_group(n) {
                                    0 => 1,
                                    per => (n / per).max(1),
                                }
                            })?;
                            let mut grouped: Vec<Vec<f32>> = Vec::with_capacity(n_groups);
                            for g in 0..n_groups {
                                let mut one_hot = vec![0.0f32; n_groups];
                                one_hot[g] =
                                    coll.policy.layer_weights.get(g).copied().unwrap_or(1.0);
                                grouped.push(scan_w(&one_hot)?);
                            }
                            Some((mode.fuse(&grouped), FusionMode::Additive.fuse(&grouped)))
                        }
                    }
                });
                gpu_fresh.unwrap_or_else(|| {
                    // Same narrowing on the CPU path: slice each window to its
                    // phase range so both backends score the identical region.
                    let decoded = host_windows
                        .get_or_init(|| windows.iter().map(SigWindow::decode).collect());
                    let wref: Vec<&[WideQSig]> = ranged
                        .iter()
                        .map(|(i, r)| &decoded[*i][r.clone()])
                        .collect();
                    // Shadows the gallery-wide slot list: `wref` holds only the
                    // windows the lens kept, so its cases must line up with it.
                    let slots: Vec<usize> = ranged.iter().map(|(i, _)| slots[*i]).collect();
                    match coll.policy.scan.fusion {
                        FusionMode::Additive => {
                            let v = score_slots_weighted(
                                p,
                                &wref,
                                &slots,
                                n,
                                &coll.policy.layer_weights,
                            );
                            (v.clone(), v)
                        }
                        mode => {
                            let grouped = score_slots_grouped(
                                p,
                                &wref,
                                &slots,
                                n,
                                &coll.policy.layer_weights,
                            );
                            if grouped.is_empty() {
                                (vec![0.0; n], vec![0.0; n])
                            } else {
                                (mode.fuse(&grouped), FusionMode::Additive.fuse(&grouped))
                            }
                        }
                    }
                })
            };

            let t_scan = Instant::now();
            let (fresh, mass_base) = scan_probe(probe);
            // Concept F: the pinned question window is a second, independent
            // scan — max-fused with the tail AFTER normalization below.
            let fresh_q: Option<Vec<f32>> = match probe_q {
                Some(q) if coll.policy.scan.question_pin && !q.is_empty() => Some(scan_probe(q).0),
                _ => None,
            };
            let scan_us = t_scan.elapsed().as_micros() as u64;
            let t_normalize = Instant::now();
            // Concept A: read each member's score against its background on the
            // collection's band, learned from the TAIL scan's raw scores (the
            // stored whole-turn signature, matching the turn-group learn path).
            // `observe` runs at seal only: a turn teaches every member's
            // background but the member it was for ([`Observe::answer`]).
            let scope = ScopeKey::collection(coll.id.raw() as u64, coll.name.as_str());
            let pairs = |v: &[f32]| -> Vec<(ChildKey, f32)> {
                coll.sections
                    .iter()
                    .zip(v)
                    .map(|(s, &v)| (ChildKey::named(s.name.clone()), v))
                    .collect()
            };
            let raw_pairs = pairs(&fresh);
            // Each scan's own probe length: a raw score is a SUM over probe
            // tokens, and the tail and question windows differ by an order of
            // magnitude, so the band has to be told which probe produced which
            // scores. See `NormConfig::probe_t_ref`.
            let t_tail = probe.len();
            let t_q = probe_q.map_or(1, |q| q.len());
            let (normed, normed_q) = {
                let cache = &*self.normalization;
                let normed = cache.normalize(&scope, &raw_pairs, t_tail);
                let normed_q = fresh_q
                    .as_deref()
                    .map(|fq| cache.normalize(&scope, &pairs(fq), t_q));
                // Segmentation: this collection learns only from a probe inside
                // its own gather scope. `tags: [tool]` means tool traffic teaches
                // the tool band and dialogue traffic does not — the band stays a
                // view of its own world, and learning is never switched off to
                // achieve that.
                if let (true, Some(source)) = (observe.teaches(&coll.policy.tags), observe.source())
                {
                    let answer = observe
                        .answer(coll.sections.iter().map(|s| s.name.as_str()))
                        .map(ChildKey::named);
                    cache.observe(&scope, source, &raw_pairs, answer.as_ref(), t_tail);
                }
                (normed, normed_q)
            };
            // Where one collection's scan time went — assembling its gallery
            // (host, walks the substrate's streams), scoring it (the arena
            // launches), and normalizing — the collection-side counterpart of
            // the per-group `belief group scan` line. Debug, not trace: a scan
            // visits one or two collections, so this is a line or two a scan.
            tracing::debug!(
                target: "candle_conversation::provenance",
                collection = %coll.name,
                windows = windows.len(),
                seal = observe.source().is_some(),
                gallery_us,
                scan_us,
                normalize_us = t_normalize.elapsed().as_micros() as u64,
                "belief collection scan"
            );
            // Fuse, don't filter: every lens is on the same 0–1000 band, so the
            // strongest evidence for a section wins and no lens can veto another.
            let finals: Vec<f32> = (0..n)
                .map(|i| {
                    let at = |v: &Option<Vec<(ChildKey, f32)>>| {
                        v.as_ref()
                            .and_then(|x| x.get(i).map(|(_, v)| *v))
                            .unwrap_or(0.0)
                    };
                    let t = normed.get(i).map(|(_, v)| *v).unwrap_or(0.0);
                    t.max(at(&normed_q))
                })
                .collect();
            if tracing::enabled!(tracing::Level::DEBUG) {
                let nonzero = finals.iter().filter(|&&s| s != 0.0).count();
                // The strongest members with every figure that made them: the
                // tail scan's raw score and its normalized reading, the question
                // window's normalized reading, and the fused final. Which lens
                // carried a member is the first question a wrong pick asks.
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&a, &b| finals[b].total_cmp(&finals[a]));
                let lens = |v: &Option<Vec<(ChildKey, f32)>>, i: usize| {
                    v.as_ref().and_then(|x| x.get(i)).map_or(0.0, |(_, s)| *s)
                };
                let top = order
                    .iter()
                    .take(6)
                    .map(|&i| {
                        format!(
                            "{}={:.0} (tail raw {:.1} → {:.0}, q raw {:.1} → {:.0})",
                            coll.sections[i].name,
                            finals[i],
                            fresh[i],
                            normed.get(i).map_or(0.0, |(_, s)| *s),
                            fresh_q
                                .as_ref()
                                .and_then(|q| q.get(i))
                                .copied()
                                .unwrap_or(0.0),
                            lens(&normed_q, i),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                tracing::trace!(
                    target: "candle_conversation::belief",
                    collection = %coll.name,
                    probe_windows = probe.len(),
                    q_windows = probe_q.map(|q| q.len()).unwrap_or(0),
                    gallery_windows = windows.len(),
                    nonzero_scores = nonzero,
                    top = %top,
                    "belief scan (normalized band)"
                );
            }
            for (s, &score) in coll.sections.iter().zip(&finals) {
                scores.set_section(s.id, score);
            }
            // Concept B: the collection's attention mass, on the UNGATED raw
            // scan (normalization compresses — and the gate removes — the
            // concentration signal mass keys on).
            scores.set_collection_mass(
                coll.id.raw(),
                attention_mass(
                    &mass_base,
                    0.0,
                    f32::MAX,
                    coll.policy.scan.mass_top_k,
                    coll.policy.scan.mass_rho,
                ),
            );
        }
        scores
    }
}
