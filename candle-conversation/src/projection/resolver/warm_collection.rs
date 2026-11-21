//! Warms each section collection's normalization backgrounds from its own
//! tag-scoped corpus, before live traffic arrives.
//!
//! Without this a restart leaves a collection's backgrounds cold, and its margins
//! collapse so near-ties flip the ranking. Each exemplar is scored against the
//! gallery without its own member, since a self-match would give that member every
//! vote and leave the other backgrounds at zero. The exemplars are gathered once and
//! scored in a few GPU launches. A failed GPU scan ends the warm-up rather than
//! falling back to a host scan that could stall boot.

use super::*;

/// Self-probes per collection MEMBER when warming a section collection's
/// backgrounds from its own tag-scoped corpus
/// ([`Conversation::warm_collection_normalization`]). Per member, not per
/// collection: each exemplar is about its own member and teaches every other
/// member's background, so a member's background is the other members'
/// exemplars — 8 each, over the catalog — and every member has one.
pub(super) const WARM_COLLECTION_PROBES_PER_MEMBER: usize = 8;

/// One exemplar the collection warm-up folds: its turn stream (the observation's
/// source), the member it is an exemplar of, and its whole-turn signature.
pub(super) struct Exemplar {
    pub(super) source: u64,
    pub(super) answer: ChildKey,
    pub(super) sig: Vec<WideQSig>,
}

/// The collection warm-up's scans: each exemplar's index under the case its
/// scan leaves out of the gallery — the member it is an exemplar of, or `None`
/// for one about no member of the collection being taught. One round per case,
/// in the order each case first appears, its probes in their own order.
pub(super) fn leave_out_rounds(
    cases: impl IntoIterator<Item = Option<usize>>,
) -> Vec<(Option<usize>, Vec<usize>)> {
    let mut rounds: Vec<(Option<usize>, Vec<usize>)> = Vec::new();
    for (probe, case) in cases.into_iter().enumerate() {
        match rounds.iter_mut().find(|(c, _)| *c == case) {
            Some((_, at)) => at.push(probe),
            None => rounds.push((case, vec![probe])),
        }
    }
    rounds
}

/// Exemplar probes per GPU gallery scan during the collection warm-up. The whole
/// corpus fits one launch comfortably (~744 probes on the tool catalog), but the
/// probe upload is `probe tokens × words per token` and grows with the corpus, so
/// the batch bounds it; each launch reuses the segment's cached index.
pub(super) const WARM_GPU_PROBE_BATCH: usize = 128;

/// What a collection warm-up did — see
/// [`Conversation::warm_collection_normalization`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CollectionWarm {
    /// Collection members the corpus had exemplars for.
    pub members: usize,
    /// Exemplar probes folded into the levels.
    pub probes: usize,
    /// Of the per-collection scans, the probes scored on the GPU gallery arena;
    /// the rest ran on the CPU.
    pub gpu_probes: usize,
}

/// Which corpus turns warm which collection member, in a deterministic order.
///
/// A gallery turn carries the collection's scope tag (what bounds the gallery)
/// plus the name of the member it was captured for; the member is therefore the
/// tag that names an actual member. Turns carrying a scope tag but no member tag
/// belong to the gallery without teaching any single member's level, and are
/// skipped rather than attributed to an arbitrary one.
///
/// Split out of [`Conversation::warm_collection_normalization`] so the tag
/// derivation, the per-member cap and the ordering are testable without a
/// substrate: the result must not depend on stream-map iteration order, or the
/// exemplars kept under the cap — and hence the backgrounds every tool query is
/// read against — would differ run to run.
pub(super) fn collection_warm_plan<'a>(
    member_names: &HashSet<&str>,
    scope_tags: &[String],
    turns: impl Iterator<Item = (&'a [String], u64, u32)>,
    per_member: usize,
) -> Vec<(String, Vec<(u64, u32)>)> {
    let mut by_member: std::collections::BTreeMap<String, Vec<(u64, u32)>> =
        std::collections::BTreeMap::new();
    for (tags, timeline, index) in turns {
        if !tags.iter().any(|t| scope_tags.contains(t)) {
            continue;
        }
        let Some(member) = tags.iter().find(|t| member_names.contains(t.as_str())) else {
            continue;
        };
        by_member
            .entry(member.clone())
            .or_default()
            .push((timeline, index));
    }
    by_member
        .into_iter()
        .map(|(member, mut keys)| {
            keys.sort_unstable();
            keys.truncate(per_member);
            (member, keys)
        })
        .collect()
}

/// Contiguous `[start, end)` sub-window bounds over a `len`-token sig, split at
/// sorted, deduped `seams`. An empty `seams` yields one window `[0, len)` — the
/// prior whole-turn behaviour. Seams at 0, at/past `len`, or that don't advance
/// are ignored, so a malformed seam can never produce an empty or inverted range.
pub(in super::super) fn subwindow_bounds(len: usize, seams: &[usize]) -> Vec<(usize, usize)> {
    let mut bounds = Vec::with_capacity(seams.len() + 1);
    let mut prev = 0usize;
    for &s in seams {
        if s > prev && s < len {
            bounds.push((prev, s));
            prev = s;
        }
    }
    bounds.push((prev, len));
    bounds
}

impl Conversation {
    /// Warm every belief-driven section COLLECTION's backgrounds from its own
    /// tag-scoped corpus (self-match), the collection analogue of
    /// [`Self::warm_ingest_normalization`]: each member's exemplars teach every
    /// other member's background.
    ///
    /// The dialogue replay ([`Self::take_warm_replay`]) already re-observes collections,
    /// but only through **dialogue** probes (`tags.is_empty()`) — and a collection's
    /// own corpus is tagged by construction (that tag is what scopes its gallery).
    /// So the turns that actually teach a member's level are exactly the ones that
    /// replay excludes, and on a workspace whose dialogue history is short or empty
    /// — every fresh install — the collection stays COLD. That is not a small
    /// difference: measured across a restart of the same substrate, the tool
    /// collection's scores fell 11–30x (`datetime` 1119 → 97) and, far worse, the
    /// MARGINS collapsed with them, so near-ties began flipping the ranking
    /// (`sqrt of …` went from `calculator` 3521 vs 77 to `kdf_derive` 116 over
    /// `calculator` 116). The backgrounds are runtime-only and rebuilt empty each
    /// load (`normalization::cache`), so without this the session that BUILDS the
    /// corpus is the only one that scores it well — and production is never that
    /// session.
    ///
    /// Additive-fusion collections only, for the reason
    /// [`Self::warm_ingest_normalization`] gives: a gated-fusion scope is learned
    /// from traffic alone.
    ///
    /// **Scored in batch, not probe by probe.** A warm-up observation is the
    /// seal-time one — the probe's whole-turn raw scores folded into the
    /// collection's backgrounds — so it needs none of the live scan's other work:
    /// no normalization (pure, and its result was discarded) and no question
    /// scan (read only by the normalized fusion). The corpus's exemplars are
    /// gathered once and scored against each collection's gallery in a few GPU
    /// launches when `arena` is given, which is why this runs on the scheduler
    /// thread. One CPU scan per probe was ~0.55 s on the tool catalog — seven to
    /// ten minutes of load for 744 probes.
    ///
    /// Call once the corpus is stable (load complete), never concurrently with an
    /// ingest writer — the same read-lock starvation hazard as the ingest warm-up.
    /// The substrate lock is taken to gather signatures and released before any
    /// scoring call.
    pub fn warm_collection_normalization(
        &self,
        schema: &Schema,
        arena: Option<&GalleryArena>,
    ) -> CollectionWarm {
        // How long the job waited for the warm pool, logged as it starts: one
        // post-recalibration boot sat in this step for over half an hour with
        // every pool thread busy and the GPU idle, and the line says whether a
        // recurrence is the queue or the work.
        let queued_at = Instant::now();
        warm_pool::run(|| {
            tracing::info!(
                target: "candle_conversation::provenance",
                queued_ms = queued_at.elapsed().as_millis() as u64,
                "normalization warm-up: started on the warm pool",
            );
            self.warm_collection_normalization_on_pool(schema, arena)
        })
    }

    /// [`Self::warm_collection_normalization`]'s body, run on the warm pool.
    pub(super) fn warm_collection_normalization_on_pool(
        &self,
        schema: &Schema,
        arena: Option<&GalleryArena>,
    ) -> CollectionWarm {
        let mut warm = CollectionWarm::default();
        let sp = &schema.system_prompt;
        for item in &sp.items {
            let SystemPromptItem::Collection(coll) = item else {
                continue;
            };
            if coll.sections.is_empty() || coll.policy.tags.is_empty() {
                continue;
            }
            if coll.policy.scan.fusion != FusionMode::Additive {
                continue;
            }
            // Group the corpus by the member each turn belongs to. A gallery turn
            // carries the collection's scope tag plus the member name it was
            // captured for, so the member is the tag that is not a scope tag —
            // the same derivation the gallery itself uses to map turn → slot.
            let member_names: HashSet<&str> =
                coll.sections.iter().map(|s| s.name.as_str()).collect();
            let planned_at = Instant::now();
            // Plan first (cheap, from declarations only), then fetch signatures for
            // exactly the turns the plan keeps — the cap does not pay to decode
            // signatures it discards.
            let plan = {
                let sub = self.inner.read();
                let decls = sub.all_turn_streams().map(|turn| {
                    (
                        turn.decl.tags.as_slice(),
                        turn.decl.timeline_id,
                        turn.index.0,
                    )
                });
                collection_warm_plan(
                    &member_names,
                    &coll.policy.tags,
                    decls,
                    WARM_COLLECTION_PROBES_PER_MEMBER,
                )
            };
            warm.members += plan.len();
            // Every planned exemplar's whole-turn signature, with the member it
            // is an exemplar of: the probe's answer, left out of that member's
            // background. The order is the plan's, and changes nothing — a
            // background is counts.
            let probes: Vec<Exemplar> = {
                let sub = self.inner.read();
                plan.iter()
                    .flat_map(|(member, keys)| keys.iter().map(move |key| (member, key)))
                    .filter_map(|(member, (tl, idx))| {
                        let sid = turn_stream_id(*tl, *idx);
                        sub.wide_sig_window(sid).map(|w| Exemplar {
                            source: sid.0,
                            answer: ChildKey::named(member.clone()),
                            sig: w.decode(),
                        })
                    })
                    .filter(|p| !p.sig.is_empty())
                    .collect()
            };
            tracing::info!(
                target: "candle_conversation::provenance",
                collection = %coll.name,
                members = plan.len(),
                probes = probes.len(),
                plan_and_fetch_ms = planned_at.elapsed().as_millis() as u64,
                "normalization warm-up: probes planned and fetched",
            );
            if probes.is_empty() {
                continue;
            }
            // The probes are this collection's corpus, so they teach every
            // collection whose scope they fall inside — its own, and any other
            // sharing a scope tag. The same set the per-probe scan over the whole
            // system prompt taught.
            let observer = Observe::Yes {
                tags: &coll.policy.tags,
                source: 0,
            };
            for item in &sp.items {
                let SystemPromptItem::Collection(taught) = item else {
                    continue;
                };
                if taught.sections.is_empty() || !observer.teaches(&taught.policy.tags) {
                    continue;
                }
                match self.fold_warm_probes(taught, &probes, arena) {
                    Some(true) => warm.gpu_probes += probes.len(),
                    Some(false) => {}
                    // It did not complete — a shutdown asked, or the GPU scan
                    // failed and the host cannot stand in for it. Either way the
                    // warm-up ends here with whatever it has already learned.
                    None => return warm,
                }
            }
            warm.probes += probes.len();
        }
        tracing::info!(
            members = warm.members,
            probes = warm.probes,
            gpu_probes = warm.gpu_probes,
            "normalization warm-up: learned per-member backgrounds for tag-scoped section \
             collections (0 ⇒ no additive belief-driven collection had a tagged corpus — \
             collection levels stay cold and scores are not comparable across members)"
        );
        warm
    }

    /// Fold `probes` into `taught`'s backgrounds — each probe's raw whole-turn
    /// scores against `taught`'s gallery, observed under the probe's stream id
    /// as the seal-time scan observes it, its own member left out. `Some(true)` when
    /// the GPU arena scored them, `Some(false)` for the CPU (no arena, a
    /// non-additive law the arena does not scan, or an empty gallery — the cases
    /// that never reach a kernel and are cheap on the host), `None` when it did
    /// not complete: a shutdown stopped it, or the GPU scan failed on a gallery
    /// the host cannot finish.
    pub(super) fn fold_warm_probes(
        &self,
        taught: &SectionCollection,
        probes: &[Exemplar],
        arena: Option<&GalleryArena>,
    ) -> Option<bool> {
        let n = taught.sections.len();
        let members: Vec<String> = taught.sections.iter().map(|s| s.name.clone()).collect();
        let gallery_started = Instant::now();
        let (windows, slots, sids) =
            self.belief_gallery(&taught.name, &taught.policy.tags, &members);
        tracing::info!(
            target: "candle_conversation::provenance",
            collection = %taught.name,
            windows = windows.len(),
            probes = probes.len(),
            gallery_ms = gallery_started.elapsed().as_millis() as u64,
            gpu = arena.is_some() && taught.policy.scan.fusion == FusionMode::Additive,
            "normalization warm-up: gallery assembled, scoring probes",
        );
        if windows.is_empty() {
            return Some(false);
        }
        let weights = &taught.policy.layer_weights;
        let fusion = taught.policy.scan.fusion;
        // **Each exemplar is scored against the gallery WITHOUT its own member.**
        //
        // The scan is late fusion: every probe token votes for the case of its
        // best-matching gallery token. An exemplar scored against a gallery that
        // holds its own window — and its member's other exemplars — self-matches,
        // so its member takes nearly every vote and every OTHER member's
        // background sample is ~0. Every background then collapses to zeros, a
        // live score lands past the top of all of them, and the band reads flat
        // and high: measured on the tool catalog, "hi, how are you doing today?"
        // put five tools at ~3000 and every probe put a dozen past 2600.
        // Leaving the member out makes each sample what the other members score
        // on a question that is NOT about them, which is what a background is.
        // One scan per member, so the probe work is the same as one batched scan.
        let case_of = |answer: &ChildKey| match answer {
            ChildKey::Named(m) => members.iter().position(|x| x == m),
            _ => None,
        };
        let rounds = leave_out_rounds(probes.iter().map(|p| case_of(&p.answer)));
        let mut raw: Vec<Vec<f32>> = vec![Vec::new(); probes.len()];
        let on_gpu = arena.is_some() && fusion == FusionMode::Additive;
        // The host path's decoded windows: decoded once, on first use.
        let host_windows: OnceCell<Vec<Vec<WideQSig>>> = OnceCell::new();
        for (left_out, at) in &rounds {
            let kept: Vec<usize> = (0..windows.len())
                .filter(|&i| Some(slots[i]) != *left_out)
                .collect();
            if kept.is_empty() {
                // The gallery holds no other member: nothing scores.
                for &p in at {
                    raw[p] = vec![0.0; n];
                }
                continue;
            }
            let queries: Vec<&[WideQSig]> = at.iter().map(|&p| probes[p].sig.as_slice()).collect();
            let scored = match arena {
                Some(arena) if on_gpu => self
                    .fold_warm_scan_gpu(taught, arena, &windows, &slots, &sids, &kept, &queries)?,
                _ => {
                    let decoded = host_windows
                        .get_or_init(|| windows.iter().map(SigWindow::decode).collect());
                    let gallery: Vec<&[WideQSig]> =
                        kept.iter().map(|&i| decoded[i].as_slice()).collect();
                    let cases: Vec<usize> = kept.iter().map(|&i| slots[i]).collect();
                    let mut out = Vec::with_capacity(queries.len());
                    for query in &queries {
                        if self.ingest_cancelled() {
                            return None;
                        }
                        out.push(score_slots_fused(
                            query, &gallery, &cases, n, weights, fusion,
                        ));
                    }
                    out
                }
            };
            for (&p, scores) in at.iter().zip(scored) {
                raw[p] = scores;
            }
        }
        let scope = ScopeKey::collection(taught.id.raw() as u64, taught.name.as_str());
        let cache = &*self.normalization;
        for (probe, scores) in probes.iter().zip(&raw) {
            let pairs: Vec<(ChildKey, f32)> = taught
                .sections
                .iter()
                .zip(scores)
                .map(|(s, &v)| (ChildKey::named(s.name.clone()), v))
                .collect();
            cache.observe(
                &scope,
                probe.source,
                &pairs,
                Some(&probe.answer),
                probe.sig.len(),
            );
        }
        Some(on_gpu)
    }

    /// [`Self::fold_warm_probes`]'s arena scan of `queries` against the gallery
    /// windows at `kept`, in [`WARM_GPU_PROBE_BATCH`] launches. `None` when it did
    /// not complete: a shutdown asked, or the scan failed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fold_warm_scan_gpu(
        &self,
        taught: &SectionCollection,
        arena: &GalleryArena,
        windows: &[SigWindow],
        slots: &[usize],
        sids: &[StreamId],
        kept: &[usize],
        queries: &[&[WideQSig]],
    ) -> Option<Vec<Vec<f32>>> {
        let weights = &taught.policy.layer_weights;
        let segments = [PagedSegment {
            windows: kept
                .iter()
                .map(|&i| PagedWindow {
                    sid: sids[i],
                    fingerprint: windows[i].fingerprint(),
                    turn: (&windows[i]).into(),
                    start: 0,
                    end: windows[i].len(),
                    case: slots[i],
                })
                .collect(),
            n_cases: taught.sections.len(),
            holder: Holder::Shared,
            groups: GroupSet::weighed(weights, arena.n_groups()),
            keep: taught.policy.scan.gallery_keep,
            law: taught.policy.scan.law,
        }];
        let mut raw = Vec::with_capacity(queries.len());
        for batch in queries.chunks(WARM_GPU_PROBE_BATCH) {
            if self.ingest_cancelled() {
                return None;
            }
            match arena.scan_weighted(&segments, batch, weights) {
                Ok(out) => raw.extend(out),
                Err(e) => {
                    // **A failed GPU scan does NOT fall back to the CPU.**
                    //
                    // The host path is a per-query walk of the whole gallery.
                    // It is the right answer for the cheap cases that never
                    // reach a kernel (no arena, an empty gallery, a fusion law
                    // the arena does not scan) — but a gallery big enough to
                    // want the GPU is one the host cannot finish. Measured: a
                    // `CUDA_ERROR_INVALID_CONTEXT` here put eight cores at 100%
                    // with the GPU idle and boot never completed — 900 s and
                    // still going, which is the same half-hour stall
                    // `warm_collection_normalization` already records.
                    //
                    // Warm-up is an optimisation: skipping it leaves this
                    // collection's levels cold for the session, which costs
                    // ranking quality. Hanging the boot costs everything.
                    tracing::warn!(
                        target: "candle_conversation::provenance",
                        collection = %taught.name,
                        windows = kept.len(),
                        probes = queries.len(),
                        "GPU warm-up scan FAILED: {e} — abandoning this \
                         collection's warm-up rather than falling back to a \
                         host scan that cannot finish a gallery this size. \
                         Its backgrounds stay COLD for this session, so its \
                         members' scores are not comparable to each other."
                    );
                    return None;
                }
            }
        }
        Some(raw)
    }
}
