//! Decides which normalization scopes a scoring pass teaches, and runs the belief
//! scans a projection selects from.
//!
//! Normalization is per scope, so an observation teaches only the scopes its tags
//! place it in. Tool traffic teaches the tool lens and dialogue does not, which
//! keeps dialogue from degrading tool ranking. A read-only pass scores without
//! teaching. Each probe's answer is left out of its own member's background, so a
//! tool's own hits do not read it low. The reproject scan covers every layer, since
//! selection needs the non-target groups' scores, while a focused projection scans
//! none of them.

use super::*;

/// A belief scan longer than this, collections and groups together, is logged
/// with its whole phase split.
pub(super) const SLOW_SCAN_US: u64 = 1_000_000;

/// Whether a scoring pass teaches the normalization levels, and — when it does —
/// which lens the observation belongs to.
///
/// Normalization is per-scope by construction, so an observation is only
/// meaningful *inside the scope it came from*. The tag set carried here is what
/// routes it: a scope learns from a probe only when the probe belongs to that
/// scope. A tag-scoped collection (the tool catalog scores its members against a
/// gallery of `tool`-tagged exemplars) therefore learns from tool traffic and is
/// untouched by dialogue traffic — which is the segmentation that keeps one lens
/// from degrading another WITHOUT switching learning off.
///
/// Getting this wrong is not a small error. Attributing dialogue turns to the
/// tool lens re-ordered its ranking rather than merely blurring it, and
/// compounded: across 24 measured conversations the correct-answer rate fell
/// 67% → 17% and provenance Top-1 fell 4/6 → 1/6.
#[derive(Clone, Copy, Debug)]
pub enum Observe<'a> {
    /// Read-only: score without teaching anything. Live reprojection and the
    /// interactive probe, neither of which is new evidence about a scope.
    No,
    /// Teach the scopes this probe belongs to. `tags` are the observed turn's
    /// gather-scope tags; a tag-scoped node learns only when they intersect its
    /// own. An empty slice is ordinary dialogue — it teaches the untagged
    /// (turn-group) scopes and no tag-scoped collection.
    ///
    /// `source` identifies the evidence — the turn's stream id — so a scope
    /// folds it exactly once however many times it is replayed.
    Yes { tags: &'a [String], source: u64 },
}

impl Observe<'_> {
    /// The evidence id to fold under, when this pass teaches anything.
    pub(super) fn source(&self) -> Option<u64> {
        match self {
            Observe::No => None,
            Observe::Yes { source, .. } => Some(*source),
        }
    }

    /// Whether a node whose gather scope is `scope_tags` learns from this pass.
    /// An untagged node (`scope_tags` empty) is unscoped and learns from any
    /// observing pass; a tag-scoped node learns only from a probe inside it.
    pub(super) fn teaches(&self, scope_tags: &[String]) -> bool {
        match self {
            Observe::No => false,
            Observe::Yes { tags, .. } => {
                scope_tags.is_empty() || scope_tags.iter().any(|s| tags.contains(s))
            }
        }
    }

    /// The one of `members` this pass is about: the member its tags name. A
    /// tool turn is tagged with the tool it called, as the warm-up's exemplars
    /// are, so its seal leaves that tool out of the tool's own background — a
    /// tool called on many turns would otherwise carry its own hits as
    /// background and read lower the more it is used. `None` when the pass
    /// teaches nothing, or its tags name no member.
    pub(super) fn answer<'m>(&self, members: impl IntoIterator<Item = &'m str>) -> Option<&'m str> {
        match self {
            Observe::No => None,
            Observe::Yes { tags, .. } => members.into_iter().find(|m| tags.iter().any(|t| t == m)),
        }
    }
}

impl Conversation {
    /// Fold one reprojection's scan into `target`'s working-set momentum
    /// (`docs/zend_working_set.md` §4.3). `candidates` is what
    /// [`Self::score_beliefs`] returned; a target whose layer declares no
    /// working set learns nothing.
    ///
    /// `hit_density` is each conversation's hit density over the stride since
    /// the last reprojection — what the model's indexer read of it
    /// (`docs/hit_driven_projection.md` §6.4); it lifts the momentum every
    /// judgment of this reprojection weighs.
    ///
    /// The boundary comes first — the slot holds `placed` now
    /// ([`Substrate::working_set_boundary`]) — in the same edit: each edit of a
    /// working set copies it from the version readers hold, and its momentum
    /// and fresh maps name every conversation it has seen.
    pub fn observe_working_set(
        &self,
        schema: &Schema,
        target: ProjectionTarget,
        placed: &[TurnKey],
        candidates: &[(GroupId, Vec<(TurnKey, f32)>)],
        hit_density: &HashMap<TimelineId, f32>,
    ) {
        let Some(cfg) = working_set_config(schema, target) else {
            self.update(|s| s.working_set_boundary(target.timeline, placed));
            return;
        };
        // A unit an aging group holds is dated by its metadata, against the
        // moment of this scan.
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64);
        let fresh = {
            let read = self.read();
            let aged = |r: &Recency, timeline: TimelineId| {
                read.custom_of(timeline)
                    .and_then(|m| m.get(&r.dated_by))
                    .map_or(1.0, |path| r.factor(path, now_ms))
            };
            working_set_fresh(schema, candidates, &aged)
        };
        self.update(|s| {
            s.working_set_boundary(target.timeline, placed);
            s.working_set_observe(
                target.timeline,
                &fresh,
                hit_density,
                &schema.hits,
                cfg.beta,
                cfg.min_momentum,
            )
        });
    }

    /// Score every belief node the projection will consult: the **target
    /// layer's collections** (the tool catalog) plus **every layer's
    /// belief-driven turn groups** (repo_map clusters, code scopes, memory tiers
    /// — each in its own non-target layer). Returns the combined
    /// [`ProjectionScores`] and, for the scheduler's turn-boundary challenger,
    /// each scored turn group's candidate `(turn, fresh_score)` list.
    ///
    /// The projection materializes *all* visible layers and belief-selects each
    /// layer's groups, so the scan must cover all layers too — scoping it to the
    /// target layer alone leaves every non-target turn group on all-zero scores
    /// (a degenerate index tie-break instead of relevance). An empty probe scores
    /// nothing.
    /// `observe` folds this probe's raw turn-group scores into the normalization
    /// backgrounds: `true` only on the once-per-turn seal scan
    /// ([`crate::conversation`]'s `last_turn_belief_scores`), `false` on every
    /// live reprojection (which only reads the levels to normalize). See
    /// `docs/provenance_score_normalization.md` §4.2.
    /// `probe_q` is the Concept F pinned question window (the current turn's
    /// user span); nodes whose policy sets `question_pin` scan it as a second
    /// probe and take the per-member max of the two normalized scans. `None`
    /// (seal-time / warm replay / hosts without a live turn) scans the tail
    /// alone, exactly as before.
    pub fn score_beliefs(
        &self,
        schema: &Schema,
        target: ProjectionTarget,
        probe: &[WideQSig],
        probe_q: Option<&[WideQSig]>,
        observe: Observe<'_>,
        arena: Option<&dyn BeliefArena>,
    ) -> (ProjectionScores, Vec<(GroupId, Vec<(TurnKey, f32)>)>) {
        let mut scores = ProjectionScores::new();
        let mut candidates: Vec<(GroupId, Vec<(TurnKey, f32)>)> = Vec::new();
        if probe.is_empty() {
            return (scores, candidates);
        }
        // The once-per-turn seal scan observes; a live reprojection only reads.
        let seal = u8::from(observe.source().is_some());
        // Collections (the tool catalog) live in the shared system prompt.
        scan_split::take();
        let t_coll = Instant::now();
        let coll_span = profile::span("beliefs:collections");
        scores =
            self.score_belief_collections(&schema.system_prompt, probe, probe_q, observe, arena);
        coll_span.end();
        let coll_us = t_coll.elapsed().as_micros() as u64;
        // Belief-driven turn groups live across every layer.
        let t_groups = Instant::now();
        let _groups_span = profile::span("beliefs:groups");
        GROUP_SCAN_SPLIT.take();
        // A focused projection scans no other layer (`Schema::focus`): it takes
        // nothing from them but locked working-set members, and a scan would
        // feed their momentum and backgrounds from a question that is not the
        // conversation's own.
        for layer in schema
            .layers
            .iter()
            .filter(|l| !schema.focus || l.id == target.layer)
        {
            candidates.extend(self.score_belief_groups(
                layer,
                target,
                probe,
                probe_q,
                &mut scores,
                observe,
                arena,
            ));
        }
        // Phase split for the reproject `scan_ms`: which side of the scan the
        // time went to (the collection scan vs the per-layer group scans), and
        // within the groups which phase — so a silent GPU→CPU fallback, a
        // growing probe or a group walk that grew with the corpus shows up
        // attributably. One line per scan.
        let groups = GROUP_SCAN_SPLIT.take();
        let arena = scan_split::take();
        // The same split in the profile table, where it sits beside the
        // reprojection that waits on it.
        let us = Duration::from_micros;
        profile::record("beliefs:groups:select", us(groups.select_us));
        profile::record("beliefs:groups:assemble", us(groups.assemble_us));
        profile::record("beliefs:groups:assemble:walk", us(groups.walk_us));
        profile::record("beliefs:groups:assemble:exchange", us(groups.exchange_us));
        profile::record("beliefs:groups:assemble:calibrations", us(groups.cal_us));
        profile::record("beliefs:groups:score:segments", us(groups.segments_us));
        profile::record("beliefs:groups:score", us(groups.score_us));
        profile::record("beliefs:groups:normalize", us(groups.normalize_us));
        profile::record("beliefs:arena:index", us(arena.index_us));
        profile::record("beliefs:arena:index:fingerprint", us(arena.fingerprint_us));
        profile::record("beliefs:arena:upload", us(arena.upload_us));
        profile::record("beliefs:arena:kernel", us(arena.kernel_us));
        profile::record("beliefs:arena:readback", us(arena.readback_us));
        profile::record("beliefs:arena:rows", us(arena.rows_us));
        profile::record("beliefs:arena:tally", us(arena.tally_us));
        let groups_us = t_groups.elapsed().as_micros() as u64;
        scan_census::record(coll_us, groups_us, &groups, &arena);
        let group_part = || {
            format!(
                "probe_windows={} collections_us={coll_us} groups_us={} groups_scanned={} \
                 group_windows={} group_turns={} select_us={} assemble_us={} score_us={} \
                 normalize_us={} files={} galleries_built={} walk_us={} exchange_us={} \
                 cal_built={} cal_windows={} cal_us={} segments_us={} \
                 seal={seal} scope_us={} tags_us={} files_us={} exclude_us={} kinds_us={} \
                 lineage_us={} release_us={} assemblies_reused={}",
                probe.len(),
                groups_us,
                groups.scanned,
                groups.windows,
                groups.turns,
                groups.select_us,
                groups.assemble_us,
                groups.score_us,
                groups.normalize_us,
                groups.files,
                groups.galleries_built,
                groups.walk_us,
                groups.exchange_us,
                groups.cal_built,
                groups.cal_windows,
                groups.cal_us,
                groups.segments_us,
                groups.scope_us,
                groups.tags_us,
                groups.files_us,
                groups.exclude_us,
                groups.kinds_us,
                groups.lineage_us,
                groups.release_us,
                groups.assemblies_reused,
            )
        };
        let arena_part = || {
            format!(
                "arena_scans={} arena_launches={} arena_segments={} arena_fresh_tokens={} \
                 arena_index_us={} arena_fingerprint_us={} arena_windows={} arena_pinned={} \
                 arena_upload_us={} arena_kernel_us={} arena_readback_us={} arena_rows_us={} \
                 arena_tally_us={} arena_kernel_gpu_us={} arena_index_built={} \
                 arena_index_build_us={} arena_index_absent={} arena_index_stale={} \
                 arena_index_narrower={} arena_held={} arena_hold_us={} arena_queue_us={} \
                 arena_view_us={} arena_built_fresh_tokens={} arena_widest_fresh={} \
                 arena_widest_tokens={} arena_widest_gpu_us={} arena_checked_turns={} \
                 arena_checks_whole={}",
                arena.scans,
                arena.launches,
                arena.segments,
                arena.fresh_tokens,
                arena.index_us,
                arena.fingerprint_us,
                arena.windows,
                arena.pinned,
                arena.upload_us,
                arena.kernel_us,
                arena.readback_us,
                arena.rows_us,
                arena.tally_us,
                arena.kernel_gpu_us,
                arena.index_built,
                arena.index_build_us,
                arena.index_absent,
                arena.index_stale,
                arena.index_narrower,
                arena.held,
                arena.hold_us,
                arena.queue_us,
                arena.view_us,
                arena.built_fresh_tokens,
                arena.widest_fresh,
                arena.widest_tokens,
                arena.widest_gpu_us,
                arena.checked_turns,
                arena.checks_whole,
            )
        };
        // A scan past a second is written whole, its groups' and its arena's
        // halves as a line each so neither is cut off where a log reader ends
        // a line: it is a conversation's first reprojection or a cold arena,
        // the one instance the event keeps is overwritten by the next scan
        // within a second, and a first token waits on it.
        if coll_us + groups_us > SLOW_SCAN_US {
            tracing::info!(
                target: "candle_conversation::provenance",
                "belief scan slow {}",
                group_part()
            );
            tracing::info!(
                target: "candle_conversation::provenance",
                "belief scan slow arena {}",
                arena_part()
            );
        }
        note("belief scan phase split", || {
            format!("{} {}", group_part(), arena_part())
        });
        (scores, candidates)
    }
}
