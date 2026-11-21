//! Carries the belief each section and turn held from one projection to the
//! next, so a turn's selection has memory.
//!
//! `PriorBelief` is built from the previous projection's selection and seeds the
//! next one, so decay and reinforcement carry across a turn's reprojections.
//! Collections keep every member, so an unselected tool's score keeps decaying.
//! Turn groups carry only the selected turns, and an unselected turn reseeds
//! fresh. `decay_scores` halves the carried scores so a fresh query can overtake
//! a stale one, and the challenger rule seats the strongest new signal in a spare
//! slot without lowering the selection threshold.

use super::*;

/// A belief node's identity: a named section collection (the tool catalog) or a
/// belief-driven turn group (repo_map clusters, code scopes). Group names are
/// globally unique in a schema, but a collection and a turn group could share a
/// name, so the two namespaces are kept distinct by the variant. Turn-group
/// members are sub-keyed by their `TurnIndex`, collection members by section name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupKey {
    Collection(String),
    TurnGroup(String),
}

/// The prior projection's belief, per belief node (collection **or** turn group),
/// used to seed the current projection so online decay/reinforcement carries
/// across a turn's reprojections. Built by the scheduler from the previous
/// [`super::ProjectionEvent`]; empty (the default) means a fresh, stateless
/// belief for this projection.
#[derive(Debug, Clone, Default)]
pub struct PriorBelief {
    pub(super) beliefs: HashMap<GroupKey, HashMap<String, (f32, bool, bool)>>,
}

impl PriorBelief {
    /// Record a section's prior belief (score + whether it was selected).
    pub fn set(
        &mut self,
        collection: &str,
        section: &str,
        score: f32,
        selected: bool,
        qualified: bool,
    ) {
        self.beliefs
            .entry(GroupKey::Collection(collection.to_string()))
            .or_default()
            .insert(section.to_string(), (score, selected, qualified));
    }

    /// Record a turn's prior belief in a belief-driven turn group (repo_map
    /// clusters, code scopes). The turn-axis analogue of [`Self::set`].
    pub fn set_turn(
        &mut self,
        group: &str,
        turn: TurnKey,
        score: f32,
        selected: bool,
        qualified: bool,
    ) {
        self.beliefs
            .entry(GroupKey::TurnGroup(group.to_string()))
            .or_default()
            .insert(turn_belief_key(turn), (score, selected, qualified));
    }

    /// Rebuild the prior belief from a completed projection's selection — every
    /// collection member's `(score, selected)`, plus each **selected** memory
    /// turn of a belief-driven turn group. The scheduler stores this on the
    /// decode state after each reprojection so the next one seeds from it,
    /// carrying the online belief across a turn's reprojections.
    ///
    /// Unlike the collection axis — which carries *every* member so an
    /// unselected tool keeps its decaying score — the turn axis carries **only
    /// selected** turns. This divergence is deliberate: the candidate set of a
    /// belief-driven group is bounded (clusters, scopes — not unbounded dialogue,
    /// which is a recency `Sequence` group and never consulted here), and an
    /// unselected turn with a strong fresh score is re-admitted by the next scan
    /// anyway, so an unselected turn reseeds fresh (score 0) instead of decaying —
    /// which keeps the carry bounded by the group's budget without losing lock-on.
    /// The live user message (`index == u32::MAX`) is the probe, not a memory.
    pub fn from_selection(sel: &super::super::ProjectionSelection) -> PriorBelief {
        let mut pb = PriorBelief::default();
        for item in &sel.system {
            if let super::super::SystemItem::Collection { name, sections, .. } = item {
                for s in sections {
                    pb.set(name, &s.name, s.score, s.selected, s.qualified);
                }
            }
        }
        for t in &sel.turns {
            if t.index != u32::MAX && t.selected {
                // Carry the belief under the turn's full `(timeline, index)`
                // identity — the persisted selection stamps the timeline exactly
                // so consumers never re-derive `group → timeline`, which is
                // ambiguous once a group holds many conversations.
                if let Some(tl) = t.timeline.and_then(TimelineId::from_raw) {
                    pb.set_turn(
                        &t.group,
                        TurnKey::new(tl, TurnIndex(t.index)),
                        t.score,
                        true,
                        t.qualified,
                    );
                }
            }
        }
        pb
    }

    /// Overlay `newer` onto this belief, per node: nodes present in `newer`
    /// replace this belief's entry wholesale; nodes absent from `newer` keep
    /// their existing state. A turn whose projection never ran a collection (e.g.
    /// tools dial off → the tools collection materialised nothing and is absent
    /// from the selection) says nothing about that node's belief, so carrying it
    /// across the turn must not erase it.
    pub fn merge_from(&mut self, newer: &PriorBelief) {
        for (key, members) in &newer.beliefs {
            self.beliefs.insert(key.clone(), members.clone());
        }
    }

    /// Scale every carried score by `factor`, turning the belief from a *hard
    /// pin* into a *soft prior* for the next turn's opening projection.
    ///
    /// The carried belief exists so a genuine multi-turn tool workflow keeps
    /// its lock-on instead of resetting to catalog order. But at full strength
    /// it also lets the *previous* turn's tool outrank the correct fresh signal
    /// for a topic-changed turn: the incoming query's evidence is prefill-Q at
    /// token 0 (the call↔definition domain gap, ≈0 for the right tool) and only
    /// builds from the assistant's decode-Q over the first ~64 tokens, so a
    /// full-strength prior owns the whole opening window and the model can
    /// commit to a wrong framing before the correct tool is ever selected.
    /// Halving the carried scores lets that fresh decode-Q overtake a stale
    /// tool within a few tokens; a real continuation re-accumulates its belief
    /// just as fast from its own decode-Q, so continuity is preserved. The
    /// `selected` flags are untouched — the score is the lever selection ranks
    /// on. See `docs/tool_selection_provenance_results.md` §24.
    pub fn decay_scores(&mut self, factor: f32) {
        for members in self.beliefs.values_mut() {
            for (score, _selected, _qualified) in members.values_mut() {
                *score *= factor;
            }
        }
    }

    /// Turn-boundary challenger rule for a top-N collection (`budget_max ≥ 3`):
    /// give the strongest FRESH signal a slot even when it's below the
    /// selection threshold, so a topic-changed turn's new intent can break in
    /// **without lowering `min_score`** (which would admit the noise floor
    /// mid-turn). Applied once, at the turn's first fresh-scored reprojection.
    ///
    /// `fresh` is `(section_name, fresh_score)` for the collection this scan.
    /// The challenger is the highest fresh-scored section that isn't already
    /// selected. It's seeded as a selected member at `seed_score` (≥
    /// `evict_score`, so it survives the belief step); if the carried selection
    /// is already at `budget_max` the lowest-scored incumbent is evicted first.
    /// A carried selection below capacity keeps all its members — the strong
    /// signals survive and the challenger simply fills a spare slot; only a full
    /// selection displaces its weakest. RelLeak then decays the challenger back
    /// out over the turn if its fresh score doesn't hold up.
    pub fn seat_turn_boundary_challenger(
        &mut self,
        key: GroupKey,
        fresh: &[(String, f32)],
        budget_max: usize,
        seed_score: f32,
    ) {
        if budget_max < 3 {
            return;
        }
        let members = self.beliefs.entry(key).or_default();
        let is_selected = |name: &str| members.get(name).map(|&(_, sel, _)| sel).unwrap_or(false);
        // The challenger: strongest fresh signal not already selected.
        let challenger = fresh
            .iter()
            .filter(|(name, score)| *score > 0.0 && !is_selected(name))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name.clone());
        let Some(challenger) = challenger else {
            return;
        };
        // Currently-selected incumbents, by carried score.
        let selected: Vec<(String, f32)> = members
            .iter()
            .filter(|(_, (_, sel, _))| *sel)
            .map(|(name, (score, _, _))| (name.clone(), *score))
            .collect();
        if selected.len() >= budget_max {
            if let Some((low, _)) = selected
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                members.insert(low.clone(), (0.0, false, false));
            }
        }
        // Seated, but NOT qualified: the challenger is deliberately placed below
        // the selection threshold ("break in without lowering `min_score`"), so it
        // has shown no evidence of crossing a bar. Marking it qualified would hand
        // it to the early-decode carry floor, which would pin it at the seed score
        // for the whole grace window — turning a speculative slot into a lock-on
        // and squatting the budget the real tool needs. It has to earn the floor by
        // reaching `min_score` on its own belief, like any other member.
        members.insert(challenger, (seed_score, true, false));
    }

    /// Aligned `(scores, selected)` for a collection's sections in declaration
    /// order — the seed for [`crate::provenance::belief_step`]. Unknown sections
    /// read `(0.0, false)`.
    pub(super) fn collection(
        &self,
        name: &str,
        sections: &[&SectionSchema],
    ) -> (Vec<f32>, Vec<bool>, Vec<bool>) {
        let map = self.beliefs.get(&GroupKey::Collection(name.to_string()));
        let mut scores = vec![0.0f32; sections.len()];
        let mut selected = vec![false; sections.len()];
        let mut qualified = vec![false; sections.len()];
        if let Some(m) = map {
            for (i, s) in sections.iter().enumerate() {
                if let Some(&(sc, sel, q)) = m.get(&s.name) {
                    scores[i] = sc;
                    selected[i] = sel;
                    qualified[i] = q;
                }
            }
        }
        (scores, selected, qualified)
    }

    /// Aligned `(scores, selected)` for a turn group's candidate turns, in the
    /// given order — the seed for [`crate::provenance::belief_step`] on the turn
    /// axis. Unknown turns read `(0.0, false)`.
    pub(super) fn turn_group(
        &self,
        group: &str,
        turns: &[(TurnKey, f32)],
    ) -> (Vec<f32>, Vec<bool>, Vec<bool>) {
        let map = self.beliefs.get(&GroupKey::TurnGroup(group.to_string()));
        let mut scores = vec![0.0f32; turns.len()];
        let mut selected = vec![false; turns.len()];
        let mut qualified = vec![false; turns.len()];
        if let Some(m) = map {
            for (i, (key, _)) in turns.iter().enumerate() {
                if let Some(&(sc, sel, q)) = m.get(&turn_belief_key(*key)) {
                    scores[i] = sc;
                    selected[i] = sel;
                    qualified[i] = q;
                }
            }
        }
        (scores, selected, qualified)
    }
}

/// Belief-map key for a turn: `"<timeline>:<index>"`.
///
/// Qualified by timeline because a belief-driven group holds many conversations
/// (`code_reading` declares one per file) and turn indices restart per timeline —
/// a bare index would make turn 3 of every file share one carried belief. Every
/// producer of a turn-group belief key (the carry, and the scheduler's
/// turn-boundary challenger) MUST route through here so they stay aligned.
pub(crate) fn turn_belief_key(turn: TurnKey) -> String {
    format!("{}:{}", turn.timeline.raw(), turn.index.0)
}

#[cfg(test)]
impl PriorBelief {
    /// A collection's member map (test-only readable view over `beliefs`).
    /// `pub(crate)` so the scheduler's resume-seeding test can assert the
    /// recovered belief's content, not merely its presence.
    pub(crate) fn coll(&self, name: &str) -> &HashMap<String, (f32, bool, bool)> {
        &self.beliefs[&GroupKey::Collection(name.to_string())]
    }

    /// A turn group's member map (test-only readable view over `beliefs`).
    pub(super) fn tgroup(&self, name: &str) -> &HashMap<String, (f32, bool, bool)> {
        &self.beliefs[&GroupKey::TurnGroup(name.to_string())]
    }

    pub(super) fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }
}
