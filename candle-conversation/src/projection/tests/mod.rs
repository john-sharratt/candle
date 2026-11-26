//! Integration tests for the projection engine, driven by YAML schemas and a mock
//! resolver.
//!
//! The mock resolver owns turn counts, scores, token counts and couplings, so each
//! test fixes what the engine sees and asserts what it emits. The cases cover
//! selection and budgets, layer and group masking, the section tree and its
//! branches, collections and their summaries, ingest and face handling, and schema
//! validation.
//!
//! - `tests_budget_and_emission`: budget spending and emission order.
//! - `tests_builder_runtime`: building and changing a prompt at runtime.
//! - `tests_collection_gates`: which collection members are shown.
//! - `tests_collection_selection`: collection selection by score and named set.
//! - `tests_collection_summary`: collection summaries and their notes.
//! - `tests_collection_tree_nodes`: collections placed as section-tree nodes.
//! - `tests_dialect_sections`: dialect sections of a system prompt.
//! - `tests_dialect_templates`: dialect templates referenced by name.
//! - `tests_exchange_and_dedup`: tool round trips and summary dedup.
//! - `tests_faces_and_ingest`: dialogue faces and ingest self-projection.
//! - `tests_layers_and_scores`: layer visibility and group scoring.
//! - `tests_optional_groups`: optional groups in a prompt.
//! - `tests_schema_validation`: schema parsing and validation.
//! - `tests_section_tree_basics`: section-tree dimensions and branches.
//! - `tests_section_tree_nesting`: nested section trees.
//! - `tests_selection`: turn selection by threshold, top-k and tag.
//! - `tests_substitution_and_zend`: template substitution and the thinking dial.
//! - `tests_turn_groups`: belief-driven turn selection and defaults.
//! - `tests_visibility`: what a projection shows, and masking.

// The fixture builders are consuming (`fn as_x(mut self) -> Self`) so they chain
// at the call site; `as_` reads better than `with_` for the flag-setting ones.
#![allow(clippy::wrong_self_convention)]

use std::collections::{HashMap, HashSet};

use super::builder::{Builder, PLAIN_PROMPT_GROUP};
use super::error::ConstructionError;
use super::hits::HitsConfig;
use super::ids::{GroupId, LayerId, Reserved, SectionId, TimelineId, TurnIndex, TurnKey};
use super::pins::Pin;
use super::project::{
    OptionalState, ProjectionMode, ProjectionTarget, SelectionState, TOOL_ROUND_SELECTOR,
};
use super::schema::{Content, CorruptTurnPolicy, DecodePriority, GatherScope, SelectionRule};
use crate::files::{FileRecord, FilesSet};
use crate::substrate::ContentResolver;
use crate::summary_tree::exchange::{exchanges, over_normals};
use crate::summary_tree::{SelectionOrigin, TurnKind};

// —— Mock resolver —————————————————————————————————————————————————————————————

/// Simple mock: explicit per-turn scores/tokens plus uniform defaults.
/// Owns turn counts so the projection engine can enumerate turns.
#[derive(Default)]
struct MockResolver {
    /// (group_raw, turn_raw) → score
    scores: HashMap<(u32, u32), f32>,
    /// (group_raw, turn_raw) → tokens
    tokens: HashMap<(u32, u32), usize>,
    /// turn_count per group
    turn_counts: HashMap<u32, u32>,
    /// section_id → score
    section_scores: HashMap<u32, f32>,
    /// section_id → token count (0 = "not sealed" for sections not present)
    section_tokens: HashMap<u32, usize>,
    default_score: f32,
    default_tokens: usize,
    /// Score-density picks returned verbatim by `summary_tree_select` —
    /// `(turn_index, origin, score)` in the chronological order the real §8
    /// selector produces (each summary node BEFORE the turns it covers).
    /// `None` ⇒ trait default (no forest → rule-based path).
    tree_picks: Option<Vec<(TurnIndex, SelectionOrigin, f32)>>,
    /// Turn indices that are summary (forest) nodes, reported by `turn_kind`
    /// as `SummaryOfSummaries`. Everything else is `Normal`.
    summary_idx: HashSet<u32>,
    /// turn_raw → the turn indices it transitively covers in the forest,
    /// reported by `node_covers`. Empty/absent ⇒ a raw leaf that covers
    /// nothing. Drives the rule-based descendant-dedup tests.
    covers: HashMap<u32, Vec<u32>>,
    /// group_raw → the turn indices that RUN ON into the next turn, standing in
    /// for the substrate's `TurnCoupling` records. A tool round-trip's call half
    /// is coupled to the response that answers it, so the two are one exchange.
    couplings: HashMap<u32, HashSet<u32>>,
    /// Stands in for "the projection target is an append-only ingest layer".
    ingest_self: bool,
    /// The groups an ingest target compounds on (`compounds_on`).
    compounds: HashSet<u32>,
    /// (group_raw, turn_raw) → the gather-scope tags the turn was written with,
    /// backing `turn_carries`. A turn absent here carries none.
    turn_tags: HashMap<(u32, u32), Vec<String>>,
    /// (group_raw, turn_raw) → the face a dialogue holds that turn by.
    faces: HashMap<(u32, u32), TurnKey>,
    /// group_raw → turns of ancestor timelines the group's conversation
    /// inherits, offered ahead of its own as the substrate offers its lineage.
    inherited: HashMap<u32, Vec<TurnKey>>,
    /// The timelines `seed_timelines` reports.
    seed: HashSet<TimelineId>,
    /// The target's files set, `files_set` reports.
    files: Option<FilesSet>,
    /// The knowledge base's file records.
    file_records: Vec<FileRecord>,
    /// Each member's section by the name it is registered under.
    members: HashMap<String, SectionId>,
}

impl MockResolver {
    fn new() -> Self {
        Self {
            default_score: 0.5,
            default_tokens: 10,
            ..Default::default()
        }
    }

    /// Register a new turn in `group`; returns its TurnIndex.
    fn append(&mut self, group: GroupId) -> TurnIndex {
        let count = self.turn_counts.entry(group.raw()).or_insert(0);
        let idx = TurnIndex(*count);
        *count += 1;
        idx
    }

    /// Mark the projection target as an append-only ingest layer — the
    /// conversation is generating its own content, not retrieving.
    fn as_ingest_self(mut self) -> Self {
        self.ingest_self = true;
        self
    }

    /// Mark `group` as one the ingest target compounds on.
    fn compounding_on(mut self, group: GroupId) -> Self {
        self.compounds.insert(group.raw());
        self
    }

    fn with_score(mut self, group: GroupId, idx: TurnIndex, score: f32) -> Self {
        self.scores.insert((group.raw(), idx.0), score);
        self
    }

    fn with_tokens(mut self, group: GroupId, idx: TurnIndex, tokens: usize) -> Self {
        self.tokens.insert((group.raw(), idx.0), tokens);
        self
    }

    fn with_default_tokens(mut self, tokens: usize) -> Self {
        self.default_tokens = tokens;
        self
    }

    fn with_default_score(mut self, score: f32) -> Self {
        self.default_score = score;
        self
    }

    /// Couple turn `idx` to the turn after it: the two (and anything the second
    /// is itself coupled to) form one exchange, selected and trimmed whole.
    fn with_coupling(mut self, group: GroupId, idx: TurnIndex) -> Self {
        self.couplings.entry(group.raw()).or_default().insert(idx.0);
        self
    }

    /// Write `idx` in `group` with `tags`, so a tagged group can scope on it.
    fn with_turn_tags(mut self, group: GroupId, idx: TurnIndex, tags: &[&str]) -> Self {
        self.turn_tags.insert(
            (group.raw(), idx.0),
            tags.iter().map(|t| t.to_string()).collect(),
        );
        self
    }

    /// Offer `turn`, on an ancestor's timeline, ahead of `group`'s own turns.
    fn with_inherited(mut self, group: GroupId, turn: TurnKey) -> Self {
        self.inherited.entry(group.raw()).or_default().push(turn);
        self
    }

    /// Name `timeline` a link of this member's seed.
    fn with_seed(mut self, timeline: TimelineId) -> Self {
        self.seed.insert(timeline);
        self
    }

    /// Give `idx` in `group` a face: the turn a dialogue holds it by.
    fn with_face(mut self, group: GroupId, idx: TurnIndex, face: TurnKey) -> Self {
        self.faces.insert((group.raw(), idx.0), face);
        self
    }

    fn with_section_score(mut self, section: SectionId, score: f32) -> Self {
        self.section_scores.insert(section.raw(), score);
        self
    }

    /// Mark a section as sealed with `tokens` tokens — projection's tool-summary
    /// gate emits the summary only when its `section_token_count` is `> 0`.
    fn with_section_tokens(mut self, section: SectionId, tokens: usize) -> Self {
        self.section_tokens.insert(section.raw(), tokens);
        self
    }

    /// Make `summary_tree_select` return `picks` verbatim (the §8 score-density
    /// path), and mark every summary-node index so `turn_kind` reports it.
    fn with_tree_picks(
        mut self,
        picks: Vec<(TurnIndex, SelectionOrigin, f32)>,
        summary_indices: &[u32],
    ) -> Self {
        self.tree_picks = Some(picks);
        self.summary_idx = summary_indices.iter().copied().collect();
        self
    }

    /// The synthetic conversation holding `group`'s turns. Each group gets its
    /// own timeline, mirroring the real model where a group is a shape and its
    /// turns live in one or more conversations under it.
    fn timeline_of(group: GroupId) -> TimelineId {
        TimelineId::for_test(group.raw() as u64)
    }

    /// Inverse of [`Self::timeline_of`] — recovers the group a mock turn key
    /// belongs to, so the `(group, index)`-keyed fixtures still resolve.
    fn group_of(turn: TurnKey) -> u32 {
        turn.timeline.raw() as u32
    }

    /// Mark turn `summary` as a summary forest node that transitively covers
    /// `covered` (the turn indices beneath it). Used to exercise the rule-based
    /// descendant-dedup without a real substrate tree.
    fn with_summary_cover(mut self, summary: u32, covered: &[u32]) -> Self {
        self.summary_idx.insert(summary);
        self.covers.insert(summary, covered.to_vec());
        self
    }
}

impl ContentResolver for MockResolver {
    fn trained_window(&self) -> Option<usize> {
        None
    }

    fn files_set(&self) -> Option<FilesSet> {
        self.files.clone()
    }

    fn files_records(&self) -> Vec<FileRecord> {
        self.file_records.clone()
    }

    fn member_section(&self, name: &str) -> Option<SectionId> {
        self.members.get(name).copied()
    }

    fn target_is_ingest_self(&self) -> bool {
        self.ingest_self
    }

    fn face_of(&self, turn: TurnKey) -> Option<TurnKey> {
        self.faces
            .get(&(Self::group_of(turn), turn.index.0))
            .copied()
    }

    fn compounds_on(&self, group: GroupId) -> bool {
        self.ingest_self && self.compounds.contains(&group.raw())
    }

    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        let tl = Self::timeline_of(group);
        let inherited = self.inherited.get(&group.raw()).into_iter().flatten();
        inherited
            .copied()
            .chain(
                (0..self.turn_counts.get(&group.raw()).copied().unwrap_or(0))
                    .map(|i| TurnKey::new(tl, TurnIndex(i))),
            )
            .collect()
    }

    fn seed_timelines(&self) -> HashSet<TimelineId> {
        self.seed.clone()
    }

    /// Exchange closure over the fixture's couplings, derived exactly as the
    /// substrate resolver does it: couplings address the `Normal` subsequence,
    /// so summary nodes are skipped before the ranges are walked.
    fn group_exchanges(&self, keys: &[TurnKey]) -> Vec<Vec<TurnKey>> {
        let mut out: Vec<Vec<TurnKey>> = Vec::new();
        let mut seen: HashSet<(u64, usize)> = HashSet::new();
        for key in keys {
            let group = Self::group_of(*key);
            let normals: Vec<TurnIndex> = (0..self.turn_counts.get(&group).copied().unwrap_or(0))
                .filter(|i| !self.summary_idx.contains(i))
                .map(TurnIndex)
                .collect();
            let raw = self.couplings.get(&group).cloned().unwrap_or_default();
            let couplings = over_normals(&raw.into_iter().collect(), &normals);
            let parts = exchanges(&couplings, normals.len());
            let Some(slot) = parts
                .iter()
                .position(|r| r.clone().any(|p| normals[p] == key.index))
            else {
                out.push(vec![*key]);
                continue;
            };
            if !seen.insert((key.timeline.raw(), slot)) {
                continue;
            }
            out.push(
                parts[slot]
                    .clone()
                    .map(|p| TurnKey::new(key.timeline, normals[p]))
                    .collect(),
            );
        }
        out
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        *self
            .tokens
            .get(&(Self::group_of(turn), turn.index.0))
            .unwrap_or(&self.default_tokens)
    }

    fn turn_score(&self, turn: TurnKey) -> f32 {
        // Mock returns a single explicit belief score.
        *self
            .scores
            .get(&(Self::group_of(turn), turn.index.0))
            .unwrap_or(&self.default_score)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        // Default 0.0 for any section the test didn't assign a score to.
        // Sections without explicit scores will lose to those with scores.
        *self.section_scores.get(&section.raw()).unwrap_or(&0.0)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        // 0 unless explicitly marked sealed — mirrors a section not present in
        // the substrate. The tool-summary gate keys off `> 0`.
        *self.section_tokens.get(&section.raw()).unwrap_or(&0)
    }

    fn turn_kind(&self, turn: TurnKey) -> TurnKind {
        if self.summary_idx.contains(&turn.index.0) {
            TurnKind::SummaryOfSummaries
        } else {
            TurnKind::Normal
        }
    }

    fn node_covers(&self, turn: TurnKey) -> Vec<TurnIndex> {
        self.covers
            .get(&turn.index.0)
            .map(|v| v.iter().copied().map(TurnIndex).collect())
            .unwrap_or_default()
    }

    fn summary_tree_select(
        &self,
        _timeline: TimelineId,
        _budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        self.tree_picks.clone()
    }

    fn turn_carries(&self, turn: TurnKey, tags: &[String]) -> bool {
        self.turn_tags
            .get(&(Self::group_of(turn), turn.index.0))
            .is_some_and(|have| have.iter().any(|t| tags.contains(t)))
    }
}

// —— Helpers ———————————————————————————————————————————————————————————————————

fn group_turn_count<'a>(
    turns: impl IntoIterator<Item = &'a super::project::ResolvedTurn>,
    gid: GroupId,
) -> usize {
    turns.into_iter().filter(|t| t.group() == gid).count()
}

fn groups_in_order<'a>(
    turns: impl IntoIterator<Item = &'a super::project::ResolvedTurn>,
) -> Vec<GroupId> {
    let mut seen = Vec::new();
    for t in turns {
        if !seen.contains(&t.group()) {
            seen.push(t.group());
        }
    }
    seen
}

// —— YAML round-trip and id assignment —————————————————————————————————————————

const SIMPLE_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "You are a helpful assistant."
    - id: values
      content: "Be honest."
layers:
  - name: ground
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 40
    groups:
      - id: facts
        selection:
          kind: top_k
          k: 3

  - name: dialogue
    window: 8000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget:
      priority: 100
      min_percent: 50
    groups:
      - id: conversation
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 8
"#;

mod tests_budget_and_emission;
mod tests_builder_runtime;
mod tests_collection_gates;
mod tests_collection_selection;
mod tests_collection_summary;
#[cfg(test)]
mod tests_collection_tree_nodes;
#[cfg(test)]
mod tests_dialect_sections;
#[cfg(test)]
mod tests_dialect_templates;
mod tests_exchange_and_dedup;
mod tests_faces_and_ingest;
mod tests_files_splice;
mod tests_layers_and_scores;
#[cfg(test)]
mod tests_optional_groups;
mod tests_schema_validation;
mod tests_section_tree_basics;
#[cfg(test)]
mod tests_section_tree_nesting;
mod tests_selection;
mod tests_substitution_and_zend;
mod tests_turn_groups;
mod tests_visibility;

use tests_collection_summary::*;
