//! Projection events — the recorded, per-decode account of what provenance
//! selected and how fast the model decoded against it.
//!
//! A [`ProjectionEvent`] is produced once per decode (at seal/Done): it pairs
//! the decode's throughput (generated-token span + wall-clock → tokens/second)
//! with the *composition* of the materialized context the model attended to,
//! bucketed into categories the GUI renders as a stacked bar map:
//!
//! - **system** — the target layer's bare system-prompt sections (the
//!   instructions) plus any structural framing.
//! - **section** — one bucket per named system-prompt *collection*
//!   (`code_read`, `repo_map`, `tools`, …) that contributed ≥ 1 section.
//! - **turns** — one bucket per conversation group that contributed ≥ 1 turn,
//!   plus the turn being submitted this decode.
//!
//! `materialized_tokens` is the sum across buckets (what provenance pulled into
//! the window); `substrate_tokens` is the total size of the on-disk store, so
//! the GUI can show "materialized M / N tokens".
//!
//! The module splits cleanly into a **pure aggregation** ([`aggregate`]) that is
//! exhaustively unit-tested with hand-built inputs, and a thin
//! [`from_projection`] adapter that classifies real [`ProjectionSegment`]s
//! against the live [`Schema`] + [`ContentResolver`].

use serde::{Deserialize, Serialize};

use super::ids::{CollectionId, GroupId, SectionId};
use super::project::{ProjectionSegment, SealedKind};
use super::schema::{Schema, SystemPromptItem};
use crate::substrate::ContentResolver;

/// Which category a materialized segment falls into. Drives the GUI bar-map
/// color and ordering (system leftmost, then section groups, then turns).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BucketKind {
    /// System-prompt instructions + structural framing.
    System,
    /// A named system-prompt collection (`code_read`, `repo_map`, `tools`).
    Section,
    /// Conversation turns selected by provenance.
    Turns,
}

impl BucketKind {
    /// Emission rank: system leftmost, section groups next, turns last.
    fn rank(self) -> u8 {
        match self {
            BucketKind::System => 0,
            BucketKind::Section => 1,
            BucketKind::Turns => 2,
        }
    }
}

/// One category of the materialized context, with its summed token count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionBucket {
    /// Display label: `"system"`, or the collection / group name.
    pub label: String,
    pub kind: BucketKind,
    pub tokens: u32,
}

/// One system-prompt section (bare or inside a collection) with the tokens it
/// holds and whether provenance selected it for this projection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedSection {
    pub name: String,
    pub tokens: u32,
    pub selected: bool,
}

/// One item of the system prompt, in materialized (declaration) order. Carries
/// the structural template glue (role/block markers) alongside content so the
/// panel shows the *complete* prompt — including the bits that close the tool
/// block and the system block — not just the retrieved content sections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SystemItem {
    /// A structural template segment (e.g. `tool_block_close`, `system_end`) —
    /// emitted verbatim as `Generated` glue. `content` is its authored marker
    /// text; shown as a glue row, never collapsible.
    Glue {
        name: String,
        content: String,
        tokens: u32,
    },
    /// A bare system-prompt content section (collapsible to its authored text).
    Section { name: String, tokens: u32 },
    /// A named collection with *every* member flagged selected/skipped — so the
    /// GUI can show what provenance picked AND skipped.
    Collection {
        name: String,
        sections: Vec<SelectedSection>,
    },
}

/// A conversation turn provenance pulled into this projection's window.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedTurn {
    /// The schema layer this turn's group belongs to (`dialogue`, `code_reading`,
    /// …) — so the panel can surface every projected memory tier, not just the
    /// dialogue conversation.
    pub layer: String,
    pub group: String,
    pub index: u32,
    pub role: String,
    pub tokens: u32,
}

/// The exact materialized-context selection for a projection — what provenance
/// pulled in, structured for the GUI's per-section "selected / not" view (vs the
/// token-count [`ProjectionBucket`]s, which are just the aggregate totals).
///
/// Only the *contributing* layer's items appear: a bare section is listed when
/// selected; a collection is listed (with all its members) when ≥1 of its
/// sections was selected — which naturally scopes the view to the projection's
/// own layer without threading a layer id through.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ProjectionSelection {
    /// The system prompt in materialized order: structural glue, bare content
    /// sections, and contributing collections interleaved exactly as emitted.
    pub system: Vec<SystemItem>,
    /// Conversation turns selected this projection.
    pub turns: Vec<SelectedTurn>,
}

/// A recorded projection: decode throughput + materialized-context composition.
///
/// All fields are `#[serde(default)]` so the persisted redo-log record stays
/// forward-compatible as the shape evolves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProjectionEvent {
    /// Generated-token index where this decode span began.
    #[serde(default)]
    pub start_token: u32,
    /// Generated-token index where it ended (== tokens generated in the span).
    #[serde(default)]
    pub end_token: u32,
    /// Wall-clock seconds the decode span took.
    #[serde(default)]
    pub seconds: f64,
    /// Decode throughput: `(end_token - start_token) / seconds`.
    #[serde(default)]
    pub tokens_per_second: f64,
    /// Sum of all bucket tokens — what provenance materialized into the window.
    #[serde(default)]
    pub materialized_tokens: u32,
    /// Total tokens stored across the whole substrate (all layers/timelines).
    #[serde(default)]
    pub substrate_tokens: u32,
    /// Per-category breakdown, ordered system → section groups → turns.
    #[serde(default)]
    pub buckets: Vec<ProjectionBucket>,
    /// Structured, per-section selection (what provenance picked AND skipped) —
    /// backs the projection bubble's substrate view.
    #[serde(default)]
    pub selection: ProjectionSelection,
}

/// One classified segment before aggregation — `(kind, label, tokens)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTokens {
    pub kind: BucketKind,
    pub label: String,
    pub tokens: u32,
}

/// Aggregate classified segments into an ordered, summed [`ProjectionEvent`].
///
/// Segments sharing a `(kind, label)` merge into one bucket (token counts
/// summed); buckets order by `kind` rank (system → section → turns) with
/// first-appearance order preserved within a kind (stable sort). Pure — no
/// schema or resolver dependency, so it is exhaustively unit-testable.
pub fn aggregate(
    segments: &[SegmentTokens],
    substrate_tokens: u32,
    start_token: u32,
    end_token: u32,
    seconds: f64,
) -> ProjectionEvent {
    let mut buckets: Vec<ProjectionBucket> = Vec::new();
    for seg in segments {
        match buckets
            .iter_mut()
            .find(|b| b.kind == seg.kind && b.label == seg.label)
        {
            Some(b) => b.tokens = b.tokens.saturating_add(seg.tokens),
            None => buckets.push(ProjectionBucket {
                label: seg.label.clone(),
                kind: seg.kind,
                tokens: seg.tokens,
            }),
        }
    }
    // Stable sort keeps first-appearance order within an equal kind rank.
    buckets.sort_by_key(|b| b.kind.rank());

    let materialized_tokens = buckets.iter().map(|b| b.tokens).sum();
    let generated = end_token.saturating_sub(start_token);
    let tokens_per_second = if seconds > 0.0 {
        generated as f64 / seconds
    } else {
        0.0
    };

    ProjectionEvent {
        start_token,
        end_token,
        seconds,
        tokens_per_second,
        materialized_tokens,
        substrate_tokens,
        buckets,
        selection: ProjectionSelection::default(),
    }
}

/// Build a [`ProjectionEvent`] from a real projection: classify each selected
/// segment against the schema (system vs named collection vs turn group), read
/// its token count from the resolver, then [`aggregate`].
///
/// Structural [`ProjectionSegment::Generated`] runs (role markers, block
/// envelopes) are framing, not content, and are skipped.
pub fn from_projection(
    segments: &[ProjectionSegment],
    schema: &Schema,
    resolver: &dyn ContentResolver,
    substrate_tokens: u32,
    start_token: u32,
    end_token: u32,
    seconds: f64,
) -> ProjectionEvent {
    let classified: Vec<SegmentTokens> = segments
        .iter()
        .filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Section(s)) => {
                let tokens = resolver.section_token_count(s.id) as u32;
                let (label, kind) = match collection_name_of(schema, s.id)
                    .or_else(|| collection_of_summary(schema, s.id))
                {
                    Some(name) => (name.to_string(), BucketKind::Section),
                    None => ("system".to_string(), BucketKind::System),
                };
                Some(SegmentTokens {
                    kind,
                    label,
                    tokens,
                })
            }
            ProjectionSegment::Sealed(SealedKind::Turn(t, _role)) => {
                let tokens = resolver.turn_token_count(t.group(), t.index()) as u32;
                let label = group_name_of(schema, t.group())
                    .unwrap_or("conversation")
                    .to_string();
                Some(SegmentTokens {
                    kind: BucketKind::Turns,
                    label,
                    tokens,
                })
            }
            ProjectionSegment::NewUserMessage { tokens } => Some(SegmentTokens {
                kind: BucketKind::Turns,
                label: "current message".to_string(),
                tokens: tokens.len() as u32,
            }),
            // Compression-internal turn-half — not part of the displayed timeline.
            ProjectionSegment::Sealed(SealedKind::TurnHalf(_)) => None,
            // Structural framing — not materialized content.
            ProjectionSegment::Generated { .. } => None,
        })
        .collect();

    let mut event = aggregate(
        &classified,
        substrate_tokens,
        start_token,
        end_token,
        seconds,
    );
    event.selection = build_selection(segments, schema, resolver);
    event
}

/// Build the structured [`ProjectionSelection`] from the selected segments and
/// the schema. A bare system section is listed when selected; a collection is
/// listed (with all its members flagged) when ≥1 member was selected — which
/// scopes the result to the contributing layer. Turns are the selected
/// conversation segments plus the current message.
fn build_selection(
    segments: &[ProjectionSegment],
    schema: &Schema,
    resolver: &dyn ContentResolver,
) -> ProjectionSelection {
    let mut selected: std::collections::HashSet<SectionId> = std::collections::HashSet::new();
    let mut turns: Vec<SelectedTurn> = Vec::new();
    // Structural template glue (`Generated`) actually emitted this projection,
    // keyed by name → its token count. `depends_on`-gated templates (e.g.
    // `tools_close`) only appear here when they really fired, so the view shows
    // exactly the framing the decode prefilled.
    let mut emitted_glue: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
    for seg in segments {
        match seg {
            ProjectionSegment::Sealed(SealedKind::Section(s)) => {
                selected.insert(s.id);
            }
            ProjectionSegment::Sealed(SealedKind::Turn(t, role)) => {
                turns.push(SelectedTurn {
                    layer: layer_name_of_group(schema, t.group())
                        .unwrap_or("")
                        .to_string(),
                    group: group_name_of(schema, t.group())
                        .unwrap_or("conversation")
                        .to_string(),
                    index: t.index().0,
                    role: role_str(*role).to_string(),
                    tokens: resolver.turn_token_count(t.group(), t.index()) as u32,
                });
            }
            ProjectionSegment::NewUserMessage { tokens } => {
                turns.push(SelectedTurn {
                    layer: "dialogue".to_string(),
                    group: "current message".to_string(),
                    index: u32::MAX,
                    role: "user".to_string(),
                    tokens: tokens.len() as u32,
                });
            }
            // Compression-internal turn-half — not a displayed dialogue turn.
            ProjectionSegment::Sealed(SealedKind::TurnHalf(_)) => {}
            ProjectionSegment::Generated { tokens, identity } => {
                emitted_glue.insert(identity.name.as_str(), tokens.len() as u32);
            }
        }
    }

    // Walk the schema in declaration order — which IS the system prompt's
    // materialized order — emitting one [`SystemItem`] per item that fired:
    // template glue (when it emitted), a selected bare section, or a collection
    // with a selected member. Template glue interleaves with content exactly as
    // the prompt is assembled, so the closing markers appear in place.
    let mut system: Vec<SystemItem> = Vec::new();
    // A collection's catalog summary shows just before its opening structural
    // marker (OUTSIDE it), matching the projected order in
    // `emit_system_prompt_items`. Captured up front, drained as it emits.
    //
    // The `selected.contains(&sum)` gate is *inherited* from the projection: a
    // summary section only lands in `selected` if `project.rs` already emitted it
    // (its own `partial && token_count > 0` gate), so the panel can't show a
    // summary the prompt didn't — no need to re-derive that gate here.
    let mut pending_summaries: std::collections::HashMap<CollectionId, (String, u32)> =
        std::collections::HashMap::new();
    for layer in &schema.layers {
        for item in &layer.system_prompt.items {
            if let SystemPromptItem::Collection(c) = item {
                if let Some(sum) = c.summary_section {
                    if selected.contains(&sum) {
                        pending_summaries.insert(
                            c.id,
                            (
                                format!("{} summary", c.name),
                                resolver.section_token_count(sum) as u32,
                            ),
                        );
                    }
                }
            }
        }
    }
    for layer in &schema.layers {
        for item in &layer.system_prompt.items {
            match item {
                SystemPromptItem::Section(s) if s.is_template => {
                    if let Some(&tokens) = emitted_glue.get(s.name.as_str()) {
                        // Emit the collection's summary just before its opening
                        // marker (e.g. `<tools>`), so it sits OUTSIDE the block —
                        // mirroring the projection.
                        if let Some(cid) = s.depends_on {
                            if let Some((name, sum_tokens)) = pending_summaries.remove(&cid) {
                                system.push(SystemItem::Section {
                                    name,
                                    tokens: sum_tokens,
                                });
                            }
                        }
                        system.push(SystemItem::Glue {
                            name: s.name.clone(),
                            content: s.content.clone(),
                            tokens,
                        });
                    }
                }
                SystemPromptItem::Section(s) if selected.contains(&s.id) => {
                    system.push(SystemItem::Section {
                        name: s.name.clone(),
                        tokens: resolver.section_token_count(s.id) as u32,
                    });
                }
                SystemPromptItem::Section(_) => {}
                SystemPromptItem::Collection(c) => {
                    // The runtime summary section (a reserved section, not a schema
                    // item) shows just before the collection's opening marker, so
                    // it was already drained above. This is the fallback for a
                    // collection with no opening marker — show it before members.
                    // Named `"<collection> summary"`; the daemon serves its text
                    // under the same key.
                    if let Some((name, tokens)) = pending_summaries.remove(&c.id) {
                        system.push(SystemItem::Section { name, tokens });
                    }
                    if c.sections.iter().any(|s| selected.contains(&s.id)) {
                        system.push(SystemItem::Collection {
                            name: c.name.clone(),
                            sections: c
                                .sections
                                .iter()
                                .map(|s| SelectedSection {
                                    name: s.name.clone(),
                                    tokens: resolver.section_token_count(s.id) as u32,
                                    selected: selected.contains(&s.id),
                                })
                                .collect(),
                        });
                    }
                }
                SystemPromptItem::SectionTree(t) => {
                    // A node emitted exactly one option's branch variant; show
                    // the node (and chosen option, for selectors) with its tokens.
                    for n in &t.nodes {
                        // A live-prefilled structural marker (`<tools>` etc.): if it
                        // fired this projection (its Generated run is in
                        // `emitted_glue`), surface it as a real glue row.
                        if n.glue.is_some() {
                            if let Some(&tokens) = emitted_glue.get(n.name.as_str()) {
                                system.push(SystemItem::Glue {
                                    name: n.name.clone(),
                                    content: n.options.first().map_or(String::new(), |o| {
                                        o.content.clone()
                                    }),
                                    tokens,
                                });
                            }
                            continue;
                        }
                        for o in &n.options {
                            if let Some(v) = o.variants.iter().find(|v| selected.contains(&v.id)) {
                                let name = if n.options.len() > 1 {
                                    format!("{}:{}", n.name, o.id)
                                } else {
                                    n.name.clone()
                                };
                                system.push(SystemItem::Section {
                                    name,
                                    tokens: resolver.section_token_count(v.id) as u32,
                                });
                            }
                        }
                        // An embedded collection node: show each selected member
                        // (its active-branch variant landed in `selected`),
                        // interleaving the collection's `member_glue` as a real
                        // structural glue row BETWEEN consecutive members —
                        // mirroring the `Generated` glue token the projection emits
                        // (and NOT baked into any member's seal), so the panel and
                        // its copy-all reproduce the exact materialized bytes.
                        if let Some(tc) = &n.collection {
                            let glue = &tc.collection.member_glue;
                            // Mirror the projection EXACTLY: it interleaves member
                            // glue only when the tokens exist (`member_glue_tokens`),
                            // so gate the panel row on the same condition (not just
                            // the non-empty string) or the panel would show a 0-token
                            // glue row the materialized prompt never contained.
                            let glue_tokens = tc
                                .collection
                                .member_glue_tokens
                                .as_ref()
                                .map(|t| t.len() as u32);
                            let mut emitted_member = false;
                            for (s, member) in tc.collection.sections.iter().zip(tc.variants.iter())
                            {
                                if let Some(v) = member.iter().find(|v| selected.contains(&v.id)) {
                                    if let (true, Some(toks)) = (emitted_member, glue_tokens) {
                                        system.push(SystemItem::Glue {
                                            name: format!("{}__member_glue", tc.collection.name),
                                            content: glue.clone(),
                                            tokens: toks,
                                        });
                                    }
                                    emitted_member = true;
                                    system.push(SystemItem::Section {
                                        name: s.name.clone(),
                                        tokens: resolver.section_token_count(v.id) as u32,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    ProjectionSelection { system, turns }
}

/// Role → wire string for [`SelectedTurn`].
fn role_str(role: crate::Role) -> &'static str {
    match role {
        crate::Role::User => "user",
        crate::Role::Assistant => "assistant",
        crate::Role::System => "system",
    }
}

/// Encode a turn's projection events to a redo-log record payload — UTF-8 JSON,
/// the same encoding `StreamDecl` uses. Paired with [`decode_events`].
pub fn encode_events(events: &[ProjectionEvent]) -> Vec<u8> {
    serde_json::to_vec(events).expect("ProjectionEvent JSON encoding is infallible")
}

/// Decode a projection-events record payload back into the turn's events.
/// Returns an empty vec on malformed input — a corrupt timeline entry must not
/// break conversation hydrate.
pub fn decode_events(payload: &[u8]) -> Vec<ProjectionEvent> {
    serde_json::from_slice(payload).unwrap_or_default()
}

/// The name of the system-prompt collection a section belongs to, or `None`
/// when the section is a bare top-level item (i.e. part of the system prompt
/// proper, not a named group).
fn collection_name_of(schema: &Schema, id: SectionId) -> Option<&str> {
    schema.layers.iter().find_map(|l| {
        l.system_prompt.items.iter().find_map(|item| match item {
            SystemPromptItem::Collection(c) if c.sections.iter().any(|s| s.id == id) => {
                Some(c.name.as_str())
            }
            _ => None,
        })
    })
}

/// The name of the collection whose *runtime summary section* is `id`. Unlike
/// [`collection_name_of`] (which matches declared members), this matches the
/// reserved summary section a collection injects before its members on partial
/// selection — that section is not a schema item, so it is classified here.
fn collection_of_summary(schema: &Schema, id: SectionId) -> Option<&str> {
    schema.layers.iter().find_map(|l| {
        l.system_prompt.items.iter().find_map(|item| match item {
            SystemPromptItem::Collection(c) if c.summary_section == Some(id) => {
                Some(c.name.as_str())
            }
            _ => None,
        })
    })
}

/// The YAML name of a group by its globally-unique id.
fn group_name_of(schema: &Schema, id: GroupId) -> Option<&str> {
    schema
        .layers
        .iter()
        .flat_map(|l| l.groups.iter())
        .find(|g| g.id == id)
        .map(|g| g.name.as_str())
}

/// The YAML name of the layer that owns the group with this id.
fn layer_name_of_group(schema: &Schema, id: GroupId) -> Option<&str> {
    schema
        .layers
        .iter()
        .find(|l| l.groups.iter().any(|g| g.id == id))
        .map(|l| l.name.as_str())
}

#[cfg(test)]
mod tests {
    use super::super::builder::Builder;
    use super::super::ids::{GroupId, Reserved, SectionId, TurnId, TurnIndex};
    use super::super::project::{
        GeneratedIdentity, ProjectionSegment, ResolvedSection, ResolvedTurn, SealedKind,
    };
    use super::super::schema::{DepthWeights, ScoreFormula};
    use super::{aggregate, from_projection, BucketKind, SegmentTokens, SystemItem};
    use crate::substrate::ContentResolver;
    use std::collections::HashMap;
    use std::sync::Arc;

    // ── aggregate (pure) ─────────────────────────────────────────────────────

    fn seg(kind: BucketKind, label: &str, tokens: u32) -> SegmentTokens {
        SegmentTokens {
            kind,
            label: label.to_string(),
            tokens,
        }
    }

    #[test]
    fn aggregate_orders_system_then_sections_then_turns() {
        // Deliberately out of order on input.
        let segs = vec![
            seg(BucketKind::Turns, "conversation", 100),
            seg(BucketKind::Section, "code_read", 800),
            seg(BucketKind::System, "system", 320),
            seg(BucketKind::Section, "repo_map", 200),
        ];
        let ev = aggregate(&segs, 10_000, 0, 480, 10.0);
        let order: Vec<(&str, BucketKind, u32)> = ev
            .buckets
            .iter()
            .map(|b| (b.label.as_str(), b.kind, b.tokens))
            .collect();
        assert_eq!(
            order,
            vec![
                ("system", BucketKind::System, 320),
                ("code_read", BucketKind::Section, 800),
                ("repo_map", BucketKind::Section, 200),
                ("conversation", BucketKind::Turns, 100),
            ]
        );
    }

    #[test]
    fn aggregate_sums_same_bucket_and_totals_materialized() {
        let segs = vec![
            seg(BucketKind::Turns, "conversation", 40),
            seg(BucketKind::Turns, "conversation", 60),
            seg(BucketKind::System, "system", 100),
        ];
        let ev = aggregate(&segs, 5_000, 10, 110, 5.0);
        // conversation merged to 100, system 100.
        assert_eq!(ev.buckets.len(), 2);
        assert_eq!(ev.materialized_tokens, 200);
        assert_eq!(ev.substrate_tokens, 5_000);
        // throughput: 100 generated tokens / 5s = 20 t/s.
        assert_eq!(ev.start_token, 10);
        assert_eq!(ev.end_token, 110);
        assert_eq!(ev.tokens_per_second, 20.0);
    }

    #[test]
    fn aggregate_zero_seconds_is_zero_tps_not_nan() {
        let ev = aggregate(&[seg(BucketKind::System, "system", 5)], 5, 0, 0, 0.0);
        assert_eq!(ev.tokens_per_second, 0.0);
        assert!(ev.tokens_per_second.is_finite());
    }

    #[test]
    fn aggregate_empty_is_empty() {
        let ev = aggregate(&[], 0, 0, 0, 1.0);
        assert!(ev.buckets.is_empty());
        assert_eq!(ev.materialized_tokens, 0);
    }

    #[test]
    fn encode_decode_events_round_trips() {
        let events = vec![
            aggregate(
                &[
                    seg(BucketKind::System, "system", 320),
                    seg(BucketKind::Section, "code_read", 800),
                ],
                42_000,
                0,
                120,
                3.0,
            ),
            aggregate(
                &[seg(BucketKind::Turns, "conversation", 540)],
                42_000,
                120,
                512,
                8.0,
            ),
        ];
        assert_eq!(super::decode_events(&super::encode_events(&events)), events);
    }

    #[test]
    fn decode_events_tolerates_garbage() {
        assert!(super::decode_events(b"not json").is_empty());
        assert!(super::decode_events(&[]).is_empty());
    }

    // ── from_projection (schema + resolver classification) ───────────────────

    const YAML: &str = r#"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    system_prompt:
      items:
        - kind: section
          id: frame
          content: "You are Zen-Code."
        - kind: collection
          name: code_read
          summary:
            chunk: 4
            categorize:
              max_tokens: 256
              system_prompt: Propose categories.
              user_prompt: Propose categories.
            assign:
              max_tokens: 128
              system_prompt: Assign by number.
              user_prompt: Assign by number.
          selection: { kind: top_k, k: 3 }
          sections:
            - id: file_a
              content: "fn a() {}"
            - id: file_b
              content: "fn b() {}"
    groups:
      - id: conversation
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 8
"#;

    /// Returns explicit token counts for sections (by raw id) and turns.
    #[derive(Default)]
    struct TokResolver {
        section_tokens: HashMap<u32, usize>,
        turn_tokens: HashMap<(u32, u32), usize>,
    }

    impl ContentResolver for TokResolver {
        fn turn_count(&self, _group: GroupId) -> u32 {
            0
        }
        fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
            *self.turn_tokens.get(&(group.raw(), index.0)).unwrap_or(&0)
        }
        fn turn_score(
            &self,
            _group: GroupId,
            _index: TurnIndex,
            _formula: ScoreFormula,
            _weights: &DepthWeights,
        ) -> f32 {
            0.0
        }
        fn section_token_count(&self, section: SectionId) -> usize {
            *self.section_tokens.get(&section.raw()).unwrap_or(&0)
        }
    }

    #[test]
    fn from_projection_buckets_system_collection_and_turns() {
        let b = Builder::from_yaml(YAML).unwrap();
        let schema = b.schema();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let conv = b.id_for_group("conversation").unwrap();
        let frame = b.id_for_section_in(dialogue, "frame").unwrap();
        let file_a = b.id_for_section_in(dialogue, "file_a").unwrap();
        let file_b = b.id_for_section_in(dialogue, "file_b").unwrap();

        let mut res = TokResolver::default();
        res.section_tokens.insert(frame.raw(), 64); // bare → system
        res.section_tokens.insert(file_a.raw(), 300); // collection code_read
        res.section_tokens.insert(file_b.raw(), 500); // collection code_read
        res.turn_tokens.insert((conv.raw(), 0), 120);
        res.turn_tokens.insert((conv.raw(), 1), 80);

        let turn = |idx: u32| {
            ProjectionSegment::Sealed(SealedKind::Turn(
                ResolvedTurn {
                    id: TurnId {
                        layer_id: dialogue,
                        group_id: conv,
                        index: TurnIndex(idx),
                    },
                },
                crate::Role::Assistant,
            ))
        };
        let section =
            |id: SectionId| ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection { id }));

        let segments = vec![
            section(frame),
            section(file_a),
            section(file_b),
            turn(0),
            turn(1),
            // structural framing — must be ignored
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![1, 2, 3]),
                identity: GeneratedIdentity {
                    name: "system_open".to_string(),
                    position: 0,
                },
            },
            ProjectionSegment::NewUserMessage {
                tokens: Arc::new(vec![9, 9, 9, 9]),
            },
        ];

        let ev = from_projection(&segments, schema, &res, 99_999, 0, 480, 12.0);

        // system (64) leftmost, then code_read (300+500=800), then turns:
        // conversation (120+80=200) and current message (4).
        let got: Vec<(&str, BucketKind, u32)> = ev
            .buckets
            .iter()
            .map(|b| (b.label.as_str(), b.kind, b.tokens))
            .collect();
        assert_eq!(
            got,
            vec![
                ("system", BucketKind::System, 64),
                ("code_read", BucketKind::Section, 800),
                ("conversation", BucketKind::Turns, 200),
                ("current message", BucketKind::Turns, 4),
            ]
        );
        // materialized excludes the skipped Generated run (3 tokens).
        assert_eq!(ev.materialized_tokens, 64 + 800 + 200 + 4);
        assert_eq!(ev.substrate_tokens, 99_999);
        assert_eq!(ev.tokens_per_second, 40.0); // 480 / 12s
    }

    const TREE_GLUE_YAML: &str = r#"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    system_prompt:
      items:
        - kind: section_tree
          nodes:
            - kind: collection
              name: tools
              selection: { kind: top_k, k: 3 }
              member_glue: "\n"
              summary:
                chunk: 4
                categorize:
                  max_tokens: 256
                  system_prompt: cat
                  user_prompt: cat
                assign:
                  max_tokens: 128
                  system_prompt: assign
                  user_prompt: assign
              sections:
                - id: tool_a
                  content: "a"
                - id: tool_b
                  content: "b"
    groups:
      - id: conversation
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 8
"#;

    #[test]
    fn from_projection_interleaves_member_glue_between_tree_collection_members() {
        use super::super::schema::SystemPromptItem;
        let mut b = Builder::from_yaml(TREE_GLUE_YAML).unwrap();
        // Tokenise so the glue carries a real token count (mock: byte → token).
        b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
            .unwrap();
        let schema = b.schema();
        let dialogue = b.id_for_layer("dialogue").unwrap();

        // The tools tree-collection node + its (single-branch) member variants.
        let tc = schema
            .layers
            .iter()
            .find(|l| l.id == dialogue)
            .unwrap()
            .system_prompt
            .items
            .iter()
            .find_map(|it| match it {
                SystemPromptItem::SectionTree(t) => {
                    t.nodes.iter().find_map(|n| n.collection.as_ref())
                }
                _ => None,
            })
            .unwrap();
        let key = tc.default_branch;
        let a = tc.member_variant(0, key).unwrap().id;
        let bb = tc.member_variant(1, key).unwrap().id;

        let mut res = TokResolver::default();
        res.section_tokens.insert(a.raw(), 10);
        res.section_tokens.insert(bb.raw(), 20);

        let section =
            |id: SectionId| ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection { id }));
        let segments = vec![section(a), section(bb)];
        let ev = from_projection(&segments, schema, &res, 0, 0, 1, 1.0);

        // The materialized system view is: member → REAL glue token → member,
        // not two members run together — so the GUI panel and its copy-all
        // reproduce the exact bytes (the glue is no longer baked into a seal).
        let sys = &ev.selection.system;
        assert_eq!(sys.len(), 3, "expected member, glue, member; got {sys:?}");
        match &sys[0] {
            SystemItem::Section { name, .. } => assert_eq!(name, "tool_a"),
            o => panic!("expected tool_a section, got {o:?}"),
        }
        match &sys[1] {
            SystemItem::Glue { content, tokens, .. } => {
                assert_eq!(content, "\n", "glue content must be the literal newline");
                assert_eq!(*tokens, 1, "glue must report its token count");
            }
            o => panic!("expected a glue row between members, got {o:?}"),
        }
        match &sys[2] {
            SystemItem::Section { name, .. } => assert_eq!(name, "tool_b"),
            o => panic!("expected tool_b section, got {o:?}"),
        }
    }

    #[test]
    fn from_projection_captures_selection_with_skipped_sections() {
        let b = Builder::from_yaml(YAML).unwrap();
        let schema = b.schema();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let conv = b.id_for_group("conversation").unwrap();
        let frame = b.id_for_section_in(dialogue, "frame").unwrap();
        let file_a = b.id_for_section_in(dialogue, "file_a").unwrap();
        let file_b = b.id_for_section_in(dialogue, "file_b").unwrap();

        let mut res = TokResolver::default();
        res.section_tokens.insert(frame.raw(), 64);
        res.section_tokens.insert(file_a.raw(), 300);
        res.section_tokens.insert(file_b.raw(), 500);
        res.turn_tokens.insert((conv.raw(), 0), 120);

        // Select frame + file_a, but NOT file_b, plus one turn.
        let segments = vec![
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection { id: frame })),
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection { id: file_a })),
            ProjectionSegment::Sealed(SealedKind::Turn(
                ResolvedTurn {
                    id: TurnId {
                        layer_id: dialogue,
                        group_id: conv,
                        index: TurnIndex(0),
                    },
                },
                crate::Role::User,
            )),
        ];

        let sel = from_projection(&segments, schema, &res, 1000, 0, 10, 1.0).selection;

        // System prompt in order: bare `frame` section, then the `code_read`
        // collection (no template items in this fixture, so no glue rows).
        assert_eq!(sel.system.len(), 2);
        match &sel.system[0] {
            SystemItem::Section { name, tokens } => {
                assert_eq!(name, "frame");
                assert_eq!(*tokens, 64);
            }
            other => panic!("expected frame section, got {other:?}"),
        }
        // The code_read collection shows BOTH members, file_b flagged skipped.
        match &sel.system[1] {
            SystemItem::Collection { name, sections } => {
                assert_eq!(name, "code_read");
                assert_eq!(sections.len(), 2);
                let a = sections.iter().find(|s| s.name == "file_a").unwrap();
                let b = sections.iter().find(|s| s.name == "file_b").unwrap();
                assert!(a.selected && a.tokens == 300);
                assert!(!b.selected && b.tokens == 500);
            }
            other => panic!("expected code_read collection, got {other:?}"),
        }

        // The selected turn is captured.
        assert_eq!(sel.turns.len(), 1);
        assert_eq!(sel.turns[0].role, "user");
        assert_eq!(sel.turns[0].index, 0);
        assert_eq!(sel.turns[0].tokens, 120);
    }

    #[test]
    fn from_projection_surfaces_emitted_template_glue_in_order() {
        use candle_transformers::models::dialect::Dialect;
        const YAML_T: &str = r#"
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    system_prompt:
      items:
        - kind: template
          id: system_open
          dialect: system_start
        - kind: section
          id: frame
          content: "You are Zen-Code."
        - kind: template
          id: system_close
          dialect: system_end
    groups:
      - id: conversation
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let b = Builder::from_yaml_with_vars_and_dialect(YAML_T, &[], Some(&dlct)).unwrap();
        let schema = b.schema();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let frame = b.id_for_section_in(dialogue, "frame").unwrap();

        let mut res = TokResolver::default();
        res.section_tokens.insert(frame.raw(), 64);

        // The assembler emits both template items as Generated glue, with the
        // content section sealed between them — exactly the shape the view must
        // reproduce so the closing markers stay visible.
        let segments = vec![
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![1, 2, 3]),
                identity: GeneratedIdentity {
                    name: "system_open".to_string(),
                    position: 0,
                },
            },
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection { id: frame })),
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![9, 9]),
                identity: GeneratedIdentity {
                    name: "system_close".to_string(),
                    position: 2,
                },
            },
        ];

        let sel = from_projection(&segments, schema, &res, 1000, 0, 10, 1.0).selection;

        assert_eq!(sel.system.len(), 3, "glue + section + glue, in order");
        match &sel.system[0] {
            SystemItem::Glue {
                name,
                content,
                tokens,
            } => {
                assert_eq!(name, "system_open");
                assert_eq!(content, "<|im_start|>system\n");
                assert_eq!(*tokens, 3);
            }
            other => panic!("expected system_open glue, got {other:?}"),
        }
        match &sel.system[1] {
            SystemItem::Section { name, tokens } => {
                assert_eq!(name, "frame");
                assert_eq!(*tokens, 64);
            }
            other => panic!("expected frame section, got {other:?}"),
        }
        match &sel.system[2] {
            SystemItem::Glue {
                name,
                content,
                tokens,
            } => {
                assert_eq!(name, "system_close");
                assert_eq!(content, "<|im_end|>\n");
                assert_eq!(*tokens, 2);
            }
            other => panic!("expected system_close glue, got {other:?}"),
        }
    }

    #[test]
    fn from_projection_summary_shows_outside_tools_markers() {
        use candle_transformers::models::dialect::Dialect;
        const YAML: &str = r#"
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: c
          user_prompt: c
        assistant:
          system_prompt: c
          user_prompt: c
    system_prompt:
      items:
        - kind: template
          id: tools_open
          dialect: tool_block_open
          depends_on: tools
        - kind: collection
          name: tools
          selection: { kind: top_k, k: 1 }
          summary:
            chunk: 4
            categorize:
              max_tokens: 256
              system_prompt: x
              user_prompt: x
            assign:
              max_tokens: 128
              system_prompt: x
              user_prompt: x
          sections:
            - id: t1
              content: "tool"
        - kind: template
          id: tools_close
          dialect: tool_block_close
          depends_on: tools
    groups:
      - id: conversation
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let mut b = Builder::from_yaml_with_vars_and_dialect(YAML, &[], Some(&dlct)).unwrap();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let tools = b.id_for_collection_in(dialogue, "tools").unwrap();
        let summary = SectionId::reserved(Reserved::ToolSummary);
        b.set_collection_summary_section(dialogue, tools, summary)
            .unwrap();
        let t1 = b.id_for_section_in(dialogue, "t1").unwrap();
        let schema = b.schema();

        let mut res = TokResolver::default();
        res.section_tokens.insert(summary.raw(), 40);
        res.section_tokens.insert(t1.raw(), 8);

        // The projected order: summary OUTSIDE, then <tools>, member, </tools>.
        let segments = vec![
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection { id: summary })),
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![1]),
                identity: GeneratedIdentity {
                    name: "tools_open".to_string(),
                    position: 0,
                },
            },
            ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection { id: t1 })),
            ProjectionSegment::Generated {
                tokens: Arc::new(vec![2]),
                identity: GeneratedIdentity {
                    name: "tools_close".to_string(),
                    position: 2,
                },
            },
        ];
        let sel = from_projection(&segments, schema, &res, 1000, 0, 10, 1.0).selection;

        // The catalog summary must appear BEFORE the `tools_open` glue — i.e.
        // outside the `<tools>` block — in the panel composition.
        let sum_pos = sel
            .system
            .iter()
            .position(
                |it| matches!(it, SystemItem::Section { name, .. } if name == "tools summary"),
            )
            .unwrap_or_else(|| panic!("summary not in composition: {:?}", sel.system));
        let open_pos = sel
            .system
            .iter()
            .position(|it| matches!(it, SystemItem::Glue { name, .. } if name == "tools_open"))
            .unwrap_or_else(|| panic!("tools_open glue not in composition: {:?}", sel.system));
        assert!(
            sum_pos < open_pos,
            "summary must show before <tools>: {:?}",
            sel.system
        );
    }
}
