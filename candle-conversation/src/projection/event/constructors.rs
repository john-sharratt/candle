//! Builds the events that are not read off a projection: the aggregation of
//! segments, and the events for staged ingest and summary nodes.
//!
//! Aggregation merges segments with the same kind and label, sums their tokens,
//! and orders the buckets system first, then sections, then turns, keeping
//! first-appearance order within a kind. It is a pure function, so tests drive it
//! with hand-built input. Ingest turns and summary nodes run no projection, so
//! their events are built from what they hold. A staged ingest event names the
//! turn and its predecessor as the references a scan follows back to the turn. A
//! summary node event lists the node first and then the turns it covers.

use super::*;

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

    ProjectionEvent {
        start_token,
        seconds,
        materialized_tokens,
        substrate_tokens,
        buckets,
        selection: ProjectionSelection::default(),
        materialized: Vec::new(),
        self_reference: false,
        groups: Vec::new(),
        pieces: Vec::new(),
        unpooled: Vec::new(),
    }
}

/// A synthesized staged event for an ingest turn (a repo_map cluster, a
/// code_read scope, the per-file summary decode).  These append-only trunks
/// run no per-turn projection, so the event is built from the trunk's known
/// composition instead of a projection walk: `system` is the layer's fixed
/// sections, `turns` the turn itself plus its immediate predecessor — the
/// resolvable `(timeline, index)` references a wide-Q scan follows to bring
/// the turn in.  Pure; the caller persists it keyed to the turn's stream id.
pub fn staged_ingest_event(
    start_token: u32,
    seconds: f64,
    system: Vec<SystemItem>,
    turns: Vec<SelectedTurn>,
) -> ProjectionEvent {
    let system_tokens: u32 = system
        .iter()
        .map(|item| match item {
            SystemItem::Glue { tokens, .. } | SystemItem::Section { tokens, .. } => *tokens,
            SystemItem::Collection {
                sections,
                member_glue_tokens,
                ..
            } => {
                let picked: Vec<u32> = sections
                    .iter()
                    .filter(|s| s.selected)
                    .map(|s| s.tokens)
                    .collect();
                // One glue run between each consecutive pair of members.
                let glue = member_glue_tokens * picked.len().saturating_sub(1) as u32;
                picked.iter().sum::<u32>() + glue
            }
        })
        .sum();
    let turn_tokens: u32 = turns.iter().map(|t| t.tokens).sum();
    let mut buckets = Vec::with_capacity(2);
    if system_tokens > 0 {
        buckets.push(ProjectionBucket {
            label: "system".to_string(),
            kind: BucketKind::System,
            tokens: system_tokens,
        });
    }
    if turn_tokens > 0 {
        buckets.push(ProjectionBucket {
            label: "turns".to_string(),
            kind: BucketKind::Turns,
            tokens: turn_tokens,
        });
    }
    ProjectionEvent {
        start_token,
        seconds,
        materialized_tokens: system_tokens + turn_tokens,
        substrate_tokens: 0,
        buckets,
        selection: ProjectionSelection { system, turns },
        materialized: Vec::new(),
        self_reference: false,
        groups: Vec::new(),
        pieces: Vec::new(),
        unpooled: Vec::new(),
    }
}

/// A synthesized event for a summariser node (SoT leaf digest or SoS rollup).
/// Compression frames run no projection, so `system` is honestly empty;
/// `selection.turns[0]` is the node itself and the rest are the turns it
/// covers — the resolvable chain a wide-Q scan follows: sig hit on the node's
/// stream → this event → node + children `(timeline, index)` keys.
pub fn summary_node_event(
    layer: &str,
    group: &str,
    timeline: u64,
    node: (TurnIndex, TurnKind, u32),
    children: Vec<(TurnIndex, TurnKind, u32)>,
) -> ProjectionEvent {
    let (node_idx, node_kind, node_tokens) = node;
    let mut turns = Vec::with_capacity(1 + children.len());
    turns.push(SelectedTurn {
        layer: layer.to_string(),
        group: group.to_string(),
        index: node_idx.0,
        role: "assistant".to_string(),
        tokens: node_tokens,
        kind: node_kind,
        reason: None,
        timeline: Some(timeline),
        // The node and its covered turns form the recorded chain; they carry no
        // belief score (compression frames run no belief scan).
        selected: true,
        score: 0.0,
        qualified: false,
    });
    for (idx, kind, tokens) in children {
        turns.push(SelectedTurn {
            layer: layer.to_string(),
            group: group.to_string(),
            index: idx.0,
            role: "assistant".to_string(),
            tokens,
            kind,
            reason: None,
            timeline: Some(timeline),
            selected: true,
            score: 0.0,
            qualified: false,
        });
    }
    let total: u32 = turns.iter().map(|t| t.tokens).sum();
    ProjectionEvent {
        start_token: 0,
        seconds: 0.0,
        materialized_tokens: total,
        substrate_tokens: 0,
        buckets: vec![ProjectionBucket {
            label: group.to_string(),
            kind: BucketKind::Turns,
            tokens: total,
        }],
        selection: ProjectionSelection {
            system: Vec::new(),
            turns,
        },
        materialized: Vec::new(),
        self_reference: false,
        groups: Vec::new(),
        pieces: Vec::new(),
        unpooled: Vec::new(),
    }
}
