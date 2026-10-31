//! A tenant's line in `GET /v1/substrate` — the lightweight top of the tree.
//!
//! No conversations, section text, or tool catalog: those are fetched
//! per-expansion, so a page's first load and its periodic refresh stay cheap
//! however large the corpus grows.

use std::path::Path;

use candle_conversation::projection::{Conversation, GroupSchema, Schema, TimelineId};
use candle_conversation::substrate::Substrate;
use serde::Serialize;

use super::segments::{segment_files, SegmentView};
use super::selection::fmt_selection;

/// The top of the tree.
#[derive(Serialize)]
pub struct SubstrateOverview {
    pub storage: Storage,
    pub counts: Counts,
    /// The projection target layer — where the live conversations live.
    pub target_layer: String,
    /// Every declared projection layer, low → high, with its config and a count
    /// of the conversations targeting it (the list is fetched on expand).
    pub layers: Vec<LayerView>,
}

/// On-disk footprint of the segmented redo log.
#[derive(Serialize)]
pub struct Storage {
    /// One entry per `seg-*.log` file, ascending by id; the last is active.
    pub segments: Vec<SegmentView>,
    /// Sum of every segment file's size on disk.
    pub total_bytes: u64,
    /// Live KV chunk records currently indexed in RAM. A record is one 32-token
    /// block of ONE attention layer, so this is a count of records, not of tokens.
    pub live_chunks: usize,
    /// Fraction of record bytes that are dead (superseded + tombstoned) and
    /// reclaimable by compaction. `None` when the persistence lock was
    /// momentarily held (a compaction in flight) — the viewer shows "—".
    pub dead_ratio: Option<f32>,
}

/// The substrate's totals. A count is always `*_count`
/// (`docs/shared_model_host.md` §3.5).
#[derive(Serialize)]
pub struct Counts {
    pub timeline_count: usize,
    pub conversation_count: usize,
    pub section_count: usize,
    /// Tokens sealed in the substrate: every live timeline's turns and the shared
    /// sections, each counted once. The layers' `tokens` sum to this less the
    /// sections and any timeline no layer lists.
    pub token_count: usize,
}

/// One projection layer's schema config plus a conversation count (the list
/// itself is fetched from the tenant's `layer/{name}` view on expand).
#[derive(Serialize)]
pub struct LayerView {
    pub name: String,
    pub description: String,
    /// Per-target token budget when this layer is the projection target.
    pub window: usize,
    /// Flex priority when another layer is the target.
    pub priority: f32,
    /// Declared turn groups (name + selection rule), in declaration order.
    pub groups: Vec<GroupView>,
    /// This layer's dial overrides for the shared system prompt, as
    /// `"<selector> = <option>"` labels. Empty inherits the section-tree defaults.
    pub dials: Vec<String>,
    /// Number of conversations targeting this layer.
    pub conversation_count: usize,
    /// Total sealed tokens across every conversation in this layer.
    pub tokens: usize,
    /// Runtime projection kill switch: `true` while this layer is toggled OFF
    /// (excluded from projection assembly). In-memory only — clears on restart.
    pub disabled: bool,
}

#[derive(Serialize)]
pub struct GroupView {
    pub name: String,
    pub selection: String,
}

/// The overview of `conv`, read under `schema`, with its redo log in `dir`.
///
/// `titler` is a timeline the tenant keeps for itself and leaves out of every
/// count; `target_layer` is the layer its live conversations are written to.
/// Reads the shared substrate through the handle — one version of it, held for
/// the walk — never a rebuild from the log.
pub fn overview(
    conv: &Conversation,
    schema: &Schema,
    titler: Option<TimelineId>,
    target_layer: String,
    dir: &Path,
) -> SubstrateOverview {
    let (segments, total_bytes) = segment_files(dir);

    let dead_ratio = conv.dead_ratio();
    let read = conv.read();
    let s = &*read;

    let counts = Counts {
        timeline_count: s.timeline_count(),
        conversation_count: s.conversation_count(),
        section_count: s.section_count(),
        token_count: s.corpus_token_total(),
    };

    // Layer metadata only — each layer carries a conversation COUNT, never the
    // list.
    let layers: Vec<LayerView> = schema
        .layers
        .iter()
        .map(|l| {
            let (conversation_count, tokens) = layer_totals(s, &l.groups, titler);
            LayerView {
                name: l.name.clone(),
                description: l.description.trim().to_string(),
                window: l.window,
                priority: l.budget.priority,
                groups: l
                    .groups
                    .iter()
                    .map(|g| GroupView {
                        name: g.name.clone(),
                        selection: fmt_selection(&g.selection),
                    })
                    .collect(),
                dials: l
                    .dials
                    .iter()
                    .map(|(sel, opt)| format!("{sel} = {opt}"))
                    .collect(),
                conversation_count,
                tokens,
                disabled: s.is_layer_disabled(&l.name),
            }
        })
        .collect();

    SubstrateOverview {
        storage: Storage {
            segments,
            total_bytes,
            live_chunks: s.live_chunk_count(),
            dead_ratio,
        },
        counts,
        target_layer,
        layers,
    }
}

/// `(conversation count, total tokens)` across the non-empty, non-tombstoned,
/// non-titler timelines targeting a layer's groups — the cheap headline numbers
/// the overview shows without materializing the conversation list.
fn layer_totals(
    s: &Substrate,
    groups: &[GroupSchema],
    titler: Option<TimelineId>,
) -> (usize, usize) {
    groups
        .iter()
        .flat_map(|g| s.timelines_for_group(g.id))
        .filter(|tl| Some(*tl) != titler && !s.is_tombstoned(*tl) && s.turn_count(*tl) > 0)
        .fold((0, 0), |(n, tok), tl| {
            (n + 1, tok + s.timeline_token_total(tl))
        })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use candle_conversation::projection::Builder;
    use serde_json::json;

    use super::*;

    const SCHEMA: &str = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: interaction
    description: "  What the character is living through.  "
    window: 9000
    summary:
      turns:
        max_tokens: 64
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: primary
        selection: { kind: conversation, recent: 4 }
  - name: dreams
    window: 4000
    summary:
      turns:
        max_tokens: 64
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: dreamt
        selection: { kind: top_k, k: 3 }
"#;

    /// An empty substrate under a schema: every declared layer with its config,
    /// nothing counted, and the segments on disk.
    #[test]
    fn an_empty_substrate_lists_its_layers_and_its_segments() {
        let builder = Builder::from_yaml(SCHEMA).unwrap();
        let conv = Conversation::new();
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("seg-000001.log"), [0u8; 4]).unwrap();

        let o = overview(
            &conv,
            builder.schema(),
            None,
            "interaction".into(),
            dir.path(),
        );
        let mut v = serde_json::to_value(&o).unwrap();
        // A fresh ephemeral log has nothing dead yet, whichever way it reads.
        v["storage"]
            .as_object_mut()
            .unwrap()
            .remove("dead_ratio")
            .unwrap();
        assert_eq!(
            v,
            json!({
                "storage": {
                    "segments": [{"id": 1, "active": true, "bytes": 4}],
                    "total_bytes": 4,
                    "live_chunks": 0,
                },
                "counts": {
                    "timeline_count": 0,
                    "conversation_count": 0,
                    "section_count": 0,
                    "token_count": 0,
                },
                "target_layer": "interaction",
                "layers": [
                    {
                        "name": "interaction",
                        "description": "What the character is living through.",
                        "window": 9000,
                        "priority": 50.0,
                        "groups": [{"name": "primary", "selection": "recent 4 + top 0"}],
                        "dials": [],
                        "conversation_count": 0,
                        "tokens": 0,
                        "disabled": false,
                    },
                    {
                        "name": "dreams",
                        "description": "",
                        "window": 4000,
                        "priority": 50.0,
                        "groups": [{"name": "dreamt", "selection": "top 3"}],
                        "dials": [],
                        "conversation_count": 0,
                        "tokens": 0,
                        "disabled": false,
                    },
                ],
            })
        );
    }

    /// A toggled layer reads as disabled.
    #[test]
    fn a_toggled_layer_reads_as_disabled() {
        let builder = Builder::from_yaml(SCHEMA).unwrap();
        let conv = Conversation::new();
        assert!(conv.toggle_layer("dreams"));
        let dir = tempfile::tempdir().unwrap();
        let o = overview(&conv, builder.schema(), None, String::new(), dir.path());
        let disabled: Vec<(&str, bool)> = o
            .layers
            .iter()
            .map(|l| (l.name.as_str(), l.disabled))
            .collect();
        assert_eq!(disabled, [("interaction", false), ("dreams", true)]);
    }
}
