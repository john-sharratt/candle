//! A tenant's `timeline/{tl}` view — one conversation's summary forest, every
//! turn decoded, fetched when the conversation's card expands.

use std::collections::{BTreeMap, HashSet};

use candle_conversation::projection::{Schema, TimelineId};
use candle_conversation::substrate::Substrate;
use candle_conversation::summary_tree::TurnKind;
use candle_conversation::turn_layout::TurnLayout;
use serde::Serialize;

/// One conversation's forest.
#[derive(Serialize)]
pub struct TimelineDetail {
    pub timeline: String,
    pub conv_id: String,
    pub label: String,
    pub archived: bool,
    pub layer: String,
    pub group: String,
    pub total_tokens: usize,
    /// The conversation's persisted `custom` metadata — content hashes, source
    /// path, and the `forked_from` lineage pointer.
    pub custom: BTreeMap<String, String>,
    /// The lineage this conversation actually projects with, oldest ancestor
    /// first, ending in itself — what `Substrate::inherited_chain` resolves
    /// `forked_from` to after dropping retired ancestors and applying the
    /// token cap. A single entry means it inherits nothing.
    pub inherited_chain: Vec<String>,
    /// Forest peaks — the orphan summary nodes that are the window entry points.
    pub peaks: Vec<u32>,
    /// Turn indices that open a coupled exchange (a tool-call turn joined with
    /// its response turn) — the `from` side of each `TurnCoupling`.
    pub couplings: Vec<u32>,
    /// How many turns `turns` holds.
    pub turn_count: usize,
    /// Every turn in append order.
    pub turns: Vec<TurnView>,
}

/// One turn / summary node in the forest.
#[derive(Serialize)]
pub struct TurnView {
    pub index: u32,
    /// `"normal"` | `"sot"` (summary of turns) | `"sos"` (summary of summaries).
    pub kind: &'static str,
    /// Tree height (0 for a normal leaf turn).
    pub height: u8,
    pub tokens: usize,
    pub no_think: bool,
    /// Child turn indices this summary node compresses (empty for a normal turn).
    pub children: Vec<u32>,
    /// `true` when this turn opens a coupled exchange (its response turn follows).
    pub coupled: bool,
    /// `true` when this node is a forest peak (a window entry point).
    pub peak: bool,
    /// Decoded user half (the derived scope for a summary node).
    pub user: String,
    /// Decoded assistant half (the summary text for a summary node).
    pub assistant: String,
    /// K/V segment layout (real vs ethereal glue, user, thinking, assistant) —
    /// present for normal turns, letting the viewer colorize the exact segments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<TurnLayout>,
}

/// The forest of the timeline whose raw id is `tl_raw`, its layer and group
/// named by `schema`. `None` for an id that is no timeline, and for one with
/// nothing sealed and no registration.
pub fn timeline(s: &Substrate, schema: &Schema, tl_raw: u64) -> Option<TimelineDetail> {
    let tl = TimelineId::from_raw(tl_raw)?;
    if s.turn_count(tl) == 0 && s.timeline_target(tl).is_none() {
        return None;
    }

    let (layer, group) = match s.timeline_target(tl) {
        Some((lid, gid)) => {
            let l = schema.layers.iter().find(|l| l.id == lid);
            (
                l.map(|l| l.name.clone()).unwrap_or_default(),
                l.and_then(|l| l.groups.iter().find(|g| g.id == gid))
                    .map(|g| g.name.clone())
                    .unwrap_or_default(),
            )
        }
        None => (String::new(), String::new()),
    };

    let peak_set: HashSet<u32> = s.peaks_of(tl).into_iter().map(|(idx, _)| idx.0).collect();
    let couplings = s.couplings_of(tl);
    let mut coupling_list: Vec<u32> = couplings.iter().copied().collect();
    coupling_list.sort_unstable();

    let mut indices: Vec<_> = s.turn_indices(tl).collect();
    indices.sort_by_key(|i| i.0);
    let turns: Vec<TurnView> = indices
        .into_iter()
        .map(|idx| {
            let (kind, height, children) = match s.tree_meta_of(tl, idx) {
                Some(m) => (
                    match m.kind {
                        TurnKind::Normal => "normal",
                        TurnKind::SummaryOfTurns => "sot",
                        TurnKind::SummaryOfSummaries => "sos",
                    },
                    m.tree_height,
                    m.children.iter().map(|c| c.0).collect::<Vec<u32>>(),
                ),
                None => ("normal", 0, Vec::new()),
            };
            TurnView {
                index: idx.0,
                kind,
                height,
                tokens: s.turn_token_count_of(tl, idx),
                no_think: s.turn_no_think(tl, idx),
                children,
                coupled: couplings.contains(&idx.0),
                peak: peak_set.contains(&idx.0),
                user: s.user_text_of(tl, idx),
                assistant: s.assistant_text_of(tl, idx),
                layout: s.turn_layout(tl, idx),
            }
        })
        .collect();

    Some(TimelineDetail {
        timeline: tl_raw.to_string(),
        conv_id: s.conv_id_of(tl).unwrap_or_default().to_string(),
        label: s.label_of(tl).unwrap_or_default().to_string(),
        archived: s.is_archived(tl),
        layer,
        group,
        total_tokens: s.timeline_token_total(tl),
        custom: s
            .custom_of(tl)
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default(),
        inherited_chain: s
            .inherited_chain(tl)
            .into_iter()
            .map(|t| t.raw().to_string())
            .collect(),
        peaks: {
            let mut p: Vec<u32> = peak_set.into_iter().collect();
            p.sort_unstable();
            p
        },
        couplings: coupling_list,
        turn_count: turns.len(),
        turns,
    })
}

#[cfg(test)]
mod tests {
    use candle_conversation::projection::{Builder, Conversation};

    use super::*;

    const SCHEMA: &str = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: interaction
    window: 9000
    summary:
      turns:
        max_tokens: 64
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: primary
        selection: { kind: top_k, k: 3 }
"#;

    /// Neither the zero id nor a timeline nothing registered has a forest.
    #[test]
    fn an_unknown_timeline_has_no_forest() {
        let builder = Builder::from_yaml(SCHEMA).unwrap();
        let conv = Conversation::new();
        let read = conv.read();
        assert!(timeline(&read, builder.schema(), 0).is_none());
        assert!(timeline(&read, builder.schema(), 7).is_none());
    }
}
