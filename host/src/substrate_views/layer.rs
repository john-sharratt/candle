//! A tenant's `layer/{name}` view — the conversations in one layer, fetched
//! when the layer's card expands.

use std::cmp::Reverse;

use candle_conversation::projection::{GroupSchema, TimelineId};
use candle_conversation::substrate::Substrate;
use serde::Serialize;

/// The conversations in one layer.
#[derive(Serialize)]
pub struct LayerConversations {
    pub conversations: Vec<ConvView>,
}

/// One conversation timeline as it sits in a layer.
#[derive(Serialize)]
pub struct ConvView {
    /// Raw timeline id as a decimal string, so a page never rounds it through
    /// a float's 2^53 mantissa.
    pub timeline: String,
    pub conv_id: String,
    /// What the tenant calls it — see [`Unit::label`].
    pub label: String,
    /// See [`Unit::repo`].
    pub repo: String,
    pub archived: bool,
    pub group: String,
    /// Turn count including summary nodes.
    pub turn_count: u32,
    /// The timeline's own sealed turn tokens — never the shared sections, which
    /// every conversation projects and none owns.
    pub tokens: usize,
    /// Summary-forest nodes (SoT + SoS) — how much has been compressed.
    pub summary_nodes: usize,
    /// See [`Unit::branches`].
    pub branches: Vec<String>,
    /// See [`Unit::commit`].
    pub commit: String,
}

/// How a tenant names one of its conversations in a layer's list.
pub struct Unit {
    /// For an ingested unit, the path inside its repository (`src/lib.rs`,
    /// `src/`, `/` for the repository's root) — the repository is
    /// [`Self::repo`]. Otherwise the conversation's title.
    pub label: String,
    /// For an ingested unit, the repository it is in; empty otherwise.
    pub repo: String,
    /// For an ingested unit, every branch whose tip holds the version this
    /// conversation read, default branch first — what tells one `candle/` row
    /// from another. Empty otherwise.
    pub branches: Vec<String>,
    /// For a file reading, the commit it read the file at — the version it
    /// holds, which every branch sharing that version shares, so the view shows
    /// this rather than [`Self::branches`]. Empty for anything else.
    pub commit: String,
}

impl Unit {
    /// A conversation named by its [`title`], with no repository, branch or
    /// commit.
    pub fn titled(s: &Substrate, tl: TimelineId) -> Self {
        Self {
            label: title(s, tl).unwrap_or_default(),
            repo: String::new(),
            branches: Vec::new(),
            commit: String::new(),
        }
    }
}

/// A conversation's sidebar label, else its `conv_id`; `None` when it has
/// neither.
pub fn title(s: &Substrate, tl: TimelineId) -> Option<String> {
    s.label_of(tl)
        .filter(|l| !l.is_empty())
        .or_else(|| s.conv_id_of(tl).filter(|c| !c.is_empty()))
        .map(|s| s.to_string())
}

/// The conversations targeting a layer's groups, largest first, each named by
/// `unit`.
///
/// Enumerates EVERY registered timeline of the groups — not just the
/// conversations a client opened — so a layer the tenant fills itself
/// surfaces its timelines too. `titler` is a timeline the tenant keeps for
/// itself and leaves out; empty and tombstoned timelines are left out too.
pub fn layer_conversations(
    s: &Substrate,
    groups: &[GroupSchema],
    titler: Option<TimelineId>,
    unit: impl Fn(&Substrate, TimelineId) -> Unit,
) -> LayerConversations {
    let mut conversations: Vec<ConvView> = Vec::new();
    for g in groups {
        for tl in s.timelines_for_group(g.id) {
            if Some(tl) == titler || s.is_tombstoned(tl) || s.turn_count(tl) == 0 {
                continue;
            }
            let summary_nodes = s
                .turn_indices(tl)
                .filter(|idx| {
                    s.tree_meta_of(tl, *idx)
                        .is_some_and(|m| m.kind.is_summary())
                })
                .count();
            let Unit {
                label,
                repo,
                branches,
                commit,
            } = unit(s, tl);
            conversations.push(ConvView {
                timeline: tl.raw().to_string(),
                conv_id: s.conv_id_of(tl).unwrap_or_default().to_string(),
                label,
                repo,
                archived: s.is_archived(tl),
                group: g.name.clone(),
                turn_count: s.turn_count(tl),
                tokens: s.timeline_token_total(tl),
                summary_nodes,
                branches,
                commit,
            });
        }
    }
    // Largest first — the most substantial timelines lead each layer.
    conversations.sort_by_key(|c| Reverse(c.turn_count));
    LayerConversations { conversations }
}

#[cfg(test)]
mod tests {
    use candle_conversation::projection::{Builder, Conversation};
    use serde_json::json;

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

    /// A layer nothing has written to lists no conversations.
    #[test]
    fn an_empty_layer_lists_nothing() {
        let builder = Builder::from_yaml(SCHEMA).unwrap();
        let conv = Conversation::new();
        let read = conv.read();
        let layer = layer_conversations(
            &read,
            &builder.schema().layers[0].groups,
            None,
            Unit::titled,
        );
        assert_eq!(
            serde_json::to_value(&layer).unwrap(),
            json!({"conversations": []})
        );
    }
}
