//! A tenant's `system-prompt` view — the shared system prompt's framing,
//! fetched when its card expands.

use candle_conversation::projection::{Schema, SystemPromptItem, SystemPromptSchema};
use candle_conversation::substrate::Substrate;
use serde::Serialize;

/// The single shared system prompt.
#[derive(Serialize)]
pub struct SystemPromptView {
    /// The layer currently targeted — whose dials pick the active section-tree
    /// branch of this shared prompt.
    pub target_layer: String,
    /// Ordered system-prompt item labels (sections / collection / tree).
    pub items: Vec<String>,
    /// Always-emit framing sections, with authored text + token cost.
    pub sections: Vec<SectionView>,
    /// Live tool-catalog size (the catalog itself loads from `/v1/tools` on
    /// expand).
    pub tool_count: usize,
}

#[derive(Serialize)]
pub struct SectionView {
    pub name: String,
    pub tokens: usize,
    pub blocks: usize,
    /// The section's authored text — what gets tokenized into these blocks.
    /// Shown when the viewer expands the section.
    pub content: String,
}

/// The shared prompt of `schema` as `s` holds it: its item labels, and its
/// always-emit framing sections with their materialized token cost. The tools
/// collection and the section tree appear as labels in `items`; `tool_count` is
/// the tenant's live catalog size.
pub fn system_prompt(
    s: &Substrate,
    schema: &Schema,
    target_layer: String,
    tool_count: usize,
) -> SystemPromptView {
    let sections: Vec<SectionView> = schema
        .system_prompt
        .items
        .iter()
        .filter_map(|it| match it {
            SystemPromptItem::Section(sec) => Some(SectionView {
                name: sec.name.clone(),
                tokens: s.section_tokens_of(sec.id).len(),
                blocks: s.section_block_count(sec.id).unwrap_or(0),
                content: sec.content.clone(),
            }),
            _ => None,
        })
        .collect();

    SystemPromptView {
        target_layer,
        items: system_prompt_labels(&schema.system_prompt),
        sections,
        tool_count,
    }
}

/// Declaration-order labels for the system prompt's items — a bare section by
/// name, a collection by name, a section tree by node count.
pub fn system_prompt_labels(sp: &SystemPromptSchema) -> Vec<String> {
    sp.items
        .iter()
        .map(|it| match it {
            SystemPromptItem::Section(s) => s.name.clone(),
            // Member count is NOT taken from `c.sections` — a collection whose
            // members are installed at runtime has none in a fresh schema
            // parse. The viewer lists the live members separately.
            SystemPromptItem::Collection(c) => format!("{} · collection", c.name),
            SystemPromptItem::SectionTree(t) => format!("section tree ({} nodes)", t.nodes.len()),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use candle_conversation::projection::{Builder, Conversation};
    use serde_json::json;

    use super::*;

    const SCHEMA: &str = r#"
system_prompt:
  sections:
    - id: stance
      content: "Speak as yourself."
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

    /// A section nothing has materialized reads with its authored text and no
    /// tokens yet.
    #[test]
    fn the_framing_sections_carry_their_text() {
        let builder = Builder::from_yaml(SCHEMA).unwrap();
        let conv = Conversation::new();
        let read = conv.read();
        let view = system_prompt(&read, builder.schema(), "interaction".into(), 9);
        assert_eq!(
            serde_json::to_value(&view).unwrap(),
            json!({
                "target_layer": "interaction",
                "items": ["stance"],
                "sections": [{
                    "name": "stance",
                    "tokens": 0,
                    "blocks": 0,
                    "content": "Speak as yourself.",
                }],
                "tool_count": 9,
            })
        );
    }
}
