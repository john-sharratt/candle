//! The prefix a collection's members seal against.
//!
//! When a conversation is created, the system prompt is walked in order and a
//! collection's members are sealed against the sections laid down before it. A
//! member submitted later has to seal against that same prefix, so this walk is
//! repeated here from the schema alone.

use super::ids::CollectionId;
use super::prefix_item::PrefixItem;
use super::schema::SystemPromptItem;

/// What is in front of the top-level collection `collection`, in the order the
/// creation walk accumulates it: bare sections, the non-template members of
/// earlier collections — or, for an earlier collection with a reserved range,
/// the range as a hole of its size on an engine trained to `native` positions
/// — and the default branch of earlier section trees.
///
/// `None` when the schema has no top-level collection by that id. A collection
/// embedded in a section tree seals once per branch, which a single submitted
/// section does not do.
pub fn prefix_before_collection(
    items: &[SystemPromptItem],
    collection: CollectionId,
    native: Option<usize>,
) -> Option<Vec<PrefixItem>> {
    let mut prefix = Vec::new();
    for item in items {
        match item {
            SystemPromptItem::Section(s) => prefix.push(PrefixItem::Section(s.id)),
            SystemPromptItem::Collection(c) => {
                if c.id == collection {
                    return Some(prefix);
                }
                match c.range {
                    Some(range) => prefix.push(PrefixItem::Hole(range.positions(native))),
                    None => prefix.extend(
                        c.sections
                            .iter()
                            .filter(|s| !s.is_template)
                            .map(|s| PrefixItem::Section(s.id)),
                    ),
                }
            }
            SystemPromptItem::SectionTree(t) => prefix.extend(
                t.default_present_ids
                    .iter()
                    .copied()
                    .map(PrefixItem::Section),
            ),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{
        sections_prefix, Builder, RangeFill, ReservedRange, SectionId, SelectionRule,
    };

    const YAML: &str = r#"
system_prompt:
  sections:
    - id: alpha
      content: "alpha"
    - id: beta
      content: "beta"
layers:
  - name: dialogue
    window: 4000
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
    groups:
      - id: convo
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 4
"#;

    fn ids(builder: &Builder, names: &[&str]) -> Vec<PrefixItem> {
        let ids: Vec<SectionId> = names
            .iter()
            .map(|n| builder.id_for_system_section(n).unwrap())
            .collect();
        sections_prefix(&ids)
    }

    #[test]
    fn a_collection_after_bare_sections_sees_them_all() {
        let mut b = Builder::from_yaml(YAML).unwrap();
        let mail = b
            .add_collection("mail", SelectionRule::AlwaysVisible, 0.0)
            .unwrap();
        b.add_section("tail", "tail", 50.0).unwrap();
        let prefix = prefix_before_collection(&b.schema().system_prompt.items, mail, None).unwrap();
        assert_eq!(prefix, ids(&b, &["alpha", "beta"]));
    }

    /// **An earlier collection with a range is a hole of its size**, whatever
    /// members it holds, sized for the engine's native context.
    #[test]
    fn an_earlier_range_is_a_hole_of_its_size() {
        let mut b = Builder::from_yaml(YAML).unwrap();
        let tools = b
            .add_collection("tools", SelectionRule::AlwaysVisible, 0.0)
            .unwrap();
        b.add_section_to_collection(tools, "t1", "one", 50.0)
            .unwrap();
        let mail = b
            .add_collection("mail", SelectionRule::AlwaysVisible, 0.0)
            .unwrap();
        let mut schema = b.schema().clone();
        for item in &mut schema.system_prompt.items {
            if let SystemPromptItem::Collection(c) = item {
                if c.id == tools {
                    c.range = Some(ReservedRange {
                        tokens: 8_192,
                        small_context: 2_048,
                        fill: RangeFill::Top,
                    });
                }
            }
        }
        let mut want = ids(&b, &["alpha", "beta"]);
        want.push(PrefixItem::Hole(8_192));
        assert_eq!(
            prefix_before_collection(&schema.system_prompt.items, mail, Some(262_144)).unwrap(),
            want
        );
        want.pop();
        want.push(PrefixItem::Hole(2_048));
        assert_eq!(
            prefix_before_collection(&schema.system_prompt.items, mail, Some(32_768)).unwrap(),
            want
        );
    }

    #[test]
    fn an_earlier_collections_members_join_the_prefix() {
        let mut b = Builder::from_yaml(YAML).unwrap();
        let tools = b
            .add_collection("tools", SelectionRule::AlwaysVisible, 0.0)
            .unwrap();
        b.add_section_to_collection(tools, "t1", "one", 50.0)
            .unwrap();
        b.add_section_to_collection(tools, "t2", "two", 50.0)
            .unwrap();
        let mail = b
            .add_collection("mail", SelectionRule::AlwaysVisible, 0.0)
            .unwrap();
        let prefix = prefix_before_collection(&b.schema().system_prompt.items, mail, None).unwrap();
        assert_eq!(prefix, ids(&b, &["alpha", "beta", "t1", "t2"]));
        let first = prefix_before_collection(&b.schema().system_prompt.items, tools, None).unwrap();
        assert_eq!(first, ids(&b, &["alpha", "beta"]));
    }

    #[test]
    fn a_collection_the_schema_lacks_has_no_prefix() {
        let b = Builder::from_yaml(YAML).unwrap();
        assert_eq!(
            prefix_before_collection(
                &b.schema().system_prompt.items,
                CollectionId::new(9999),
                None
            ),
            None
        );
    }
}
