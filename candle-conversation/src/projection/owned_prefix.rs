//! The prefix a collection's members seal against.
//!
//! When a conversation is created, the system prompt is walked in order and a
//! collection's members are sealed against the sections laid down before it. A
//! member submitted later has to seal against that same prefix, so this walk is
//! repeated here from the schema alone.

use super::ids::{CollectionId, SectionId};
use super::schema::SystemPromptItem;

/// The section ids in front of the top-level collection `collection`, in the
/// order the creation walk accumulates them: bare sections, the non-template
/// members of earlier collections, and the default branch of earlier section
/// trees.
///
/// `None` when the schema has no top-level collection by that id. A collection
/// embedded in a section tree seals once per branch, which a single submitted
/// section does not do.
pub fn prefix_before_collection(
    items: &[SystemPromptItem],
    collection: CollectionId,
) -> Option<Vec<SectionId>> {
    let mut prefix = Vec::new();
    for item in items {
        match item {
            SystemPromptItem::Section(s) => prefix.push(s.id),
            SystemPromptItem::Collection(c) => {
                if c.id == collection {
                    return Some(prefix);
                }
                prefix.extend(c.sections.iter().filter(|s| !s.is_template).map(|s| s.id));
            }
            SystemPromptItem::SectionTree(t) => {
                prefix.extend(t.default_present_ids.iter().copied())
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{Builder, SelectionRule};

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

    fn ids(builder: &Builder, names: &[&str]) -> Vec<SectionId> {
        names
            .iter()
            .map(|n| builder.id_for_system_section(n).unwrap())
            .collect()
    }

    #[test]
    fn a_collection_after_bare_sections_sees_them_all() {
        let mut b = Builder::from_yaml(YAML).unwrap();
        let mail = b
            .add_collection("mail", SelectionRule::AlwaysVisible, 0.0)
            .unwrap();
        b.add_section("tail", "tail", 50.0).unwrap();
        let prefix = prefix_before_collection(&b.schema().system_prompt.items, mail).unwrap();
        assert_eq!(prefix, ids(&b, &["alpha", "beta"]));
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
        let prefix = prefix_before_collection(&b.schema().system_prompt.items, mail).unwrap();
        assert_eq!(prefix, ids(&b, &["alpha", "beta", "t1", "t2"]));
        let first = prefix_before_collection(&b.schema().system_prompt.items, tools).unwrap();
        assert_eq!(first, ids(&b, &["alpha", "beta"]));
    }

    #[test]
    fn a_collection_the_schema_lacks_has_no_prefix() {
        let b = Builder::from_yaml(YAML).unwrap();
        assert_eq!(
            prefix_before_collection(&b.schema().system_prompt.items, CollectionId::new(9999)),
            None
        );
    }
}
