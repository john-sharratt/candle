//! The prefix a collection's members, or its catalog summary, seal against.
//!
//! When a conversation is created, the system prompt is walked in order and a
//! collection's members are sealed against the sections laid down before it. A
//! member submitted later has to seal against that same prefix, and so does a
//! catalog summary sealed at startup, so this walk is repeated here from the
//! schema alone — every item in its slot (`docs/placed_section_layout.md`
//! §3.1c), as the creation walk and every projection lay it down.

use super::ids::{CollectionId, SectionId};
use super::prefix_item::PrefixItem;
use super::schema::SystemPromptItem;
use super::schema::{collection_of, section_slot, summary_points, tree_prefix};

/// What is in front of the top-level collection `collection`, in the order the
/// creation walk accumulates it: bare sections, the non-template members of
/// earlier collections — or, for an earlier collection with a reserved range,
/// the range as a hole of its size on an engine trained to `native` positions
/// — the default branch of earlier section trees, every node in its slot, each
/// section padded to its slot, and every catalog summary's range as a hole
/// where it stands, the collection's own included. `tokens` is a sealed
/// section's length.
///
/// `None` when the schema has no top-level collection by that id. A collection
/// embedded in a section tree seals once per branch, which a single submitted
/// section does not do.
pub fn prefix_before_collection(
    items: &[SystemPromptItem],
    collection: CollectionId,
    native: Option<usize>,
    tokens: impl Fn(SectionId) -> usize,
) -> Option<Vec<PrefixItem>> {
    let at = items
        .iter()
        .position(|it| matches!(it, SystemPromptItem::Collection(c) if c.id == collection))?;
    Some(prefix_to(items, at, None, native, &tokens))
}

/// What is in front of the top-level collection `collection`'s catalog
/// summary slot: the prompt as [`prefix_before_collection`] lays it down, up
/// to where the slot stands — just before the first template gated on the
/// collection. `None` when the collection declares no `summary_range`.
pub fn prefix_before_summary(
    items: &[SystemPromptItem],
    collection: CollectionId,
    native: Option<usize>,
    tokens: impl Fn(SectionId) -> usize,
) -> Option<Vec<PrefixItem>> {
    let at = summary_points(items)
        .into_iter()
        .find_map(|(at, cids)| cids.contains(&collection).then_some(at))?;
    Some(prefix_to(items, at, Some(collection), native, &tokens))
}

/// `items[..end]` laid down in their slots, then the summary slots standing at
/// `end` — all of them, or with `summary` only those ahead of its own.
fn prefix_to(
    items: &[SystemPromptItem],
    end: usize,
    summary: Option<CollectionId>,
    native: Option<usize>,
    tokens: &impl Fn(SectionId) -> usize,
) -> Vec<PrefixItem> {
    let summary_slots = summary_points(items);
    let summary_hole = |cid: CollectionId| {
        collection_of(items, cid)
            .and_then(|c| c.summary_range)
            .map(|range| PrefixItem::Hole(range.positions(native)))
    };
    let mut prefix = Vec::new();
    for (at, item) in items.iter().enumerate().take(end + 1) {
        for &cid in summary_slots.get(&at).into_iter().flatten() {
            if at == end && Some(cid) == summary {
                return prefix;
            }
            prefix.extend(summary_hole(cid));
        }
        if at == end {
            break;
        }
        match item {
            SystemPromptItem::Section(s) => {
                prefix.push(PrefixItem::Section(s.id));
                let t = tokens(s.id);
                let pad = section_slot(s, t) - t;
                if pad > 0 {
                    prefix.push(PrefixItem::Hole(pad));
                }
            }
            SystemPromptItem::Collection(c) => match c.range {
                Some(range) => prefix.push(PrefixItem::Hole(range.positions(native))),
                None => prefix.extend(
                    c.sections
                        .iter()
                        .filter(|s| !s.is_template)
                        .map(|s| PrefixItem::Section(s.id)),
                ),
            },
            SystemPromptItem::SectionTree(t) => {
                prefix.extend(tree_prefix(t, t.nodes.len(), &t.default_selection, tokens));
            }
        }
    }
    prefix
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
        let prefix =
            prefix_before_collection(&b.schema().system_prompt.items, mail, None, |_| 0).unwrap();
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
            prefix_before_collection(&schema.system_prompt.items, mail, Some(262_144), |_| 0)
                .unwrap(),
            want
        );
        want.pop();
        want.push(PrefixItem::Hole(2_048));
        assert_eq!(
            prefix_before_collection(&schema.system_prompt.items, mail, Some(32_768), |_| 0)
                .unwrap(),
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
        let prefix =
            prefix_before_collection(&b.schema().system_prompt.items, mail, None, |_| 0).unwrap();
        assert_eq!(prefix, ids(&b, &["alpha", "beta", "t1", "t2"]));
        let first =
            prefix_before_collection(&b.schema().system_prompt.items, tools, None, |_| 0).unwrap();
        assert_eq!(first, ids(&b, &["alpha", "beta"]));
    }

    #[test]
    fn a_collection_the_schema_lacks_has_no_prefix() {
        let b = Builder::from_yaml(YAML).unwrap();
        assert_eq!(
            prefix_before_collection(
                &b.schema().system_prompt.items,
                CollectionId::new(9999),
                None,
                |_| 0
            ),
            None
        );
    }
}
