//! The positions each item of the system prompt always takes
//! (`docs/placed_section_layout.md` §3.1c).
//!
//! An item takes the positions of its largest form, and a dynamic one at least
//! its declared `min_tokens` — the margin that lets its text change without
//! moving anything after it: what it emits and a hole of the rest. A ranged
//! collection's slot is its range; this file holds the slots that are
//! computed — a section's, each tree node's and so each tree's, and where a
//! collection's catalog summary slot stands — and the slotted prefix inside a
//! tree, so every walk of the prompt lays them down alike.

use std::collections::HashMap;

use super::*;
use crate::projection::PrefixItem;

/// The positions section `s` always takes: its tokens, or its `min_tokens`
/// when that is more. A gated section that does not emit is a hole of it.
pub fn section_slot(s: &SectionSchema, tokens: usize) -> usize {
    tokens.max(s.min_tokens)
}

/// What `node`'s longest option takes, `tokens` giving a sealed variant's
/// length and a glue node its marker's: the most the node can emit.
pub fn node_options_tokens(node: &TreeNode, tokens: impl Fn(SectionId) -> usize) -> usize {
    if let Some(glue) = &node.glue {
        return glue.tokens.as_ref().map_or(0, |t| t.len());
    }
    node.options
        .iter()
        .flat_map(|o| o.variants.iter())
        .map(|v| tokens(v.id))
        .max()
        .unwrap_or(0)
}

/// The positions tree node `node` always takes: its longest option, or its
/// `min_tokens` when that is more. A prefix-transparent embedded collection
/// takes none of its own: what it emits moves what follows it.
pub fn node_slot(node: &TreeNode, tokens: impl Fn(SectionId) -> usize) -> usize {
    if node.collection.is_some() {
        return 0;
    }
    node_options_tokens(node, tokens).max(node.min_tokens)
}

/// The positions `tree` always takes: the sum of its nodes' slots.
pub fn tree_slot(tree: &SectionTree, tokens: impl Fn(SectionId) -> usize) -> usize {
    tree.nodes.iter().map(|n| node_slot(n, &tokens)).sum()
}

/// The per-dim selection a packed ancestor `key` over `tree`'s first
/// `dim_count` dims names, every later dim at its default.
pub fn unpack_selection(tree: &SectionTree, key: u32, dim_count: usize) -> Vec<u8> {
    let mut rest = key;
    tree.dims
        .iter()
        .enumerate()
        .map(|(d, dim)| {
            if d < dim_count {
                let radix = dim.option_count.max(1) as u32;
                let v = (rest % radix) as u8;
                rest /= radix;
                v
            } else {
                dim.default_option
            }
        })
        .collect()
}

/// `tree`'s nodes before `upto` under `selection`, each in its slot: the
/// node's emitted variant, when it has one in that branch, and a hole of the
/// rest of the slot. A glue node is live-prefilled rather than sealed, so it
/// is a hole of its slot; an embedded collection takes nothing. What a variant
/// at node `upto` seals behind, inside the tree, and what the whole tree lays
/// down with `upto` its length.
pub fn tree_prefix(
    tree: &SectionTree,
    upto: usize,
    selection: &[u8],
    tokens: impl Fn(SectionId) -> usize,
) -> Vec<PrefixItem> {
    let mut out = Vec::new();
    for node in &tree.nodes[..upto] {
        if node.collection.is_some() {
            continue;
        }
        let slot = node_slot(node, &tokens);
        let emitted = if node.glue.is_some() {
            None
        } else {
            node.options
                .get(node.chosen(selection))
                .and_then(|o| o.variant_for(tree.pack(selection, node.ancestor_dims)))
                .map(|v| v.id)
        };
        let used = emitted.map_or(0, &tokens);
        if let Some(id) = emitted {
            out.push(PrefixItem::Section(id));
        }
        if slot > used {
            out.push(PrefixItem::Hole(slot - used));
        }
    }
    out
}

/// Where each collection's catalog summary slot stands: the index of the item
/// it goes in front of, mapped to the collections whose slots go there. That
/// is the first template gated on the collection (`depends_on`) declared ahead
/// of it — the `<tools>` the summary is read just outside — or the collection
/// itself when none is. Only collections declaring a `summary_range` have one.
pub fn summary_points(items: &[SystemPromptItem]) -> HashMap<usize, Vec<CollectionId>> {
    let mut out: HashMap<usize, Vec<CollectionId>> = HashMap::new();
    for (at, item) in items.iter().enumerate() {
        let SystemPromptItem::Collection(c) = item else {
            continue;
        };
        if c.summary_range.is_none() {
            continue;
        }
        let template = items[..at].iter().position(|it| {
            matches!(it, SystemPromptItem::Section(s) if s.is_template && s.depends_on == Some(c.id))
        });
        out.entry(template.unwrap_or(at)).or_default().push(c.id);
    }
    out
}

/// The collection `id` among `items`.
pub fn collection_of(items: &[SystemPromptItem], id: CollectionId) -> Option<&SectionCollection> {
    items.iter().find_map(|it| match it {
        SystemPromptItem::Collection(c) if c.id == id => Some(c),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use candle_transformers::models::dialect::Dialect;

    use super::*;
    use crate::projection::Builder;

    const YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: selector
          id: length
          default: short
          options:
            - id: short
              content: "s"
            - id: long
              content: "l"
        - kind: optional
          id: extra
          content: "x"
          default: absent
    - kind: section
      id: overview
      content: "o"
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
    - kind: collection
      name: tools
      selection: { kind: top_k, k: 2 }
      range: { tokens: 100 }
      summary_range: { tokens: 50 }
      sections:
        - id: a
          content: "A"
    - kind: collection
      name: files
      selection: { kind: always_visible }
      summary_range: { tokens: 20 }
      sections: []
layers:
  - name: dialogue
    window: 4000
    summary:
      turns:
        max_tokens: 256
        user: { system_prompt: c, user_prompt: c }
        assistant: { system_prompt: c, user_prompt: c }
    score_formula: max
    budget: { priority: 100 }
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

    fn builder() -> Builder {
        from_yaml(YAML)
    }

    /// `yaml` built under ChatML, whose catalog has the `<tools>` markers.
    fn from_yaml(yaml: &str) -> Builder {
        Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&Dialect::chat_ml())).unwrap()
    }

    fn tree(b: &Builder) -> &SectionTree {
        b.schema().system_prompt.section_trees().next().unwrap()
    }

    /// Each variant's length, keyed on which option it belongs to: `short`
    /// 10, `long` 40, `extra` 7.
    fn lengths(b: &Builder) -> impl Fn(SectionId) -> usize + '_ {
        move |id| {
            let t = tree(b);
            for node in &t.nodes {
                for o in &node.options {
                    if o.variants.iter().any(|v| v.id == id) {
                        return match o.id.as_str() {
                            "short" => 10,
                            "long" => 40,
                            _ => 7,
                        };
                    }
                }
            }
            0
        }
    }

    /// **A tree takes each node's longest option**: the selector's `long`
    /// (40) and the optional's present content (7), whatever is selected.
    #[test]
    fn a_tree_takes_each_nodes_longest_option() {
        let b = builder();
        assert_eq!(tree_slot(tree(&b), lengths(&b)), 47);
    }

    /// **A node's minimum is its slot until its options outgrow it**: the
    /// selector's 100 over options of 40, its 40 over a minimum of 20; the
    /// tree's slot is its nodes' together.
    #[test]
    fn a_nodes_minimum_holds_until_its_options_outgrow_it() {
        let b = builder();
        let mut t = tree(&b).clone();
        t.nodes[0].min_tokens = 100;
        assert_eq!(node_slot(&t.nodes[0], lengths(&b)), 100);
        assert_eq!(node_options_tokens(&t.nodes[0], lengths(&b)), 40);
        assert_eq!(tree_slot(&t, lengths(&b)), 107);
        t.nodes[0].min_tokens = 20;
        assert_eq!(node_slot(&t.nodes[0], lengths(&b)), 40);
    }

    /// **A schema's `min_tokens` reaches its node and its section.**
    #[test]
    fn min_tokens_is_read_from_the_schema() {
        let yaml = YAML
            .replacen(
                "          id: extra\n",
                "          id: extra\n          min_tokens: 64\n",
                1,
            )
            .replacen(
                "      id: overview\n",
                "      id: overview\n      min_tokens: 32\n",
                1,
            );
        let b = from_yaml(&yaml);
        assert_eq!(tree(&b).nodes[1].min_tokens, 64);
        assert_eq!(tree_slot(tree(&b), lengths(&b)), 40 + 64);
        let overview = b
            .schema()
            .system_prompt
            .all_sections()
            .find(|s| s.name == "overview")
            .unwrap();
        assert_eq!(section_slot(overview, 3), 32);
        assert_eq!(section_slot(overview, 50), 50);
    }

    /// **The tree's default branch lies in its slots**: `short` (10) and a hole
    /// of 30 to the selector's slot, then the absent optional's slot (7) a
    /// hole; and inside the tree, the optional's variants seal behind the
    /// selector's whole slot whichever option it holds.
    #[test]
    fn the_default_branch_lies_in_its_slots() {
        let b = builder();
        let t = tree(&b);
        let short = t.nodes[0].options[0].variants[0].id;
        assert_eq!(
            tree_prefix(t, t.nodes.len(), &t.default_selection, lengths(&b)),
            vec![
                PrefixItem::Section(short),
                PrefixItem::Hole(30),
                PrefixItem::Hole(7)
            ]
        );
        let long = t.nodes[0].options[1].variants[0].id;
        assert_eq!(
            tree_prefix(t, 1, &[1, 1], lengths(&b)),
            vec![PrefixItem::Section(long)]
        );
        let positions = |p: Vec<PrefixItem>| -> usize {
            p.iter()
                .map(|i| match i {
                    PrefixItem::Section(id) => lengths(&b)(*id),
                    PrefixItem::Hole(n) => *n,
                })
                .sum()
        };
        assert_eq!(positions(tree_prefix(t, 1, &[0, 1], lengths(&b))), 40);
        assert_eq!(positions(tree_prefix(t, 1, &[1, 1], lengths(&b))), 40);
    }

    /// **A packed ancestor key unpacks to its selection**, later dims at their
    /// defaults: key 1 over the first dim is `long`.
    #[test]
    fn a_key_unpacks_to_its_selection() {
        let b = builder();
        let t = tree(&b);
        assert_eq!(unpack_selection(t, 1, 1), vec![1, 1]);
        assert_eq!(unpack_selection(t, 0, 0), t.default_selection);
        assert_eq!(unpack_selection(t, 3, 2), vec![1, 1]);
    }

    /// **A summary slot stands before the first template gated on its
    /// collection, or before the collection when none is.**
    #[test]
    fn a_summary_slot_stands_before_its_template() {
        let b = builder();
        let items = &b.schema().system_prompt.items;
        let id = |name: &str| match items
            .iter()
            .find(|it| matches!(it, SystemPromptItem::Collection(c) if c.name == name))
        {
            Some(SystemPromptItem::Collection(c)) => c.id,
            _ => unreachable!(),
        };
        let points = summary_points(items);
        assert_eq!(points.get(&2), Some(&vec![id("tools")]), "{points:?}");
        assert_eq!(points.get(&4), Some(&vec![id("files")]), "{points:?}");
        assert_eq!(points.len(), 2);
    }
}
