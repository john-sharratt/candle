//! Every item of the system prompt takes a fixed slot
//! (`docs/placed_section_layout.md` §3.1c): whatever the dials, the tools mode,
//! the selection and the journal hold, the section after the knowledge base's
//! range stands at the same position — and at the one the ingest seals its
//! members behind.

use candle_transformers::models::dialect::Dialect;

use super::*;
use crate::projection::prompt_tokens::segment_tokens;
use crate::projection::{prefix_before_collection, PrefixItem, ProjectionSegment, SealedKind};

/// A prompt with every kind of variable item ahead of a ranged knowledge
/// base: a selector tree, a journal-gated intro and its range, the two
/// configured/unconfigured grounding variants, a configured-gated overview,
/// the `<tools>` markers gated on a selection, the tools range and its
/// summary's range. `tools` is the tools collection's members.
fn yaml(journal: &str, tools: &str) -> String {
    format!(
        r#"
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
    - kind: section
      id: journal_intro
      content: "j"
      depends_on: journal
    - kind: collection
      name: journal
      selection: {{ kind: always_visible }}
      range: {{ tokens: 50, fill: bottom }}
      sections: {journal}
    - kind: section
      id: grounding
      content: "g"
      depends_on_configured: tools
    - kind: section
      id: grounding_no_tools
      content: "n"
      depends_on_unconfigured: tools
    - kind: section
      id: overview
      content: "o"
      depends_on_configured: tools
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
    - kind: collection
      name: tools
      selection: {{ kind: top_k, k: 1 }}
      range: {{ tokens: 100 }}
      summary_range: {{ tokens: 30 }}
      sections: {tools}
    - kind: template
      id: tools_close
      dialect: tool_block_close
      depends_on: tools
    - kind: section
      id: files_open
      content: "f"
    - kind: collection
      name: files
      selection: {{ kind: always_visible }}
      range: {{ tokens: 200 }}
      sections: []
    - kind: section
      id: close
      content: "c"
layers:
  - name: dialogue
    window: 4000
    summary:
      turns:
        max_tokens: 256
        user: {{ system_prompt: c, user_prompt: c }}
        assistant: {{ system_prompt: c, user_prompt: c }}
    score_formula: max
    budget: {{ priority: 100 }}
    groups:
      - id: convo
        selection: {{ kind: always_visible }}
"#
    )
}

/// `yaml` built under ChatML, whose catalog has the `<tools>` markers.
fn build(yaml: &str) -> Result<Builder, ConstructionError> {
    Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&Dialect::chat_ml()))
}

const TWO_TOOLS: &str = "[{ id: tool_a, content: \"A\" }, { id: tool_b, content: \"B\" }]";
const ONE_ENTRY: &str = "[{ id: entry, content: \"E\" }]";

/// Every section's tokens, by name: the tree's `short` 10 and `long` 40.
const TOKENS: &[(&str, usize)] = &[
    ("journal_intro", 25),
    ("entry", 6),
    ("grounding", 45),
    ("grounding_no_tools", 5),
    ("overview", 44),
    ("tools_open", 4),
    ("tool_a", 20),
    ("tool_b", 20),
    ("tools_close", 4),
    ("files_open", 3),
    ("close", 2),
];

/// The tokens of each variant of the tree, keyed on its option.
fn tree_tokens(b: &Builder) -> Vec<(SectionId, usize)> {
    let tree = b.schema().system_prompt.section_trees().next().unwrap();
    tree.nodes
        .iter()
        .flat_map(|n| n.options.iter())
        .flat_map(|o| {
            let t = if o.id == "short" { 10 } else { 40 };
            o.variants.iter().map(move |v| (v.id, t))
        })
        .collect()
}

/// `b`'s resolver: every section's tokens, a 12-token catalog summary, and
/// `tool_a` scored so a top-1 of two selects it.
fn resolver(b: &mut Builder, summarised: bool) -> MockResolver {
    let mut r = MockResolver::new();
    for &(name, t) in TOKENS {
        if let Some(id) = b.id_for_system_section(name) {
            r = r.with_section_tokens(id, t);
        }
    }
    for (id, t) in tree_tokens(b) {
        r = r.with_section_tokens(id, t);
    }
    if let Some(a) = b.id_for_system_section("tool_a") {
        r = r.with_section_score(a, 0.9);
    }
    if summarised {
        if let Some(tools) = b.id_for_system_collection("tools") {
            let summary = SectionId::reserved(Reserved::ToolSummary);
            b.set_collection_summary_section(tools, summary).unwrap();
            r = r.with_section_tokens(summary, 12);
        }
    }
    r
}

/// The position `close` is read at under `selection`.
fn close_at(b: &Builder, r: &MockResolver, selection: &SelectionState) -> usize {
    let target = ProjectionTarget {
        layer: b.id_for_layer("dialogue").unwrap(),
        group: b.id_for_group("convo").unwrap(),
        timeline: TimelineId::for_test(1),
    };
    let segments = b
        .project_with_selection(target, r, ProjectionMode::Decode, selection)
        .segments;
    let close = b.id_for_system_section("close").unwrap();
    let at = segments
        .iter()
        .position(
            |s| matches!(s, ProjectionSegment::Sealed(SealedKind::Section(x)) if x.id == close),
        )
        .expect("close is projected");
    segment_tokens(&segments[..at], r)
}

/// The tree's slot (40), the journal intro and range (25 + 50), both grounding
/// variants (45 + 5), the overview (44), the summary's range (30), the tool
/// markers and range (4 + 100 + 4), the files heading and range (3 + 200).
const CLOSE_AT: usize = 40 + 25 + 50 + 45 + 5 + 44 + 30 + 4 + 100 + 4 + 3 + 200;

/// **The section after the knowledge base stands at one position on every
/// projection**: tools selected with the summary, nothing scored, the tree's
/// longer option, a journal entry, and the tools switched off.
#[test]
fn what_follows_the_knowledge_base_stands_still() {
    let mut b = build(&yaml("[]", TWO_TOOLS)).unwrap();
    let r = resolver(&mut b, true);
    let default = SelectionState::new();
    assert_eq!(
        close_at(&b, &r, &default),
        CLOSE_AT,
        "a tool selected, summarised"
    );

    let mut long = SelectionState::new();
    long.select("length", "long");
    assert_eq!(
        close_at(&b, &r, &long),
        CLOSE_AT,
        "the tree's longer option"
    );

    let mut bare = build(&yaml("[]", TWO_TOOLS)).unwrap();
    let mut unscored = resolver(&mut bare, false);
    if let Some(a) = bare.id_for_system_section("tool_a") {
        unscored = unscored.with_section_score(a, 0.0);
    }
    assert_eq!(close_at(&bare, &unscored, &default), CLOSE_AT, "no summary");

    let mut journal = build(&yaml(ONE_ENTRY, TWO_TOOLS)).unwrap();
    let rj = resolver(&mut journal, true);
    assert_eq!(
        close_at(&journal, &rj, &default),
        CLOSE_AT,
        "a journal entry"
    );

    let mut off = build(&yaml("[]", "[]")).unwrap();
    let ro = resolver(&mut off, false);
    assert_eq!(close_at(&off, &ro, &default), CLOSE_AT, "tools off");
}

/// **The ingest seals behind the same positions**: the prefix before the
/// knowledge base, plus its range, is where every projection reads `close`.
#[test]
fn the_ingest_prefix_takes_the_same_positions() {
    let mut b = build(&yaml("[]", TWO_TOOLS)).unwrap();
    let r = resolver(&mut b, true);
    let files = b.id_for_system_collection("files").unwrap();
    let prefix = prefix_before_collection(&b.schema().system_prompt.items, files, None, |id| {
        r.section_token_count(id)
    })
    .unwrap();
    let positions: usize = prefix
        .iter()
        .map(|p| match p {
            PrefixItem::Section(id) => r.section_token_count(*id),
            PrefixItem::Hole(n) => *n,
        })
        .sum();
    assert_eq!(positions + 200, CLOSE_AT);
}
