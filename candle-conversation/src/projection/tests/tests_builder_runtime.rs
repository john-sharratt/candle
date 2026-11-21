//! Tests for building a system prompt from YAML and changing it at runtime.
//!
//! Sections and collections are emitted in declaration order, and a section added
//! at runtime takes an id that no compression prompt already holds. Those hidden
//! ids must be skipped, because a reused one would let a tool's content overwrite
//! the compression prompt. Names stay unique within a layer, and a member added
//! with its own id projects from its own builder alone.

use super::*;

// —— Section + collection emission tests ——————————————————————————————————————
//
// These tests cover the `SystemPromptItem` model: the layer's
// `system_prompt` is an ordered list of items, each either a single
// always-emit section or a `SectionCollection` with its own selection
// rule (TopK / Single / AlwaysVisible).  Emission preserves declaration
// order; selection picks by salience (provenance-derived score).

pub(super) const SECTIONS_YAML_FLAT: &str = r#"
system_prompt:
  sections:
    - id: alpha
      content: "alpha"
    - id: beta
      content: "beta"
    - id: gamma
      content: "gamma"
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

#[test]
pub(super) fn flat_sections_yaml_emits_all_in_declaration_order() {
    // Legacy `sections:` shortcut — every section is always-emit.
    let b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let alpha = b.id_for_system_section("alpha").unwrap();
    let beta = b.id_for_system_section("beta").unwrap();
    let gamma = b.id_for_system_section("gamma").unwrap();

    let resolver = MockResolver::new();
    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    assert_eq!(p.sealed_sections().count(), 3);
    assert_eq!(p.sealed_sections().next().unwrap().id, alpha);
    assert_eq!(p.sealed_sections().nth(1).unwrap().id, beta);
    assert_eq!(p.sealed_sections().nth(2).unwrap().id, gamma);
}

// —— Builder runtime mutation —————————————————————————————————————————————————

#[test]
pub(super) fn add_section_appends_at_end_with_unique_id() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let n_before = b.schema().system_prompt.items.len();
    let new_id = b.add_section("newsec", "new content", 75.0).unwrap();
    assert_eq!(b.schema().system_prompt.items.len(), n_before + 1);
    assert_eq!(b.id_for_system_section("newsec"), Some(new_id));
}

#[test]
pub(super) fn add_section_duplicate_name_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let result = b.add_section("alpha", "dup", 50.0);
    assert!(matches!(
        result,
        Err(super::super::error::ConstructionError::DuplicateSectionName(ref n)) if n == "alpha"
    ));
}

#[test]
pub(super) fn add_collection_appends_and_returns_id() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let cid = b
        .add_collection(
            "tools",
            super::super::schema::SelectionRule::TopK { k: 3 },
            0.0,
        )
        .unwrap();
    assert_eq!(b.id_for_system_collection("tools"), Some(cid));
}

#[test]
pub(super) fn runtime_section_ids_stay_disjoint_from_compression_prompts() {
    // Regression: a layer's compression-prompt sections (and collection
    // summaries) live in `layer.summary` / the collection summary, NOT in
    // `system_prompt.items`, and for a trailing layer get the highest section
    // ids. Runtime allocation (`add_section_to_collection`, the daemon's tool
    // catalog) must skip past them — otherwise a tool section reuses a
    // compression-prompt id and `ensure_summary_section` injects the tool's
    // content as the compression prompt, collapsing the compression decode into
    // a degenerate loop (the live-daemon bug this guards).
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let cid = b.id_for_system_collection("tools").unwrap();

    // The hidden (compression-prompt + collection-summary) section ids — those
    // in `all_section_ids` but not visible in `system_prompt.items`.
    let hidden: Vec<u32> = {
        let layer = b.schema().layers.iter().find(|l| l.id == dialogue).unwrap();
        let visible: std::collections::HashSet<u32> = b
            .schema()
            .system_prompt
            .all_sections()
            .map(|s| s.id.raw())
            .collect();
        layer
            .all_section_ids()
            .into_iter()
            .filter(|id| !visible.contains(id))
            .collect()
    };
    assert!(
        !hidden.is_empty(),
        "layer must own hidden compression-prompt / collection-summary sections"
    );

    for t in 0..6 {
        let id = b
            .add_section_to_collection(cid, format!("rt_{t}"), "x", 50.0)
            .unwrap();
        assert!(
            !hidden.contains(&id.raw()),
            "runtime section id {} reuses a hidden compression-prompt id {:?}",
            id.raw(),
            hidden,
        );
    }
}

#[test]
pub(super) fn add_section_to_collection_appends_in_collection() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let cid = b
        .add_collection(
            "tools",
            super::super::schema::SelectionRule::TopK { k: 2 },
            0.0,
        )
        .unwrap();
    let t1 = b
        .add_section_to_collection(cid, "t1", "tool1", 50.0)
        .unwrap();
    let t2 = b
        .add_section_to_collection(cid, "t2", "tool2", 50.0)
        .unwrap();
    let t3 = b
        .add_section_to_collection(cid, "t3", "tool3", 50.0)
        .unwrap();
    // Section names are layer-scoped: t1/t2/t3 are unique even though
    // they're nested in a collection.
    assert_eq!(b.id_for_system_section("t1"), Some(t1));
    assert_eq!(b.id_for_system_section("t2"), Some(t2));
    assert_eq!(b.id_for_system_section("t3"), Some(t3));

    // Score them and verify projection top-k=2 picks the right two.
    let resolver = MockResolver::new()
        .with_section_score(t1, 0.1)
        .with_section_score(t2, 0.9)
        .with_section_score(t3, 0.5);

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    // Static `alpha`/`beta`/`gamma` from SECTIONS_YAML_FLAT plus the
    // top-2 tools (t2, t3) from the collection.
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert!(ids.contains(&t2), "t2 (score 0.9) must survive top-2");
    assert!(ids.contains(&t3), "t3 (score 0.5) must survive top-2");
    assert!(!ids.contains(&t1), "t1 (score 0.1) must be filtered");
}

#[test]
pub(super) fn add_section_to_unknown_collection_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    // Construct a CollectionId that no add_collection call has issued.
    let bogus = super::super::ids::CollectionId::new(9999);
    let r = b.add_section_to_collection(bogus, "x", "y", 50.0);
    assert!(matches!(
        r,
        Err(super::super::error::ConstructionError::UnknownCollection(_))
    ));
}

#[test]
pub(super) fn a_member_added_with_its_own_id_projects_from_that_builder_only() {
    let mut base = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = base.id_for_layer("dialogue").unwrap();
    let convo = base.id_for_group("convo").unwrap();
    let cid = base
        .add_collection(
            "mail",
            super::super::schema::SelectionRule::AlwaysVisible,
            0.0,
        )
        .unwrap();
    let mut mine = base.clone();
    let own = SectionId::new(0x8000_0001);
    mine.add_section_to_collection_with_id(cid, own, "inbox", "three new messages", 50.0)
        .unwrap();

    assert_eq!(mine.id_for_system_section("inbox"), Some(own));
    assert_eq!(base.id_for_system_section("inbox"), None);

    let target = ProjectionTarget {
        layer: dialogue,
        group: convo,
        timeline: TimelineId::for_test(1),
    };
    let resolver = MockResolver::new();
    let in_mine: Vec<SectionId> = mine
        .project(target, &resolver)
        .sealed_sections()
        .map(|s| s.id)
        .collect();
    let in_base: Vec<SectionId> = base
        .project(target, &resolver)
        .sealed_sections()
        .map(|s| s.id)
        .collect();
    assert!(in_mine.contains(&own));
    assert!(!in_base.contains(&own));
}

#[test]
pub(super) fn adding_a_member_with_an_id_rejects_a_taken_name_and_a_bad_collection() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let cid = b
        .add_collection(
            "mail",
            super::super::schema::SelectionRule::AlwaysVisible,
            0.0,
        )
        .unwrap();
    assert!(matches!(
        b.add_section_to_collection_with_id(cid, SectionId::new(0x8000_0001), "alpha", "x", 50.0),
        Err(super::super::error::ConstructionError::DuplicateSectionName(_))
    ));
    assert!(matches!(
        b.add_section_to_collection_with_id(
            super::super::ids::CollectionId::new(9999),
            SectionId::new(0x8000_0001),
            "x",
            "y",
            50.0
        ),
        Err(super::super::error::ConstructionError::UnknownCollection(_))
    ));
    assert!(matches!(
        b.add_section_to_collection_with_id(cid, SectionId::new(0x8000_0001), "x", "y", 0.0),
        Err(super::super::error::ConstructionError::InvalidPriority { .. })
    ));
}

#[test]
pub(super) fn duplicate_collection_name_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    b.add_collection(
        "tools",
        super::super::schema::SelectionRule::AlwaysVisible,
        0.0,
    )
    .unwrap();
    let r = b.add_collection(
        "tools",
        super::super::schema::SelectionRule::AlwaysVisible,
        0.0,
    );
    assert!(matches!(
        r,
        Err(super::super::error::ConstructionError::DuplicateCollectionName(_))
    ));
}

#[test]
pub(super) fn add_section_invalid_priority_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let result = b.add_section("newsec", "content", 0.0);
    assert!(matches!(
        result,
        Err(super::super::error::ConstructionError::InvalidPriority { .. })
    ));
}
