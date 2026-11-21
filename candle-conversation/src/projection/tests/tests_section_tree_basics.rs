//! Tests for dialect-template parsing and for section trees: their dimensions,
//! branches and sealed ids.
//!
//! A dimension multiplies only the sections below it, so two branches share their
//! prefix exactly as far as their dimensions agree, and a prefix never includes a
//! collection's runtime members. A gated dimension is masked back to its default
//! when its group is absent. Every declared tree is returned, not only the first,
//! and a projection with no generated run has no hole for the engine to fill.

use super::*;

// ── Dialect-template parsing ─────────────────────────────────────────────────
//
// These tests cover the `kind: template` system-prompt item that references
// a [`DialectTemplate`] catalog entry by snake-case name.  The resolved
// string lands in a `SectionSchema` with `is_template = true`; the
// projection assembler routes those items through live prefill so their
// K/V stays attention-correct under the runtime left context.

pub(super) mod dialect_templates {
    use super::*;
    use crate::projection::builder::Builder;
    use crate::projection::error::ConstructionError;
    use crate::projection::project::{ProjectionSegment, SealedKind};
    use crate::projection::schema::SystemPromptItem;
    use candle_transformers::models::dialect::Dialect;

    const TEMPLATE_YAML: &str = r#"
system_prompt:
  items:
    - kind: template
      id: system_open
      dialect: system_start
    - kind: section
      id: frame
      content: "You are a senior engineer."
    - kind: template
      id: system_close
      dialect: system_end
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

    #[test]
    fn template_item_resolves_to_dialect_content() {
        let dlct = Dialect::chat_ml();
        let b = Builder::from_yaml_with_vars_and_dialect(TEMPLATE_YAML, &[], Some(&dlct)).unwrap();

        let mut found: Vec<(&str, &str, bool)> = Vec::new();
        for it in &b.schema().system_prompt.items {
            if let SystemPromptItem::Section(s) = it {
                found.push((s.name.as_str(), s.content.as_str(), s.is_template));
            }
        }
        assert_eq!(
            found,
            vec![
                ("system_open", "<|im_start|>system\n", true),
                ("frame", "You are a senior engineer.", false),
                ("system_close", "<|im_end|>\n", true),
            ],
        );
    }

    #[test]
    fn template_without_dialect_errors() {
        let err = Builder::from_yaml_with_vars_and_dialect(TEMPLATE_YAML, &[], None).unwrap_err();
        assert!(
            matches!(err, ConstructionError::DialectRequired { ref item } if item == "system_prompt/system_open"),
            "got {err:?}",
        );
    }

    #[test]
    fn template_unknown_dialect_name_errors() {
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: bogus
      dialect: not_a_real_template
    - kind: section
      id: frame
      content: "x"
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let err = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap_err();
        assert!(
            matches!(
                err,
                ConstructionError::UnknownDialectTemplate { ref item, ref name }
                if item == "system_prompt/bogus" && name == "not_a_real_template"
            ),
            "got {err:?}",
        );
    }

    #[test]
    fn empty_template_content_is_dropped() {
        // Llama3 has no `no_think_prefix` (empty string), so a
        // `kind: template` referencing it must be filtered at build time —
        // projection never sees an empty section.
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: maybe_no_think
      dialect: no_think_prefix
    - kind: section
      id: frame
      content: "x"
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::llama3();
        let b = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap();
        let names: Vec<&str> = b
            .schema()
            .system_prompt
            .items
            .iter()
            .filter_map(|it| match it {
                SystemPromptItem::Section(s) => Some(s.name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["frame"]);
        // The dropped template item is also absent from the name map.
        assert!(b.id_for_system_section("maybe_no_think").is_none());
    }

    #[test]
    fn template_depends_on_resolves_to_collection_id() {
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
    - kind: collection
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: Propose a few functional categories for the sections.
          user_prompt: Propose categories for the content above.
        assign:
          max_tokens: 128
          system_prompt: Assign each section to a category by number.
          user_prompt: Assign each section above to a category number.
      name: tools
      selection: { kind: top_k, k: 1 }
      sections:
        - id: t1
          content: "tool one"
    - kind: template
      id: tools_close
      dialect: tool_block_close
      depends_on: tools
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let b = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap();
        let tools_cid = b.id_for_system_collection("tools").unwrap();

        let template_deps: Vec<Option<super::super::super::ids::CollectionId>> = b
            .schema()
            .system_prompt
            .items
            .iter()
            .filter_map(|it| match it {
                SystemPromptItem::Section(s) if s.is_template => Some(s.depends_on),
                _ => None,
            })
            .collect();
        assert_eq!(template_deps, vec![Some(tools_cid), Some(tools_cid)]);
    }

    /// **The feasibility gate, at the projection level: zero glue islands.**
    ///
    /// Every `ProjectionSegment::Generated` run is a hole the engine must
    /// gap-fill, and a recurrence cannot compute the output of a token inserted
    /// mid-sequence — its state has already accumulated past the hole. So a
    /// projection that emits even one is a projection this lineage cannot run,
    /// and the wave refuses it (`reserve_glue_island`).
    ///
    /// The structural envelope (`system_open` … `system_close`) around a content
    /// section is the shape every system prompt has, and all three sections must
    /// now arrive as sealed K/V.
    #[test]
    fn a_projection_reserves_no_glue_islands() {
        let dlct = Dialect::chat_ml();
        let mut b =
            Builder::from_yaml_with_vars_and_dialect(TEMPLATE_YAML, &[], Some(&dlct)).unwrap();
        b.tokenize_templates::<std::convert::Infallible, _>(|s| {
            Ok(s.bytes().map(u32::from).collect())
        })
        .unwrap();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let convo = b.id_for_group("convo").unwrap();
        let proj = b.project(
            ProjectionTarget {
                layer: dialogue,
                group: convo,
                timeline: TimelineId::for_test(1),
            },
            &MockResolver::new(),
        );

        let generated: Vec<&str> = proj
            .segments
            .iter()
            .filter_map(|seg| match seg {
                ProjectionSegment::Generated { identity, .. } => Some(identity.name.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            generated.is_empty(),
            "projection reserved glue islands, which a recurrent model cannot \
             fill: {generated:?}"
        );

        // All three — the two structural templates and the content section —
        // arrive as sealed K/V.
        let sealed = proj
            .segments
            .iter()
            .filter(|seg| matches!(seg, ProjectionSegment::Sealed(SealedKind::Section(_))))
            .count();
        assert_eq!(
            sealed,
            3,
            "system_open + frame + system_close must all seal: {:?}",
            proj.segments.len()
        );
    }

    #[test]
    fn template_closing_after_collection_still_emits() {
        // Regression: `tools_close` is declared AFTER the `tools` collection
        // (with `depends_on: tools`). The emission pass drains the
        // collection-results map as each collection emits, so the close marker
        // must be gated on a set captured up front — otherwise it reads the
        // drained map, sees `None`, and the tool block is never closed in the
        // materialized context (an unclosed `<tools>` block reaches the model).
        use super::super::super::project::ProjectionSegment;
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
    - kind: collection
      name: tools
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: Propose categories.
          user_prompt: Propose categories.
        assign:
          max_tokens: 128
          system_prompt: Assign by number.
          user_prompt: Assign by number.
      selection: { kind: top_k, k: 1 }
      sections:
        - id: t1
          content: "tool one"
    - kind: template
      id: tools_close
      dialect: tool_block_close
      depends_on: tools
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let mut b = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap();
        b.tokenize_templates::<std::convert::Infallible, _>(|s| {
            Ok(s.bytes().map(u32::from).collect())
        })
        .unwrap();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let convo = b.id_for_group("convo").unwrap();
        let t1 = b.id_for_system_section("t1").unwrap();

        // Score t1 so the tools collection materialises (non-empty) — the
        // precondition for both `tools_open` and `tools_close` to emit.
        let resolver = MockResolver::new().with_section_score(t1, 0.9);
        let proj = b.project(
            ProjectionTarget {
                layer: dialogue,
                group: convo,
                timeline: TimelineId::for_test(1),
            },
            &resolver,
        );

        // Template sections seal like any other, so they emit as
        // `Sealed(Section)` and are identified by id. What this test is about is
        // unchanged: a `depends_on` template emits exactly when its collection
        // materialises, and one declared AFTER the collection still emits.
        let tools_open = b.id_for_system_section("tools_open").unwrap();
        let tools_close = b.id_for_system_section("tools_close").unwrap();
        let sealed: Vec<SectionId> = proj
            .segments
            .iter()
            .filter_map(|seg| match seg {
                ProjectionSegment::Sealed(SealedKind::Section(s)) => Some(s.id),
                _ => None,
            })
            .collect();
        assert!(
            sealed.contains(&tools_open),
            "tools_open should emit, got {sealed:?}",
        );
        assert!(
            sealed.contains(&tools_close),
            "tools_close (declared after the collection) must still emit, got {sealed:?}",
        );

        // And the whole projection is now gap-fill free: no `Generated` run
        // means no island for the engine to fill mid-sequence.
        let generated: Vec<&str> = proj
            .segments
            .iter()
            .filter_map(|seg| match seg {
                ProjectionSegment::Generated { identity, .. } => Some(identity.name.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            generated.is_empty(),
            "a projection must reserve no glue islands: {generated:?}",
        );
    }

    #[test]
    fn template_depends_on_unknown_collection_errors() {
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: nonexistent
    - kind: section
      id: frame
      content: "x"
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let err = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap_err();
        assert!(
            matches!(err, ConstructionError::UnknownCollection(_)),
            "got {err:?}",
        );
    }

    #[test]
    fn template_id_collision_with_section_errors() {
        let yaml = r#"
system_prompt:
  items:
    - kind: section
      id: dup
      content: "x"
    - kind: template
      id: dup
      dialect: system_start
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let err = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap_err();
        assert!(
            matches!(err, ConstructionError::DuplicateSectionName(ref n) if n == "dup"),
            "got {err:?}",
        );
    }

    #[test]
    fn existing_from_yaml_still_works_without_dialect() {
        // Schemas with no `kind: template` items must continue to parse via
        // the dialect-less entry points.
        let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
        assert!(b.id_for_layer("dialogue").is_some());
    }

    // —— Section tree (optional toggles + N-way selectors) ————————————————————————

    const TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: role
          content: "You are an assistant."
        - kind: selector
          id: length
          default: standard
          options:
            - id: terse
              content: "Be terse."
            - id: standard
              content: "Be balanced."
            - id: verbose
              content: "Be verbose."
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

    fn tree_target(b: &Builder) -> ProjectionTarget {
        ProjectionTarget {
            layer: b.id_for_layer("dialogue").unwrap(),
            group: b.id_for_group("convo").unwrap(),
            timeline: TimelineId::for_test(1),
        }
    }

    fn dialogue_tree(b: &Builder) -> &crate::projection::SectionTree {
        b.schema()
            .system_prompt
            .section_trees()
            .next()
            .expect("dialogue layer has a section_tree")
    }

    /// The sealed variant id of `node`'s `option` for the given dim selection.
    fn opt_var(b: &Builder, node: &str, option: &str, selection: &[u8]) -> SectionId {
        let tree = dialogue_tree(b);
        let n = tree.nodes.iter().find(|n| n.name == node).unwrap();
        let o = n.options.iter().find(|o| o.id == option).unwrap();
        o.variant_for(tree.pack(selection, n.ancestor_dims))
            .expect("variant sealed for this branch")
            .id
    }

    /// Two trees, with the interesting node declared only in the SECOND — the
    /// shape zend's bundled schema has and the one a single-tree reader misses.
    const TWO_TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: selector
          id: length
          default: standard
          options:
            - id: terse
              content: "Be terse."
            - id: standard
              content: "Be balanced."
    - kind: section_tree
      nodes:
        - kind: selector
          id: examples
          default: absent
          options:
            - id: absent
              content: ""
            - id: folder
              content: "Worked folder example."
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

    /// **Every declared section tree comes back, not just the first.**
    ///
    /// This accessor was `section_tree() -> Option<&SectionTree>` on the stated
    /// premise that "a schema has at most one". zend's bundled schema declares
    /// two, so every node in the second was invisible to all three callers —
    /// and invisibly so, because reading the first tree succeeds. A branch
    /// validation passed a schema missing the exact option it exists to pin,
    /// and its test asserted a valid schema was rejected while the real one
    /// went unchecked.
    #[test]
    fn every_declared_section_tree_is_returned() {
        let b = Builder::from_yaml(TWO_TREE_YAML).unwrap();
        let trees: Vec<&crate::projection::SectionTree> =
            b.schema().system_prompt.section_trees().collect();
        assert_eq!(trees.len(), 2, "both declared trees must come back");
        assert!(
            trees[0].nodes.iter().all(|n| n.name != "examples"),
            "the fixture must split the nodes across the two trees, or this \
             test would pass against a single-tree schema",
        );
        assert!(
            trees[1].nodes.iter().any(|n| n.name == "examples"),
            "the second tree's nodes are reachable",
        );
    }

    /// **T6.1 — branches share their prefix exactly as far as their dims agree.**
    ///
    /// A recurrent checkpoint is only valid for an *ordered prefix*: state does
    /// not compose, so `state([A][B])` is not any function of `state([A])` and
    /// `state([B])`. That makes the prefix identity the whole correctness
    /// argument, and this is the assertion that pins it — two selections that
    /// differ only in a later dim must produce the same section ids up to the
    /// node that first sees the difference, and different ones from there.
    ///
    /// Catches a key derived from the raw `selection` instead of
    /// `pack(selection, node.ancestor_dims)`: that version makes every node's id
    /// depend on dims declared *after* it, so two branches that genuinely share a
    /// prefix would be handed different checkpoints and the sharing in
    /// [`SectionTree::leaf_selections`]'s ordering would silently buy nothing.
    #[test]
    fn branch_prefix_shares_up_to_the_first_differing_dim() {
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        // dims: no_think (radix 2) then length (radix 3).
        assert_eq!(tree.dims.len(), 2);

        // Differ in the LAST dim only: everything above `length` is shared.
        let terse = tree.branch_prefix_ids(&[0, 0]);
        let verbose = tree.branch_prefix_ids(&[0, 2]);
        assert_eq!(
            terse.len(),
            verbose.len(),
            "same node count on both branches"
        );
        let first_diff = terse
            .iter()
            .zip(verbose.iter())
            .position(|(a, b)| a != b)
            .expect("the two branches must differ somewhere");
        assert_eq!(
            first_diff,
            terse.len() - 1,
            "only the `length` node itself should differ: {terse:?} vs {verbose:?}"
        );

        // Differ in the FIRST dim: the divergence starts at the top.
        let no_think_off = tree.branch_prefix_ids(&[1, 0]);
        assert_ne!(
            terse.first(),
            no_think_off.first(),
            "flipping the first dim must change the branch from its first node"
        );
    }

    /// **T6.2 — a branch's prefix is invariant to which collection members
    /// projection selects.**
    ///
    /// The prefix carries the placeholder node's own sealed anchor (`noop
    /// anchor`), never the collection's runtime top-k. That is what makes a
    /// checkpoint keyable at all: members are scored and top-k selected per
    /// turn, so a prefix that included them would be a different prefix on every
    /// projection and no key could name it.
    ///
    /// It is also not a new concession. `is_collection_member` already excludes
    /// members from the content-address prefix chain, so every sealed K/V below a
    /// collection is *already* conditioned on a prefix that ignores which members
    /// are present — this makes the state approximate the collection the same
    /// way, rather than differently.
    #[test]
    fn branch_prefix_carries_the_placeholder_not_the_collection_members() {
        let b = Builder::from_yaml(GROUP_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        let ids = tree.branch_prefix_ids(&tree.default_selection);

        let anchor_node = tree
            .nodes
            .iter()
            .find(|n| n.name == "noop_tool")
            .expect("fixture has the placeholder node");
        assert!(
            anchor_node.inject_collection.is_some(),
            "`noop_tool` is the inject_collection placeholder"
        );
        let anchor = anchor_node.options[0]
            .variant_for(tree.pack(&tree.default_selection, anchor_node.ancestor_dims))
            .expect("the placeholder seals its own anchor per branch")
            .id;
        assert!(
            ids.contains(&anchor),
            "the placeholder's sealed anchor is what nodes below it attend to, so \
             the state has to be computed over it too"
        );

        // No member of the embedded collection may appear.
        let members: Vec<SectionId> = tree
            .nodes
            .iter()
            .filter_map(|n| n.collection.as_ref())
            .flat_map(|tc| tc.variants.iter().flatten().map(|v| v.id))
            .collect();
        assert!(!members.is_empty(), "fixture has sealed collection members");
        for m in &members {
            assert!(
                !ids.contains(m),
                "collection member {m:?} leaked into the branch prefix — the \
                 checkpoint would be valid only for the one top-k that produced it"
            );
        }
    }

    /// **An out-of-scope gated dim names the same branch whatever it says.**
    ///
    /// `inner` lives inside `grp`'s present branch, so the absent side was
    /// sealed at `inner`'s default and never multiplied by it. Two selections
    /// that differ only in an out-of-scope dim are therefore the *same* branch,
    /// and `branch_prefix_ids` has to say so — a checkpoint keyed off a
    /// distinction the sealed variants never made would be computed twice and
    /// read once, or worse, read under the wrong key.
    #[test]
    fn an_out_of_scope_gated_dim_resolves_to_one_branch() {
        let b = Builder::from_yaml(GATED_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        assert_eq!(tree.dims.len(), 2, "grp, inner");
        let grp = tree
            .dims
            .iter()
            .position(|d| d.selector_id == "grp")
            .expect("grp dim");
        let inner = 1 - grp;

        let mut absent_a = vec![0u8; 2];
        absent_a[grp] = 1; // group absent
        absent_a[inner] = 0;
        let mut absent_b = absent_a.clone();
        absent_b[inner] = 1; // …and a non-default inner, which is out of scope

        assert_eq!(
            tree.branch_prefix_ids(&absent_a),
            tree.branch_prefix_ids(&absent_b),
            "the group is absent, so `inner` never multiplied this side — both \
             selections are the one sealed branch"
        );

        // In scope, the same dim genuinely does select a different branch.
        let mut present_a = vec![0u8; 2];
        present_a[grp] = 0;
        present_a[inner] = 0;
        let mut present_b = present_a.clone();
        present_b[inner] = 1;
        assert_ne!(
            tree.branch_prefix_ids(&present_a),
            tree.branch_prefix_ids(&present_b),
            "with the group present, `inner` is a real dimension"
        );
    }

    #[test]
    fn section_tree_default_emits_default_selection() {
        use crate::projection::{ProjectionMode, ResolvedSelection, SelectionState};
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let _dialogue = b.id_for_layer("dialogue").unwrap();
        // Declared names resolve to the default-selection variant.
        let no_think = b.id_for_system_section("no_think").unwrap();
        let role = b.id_for_system_section("role").unwrap();
        let length = b.id_for_system_section("length").unwrap();
        let resolver = MockResolver::new();

        // Empty state → defaults: no_think present, role, length=standard.
        let proj = b.project_with_selection(
            tree_target(&b),
            &resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        );
        let ids: Vec<SectionId> = proj.sealed_sections().map(|s| s.id).collect();
        assert_eq!(ids, vec![no_think, role, length]);
        // Both selectors report the option they emitted, by id.
        assert_eq!(
            proj.selections,
            vec![
                ResolvedSelection {
                    selector: "no_think".to_string(),
                    option: "present".to_string(),
                },
                ResolvedSelection {
                    selector: "length".to_string(),
                    option: "standard".to_string(),
                },
            ]
        );
    }

    #[test]
    fn section_tree_optional_absent_drops_node_and_reprefixes() {
        use crate::projection::{ProjectionMode, SelectionState};
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let resolver = MockResolver::new();

        let mut sel = SelectionState::new();
        sel.select("no_think", "absent");
        let proj =
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel);

        // no_think absent ⇒ it emits nothing; role + length still emit, but as
        // their ABSENT-branch variants (sealed without no_think above them).
        let ids: Vec<SectionId> = proj.sealed_sections().map(|s| s.id).collect();
        assert_eq!(
            ids,
            vec![
                opt_var(&b, "role", "content", &[1, 1]),
                opt_var(&b, "length", "standard", &[1, 1]),
            ]
        );
        // Distinct from the present-branch (default) variants.
        assert_ne!(
            opt_var(&b, "role", "content", &[1, 1]),
            opt_var(&b, "role", "content", &[0, 1]),
        );
    }

    #[test]
    fn section_tree_selector_override_emits_chosen_option() {
        use crate::projection::{ProjectionMode, ResolvedSelection, SelectionState};
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let resolver = MockResolver::new();

        let mut sel = SelectionState::new();
        sel.select("length", "verbose");
        let proj =
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel);

        let ids: Vec<SectionId> = proj.sealed_sections().map(|s| s.id).collect();
        // no_think present (default), role, length=verbose (overridden).
        assert_eq!(ids[2], opt_var(&b, "length", "verbose", &[0, 2]));
        assert_ne!(ids[2], opt_var(&b, "length", "standard", &[0, 1]));
        assert!(proj.selections.contains(&ResolvedSelection {
            selector: "length".to_string(),
            option: "verbose".to_string(),
        }));
    }

    /// A two-layer schema where a NON-dialogue layer carries `dials:` for the
    /// shared prompt's section-tree selector. Projecting that layer seeds the
    /// selector from its dial (beneath the tree default); the caller's per-turn
    /// selection still overrides it. This is the unified-prompt contract: layers
    /// differ only by the branch their dials pick, not by a prompt of their own.
    const DIALS_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: section
          id: role
          content: "You are an assistant."
        - kind: selector
          id: length
          default: standard
          options:
            - id: terse
              content: "Be terse."
            - id: standard
              content: "Be balanced."
            - id: verbose
              content: "Be verbose."
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
  - name: analysis
    window: 1000
    dials:
      length: verbose
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: analysis_group
        selection: { kind: always_visible }
"#;

    #[test]
    fn layer_dials_seed_section_tree_selection() {
        use crate::projection::{ProjectionMode, ResolvedSelection, SelectionState};
        let b = Builder::from_yaml(DIALS_YAML).unwrap();
        let resolver = MockResolver::new();
        let length_is = |proj: &crate::projection::Projection, opt: &str| {
            proj.selections.contains(&ResolvedSelection {
                selector: "length".to_string(),
                option: opt.to_string(),
            })
        };
        let target = |layer: &str, group: &str, tl: u64| ProjectionTarget {
            layer: b.id_for_layer(layer).unwrap(),
            group: b.id_for_group(group).unwrap(),
            timeline: TimelineId::for_test(tl),
        };

        // dialogue has no dials → the section-tree's authored default (standard).
        let proj = b.project_with_selection(
            target("dialogue", "convo", 1),
            &resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        );
        assert!(
            length_is(&proj, "standard"),
            "dialogue (no dials) uses the tree default: {:?}",
            proj.selections
        );

        // analysis carries `length: verbose`; with no caller selection the layer
        // dial seeds the branch.
        let ana = target("analysis", "analysis_group", 2);
        let proj = b.project_with_selection(
            ana,
            &resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        );
        assert!(
            length_is(&proj, "verbose"),
            "analysis layer's dial seeds length=verbose: {:?}",
            proj.selections
        );

        // A caller's per-turn selection overrides the layer dial.
        let mut sel = SelectionState::new();
        sel.select("length", "terse");
        let proj = b.project_with_selection(ana, &resolver, ProjectionMode::Decode, &sel);
        assert!(
            length_is(&proj, "terse"),
            "caller selection beats the layer dial: {:?}",
            proj.selections
        );
    }

    #[test]
    fn section_tree_seals_full_cross_product_with_correct_prefixes() {
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        let no_think = tree.nodes.iter().find(|n| n.name == "no_think").unwrap();
        let role = tree.nodes.iter().find(|n| n.name == "role").unwrap();
        let length = tree.nodes.iter().find(|n| n.name == "length").unwrap();

        // Two dims: no_think (radix 2), length (radix 3); defaults present/standard.
        assert_eq!(tree.dims.len(), 2);
        assert_eq!(tree.default_selection, vec![0, 1]);

        // no_think: `present` has 1 variant (root), `absent` is empty (0).
        let present = no_think.options.iter().find(|o| o.id == "present").unwrap();
        let absent = no_think.options.iter().find(|o| o.id == "absent").unwrap();
        assert_eq!(present.variants.len(), 1);
        assert!(present.variants[0].in_tree_prefix.is_empty());
        assert!(absent.variants.is_empty());

        // role (mandatory): one variant per no_think branch (2).
        let role_opt = &role.options[0];
        assert_eq!(role_opt.variants.len(), 2);
        assert_eq!(
            role_opt.variant_for(0).unwrap().in_tree_prefix,
            vec![present.variants[0].id] // present branch attends to no_think
        );
        assert!(role_opt.variant_for(1).unwrap().in_tree_prefix.is_empty()); // absent branch

        // length: each of 3 options × 2 no_think branches = 6 variants.
        assert_eq!(
            length
                .options
                .iter()
                .map(|o| o.variants.len())
                .sum::<usize>(),
            6
        );
        let std = length.options.iter().find(|o| o.id == "standard").unwrap();
        // length.standard on the present branch attends to [no_think, role].
        assert_eq!(
            std.variant_for(0).unwrap().in_tree_prefix,
            vec![present.variants[0].id, role_opt.variant_for(0).unwrap().id]
        );
    }

    /// A nested `section_tree` whose inner selector sits BELOW static content +
    /// a fake-tool anchor.  The point: content above the nested tree multiplies
    /// only by the OUTER selector (no_think), never the inner one (effort).
    const NESTED_TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: framing
          content: "You are an assistant."
        - kind: section
          id: noop_tool
          content: "noop anchor tool"
        - kind: section_tree
          nodes:
            - kind: selector
              id: effort
              default: balanced
              options:
                - id: off
                  content: "Answer directly."
                - id: balanced
                  content: "Reason."
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

    #[test]
    fn nested_section_tree_inner_selector_inherits_outer_dims() {
        let b = Builder::from_yaml(NESTED_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);

        // The inner `effort` selector was lifted into the outer tree's dim list,
        // declared AFTER no_think.
        assert_eq!(tree.dims.len(), 2);
        assert_eq!(tree.dims[0].selector_id, "no_think");
        assert_eq!(tree.dims[1].selector_id, "effort");
        assert_eq!(tree.default_selection, vec![0, 1]); // present, balanced

        let noop = tree.nodes.iter().find(|n| n.name == "noop_tool").unwrap();
        let effort = tree.nodes.iter().find(|n| n.name == "effort").unwrap();

        // The anchor sits above the nested tree → it multiplies ONLY by the
        // outer dim (no_think, radix 2). effort never multiplies it.
        assert_eq!(noop.ancestor_dims, 1);
        assert_eq!(noop.options[0].variants.len(), 2);

        // The inner selector fans out across the outer dim: 2 options × 2
        // no_think branches = 4 variants; its ancestor width is the outer dim.
        assert_eq!(effort.ancestor_dims, 1);
        assert_eq!(
            effort
                .options
                .iter()
                .map(|o| o.variants.len())
                .sum::<usize>(),
            4
        );
    }

    #[test]
    fn nested_section_tree_inner_variant_prefix_includes_outer_anchor() {
        let b = Builder::from_yaml(NESTED_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        let node = |name: &str| tree.nodes.iter().find(|n| n.name == name).unwrap();

        // present-no_think branch: ancestor key over [no_think] = 0.
        let key = tree.pack(&[0, 0], 1);
        let no_think_present = node("no_think")
            .options
            .iter()
            .find(|o| o.id == "present")
            .unwrap();
        let framing_var = node("framing").options[0].variant_for(0).unwrap();
        let noop_var = node("noop_tool").options[0].variant_for(0).unwrap();
        let off_var = node("effort")
            .options
            .iter()
            .find(|o| o.id == "off")
            .unwrap()
            .variant_for(key)
            .unwrap();

        // effort.off on the present branch seals against the full outer chain,
        // anchor included: [no_think.present, framing, noop_tool].
        assert_eq!(
            off_var.in_tree_prefix,
            vec![no_think_present.variants[0].id, framing_var.id, noop_var.id]
        );
    }

    #[test]
    fn nested_section_tree_outer_content_stable_across_inner_selection() {
        use crate::projection::{ProjectionMode, SelectionState};
        let b = Builder::from_yaml(NESTED_TREE_YAML).unwrap();
        let resolver = MockResolver::new();

        let proj = |effort: &str| {
            let mut sel = SelectionState::new();
            sel.select("effort", effort);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect::<Vec<_>>()
        };
        let off = proj("off");
        let balanced = proj("balanced");

        // Four sections emit: no_think, framing, noop_tool, effort. The first
        // three (static content + anchor) are IDENTICAL across the inner
        // directive selection — only the last differs. That is the continuity
        // win: switching the inner selector never re-prefills the content above.
        assert_eq!(off.len(), 4);
        assert_eq!(off[..3], balanced[..3]);
        assert_ne!(off[3], balanced[3]);
    }

    /// A collection embedded as a tree node, sitting UNDER no_think but ABOVE
    /// the noop anchor + inner directive tree.  The tools must seal ×2 (no_think)
    /// and be prefix-transparent: the anchor + directives below seal as if the
    /// variable tool members were not there.
    const COLLECTION_TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: framing
          content: "You are an assistant."
        - kind: collection
          name: tools
          selection: { kind: top_k, k: 2 }
          summary:
            chunk: 4
            categorize:
              max_tokens: 256
              system_prompt: cat
              user_prompt: cat
            assign:
              max_tokens: 128
              system_prompt: assign
              user_prompt: assign
          sections:
            - id: tool_a
              content: "tool a"
            - id: tool_b
              content: "tool b"
        - kind: section
          id: noop_tool
          content: "noop anchor"
        - kind: section_tree
          nodes:
            - kind: selector
              id: effort
              default: balanced
              options:
                - id: off
                  content: "direct"
                - id: balanced
                  content: "reason"
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

    #[test]
    fn collection_tree_node_members_seal_per_outer_branch() {
        let b = Builder::from_yaml(COLLECTION_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);

        // no_think + effort are the dims; the collection is NOT a dim.
        assert_eq!(tree.dims.len(), 2);
        assert_eq!(tree.dims[0].selector_id, "no_think");
        assert_eq!(tree.dims[1].selector_id, "effort");

        let tools = tree.nodes.iter().find(|n| n.name == "tools").unwrap();
        let tc = tools
            .collection
            .as_ref()
            .expect("tools is a collection node");
        // Two members, each sealed ×2 (no_think present/absent).
        assert_eq!(tc.variants.len(), 2);
        assert_eq!(tc.variants[0].len(), 2);
        assert_eq!(tc.variants[1].len(), 2);
        // present vs absent no_think → distinct member seals.
        assert_ne!(
            tc.member_variant(0, 0).unwrap().id,
            tc.member_variant(0, 1).unwrap().id
        );
        // The canonical (default-branch = no_think present, key 0) ids back the
        // SectionCollection used for scoring + depends_on gating.
        assert_eq!(tc.collection.sections.len(), 2);
        assert_eq!(
            tc.collection.sections[0].id,
            tc.member_variant(0, 0).unwrap().id
        );
    }

    #[test]
    fn collection_tree_node_is_prefix_transparent() {
        let b = Builder::from_yaml(COLLECTION_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        let node = |name: &str| tree.nodes.iter().find(|n| n.name == name).unwrap();

        let no_think_present = node("no_think")
            .options
            .iter()
            .find(|o| o.id == "present")
            .unwrap();
        let framing_var = node("framing").options[0].variant_for(0).unwrap();

        // The tool members seal against the static base above them only:
        // [no_think.present, framing] on the present branch.
        let tc = node("tools").collection.as_ref().unwrap();
        assert_eq!(
            tc.member_variant(0, 0).unwrap().in_tree_prefix,
            vec![no_think_present.variants[0].id, framing_var.id]
        );

        // The noop anchor sits AFTER the collection, but its prefix is still just
        // [no_think, framing] — the variable tool members are NOT in it.
        let noop_var = node("noop_tool").options[0].variant_for(0).unwrap();
        assert_eq!(
            noop_var.in_tree_prefix,
            vec![no_think_present.variants[0].id, framing_var.id]
        );

        // The inner effort selector anchors on the noop, still excluding tools.
        let off_present = node("effort")
            .options
            .iter()
            .find(|o| o.id == "off")
            .unwrap()
            .variant_for(tree.pack(&[0, 0], 1))
            .unwrap();
        assert_eq!(
            off_present.in_tree_prefix,
            vec![no_think_present.variants[0].id, framing_var.id, noop_var.id]
        );
    }

    #[test]
    fn collection_tree_node_projects_active_branch_stable_across_directive() {
        use crate::projection::{ProjectionMode, SelectionState};
        let b = Builder::from_yaml(COLLECTION_TREE_YAML).unwrap();
        let resolver = MockResolver::new();

        let proj = |no_think: &str, effort: &str| {
            let mut sel = SelectionState::new();
            sel.select("no_think", no_think);
            sel.select("effort", effort);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect::<Vec<_>>()
        };

        // present branch: no_think, framing, tool_a, tool_b, noop, effort = 6.
        let base = proj("present", "balanced");
        assert_eq!(base.len(), 6);

        // Switching the inner directive does NOT change the static/tool prefix —
        // the collection never re-prefills for the directive below it.
        let other = proj("present", "off");
        assert_eq!(base[..5], other[..5]);
        assert_ne!(base[5], other[5]);

        // Switching no_think DOES swap the tools to their absent-branch seals.
        let tools = dialogue_tree(&b)
            .nodes
            .iter()
            .find(|n| n.name == "tools")
            .unwrap()
            .collection
            .as_ref()
            .unwrap();
        let present = proj("present", "balanced");
        let absent = proj("absent", "balanced");
        assert!(present.contains(&tools.member_variant(0, 0).unwrap().id));
        assert!(absent.contains(&tools.member_variant(0, 1).unwrap().id));
        assert!(!absent.contains(&tools.member_variant(0, 0).unwrap().id));
    }

    #[test]
    fn inject_collection_replaces_placeholder_with_collection_selection() {
        use crate::projection::{ProjectionMode, ProjectionSegment, SealedKind, SelectionState};
        // The noop placeholder declares `inject_collection: tools`: it still seals
        // its own anchor (the directive below attends to it) but at projection its
        // content is REPLACED by the tools collection's top-k, and the collection
        // itself defers its own emission.  `member_glue` puts a REAL newline token
        // between the selected tools (not baked into any tool's seal).
        let yaml = COLLECTION_TREE_YAML
            .replace(
                "        - kind: section\n          id: noop_tool\n          content: \"noop anchor\"",
                "        - kind: section\n          id: noop_tool\n          inject_collection: tools\n          content: \"noop anchor\"",
            )
            .replace(
                "          selection: { kind: top_k, k: 2 }",
                "          selection: { kind: top_k, k: 2 }\n          member_glue: \"\\n\"",
            );
        let mut b = Builder::from_yaml(&yaml).unwrap();
        // The glue is a live-prefilled structural token, so it must be tokenised
        // before projection (mock: one byte → one token, so "\n" → [10]).
        b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
            .unwrap();

        // Wiring: the tools collection is marked deferred, and the noop node points
        // at it.
        let cid = b.id_for_system_collection("tools").unwrap();
        let tree = dialogue_tree(&b);
        let tools = tree.nodes.iter().find(|n| n.name == "tools").unwrap();
        assert!(
            tools.collection.as_ref().unwrap().deferred_projection,
            "the injected collection must defer its own projection"
        );
        let noop = tree.nodes.iter().find(|n| n.name == "noop_tool").unwrap();
        assert_eq!(noop.inject_collection, Some(cid));

        // The noop STILL seals its own anchor variant (so the directive below it
        // anchors on a stable prefix) — that seal exists even though it isn't
        // projected.
        let noop_anchor = noop.options[0].variant_for(0).unwrap().id;
        let tc = tools.collection.as_ref().unwrap();
        let tool_a = tc.member_variant(0, 0).unwrap().id;
        let tool_b = tc.member_variant(1, 0).unwrap().id;
        let framing = opt_var(&b, "framing", "content", &[0]);
        let effort = opt_var(&b, "effort", "balanced", &[0, 0]);

        // Project the present/balanced branch and read the FULL segment list.
        let resolver = MockResolver::new();
        let mut sel = SelectionState::new();
        sel.select("no_think", "present");
        sel.select("effort", "balanced");
        let proj =
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel);

        let ids: Vec<SectionId> = proj.sealed_sections().map(|s| s.id).collect();
        // The placeholder's own anchor is NOT emitted; the two real tools take its
        // place, and the directive below still emits.
        assert!(
            !ids.contains(&noop_anchor),
            "the noop anchor must not appear in the projection"
        );
        assert!(ids.contains(&tool_a) && ids.contains(&tool_b));
        let pos = |id: SectionId| ids.iter().position(|x| *x == id).unwrap();
        assert!(pos(framing) < pos(tool_a));
        assert!(pos(tool_a) < pos(tool_b));
        assert!(pos(tool_b) < pos(effort));

        // The glue is a REAL token between the two tools: in the full segment list,
        // tool_a → Generated("\n" = [10]) → tool_b, contiguous and in that order.
        let seg_pos = |id: SectionId| {
            proj.segments
                .iter()
                .position(|s| {
                    matches!(s, ProjectionSegment::Sealed(SealedKind::Section(r)) if r.id == id)
                })
                .unwrap()
        };
        let a = seg_pos(tool_a);
        let bseg = seg_pos(tool_b);
        assert_eq!(
            bseg,
            a + 2,
            "exactly one segment sits between the two tools"
        );
        match &proj.segments[a + 1] {
            ProjectionSegment::Generated { tokens, .. } => {
                assert_eq!(
                    tokens.as_ref(),
                    &vec![10u32],
                    "glue must be the newline token"
                );
            }
            other => panic!("expected a Generated glue token between tools, got {other:?}"),
        }
        // No glue leads the first tool (the segment before tool_a is not the glue).
        assert!(
            !matches!(&proj.segments[a - 1], ProjectionSegment::Generated { tokens, .. } if tokens.as_ref() == &vec![10u32]),
            "glue must not lead the first selected tool"
        );
    }

    /// A collection embedded in a section tree resolves `Named` the way a
    /// top-level one does — the members its selector holds and nothing else —
    /// rather than selecting nothing because the tree path is score-based.
    #[test]
    fn a_tree_collection_emits_the_members_its_named_selector_holds() {
        use crate::projection::{ProjectionMode, SelectionState};
        let yaml = COLLECTION_TREE_YAML.replace(
            "          selection: { kind: top_k, k: 2 }",
            "          selection: { kind: named, selector: pick }",
        );
        assert_ne!(
            yaml, COLLECTION_TREE_YAML,
            "the fixture's selection line moved"
        );
        let b = Builder::from_yaml(&yaml).unwrap();
        let tree = dialogue_tree(&b);
        let tc = tree
            .nodes
            .iter()
            .find(|n| n.name == "tools")
            .and_then(|n| n.collection.as_ref())
            .unwrap();
        let tool_a = tc.member_variant(0, 0).unwrap().id;
        let tool_b = tc.member_variant(1, 0).unwrap().id;
        let resolver = MockResolver::new();
        let project = |sel: &SelectionState| -> Vec<SectionId> {
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        let mut sel = SelectionState::new();
        sel.select("no_think", "present");
        sel.select("effort", "balanced");
        let unset = project(&sel);
        assert!(
            !unset.contains(&tool_a) && !unset.contains(&tool_b),
            "an unset selector names nothing"
        );

        sel.select_all("pick", ["tool_b"]);
        let picked = project(&sel);
        assert!(picked.contains(&tool_b), "{picked:?}");
        assert!(!picked.contains(&tool_a), "{picked:?}");
    }

    /// An `optional_group` gates a whole sub-tree (markers + collection + inject
    /// placeholder) on a binary dim: `absent` omits the entire block (markers
    /// included), and nodes BELOW it seal distinctly per (tools present/absent).
    const GROUP_TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: framing
          content: "You are an assistant."
        - kind: optional_group
          id: tools_on
          default: present
          nodes:
            - kind: section
              id: tools_open
              content: "<tools>"
            - kind: collection
              name: tools
              selection: { kind: top_k, k: 2 }
              member_glue: "\n"
              summary:
                chunk: 4
                categorize:
                  max_tokens: 256
                  system_prompt: cat
                  user_prompt: cat
                assign:
                  max_tokens: 128
                  system_prompt: assign
                  user_prompt: assign
              sections:
                - id: tool_a
                  content: "tool a"
                - id: tool_b
                  content: "tool b"
            - kind: section
              id: noop_tool
              inject_collection: tools
              content: "noop anchor"
            - kind: section
              id: tools_close
              content: "</tools>"
        - kind: section
          id: directive
          content: "Respond."
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

    #[test]
    fn optional_group_omits_subtree_when_absent_and_reprefixes_below() {
        use crate::projection::{ProjectionMode, SelectionState};
        let mut b = Builder::from_yaml(GROUP_TREE_YAML).unwrap();
        b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
            .unwrap();
        let resolver = MockResolver::new();

        let proj = |no_think: &str, tools: &str| -> Vec<SectionId> {
            let mut sel = SelectionState::new();
            sel.select("no_think", no_think);
            sel.select("tools_on", tools);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        // `tools_on` is a real dim (after no_think).
        let tree = dialogue_tree(&b);
        assert_eq!(tree.dims.len(), 2);
        assert_eq!(tree.dims[0].selector_id, "no_think");
        assert_eq!(tree.dims[1].selector_id, "tools_on");

        let tc = tree
            .nodes
            .iter()
            .find(|n| n.name == "tools")
            .unwrap()
            .collection
            .as_ref()
            .unwrap();
        // Members seal ONLY under the present branch — ×no_think (2), not ×4.
        let member_ids: std::collections::HashSet<SectionId> =
            tc.variants.iter().flatten().map(|v| v.id).collect();
        assert_eq!(
            member_ids.len(),
            2 * 2,
            "2 members × 2 no_think (present only)"
        );

        let present = proj("present", "present");
        let absent = proj("present", "absent");

        // Present: the whole block emits (markers + ≥1 tool).
        let topen = opt_var(&b, "tools_open", "content", &[0, 0]);
        let tclose = opt_var(&b, "tools_close", "content", &[0, 0]);
        assert!(present.contains(&topen) && present.contains(&tclose));
        assert!(
            present.iter().any(|id| member_ids.contains(id)),
            "present branch must inject tools"
        );

        // Absent: NOTHING tool-related — markers, members, and the noop anchor all
        // gone (the block is omitted, not an empty <tools></tools> shell).
        assert!(!absent.contains(&topen) && !absent.contains(&tclose));
        assert!(
            !absent.iter().any(|id| member_ids.contains(id)),
            "absent branch must emit no tools"
        );

        // The directive below seals DISTINCTLY per tools state (the permutation
        // increase) and both variants project on their branch.
        let dir_present = opt_var(&b, "directive", "content", &[0, 0]);
        let dir_absent = opt_var(&b, "directive", "content", &[0, 1]);
        assert_ne!(
            dir_present, dir_absent,
            "directive must seal a distinct variant per tools-on/off"
        );
        assert!(present.contains(&dir_present));
        assert!(absent.contains(&dir_absent));
    }

    /// A `selector` declared INSIDE an `optional_group` present branch is a GATED
    /// dim: it multiplies only the present side, the absent side seals at the
    /// selector's default, and `pack` masks an out-of-scope runtime value back to
    /// that default — so a node after the group projects correctly for EVERY
    /// combination, including the dangerous one (group absent + non-default inner).
    /// A `selector` (`inner`) declared inside an `optional_group`'s present
    /// branch, so `inner` is a GATED dim: it multiplies only the present side.
    const GATED_TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional_group
          id: grp
          default: present
          nodes:
            - kind: selector
              id: inner
              default: a
              options:
                - id: a
                  content: "INNER-A"
                - id: b
                  content: "INNER-B"
          absent:
            - kind: section
              id: alt
              content: "ALT"
        - kind: section
          id: after
          content: "AFTER"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

    #[test]
    fn selector_inside_optional_group_is_gated() {
        use crate::projection::{ProjectionMode, SelectionState};
        // No panic at build (the whole point) — the absent side never multiplied
        // by `inner`, yet every branch's assignment stays width-consistent.
        let b = Builder::from_yaml(GATED_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);

        // `inner` is gated by `grp` being present (option 0).
        let grp_idx = tree
            .dims
            .iter()
            .position(|d| d.selector_id == "grp")
            .unwrap();
        let inner_dim = tree.dims.iter().find(|d| d.selector_id == "inner").unwrap();
        assert_eq!(inner_dim.gate, Some((grp_idx, 0)));

        // `after` seals exactly THREE variants — (present,a), (present,b), (absent)
        // — NOT four: the absent side does not multiply by `inner`.
        let after = tree.nodes.iter().find(|n| n.name == "after").unwrap();
        assert_eq!(after.options[0].variants.len(), 3);
        let after_a = after.options[0]
            .variant_for(tree.pack(&[0, 0], 2))
            .unwrap()
            .id;
        let after_b = after.options[0]
            .variant_for(tree.pack(&[0, 1], 2))
            .unwrap()
            .id;
        let after_absent = after.options[0]
            .variant_for(tree.pack(&[1, 0], 2))
            .unwrap()
            .id;
        assert_ne!(after_a, after_b);

        let resolver = MockResolver::new();
        let proj = |grp: &str, inner: &str| -> Vec<SectionId> {
            let mut sel = SelectionState::new();
            sel.select("grp", grp);
            sel.select("inner", inner);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        // Present: `after` follows the inner selection; the inner content emits.
        assert!(proj("present", "a").contains(&after_a));
        assert!(proj("present", "b").contains(&after_b));

        // THE DANGEROUS CASE: group absent + a NON-DEFAULT inner value. `pack`
        // masks `inner` back to its default, so `after` lands on its single absent
        // variant — no build panic, no silent missing section.
        assert!(proj("absent", "b").contains(&after_absent));
        assert!(proj("absent", "a").contains(&after_absent));
        assert!(!proj("absent", "b").contains(&after_a));
        assert!(!proj("absent", "b").contains(&after_b));

        // The `inner` selector itself only emits when the group is present.
        let inner_a_id = opt_var(&b, "inner", "a", &[0]);
        assert!(proj("present", "a").contains(&inner_a_id));
        assert!(!proj("absent", "a").contains(&inner_a_id));
    }

    /// An `optional_group` nested inside another: the inner group's toggle is
    /// itself gated by the outer group's side, so `dim_active` chains both gates.
    /// A node after both groups must project correctly even when the inner group
    /// is selected present while the outer is absent (inner is out of scope).
    #[test]
    fn nested_optional_group_chains_gates() {
        use crate::projection::{ProjectionMode, SelectionState};
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional_group
          id: outer
          default: present
          nodes:
            - kind: optional_group
              id: inner
              default: present
              nodes:
                - kind: section
                  id: deep
                  content: "DEEP"
              absent:
                - kind: section
                  id: inner_alt
                  content: "INNER-ALT"
          absent:
            - kind: section
              id: outer_alt
              content: "OUTER-ALT"
        - kind: section
          id: tail
          content: "TAIL"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let b = Builder::from_yaml(yaml).unwrap();
        let tree = dialogue_tree(&b);

        let outer_idx = tree
            .dims
            .iter()
            .position(|d| d.selector_id == "outer")
            .unwrap();
        let inner = tree.dims.iter().find(|d| d.selector_id == "inner").unwrap();
        // The nested group's toggle is gated by the OUTER group being present.
        assert_eq!(inner.gate, Some((outer_idx, 0)));

        // `tail` seals three variants: (outer present, inner present),
        // (outer present, inner absent), (outer absent) — the absent outer side
        // collapses the inner dim.
        let tail = tree.nodes.iter().find(|n| n.name == "tail").unwrap();
        assert_eq!(tail.options[0].variants.len(), 3);
        let tail_pp = tail.options[0]
            .variant_for(tree.pack(&[0, 0], 2))
            .unwrap()
            .id;
        let tail_pa = tail.options[0]
            .variant_for(tree.pack(&[0, 1], 2))
            .unwrap()
            .id;
        let tail_absent = tail.options[0]
            .variant_for(tree.pack(&[1, 0], 2))
            .unwrap()
            .id;

        let resolver = MockResolver::new();
        let proj = |outer: &str, inner: &str| -> Vec<SectionId> {
            let mut sel = SelectionState::new();
            sel.select("outer", outer);
            sel.select("inner", inner);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        assert!(proj("present", "present").contains(&tail_pp));
        assert!(proj("present", "absent").contains(&tail_pa));
        // Outer absent + inner present: inner is out of scope, masked away — `tail`
        // lands on its single absent variant, never the inner-present one.
        assert!(proj("absent", "present").contains(&tail_absent));
        assert!(!proj("absent", "present").contains(&tail_pp));
    }

    /// A `dialect:` mandatory tree section is LIVE-PREFILLED glue, not a sealed
    /// section: it allocates no sealed variant, is prefix-transparent (the node
    /// below seals as if it weren't there), and emits a `Generated` run gated to
    /// the branch it lives in (here, tools-on).
    #[test]
    fn dialect_tree_section_is_live_prefilled_glue() {
        use crate::projection::{ProjectionMode, ProjectionSegment, SelectionState};
        use candle_transformers::models::dialect::Dialect;
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: optional_group
          id: tools_on
          default: present
          nodes:
            - kind: section
              id: tools_open
              dialect: tool_block_open
            - kind: section
              id: inner
              content: "inner"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let dlct = Dialect::chat_ml();
        let mut b = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap();
        b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
            .unwrap();

        let tree = dialogue_tree(&b);
        let tools_open = tree.nodes.iter().find(|n| n.name == "tools_open").unwrap();
        assert!(tools_open.glue.is_some(), "a dialect section is glue");
        assert!(
            tools_open.options[0].variants.is_empty(),
            "glue allocates no sealed variant"
        );

        // Prefix-transparent: `inner` (present branch, right after the glue) anchors
        // only on no_think — the glue contributes NOTHING to its prefix.
        let no_think_present = tree
            .nodes
            .iter()
            .find(|n| n.name == "no_think")
            .unwrap()
            .options
            .iter()
            .find(|o| o.id == "present")
            .unwrap()
            .variants[0]
            .id;
        let inner = tree.nodes.iter().find(|n| n.name == "inner").unwrap();
        let inner_present = inner.options[0].variant_for(tree.pack(&[0, 0], 2)).unwrap();
        assert_eq!(
            inner_present.in_tree_prefix,
            vec![no_think_present],
            "glue must not appear in the K/V prefix below it"
        );

        // Projection: tools on → a `Generated` run named `tools_open`; tools off →
        // none (gated to the present branch).
        let resolver = MockResolver::new();
        let has_glue = |tools: &str| {
            let mut sel = SelectionState::new();
            sel.select("no_think", "present");
            sel.select("tools_on", tools);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .segments
                .iter()
                .any(|s| {
                    matches!(
                        s,
                        ProjectionSegment::Generated { identity, tokens }
                            if identity.name == "tools_open" && !tokens.is_empty()
                    )
                })
        };
        assert!(
            has_glue("present"),
            "glue emits a Generated run when tools on"
        );
        assert!(!has_glue("absent"), "glue is omitted when tools off");

        // The GUI event surfaces the marker as a real glue ROW, never a section.
        let mut sel = SelectionState::new();
        sel.select("no_think", "present");
        sel.select("tools_on", "present");
        let proj =
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel);
        let ev = crate::projection::from_projection(
            &proj.segments,
            b.schema(),
            &resolver,
            &crate::projection::SelectionScores::default(),
            0,
            0,
            1.0,
        );
        use crate::projection::SystemItem;
        assert!(
            ev.selection.system.iter().any(|it| matches!(
                it, SystemItem::Glue { name, .. } if name == "tools_open"
            )),
            "marker renders as a glue row"
        );
        assert!(
            !ev.selection.system.iter().any(|it| matches!(
                it, SystemItem::Section { name, .. } if name == "tools_open"
            )),
            "marker is NOT a section row"
        );
    }

    /// An `optional_group` with an `absent:` branch carries DIFFERENT content per
    /// branch under the SAME binary dim (no new permutations): present emits the
    /// tools-aware grounding, absent the tools-free one, and the directive below
    /// genuinely anchors on whichever grounding fired in its branch.
    #[test]
    fn optional_group_absent_branch_carries_alternative_grounding() {
        use crate::projection::{ProjectionMode, SelectionState};
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: frame
          content: "frame"
        - kind: optional_group
          id: tools_on
          default: present
          nodes:
            - kind: section
              id: tools_open
              content: "<tools>"
            - kind: section
              id: grounding_tools
              content: "fetch it with a tool"
          absent:
            - kind: section
              id: grounding_no_tools
              content: "say so rather than guessing"
        - kind: section
          id: directive
          content: "respond"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let b = Builder::from_yaml(yaml).unwrap();
        let resolver = MockResolver::new();
        let proj = |tools: &str| -> Vec<SectionId> {
            let mut sel = SelectionState::new();
            sel.select("no_think", "present");
            sel.select("tools_on", tools);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        // present: no_think=0, tools=0 ; absent: no_think=0, tools=1.
        let g_tools = opt_var(&b, "grounding_tools", "content", &[0, 0]);
        let g_no = opt_var(&b, "grounding_no_tools", "content", &[0, 1]);

        let present = proj("present");
        let absent = proj("absent");
        assert!(
            present.contains(&g_tools) && !present.contains(&g_no),
            "present branch shows the tools grounding only"
        );
        assert!(
            absent.contains(&g_no) && !absent.contains(&g_tools),
            "absent branch shows the tools-free grounding only"
        );

        // grounding_tools seals ONLY ×no_think on the present branch, grounding_no_tools
        // ONLY ×no_think on the absent branch — 2 + 2 = 4 total, no extra dim.
        let tree = dialogue_tree(&b);
        let gt = tree
            .nodes
            .iter()
            .find(|n| n.name == "grounding_tools")
            .unwrap();
        let gn = tree
            .nodes
            .iter()
            .find(|n| n.name == "grounding_no_tools")
            .unwrap();
        assert_eq!(gt.options[0].variants.len(), 2);
        assert_eq!(gn.options[0].variants.len(), 2);

        // KV continuity: the directive below anchors on the grounding that fired
        // in ITS branch (and never the other branch's grounding).
        let dir = tree.nodes.iter().find(|n| n.name == "directive").unwrap();
        let dir_present = dir.options[0].variant_for(tree.pack(&[0, 0], 2)).unwrap();
        let dir_absent = dir.options[0].variant_for(tree.pack(&[0, 1], 2)).unwrap();
        assert!(dir_present.in_tree_prefix.contains(&g_tools));
        assert!(!dir_present.in_tree_prefix.contains(&g_no));
        assert!(dir_absent.in_tree_prefix.contains(&g_no));
        assert!(!dir_absent.in_tree_prefix.contains(&g_tools));
    }

    /// The tool-catalog overview is a REAL sealed tree section before `<tools>`:
    /// `tools_open` anchors on it (its variant id is in `tools_open`'s prefix),
    /// and the content is rewritable pre-prefill WITHOUT changing the variant id
    /// — so the K/V chain stays intact while the daemon writes the real overview.
    #[test]
    fn tool_summary_section_anchors_chain_and_content_is_settable() {
        use crate::projection::{ProjectionMode, SelectionState};
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: section
          id: frame
          content: "frame"
        - kind: section
          id: tool_summary
          content: "placeholder"
        - kind: section
          id: tools_open
          content: "<tools>"
        - kind: section
          id: directive
          content: "respond"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let mut b = Builder::from_yaml(yaml).unwrap();
        let _dialogue = b.id_for_layer("dialogue").unwrap();

        let var = |b: &Builder, name: &str| {
            dialogue_tree(b)
                .nodes
                .iter()
                .find(|n| n.name == name)
                .unwrap()
                .options[0]
                .variant_for(0)
                .unwrap()
                .id
        };
        let summary_id = var(&b, "tool_summary");
        // `tools_open` (and everything below) genuinely attends the summary: its
        // sealed variant id is in `tools_open`'s in-tree prefix.
        let tools_open_node = dialogue_tree(&b)
            .nodes
            .iter()
            .find(|n| n.name == "tools_open")
            .unwrap()
            .clone();
        assert!(
            tools_open_node.options[0]
                .variant_for(0)
                .unwrap()
                .in_tree_prefix
                .contains(&summary_id),
            "tools_open must anchor on the tool_summary K/V"
        );

        // Rewriting the content pre-prefill keeps the SAME variant id (the chain
        // is unchanged; only the bytes the prefill seals differ).
        b.set_tree_section_content("tool_summary", "The tools are grouped by purpose.")
            .unwrap();
        assert_eq!(
            var(&b, "tool_summary"),
            summary_id,
            "variant id must not move"
        );
        let node = dialogue_tree(&b)
            .nodes
            .iter()
            .find(|n| n.name == "tool_summary")
            .unwrap()
            .clone();
        assert_eq!(node.options[0].content, "The tools are grouped by purpose.");

        // Setting an unknown section errors.
        assert!(b.set_tree_section_content("nope", "x").is_err());

        // Projects in order: frame → tool_summary → <tools> → directive.
        let resolver = MockResolver::new();
        let ids: Vec<SectionId> = b
            .project_with_selection(
                tree_target(&b),
                &resolver,
                ProjectionMode::Decode,
                &SelectionState::new(),
            )
            .sealed_sections()
            .map(|s| s.id)
            .collect();
        let pos = |id: SectionId| ids.iter().position(|x| *x == id).unwrap();
        assert!(pos(var(&b, "frame")) < pos(summary_id));
        assert!(pos(summary_id) < pos(var(&b, "tools_open")));
    }

    #[test]
    fn runtime_add_to_tree_collection_seals_per_branch_no_alias() {
        // Like COLLECTION_TREE_YAML but the tools collection starts empty — the
        // real catalog is installed at runtime (the daemon's install_tool_catalog).
        let yaml = COLLECTION_TREE_YAML.replace(
            "          sections:\n            - id: tool_a\n              content: \"tool a\"\n            - id: tool_b\n              content: \"tool b\"",
            "          sections: []",
        );
        let mut b = Builder::from_yaml(&yaml).unwrap();
        let _dialogue = b.id_for_layer("dialogue").unwrap();
        let cid = b.id_for_system_collection("tools").unwrap();

        // Runtime add, exactly as install_tool_catalog does (by CollectionId).
        let ws = b
            .add_section_to_collection(cid, "web_search", "ws def", 100.0)
            .unwrap();
        let calc = b
            .add_section_to_collection(cid, "calc", "calc def", 100.0)
            .unwrap();

        let tree = dialogue_tree(&b);
        let tools = tree
            .nodes
            .iter()
            .find(|n| n.name == "tools")
            .unwrap()
            .collection
            .as_ref()
            .unwrap();

        // Two members, each sealed ×2 (no_think branches).
        assert_eq!(tools.variants.len(), 2);
        assert_eq!(tools.variants[0].len(), 2);
        assert_eq!(tools.variants[1].len(), 2);

        // Each member's canonical id is its default-branch variant, and resolves
        // by name.
        assert_eq!(
            ws,
            tools.member_variant(0, tools.default_branch).unwrap().id
        );
        assert_eq!(b.id_for_system_section("web_search").unwrap(), ws);

        // No id aliasing: every sealed id across both members + both branches is
        // distinct (the ×branch block sits above the prior max each add).
        let mut ids: Vec<_> = tools
            .variants
            .iter()
            .flatten()
            .map(|v| v.id.raw())
            .collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "all per-branch member ids must be unique");
        // calc was added after web_search → its block is strictly higher.
        assert!(calc.raw() > ws.raw());
    }

    #[test]
    fn section_tree_invalid_optional_default_rejected() {
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: x
          content: "x"
          default: maybe
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let err = Builder::from_yaml(yaml).unwrap_err();
        assert!(
            matches!(err, ConstructionError::InvalidToggleDefault { ref value, .. } if value == "maybe"),
            "got {err:?}",
        );
    }

    #[test]
    fn section_tree_unknown_selector_default_rejected() {
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: selector
          id: length
          default: nope
          options:
            - id: terse
              content: "a"
            - id: standard
              content: "b"
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
        let err = Builder::from_yaml(yaml).unwrap_err();
        assert!(
            matches!(err, ConstructionError::UnknownTreeOption { ref option, .. } if option == "nope"),
            "got {err:?}",
        );
    }
}
