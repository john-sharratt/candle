//! Tests for variable substitution in schema templates, and for the zend thinking
//! dial.
//!
//! A placeholder must name a bound variable, and a schema with any placeholder
//! left unsubstituted is refused at parse. The thinking dial uses the checkpoint's
//! own effort vocabulary, and `balanced` is silent: it seals no variant and emits
//! no segment, so the default path reproduces the vendor template. `off` stays
//! non-empty, telling the model to answer without deliberating.

use super::*;

// —— Variable substitution ————————————————————————————————————————————————————

pub(super) const SUBST_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "Hello {name}, welcome to {project}."
layers:
  - name: dialogue
    window: 8000
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
    budget: { priority: 100 }
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;

#[test]
pub(super) fn substitution_replaces_placeholders() {
    let b = Builder::from_yaml_with_vars(SUBST_YAML, &[("name", "Alice"), ("project", "Candle")])
        .unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
    let section = b.section(frame).unwrap();
    assert_eq!(section.content, "Hello Alice, welcome to Candle.");
}

#[test]
pub(super) fn substitution_missing_var_is_error() {
    let err = Builder::from_yaml_with_vars(SUBST_YAML, &[("name", "Alice")]).unwrap_err();
    match err {
        super::super::error::ConstructionError::UnresolvedVariable { name } => {
            assert_eq!(name, "project");
        }
        other => panic!("expected UnresolvedVariable, got {other:?}"),
    }
}

#[test]
pub(super) fn substitution_catches_template_typo() {
    let yaml = r#"
system_prompt:
  sections:
    - id: frame
      content: "Welcome to {wokrspace}."
layers:
  - name: dialogue
    window: 8000
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
    budget: { priority: 100 }
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml_with_vars(yaml, &[("workspace", "candle")]).unwrap_err();
    match err {
        super::super::error::ConstructionError::UnresolvedVariable { name } => {
            assert_eq!(name, "wokrspace");
        }
        other => panic!("expected UnresolvedVariable, got {other:?}"),
    }
}

#[test]
pub(super) fn substitution_leaves_non_identifier_braces_alone() {
    let yaml = r#"
system_prompt:
  sections:
    - id: frame
      content: 'Example JSON: {"tool": "search", "args": {}}'
layers:
  - name: dialogue
    window: 8000
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
    budget: { priority: 100 }
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml_with_vars(yaml, &[]).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
    let section = b.section(frame).unwrap();
    assert_eq!(
        section.content,
        r#"Example JSON: {"tool": "search", "args": {}}"#
    );
}

#[test]
pub(super) fn from_yaml_rejects_any_placeholder() {
    let yaml = r#"
system_prompt:
  sections:
    - id: frame
      content: "Welcome to {workspace}."
layers:
  - name: dialogue
    window: 8000
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
    budget: { priority: 100 }
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::UnresolvedVariable { .. }
    ));
}

#[test]
pub(super) fn substitution_baked_into_immutable_schema() {
    let b = Builder::from_yaml_with_vars(SUBST_YAML, &[("name", "A"), ("project", "B")]).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
    let content_first = b.section(frame).unwrap().content.clone();
    let content_second = b.section(frame).unwrap().content.clone();
    assert_eq!(content_first, content_second);
    assert_eq!(content_first, "Hello A, welcome to B.");
}

/// **The thinking dial speaks the checkpoint's own reasoning-effort vocabulary,
/// and `balanced` is silent.**
///
/// Qwen3.8's chat template carries three trained effort levels — `low`,
/// `medium`, `xhigh` — each a system-prompt instruction placed ahead of the
/// turn (never text written into the `<think>` block). The dial reuses those
/// strings verbatim so it pulls the lever the checkpoint was tuned for.
///
/// `balanced` is the template's `medium`, which injects nothing at all. It must
/// therefore carry no content and seal NO variant: an empty section in the
/// prompt is a different thing from no section, and only the latter reproduces
/// the vendor template on the default path.
///
/// `off` is asserted non-empty on purpose, though it is no longer what keeps
/// the block shut — the `Off` steering tree closes it structurally, on the
/// token after `<think>`. What this directive still does is tell the model to
/// answer without deliberating, so it does not open with the throat-clearing it
/// would otherwise put in a block it is about to be denied.
#[test]
pub(super) fn zend_thinking_effort_uses_the_checkpoints_own_vocabulary() {
    use candle_transformers::models::dialect::Dialect;
    let yaml = include_str!("../../../../zend/src/prompts/projection.yaml");
    let dlct = Dialect::chat_ml();
    let b = Builder::from_yaml_with_vars_and_dialect(
        yaml,
        &[("workspace", "candle"), ("model", "test-model")],
        Some(&dlct),
    )
    .expect("zend projection.yaml must parse");

    let node = b
        .schema()
        .system_prompt
        .section_trees()
        .flat_map(|t| t.nodes.iter())
        .find(|n| n.name == "thinking_effort")
        .expect("the dialogue schema declares a thinking_effort selector");

    let opt = |id: &str| {
        node.options
            .iter()
            .find(|o| o.id == id)
            .unwrap_or_else(|| panic!("thinking_effort has no {id:?} option"))
    };

    // `balanced` == the vendor's `medium`: no content, hence no sealed variant,
    // hence no emitted segment on the default path.
    assert_eq!(opt("balanced").content, "", "balanced must emit nothing");
    assert!(
        opt("balanced").variants.is_empty(),
        "an option with no content seals no variant"
    );

    // The rungs that do speak carry the template's own strings.
    assert!(
        opt("quick")
            .content
            .starts_with("Reasoning effort is set to low."),
        "quick must use the template's `low` string, got {:?}",
        opt("quick").content
    );
    for id in ["deep", "exhaustive"] {
        assert!(
            opt(id)
                .content
                .starts_with("Reasoning effort is set to xhigh."),
            "{id} must use the template's `xhigh` string, got {:?}",
            opt(id).content
        );
    }
    // Exhaustive EXTENDS xhigh rather than replacing it — the model has three
    // trained levels and this dial has five rungs, so the top two share an
    // anchor and separate on this extra directive plus their token budgets.
    assert!(
        opt("exhaustive").content.len() > opt("deep").content.len(),
        "exhaustive must add a directive on top of the xhigh anchor"
    );

    // `off`'s prose is the only thing suppressing the block on this family.
    assert!(
        !opt("off").content.is_empty(),
        "off's prose is load-bearing — emptying it silently disables the dial"
    );
    assert!(!opt("off").variants.is_empty(), "off must seal a variant");
}

/// Sanity check: the live zend projection schema parses + validates.
#[test]
pub(super) fn zend_projection_yaml_parses() {
    use candle_transformers::models::dialect::Dialect;
    let yaml = include_str!("../../../../zend/src/prompts/projection.yaml");
    let dlct = Dialect::chat_ml();
    let b = Builder::from_yaml_with_vars_and_dialect(
        yaml,
        &[("workspace", "candle"), ("model", "test-model")],
        Some(&dlct),
    )
    .expect("zend projection.yaml must parse");

    let expected_layers = ["repo_map", "code_reading", "dialogue"];
    for name in &expected_layers {
        assert!(
            b.id_for_layer(name).is_some(),
            "layer {name:?} missing from zend schema"
        );
    }

    let last = b.schema().layers.last().expect("at least one layer");
    assert_eq!(last.name, "dialogue", "dialogue must be the topmost layer");

    assert!(b.id_for_group("primary_conversation").is_some());

    // Every layer's summary content is a model decode. `repo_map` was the one
    // structural holdout while its turns were directory listings; it now ingests
    // one conversation per folder ending in a decoded description of what the
    // folder is for, so a deterministic merge of names would throw away the only
    // thing those turns carry that a listing does not.
    let mode_of = |layer: &str| {
        let id = b.id_for_layer(layer).unwrap();
        b.schema()
            .layers
            .iter()
            .find(|l| l.id == id)
            .unwrap()
            .summary
            .summaries
            .as_ref()
            .expect("layer has a summaries level")
            .content
    };
    for single in ["repo_map", "code_reading", "dialogue"] {
        assert_eq!(
            mode_of(single),
            Content::Decode,
            "{single} should stay a model decode"
        );
    }
}
