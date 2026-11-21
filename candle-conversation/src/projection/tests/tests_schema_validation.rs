//! Tests for parsing and validating a projection schema from YAML and the builder.
//!
//! A schema with a missing or empty system prompt, a duplicate name, an unknown
//! selection kind or a percentage over 100 is refused with the error that names
//! it, and a silent fall-through to a default is refused as well. Pinned ids
//! survive removed layers, and a schema pinning only some of its layers is an
//! error. Lifting a tool call's repetition penalties is opt-in, so silence keeps
//! them.

use super::*;

/// **Lifting repetition penalties inside tool calls is opt-in, and silence
/// means no.**
///
/// It suits a schema whose tool arguments are quotations — paths, identifiers,
/// numbers that are only right when they repeat. It is the reverse for a schema
/// whose arguments are prose: a cast decoded every utterance with presence,
/// frequency, repeat and DRY all off, and one character said the same sentence
/// word for word for a hundred turns. A default of "on" would keep doing that
/// to any schema that had not heard of the flag.
#[test]
pub(super) fn tool_calls_keep_their_penalties_unless_the_schema_asks_otherwise() {
    let quiet = Builder::from_yaml(SIMPLE_YAML).unwrap();
    assert!(
        !quiet.schema().free_tool_calls_from_penalties,
        "a schema that says nothing must keep the penalties it configured"
    );

    let asked = Builder::from_yaml(&format!(
        "free_tool_calls_from_penalties: true\n{SIMPLE_YAML}"
    ))
    .unwrap();
    assert!(asked.schema().free_tool_calls_from_penalties);

    // A programmatic schema asks through the builder — the passthrough's way.
    let built = Builder::from_yaml(SIMPLE_YAML)
        .unwrap()
        .free_tool_calls_from_penalties();
    assert!(built.schema().free_tool_calls_from_penalties);
}

/// `hits:` reaches the schema field by field, a block left out is the
/// default, and a block the counting cannot run is refused at parse.
#[test]
pub(super) fn the_hits_block_parses_and_is_validated() {
    let quiet = Builder::from_yaml(SIMPLE_YAML).unwrap();
    assert_eq!(quiet.schema().hits, HitsConfig::default());

    let set = Builder::from_yaml(&format!(
        "hits:\n  layers: [50]\n  gain: 2.5\n{SIMPLE_YAML}"
    ))
    .unwrap();
    assert_eq!(set.schema().hits.layers.percentiles(), &[50]);
    assert_eq!(set.schema().hits.gain, 2.5);
    assert_eq!(set.schema().hits.beta, 0.2, "the default leak");

    for bad in [
        "layers: []",
        "layers: [120]",
        "gain: -1",
        "beta: 0",
        "beta: 2",
    ] {
        let err = Builder::from_yaml(&format!("hits:\n  {bad}\n{SIMPLE_YAML}"))
            .err()
            .unwrap();
        assert!(
            matches!(err, ConstructionError::InvalidHits(_)),
            "{bad}: {err}"
        );
    }
    let unknown = Builder::from_yaml(&format!("hits:\n  gian: 2\n{SIMPLE_YAML}"));
    assert!(unknown.is_err(), "a misspelt hits key must not be ignored");
}

/// An item flagged `pin: true` is pinned and an unflagged one is not; flags
/// with no `pin_tokens` to hold them are refused.
#[test]
pub(super) fn system_prompt_items_are_pinned_by_their_flag() {
    const PROMPT: &str = "system_prompt:\n  sections:\n    - id: frame\n      content: \"You are a helpful assistant.\"\n    - id: values\n      content: \"Be honest.\"\n";
    assert!(SIMPLE_YAML.contains(PROMPT), "the fixture's prompt block");
    let with = |prompt: &str| SIMPLE_YAML.replacen(PROMPT, prompt, 1);
    let quiet = Builder::from_yaml(SIMPLE_YAML).unwrap();
    assert!(quiet.schema().system_prompt.pins.is_empty());

    let flagged = |bound: &str| {
        format!(
            "system_prompt:\n{bound}  items:\n    - kind: section\n      id: frame\n      pin: true\n      content: \"You are a helpful assistant.\"\n    - kind: section\n      id: values\n      content: \"Be honest.\"\n"
        )
    };
    let pinned = Builder::from_yaml(&with(&flagged("  pin_tokens: 64\n"))).unwrap();
    let sp = &pinned.schema().system_prompt;
    let frame = pinned.id_for_system_section("frame").unwrap();
    let values = pinned.id_for_system_section("values").unwrap();
    assert_eq!(sp.pins.tokens, 64);
    assert_eq!(sp.pins.pin_of(sp, frame), Pin::Whole);
    assert_eq!(sp.pins.pin_of(sp, values), Pin::No, "unflagged, unpinned");

    let unbounded = Builder::from_yaml(&with(&flagged("")));
    assert!(matches!(unbounded, Err(ConstructionError::InvalidPins(_))));
}

#[test]
pub(super) fn yaml_parses_and_assigns_ids() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();

    // Layers assigned in declaration order.
    let ground = b.id_for_layer("ground").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert!(ground.raw() < dialogue.raw());

    // Groups globally unique.
    let facts = b.id_for_group("facts").unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    assert_ne!(facts, conv);
    assert!(facts.raw() < conv.raw());

    // Sections (per-layer scoped — both live under `dialogue` here).
    let frame = b.id_for_system_section("frame").unwrap();
    let values = b.id_for_system_section("values").unwrap();
    assert!(frame.raw() < values.raw());
}

#[test]
pub(super) fn policy_inherits_default_then_layer_overrides() {
    let yaml = SIMPLE_YAML
        .replace(
            "layers:",
            "default_policy:\n  preset: high_recall_scope\nlayers:",
        )
        .replace(
            "  - name: dialogue\n    window: 8000\n",
            "  - name: dialogue\n    window: 8000\n    policy:\n      beta: 0.5\n",
        );
    let b = Builder::from_yaml(&yaml).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    // ground inherits the schema default preset verbatim.
    assert_eq!(
        b.layer(ground).unwrap().policy.config,
        super::super::PolicyPreset::HighRecallScope.config()
    );
    // dialogue overrides only beta on top of the inherited high-recall base.
    let dp = b.layer(dialogue).unwrap().policy.config;
    assert_eq!(dp.beta, 0.5);
    assert_eq!(
        dp.budget_max,
        super::super::PolicyPreset::HighRecallScope
            .config()
            .budget_max
    );
}

#[test]
pub(super) fn policy_unknown_preset_is_error() {
    let yaml = SIMPLE_YAML.replace("layers:", "default_policy:\n  preset: nope\nlayers:");
    assert!(matches!(
        Builder::from_yaml(&yaml),
        Err(super::super::ConstructionError::UnknownPolicyPreset { .. })
    ));
}

#[test]
pub(super) fn policy_evict_above_min_is_error() {
    let yaml = SIMPLE_YAML.replace(
        "  - name: dialogue\n    window: 8000\n",
        "  - name: dialogue\n    window: 8000\n    policy:\n      min_score: 5\n      evict_score: 10\n",
    );
    assert!(matches!(
        Builder::from_yaml(&yaml),
        Err(super::super::ConstructionError::InvalidPolicy { .. })
    ));
}

#[test]
pub(super) fn gather_scope_defaults_to_shared_and_parses_conversation() {
    // Unset → Shared.
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert_eq!(b.layer(dialogue).unwrap().gather_scope, GatherScope::Shared);

    // Flagging the dialogue layer parses to Conversation.
    let yaml = SIMPLE_YAML.replace(
        "  - name: dialogue\n    window: 8000",
        "  - name: dialogue\n    gather_scope: conversation\n    window: 8000",
    );
    let b2 = Builder::from_yaml(&yaml).unwrap();
    let d2 = b2.id_for_layer("dialogue").unwrap();
    assert_eq!(
        b2.layer(d2).unwrap().gather_scope,
        GatherScope::Conversation
    );
}

#[test]
pub(super) fn unknown_layer_name_returns_none() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    assert!(b.id_for_layer("nonexistent").is_none());
}

/// `gathered` is RUNTIME state, never a YAML field: every declared layer starts
/// gathered and only `set_layer_gathered` clears it.
///
/// The return value is how a caller learns the name was a layer at all.
/// `--disable-layer` also names section *collections* (`response`, `mood`),
/// which are suppressed on a different path entirely, so "not found" is an
/// ordinary outcome there and must never be read as "cleared". And the flag
/// restores: it is one boot's decision, not a deletion.
#[test]
pub(super) fn every_declared_layer_is_gathered_until_the_mutator_clears_it() {
    let mut b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert!(
        b.layer(ground).unwrap().gathered,
        "a declared layer gathers by default"
    );
    assert!(b.layer(dialogue).unwrap().gathered);

    assert!(b.set_layer_gathered("ground", false), "known layer found");
    assert!(!b.layer(ground).unwrap().gathered);
    assert!(
        b.layer(dialogue).unwrap().gathered,
        "clearing one layer must not touch another",
    );

    // A name that is not a layer changes nothing, and says so.
    assert!(!b.set_layer_gathered("response", false));
    assert!(!b.layer(ground).unwrap().gathered);
    assert!(b.layer(dialogue).unwrap().gathered);

    assert!(b.set_layer_gathered("ground", true));
    assert!(b.layer(ground).unwrap().gathered);
}

/// A plain-prompt schema's dialogue window is its fixed 32,768 until the
/// mutator sets another; a name that is not a layer changes nothing.
#[test]
pub(super) fn a_layer_window_is_set_by_name() {
    let mut b = Builder::for_plain_prompt("You are helpful.", SectionId::new(7));
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert_eq!(b.layer(dialogue).unwrap().window, 32_768);
    assert!(b.set_layer_window("dialogue", 262_144));
    assert_eq!(b.layer(dialogue).unwrap().window, 262_144);
    assert!(!b.set_layer_window("response", 1));
    assert_eq!(b.layer(dialogue).unwrap().window, 262_144);
}

#[test]
pub(super) fn decode_priority_defaults_to_low_and_parses_normal_and_high() {
    // Unset → Low.
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert_eq!(
        b.layer(dialogue).unwrap().decode_priority,
        DecodePriority::Low
    );

    // `normal` and `high` parse to their variants.
    for (word, want) in [
        ("normal", DecodePriority::Normal),
        ("high", DecodePriority::High),
        ("low", DecodePriority::Low),
    ] {
        let yaml = SIMPLE_YAML.replace(
            "  - name: dialogue\n    window: 8000",
            &format!("  - name: dialogue\n    decode_priority: {word}\n    window: 8000"),
        );
        let b = Builder::from_yaml(&yaml).unwrap();
        let d = b.id_for_layer("dialogue").unwrap();
        assert_eq!(
            b.layer(d).unwrap().decode_priority,
            want,
            "decode_priority: {word}"
        );
    }
}

#[test]
pub(super) fn on_corrupt_turn_defaults_to_drop_conversation_and_parses_drop_turn() {
    // Unset → DropConversation: the ingest-layer default, where one file is
    // one timeline and dropping it triggers a clean re-ingest.
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert_eq!(
        b.layer(dialogue).unwrap().on_corrupt_turn,
        CorruptTurnPolicy::DropConversation
    );

    // Both spellings parse to their variants. A silent fall-through to the
    // default here would make the dialogue layer drop whole conversations on
    // one corrupt turn — the exact failure the policy exists to prevent.
    for (word, want) in [
        ("drop_turn", CorruptTurnPolicy::DropTurn),
        ("drop_conversation", CorruptTurnPolicy::DropConversation),
    ] {
        let yaml = SIMPLE_YAML.replace(
            "  - name: dialogue\n    window: 8000",
            &format!("  - name: dialogue\n    on_corrupt_turn: {word}\n    window: 8000"),
        );
        let b = Builder::from_yaml(&yaml).unwrap();
        let d = b.id_for_layer("dialogue").unwrap();
        assert_eq!(
            b.layer(d).unwrap().on_corrupt_turn,
            want,
            "on_corrupt_turn: {word}"
        );
    }
}

#[test]
pub(super) fn decode_priority_ratio_is_decode_tokens_per_prefill() {
    // The airtime ratio R the continuous-fair-wave throttle keys on.
    assert_eq!(DecodePriority::Low.ratio(), 1);
    assert_eq!(DecodePriority::Normal.ratio(), 16);
    assert_eq!(DecodePriority::High.ratio(), 64);
    // Always >= 1 so a prefill can never be fully starved.
    for p in [
        DecodePriority::Low,
        DecodePriority::Normal,
        DecodePriority::High,
    ] {
        assert!(p.ratio() >= 1);
    }
}

#[test]
pub(super) fn builder_fallback_dialogue_layer_is_high_priority() {
    // The template-less fallback IS the dialogue layer, so it inherits the
    // interactive priority even without a YAML declaration.
    let b = Builder::for_plain_prompt("You are a helpful assistant.", SectionId::new(1));
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert_eq!(
        b.layer(dialogue).unwrap().decode_priority,
        DecodePriority::High
    );
}

#[test]
pub(super) fn schema_accessors_work() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let layer = b.layer(ground).unwrap();
    assert_eq!(layer.name, "ground");

    let facts = b.id_for_group("facts").unwrap();
    let group = b.group(facts).unwrap();
    assert_eq!(group.name, "facts");

    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
    let section = b.section(frame).unwrap();
    assert_eq!(section.name, "frame");
}

// —— YAML validation errors ————————————————————————————————————————————————————

#[test]
pub(super) fn yaml_duplicate_group_name_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
    - id: stub
      content: "stub"
layers:
  - name: layer1
    window: 9000
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
    groups:
      - id: dup
        selection: { kind: always_visible }
  - name: layer2
    window: 9000
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
    groups:
      - id: dup
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(
        matches!(
            err,
            super::super::error::ConstructionError::DuplicateGroupName(_)
        ),
        "expected DuplicateGroupName, got {:?}",
        err
    );
}

/// A schema of two layers, each with one group, the layer and group ids as
/// given (`""` for none).
pub(super) fn pinned_yaml(first: (&str, &str), second: (&str, &str)) -> String {
    let layer = |name: &str, group: &str, ids: (&str, &str)| {
        format!(
            r#"
  - name: {name}
    {}
    window: 9000
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
    groups:
      - id: {group}
        {}
        selection: {{ kind: always_visible }}"#,
            ids.0, ids.1
        )
    };
    format!(
        r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
    - id: stub
      content: "stub"
layers:{}{}
"#,
        layer("low", "low_g", first),
        layer("dialogue", "dialogue_g", second)
    )
}

/// **Pinned ids are the ids, wherever the layer sits**: a schema that dropped
/// the layers between these two keeps the dialogue at the numbers its
/// persisted conversations carry.
#[test]
pub(super) fn yaml_pinned_persisted_ids_survive_removed_layers() {
    let yaml = pinned_yaml(
        ("persisted_id: 2", "persisted_id: 2"),
        ("persisted_id: 10", "persisted_id: 10"),
    );
    let builder = Builder::from_yaml(&yaml).unwrap();
    assert_eq!(builder.id_for_layer("low").unwrap().raw(), 2);
    assert_eq!(builder.id_for_layer("dialogue").unwrap().raw(), 10);
    assert_eq!(builder.id_for_group("low_g").unwrap().raw(), 2);
    assert_eq!(builder.id_for_group("dialogue_g").unwrap().raw(), 10);

    let positional = Builder::from_yaml(&pinned_yaml(("", ""), ("", ""))).unwrap();
    assert_eq!(positional.id_for_layer("dialogue").unwrap().raw(), 2);
    assert_eq!(positional.id_for_group("dialogue_g").unwrap().raw(), 2);
}

/// A schema pinning only some of its layers is refused.
#[test]
pub(super) fn yaml_partly_pinned_persisted_ids_are_an_error() {
    let yaml = pinned_yaml(
        ("persisted_id: 2", "persisted_id: 2"),
        ("", "persisted_id: 10"),
    );
    let err = Builder::from_yaml(&yaml).unwrap_err();
    assert!(
        matches!(err, super::super::error::ConstructionError::PersistedIds(_)),
        "expected PersistedIds, got {err:?}"
    );
}

#[test]
pub(super) fn yaml_duplicate_section_name_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "A"
    - id: s1
      content: "B"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::DuplicateSectionName(_)
    ));
}

#[test]
pub(super) fn yaml_unknown_selection_kind_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: fuzzy_pick }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::UnknownSelectionKind(_)
    ));
}

#[test]
pub(super) fn yaml_invalid_priority_zero_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: always_visible }
        budget: { priority: 0 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::InvalidPriority { .. }
    ));
}

#[test]
pub(super) fn yaml_invalid_percentage_over_100_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 150 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::InvalidPercentage { .. }
    ));
}

// —— Construction validation errors ———————————————————————————————————————————

#[test]
pub(super) fn construction_sibling_min_percent_exceeds_100_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp1
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 60 }
      - id: grp2
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 60 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::MinPercentExceedsTotal { .. }
    ));
}

#[test]
pub(super) fn construction_max_less_than_min_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 60, max_percent: 40 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::MaxLessThanMin { .. }
    ));
}

#[test]
pub(super) fn construction_top_k_zero_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: top_k, k: 0 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::InvalidTopK { .. }
    ));
}

#[test]
pub(super) fn construction_conversation_both_zero_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection:
          kind: conversation
          recent: 0
          historical_top_k: 0
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::InvalidConversationK { .. }
    ));
}

#[test]
pub(super) fn construction_negative_score_threshold_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: always_visible }
        score_threshold: -0.5
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::NegativeScoreThreshold { .. }
    ));
}

#[test]
pub(super) fn yaml_top_k_missing_k_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: top_k }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::InvalidTopK { .. }
    ));
}

#[test]
pub(super) fn missing_system_prompt_is_parse_error() {
    // The top-level system_prompt is a required field; omitting it entirely is a
    // deserialization (parse) error before construction validation runs.
    let yaml = r#"
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::YamlParse(_)
    ));
}

#[test]
pub(super) fn empty_system_prompt_is_construction_error() {
    // The unified top-level system_prompt is required and must be non-empty;
    // an empty section list fails construction with EmptySystemPrompt.
    let yaml = r#"
system_prompt:
  sections: []
layers:
  - name: layer
    window: 9000
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
    groups:
      - id: grp
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::super::error::ConstructionError::EmptySystemPrompt
    ));
}

#[test]
pub(super) fn yaml_section_priority_field_parses() {
    const YAML: &str = r#"
system_prompt:
  items:
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
      name: pick
      selection: { kind: top_k, k: 1 }
      sections:
        - id: low_priority
          content: "L"
          priority: 10
        - id: high_priority
          content: "H"
          priority: 1000
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
    budget: { priority: 100 }
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let high = b.id_for_system_section("high_priority").unwrap();
    let low = b.id_for_system_section("low_priority").unwrap();

    // The `priority:` field parses (the YAML built); selection is now
    // belief-driven by score, so the higher-scored section is the single pick.
    let resolver = MockResolver::new()
        .with_section_score(high, 1.0)
        .with_section_score(low, 0.1);
    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(p.sealed_sections().count(), 1);
    assert_eq!(p.sealed_sections().next().unwrap().id, high);
}
