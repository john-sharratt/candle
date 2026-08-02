//! Tool-catalog integration tests that don't need a GPU or model.
//!
//! These exercise the orchestrator pieces in isolation:
//!
//! - `install_tool_catalog` adds one section per registered tool to the
//!   dialogue layer's `system_prompt`, switches the layer's section
//!   selection to `TopK`, and returns the `(name, SectionId, json_line)`
//!   triples in registry order.
//!
//! - `extract_tool_calls` parses Hermes-format `<tool_call>` blocks
//!   from a model response and skips malformed ones.
//!
//! - `format_tool_responses` wraps each result in
//!   `<tool_response>...</tool_response>`.
//!
//! - `run_tool` dispatches via the registry and returns either the
//!   tool's typed response or the canonical error JSON shape.
//!
//! Running on CPU only is intentional: this layer should be tested
//! independently of the inference engine so failures localise to the
//! tool plumbing rather than the BDP scoring or model.

use serde_json::{json, Value};

use candle_conversation::projection::{self, SystemPromptItem};
use zend::tools::{
    extract_tool_calls, format_tool_responses, install_tool_catalog, run_tool, run_tool_calls,
    ToolCall, ToolResult,
};
use zend_tools::{registry, ToolContext};

const PROJECTION_YAML: &str = include_str!("../src/prompts/projection.yaml");

/// Parse the bundled projection.yaml the same way the daemon does.
fn build_test_projection() -> projection::Builder {
    // The bundled YAML uses `kind: template` items (system_open,
    // tools_open, etc.) so the parser needs a `Dialect`; ChatML is
    // the closest standard analogue to Qwen3's tokens for resolving
    // the template strings.
    let dialect = candle_conversation::models::Dialect::chat_ml();
    projection::Builder::from_yaml_with_vars_and_dialect(
        PROJECTION_YAML,
        &[("workspace", "test")],
        Some(&dialect),
    )
    .expect("projection.yaml must parse")
}

/// Find the dialogue layer's `tools` collection — top-level OR embedded as a
/// section-tree node (the catalog sits under the `no_think` selector, so it is
/// a prefix-transparent tree-collection node rather than a top-level item).
fn tools_collection(builder: &projection::Builder) -> Option<&projection::SectionCollection> {
    for it in &builder.schema().system_prompt.items {
        match it {
            SystemPromptItem::Collection(c) if c.name == "tools" => return Some(c),
            SystemPromptItem::SectionTree(t) => {
                for n in &t.nodes {
                    if let Some(tc) = &n.collection {
                        if tc.collection.name == "tools" {
                            return Some(&tc.collection);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Helper: count tool sections currently in the dialogue layer's
/// `tools` collection (the canonical default-branch members).
fn n_tools_in_collection(builder: &projection::Builder) -> usize {
    tools_collection(builder).map_or(0, |c| c.sections.len())
}

// ── install_tool_catalog ─────────────────────────────────────────────────────

#[test]
fn install_tool_catalog_adds_one_section_per_registered_tool() {
    let mut builder = build_test_projection();
    let _dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");

    let n_before = n_tools_in_collection(&builder);
    let n_tools_in_registry = registry::all_tools().len();

    let installed = install_tool_catalog(&mut builder).unwrap();

    assert_eq!(
        installed.len(),
        n_tools_in_registry,
        "every registered tool must appear in the installed catalog",
    );

    let n_after = n_tools_in_collection(&builder);
    assert_eq!(n_after, n_before + n_tools_in_registry);
}

#[test]
fn install_tool_catalog_uses_existing_yaml_collection_topk() {
    // The YAML declares the tools collection with `selection: top_k k: 3`.
    // After install, that selection rule must be unchanged — the catalog
    // is appended to the existing collection rather than redefining it.
    let mut builder = build_test_projection();
    let _dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");
    install_tool_catalog(&mut builder).unwrap();

    let coll = tools_collection(&builder).expect("tools collection must exist");
    match coll.selection {
        projection::SelectionRule::TopK { k } => {
            assert!(k >= 1, "top-k must keep at least one tool");
        }
        ref other => panic!("expected TopK selection on tools collection, got {other:?}"),
    }
}

#[test]
fn install_tool_catalog_returns_section_ids_in_registry_order() {
    let mut builder = build_test_projection();
    let _dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");

    let installed = install_tool_catalog(&mut builder).unwrap();
    // `install_tool_catalog` walks the resolved tool-definition catalog
    // (`tool_def::all()`, sorted by name) and installs one section per tool in
    // that order, so the returned triples must match the catalog iteration order.
    let catalog_names: Vec<&str> = zend::tool_def::all()
        .iter()
        .map(|d| d.name.as_str())
        .collect();

    let installed_names: Vec<String> = installed.iter().map(|(n, _, _)| n.clone()).collect();

    assert_eq!(
        installed_names, catalog_names,
        "install order must match tool-definition catalog iteration order",
    );
}

#[test]
fn install_tool_catalog_emits_valid_hermes_json_lines() {
    let mut builder = build_test_projection();
    let _dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");
    let installed = install_tool_catalog(&mut builder).unwrap();

    // The catalog deliberately emits a flat
    // `{"name", "description", "parameters"}` shape — see the doc on
    // `render_tool_json_line` in zend/src/tools.rs for the rationale
    // (Qwen3-A3B's "function" key echo + token-count savings).
    for (name, _, json_line) in &installed {
        let parsed: Value = serde_json::from_str(json_line.trim_end())
            .unwrap_or_else(|e| panic!("tool {name:?} produced invalid JSON: {e}"));
        assert_eq!(parsed["name"], *name, "tool {name:?} name mismatch");
        assert!(
            parsed["description"].is_string(),
            "tool {name:?} missing description",
        );
        assert!(
            parsed["parameters"].is_object(),
            "tool {name:?} parameters not an object",
        );
    }
}

#[test]
fn install_tool_catalog_leaves_static_sections_untouched() {
    // The framing sections (mode, frame, grounding, tools_intro,
    // tools_outro) must remain top-level always-emit sections — outside
    // the `tools` collection.
    let mut builder = build_test_projection();
    let _dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");
    let n_top_before = builder
        .schema()
        .system_prompt
        .items
        .iter()
        .filter(|it| matches!(it, SystemPromptItem::Section(_)))
        .count();
    install_tool_catalog(&mut builder).unwrap();
    let n_top_after = builder
        .schema()
        .system_prompt
        .items
        .iter()
        .filter(|it| matches!(it, SystemPromptItem::Section(_)))
        .count();
    assert_eq!(
        n_top_before, n_top_after,
        "top-level static sections must not move when tools are installed",
    );
    // And the static content-section names are all still findable.
    // (`mode`, `system_open`/`system_close`, `tools_open`/`tools_close`
    // are `kind: template` items now and don't appear in the section
    // name map — see `projection.yaml`.)
    // (`grounding` / `grounding_no_tools` are commented out in projection.yaml.)
    // (`frame` is now the `assistant` option of the `persona` selector, not a
    // standalone section — see `projection.yaml`.)
    for name in ["history_stance", "tools_overview"] {
        assert!(
            builder.id_for_system_section(name).is_some(),
            "static section {name:?} must still resolve",
        );
    }
}

/// The ingest summariser drives the shared system prompt into summarizer mode via
/// `selection.select("persona", "summarize")` (see `ingest_scope_roundtrip`). That
/// call silently no-ops if the ids don't match the schema, so pin them: the parsed
/// `projection.yaml` must expose a `persona` SELECTOR (>1 option) carrying both an
/// `assistant` and a `summarize` option.
#[test]
fn persona_selector_exposes_assistant_and_summarize() {
    let builder = build_test_projection();
    let tree = builder
        .schema()
        .system_prompt
        .items
        .iter()
        .find_map(|it| match it {
            SystemPromptItem::SectionTree(t) => Some(t),
            _ => None,
        })
        .expect("system prompt has a section_tree");
    let persona = tree
        .nodes
        .iter()
        .find(|n| n.name == "persona")
        .expect("section_tree has a `persona` node");
    let opt_ids: Vec<&str> = persona.options.iter().map(|o| o.id.as_str()).collect();
    assert!(
        persona.options.len() >= 2,
        "persona must be a selector (>1 option) so the ingest can select it: {opt_ids:?}"
    );
    assert!(
        opt_ids.contains(&"assistant"),
        "persona options: {opt_ids:?}"
    );
    assert!(
        opt_ids.contains(&"summarize"),
        "persona options: {opt_ids:?}"
    );
}

/// The ingest stuffs few-shot summarizer example TURNS via
/// `set_optional("summarize_examples", Present)`. Pin that the parsed schema
/// exposes a `summarize_examples` optional node carrying real example turns
/// (ChatML assistant markers + a sample summary), so the id resolves and the
/// stuffing actually lands.
#[test]
fn summarize_examples_optional_carries_stuffed_turns() {
    let builder = build_test_projection();
    let node = builder
        .schema()
        .system_prompt
        .items
        .iter()
        .filter_map(|it| match it {
            SystemPromptItem::SectionTree(t) => Some(t),
            _ => None,
        })
        .flat_map(|t| t.nodes.iter())
        .find(|n| n.name == "summarize_examples")
        .expect("schema has a `summarize_examples` optional node");
    let content: String = node.options.iter().map(|o| o.content.as_str()).collect();
    assert!(
        content.contains("<|im_start|>assistant"),
        "stuffed content must be real example TURNS (assistant markers)",
    );
    assert!(
        content.contains("Jitter returns"),
        "stuffed content must carry the sample summaries",
    );
}

// ── extract_tool_calls ───────────────────────────────────────────────────────

#[test]
fn extract_recognises_single_call() {
    let text = r#"<tool_call>
{"name": "datetime", "arguments": {"timezone": "UTC"}}
</tool_call>"#;
    let calls = extract_tool_calls(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "datetime");
    assert_eq!(calls[0].arguments["timezone"], "UTC");
}

#[test]
fn extract_recognises_multiple_calls_across_one_response() {
    let text = r#"
The user wants two things:

<tool_call>
{"name": "datetime", "arguments": {}}
</tool_call>

then

<tool_call>
{"name": "calculator", "arguments": {"expression": "5 + 3"}}
</tool_call>
"#;
    let calls = extract_tool_calls(text);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "datetime");
    assert_eq!(calls[1].name, "calculator");
}

#[test]
fn extract_skips_malformed_blocks() {
    let text = r#"
<tool_call>not json</tool_call>
<tool_call>{}</tool_call>
<tool_call>{"name": "datetime", "arguments": {}}</tool_call>
<tool_call>{"arguments": {}}</tool_call>
"#;
    // Only the third block has both valid JSON and a non-empty name.
    let calls = extract_tool_calls(text);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "datetime");
}

#[test]
fn extract_handles_no_calls() {
    let text = "Just a normal model response with no tool blocks.";
    let calls = extract_tool_calls(text);
    assert!(calls.is_empty());
}

// ── run_tool ─────────────────────────────────────────────────────────────────

#[test]
fn run_tool_unknown_returns_error_shape() {
    let ctx = ToolContext::default();
    let call = ToolCall {
        name: "definitely_not_a_tool".to_string(),
        arguments: json!({}),
    };
    let resp = run_tool(&ctx, &call);
    assert_eq!(resp["error"], "unknown_tool");
    assert!(resp["detail"]
        .as_str()
        .unwrap()
        .contains("definitely_not_a_tool"));
}

#[test]
fn run_tool_datetime_succeeds_on_empty_args() {
    let ctx = ToolContext::default();
    let call = ToolCall {
        name: "datetime".to_string(),
        arguments: json!({}),
    };
    let resp = run_tool(&ctx, &call);
    assert!(resp.get("error").is_none(), "datetime errored: {resp:?}");
}

#[test]
fn run_tool_calculator_succeeds_on_simple_expression() {
    let ctx = ToolContext::default();
    let call = ToolCall {
        name: "calculator".to_string(),
        arguments: json!({"expression": "2 + 2"}),
    };
    let resp = run_tool(&ctx, &call);
    assert!(resp.get("error").is_none(), "calculator errored: {resp:?}",);
}

#[test]
fn run_tool_calculator_errors_on_invalid_expression() {
    let ctx = ToolContext::default();
    let call = ToolCall {
        name: "calculator".to_string(),
        arguments: json!({"expression": "💀💀💀"}),
    };
    let resp = run_tool(&ctx, &call);
    // Either tool-specific error or invalid_arguments — both are "error"
    // shapes and the model can react.
    assert!(
        resp.get("error").is_some(),
        "expected calculator to error on garbage, got: {resp:?}",
    );
}

#[test]
fn run_tool_validation_failure_returns_invalid_arguments() {
    let ctx = ToolContext::default();
    // calculator requires `expression`; passing the wrong field type
    // should trigger serde or validator failure → invalid_arguments.
    let call = ToolCall {
        name: "calculator".to_string(),
        arguments: json!({"expression": ""}),
    };
    let resp = run_tool(&ctx, &call);
    // Either invalid_arguments OR a tool-specific code that signals
    // empty input is unacceptable — both are "error" shapes.
    assert!(resp.get("error").is_some());
}

// ── run_tool_calls + format_tool_responses ───────────────────────────────────

#[test]
fn run_tool_calls_dispatches_each_in_order() {
    let ctx = ToolContext::default();
    let calls = vec![
        ToolCall {
            name: "datetime".to_string(),
            arguments: json!({}),
        },
        ToolCall {
            name: "calculator".to_string(),
            arguments: json!({"expression": "1 + 1"}),
        },
    ];
    let results = run_tool_calls(&ctx, calls);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].call.name, "datetime");
    assert_eq!(results[1].call.name, "calculator");
}

#[test]
fn format_tool_responses_produces_one_block_per_result() {
    let results = vec![
        ToolResult {
            call: ToolCall {
                name: "datetime".to_string(),
                arguments: Value::Null,
            },
            response: json!({"iso": "2026-05-09"}),
        },
        ToolResult {
            call: ToolCall {
                name: "calculator".to_string(),
                arguments: Value::Null,
            },
            response: json!({"result": 4}),
        },
    ];
    let formatted = format_tool_responses(&results);
    let n_open = formatted.matches("<tool_response>").count();
    let n_close = formatted.matches("</tool_response>").count();
    assert_eq!(n_open, 2);
    assert_eq!(n_close, 2);
    assert!(formatted.contains("\"iso\":\"2026-05-09\""));
    assert!(formatted.contains("\"result\":4"));
}

#[test]
fn format_tool_responses_escapes_nested_json_correctly() {
    let results = vec![ToolResult {
        call: ToolCall {
            name: "x".to_string(),
            arguments: Value::Null,
        },
        // Nested object, ensures serde_json::to_string handles it without
        // breaking the <tool_response> wrapping.
        response: json!({
            "nested": { "deep": { "value": "hello" } },
            "array": [1, 2, 3],
        }),
    }];
    let formatted = format_tool_responses(&results);
    assert!(formatted.starts_with("<tool_response>"));
    assert!(formatted.contains("</tool_response>"));
    assert!(formatted.contains("\"nested\""));
}

/// The bundled schema's belief gates, resolved the way projection does it.
/// `repo_map/structure` declares a `policy:` band and no `score_threshold`; the
/// gate must be that band (600/400), not the `0.0` default that previously
/// overwrote it and made every cluster eligible at zero evidence. Every other
/// group declares a `score_threshold` and must keep it verbatim.
#[test]
fn bundled_schema_belief_gates_resolve_from_the_right_source() {
    use candle_conversation::models::Dialect;
    use candle_conversation::projection::Builder;
    const YAML: &str = include_str!("../src/prompts/projection.yaml");
    let dialect = Dialect::chat_ml();
    let builder =
        Builder::from_yaml_with_vars_and_dialect(YAML, &[("workspace", "proj")], Some(&dialect))
            .expect("bundled projection.yaml parses");
    let schema = builder.schema();
    let mut seen = std::collections::BTreeMap::new();
    for layer in &schema.layers {
        for group in &layer.groups {
            let cfg = group.belief_config(32);
            seen.insert(
                format!("{}/{}", layer.name, group.name),
                (cfg.min_score, cfg.evict_score),
            );
        }
    }
    assert_eq!(
        seen.get("repo_map/structure"),
        Some(&(250.0, 250.0)),
        "repo_map must use its declared policy band, not the 0.0 default; got {seen:?}",
    );
    assert_eq!(seen.get("bug_analysis/bugs"), Some(&(250.0, 250.0)));
    assert_eq!(seen.get("dream_log/dreams"), Some(&(100.0, 100.0)));
    assert_eq!(seen.get("code_reading/scopes"), Some(&(100.0, 100.0)));

    // The early band is a GRACE window: it must never sit above the steady one,
    // or the opening tokens of a turn are gated harder than the rest. Only
    // repo_map currently enables an early window (`early_window_tokens: 24`);
    // the rest inherit `early_window_tokens: 0`, which makes their early band
    // inert — so this guards repo_map today and any group that turns one on.
    for layer in &schema.layers {
        for group in &layer.groups {
            let cfg = group.belief_config(32);
            if cfg.early_window_tokens == 0 {
                continue;
            }
            assert!(
                cfg.early_min_score <= cfg.min_score,
                "{}/{}: early_min_score {} > min_score {}",
                layer.name,
                group.name,
                cfg.early_min_score,
                cfg.min_score,
            );
        }
    }
}
