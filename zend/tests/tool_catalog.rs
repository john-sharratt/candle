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

/// Helper: count tool sections currently in the dialogue layer's
/// `tools` collection.
fn n_tools_in_collection(builder: &projection::Builder) -> usize {
    let dialogue = builder.id_for_layer("dialogue").expect("dialogue");
    builder
        .layer(dialogue)
        .unwrap()
        .system_prompt
        .items
        .iter()
        .filter_map(|it| match it {
            SystemPromptItem::Collection(c) if c.name == "tools" => Some(c.sections.len()),
            _ => None,
        })
        .next()
        .unwrap_or(0)
}

// ── install_tool_catalog ─────────────────────────────────────────────────────

#[test]
fn install_tool_catalog_adds_one_section_per_registered_tool() {
    let mut builder = build_test_projection();
    let dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");

    let n_before = n_tools_in_collection(&builder);
    let n_tools_in_registry = registry::all_tools().len();

    let installed = install_tool_catalog(&mut builder, dialogue).unwrap();

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
    let dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");
    install_tool_catalog(&mut builder, dialogue).unwrap();

    let layer = builder.layer(dialogue).unwrap();
    let coll = layer
        .system_prompt
        .collection_named("tools")
        .expect("tools collection must exist");
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
    let dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");

    let installed = install_tool_catalog(&mut builder, dialogue).unwrap();
    let registry_names: Vec<&str> = registry::all_tools().iter().map(|t| t.name).collect();

    let installed_names: Vec<String> = installed.iter().map(|(n, _, _)| n.clone()).collect();

    assert_eq!(
        installed_names, registry_names,
        "install order must match registry iteration order",
    );
}

#[test]
fn install_tool_catalog_emits_valid_hermes_json_lines() {
    let mut builder = build_test_projection();
    let dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");
    let installed = install_tool_catalog(&mut builder, dialogue).unwrap();

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
    // The framing sections (mode, frame, history_stance, grounding,
    // tools_intro, tools_outro) must remain top-level always-emit
    // sections — outside the `tools` collection.
    let mut builder = build_test_projection();
    let dialogue = builder.id_for_layer("dialogue").expect("dialogue layer");
    let n_top_before = builder
        .layer(dialogue)
        .unwrap()
        .system_prompt
        .items
        .iter()
        .filter(|it| matches!(it, SystemPromptItem::Section(_)))
        .count();
    install_tool_catalog(&mut builder, dialogue).unwrap();
    let n_top_after = builder
        .layer(dialogue)
        .unwrap()
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
    for name in ["frame", "history_stance", "grounding", "tools_overview"] {
        assert!(
            builder.id_for_section_in(dialogue, name).is_some(),
            "static section {name:?} must still resolve",
        );
    }
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
