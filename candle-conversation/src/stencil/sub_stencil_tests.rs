//! The standalone invoke-body sub-stencil (effector design §11): a `{ … }` object
//! compiled from a body schema's parameters, simulated on its own and spliced into
//! a turn's call by `compile_action_loop_with_body`.

use std::sync::Arc;

use serde_json::{json, Value};

use super::compile::compile;
use super::sim::{lowest_arm_policy, simulate, Oracle, SimRun};
use super::tool_call::{
    compile_action_loop, compile_action_loop_with_body, compile_invoke_body_tree, Param,
    ToolCallEnvelope, ToolSpec, INVOKE_BODY_TREE_LABEL,
};
use super::tree::StencilTree;
use super::vocab::{TestVocab, TokenId};
use super::{NodeSpec, Observe, TreeSpec};

const FREE: TokenId = b'x' as TokenId;

fn params(schema: &Value) -> Vec<Param> {
    ToolSpec::from_json_schema("body", schema).params
}

fn tree(schema: &Value, v: &TestVocab) -> Arc<StencilTree> {
    let spec = compile_invoke_body_tree(&params(schema)).unwrap();
    Arc::new(compile(&spec, v).unwrap())
}

fn run_lowest(schema: &Value, v: &TestVocab) -> (String, SimRun) {
    let sim = simulate(tree(schema, v), v, lowest_arm_policy(FREE), 20_000)
        .unwrap_or_else(|e| panic!("simulation failed: {e}"));
    let text = sim.text(v);
    assert!(
        !sim.observes.contains(&Observe::Bailed),
        "the walk bailed: {text:?}"
    );
    (text, sim)
}

#[test]
fn tree_carries_the_invoke_body_label_and_a_closing_brace_bail() {
    let spec = compile_invoke_body_tree(&params(&json!({
        "type": "object",
        "properties": {"on": {"type": "boolean"}}
    })))
    .unwrap();
    assert_eq!(spec.label, INVOKE_BODY_TREE_LABEL);
    assert_eq!(spec.bail, "}");
}

#[test]
fn empty_schema_compiles_to_an_empty_object_and_nothing_else() {
    let v = TestVocab::new();
    let sim = simulate(
        tree(&json!({"type": "object", "properties": {}}), &v),
        &v,
        Oracle::Scripted(Vec::new()),
        100,
    )
    .unwrap();
    assert_eq!(sim.text(&v), "{}");
    assert!(
        sim.observes.is_empty(),
        "no decode happens for an empty body"
    );
}

#[test]
fn a_boolean_field_is_forced_to_a_json_boolean() {
    let v = TestVocab::new();
    let (text, _) = run_lowest(
        &json!({
            "type": "object",
            "properties": {"on": {"type": "boolean"}},
            "required": ["on"]
        }),
        &v,
    );
    let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
    assert!(parsed["on"].is_boolean(), "got {text:?}");
}

#[test]
fn a_string_enum_field_takes_only_a_listed_value() {
    let v = TestVocab::new();
    let (text, _) = run_lowest(
        &json!({
            "type": "object",
            "properties": {"verb": {"type": "string", "enum": ["open", "close", "lock"]}},
            "required": ["verb"]
        }),
        &v,
    );
    let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
    let verb = parsed["verb"].as_str().expect("verb is a string");
    assert!(
        ["open", "close", "lock"].contains(&verb),
        "enum value outside the set: {verb:?}"
    );
}

#[test]
fn a_nested_object_is_recursed_and_the_whole_body_parses() {
    let v = TestVocab::new();
    let (text, _) = run_lowest(
        &json!({
            "type": "object",
            "properties": {
                "target": {
                    "type": "object",
                    "properties": {"hard": {"type": "boolean"}},
                    "required": ["hard"]
                },
                "note": {"type": "string"}
            },
            "required": ["target", "note"]
        }),
        &v,
    );
    let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
    assert!(parsed["target"]["hard"].is_boolean(), "got {text:?}");
    assert!(parsed["note"].is_string(), "got {text:?}");
}

#[test]
fn a_body_ends_the_walk_at_its_closing_brace() {
    let v = TestVocab::new();
    let (text, _) = run_lowest(
        &json!({
            "type": "object",
            "properties": {"on": {"type": "boolean"}},
            "required": ["on"]
        }),
        &v,
    );
    assert!(text.starts_with('{') && text.ends_with('}'), "got {text:?}");
}

#[test]
fn splicing_no_body_is_byte_identical_to_the_plain_action_loop() {
    let tools = vec![ToolSpec::from_json_schema(
        "invoke",
        &json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "body": {"type": "object"}},
            "required": ["path", "body"]
        }),
    )];
    let env = ToolCallEnvelope::qwen3();
    let plain = compile_action_loop(&tools, &env, 2, "<|im_end|>", None).unwrap();
    let spliced = compile_action_loop_with_body(&tools, &env, 2, "<|im_end|>", None, None).unwrap();
    assert_eq!(plain.nodes.len(), spliced.nodes.len());
    assert_eq!(plain.root, spliced.root);
}

#[test]
fn splicing_a_body_types_that_tools_parameter_and_leaves_others_alone() {
    let tools = vec![
        ToolSpec::from_json_schema(
            "invoke",
            &json!({
                "type": "object",
                "properties": {"path": {"type": "string"}, "body": {"type": "object"}},
                "required": ["path", "body"]
            }),
        ),
        ToolSpec::from_json_schema(
            "look",
            &json!({
                "type": "object",
                "properties": {"body": {"type": "object"}},
                "required": ["body"]
            }),
        ),
    ];
    let env = ToolCallEnvelope::qwen3();
    let body = compile_invoke_body_tree(&params(&json!({
        "type": "object",
        "properties": {"on": {"type": "boolean"}},
        "required": ["on"]
    })))
    .unwrap();
    let plain = compile_action_loop(&tools, &env, 1, "<|im_end|>", None).unwrap();
    let spliced = compile_action_loop_with_body(
        &tools,
        &env,
        1,
        "<|im_end|>",
        None,
        Some(("invoke", "body", &body)),
    )
    .unwrap();
    assert_ne!(
        plain.nodes.len(),
        spliced.nodes.len(),
        "the typed body adds nodes to the armed tool's call"
    );
    let offers_boolean = |spec: &TreeSpec| {
        spec.nodes.iter().any(|n| {
            matches!(n, NodeSpec::Branch { arms }
                if arms.iter().any(|(label, _)| label == " true"))
        })
    };
    assert!(
        !offers_boolean(&plain),
        "free-typed values offer no boolean choice"
    );
    assert!(
        offers_boolean(&spliced),
        "the body's boolean choice is spliced in"
    );
    compile(&spliced, &TestVocab::new()).expect("the spliced turn grammar compiles");
}
