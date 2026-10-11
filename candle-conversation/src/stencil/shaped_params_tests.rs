//! An enum parameter whose values bring their own sibling fields
//! ([`Param::shapes`]) — the `invoke` call's `url`, each URL carrying the typed
//! `body` its endpoint accepts.
//!
//! What is pinned here: every path of the grammar pairs a URL with a body of
//! *that URL's* schema and no other; a body a URL requires cannot be skipped;
//! and the tree grows with the number of distinct schemas, not with the number
//! of URLs.

use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::Value;

use super::compile::compile;
use super::invoke_body_tests::{all_paths, extract_call, QUOTE};
use super::sim::SimRun;
use super::spec::TreeSpec;
use super::tool_call::{compile_action_loop, Param, ParamType, ToolCallEnvelope, ToolSpec};
use super::tree::StencilTree;
use super::vocab::TestVocab;

// ── fixtures ─────────────────────────────────────────────────────────────────

fn param(name: &str, ty: ParamType, required: bool) -> Param {
    Param {
        name: name.to_string(),
        ty,
        required,
        enum_values: None,
        items: None,
        min_items: 0,
        max_items: None,
        properties: None,
        nullable: false,
        minimum: None,
        max_tokens: None,
        requires: Vec::new(),
        shapes: Vec::new(),
    }
}

fn enum_param(name: &str, values: &[&str]) -> Param {
    Param {
        enum_values: Some(values.iter().map(|v| v.to_string()).collect()),
        ..param(name, ParamType::String, true)
    }
}

/// A `body` object argument with the given fields.
fn body(fields: Vec<Param>) -> Param {
    Param {
        properties: Some(fields),
        ..param("body", ParamType::Object, true)
    }
}

/// The `invoke` tool: one required `url` enum, each value bringing its body.
fn invoke_tool(urls: &[(String, Vec<Param>)]) -> ToolSpec {
    let mut url = param("url", ParamType::String, true);
    url.enum_values = Some(urls.iter().map(|(u, _)| u.clone()).collect());
    url.shapes = urls
        .iter()
        .map(|(u, fields)| (u.clone(), vec![body(fields.clone())]))
        .collect();
    ToolSpec {
        name: "invoke".to_string(),
        params: vec![url],
    }
}

fn spec_of(tool: ToolSpec) -> TreeSpec {
    compile_action_loop(&[tool], &ToolCallEnvelope::qwen3(), 1, "<|im_end|>", None)
        .expect("the catalog compiles")
}

fn lowered(spec: &TreeSpec, v: &TestVocab) -> StencilTree {
    compile(spec, v).expect("the spec lowers")
}

/// `(url, body)` of every path of a one-call `invoke` turn.
fn pairs(urls: &[(String, Vec<Param>)], v: &TestVocab) -> Vec<(String, Value)> {
    let tree = Arc::new(lowered(&spec_of(invoke_tool(urls)), v));
    all_paths(&tree, v, QUOTE)
        .iter()
        .map(|run: &SimRun| {
            let text = run.text(v);
            let call: Value = serde_json::from_str(extract_call(&text))
                .unwrap_or_else(|e| panic!("call is not JSON: {text:?}: {e}"));
            assert_eq!(call["name"], "invoke", "{text:?}");
            (
                call["arguments"]["url"]
                    .as_str()
                    .expect("a url")
                    .to_string(),
                call["arguments"]["body"].clone(),
            )
        })
        .collect()
}

fn lift_call() -> (String, Vec<Param>) {
    ("http://local/lift/1/call".to_string(), Vec::new())
}

fn lift_use() -> (String, Vec<Param>) {
    (
        "http://local/lift/1/use".to_string(),
        vec![enum_param("floor", &["command", "port"])],
    )
}

fn chronicle_add() -> (String, Vec<Param>) {
    (
        "http://local/chronicle/2/add_entry".to_string(),
        vec![
            param("to", ParamType::String, true),
            param("what", ParamType::String, true),
        ],
    )
}

// ── what a URL admits ────────────────────────────────────────────────────────

#[test]
fn each_url_is_paired_with_exactly_its_own_body() {
    let v = TestVocab::new();
    let got: BTreeSet<(String, String)> = pairs(&[lift_call(), lift_use(), chronicle_add()], &v)
        .into_iter()
        .map(|(u, b)| (u, b.to_string()))
        .collect();
    let want: BTreeSet<(String, String)> = [
        ("http://local/lift/1/call", r#"{}"#),
        ("http://local/lift/1/use", r#"{"floor":"command"}"#),
        ("http://local/lift/1/use", r#"{"floor":"port"}"#),
        (
            "http://local/chronicle/2/add_entry",
            r#"{"to":"","what":""}"#,
        ),
    ]
    .into_iter()
    .map(|(u, b)| (u.to_string(), b.to_string()))
    .collect();
    assert_eq!(got, want);
}

#[test]
fn a_body_a_url_requires_cannot_be_skipped() {
    let v = TestVocab::new();
    for (url, b) in pairs(&[lift_call(), lift_use()], &v) {
        if url.ends_with("/use") {
            assert!(
                b.get("floor").is_some(),
                "`use` with an empty or foreign body is unreachable: {b}"
            );
        }
    }
}

#[test]
fn a_url_that_takes_no_fields_can_only_carry_the_empty_object() {
    let v = TestVocab::new();
    let calls: Vec<Value> = pairs(&[lift_call(), lift_use()], &v)
        .into_iter()
        .filter(|(u, _)| u.ends_with("/call"))
        .map(|(_, b)| b)
        .collect();
    assert_eq!(calls, vec![serde_json::json!({})]);
}

#[test]
fn a_shape_replaces_a_sibling_of_the_same_name() {
    // A free optional `body` beside the url would let a string through; the
    // shape's required object takes its place on every path.
    let v = TestVocab::new();
    let mut tool = invoke_tool(&[lift_use()]);
    tool.params.push(param("body", ParamType::String, false));
    let tree = Arc::new(lowered(&spec_of(tool), &v));
    let runs = all_paths(&tree, &v, QUOTE);
    assert_eq!(runs.len(), 2, "one path per floor");
    for run in runs {
        let text = run.text(&v);
        let call: Value = serde_json::from_str(extract_call(&text)).unwrap();
        assert!(call["arguments"]["body"].is_object(), "{text:?}");
    }
}

// ── one sub-tree per distinct shape ──────────────────────────────────────────

fn urls_with(shape: &[Param], count: usize) -> Vec<(String, Vec<Param>)> {
    (0..count)
        .map(|i| (format!("http://local/station/{i}/verb"), shape.to_vec()))
        .collect()
}

/// A body with `n` required enum fields of eight values each — large enough
/// that a duplicate would be unmistakable in the node count.
fn wide_shape(n: usize) -> Vec<Param> {
    (0..n)
        .map(|i| {
            enum_param(
                &format!("field{i}"),
                &["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"],
            )
        })
        .collect()
}

#[test]
fn urls_that_share_a_schema_share_its_nodes() {
    let shape = wide_shape(6);
    let few = spec_of(invoke_tool(&urls_with(&shape, 2)));
    let many = spec_of(invoke_tool(&urls_with(&shape, 200)));
    assert_eq!(
        few.nodes.len(),
        many.nodes.len(),
        "two hundred URLs with one schema build the schema once"
    );
}

#[test]
fn a_new_schema_adds_nodes_and_a_repeated_one_does_not() {
    let a = wide_shape(6);
    let b = wide_shape(5);
    let one: Vec<_> = urls_with(&a, 3);
    let mut two = one.clone();
    two.extend(
        urls_with(&b, 3)
            .into_iter()
            .map(|(u, f)| (format!("{u}-b"), f)),
    );
    let mut repeat = two.clone();
    repeat.extend(
        urls_with(&a, 3)
            .into_iter()
            .map(|(u, f)| (format!("{u}-again"), f)),
    );
    let n_one = spec_of(invoke_tool(&one)).nodes.len();
    let n_two = spec_of(invoke_tool(&two)).nodes.len();
    let n_repeat = spec_of(invoke_tool(&repeat)).nodes.len();
    assert!(n_two > n_one, "a second schema is built: {n_one} {n_two}");
    assert_eq!(n_two, n_repeat, "a repeated schema is not");
}

#[test]
fn a_urls_marginal_cost_does_not_depend_on_the_size_of_its_body() {
    // The lowered tree pays per URL for the arm's own bytes (the trie) and for
    // nothing behind it: a URL whose body is forty enum fields costs what a URL
    // with an empty body costs, once the body exists.
    let v = TestVocab::new();
    let grow = |shape: &[Param]| {
        let small = lowered(&spec_of(invoke_tool(&urls_with(shape, 10))), &v).len();
        let large = lowered(&spec_of(invoke_tool(&urls_with(shape, 110))), &v).len();
        (large - small) / 100
    };
    let empty = grow(&[]);
    let wide = grow(&wide_shape(40));
    assert_eq!(
        empty, wide,
        "per-URL growth: {empty} nodes for `{{}}`, {wide} for forty fields"
    );
    assert!(
        empty <= 64,
        "a URL costs its own bytes and a constant: {empty}"
    );
}

#[test]
fn a_full_world_of_urls_stays_within_a_node_budget() {
    // Two hundred stations, each with several verbs over a dozen schemas: the
    // shape of the widest live room.
    let v = TestVocab::new();
    let shapes: Vec<Vec<Param>> = (1..=12).map(wide_shape).collect();
    let mut urls = Vec::new();
    for station in 0..200 {
        for (verb, shape) in shapes.iter().enumerate().take(4 + station % 5) {
            urls.push((
                format!("http://local/station/{station}/verb{verb}"),
                shape.clone(),
            ));
        }
    }
    let spec = spec_of(invoke_tool(&urls));
    let tree = lowered(&spec, &v);
    assert!(
        tree.len() < 100_000,
        "{} URLs lowered to {} nodes",
        urls.len(),
        tree.len()
    );
}
