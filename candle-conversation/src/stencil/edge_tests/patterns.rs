//! Values held to a pattern, against the failures a grammar can cause rather
//! than prevent: a walk with nothing legal to write, a value that can never
//! close, a turn that does not end.

use std::sync::Arc;

use regex_automata::meta::Regex;
use serde_json::{json, Value};

use super::{parse, Rng, EOS};
use crate::stencil::compile::compile;
use crate::stencil::mask::AllowedSet;
use crate::stencil::session::Observe;
use crate::stencil::sim::{simulate, Oracle, SimRun};
use crate::stencil::tool_call::{compile_tool_call_tree, ToolCallEnvelope, ToolSpec};
use crate::stencil::tree::StencilTree;
use crate::stencil::vocab::{TestVocab, TokenId};

const HEX: &str = "[0-9a-f]{4}";
const SCHEME: &str = "https?://[a-z.]+";
const SEGMENT: &str = "[a-z]+(/[a-z]+)*";

/// Tokens that cut a value the ways a real tokenizer does: merged with its
/// quotes, with the separator after it, across a pattern's literal.
const MERGED: &[&str] = &[
    " \"", " \"a", " \"http", "\",", "\"}", "ab", "12", "://", "https",
];

fn vocab() -> TestVocab {
    MERGED
        .iter()
        .enumerate()
        .fold(TestVocab::new(), |v, (i, s)| {
            v.with_special(s, 300 + i as TokenId)
        })
}

fn tree(v: &TestVocab) -> Arc<StencilTree> {
    let tool = ToolSpec::from_json_schema(
        "fetch",
        &json!({
            "type": "object",
            "properties": {
                "id": {"type": "string", "pattern": HEX},
                "url": {"type": ["string", "null"], "pattern": SCHEME},
                "paths": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "string", "pattern": SEGMENT},
                },
                "note": {"type": "string"},
            },
            "required": ["id", "paths"],
        }),
    );
    let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
    Arc::new(compile(&spec, v).unwrap())
}

/// A model picking anything the mask leaves, and hostile bytes and EOS where
/// nothing masks it.
fn adversary(seed: u64) -> Oracle {
    let mut rng = Rng(seed);
    Oracle::Policy(Box::new(
        move |allowed: Option<&AllowedSet>| match allowed {
            Some(set) => {
                assert!(!set.is_empty(), "a mask left nothing to write");
                set.tokens()[rng.below(set.len())]
            }
            None => match rng.below(10) {
                0 => EOS,
                1 => b'"' as TokenId,
                _ => b'x' as TokenId,
            },
        },
    ))
}

fn held(value: &Value, pattern: &str) -> Result<(), String> {
    let text = value.as_str().ok_or("not a string")?;
    let whole = Regex::new(&format!("^(?:{pattern})$")).unwrap();
    match whole.is_match(text) {
        true => Ok(()),
        false => Err(format!("{text:?} is not {pattern}")),
    }
}

fn check(run: &SimRun, v: &TestVocab, seed: u64) {
    let text = run.text(v);
    assert!(
        !run.observes.contains(&Observe::Bailed),
        "seed {seed}: bailed: {text:?}"
    );
    let call = parse(&text);
    let args = &call["arguments"];
    held(&args["id"], HEX).unwrap_or_else(|why| panic!("seed {seed}: {why}"));
    if args.get("url").is_some() {
        held(&args["url"], SCHEME).unwrap_or_else(|why| panic!("seed {seed}: {why}"));
    }
    for path in args["paths"].as_array().unwrap() {
        held(path, SEGMENT).unwrap_or_else(|why| panic!("seed {seed}: {why}"));
    }
}

/// **Whatever a model does under the masks, the call ends, parses and matches**:
/// no mask is ever empty, no walk runs away, no value breaks its pattern.
#[test]
fn every_walk_ends_in_a_call_that_holds_to_its_patterns() {
    let v = vocab();
    let tree = tree(&v);
    for seed in 0..3000 {
        let run = simulate(tree.clone(), &v, adversary(seed), 40_000)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        check(&run, &v, seed);
    }
}

/// **A value that never stops runs into the span's limit and closes there**: a
/// long language cannot hold a turn open. What it closed on may be short of the
/// pattern, which its tool refuses; the turn still ends.
#[test]
fn a_value_that_never_stops_is_closed_by_its_limit() {
    let v = TestVocab::new();
    let tool = ToolSpec::from_json_schema(
        "fetch",
        &json!({
            "type": "object",
            "properties": {"path": {"type": "string", "pattern": "[a-z]+", "maxLength": 8}},
            "required": ["path"],
        }),
    );
    let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
    let tree = Arc::new(compile(&spec, &v).unwrap());
    let never_closes = Oracle::Policy(Box::new(|allowed: Option<&AllowedSet>| {
        let set = allowed.expect("the value is masked");
        let quote = b'"' as TokenId;
        let letter = set.tokens().iter().find(|&&t| t != quote);
        *letter.unwrap_or(&quote)
    }));
    let run = simulate(tree, &v, never_closes, 1_000).expect("the walk ended");
    let call = parse(&run.text(&v));
    assert_eq!(call["arguments"]["path"], "aaaaaaaa");
}

/// **A pattern that cannot be held to leaves the value free**, and the call
/// still decodes: a bad constraint costs the constraint, never the turn.
#[test]
fn a_pattern_that_cannot_build_leaves_the_value_free() {
    let v = TestVocab::new();
    let tool = ToolSpec::from_json_schema(
        "fetch",
        &json!({
            "type": "object",
            "properties": {"path": {"type": "string", "pattern": "("}},
            "required": ["path"],
        }),
    );
    let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
    let tree = Arc::new(compile(&spec, &v).unwrap());
    let run = simulate(tree, &v, adversary(1), 10_000).expect("the walk ended");
    parse(&run.text(&v));
}
