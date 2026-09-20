//! **A closed set is a choice.** Where the schema names every value a field
//! can take — `true`/`false`, an enum, `null` — the grammar offers them under
//! the mask, so a model cannot spell one wrong: `True` is not repaired here,
//! it is never offered. And where the schema allows `null`, the grammar offers
//! it, so a nullable field can actually be left empty rather than forced to a
//! value.
//!
//! Driven byte by byte against a target call: prefills must match it, each
//! masked decode must allow its byte, and a free span writes it.

use std::sync::Arc;

use serde_json::{json, Value};

use crate::stencil::compile::compile;
use crate::stencil::session::{StencilAction, StencilSession};
use crate::stencil::tool_call::{compile_tool_call_tree, ToolCallEnvelope, ToolSpec};
use crate::stencil::tree::StencilTree;
use crate::stencil::vocab::{TestVocab, TokenId, Vocab};

use super::{json_body, parse};

/// Every way a schema says a field is a closed set or may be `null`.
fn settings() -> ToolSpec {
    ToolSpec::from_json_schema(
        "settings",
        &json!({
            "type": "object",
            "properties": {
                "enabled": {"type": ["boolean", "null"]},
                "mode": {"enum": ["fast", "slow", null]},
                "label": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "limit": {"type": "string", "nullable": true},
                "strict": {"type": "boolean"},
                "level": {"type": "string", "enum": ["low", "high"]},
                "switches": {"type": "array", "items": {"type": "boolean"}},
                "modes": {"type": "array", "items": {"enum": ["a", "ab"]}},
                "maybe": {"type": "array", "items": {"type": ["boolean", "null"]}},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                "range": {
                    "anyOf": [
                        {"type": "object",
                         "properties": {"start": {"type": "integer"}},
                         "required": ["start"]},
                        {"type": "null"}
                    ]
                }
            },
            "required": [
                "enabled", "mode", "label", "limit", "strict", "level",
                "switches", "modes", "maybe", "tags", "range"
            ]
        }),
    )
}

fn tree(v: &TestVocab) -> Arc<StencilTree> {
    let spec = compile_tool_call_tree(&[settings()], &ToolCallEnvelope::qwen3()).unwrap();
    Arc::new(compile(&spec, v).unwrap())
}

/// Where a target stopped being writable.
#[derive(Debug, PartialEq)]
enum Refused {
    /// The grammar masked out this byte.
    Masked { at: usize, rest: String },
    /// The grammar wrote something else here.
    Prefilled { at: usize, wrote: String },
}

/// Write `target` through the grammar, or say where it would not go.
fn write(target: &str) -> Result<String, Refused> {
    let v = TestVocab::new();
    let mut session = StencilSession::new(tree(&v));
    let bytes = target.as_bytes();
    let mut at = 0usize;
    let mut out: Vec<TokenId> = Vec::new();
    loop {
        match session.next_action() {
            StencilAction::Prefill(toks) => {
                let wrote = v.decode(&toks);
                if !bytes[at..].starts_with(&wrote) {
                    return Err(Refused::Prefilled {
                        at,
                        wrote: String::from_utf8_lossy(&wrote).into_owned(),
                    });
                }
                at += wrote.len();
                out.extend(toks);
            }
            StencilAction::MaskedDecode(set) => {
                let b = *bytes.get(at).expect("the target ended inside the call");
                if !set.contains(b as TokenId) {
                    return Err(Refused::Masked {
                        at,
                        rest: target[at..].to_string(),
                    });
                }
                session.observe(b as TokenId, &[b]).unwrap();
                out.push(b as TokenId);
                at += 1;
            }
            StencilAction::FreeDecode { .. } => {
                let b = bytes[at];
                session.observe(b as TokenId, &[b]).unwrap();
                out.push(b as TokenId);
                at += 1;
            }
            StencilAction::Exit => break,
        }
    }
    Ok(String::from_utf8_lossy(&v.decode(&out)).into_owned())
}

/// A whole `settings` call, with `values` written in schema order.
fn call(values: [&str; 11]) -> String {
    let keys = [
        "enabled", "mode", "label", "limit", "strict", "level", "switches", "modes", "maybe",
        "tags", "range",
    ];
    let args: Vec<String> = keys
        .iter()
        .zip(values)
        .map(|(k, v)| format!("\"{k}\": {v}"))
        .collect();
    format!(
        "<tool_call>\n{{\"name\": \"settings\", \"arguments\": {{{}}}}}\n</tool_call>",
        args.join(", ")
    )
}

/// The ordinary values: every field writable, and the call means them.
const TYPICAL: [&str; 11] = [
    "true",
    "\"fast\"",
    "\"note\"",
    "\"10\"",
    "false",
    "\"high\"",
    "[true, false]",
    "[\"ab\", \"a\"]",
    "[true, null]",
    "[\"x\"]",
    "{\"start\": 3}",
];

fn arguments(text: &str) -> Value {
    parse(text)["arguments"].clone()
}

#[test]
fn every_field_takes_its_ordinary_values() {
    let target = call(TYPICAL);
    let text = write(&target).unwrap();
    assert_eq!(text, target);
    serde_json::from_str::<Value>(json_body(&text)).unwrap();
    assert_eq!(
        arguments(&text),
        json!({
            "enabled": true, "mode": "fast", "label": "note", "limit": "10", "strict": false,
            "level": "high", "switches": [true, false], "modes": ["ab", "a"],
            "maybe": [true, null], "tags": ["x"], "range": {"start": 3}
        })
    );
}

/// **Every way of saying "may be null" offers `null`**: a `null` in the type
/// list, in the enum, `nullable: true`, and `anyOf` with `{"type": "null"}` —
/// for a boolean, an enum, a string, an array and an object alike.
#[test]
fn every_nullable_field_can_be_null() {
    let target = call([
        "null",
        "null",
        "null",
        "null",
        "true",
        "\"low\"",
        "[]",
        "[]",
        "[null, null]",
        "null",
        "null",
    ]);
    let text = write(&target).unwrap_or_else(|r| panic!("{r:?}"));
    let args = arguments(&text);
    for key in ["enabled", "mode", "label", "limit", "tags", "range"] {
        assert_eq!(args[key], Value::Null, "{key}");
    }
    assert_eq!(args["maybe"], json!([null, null]));
}

/// **And a field that may not be null is not offered it.** Each of these puts
/// `null` where the schema does not allow one. A choice refuses it under the
/// mask; a value with one way to begin (`[`, `"`) has that opening prefilled,
/// and the target no longer matches — either way, at the field, never after.
#[test]
fn a_field_that_may_not_be_null_refuses_it() {
    for (field, key) in [
        (4usize, "strict"),
        (5, "level"),
        (6, "switches"),
        (7, "modes"),
    ] {
        let mut values = TYPICAL;
        values[field] = "null";
        let target = call(values);
        let value_at = target.find(&format!("\"{key}\": null")).unwrap() + key.len() + 4;
        match write(&target) {
            Err(Refused::Masked { at, rest }) => {
                assert!(rest.starts_with("null") && at == value_at, "{key}: {rest}")
            }
            Err(Refused::Prefilled { at, wrote }) => {
                assert!(at <= value_at, "{key}: refused late, writing {wrote:?}")
            }
            Ok(text) => panic!("{key}: `null` was accepted: {text}"),
        }
    }
    // An element that may not be null either.
    let mut values = TYPICAL;
    values[6] = "[true, null]";
    assert!(
        matches!(write(&call(values)), Err(Refused::Masked { rest, .. }) if rest.starts_with("null")),
        "a boolean array took a null element"
    );
}

/// **A closed set cannot be misspelled.** The wrong spelling of a literal, a
/// value outside the enum, a quoted boolean — each is refused under the mask at
/// the byte where it leaves the set, so no repair is ever needed.
#[test]
fn a_closed_set_refuses_every_other_spelling() {
    let cases: &[(usize, &str, &str)] = &[
        (4, "True", "True"),
        (4, "TRUE", "TRUE"),
        (4, "\"true\"", "\"true\""),
        (4, "1", "1"),
        (0, "None", "None"),
        (0, "Null", "Null"),
        (1, "\"medium\"", "medium\""),
        (1, "\"Fast\"", "Fast\""),
        (1, "fast", "fast"),
        // A prefix of `"high"` runs until it has to end: at its quote.
        (5, "\"hi\"", "\", \"switches\""),
        (5, "\"lo\"", "\", \"switches\""),
        (5, "\"highest\"", "est\""),
        (6, "[True]", "True]"),
        (6, "[1, 0]", "1, 0]"),
        (7, "[\"b\"]", "b\"]"),
        (7, "[\"abc\"]", "c\"]"),
        (8, "[False, None]", "False, None]"),
    ];
    for &(field, written, refused_at) in cases {
        let mut values = TYPICAL;
        values[field] = written;
        match write(&call(values)) {
            Err(Refused::Masked { rest, .. }) => assert!(
                rest.starts_with(refused_at),
                "{written:?} was refused at {rest:?}, not at {refused_at:?}"
            ),
            other => panic!("{written:?} was accepted: {other:?}"),
        }
    }
}

/// An enum value that is a prefix of another stays distinguishable: `"a"` and
/// `"ab"` differ at the closing quote, which rides on each arm.
#[test]
fn an_enum_value_that_prefixes_another_is_still_choosable() {
    for modes in ["[\"a\"]", "[\"ab\"]", "[\"a\", \"ab\", \"a\"]"] {
        let mut values = TYPICAL;
        values[7] = modes;
        let text = write(&call(values)).unwrap_or_else(|r| panic!("{modes}: {r:?}"));
        assert_eq!(
            arguments(&text)["modes"],
            serde_json::from_str::<Value>(modes).unwrap()
        );
    }
}

/// An array of booleans is a choice at every element — `true`, `false`, or the
/// close — so it runs to any length up to the bound, and no further.
#[test]
fn an_array_of_booleans_is_a_choice_at_every_element() {
    use crate::stencil::tool_call::MAX_ARRAY_ELEMENTS;

    let long: Vec<&str> = (0..MAX_ARRAY_ELEMENTS)
        .map(|i| if i % 3 == 0 { "true" } else { "false" })
        .collect();
    let mut values = TYPICAL;
    let full = format!("[{}]", long.join(", "));
    values[6] = &full;
    let text = write(&call(values)).unwrap();
    assert_eq!(
        arguments(&text)["switches"].as_array().map(Vec::len),
        Some(MAX_ARRAY_ELEMENTS)
    );

    // Past the bound the grammar writes the close itself.
    let over = format!("[{}, true]", long.join(", "));
    values[6] = &over;
    assert!(
        matches!(write(&call(values)), Err(Refused::Prefilled { wrote, .. }) if wrote.starts_with(']')),
        "an element past the bound was accepted"
    );
}
