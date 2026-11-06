//! A call held to the values that exist (`docs/stencil_tree.md` §9.2): a
//! file's anchors for `part`, the part's lines for `from`, each depending on
//! what the call has already written.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use serde_json::json;

use super::compile::compile;
use super::driver::{StencilDriver, StepMask};
use super::held_values::{FieldHold, HeldValues, ValueHolds};
use super::mask::AllowedSet;
use super::tool_call::{compile_tool_call_tree, ToolCallEnvelope, ToolSpec};
use super::trigger::TriggerRegistry;
use super::vocab::{TestVocab, TokenId, Vocab};

const TRIGGER: TokenId = 1000;

/// An edit: a file, a part of it by anchor, and optionally a line of it.
fn edit_tool() -> ToolSpec {
    ToolSpec::from_json_schema(
        "file_edit",
        &json!({
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "path": {"type": "string"},
                "part": {"type": "string", "pattern": "[A-Za-z0-9_-]{4}"},
                "from": {"type": "string", "pattern": "\\+[1-9][0-9]?"}
            },
            "required": ["repo", "path", "part"]
        }),
    )
}

/// `a.rs` in `r` holds parts `KehT` (three lines) and `r7Lm` (one).
fn holds() -> ValueHolds {
    let at = |path: &str| vec!["r".to_string(), path.to_string()];
    ValueHolds {
        fields: vec![
            FieldHold {
                tools: vec!["file_edit".into()],
                field: "part".into(),
                on: vec!["repo".into(), "path".into()],
                table: HashMap::from([(at("a.rs"), vec!["KehT".into(), "r7Lm".into()])]),
            },
            FieldHold {
                tools: vec!["file_edit".into()],
                field: "from".into(),
                on: vec!["repo".into(), "path".into(), "part".into()],
                table: HashMap::from([
                    (
                        vec!["r".into(), "a.rs".into(), "KehT".into()],
                        vec!["+1".into(), "+2".into(), "+3".into()],
                    ),
                    (
                        vec!["r".into(), "a.rs".into(), "r7Lm".into()],
                        vec!["+1".into()],
                    ),
                ]),
            },
        ],
    }
}

/// A second tool, so a call's name is a choice the walk reads.
fn read_tool() -> ToolSpec {
    ToolSpec::from_json_schema(
        "file_read",
        &json!({
            "type": "object",
            "properties": {"repo": {"type": "string"}, "path": {"type": "string"}},
            "required": ["repo", "path"]
        }),
    )
}

fn driver(v: &TestVocab) -> StencilDriver {
    let tools = [edit_tool(), read_tool()];
    let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
    let tree = Arc::new(compile(&spec, v).unwrap());
    let registry = TriggerRegistry::new()
        .with_trigger(TRIGGER, tree)
        .holding_values(TRIGGER, Arc::new(HeldValues::build(&holds(), v)));
    registry.driver_for(TRIGGER).unwrap()
}

/// The value being written in `text`, when one is: its field and what of it
/// — from its opening quote — is written.
fn writing(text: &str) -> Option<(&str, &str)> {
    let at = text.rfind("\": ")?;
    let rest = &text[at + 3..];
    if rest.matches('"').count() >= 2 {
        return None;
    }
    let key = &text[..at];
    Some((&key[key.rfind('"')? + 1..], rest))
}

/// Walk a call that gives each field its value in `values`, until `stop`
/// holds of what is written; answer the text and the allowed bytes at the
/// step that stopped it.
fn walk(
    v: &TestVocab,
    values: &[(&str, &str)],
    stop: impl Fn(&str) -> bool,
) -> (String, BTreeSet<u8>) {
    let mut values: HashMap<&str, &str> = values.iter().copied().collect();
    values.insert("name", "file_edit");
    let mut d = driver(v);
    let mut text = String::new();
    for _ in 0..400 {
        let set: Option<AllowedSet> = match d.step() {
            StepMask::Prefill(run) => {
                text.push_str(&String::from_utf8(v.decode(&run)).unwrap());
                continue;
            }
            StepMask::Branch(set) => Some(set),
            StepMask::Free { .. } => None,
            StepMask::Done => return (text, BTreeSet::new()),
        };
        let allowed = |set: &Option<AllowedSet>| -> BTreeSet<u8> {
            set.as_ref()
                .map(|s| {
                    s.tokens()
                        .iter()
                        .filter_map(|&t| v.token_bytes(t).first().copied())
                        .collect()
                })
                .unwrap_or_default()
        };
        if stop(&text) {
            return (text, allowed(&set));
        }
        let only = set
            .as_ref()
            .filter(|s| s.tokens().len() == 1)
            .and_then(|s| v.token_bytes(s.tokens()[0]).first().copied());
        let byte = if let Some(only) = only {
            only
        } else if text.ends_with(':') {
            b' '
        } else if let Some((field, written)) = writing(&text) {
            let want = format!("\"{}\"", values[field]);
            want.as_bytes()[written.len()]
        } else {
            // A choice between fields: `from` while it is still to write.
            if values.contains_key("from") && !text.contains("\"from\"") {
                b','
            } else {
                b'}'
            }
        };
        let token = byte as TokenId;
        if let Some(set) = &set {
            assert!(
                set.tokens().contains(&token),
                "{:?} is not allowed after {text:?}: {:?}",
                byte as char,
                allowed(&Some(set.clone()))
            );
        }
        text.push(byte as char);
        d.accept(token, &[byte]);
    }
    panic!("the walk did not finish: {text:?}");
}

fn bytes(s: &str) -> BTreeSet<u8> {
    s.bytes().collect()
}

/// **A part is one of the file's anchors**: writing `part` in `a.rs`, the
/// first character offered is the first of `KehT` or `r7Lm` and no other.
#[test]
fn a_held_part_is_one_of_the_files_anchors() {
    let v = TestVocab::new();
    let values = [("repo", "r"), ("path", "a.rs"), ("part", "KehT")];
    let (_, allowed) = walk(&v, &values, |t| t.ends_with("\"part\": \""));
    assert_eq!(allowed, bytes("Kr"));
}

/// **A line is one of the part's**: after `KehT`, `from` is `+1` to `+3`;
/// after `r7Lm`, `+1` alone — so `+4`, and `last`, cannot be written.
#[test]
fn a_held_line_is_one_of_the_parts_lines() {
    let v = TestVocab::new();
    let at = |part: &'static str| {
        [
            ("repo", "r"),
            ("path", "a.rs"),
            ("part", part),
            ("from", "+1"),
        ]
    };
    let (_, three) = walk(&v, &at("KehT"), |t| t.ends_with("\"from\": \"+"));
    assert_eq!(three, bytes("123"));
    let (_, one) = walk(&v, &at("r7Lm"), |t| t.ends_with("\"from\": \"+"));
    assert_eq!(one, bytes("1"));
    let (_, opening) = walk(&v, &at("KehT"), |t| t.ends_with("\"from\": \""));
    assert_eq!(opening, bytes("+"), "`last` cannot begin");
}

/// **A held call writes through to its end**, and a file nothing holds keeps
/// the field's own pattern: any anchor character may begin its part.
#[test]
fn a_held_call_completes_and_an_unheld_file_keeps_its_pattern() {
    let v = TestVocab::new();
    let held = [
        ("repo", "r"),
        ("path", "a.rs"),
        ("part", "KehT"),
        ("from", "+2"),
    ];
    let (text, _) = walk(&v, &held, |_| false);
    assert!(
        text.contains("\"part\": \"KehT\"") && text.contains("\"from\": \"+2\""),
        "{text}"
    );
    let free = [("repo", "r"), ("path", "b.rs"), ("part", "ZZZZ")];
    let (_, allowed) = walk(&v, &free, |t| t.ends_with("\"part\": \""));
    assert!(
        allowed.contains(&b'Z') && allowed.contains(&b'9'),
        "{allowed:?}"
    );
}
