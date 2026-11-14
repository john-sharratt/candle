//! A call held to the values that exist (`docs/stencil_tree.md` §9.2): a
//! file's anchors for `part`, the part's lines for `from`, each depending on
//! what the call has already written.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use serde_json::json;

use super::compile::compile;
use super::driver::{StencilDriver, StepMask};
use super::held_values::{FieldHold, HeldRules, HeldValues, ValueHolds};
use super::mask::AllowedSet;
use super::session::{StencilAction, StencilSession};
use super::spec::{NodeSpec, TreeSpec};
use super::tool_call::{compile_tool_call_tree, ToolCallEnvelope, ToolSpec};
use super::tree::FreeTextLimits;
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
        .holding_values(
            TRIGGER,
            Arc::new(HeldValues::build(&holds(), v, &mut HeldRules::default())),
        );
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
    walk_on(v, driver(v), "file_edit", values, stop)
}

/// [`walk`] on `d`, a call of `tool`.
fn walk_on(
    v: &TestVocab,
    mut d: StencilDriver,
    tool: &str,
    values: &[(&str, &str)],
    stop: impl Fn(&str) -> bool,
) -> (String, BTreeSet<u8>) {
    let mut values: HashMap<&str, &str> = values.iter().copied().collect();
    values.insert("name", tool);
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

/// A listing whose `repo` is one of `repos` — an enum, as the daemon writes a
/// workspace's repositories into it — and whose `path` names a directory.
fn list_tool(repos: &[&str]) -> ToolSpec {
    ToolSpec::from_json_schema(
        "file_list",
        &json!({
            "type": "object",
            "properties": {
                "repo": {"type": "string", "enum": repos},
                "path": {"type": "string"}
            },
            "required": ["repo", "path"]
        }),
    )
}

/// `r` has `src` and `docs`; `s` has `crates`.
fn dirs() -> ValueHolds {
    ValueHolds {
        fields: vec![FieldHold {
            tools: vec!["file_list".into()],
            field: "path".into(),
            on: vec!["repo".into()],
            table: HashMap::from([
                (vec!["r".into()], vec!["src".into(), "docs".into()]),
                (vec!["s".into()], vec!["crates".into()]),
            ]),
        }],
    }
}

fn list_driver(v: &TestVocab, repos: &[&str]) -> StencilDriver {
    let spec = compile_tool_call_tree(&[list_tool(repos), read_tool()], &ToolCallEnvelope::qwen3())
        .unwrap();
    let tree = Arc::new(compile(&spec, v).unwrap());
    TriggerRegistry::new()
        .with_trigger(TRIGGER, tree)
        .holding_values(
            TRIGGER,
            Arc::new(HeldValues::build(&dirs(), v, &mut HeldRules::default())),
        )
        .driver_for(TRIGGER)
        .unwrap()
}

/// **A field held on an enum is held by the value the enum's branch chose.**
/// `repo` is a choice, not a span, so nothing but the branch can say which
/// repository the call names: in `r` a listing's path begins `s` or `d`, in
/// `s` only `c` — the directory `s` has. Read from spans alone, the path was
/// free in every repository, and a live listing of `crates` in a repository
/// without one came back `not_found`.
#[test]
fn a_field_held_on_an_enum_reads_the_value_it_chose() {
    let v = TestVocab::new();
    let at_path = |t: &str| t.ends_with("\"path\": \"");
    let (_, in_r) = walk_on(
        &v,
        list_driver(&v, &["r", "s"]),
        "file_list",
        &[("repo", "r"), ("path", "src")],
        at_path,
    );
    assert_eq!(in_r, bytes("ds"));
    let (_, in_s) = walk_on(
        &v,
        list_driver(&v, &["r", "s"]),
        "file_list",
        &[("repo", "s"), ("path", "crates")],
        at_path,
    );
    assert_eq!(in_s, bytes("c"));
}

/// **A one-value enum is written by the key, and still held by**: with `s`
/// the workspace's only repository, `repo` has no branch to choose it, and the
/// path is held to `s`'s directories all the same.
#[test]
fn a_one_value_enum_holds_by_its_value() {
    let v = TestVocab::new();
    let (text, allowed) = walk_on(
        &v,
        list_driver(&v, &["s"]),
        "file_list",
        &[("path", "crates")],
        |t| t.ends_with("\"path\": \""),
    );
    assert!(text.contains("\"repo\": \"s\""), "{text}");
    assert_eq!(allowed, bytes("c"));
}

/// **A spill into a held value keeps to the held rule**: a token that closes
/// the name and writes the value's first byte is offered only when that byte
/// begins a held value. `"s` would begin `src/x.rs`, which the field's own
/// pattern allows and the held set does not have; taken, the held rule could
/// only have left the rest of the value free — the path the readings lost
/// their first directory by.
#[test]
fn a_spill_into_a_held_value_keeps_to_the_held_rule() {
    const QUOTE_S: TokenId = 301;
    const QUOTE_C: TokenId = 302;
    let v = TestVocab::new()
        .with_special("\"s", QUOTE_S)
        .with_special("\"c", QUOTE_C);
    let mut spec = TreeSpec::new("spill");
    let end = spec.push(NodeSpec::End);
    let path = spec.push(NodeSpec::Value {
        pattern: "[a-z/.-]+".into(),
        opened: true,
        limits: FreeTextLimits::json_string(),
        next: end,
    });
    let name = spec.push(NodeSpec::Branch {
        arms: vec![("ab\"".into(), path), ("cd\"".into(), path)],
    });
    spec.root = name;
    spec.keys.insert(path, Arc::from("path"));
    spec.name_branches.insert(name);
    let tree = Arc::new(compile(&spec, &v).unwrap());
    let holds = ValueHolds {
        fields: vec![FieldHold {
            tools: vec!["cd".into()],
            field: "path".into(),
            on: vec![],
            table: HashMap::from([(vec![], vec!["candle-conversation/x.rs".into()])]),
        }],
    };
    let mut session = StencilSession::new(tree);
    session.hold_values(Arc::new(HeldValues::build(
        &holds,
        &v,
        &mut HeldRules::default(),
    )));
    for (token, bytes) in [(b'c' as TokenId, &b"c"[..]), (b'd' as TokenId, b"d")] {
        assert!(matches!(
            session.next_action(),
            StencilAction::MaskedDecode(_)
        ));
        session.observe(token, bytes).unwrap();
    }
    let StencilAction::MaskedDecode(after_name) = session.next_action() else {
        panic!("the name's close is a choice");
    };
    assert!(
        after_name.tokens().contains(&QUOTE_C),
        "`\"c` begins the held path"
    );
    assert!(
        !after_name.tokens().contains(&QUOTE_S),
        "`\"s` would begin a path the set does not hold"
    );
    session.observe(QUOTE_C, b"\"c").unwrap();
    let StencilAction::MaskedDecode(next) = session.next_action() else {
        panic!("the path is held");
    };
    assert_eq!(next.tokens(), [b'a' as TokenId]);
}

/// **A name finished by a token that runs on names its call**: `cd"` closed by
/// `"K`, which also writes the part's first character, is a call of `cd`, so
/// the part goes on held to `cd`'s `KehT` — `e` and nothing else.
#[test]
fn a_name_spilled_into_its_value_still_names_the_call() {
    const QUOTE_K: TokenId = 300;
    let v = TestVocab::new().with_special("\"K", QUOTE_K);
    let mut spec = TreeSpec::new("spill");
    let end = spec.push(NodeSpec::End);
    let part = spec.push(NodeSpec::Value {
        pattern: "[A-Za-z0-9]{4}".into(),
        opened: true,
        limits: FreeTextLimits::json_string(),
        next: end,
    });
    let name = spec.push(NodeSpec::Branch {
        arms: vec![("ab\"".into(), part), ("cd\"".into(), part)],
    });
    spec.root = name;
    spec.keys.insert(part, Arc::from("part"));
    spec.name_branches.insert(name);
    let tree = Arc::new(compile(&spec, &v).unwrap());
    let holds = ValueHolds {
        fields: vec![FieldHold {
            tools: vec!["cd".into()],
            field: "part".into(),
            on: vec![],
            table: HashMap::from([(vec![], vec!["KehT".into()])]),
        }],
    };
    let mut session = StencilSession::new(tree);
    session.hold_values(Arc::new(HeldValues::build(
        &holds,
        &v,
        &mut HeldRules::default(),
    )));
    for (token, bytes) in [(b'c' as TokenId, &b"c"[..]), (b'd' as TokenId, b"d")] {
        assert!(matches!(
            session.next_action(),
            StencilAction::MaskedDecode(_)
        ));
        session.observe(token, bytes).unwrap();
    }
    let StencilAction::MaskedDecode(after_name) = session.next_action() else {
        panic!("the name's close is a choice");
    };
    assert!(after_name.tokens().contains(&QUOTE_K), "`\"K` spills");
    session.observe(QUOTE_K, b"\"K").unwrap();
    let StencilAction::MaskedDecode(next) = session.next_action() else {
        panic!("the part is held");
    };
    assert_eq!(next.tokens(), [b'e' as TokenId]);
}

/// **A catalog of one tool still names its calls, and a function block's raw
/// values are held.** A file's reading offers `file_read` alone, so no branch
/// chooses the tool: the grammar writes the name, and the walk names the call
/// where it does. Held there, each raw value can be only its file's: with
/// `src` as a directory of the repository root, `candle-conversation/…` is
/// still the one path the walk can write.
#[test]
fn a_lone_tools_raw_values_are_held() {
    let v = TestVocab::new();
    let spec = compile_tool_call_tree(&[read_tool()], &ToolCallEnvelope::qwen35()).unwrap();
    let tree = Arc::new(compile(&spec, &v).unwrap());
    let one = |field: &str, value: &str| FieldHold {
        tools: vec!["file_read".into()],
        field: field.into(),
        on: Vec::new(),
        table: HashMap::from([(Vec::new(), vec![value.to_string()])]),
    };
    let holds = ValueHolds {
        fields: vec![
            one("repo", "candle"),
            one("path", "candle-conversation/src/cancel.rs"),
        ],
    };
    let mut d = TriggerRegistry::new()
        .with_trigger(TRIGGER, tree)
        .holding_values(
            TRIGGER,
            Arc::new(HeldValues::build(&holds, &v, &mut HeldRules::default())),
        )
        .driver_for(TRIGGER)
        .unwrap();
    let mut text = String::new();
    for _ in 0..400 {
        let set = match d.step() {
            StepMask::Prefill(run) => {
                text.push_str(&String::from_utf8(v.decode(&run)).unwrap());
                continue;
            }
            StepMask::Branch(set) => set,
            StepMask::Free { .. } => panic!("a value decodes free after {text:?}"),
            StepMask::Done => break,
        };
        let offered: Vec<u8> = set
            .tokens()
            .iter()
            .filter_map(|&t| v.token_bytes(t).first().copied())
            .collect();
        // One way on inside a held value; at its close, the layout's newline,
        // then the next tag.
        let byte = match offered.as_slice() {
            [only] => *only,
            _ if offered.contains(&b'\n') && !text.ends_with('\n') => b'\n',
            _ => b'<',
        };
        assert!(
            offered.contains(&byte),
            "{byte:?} after {text:?}: {offered:?}"
        );
        text.push(byte as char);
        d.accept(byte as TokenId, &[byte]);
    }
    assert!(
        text.contains("<parameter=repo>\ncandle</parameter>"),
        "{text:?}"
    );
    assert!(
        text.contains("<parameter=path>\ncandle-conversation/src/cancel.rs</parameter>"),
        "{text:?}"
    );
}
