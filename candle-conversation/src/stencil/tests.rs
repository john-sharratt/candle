//! End-to-end scenario suites: drive whole grammars with the simulator, assert
//! exact emitted text, and re-parse the output as JSON.  Component-level edge
//! cases live in each file's own `#[cfg(test)]`; this file is the integration
//! net, including the tool-library construction tests.

use std::sync::Arc;

use super::compile::compile;
use super::session::Observe;
use super::sim::{simulate, Oracle};
use super::tool_call::{compile_tool_call_tree, parse_tools, ToolCallEnvelope};
use super::vocab::{TestVocab, TokenId, Vocab};

/// Decode-step tokens for a byte string (one token per byte).
fn bytes_of(s: &str) -> Vec<TokenId> {
    s.bytes().map(|b| b as TokenId).collect()
}

/// The JSON body of a `<tool_call>…</tool_call>` envelope.
fn json_body(text: &str) -> &str {
    text.trim_start_matches("<tool_call>\n")
        .trim_end_matches("\n</tool_call>")
}

fn three_tool_catalog() -> Vec<super::tool_call::ToolSpec> {
    parse_tools(
        r#"[
          {"name":"read_file","params":[{"name":"path","type":"string","required":true}]},
          {"name":"write_file","params":[
              {"name":"path","type":"string","required":true},
              {"name":"create","type":"boolean","required":false}
          ]},
          {"name":"set_mode","params":[
              {"name":"mode","type":"string","required":true,"enum":["read","write","exec"]}
          ]}
        ]"#,
    )
    .unwrap()
}

/// Compile a catalog into a walkable tool-call tree under the Qwen3 envelope —
/// the one-liner every full-walk test starts from.
fn tree_of(catalog: &[super::tool_call::ToolSpec], v: &TestVocab) -> Arc<super::tree::StencilTree> {
    Arc::new(
        compile(
            &compile_tool_call_tree(catalog, &ToolCallEnvelope::qwen3()).unwrap(),
            v,
        )
        .unwrap(),
    )
}

// ── Full tool-call walks (Scripted oracle, exact output, JSON-valid) ─────────

#[test]
fn read_file_minimal_call() {
    let v = TestVocab::new();
    let tree = tree_of(&three_tool_catalog(), &v);
    // Decode steps: tool name, then the path value + closing quote.
    let mut script = bytes_of("read_file\"");
    script.extend(bytes_of("src/main.rs\""));
    let run = simulate(tree, &v, Oracle::Scripted(script), 1000).unwrap();
    let text = run.text(&v);
    assert_eq!(
        text,
        "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/main.rs\"}}\n</tool_call>"
    );
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(parsed["name"], "read_file");
    assert_eq!(parsed["arguments"]["path"], "src/main.rs");
    assert_eq!(run.healed_bytes, 0);
}

#[test]
fn write_file_with_optional_included() {
    let v = TestVocab::new();
    let tree = tree_of(&three_tool_catalog(), &v);
    // name "write_file"; path value "a.txt\""; then the optional gate must be
    // chosen via the `create` arm — but that's a Branch, so the decode step that
    // selects it is the first token of `, "create": ` which is ','.
    let mut script = bytes_of("write_file\"");
    script.extend(bytes_of("a.txt\"")); // path value + close quote
                                        // optional gate: choose ", \"create\": " (starts with ',') then boolean "true"
    script.extend(bytes_of(", \"create\": ")); // walk the gate arm trie
    script.extend(bytes_of("true")); // boolean branch
    let run = simulate(tree, &v, Oracle::Scripted(script), 2000).unwrap();
    let text = run.text(&v);
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(parsed["name"], "write_file");
    assert_eq!(parsed["arguments"]["path"], "a.txt");
    assert_eq!(parsed["arguments"]["create"], true);
}

#[test]
fn write_file_with_optional_skipped() {
    let v = TestVocab::new();
    let tree = tree_of(&three_tool_catalog(), &v);
    let mut script = bytes_of("write_file\"");
    script.extend(bytes_of("a.txt\""));
    // optional gate: choose the close arm "}}\n</tool_call>" (starts with '}').
    script.extend(bytes_of("}}\n</tool_call>"));
    let run = simulate(tree, &v, Oracle::Scripted(script), 2000).unwrap();
    let text = run.text(&v);
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(parsed["name"], "write_file");
    assert_eq!(parsed["arguments"]["path"], "a.txt");
    assert!(parsed["arguments"].get("create").is_none());
}

#[test]
fn set_mode_enum_value() {
    let v = TestVocab::new();
    let tree = tree_of(&three_tool_catalog(), &v);
    // name "set_mode"; enum value "exec" (inside the value's quote branch). The
    // only param is required, so the close is prefilled — no further decode.
    let mut script = bytes_of("set_mode\"");
    script.extend(bytes_of("exec\"")); // enum branch
    let run = simulate(tree, &v, Oracle::Scripted(script), 2000).unwrap();
    let text = run.text(&v);
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(parsed["name"], "set_mode");
    assert_eq!(parsed["arguments"]["mode"], "exec");
}

// ── Calculator tool: does the stencil mangle the `expression` value? ─────────
//
// Reproduces the live daemon's tool-call path for the calculator tool, built the
// SAME way the daemon builds it (`zend/src/session.rs`): `ToolSpec::from_json_
// schema` over each tool's schemars JSON Schema.  Several tools so the name is a
// real Branch (as in the 93-tool live catalog), not a folded single-tool static.
fn calc_catalog() -> Vec<super::tool_call::ToolSpec> {
    use super::tool_call::ToolSpec;
    vec![
        ToolSpec::from_json_schema(
            "datetime",
            &serde_json::json!({
                "type": "object",
                "properties": { "timezone": { "type": "string" } },
            }),
        ),
        ToolSpec::from_json_schema(
            "calculator",
            &serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic/scientific expression to evaluate",
                    }
                },
                "required": ["expression"],
            }),
        ),
        ToolSpec::from_json_schema(
            "write",
            &serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" }, "content": { "type": "string" } },
                "required": ["path", "content"],
            }),
        ),
    ]
}

fn calc_tree(v: &TestVocab) -> Arc<super::tree::StencilTree> {
    tree_of(&calc_catalog(), v)
}

// The model selects `calculator` and free-decodes `sqrt(324523452345)` for the
// expression.  If the stencil is faithful, the emitted JSON carries that exact
// expression — proving the stencil is NOT the source of the live `" , "` garbage.
#[test]
fn calculator_passes_sqrt_expression_verbatim() {
    let v = TestVocab::new();
    let tree = calc_tree(&v);
    let mut script = bytes_of("calculator\""); // name branch + closing quote
    script.extend(bytes_of("sqrt(324523452345)\"")); // free expression value + close
    let run = simulate(tree, &v, Oracle::Scripted(script), 2000).unwrap();
    let text = run.text(&v);
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(parsed["name"], "calculator");
    assert_eq!(parsed["arguments"]["expression"], "sqrt(324523452345)");
    // No mid-token heals, no forced close — the value span ran to the model's own
    // closing quote, exactly as long as the model drove it.
    assert_eq!(run.healed_bytes, 0);
    assert_eq!(run.forced_closes, 0);
}

// The live daemon produced `{"expression": " , "}`.  This confirms the stencil is
// a faithful pass-through of whatever the model free-decodes: feed the SAME bytes
// the daemon observed and the stencil reflects them unchanged.  Together with the
// test above, this proves the garbage originated in the model's decode (a context
// / logits problem), not in constrained decoding.
#[test]
fn calculator_reflects_model_value_verbatim_even_when_garbage() {
    let v = TestVocab::new();
    let tree = calc_tree(&v);
    let mut script = bytes_of("calculator\"");
    script.extend(bytes_of(" , \"")); // the exact garbage the daemon emitted, + close
    let run = simulate(tree, &v, Oracle::Scripted(script), 2000).unwrap();
    let text = run.text(&v);
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(parsed["name"], "calculator");
    assert_eq!(parsed["arguments"]["expression"], " , ");
}

// Numbers and arrays are emitted as JSON values, lookahead-terminated and
// pushed back to the close — exercises the session push-back path.
#[test]
fn integer_value_via_pushback() {
    let v = TestVocab::new();
    let tools = parse_tools(
        r#"[{"name":"wait","params":[{"name":"secs","type":"integer","required":true}]}]"#,
    )
    .unwrap();
    let tree = Arc::new(
        compile(
            &compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap(),
            &v,
        )
        .unwrap(),
    );
    // Single tool ⇒ the name folds to a prefilled static; only the value is
    // decoded. value "30" then the args-close '}' (lookahead delimiter).
    let script = bytes_of("30}");
    let run = simulate(tree, &v, Oracle::Scripted(script), 2000).unwrap();
    let text = run.text(&v);
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(parsed["name"], "wait");
    assert_eq!(parsed["arguments"]["secs"], 30);
}

#[test]
fn array_value_via_pushback() {
    let v = TestVocab::new();
    let tools = parse_tools(
        r#"[{"name":"pick","params":[{"name":"opts","type":"array","required":true}]}]"#,
    )
    .unwrap();
    let tree = Arc::new(
        compile(
            &compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap(),
            &v,
        )
        .unwrap(),
    );
    // a nested array whose inner commas/brackets must not terminate early.
    let script = bytes_of("[1,[2,3],4]}");
    let run = simulate(tree, &v, Oracle::Scripted(script), 2000).unwrap();
    let text = run.text(&v);
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(
        parsed["arguments"]["opts"],
        serde_json::json!([1, [2, 3], 4])
    );
}

// ── Adversarial value contents ──────────────────────────────────────────────

#[test]
fn string_value_with_escaped_quote() {
    let v = TestVocab::new();
    let tree = tree_of(&three_tool_catalog(), &v);
    // path value contains an escaped quote: a\"b  — must not close early.
    let mut script = bytes_of("read_file\"");
    script.extend(bytes_of("a\\\"b\"")); // a \ " b "  → closes only at the final "
    let run = simulate(tree, &v, Oracle::Scripted(script), 2000).unwrap();
    let text = run.text(&v);
    let parsed: serde_json::Value = serde_json::from_str(json_body(&text)).unwrap();
    assert_eq!(parsed["arguments"]["path"], "a\"b");
}

// ── Construction equivalence (builder == yaml for the same grammar) ──────────

#[test]
fn builder_and_yaml_agree() {
    use super::builder::StencilTreeBuilder;
    use super::spec::TreeSpec;

    let built = StencilTreeBuilder::new("t")
        .root("open")
        .static_node("open", "\"", "name")
        .branch("name", &[("ab", "close"), ("cd", "close")])
        .static_node("close", "\"", "done")
        .end("done")
        .build()
        .unwrap();

    let yaml = r#"
label: t
root: open
nodes:
  - id: open
    static: "\""
    next: name
  - id: name
    branch:
      - match: "ab"
        next: close
      - match: "cd"
        next: close
  - id: close
    static: "\""
    next: done
  - id: done
    end: true
"#;
    let from_yaml = TreeSpec::from_yaml(yaml).unwrap();

    let v = TestVocab::new();
    let tb = Arc::new(compile(&built, &v).unwrap());
    let ty = Arc::new(compile(&from_yaml, &v).unwrap());

    // Same script through both → identical emitted text.
    let script = bytes_of("ab");
    let rb = simulate(tb, &v, Oracle::Scripted(script.clone()), 100).unwrap();
    let ry = simulate(ty, &v, Oracle::Scripted(script), 100).unwrap();
    assert_eq!(rb.text(&v), ry.text(&v));
    assert_eq!(rb.text(&v), "\"ab\"");
}

// ── Tool-library construction: every path of every tool simulates valid JSON ─

#[test]
fn every_tool_path_yields_valid_json() {
    let v = TestVocab::new();
    let catalog = three_tool_catalog();
    let tree = Arc::new(
        compile(
            &compile_tool_call_tree(&catalog, &ToolCallEnvelope::qwen3()).unwrap(),
            &v,
        )
        .unwrap(),
    );

    // A set of scripted decode sequences exercising every tool, both optional
    // states, and the enum. Each must produce parseable JSON with name ∈ catalog.
    let names: std::collections::HashSet<&str> = catalog.iter().map(|t| t.name.as_str()).collect();

    let mut scripts: Vec<Vec<TokenId>> = Vec::new();
    // read_file
    scripts.push([bytes_of("read_file\""), bytes_of("x\"")].concat());
    // write_file, create skipped
    scripts.push(
        [
            bytes_of("write_file\""),
            bytes_of("y\""),
            bytes_of("}}\n</tool_call>"),
        ]
        .concat(),
    );
    // write_file, create=false
    scripts.push(
        [
            bytes_of("write_file\""),
            bytes_of("y\""),
            bytes_of(", \"create\": "),
            bytes_of("false"),
            bytes_of("}}\n</tool_call>"),
        ]
        .concat(),
    );
    // set_mode each enum value, count skipped
    for mode in ["read", "write", "exec"] {
        scripts.push(
            [
                bytes_of("set_mode\""),
                bytes_of(&format!("{mode}\"")),
                bytes_of("}}\n</tool_call>"),
            ]
            .concat(),
        );
    }

    for script in scripts {
        let run = simulate(Arc::clone(&tree), &v, Oracle::Scripted(script), 4000).unwrap();
        let text = run.text(&v);
        let parsed: serde_json::Value = serde_json::from_str(json_body(&text))
            .unwrap_or_else(|e| panic!("not JSON: {text:?}: {e}"));
        let name = parsed["name"].as_str().unwrap();
        assert!(names.contains(name), "name {name:?} not in catalog");
        assert!(parsed["arguments"].is_object());
    }
}

// ── Failure modes ───────────────────────────────────────────────────────────

#[test]
fn out_of_mask_token_bails_gracefully() {
    let v = TestVocab::new();
    let tree = tree_of(&three_tool_catalog(), &v);
    // First decode is the tool-name branch; 'Z' is not a legal first byte.  The
    // failsafe bails: it does NOT error, it emits the bail (close) tokens and
    // exits so the partial output is terminated.
    let run = simulate(tree, &v, Oracle::Scripted(vec![b'Z' as u32]), 100).unwrap();
    assert!(run.observes.contains(&Observe::Bailed));
    let text = run.text(&v);
    // The bad byte is present, followed by the bail close that terminates the
    // tool-call block.
    assert!(text.contains('Z'));
    assert!(
        text.ends_with("</tool_call>"),
        "bail must terminate the block: {text:?}"
    );
}

#[test]
fn forced_close_on_runaway_string() {
    use super::builder::StencilTreeBuilder;
    use super::terminator::Terminator;
    use super::tree::FreeTextLimits;

    let v = TestVocab::new();
    // A string span with a tiny hard limit and a never-closing value.
    let limits = FreeTextLimits {
        ramp_start: None,
        ramp_len: 0,
        boost: 0.0,
        forced_after: 3,
    };
    let spec = StencilTreeBuilder::new("t")
        .root("v")
        .free_text("v", Terminator::JsonString, false, limits, "done")
        .end("done")
        .build()
        .unwrap();
    let tree = Arc::new(compile(&spec, &v).unwrap());
    // Emit non-closing bytes forever; the hard limit force-closes at 3.
    let policy = Oracle::Policy(Box::new(|_| b'x' as TokenId));
    let run = simulate(tree, &v, policy, 100).unwrap();
    assert_eq!(run.forced_closes, 1);
    assert!(run
        .observes
        .iter()
        .any(|o| matches!(o, Observe::SpanForcedClosed)));
}

#[test]
fn eos_ends_a_free_span() {
    use super::builder::StencilTreeBuilder;
    use super::terminator::Terminator;
    use super::tree::FreeTextLimits;

    let v = TestVocab::new();
    let spec = StencilTreeBuilder::new("t")
        .root("v")
        .free_text(
            "v",
            Terminator::JsonString,
            true, // eos_ends
            FreeTextLimits::json_string(),
            "done",
        )
        .end("done")
        .build()
        .unwrap();
    let tree = Arc::new(compile(&spec, &v).unwrap());
    // Emit a couple of bytes then EOS.
    let eos = v.eos();
    let run = simulate(
        tree,
        &v,
        Oracle::Scripted(vec![b'h' as u32, b'i' as u32, eos]),
        100,
    )
    .unwrap();
    assert!(run.observes.iter().any(|o| matches!(o, Observe::SpanEos)));
}
