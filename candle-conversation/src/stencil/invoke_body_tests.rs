//! A typed object argument — the shape an `invoke` call's `body` takes —
//! driven exhaustively by the simulator.
//!
//! Every path of a compiled call is enumerated (the `PickArm`-style policy of
//! `stencil_tree.md` §14) and the body it emits is re-parsed as JSON and checked
//! against the schema: string enums decode to *exactly* the schema's values,
//! booleans to `true`/`false`, optionals include and skip cleanly with no stray
//! comma, and the weakly-typed fields (integer/number/array/object) shape valid
//! JSON. Real effector schemas (the lift's `use` body, a station verb) and the
//! adversarial free-text cases round it out.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;
use std::sync::Arc;

use serde_json::Value;

use super::compile::compile;
use super::session::Observe;
use super::sim::{simulate, Oracle, SimRun};
use super::tool_call::{compile_action_loop, Param, ParamType, ToolCallEnvelope, ToolSpec};
use super::tree::{FreeTextLimits, StencilTree};
use super::vocab::{TestVocab, TokenId};

// ── harness ──────────────────────────────────────────────────────────────────

/// The closing quote — the free-text token that ends every `JsonString` value at
/// once, so an enumeration's incidental free-string fields decode to `""`.
pub(super) const QUOTE: TokenId = b'"' as TokenId;

/// One call to a one-tool catalog whose single required argument is `body`, an
/// object typed by `schema`: the smallest turn that exercises a typed object
/// argument.
fn body_tree(schema: &Value, v: &TestVocab) -> Arc<StencilTree> {
    let fields = ToolSpec::from_json_schema("body", schema).params;
    let tool = ToolSpec {
        name: "t".to_string(),
        params: vec![Param {
            name: "body".to_string(),
            ty: ParamType::Object,
            required: true,
            enum_values: None,
            items: None,
            min_items: 0,
            properties: Some(fields),
            nullable: false,
            minimum: None,
            requires: Vec::new(),
            shapes: Vec::new(),
        }],
    };
    let spec = compile_action_loop(&[tool], &ToolCallEnvelope::qwen3(), 1, "<|im_end|>", None)
        .expect("the body compiles");
    Arc::new(compile(&spec, v).expect("the body lowers"))
}

/// Run one path: follow `plan`'s arm choices at each branch (defaulting to arm 0
/// beyond it), emitting `free` in any free span. Returns the run and the arity of
/// every branch it met, so the enumerator knows where it may still fork.
fn run_plan(
    tree: Arc<StencilTree>,
    v: &TestVocab,
    free: TokenId,
    plan: &[usize],
) -> (SimRun, Vec<usize>) {
    let arities = Rc::new(RefCell::new(Vec::new()));
    let step = Rc::new(RefCell::new(0usize));
    let plan = plan.to_vec();
    let ar = Rc::clone(&arities);
    let st = Rc::clone(&step);
    let oracle = Oracle::Policy(Box::new(move |allowed| match allowed {
        Some(set) => {
            let i = *st.borrow();
            *st.borrow_mut() += 1;
            ar.borrow_mut().push(set.len());
            let choice = plan
                .get(i)
                .copied()
                .unwrap_or(0)
                .min(set.len().saturating_sub(1));
            set.tokens()[choice]
        }
        None => free,
    }));
    let run = simulate(tree, v, oracle, 20_000).expect("the body simulates");
    let a = arities.borrow().clone();
    (run, a)
}

/// Every complete path through a compiled call (a depth-first enumeration over
/// branch arms — the `PickArm` oracle of §14.1).
pub(super) fn all_paths(tree: &Arc<StencilTree>, v: &TestVocab, free: TokenId) -> Vec<SimRun> {
    fn rec(
        tree: &Arc<StencilTree>,
        v: &TestVocab,
        free: TokenId,
        plan: Vec<usize>,
        out: &mut Vec<SimRun>,
    ) {
        let (run, arities) = run_plan(Arc::clone(tree), v, free, &plan);
        if plan.len() >= arities.len() {
            out.push(run);
            return;
        }
        let i = plan.len();
        for j in 0..arities[i] {
            let mut p = plan.clone();
            p.push(j);
            rec(tree, v, free, p, out);
        }
    }
    let mut out = Vec::new();
    rec(tree, v, free, Vec::new(), &mut out);
    out
}

/// The JSON of a single-call turn, between the tool-call markers.
pub(super) fn extract_call(text: &str) -> &str {
    text.trim_start_matches("<tool_call>\n")
        .split("\n</tool_call>")
        .next()
        .unwrap()
}

/// Parse a run's call as JSON and return its `body` argument, failing loudly
/// with the text.
fn body_json(run: &SimRun, v: &TestVocab) -> Value {
    let text = run.text(v);
    let call: Value = serde_json::from_str(extract_call(&text))
        .unwrap_or_else(|e| panic!("call is not JSON: {text:?}: {e}"));
    call["arguments"]["body"].clone()
}

/// Every distinct JSON object a body can take, across all its paths.
fn all_bodies(schema: &Value, v: &TestVocab) -> Vec<Value> {
    let tree = body_tree(schema, v);
    all_paths(&tree, v, QUOTE)
        .iter()
        .map(|r| body_json(r, v))
        .collect()
}

// ── empty and scalar bodies ──────────────────────────────────────────────────

#[test]
fn an_empty_schema_is_the_empty_object_and_nothing_else() {
    let v = TestVocab::new();
    let bodies = all_bodies(
        &serde_json::json!({ "type": "object", "properties": {} }),
        &v,
    );
    assert_eq!(
        bodies,
        vec![serde_json::json!({})],
        "empty body is `{{}}` only"
    );
}

#[test]
fn a_required_string_enum_decodes_to_exactly_the_schema_values() {
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "floor": { "type": "string", "enum": ["command", "casting", "port"] } },
        "required": ["floor"]
    });
    let bodies = all_bodies(&schema, &v);
    // Every path parses, carries a `floor`, and its value is one of the enum's.
    let got: BTreeSet<String> = bodies
        .iter()
        .map(|b| b["floor"].as_str().expect("floor is a string").to_string())
        .collect();
    let expected: BTreeSet<String> = ["command", "casting", "port"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(
        got, expected,
        "the enum arms are exactly the schema's values"
    );
    assert_eq!(bodies.len(), 3, "one path per enum value, no more");
}

#[test]
fn a_boolean_field_enumerates_true_and_false() {
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "powered": { "type": "boolean" } },
        "required": ["powered"]
    });
    let vals: BTreeSet<bool> = all_bodies(&schema, &v)
        .iter()
        .map(|b| b["powered"].as_bool().expect("a bool"))
        .collect();
    assert_eq!(vals, BTreeSet::from([true, false]));
}

#[test]
fn a_free_string_field_is_a_valid_string() {
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "note": { "type": "string" } },
        "required": ["note"]
    });
    // The free-string span closes on the first quote → an empty string value.
    let bodies = all_bodies(&schema, &v);
    assert_eq!(bodies.len(), 1);
    assert_eq!(bodies[0]["note"], serde_json::json!(""));
}

// ── optionals: include and skip, no stray comma ──────────────────────────────

#[test]
fn an_optional_enum_is_offered_both_included_and_skipped() {
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "floor": { "type": "string", "enum": ["command", "port"] },
            "gentle": { "type": "boolean" }
        },
        "required": ["floor"]
    });
    let bodies = all_bodies(&schema, &v);
    // Paths: {floor} skipped-optional, plus {floor, gentle:true|false} — for each
    // floor. Every one parses and none has a trailing/leading comma.
    assert!(
        bodies.iter().any(|b| b.get("gentle").is_none()),
        "the optional can be skipped: {bodies:?}"
    );
    assert!(
        bodies
            .iter()
            .any(|b| b["gentle"] == serde_json::json!(true)),
        "the optional can be included: {bodies:?}"
    );
    // Each floor appears both with and without the optional → 2 floors × 3 (skip,
    // true, false) = 6 paths.
    assert_eq!(bodies.len(), 6, "{bodies:?}");
    for b in &bodies {
        assert!(b["floor"].is_string());
    }
}

#[test]
fn two_optionals_enumerate_every_subset_cleanly() {
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "a": { "type": "boolean" },
            "b": { "type": "boolean" }
        }
    });
    let bodies = all_bodies(&schema, &v);
    // Subsets: {}, {a}, {b}, {a,b} — with booleans each having two values, but the
    // key point is that every combination parses and no path is malformed.
    assert!(bodies.iter().any(|b| b.as_object().unwrap().is_empty()));
    assert!(bodies
        .iter()
        .any(|b| b.get("a").is_some() && b.get("b").is_none()));
    assert!(bodies
        .iter()
        .any(|b| b.get("a").is_none() && b.get("b").is_some()));
    assert!(bodies
        .iter()
        .any(|b| b.get("a").is_some() && b.get("b").is_some()));
}

// ── weakly-typed fields: shaped, valid JSON (scripted values) ────────────────

/// Compile a single-field body and script the one free value, then parse.
fn one_field_body(schema: &Value, value_script: &str) -> Value {
    let v = TestVocab::new();
    let tree = body_tree(schema, &v);
    let script: Vec<TokenId> = value_script.bytes().map(|b| b as TokenId).collect();
    let run = simulate(tree, &v, Oracle::Scripted(script), 5000).unwrap();
    body_json(&run, &v)
}

#[test]
fn an_integer_field_shapes_a_valid_number() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "count": { "type": "integer" } },
        "required": ["count"]
    });
    // The digits, then the object close: an integer is written under the mask,
    // and its end is the close that follows it.
    let body = one_field_body(&schema, "42}");
    assert_eq!(body["count"], serde_json::json!(42));
}

#[test]
fn a_number_field_shapes_a_valid_float() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "rate": { "type": "number" } },
        "required": ["rate"]
    });
    let body = one_field_body(&schema, "1.5}");
    assert_eq!(body["rate"], serde_json::json!(1.5));
}

#[test]
fn an_array_field_shapes_a_valid_nested_array() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "opts": { "type": "array" } },
        "required": ["opts"]
    });
    // A nested array whose inner brackets/commas must not close the value early.
    let body = one_field_body(&schema, " [1,[2,3],4]}");
    assert_eq!(body["opts"], serde_json::json!([1, [2, 3], 4]));
}

#[test]
fn an_array_of_enum_elements_is_grammar_enforced_not_just_shaped() {
    // A *guided* element type (an enum, a boolean, an object with declared
    // properties) makes `build_array` constrain every element to its own
    // schema, not just the array's brackets and commas — so this compiles to
    // a real branch choice at each element, never a free span. The opening
    // `[` rides on the key's own lead-in (folded in by `build_value`), so the
    // script picks up right after it: each element's own quoted arm, `, `
    // between them, `]` to close.
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "picks": { "type": "array", "items": { "type": "string", "enum": ["a", "b"] } }
        },
        "required": ["picks"]
    });
    let body = one_field_body(&schema, "\"a\", \"b\"]}");
    assert_eq!(body["picks"], serde_json::json!(["a", "b"]));
}

#[test]
fn an_object_field_shapes_valid_json() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "meta": { "type": "object" } },
        "required": ["meta"]
    });
    // A bare `object` with no declared properties is any structurally-valid JSON
    // object — string-aware balancing keeps a `}` inside a string from closing it.
    let body = one_field_body(&schema, " {\"k\": \"a}b\"}}");
    assert_eq!(body["meta"], serde_json::json!({ "k": "a}b" }));
}

// ── nested object (recurse), enforced field-by-field ─────────────────────────

#[test]
fn a_nested_object_is_recursed_and_its_enum_is_enforced() {
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": { "mode": { "type": "string", "enum": ["read", "write"] } },
                "required": ["mode"]
            }
        },
        "required": ["config"]
    });
    let bodies = all_bodies(&schema, &v);
    let modes: BTreeSet<String> = bodies
        .iter()
        .map(|b| {
            b["config"]["mode"]
                .as_str()
                .expect("nested mode")
                .to_string()
        })
        .collect();
    assert_eq!(
        modes,
        BTreeSet::from(["read".to_string(), "write".to_string()]),
        "the nested enum is enforced exactly: {bodies:?}"
    );
}

// ── nullable ─────────────────────────────────────────────────────────────────

#[test]
fn a_nullable_string_enum_is_constrained_as_its_base_type() {
    // `["string","null"]` takes the non-null member as the enum's base type,
    // and — like every nullable field (`every_nullable_field_can_be_null`,
    // `edge_tests::choices`) — `null` is offered beside it as its own arm.
    // So the value is either the enum's own constrained set, or `null`.
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "who": { "type": ["string", "null"], "enum": ["ada", "bram"] }
        },
        "required": ["who"]
    });
    let vals: BTreeSet<Option<String>> = all_bodies(&schema, &v)
        .iter()
        .map(|b| b["who"].as_str().map(str::to_string))
        .collect();
    assert_eq!(
        vals,
        BTreeSet::from([Some("ada".to_string()), Some("bram".to_string()), None])
    );
}

// ── real effector schemas ────────────────────────────────────────────────────

/// The lift's `use` body, exactly as `lift.rs::schema` builds it: a `floor` enum
/// filled from `world.floor_names()`.
fn lift_use_body(floors: &[&str]) -> Value {
    serde_json::json!({
        "type": "object",
        "properties": { "floor": { "type": "string", "enum": floors } },
        "required": ["floor"]
    })
}

#[test]
fn the_lift_use_body_enumerates_exactly_the_worlds_floor_names() {
    let v = TestVocab::new();
    // A representative live floor set — the enum carried raw from `floor_names()`.
    let floors = ["command", "casting", "port", "the-drowned-level"];
    let bodies = all_bodies(&lift_use_body(&floors), &v);
    let got: BTreeSet<String> = bodies
        .iter()
        .map(|b| b["floor"].as_str().unwrap().to_string())
        .collect();
    let expected: BTreeSet<String> = floors.iter().map(|s| s.to_string()).collect();
    assert_eq!(got, expected, "the floor arms are the world's own, raw");
}

#[test]
fn the_lift_call_body_takes_no_fields() {
    // `call` takes no body — an empty-properties schema arms `{}` only.
    let v = TestVocab::new();
    let bodies = all_bodies(
        &serde_json::json!({ "type": "object", "properties": {}, "required": [] }),
        &v,
    );
    assert_eq!(bodies, vec![serde_json::json!({})]);
}

#[test]
fn a_station_verb_with_two_required_strings_round_trips() {
    // `chronicle add_entry`, as `station.rs::body_schema` builds it: `to`,`what`,
    // both required strings. Scripted with distinct values in declared order.
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "to": { "type": "string" }, "what": { "type": "string" } },
        "required": ["to", "what"]
    });
    let tree = body_tree(&schema, &v);
    // to = "the third era", what = "the year it fell" — two free-string spans.
    // The model writes its own opening quote (`Terminator::JsonStringValue`
    // distinguishes `""` from `"...` by that first byte), so each value's
    // script leads with `"`.
    let mut script: Vec<TokenId> = "\"the third era\"".bytes().map(|b| b as TokenId).collect();
    script.extend("\"the year it fell\"".bytes().map(|b| b as TokenId));
    let run = simulate(tree, &v, Oracle::Scripted(script), 5000).unwrap();
    let body = body_json(&run, &v);
    assert_eq!(body["to"], serde_json::json!("the third era"));
    assert_eq!(body["what"], serde_json::json!("the year it fell"));
}

// ── adversarial free-text ────────────────────────────────────────────────────

#[test]
fn a_string_value_with_escapes_does_not_close_early() {
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "note": { "type": "string" } },
        "required": ["note"]
    });
    let tree = body_tree(&schema, &v);
    // note = a\"b\\c — an escaped quote and an escaped backslash, then the
    // close. The model writes its own opening quote first (`Terminator::
    // JsonStringValue`).
    let script: Vec<TokenId> = "\"a\\\"b\\\\c\"".bytes().map(|b| b as TokenId).collect();
    let run = simulate(tree, &v, Oracle::Scripted(script), 5000).unwrap();
    let body = body_json(&run, &v);
    assert_eq!(body["note"], serde_json::json!("a\"b\\c"));
    assert_eq!(run.healed_bytes, 0);
}

#[test]
fn a_runaway_string_value_is_force_closed_and_still_parses() {
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "note": { "type": "string" } },
        "required": ["note"]
    });
    let tree = body_tree(&schema, &v);
    // The model writes its own opening quote first (`Terminator::
    // JsonStringValue`; any other first byte reads as skipping the value),
    // then non-closing bytes forever so json_string's forced_after force-closes.
    let mut wrote_quote = false;
    let run = simulate(
        tree,
        &v,
        Oracle::Policy(Box::new(move |_| {
            if !wrote_quote {
                wrote_quote = true;
                b'"' as TokenId
            } else {
                b'x' as TokenId
            }
        })),
        100_000,
    )
    .unwrap();
    assert_eq!(run.forced_closes, 1, "the runaway is force-closed once");
    assert!(run
        .observes
        .iter()
        .any(|o| matches!(o, Observe::SpanForcedClosed)));
    // The forced close writes the quote its terminator would have consumed, so
    // the body is still a parseable JSON object.
    let body = body_json(&run, &v);
    assert!(body["note"].is_string());
}

#[test]
fn an_out_of_mask_token_bails_without_panicking() {
    let v = TestVocab::new();
    // A body whose first decision is an enum branch; a byte not on any arm bails.
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "floor": { "type": "string", "enum": ["command", "port"] } },
        "required": ["floor"]
    });
    let tree = body_tree(&schema, &v);
    let run = simulate(tree, &v, Oracle::Scripted(vec![b'Z' as TokenId]), 100).unwrap();
    assert!(
        run.observes.contains(&Observe::Bailed),
        "an illegal first token bails: {:?}",
        run.observes
    );
}

// ── FreeTextLimits sanity: the body's spans carry a runaway guard ────────────

#[test]
fn every_free_span_in_a_body_has_a_forced_after_guard() {
    // A compiled body must satisfy the compile invariant `forced_after > 0` on
    // every free span; building one with a mix of free and typed fields proves
    // the compiler accepts it (a zero guard is a BuildError).
    let v = TestVocab::new();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "text": { "type": "string" },
            "count": { "type": "integer" }
        },
        "required": ["text", "count"]
    });
    // Compiles without error ⇒ the invariant held.
    let _ = body_tree(&schema, &v);
    // And the default json_string limit is a real guard.
    assert!(FreeTextLimits::json_string().forced_after > 0);
}
