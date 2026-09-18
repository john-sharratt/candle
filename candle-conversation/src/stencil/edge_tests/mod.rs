//! Adversarial inputs against the tool-call stencil: a model that writes the
//! wrong bracket, merges a delimiter into its value, stops mid-structure, or
//! writes something that is not JSON at all. Every case must come out as a call
//! that parses — the stencil's job is to bring it back to sane, not to hope.
//!
//! - `structure` — the grammar-written arrays and objects, and the delimiters a
//!   model gets wrong around them.
//! - `values` — the free spans inside them: empty, cut short, malformed.
//! - `intent` — repairs that keep the meaning (`True` is `true`), asserted on
//!   the parsed value under every way a tokenizer could cut the text.
//! - `choices` — closed sets and nullable fields in the grammar itself: offered
//!   under the mask, so a wrong spelling is never written at all.
//! - `fuzz` — seeded random adversaries over whole calls, checked against the
//!   schema's shape.

mod choices;
mod fuzz;
mod intent;
mod structure;
mod values;

use std::sync::Arc;

use serde_json::Value;

use super::compile::compile;
use super::session::Observe;
use super::sim::{simulate, Oracle, SimRun};
use super::tests::{bytes_of, cline_catalog, json_body};
use super::tool_call::{compile_tool_call_tree, ToolCallEnvelope, ToolSpec};
use super::tree::StencilTree;
use super::vocab::{TestVocab, TokenId};

/// The end token `TestVocab` treats as EOS.
const EOS: TokenId = 256;

/// SplitMix64: a deterministic generator with no dependency, so the seed alone
/// names a run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

/// Cline's two tools plus `pack`, whose `values` array has no element schema
/// and so stays one free value — the span the malformed-value cases attack.
fn catalog() -> Vec<ToolSpec> {
    let mut tools = cline_catalog();
    tools.push(ToolSpec::from_json_schema(
        "pack",
        &serde_json::json!({
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": true},
                "label": {"type": "string"}
            },
            "required": ["values"]
        }),
    ));
    tools
}

fn tree(v: &TestVocab) -> Arc<StencilTree> {
    let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen3()).unwrap();
    Arc::new(compile(&spec, v).unwrap())
}

/// One decode step of a script: a run of byte tokens, or one named token.
enum Step<'a> {
    Text(&'a str),
    Token(TokenId),
}

fn script(steps: &[Step]) -> Vec<TokenId> {
    steps
        .iter()
        .flat_map(|s| match s {
            Step::Text(t) => bytes_of(t),
            Step::Token(id) => vec![*id],
        })
        .collect()
}

/// Run `steps` to the end of the call and return the emitted text, asserting
/// the walk never bailed and the call parses.
fn run(v: &TestVocab, steps: &[Step]) -> (String, SimRun) {
    let sim = simulate(tree(v), v, Oracle::Scripted(script(steps)), 20_000)
        .unwrap_or_else(|e| panic!("simulation failed: {e}"));
    let text = sim.text(v);
    assert!(
        !sim.observes.contains(&Observe::Bailed),
        "the walk bailed: {text:?}"
    );
    parse(&text);
    (text, sim)
}

/// The call's JSON, or a panic naming the text that is not.
fn parse(text: &str) -> Value {
    serde_json::from_str(json_body(text)).unwrap_or_else(|e| panic!("not JSON: {text:?}: {e}"))
}

/// The emitted text of a whole call whose arguments are `args`.
fn call(name: &str, args: &str) -> String {
    format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {{{args}}}}}\n</tool_call>")
}
