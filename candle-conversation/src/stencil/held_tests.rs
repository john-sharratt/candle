//! A tool call held to the tools its turn was given
//! ([`TriggerRegistry::holding`](super::TriggerRegistry::holding)): the
//! catalog's grammar compiled once, and each name branch narrowed per turn.

use std::collections::BTreeSet;
use std::sync::Arc;

use super::compile::compile;
use super::driver::{StencilDriver, StepMask};
use super::held_arms::held_branches;
use super::named_arm::arm_name;
use super::tool_call::{compile_tool_call_tree, parse_tools, ToolCallEnvelope};
use super::tree::StencilTree;
use super::trigger::TriggerRegistry;
use super::vocab::{TestVocab, TokenId, Vocab};

const CATALOG: &str = r#"[
    {"name":"file_list","params":[{"name":"path","type":"string","required":true}]},
    {"name":"file_read","params":[{"name":"path","type":"string","required":true}]},
    {"name":"write","params":[
        {"name":"mode","type":"string","enum":["append","replace","write"],"required":true}]}]"#;

const TRIGGER: TokenId = 1000;

fn tree(v: &TestVocab) -> Arc<StencilTree> {
    let spec =
        compile_tool_call_tree(&parse_tools(CATALOG).unwrap(), &ToolCallEnvelope::qwen3()).unwrap();
    Arc::new(compile(&spec, v).unwrap())
}

/// The driver a call opens with: held to `tools`, or the whole catalog when
/// `None`.
fn driver(v: &TestVocab, tools: Option<&[&str]>) -> StencilDriver {
    let t = tree(v);
    let mut registry = TriggerRegistry::new();
    registry.register(TRIGGER, Arc::clone(&t));
    if let Some(tools) = tools {
        let names: Vec<String> = tools.iter().map(|s| s.to_string()).collect();
        let held = held_branches(&t, &names, |t| String::from_utf8(v.decode(t)).ok());
        registry = registry.holding(TRIGGER, Arc::new(held));
    }
    registry.driver_for(TRIGGER).unwrap()
}

fn name_in(text: &str) -> String {
    let key = "\"name\": \"";
    let at = text
        .find(key)
        .unwrap_or_else(|| panic!("no name in {text:?}"))
        + key.len();
    arm_name(&text.as_bytes()[at..]).to_string()
}

/// Every text a call can be written as up to its first free value (or its
/// end): each allowed token taken at every branch.
fn every_path(start: StencilDriver, v: &TestVocab) -> Vec<String> {
    let mut done = Vec::new();
    let mut stack = vec![(start, Vec::new())];
    while let Some((mut d, mut text)) = stack.pop() {
        match d.step() {
            StepMask::Prefill(run) => {
                text.extend(v.decode(&run));
                stack.push((d, text));
            }
            StepMask::Branch(set) => {
                for &t in set.tokens() {
                    let mut next = d.clone();
                    let mut written = text.clone();
                    written.extend(v.token_bytes(t));
                    next.accept(t, &v.token_bytes(t));
                    stack.push((next, written));
                }
            }
            StepMask::Free { .. } | StepMask::Done => done.push(String::from_utf8(text).unwrap()),
        }
        assert!(stack.len() < 10_000, "the walk did not converge");
    }
    done
}

fn names(paths: &[String]) -> BTreeSet<String> {
    paths.iter().map(|p| name_in(p)).collect()
}

/// **Unheld, a call can name any tool in the catalog; held, only the given
/// ones.** Every path through the call is walked, so a token that could lead
/// to an ungiven tool anywhere in the name would show up as its name here.
#[test]
fn a_held_call_can_name_only_a_given_tool() {
    let v = TestVocab::new();
    assert_eq!(
        names(&every_path(driver(&v, None), &v)),
        BTreeSet::from(["file_list".into(), "file_read".into(), "write".into()])
    );
    assert_eq!(
        names(&every_path(driver(&v, Some(&["write", "file_read"])), &v)),
        BTreeSet::from(["file_read".into(), "write".into()])
    );
    assert_eq!(
        names(&every_path(driver(&v, Some(&["file_list"])), &v)),
        BTreeSet::from(["file_list".into()])
    );
}

/// **The hold ends with its branch.** `write`'s `mode` enum — one of whose
/// values is spelled like a tool — still offers every value after a name held
/// to `write`.
#[test]
fn the_hold_ends_with_the_name_branch() {
    let v = TestVocab::new();
    let modes: BTreeSet<String> = every_path(driver(&v, Some(&["write"])), &v)
        .iter()
        .map(|p| {
            let key = "\"mode\": \"";
            let at = p.find(key).unwrap_or_else(|| panic!("no mode in {p:?}")) + key.len();
            arm_name(&p.as_bytes()[at..]).to_string()
        })
        .collect();
    assert_eq!(
        modes,
        BTreeSet::from(["append".into(), "replace".into(), "write".into()])
    );
}

/// **A call held to one tool writes it whatever is sampled**: with `file_read`
/// alone given, the lowest-token walk — which writes `file_list` unheld —
/// writes `file_read`.
#[test]
fn a_call_held_to_one_tool_writes_it() {
    let v = TestVocab::new();
    for (tools, want) in [(None, "file_list"), (Some(&["file_read"][..]), "file_read")] {
        let mut d = driver(&v, tools);
        let mut text = Vec::new();
        loop {
            match d.step() {
                StepMask::Prefill(run) => text.extend(v.decode(&run)),
                StepMask::Branch(set) => {
                    let t = set.tokens()[0];
                    text.extend(v.token_bytes(t));
                    d.accept(t, &v.token_bytes(t));
                }
                StepMask::Free { .. } | StepMask::Done => break,
            }
        }
        assert_eq!(name_in(&String::from_utf8(text).unwrap()), want);
    }
}
