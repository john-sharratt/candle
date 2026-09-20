//! Tool errors the model can correct itself from.
//!
//! The prompt carries full definitions for only the few tools provenance
//! selects; every other tool appears by name alone in the catalog listing. A
//! call to one of those is still dispatched, so the model's first attempt may
//! guess its arguments — and the error it gets back is the place to teach it.
//!
//! - `invalid_arguments` gains the tool's `parameters` schema, so the next
//!   attempt is written against the real signature rather than another guess.
//! - A name no tool answers to is refused with the nearest real names, so a
//!   misremembered name (`web_get`, `read_file`) lands on its tool one call
//!   later instead of ending the turn in "that tool does not exist". Only
//!   tools the caller's grants can run are suggested — a suggestion the next
//!   call would find `not_permitted` teaches nothing — and an alias close to
//!   the name suggests the tool it stands for.

use std::iter;

use serde_json::{json, Value};
use zend_tools::{registry, Grants};

use crate::tool_def;

/// Error code a call with unparseable or invalid arguments carries.
const INVALID_ARGUMENTS: &str = "invalid_arguments";

/// How many near names an unknown-tool refusal offers.
const SUGGESTIONS: usize = 3;

/// `response` to a call of `name`, with the tool's parameter schema attached
/// when the call's arguments were refused.
pub fn with_guidance(name: &str, mut response: Value) -> Value {
    if response.get("error").and_then(Value::as_str) != Some(INVALID_ARGUMENTS) {
        return response;
    }
    if let (Some(def), Some(obj)) = (tool_def::find(name), response.as_object_mut()) {
        obj.insert("parameters".to_string(), def.parameters.clone());
        obj.insert(
            "hint".to_string(),
            json!("call again with arguments that match `parameters`"),
        );
    }
    response
}

/// The refusal for a call to `name`, which no tool answers to, naming the
/// real tools nearest to it among those `grants` can run.
pub fn unknown_tool(name: &str, grants: Grants) -> Value {
    // Each candidate spelling (a canonical name or one of its aliases) with
    // the canonical name it stands for.
    let spellings: Vec<(&str, &str)> = registry::all_tools()
        .iter()
        .filter(|t| grants.require_all(t.requires).is_ok())
        .flat_map(|t| {
            iter::once(t.name)
                .chain(registry::aliases(t.name).iter().copied())
                .map(move |spelling| (spelling, t.name))
        })
        .collect();
    let near = nearest(name, &spellings);
    json!({
        "error": "unknown_tool",
        "detail": format!(
            "no tool named {name:?}; the nearest tools are {}",
            near.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ")
        ),
        "did_you_mean": near,
    })
}

/// Up to [`SUGGESTIONS`] distinct tools, nearest to `query` first. Each
/// `(spelling, tool)` is ranked by its spelling — most shared `_`-separated
/// words, then smallest edit distance — and suggests its tool.
fn nearest<'a>(query: &str, spellings: &[(&str, &'a str)]) -> Vec<&'a str> {
    let query = query.to_ascii_lowercase();
    let words: Vec<&str> = query
        .split(['_', '-', ' '])
        .filter(|w| !w.is_empty())
        .collect();
    let mut ranked: Vec<(usize, usize, &str)> = spellings
        .iter()
        .map(|&(spelling, tool)| {
            let shared = spelling.split('_').filter(|w| words.contains(w)).count();
            (shared, edit_distance(&query, spelling), tool)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(b.2)));
    let mut out: Vec<&str> = Vec::with_capacity(SUGGESTIONS);
    for (_, _, tool) in ranked {
        if !out.contains(&tool) {
            out.push(tool);
            if out.len() == SUGGESTIONS {
                break;
            }
        }
    }
    out
}

/// Levenshtein distance between `a` and `b`, over bytes (tool names are ASCII).
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, &ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, &cb) in b.iter().enumerate() {
            let substitute = prev[j] + usize::from(ca != cb);
            cur[j + 1] = substitute.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_distance_counts_single_edits() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("write", "write"), 0);
        assert_eq!(edit_distance("write", "wrote"), 1);
        assert_eq!(edit_distance("file_read", "file_reed"), 1);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    fn suggested(refusal: &Value) -> Vec<&str> {
        refusal["did_you_mean"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect()
    }

    /// **A misremembered name lands on its tool** — and a name close to an
    /// alias suggests the tool it stands for, once.
    #[test]
    fn an_unknown_tool_names_its_nearest_real_tools() {
        let refusal = unknown_tool("file_writ", Grants::ALL);
        assert_eq!(refusal["error"], "unknown_tool");
        let near = suggested(&refusal);
        assert_eq!(near.len(), SUGGESTIONS);
        assert_eq!(
            near[0], "write",
            "`file_write` is an alias of `write`: {near:?}"
        );

        let tools = |names: &[&'static str]| -> Vec<(&'static str, &'static str)> {
            names.iter().map(|&n| (n, n)).collect()
        };
        assert_eq!(
            nearest("weathr", &tools(&["weather", "write", "web_fetch"]))[0],
            "weather"
        );
        assert_eq!(
            nearest("web_get", &tools(&["weather", "web_fetch", "write"]))[0],
            "web_fetch"
        );
        assert_eq!(
            nearest(
                "x",
                &[("write", "write"), ("file_write", "write"), ("xy", "xy")]
            ),
            ["xy", "write"],
            "each tool once, however many spellings match"
        );
    }

    /// Only tools the caller can run are suggested.
    #[test]
    fn suggestions_are_tools_the_grants_can_run() {
        for tool in suggested(&unknown_tool("sub_runn", Grants::NONE)) {
            let t = registry::find(tool).unwrap();
            assert!(t.requires.is_empty(), "`{tool}` needs {:?}", t.requires);
        }
        assert!(suggested(&unknown_tool("sub_runn", Grants::ALL)).contains(&"sub_run"));
    }

    /// **A refused call learns the signature.** Any other response, success
    /// or a different error, is returned unchanged.
    #[test]
    fn refused_arguments_carry_the_tools_parameters() {
        let refused = json!({"error": "invalid_arguments", "detail": "missing field `city`"});
        let guided = with_guidance("weather", refused);
        assert_eq!(
            guided["parameters"],
            tool_def::find("weather").unwrap().parameters
        );
        assert!(guided["hint"].is_string());

        let other = json!({"error": "not_found", "detail": "x"});
        assert_eq!(with_guidance("weather", other.clone()), other);
        let ok = json!({"temperature": 12});
        assert_eq!(with_guidance("weather", ok.clone()), ok);
    }
}
