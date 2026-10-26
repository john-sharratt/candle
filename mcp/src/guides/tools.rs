//! `guide_list` and `guide_read`: the aggregator's own tools, answered here
//! rather than by a backend.

use mcp_server::protocol::text_result;
use serde_json::{json, Value};

use super::library::Library;

pub const LIST: &str = "guide_list";
pub const READ: &str = "guide_read";

/// Whether `name` is one of the guide tools.
pub fn is_guide_tool(name: &str) -> bool {
    name == LIST || name == READ
}

/// The two tools' definitions, the catalogue's names in `guide_list`'s.
pub fn definitions(library: &Library) -> Vec<Value> {
    vec![
        json!({
            "name": LIST,
            "description": format!(
                "List the guides: the methods for doing work well with these tools — delegating, \
                 waiting, verifying, research, code review, deploying, managing bots and more — each \
                 with when to use it. Call this once at the start of any conversation that uses \
                 these tools, before starting work, then read the guide for the work with \
                 guide_read. Guides: {}.",
                library.names().join(", ")
            ),
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
        }),
        json!({
            "name": READ,
            "description": "Read one guide: the method for a kind of work — how to brief a worker for \
                it, what to verify, what to report. Read the guide before starting work of its \
                kind; guide_list names them all.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The guide's name, as guide_list gives it."}
                },
                "required": ["name"],
                "additionalProperties": false
            }
        }),
    ]
}

/// Answer the guide tool `name` with `args`.
pub fn call(library: &Library, name: &str, args: &Value) -> Value {
    if name == LIST {
        return text_result(&library.catalogue(), false);
    }
    let wanted = args
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default();
    match library.get(wanted) {
        Some(g) => text_result(&g.body, false),
        None => text_result(
            &format!(
                "there is no guide `{wanted}`; the guides are: {}",
                library.names().join(", ")
            ),
            true,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::guides::Guide;

    fn library() -> Library {
        Library::new(vec![Guide {
            name: "research".into(),
            description: "A question answered from the web.".into(),
            tools: vec!["repo_task_start".into()],
            body: "# Research\n\nGather.".into(),
        }])
        .unwrap()
    }

    #[test]
    fn the_list_names_every_guide_and_says_to_call_it_first() {
        let defs = definitions(&library());
        assert_eq!(defs[0]["name"], LIST);
        let described = defs[0]["description"].as_str().unwrap();
        assert!(described.contains("Call this once at the start"));
        assert!(described.ends_with("Guides: research."));
        assert_eq!(defs[1]["name"], READ);
    }

    #[test]
    fn a_guide_is_read_by_name_and_an_unknown_one_names_the_rest() {
        let l = library();
        assert_eq!(
            call(&l, READ, &json!({"name": " research "})),
            text_result("# Research\n\nGather.", false)
        );
        assert_eq!(
            call(&l, READ, &json!({"name": "nope"})),
            text_result("there is no guide `nope`; the guides are: research", true)
        );
        assert!(call(&l, LIST, &json!({}))["content"][0]["text"]
            .as_str()
            .unwrap()
            .ends_with("- research — A question answered from the web."));
    }
}
