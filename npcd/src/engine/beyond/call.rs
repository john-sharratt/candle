//! Where each act beyond the world runs, and the arguments it runs with: its
//! tool's own, each turned back from how the grammar carried it
//! ([`super::carry`]). `Err` says what the act's arguments did not hold.

use serde_json::{Map, Value};

use super::acts::{self, Reached};
use super::carry::back;
use super::notes::{NotesCall, LIST_MOST, SEARCH_MOST};

/// A call of a tool's, with its arguments as the tool takes them.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub tool: &'static str,
    pub args: Value,
}

/// Where an act runs.
#[derive(Clone, Debug, PartialEq)]
pub enum Call {
    /// zend's tool, run here: the world outside, a sum, the guides, and the
    /// character's own journal.
    Here(ToolCall),
    /// The world's notebook.
    Notebook(NotesCall),
    /// npcd's own MCP tool, called as Zen: the pictures.
    Npcd(ToolCall),
}

/// The acts npcd's own MCP server answers.
const NPCD_TOOLS: [&str; 2] = ["image_generate", "image_edit"];

/// The call act `tool` makes with `args`.
pub fn call_of(tool: &str, args: &Map<String, Value>) -> Result<Call, String> {
    let act = acts::of(tool).ok_or_else(|| format!("`{tool}` is no act beyond the world."))?;
    let args = carried_back(act, args)?;
    let name = act.tool.name;
    if name.starts_with("notes_") {
        return notes_call(name, &args).map(Call::Notebook);
    }
    let call = ToolCall {
        tool: name,
        args: Value::Object(args),
    };
    Ok(match NPCD_TOOLS.contains(&name) {
        true => Call::Npcd(call),
        false => Call::Here(call),
    })
}

/// `args` as `act`'s tool takes them: each under its tool's name for it,
/// turned back from how it was carried, an argument the act does not declare
/// left out.
fn carried_back(act: &Reached, args: &Map<String, Value>) -> Result<Map<String, Value>, String> {
    let mut out = Map::new();
    for c in &act.carries {
        if let Some(v) = args.get(c.said).filter(|v| !v.is_null()) {
            out.insert(c.param.to_string(), back(c.carry, c.said, v)?);
        }
    }
    Ok(out)
}

/// The notebook's call for `notes_*` act `name`, from its tool's arguments.
fn notes_call(name: &str, args: &Map<String, Value>) -> Result<NotesCall, String> {
    let text = |key: &str| {
        args.get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string()
    };
    let need = |key: &str| {
        Some(text(key))
            .filter(|v| !v.is_empty())
            .ok_or_else(|| format!("Say the `{key}`."))
    };
    let tags: Vec<String> = args
        .get("tags")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect();
    let most = |default: usize| {
        args.get("max_results")
            .and_then(Value::as_u64)
            .map_or(default, |n| n as usize)
    };
    Ok(match name {
        "notes_write" => NotesCall::Write {
            key: need("key")?,
            content: text("content"),
            tags,
        },
        "notes_read" => NotesCall::Read { key: need("key")? },
        "notes_search" => {
            let query = text("query");
            if query.is_empty() && tags.is_empty() {
                return Err("Say what to search for: the `query`, or `tags`.".into());
            }
            NotesCall::Search {
                query,
                tags,
                most: most(SEARCH_MOST),
            }
        }
        "notes_list" => NotesCall::List {
            prefix: text("prefix"),
            tags,
            most: most(LIST_MOST),
        },
        other => return Err(format!("The notebook has no `{other}`.")),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn call(tool: &str, v: Value) -> Result<Call, String> {
        call_of(tool, v.as_object().unwrap())
    }

    fn here(tool: &'static str, args: Value) -> Result<Call, String> {
        Ok(Call::Here(ToolCall { tool, args }))
    }

    /// **Each act is its tool, with its tool's own arguments**: what the
    /// grammar carried as a string turned back into the number or the object
    /// the tool takes, and nothing the tool does not declare.
    #[test]
    fn each_act_is_its_tool_with_its_own_arguments() {
        assert_eq!(
            call("weather", json!({"location": "Sydney", "forecast_days": 2})),
            here("weather", json!({"location": "Sydney", "forecast_days": 2}))
        );
        assert_eq!(
            call(
                "unit_convert",
                json!({"value": "72", "from": "fahrenheit", "to": "celsius", "extra": 1})
            ),
            here(
                "unit_convert",
                json!({"value": 72.0, "from": "fahrenheit", "to": "celsius"})
            )
        );
        assert_eq!(
            call(
                "http_request",
                json!({"session_id": "s1", "path": "https://api.example.com/x",
                       "method": "POST", "headers": "Accept: application/json"})
            ),
            here(
                "http_request",
                json!({"session_id": "s1", "path": "https://api.example.com/x",
                       "method": "POST", "headers": {"Accept": "application/json"}})
            )
        );
        assert_eq!(
            call("journal_read", json!({"journal": "self", "entry": 3})),
            here("journal_read", json!({"journal": "self", "entry": 3}))
        );
        assert_eq!(call("guide_list", json!({})), here("guide_list", json!({})));
        assert_eq!(
            call("guide_read", json!({"guide": "deploy"})),
            here("guide_read", json!({"name": "deploy"}))
        );
    }

    /// **What was carried and does not spell its kind is said**, and an act
    /// that is none.
    #[test]
    fn what_an_act_did_not_hold_is_said() {
        assert_eq!(
            call(
                "unit_convert",
                json!({"value": "ten", "from": "m", "to": "ft"})
            ),
            Err("`value` is a number, not \"ten\".".into())
        );
        assert_eq!(
            call("cluster_status", json!({})),
            Err("`cluster_status` is no act beyond the world.".into())
        );
        assert_eq!(
            call("notes_search", json!({})),
            Err("Say what to search for: the `query`, or `tags`.".into())
        );
    }

    /// **The pictures are npcd's own tools, and the notes the world's
    /// notebook**, each with its tool's arguments.
    #[test]
    fn pictures_go_to_npcd_and_notes_to_the_notebook() {
        assert_eq!(
            call("image_generate", json!({"prompt": "a moon", "width": 768})),
            Ok(Call::Npcd(ToolCall {
                tool: "image_generate",
                args: json!({"prompt": "a moon", "width": 768})
            }))
        );
        assert_eq!(
            call(
                "notes_write",
                json!({"key": "house/wifi", "content": " Moonbase ", "tags": ["house"]})
            ),
            Ok(Call::Notebook(NotesCall::Write {
                key: "house/wifi".into(),
                content: "Moonbase".into(),
                tags: vec!["house".into()],
            }))
        );
        assert_eq!(
            call("notes_list", json!({"prefix": "house/", "max_results": 5})),
            Ok(Call::Notebook(NotesCall::List {
                prefix: "house/".into(),
                tags: Vec::new(),
                most: 5,
            }))
        );
        assert_eq!(
            call("notes_search", json!({"query": "router"})),
            Ok(Call::Notebook(NotesCall::Search {
                query: "router".into(),
                tags: Vec::new(),
                most: 10,
            }))
        );
    }
}
