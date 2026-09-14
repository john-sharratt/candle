//! OpenAI tool calling on the passthrough, in Qwen3's JSON-block format.
//!
//! A client that runs its own tools (Cline) sends their definitions in the
//! request's `tools` array, expects the model's calls back as `tool_calls`, and
//! returns each call and its result in the next request's history. The model
//! knows tools only as text: the definitions in a `# Tools` block closing the
//! system prompt, each call a `<tool_call>{"name": …, "arguments": {…}}</tool_call>`
//! block in its reply, each result a `<tool_response>` in the following user
//! turn. This module is the translation both ways.
//!
//! The JSON block rather than this lineage's function element
//! (`<function=…><parameter=…>`), whose training reads weaker on this
//! checkpoint. A function element the model writes anyway is still read, typed
//! through the client's own parameter schemas.

use serde_json::Value;

use candle_conversation::stencil::{function_blocks_to_json, ToolSpec};
use candle_conversation::TurnText;

use crate::lenient_json::repair_escapes;
use crate::types::ResponseFunction;

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

/// One call the model made.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub name: String,
    pub arguments: Value,
}

/// The `# Tools` block Qwen3's chat template closes the system prompt with.
pub fn tools_prompt(tools: &[Value]) -> String {
    let mut s = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for tool in tools {
        s.push('\n');
        s.push_str(&tool.to_string());
    }
    s.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and \
         arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    s
}

/// A call as the model writes it. `arguments` is the object, or OpenAI's wire
/// form of it — the object serialized as a string.
pub fn render_call(name: &str, arguments: &Value) -> String {
    let arguments = match arguments {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| arguments.clone()),
        other => other.clone(),
    };
    format!(
        "{CALL_OPEN}\n{{\"name\": {}, \"arguments\": {arguments}}}\n{CALL_CLOSE}",
        Value::String(name.to_string())
    )
}

/// A tool result as the model reads it: the wrapper as markup, and the content
/// — whatever the tool returned, a file's text included — literal, so no part
/// of it can become a control token.
pub fn render_response(content: &str) -> TurnText {
    TurnText::markup("<tool_response>")
        .then_literal(format!("\n{content}\n"))
        .then_markup("</tool_response>")
}

/// The client's tool definitions as the specs that type a function element's
/// arguments. A definition with no name is skipped.
pub fn specs(tools: &[Value]) -> Vec<ToolSpec> {
    tools
        .iter()
        .filter_map(|tool| {
            let function = tool.get("function")?;
            let name = function.get("name")?.as_str()?;
            let schema = function.get("parameters").unwrap_or(&Value::Null);
            Some(ToolSpec::from_json_schema(name, schema))
        })
        .collect()
}

/// Read one `<tool_call>` block's body — the text between its markers — as a
/// call, or `None` when it is not one. A body that fails to parse only for a
/// raw Windows path in a string is read with that path's backslashes literal
/// ([`repair_escapes`]).
pub fn parse_call(body: &str, specs: &[ToolSpec]) -> Option<Call> {
    let translated = function_blocks_to_json(body, specs);
    let json = translated.as_deref().unwrap_or(body).trim();
    let value: Value = serde_json::from_str(json)
        .ok()
        .or_else(|| serde_json::from_str(&repair_escapes(json)?).ok())?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = match value.get("arguments") {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(v) => v.clone(),
        None => Value::Object(Default::default()),
    };
    Some(Call { name, arguments })
}

/// Split a reply into its prose and its calls, in order. A block that does not
/// read as a call stays in the prose as the model wrote it, as does an unclosed
/// one.
pub fn split_calls(text: &str, specs: &[ToolSpec]) -> (String, Vec<Call>) {
    let mut prose = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find(CALL_OPEN) {
        let after = &rest[open + CALL_OPEN.len()..];
        let Some(close) = after.find(CALL_CLOSE) else {
            break;
        };
        let block_end = open + CALL_OPEN.len() + close + CALL_CLOSE.len();
        match parse_call(&after[..close], specs) {
            Some(call) => {
                prose.push_str(&rest[..open]);
                calls.push(call);
            }
            None => prose.push_str(&rest[..block_end]),
        }
        rest = &rest[block_end..];
    }
    prose.push_str(rest);
    (prose, calls)
}

/// An assistant reply as compared across requests: its prose trimmed, then
/// each call in one fixed rendering — so the call a client sends back matches
/// the one the model wrote, whatever layout and key order each carried.
pub fn canonical(text: &str) -> String {
    let (prose, calls) = split_calls(text, &[]);
    let mut out = prose.trim().to_string();
    for call in calls {
        out.push('\n');
        out.push_str(&render_call(&call.name, &call.arguments));
    }
    out
}

/// A call in OpenAI's wire form: the arguments object serialized.
pub fn wire_function(call: &Call) -> ResponseFunction {
    ResponseFunction {
        name: call.name.clone(),
        arguments: call.arguments.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::stencil::ParamType;
    use serde_json::json;

    #[test]
    fn a_call_renders_as_the_json_block_the_model_writes() {
        assert_eq!(
            render_call("read_files", &json!({"paths": ["a.md"]})),
            "<tool_call>\n{\"name\": \"read_files\", \"arguments\": {\"paths\":[\"a.md\"]}}\n</tool_call>"
        );
        // OpenAI's wire form — the arguments serialized — renders the same.
        assert_eq!(
            render_call(
                "read_files",
                &Value::String(r#"{"paths": ["a.md"]}"#.into())
            ),
            render_call("read_files", &json!({"paths": ["a.md"]}))
        );
    }

    #[test]
    fn the_tools_block_lists_each_definition_on_its_own_line() {
        let prompt = tools_prompt(&[
            json!({"type": "function", "function": {"name": "a"}}),
            json!({"type": "function", "function": {"name": "b"}}),
        ]);
        assert!(prompt.starts_with("# Tools\n\n"));
        assert!(prompt.contains(
            "<tools>\n{\"function\":{\"name\":\"a\"},\"type\":\"function\"}\n\
             {\"function\":{\"name\":\"b\"},\"type\":\"function\"}\n</tools>"
        ));
        assert!(prompt.ends_with(
            "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
        ));
    }

    /// The grammar's inputs: each named definition becomes a spec with exactly
    /// the parameters its schema declares — which is what makes an undeclared
    /// argument (`commands2`) unwritable once a call opens.
    #[test]
    fn the_clients_schemas_become_the_call_grammars_specs() {
        let specs = specs(&[
            json!({"type": "function", "function": {
                "name": "run_commands",
                "parameters": {
                    "type": "object", "additionalProperties": false,
                    "properties": {"commands": {"type": "array", "items": {"type": "string"}}},
                    "required": ["commands"]
                }
            }}),
            json!({"type": "function"}),
        ]);
        assert_eq!(specs.len(), 1, "a definition without a name is skipped");
        assert_eq!(specs[0].name, "run_commands");
        let params: Vec<(&str, bool, ParamType)> = specs[0]
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.required, p.ty))
            .collect();
        assert_eq!(params, [("commands", true, ParamType::Array)]);
    }

    #[test]
    fn calls_split_from_the_prose_around_them() {
        let (prose, calls) = split_calls(
            "Reading.\n\n<tool_call>\n{\"name\": \"read_files\", \"arguments\": {\"paths\": [\"a.md\"]}}\n</tool_call>",
            &[],
        );
        assert_eq!(prose, "Reading.\n\n");
        assert_eq!(
            calls,
            vec![Call {
                name: "read_files".into(),
                arguments: json!({"paths": ["a.md"]}),
            }]
        );
    }

    /// The call a Cline turn ended on: raw Windows paths in the arguments. It
    /// reads as a call carrying the paths the model meant, not as prose.
    #[test]
    fn a_call_with_raw_windows_paths_is_a_call() {
        let text = "<tool_call>\n{\"name\": \"run_commands\", \"arguments\": {\"commands\":  \
                    [\"Get-ChildItem -Path c:/Users/johna/prog/candle -Recurse -Depth 2\", \
                    \"dir C:\\Users\\johna\\prog\\candle\\*.md /B\"]}}\n</tool_call>";
        let (prose, calls) = split_calls(text, &[]);
        assert_eq!(prose, "");
        assert_eq!(
            calls,
            vec![Call {
                name: "run_commands".into(),
                arguments: json!({"commands": [
                    "Get-ChildItem -Path c:/Users/johna/prog/candle -Recurse -Depth 2",
                    r"dir C:\Users\johna\prog\candle\*.md /B",
                ]}),
            }]
        );
    }

    #[test]
    fn a_block_that_is_not_a_call_stays_prose() {
        let text = "a <tool_call>not json</tool_call> b";
        assert_eq!(split_calls(text, &[]), (text.to_string(), Vec::new()));
        let unclosed = "a <tool_call>{\"name\": \"x\"";
        assert_eq!(
            split_calls(unclosed, &[]),
            (unclosed.to_string(), Vec::new())
        );
    }

    /// The lineage's own element, typed through the client's schema: an array
    /// parameter arrives as an array, not as the text of one.
    #[test]
    fn a_function_element_is_read_through_the_clients_schema() {
        let tools = [json!({"type": "function", "function": {
            "name": "read_files",
            "parameters": {"type": "object", "properties": {"paths": {"type": "array"}}}
        }})];
        let text = "<tool_call>\n<function=read_files>\n<parameter=paths>\n[\"a.md\"]\n</parameter>\n</function>\n</tool_call>";
        let (prose, calls) = split_calls(text, &specs(&tools));
        assert_eq!(prose, "");
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].name, "read_files");
        assert_eq!(calls[0].arguments, json!({"paths": ["a.md"]}));
    }

    #[test]
    fn a_call_the_client_sends_back_matches_the_one_the_model_wrote() {
        let model = "Reading.\n\n<tool_call>\n{\"arguments\": {\"paths\": [\"a.md\"]}, \"name\": \"read_files\"}\n</tool_call>\n";
        let client = format!(
            "Reading.\n{}",
            render_call("read_files", &Value::String(r#"{"paths":["a.md"]}"#.into()))
        );
        assert_eq!(canonical(model), canonical(&client));
    }
}
