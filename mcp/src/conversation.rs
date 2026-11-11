//! A call's conversation, as the aggregator reads it (`docs/private_mode.md`
//! §3): the `ref` every tool but `get_user_context` requires, taken out of
//! the arguments and carried to the backend in `_meta`, and the schema field
//! every listed tool is given for it.

use serde_json::{json, Value};

use crate::context::GET;

/// The argument a conversation's ref is passed in.
pub const REF: &str = "ref";

/// What a call naming no ref is answered.
pub const NO_REF: &str = "this call names no `ref`: call get_user_context first — the first line \
    it answers is this conversation's ref — then pass it as `ref` on every Zen call";

/// What a call naming a ref not issued to its caller is answered.
pub const FORGED: &str = "that `ref` is not one this server issued to you: call get_user_context \
    without `ref` to begin a conversation, and pass the ref it answers";

/// What a call naming an expired ref is answered.
pub const EXPIRED: &str = "this conversation's `ref` has expired (unused for four hours): call \
    get_user_context with it to renew it, and pass the ref that answers from then on";

/// The ref `args` name, and the arguments without it — what the backend's
/// tool is called with.
pub fn take_ref(args: &Value) -> (Option<String>, Value) {
    let mut rest = args.clone();
    let named = rest
        .as_object_mut()
        .and_then(|fields| fields.remove(REF))
        .and_then(|v| v.as_str().map(str::trim).map(str::to_string))
        .filter(|s| !s.is_empty());
    (named, rest)
}

/// The schema field every tool but `get_user_context` takes its ref in.
fn ref_field() -> Value {
    json!({
        "type": "string",
        "description": "This conversation's ref: the first line get_user_context answered. Every call \
            needs it."
    })
}

/// Give every tool in `tools` but `get_user_context` — whose own schema
/// names its `ref` — a required `ref`.
pub fn require_ref(tools: &mut [Value]) {
    for tool in tools {
        if tool.get("name").and_then(Value::as_str) == Some(GET) {
            continue;
        }
        let Some(schema) = tool.get_mut("inputSchema").and_then(Value::as_object_mut) else {
            continue;
        };
        let properties = schema.entry("properties").or_insert_with(|| json!({}));
        if let Some(p) = properties.as_object_mut() {
            p.insert(REF.into(), ref_field());
        }
        let required = schema.entry("required").or_insert_with(|| json!([]));
        if let Some(r) = required.as_array_mut() {
            if !r.iter().any(|v| v == REF) {
                r.push(json!(REF));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ref_is_taken_out_of_the_arguments() {
        let (named, rest) = take_ref(&json!({"ref": " r1 ", "journal": "here"}));
        assert_eq!(named.as_deref(), Some("r1"));
        assert_eq!(rest, json!({"journal": "here"}));
        assert_eq!(take_ref(&json!({"ref": ""})).0, None);
        assert_eq!(take_ref(&json!({"ref": 3})), (None, json!({})));
        assert_eq!(take_ref(&json!({})).0, None);
        assert_eq!(take_ref(&json!(null)), (None, json!(null)));
    }

    /// **Every tool but `get_user_context` requires `ref`**, beside what it
    /// already took, and a schema that already requires it is left so.
    #[test]
    fn every_tool_but_get_user_context_requires_a_ref() {
        let mut tools = vec![
            json!({"name": GET, "inputSchema": {"type": "object", "properties": {"ref": {"type": "string"}}}}),
            json!({"name": "journal_read", "inputSchema": {"type": "object",
                "properties": {"journal": {"type": "string"}}, "required": ["journal"],
                "additionalProperties": false}}),
            json!({"name": "guide_list", "inputSchema": {"type": "object", "properties": {}}}),
            json!({"name": "bare", "inputSchema": {"type": "object"}}),
        ];
        require_ref(&mut tools);
        require_ref(&mut tools);
        assert_eq!(tools[0]["inputSchema"].get("required"), None);
        assert_eq!(
            tools[1]["inputSchema"]["required"],
            json!(["journal", "ref"])
        );
        assert_eq!(tools[1]["inputSchema"]["properties"]["ref"], ref_field());
        assert_eq!(tools[1]["inputSchema"]["additionalProperties"], false);
        assert_eq!(tools[2]["inputSchema"]["required"], json!(["ref"]));
        assert_eq!(tools[3]["inputSchema"]["properties"]["ref"], ref_field());
    }
}
