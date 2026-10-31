//! The flat `{name, params}` call form a `candle_conversation::stencil::
//! ToolSpec` deserializes from, built without an engine: every call an answer
//! is held to — a journal's entry, its fold, a memory journal's summary — is
//! assembled from these.

use serde_json::{json, Value};

pub fn text(name: &str, required: bool) -> Value {
    json!({ "name": name, "type": "string", "required": required })
}

/// A string the grammar closes once it runs to `max_tokens`.
pub fn text_upto(name: &str, required: bool, max_tokens: u32) -> Value {
    json!({ "name": name, "type": "string", "required": required, "max_tokens": max_tokens })
}

/// A string that is one of `values`, which the grammar offers and nothing else.
pub fn choice(name: &str, required: bool, values: &[String]) -> Value {
    json!({ "name": name, "type": "string", "required": required, "enum": values })
}

/// A list of `item`, between `min_items` and `max_items` long.
///
/// **The most is the grammar's to hold, not only the check's.** Held only by
/// [`crate::verify`], a writer that went on past it — a worker whose research
/// turned up fifteen findings — had its whole entry refused, every valid claim
/// with it, and with one round to write in, kept nothing at all. Offered no
/// element past the most, it closes the list and keeps the entry.
pub fn list_of(
    name: &str,
    required: bool,
    min_items: usize,
    max_items: usize,
    item: Value,
) -> Value {
    json!({
        "name": name,
        "type": "array",
        "required": required,
        "min_items": min_items,
        "max_items": max_items,
        "items": item,
    })
}

pub fn object(name: &str, required: bool, properties: Vec<Value>) -> Value {
    json!({ "name": name, "type": "object", "required": required, "properties": properties })
}

/// The call named `name`, taking `params`.
pub fn call(name: &str, params: Vec<Value>) -> Value {
    json!({ "name": name, "params": params })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_call_is_its_name_and_params() {
        let c = call(
            "x",
            vec![text("t", true), choice("c", false, &["a".into()])],
        );
        assert_eq!(
            c,
            json!({"name": "x", "params": [
                {"name": "t", "type": "string", "required": true},
                {"name": "c", "type": "string", "required": false, "enum": ["a"]},
            ]})
        );
        let l = list_of("l", true, 1, 3, object("", true, vec![]));
        assert_eq!(
            l,
            json!({"name": "l", "type": "array", "required": true, "min_items": 1,
                   "max_items": 3, "items": {"name": "", "type": "object",
                   "required": true, "properties": []}})
        );
    }
}
