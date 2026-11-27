//! An act's list argument — the steps of a mission — as the grammar holds it,
//! as the routes publish it, and as the act reads it.
//!
//! **A list that must not be empty cannot be written empty.** `steps` was a
//! string read one to a line, and a free string may be `""`: asked to have a
//! Drone research a repository, Zen's conversation posted a brief with
//! `"steps": ""` twice where the `coding-task` job was the work, and the table
//! refused the post each time. Held as a list of at least one line, each
//! opening on something written, the grammar offers no `]` before the first
//! step and no closing quote before a step's first character.

use candle_conversation::stencil::{Param as StencilParam, ParamType, Pick};
use serde_json::{json, Map, Value};

/// The catalog's type name for a list argument.
pub const LIST: &str = "array";

/// The most entries a list argument holds. A mission longer than this is
/// several missions.
pub const LIST_MOST: usize = 12;

/// The most tokens one entry may run to: a step is a line, not a passage.
pub const LINE_TOKENS: u32 = 128;

/// What one entry may spell: a single line opening on something written. No
/// quote and no backslash, which a value held to a pattern never carries.
pub const LINE: &str = r#"[^\s"\\][^"\\\r\n]*"#;

/// The schema every entry of a list argument follows, for the grammar.
pub fn line_item() -> StencilParam {
    StencilParam {
        name: String::new(),
        ty: ParamType::String,
        required: true,
        enum_values: None,
        items: None,
        min_items: 0,
        max_items: None,
        properties: None,
        nullable: false,
        minimum: None,
        maximum: None,
        max_tokens: Some(LINE_TOKENS),
        requires: Vec::new(),
        shapes: Vec::new(),
        pick: Pick::None,
        pattern: Some(LINE.to_string()),
    }
}

/// A catalog parameter of type `ty` as a JSON Schema property: a list argument
/// with its entries' type and bounds, anything else by its type alone.
pub fn wire_type(ty: &str) -> Value {
    match ty {
        LIST => json!({
            "type": LIST,
            "items": {"type": "string", "pattern": LINE},
            "minItems": 1,
            "maxItems": LIST_MOST,
        }),
        other => json!({ "type": other }),
    }
}

/// The entries of list argument `key`, each trimmed, blanks left out.
/// `None` when it is absent, not a list, or holds nothing written.
pub fn list_arg(args: &Map<String, Value>, key: &str) -> Option<Vec<String>> {
    let entries: Vec<String> = args
        .get(key)?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    (!entries.is_empty()).then_some(entries)
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::{TestVocab, ValueRule};

    use super::*;

    fn args(v: Value) -> Map<String, Value> {
        v.as_object().unwrap().clone()
    }

    #[test]
    fn a_list_reads_its_written_entries_in_order() {
        let a = args(json!({"steps": [" go to the ward room ", "", "  ", "scan the panel"]}));
        assert_eq!(
            list_arg(&a, "steps"),
            Some(vec![
                "go to the ward room".to_string(),
                "scan the panel".to_string()
            ])
        );
    }

    #[test]
    fn a_list_with_nothing_written_is_no_list() {
        assert_eq!(list_arg(&args(json!({"steps": []})), "steps"), None);
        assert_eq!(list_arg(&args(json!({"steps": ["", " "]})), "steps"), None);
        assert_eq!(list_arg(&args(json!({"steps": "go"})), "steps"), None);
        assert_eq!(list_arg(&args(json!({})), "steps"), None);
    }

    #[test]
    fn a_list_is_published_with_its_bounds() {
        assert_eq!(
            wire_type(LIST),
            json!({
                "type": "array",
                "items": {"type": "string", "pattern": r#"[^\s"\\][^"\\\r\n]*"#},
                "minItems": 1,
                "maxItems": 12,
            })
        );
        assert_eq!(wire_type("string"), json!({"type": "string"}));
    }

    /// A pattern the stencil cannot hold to is left unenforced with a warning,
    /// so the line rule is built here, where a failure is a test.
    #[test]
    fn the_line_rule_builds() {
        ValueRule::new(LINE, &TestVocab::new()).expect("a step's pattern must be a rule");
        let item = line_item();
        assert_eq!(item.pattern.as_deref(), Some(LINE));
        assert_eq!(item.max_tokens, Some(128));
    }
}
