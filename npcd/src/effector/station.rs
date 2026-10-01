//! What a placed station affords: the verbs mounted under its url.
//!
//! A verb is an act that has moved onto the device — a [`tools::routed`] tool —
//! offered at the parts its `at` names. Its name is the verb's name, and its
//! parameters are the verb's body: [`body_schema`] writes them as the JSON-Schema
//! object an `OPTIONS` answer carries, which is what types the `invoke` a
//! character writes next.

use serde_json::{json, Map, Value};

use crate::engine::tools::{self, Tool, CATALOG};

/// The routed tools offered at a part, in catalog order.
fn tools_at(part_id: &str) -> impl Iterator<Item = &'static Tool> + '_ {
    CATALOG
        .iter()
        .filter(move |t| tools::routed(t.name) && t.at.contains(&part_id))
}

/// The verb names a part affords. A seat, which only holds somebody, has none.
pub fn verbs_of(part_id: &str) -> Vec<&'static str> {
    tools_at(part_id).map(|t| t.name).collect()
}

/// The tool a verb of a part is, `None` when the part does not afford it.
pub fn tool_of(part_id: &str, verb: &str) -> Option<&'static Tool> {
    tools_at(part_id).find(|t| t.name == verb)
}

/// The JSON-Schema object a tool's parameters make: what the verb's `invoke`
/// body must be.
pub fn body_schema(tool: &Tool) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for param in tool.params {
        properties.insert(
            param.name.to_string(),
            json!({ "type": param.ty, "description": param.description }),
        );
        if param.required {
            required.push(param.name);
        }
    }
    json!({ "type": "object", "properties": properties, "required": required })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_command_table_affords_the_mission_verbs() {
        assert_eq!(
            verbs_of("order-table"),
            vec!["collect_mission", "report_done", "report_stuck"]
        );
    }

    #[test]
    fn a_seat_affords_nothing() {
        assert!(verbs_of("seat").is_empty());
        assert!(verbs_of("not-a-part").is_empty());
    }

    #[test]
    fn a_verb_is_found_only_at_the_part_that_affords_it() {
        assert!(tool_of("order-table", "report_done").is_some());
        assert!(tool_of("order-table", "tell").is_none());
        assert!(tool_of("seat", "report_done").is_none());
    }

    #[test]
    fn a_verb_without_parameters_has_an_empty_body() {
        let tool = tool_of("order-table", "collect_mission").unwrap();
        assert_eq!(
            body_schema(tool),
            json!({ "type": "object", "properties": {}, "required": [] })
        );
    }

    #[test]
    fn a_verbs_parameters_are_its_body() {
        let tool = tool_of("order-table", "report_stuck").unwrap();
        let schema = body_schema(tool);
        assert_eq!(schema["required"], json!(["why"]));
        assert_eq!(schema["properties"]["why"]["type"], "string");
        assert!(schema["properties"]["why"]["description"]
            .as_str()
            .unwrap()
            .starts_with("What stopped you"));
    }
}
