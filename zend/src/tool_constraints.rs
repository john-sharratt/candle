//! What the grammar holds a call to, beyond the shape a tool's schema gives.
//!
//! `prompts/tool_constraints.yaml` overlays a tool's `parameters` for the stencil
//! alone. The prompt and [`ToolDef::calibration_marker`] read `parameters` as the
//! tool's file has it: written there, a constraint would grow every prompt and
//! send each tool it touched back through calibration on both machines.
//!
//! [`ToolDef::calibration_marker`]: crate::tool_def::ToolDef::calibration_marker

use std::sync::OnceLock;

use serde_json::{Map, Value};

/// The overlays, by tool.
const CONSTRAINTS: &str = include_str!("prompts/tool_constraints.yaml");

fn overlays() -> &'static Map<String, Value> {
    static OVERLAYS: OnceLock<Map<String, Value>> = OnceLock::new();
    OVERLAYS
        .get_or_init(|| serde_yaml::from_str(CONSTRAINTS).expect("tool_constraints.yaml parses"))
}

/// `parameters` as the grammar reads it for `tool`.
pub fn grammar_schema(tool: &str, parameters: &Value) -> Value {
    let mut schema = parameters.clone();
    if let Some(overlay) = overlays().get(tool) {
        merge(&mut schema, overlay);
    }
    schema
}

/// Lay `overlay` over `base`: objects field by field, an `allOf` added to what
/// is there, anything else replaced.
fn merge(base: &mut Value, overlay: &Value) {
    let (Value::Object(base), Value::Object(overlay)) = (base, overlay) else {
        return;
    };
    for (key, value) in overlay {
        match (base.get_mut(key), value) {
            (Some(Value::Array(rules)), Value::Array(more)) if key == "allOf" => {
                rules.extend(more.iter().cloned());
            }
            (Some(existing @ Value::Object(_)), Value::Object(_)) => merge(existing, value),
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_conversation::stencil::{ParamType, ToolSpec};
    use regex::Regex;
    use serde_json::json;

    use super::*;
    use crate::tool_def;
    use crate::tools::extract_tool_calls;

    /// **An overlay reaches only the grammar**: the prompt's line for a
    /// constrained tool is what its own file says.
    #[test]
    fn the_prompt_and_calibration_never_see_a_constraint() {
        for def in tool_def::all() {
            if overlays().contains_key(&def.name) {
                let line = def.json_line();
                assert!(!line.contains("\"pattern\""), "{}: {line}", def.name);
            }
        }
    }

    /// **Every overlay names a tool that exists and fields it has**: a renamed
    /// field would otherwise leave a constraint that silently does nothing.
    #[test]
    fn every_overlay_names_a_real_tool_and_real_fields() {
        for (tool, overlay) in overlays() {
            let def = tool_def::all()
                .iter()
                .find(|d| d.name == *tool)
                .unwrap_or_else(|| panic!("{tool} is not a tool"));
            for field in overlay["properties"].as_object().into_iter().flatten() {
                assert!(
                    def.parameters["properties"].get(field.0).is_some(),
                    "{tool} has no field {}",
                    field.0
                );
            }
            for (name, schema) in overlay["definitions"].as_object().into_iter().flatten() {
                for field in schema["properties"].as_object().into_iter().flatten() {
                    let known = &def.parameters["definitions"][name]["properties"];
                    assert!(
                        known.get(field.0).is_some(),
                        "{tool}: {name} has no field {}",
                        field.0
                    );
                }
            }
        }
    }

    /// Every pattern compiles, and to a string field with no enum: a pattern
    /// anywhere else is ignored by the grammar.
    #[test]
    fn every_pattern_lands_on_a_free_string() {
        for (tool, overlay) in overlays() {
            let def = tool_def::all().iter().find(|d| d.name == *tool).unwrap();
            let spec = ToolSpec::from_json_schema(tool, &grammar_schema(tool, &def.parameters));
            for (field, schema) in overlay["properties"].as_object().into_iter().flatten() {
                let Some(pattern) = schema.get("pattern").and_then(Value::as_str) else {
                    continue;
                };
                Regex::new(pattern).unwrap_or_else(|e| panic!("{tool}.{field}: {e}"));
                let param = spec.params.iter().find(|p| p.name == *field).unwrap();
                assert_eq!(param.ty, ParamType::String, "{tool}.{field}");
                assert_eq!(
                    param.pattern.as_deref(),
                    Some(pattern),
                    "{tool}.{field} would be unenforced"
                );
            }
        }
    }

    /// The patterns at one level of `overlay`, applied to `args`.
    fn check(tool: &str, overlay: &Value, args: &Value, failures: &mut Vec<String>) {
        for (field, schema) in overlay["properties"].as_object().into_iter().flatten() {
            let (Some(pattern), Some(value)) = (
                schema.get("pattern").and_then(Value::as_str),
                args.get(field).and_then(Value::as_str),
            ) else {
                continue;
            };
            let whole = Regex::new(&format!("^(?:{pattern})$")).unwrap();
            if !whole.is_match(value) {
                failures.push(format!("{tool}.{field}: {value:?} is not {pattern}"));
            }
        }
    }

    /// **Every recorded call still decodes.** The catalog's worked examples are
    /// what real calls to these tools look like, so a constraint one of them
    /// breaks is taking something away, however reasonable it read.
    #[test]
    fn every_recorded_call_keeps_to_the_constraints() {
        let mut failures = Vec::new();
        let mut checked = 0;
        for def in tool_def::all() {
            let Some(overlay) = overlays().get(&def.name) else {
                continue;
            };
            for example in &def.examples {
                let text = example.replace("<|projection|>", "");
                for call in extract_tool_calls(&text) {
                    if call.name == def.name {
                        checked += 1;
                        check(&def.name, overlay, &call.arguments, &mut failures);
                    }
                }
            }
        }
        assert!(checked > 40, "only {checked} calls were checked");
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// An object is laid over field by field, and an `allOf` adds to the
    /// tool's own rules rather than replacing them.
    #[test]
    fn an_overlay_adds_to_what_a_schema_has() {
        let mut base = json!({
            "properties": {"a": {"type": "string", "description": "d"}},
            "allOf": [{"if": 1}],
        });
        merge(
            &mut base,
            &json!({
                "properties": {"a": {"pattern": "x"}},
                "allOf": [{"if": 2}],
            }),
        );
        assert_eq!(
            base,
            json!({
                "properties": {"a": {"type": "string", "description": "d", "pattern": "x"}},
                "allOf": [{"if": 1}, {"if": 2}],
            })
        );
    }
}
