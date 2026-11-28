//! How a tool's parameter travels through a character's grammar, and back to
//! the tool (`docs/zen_moon_mind.md` §16).
//!
//! A definition's parameters are JSON Schema. The grammar bounds four kinds
//! of value — a string, an integer, a boolean and a list of lines
//! (`tools::tests_grammar::every_parameter_is_a_type_the_grammar_can_bound`) —
//! because a `number`, an `object` or any JSON value compiles to a tree that
//! never ends. So each parameter is carried as the nearest of the four, and
//! turned back into what the tool takes when the act is performed:
//!
//! | Schema | Carried as | Back |
//! |---|---|---|
//! | `string`, `integer`, `boolean` | itself | as it is |
//! | an array of strings | a list | as it is |
//! | `number` | a string, `"72.5"` | the number it spells |
//! | an object of strings (headers, a query) | a string, one `name: value` to a line | the object |
//! | anything else | a string of JSON | the JSON it spells |
//!
//! A `null` in a type (`["integer", "null"]`) is how a definition says the
//! parameter is optional; the grammar says that by leaving it out.
//!
//! A parameter is carried under its own name, except one the act's line
//! already spends — `tool`, `name` and `arguments` name the act itself
//! (`engine::act`), so an argument called that would be read as the act's
//! name. That one is carried under the first word of its tool's name:
//! `guide_read`'s `name` is its `guide`.

use serde_json::{Map, Value};

use crate::engine::tools::LIST;

/// What an act's line spends on the act itself.
const SPENT: [&str; 3] = ["tool", "name", "arguments"];

/// The name tool `tool`'s parameter `param` is carried under.
pub fn said_as(tool: &'static str, param: &'static str) -> &'static str {
    match SPENT.contains(&param) {
        true => tool.split('_').next().unwrap_or(tool),
        false => param,
    }
}

/// How one parameter is carried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Carry {
    /// As the grammar's own type: a string, an integer, a boolean or a list.
    Itself,
    /// A number, spelled as a string.
    Number,
    /// An object of strings, one `name: value` to a line.
    Lines,
    /// Any other value, as JSON text.
    Json,
}

impl Carry {
    /// What the parameter's description says of how it is written, after the
    /// definition's own words.
    pub fn note(self) -> &'static str {
        match self {
            Carry::Itself => "",
            Carry::Number => " Written as a number in a string: \"72.5\".",
            Carry::Lines => " Written one `name: value` to a line.",
            Carry::Json => " Written as JSON.",
        }
    }
}

/// The schema's type names, `null` aside.
fn types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(t)) => vec![t.as_str()],
        Some(Value::Array(ts)) => ts
            .iter()
            .filter_map(Value::as_str)
            .filter(|t| *t != "null")
            .collect(),
        _ => Vec::new(),
    }
}

/// The grammar's type for a parameter of `schema`, and how it is carried.
pub fn carried(schema: &Value) -> (&'static str, Carry) {
    match types(schema).as_slice() {
        ["string"] => ("string", Carry::Itself),
        ["integer"] => ("integer", Carry::Itself),
        ["boolean"] => ("boolean", Carry::Itself),
        ["number"] => ("string", Carry::Number),
        ["array"] if types(&schema["items"]) == ["string"] => (LIST, Carry::Itself),
        ["object"] if types(&schema["additionalProperties"]) == ["string"] => {
            ("string", Carry::Lines)
        }
        _ => ("string", Carry::Json),
    }
}

/// `value`, as carried, turned back into what the tool takes.
pub fn back(carry: Carry, name: &str, value: &Value) -> Result<Value, String> {
    let text = || {
        value
            .as_str()
            .map(str::trim)
            .ok_or_else(|| format!("`{name}` is written as text."))
    };
    match carry {
        Carry::Itself => Ok(value.clone()),
        Carry::Number => {
            let said = text()?;
            said.replace(',', "")
                .parse::<f64>()
                .map(Value::from)
                .map_err(|_| format!("`{name}` is a number, not \"{said}\"."))
        }
        Carry::Lines => {
            let mut out = Map::new();
            for line in text()?.lines().map(str::trim).filter(|l| !l.is_empty()) {
                let (k, v) = line
                    .split_once(':')
                    .filter(|(k, _)| !k.trim().is_empty())
                    .ok_or_else(|| format!("`{line}` in `{name}` is no `name: value` line."))?;
                out.insert(k.trim().to_string(), Value::from(v.trim()));
            }
            Ok(Value::Object(out))
        }
        Carry::Json => {
            let said = text()?;
            serde_json::from_str(said).map_err(|e| format!("`{name}` is not JSON: {e}."))
        }
    }
}

/// `value`, as the tool takes it, written as `carry` carries it: what a
/// definition's example call becomes in the act's own example.
pub fn there(carry: Carry, value: &Value) -> Value {
    match carry {
        Carry::Itself => value.clone(),
        Carry::Number => Value::from(match value {
            Value::Number(n) => n.to_string(),
            other => other.to_string(),
        }),
        Carry::Lines => Value::from(
            value
                .as_object()
                .into_iter()
                .flatten()
                .map(|(k, v)| format!("{k}: {}", v.as_str().unwrap_or_default()))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        Carry::Json => Value::from(value.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **Each schema is carried as the nearest type the grammar bounds.**
    #[test]
    fn each_schema_is_carried_as_the_nearest_bounded_type() {
        assert_eq!(
            carried(&json!({"type": "string"})),
            ("string", Carry::Itself)
        );
        assert_eq!(
            carried(&json!({"type": ["integer", "null"], "format": "uint8"})),
            ("integer", Carry::Itself)
        );
        assert_eq!(
            carried(&json!({"type": "boolean"})),
            ("boolean", Carry::Itself)
        );
        assert_eq!(
            carried(&json!({"type": "number"})),
            ("string", Carry::Number)
        );
        assert_eq!(
            carried(&json!({"type": ["array", "null"], "items": {"type": "string"}})),
            (LIST, Carry::Itself)
        );
        assert_eq!(
            carried(
                &json!({"type": ["object", "null"], "additionalProperties": {"type": "string"}})
            ),
            ("string", Carry::Lines)
        );
        assert_eq!(carried(&json!({})), ("string", Carry::Json));
        assert_eq!(
            carried(&json!({"type": "array", "items": {"type": "integer"}})),
            ("string", Carry::Json)
        );
    }

    /// **What was carried comes back as the tool takes it**, and what does not
    /// spell its kind is said.
    #[test]
    fn what_was_carried_comes_back() {
        assert_eq!(back(Carry::Itself, "n", &json!(3)), Ok(json!(3)));
        assert_eq!(
            back(Carry::Number, "value", &json!(" 1,250.5 ")),
            Ok(json!(1250.5))
        );
        assert_eq!(
            back(Carry::Number, "value", &json!("ten")),
            Err("`value` is a number, not \"ten\".".into())
        );
        assert_eq!(
            back(
                Carry::Lines,
                "headers",
                &json!("Accept: text/html\nX-Key:  k1 ")
            ),
            Ok(json!({"Accept": "text/html", "X-Key": "k1"}))
        );
        assert_eq!(
            back(Carry::Lines, "headers", &json!("no colon")),
            Err("`no colon` in `headers` is no `name: value` line.".into())
        );
        assert_eq!(
            back(Carry::Json, "body_json", &json!(r#"{"a":1}"#)),
            Ok(json!({"a": 1}))
        );
        assert!(back(Carry::Json, "body_json", &json!("{")).is_err());
    }

    /// **A parameter is carried under its own name, unless the act's line
    /// spends it.**
    #[test]
    fn a_spent_name_is_carried_under_its_tools_first_word() {
        assert_eq!(said_as("guide_read", "name"), "guide");
        assert_eq!(said_as("random", "tool"), "random");
        assert_eq!(said_as("weather", "location"), "location");
    }

    /// **An example's value is written as it is carried**, and comes back to
    /// what it was.
    #[test]
    fn an_example_value_is_written_as_it_is_carried_and_back() {
        for (carry, v) in [
            (Carry::Number, json!(72.5)),
            (Carry::Lines, json!({"Accept": "text/html"})),
            (Carry::Json, json!({"a": [1, 2]})),
            (Carry::Itself, json!("x")),
        ] {
            assert_eq!(
                back(carry, "p", &there(carry, &v)),
                Ok(v.clone()),
                "{carry:?}"
            );
        }
        assert_eq!(there(Carry::Number, &json!(72)), json!("72"));
        assert_eq!(
            there(Carry::Lines, &json!({"A": "1", "B": "2"})),
            json!("A: 1\nB: 2")
        );
    }
}
