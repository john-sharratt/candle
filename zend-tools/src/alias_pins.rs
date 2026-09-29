//! Arguments an alias fixes by its name.
//!
//! An alias answers for its canonical tool ([`crate::registry::find`]) with the
//! tool's whole request, so a name that means one mode could decode another:
//! measured live, a model asked to cherry-pick called `git_cherry_pick` with
//! `from: revert`, undid a commit its branch never had, and settled the
//! conflict that made by deleting a line. A name that says what it does fixes
//! that argument. The grammar writes the value — zend compiles each alias with
//! the field narrowed to it — and dispatch holds to it for a call that did not
//! come through the grammar: a missing value is filled in, a different one is
//! refused, naming the tool that does what was asked.

use std::borrow::Cow;

use serde_json::{json, Map, Value};

/// `(alias, [(field, value)])`: what each alias fixes.
const PINS: &[(&str, &[(&str, &str)])] = &[
    ("git_pick", &[("from", "cherry_pick")]),
    ("cherry_pick", &[("from", "cherry_pick")]),
    ("git_cherry_pick", &[("from", "cherry_pick")]),
    ("backport", &[("from", "cherry_pick")]),
    ("git_revert", &[("from", "revert")]),
    ("revert_commit", &[("from", "revert")]),
    ("git_apply", &[("from", "patch")]),
    ("apply_patch", &[("from", "patch")]),
    ("land_patch", &[("from", "patch")]),
    ("apply_diff", &[("from", "patch")]),
    ("git_tag", &[("kind", "tag")]),
    ("create_tag", &[("kind", "tag"), ("action", "create")]),
    ("make_tag", &[("kind", "tag"), ("action", "create")]),
    ("delete_tag", &[("kind", "tag"), ("action", "delete")]),
    ("git_branch", &[("kind", "branch")]),
    ("create_branch", &[("kind", "branch"), ("action", "create")]),
    ("new_branch", &[("kind", "branch"), ("action", "create")]),
    ("delete_branch", &[("kind", "branch"), ("action", "delete")]),
    ("move_branch", &[("kind", "branch"), ("action", "move")]),
];

/// The arguments `name` fixes — empty for a canonical name or an alias that
/// fixes none.
pub fn pins(name: &str) -> &'static [(&'static str, &'static str)] {
    PINS.iter()
        .find(|(alias, _)| *alias == name)
        .map_or(&[], |(_, pins)| *pins)
}

/// `args` as a call under `name` means them: each argument the name fixes
/// filled in where it is missing. A different value is refused with the error
/// the model reads — the name and the argument disagree, and which one was
/// meant cannot be told.
pub fn apply<'a>(name: &str, args: &'a Value) -> Result<Cow<'a, Value>, Value> {
    let pins = pins(name);
    if pins.is_empty() {
        return Ok(Cow::Borrowed(args));
    }
    let mut fixed: Map<String, Value> = match args {
        Value::Object(map) => map.clone(),
        Value::Null => Map::new(),
        _ => return Ok(Cow::Borrowed(args)),
    };
    for (field, value) in pins {
        match fixed.get(*field) {
            None | Some(Value::Null) => {
                fixed.insert((*field).to_string(), Value::String((*value).to_string()));
            }
            Some(Value::String(given)) if given == value => {}
            Some(given) => {
                return Err(json!({
                    "error": "invalid_arguments",
                    "detail": format!(
                        "`{name}` means `{field}: {value}`, but the call gives `{field}: \
                         {given}`. Call it again with `{field}` left out or set to `{value}` — \
                         or, if `{given}` is what you mean, call the tool whose name says so"
                    ),
                }));
            }
        }
    }
    Ok(Cow::Owned(Value::Object(fixed)))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::{aliases, find};

    /// **A pinned argument is filled in, kept when it agrees, and refused
    /// when it does not**; a name with no pins passes its arguments through.
    #[test]
    fn a_pin_fills_in_agrees_or_refuses() {
        assert_eq!(
            apply("git_cherry_pick", &json!({"repo": "app", "commit": "c"})).unwrap(),
            Cow::<Value>::Owned(json!({"repo": "app", "commit": "c", "from": "cherry_pick"}))
        );
        let agreeing = json!({"repo": "app", "from": "revert", "commit": "c"});
        assert_eq!(apply("git_revert", &agreeing).unwrap().as_ref(), &agreeing);
        let refused = apply("git_cherry_pick", &json!({"from": "revert"})).unwrap_err();
        assert_eq!(refused["error"], "invalid_arguments");
        assert_eq!(
            refused["detail"],
            "`git_cherry_pick` means `from: cherry_pick`, but the call gives `from: \"revert\"`. \
             Call it again with `from` left out or set to `cherry_pick` — or, if `\"revert\"` is \
             what you mean, call the tool whose name says so"
        );
        let plain = json!({"repo": "app", "from": "revert"});
        assert!(matches!(apply("git_commit", &plain), Ok(Cow::Borrowed(_))));
    }

    /// **Every pin belongs to an alias of the tool it runs, on a field that
    /// tool's request has, with a value that field allows** — a pin that
    /// drifted from the schema would fix an argument to something refused.
    #[test]
    fn every_pin_is_a_legal_argument_of_its_tool() {
        for (alias, pins) in PINS {
            let tool = find(alias).unwrap_or_else(|| panic!("{alias} runs no tool"));
            assert!(
                aliases(tool.name).contains(alias),
                "{alias} is not an alias of {}",
                tool.name
            );
            let schema = (tool.schema)();
            for (field, value) in *pins {
                let property = &schema["properties"][field];
                let named = property["allOf"][0]["$ref"]
                    .as_str()
                    .or_else(|| property["$ref"].as_str())
                    .and_then(|r| r.strip_prefix("#/definitions/"));
                let enum_schema = match named {
                    Some(name) => &schema["definitions"][name],
                    None => property,
                };
                // An enum whose variants carry documentation is written as a
                // `oneOf` of one-value schemas; a plain one as `enum`.
                let values: Vec<&Value> = match enum_schema["oneOf"].as_array() {
                    Some(arms) => arms
                        .iter()
                        .flat_map(|arm| arm["enum"].as_array().into_iter().flatten())
                        .collect(),
                    None => enum_schema["enum"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .collect(),
                };
                assert!(
                    values.iter().any(|x| *x == value),
                    "{alias}: {field} does not allow {value} ({values:?})"
                );
            }
        }
    }
}
