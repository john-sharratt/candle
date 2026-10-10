//! Filling a step's prompt template.
//!
//! A placeholder is `{name}`, where `name` is a letter or `_` followed by
//! letters, digits or `_`. Any other brace — `{ }`, `{"a": 1}`, `{3}` — is
//! text and is left alone, so a prompt may quote JSON or code.
//!
//! **A placeholder with no value is an error, never a silent blank.** The
//! caller passes every value the step may use, an empty string included; a
//! name it did not pass is a typo in the template or a value the engine does
//! not provide, and either way the step's taker would be handed a prompt with
//! a hole in it. [`fill`] refuses, naming every such placeholder.
//!
//! Values are inserted verbatim and are not scanned again, so a value that
//! itself contains `{objective}` arrives as that literal text.

use std::collections::BTreeMap;

/// One placeholder's place in a template: its byte range and its name.
struct Slot<'a> {
    start: usize,
    end: usize,
    name: &'a str,
}

fn slots(template: &str) -> Vec<Slot<'_>> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(offset) = template[from..].find('{') {
        let open = from + offset;
        let rest = &template[open + 1..];
        let len = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let name = &rest[..len];
        let starts_well = name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_');
        if starts_well && rest[len..].starts_with('}') {
            found.push(Slot {
                start: open,
                end: open + 1 + len + 1,
                name,
            });
            from = open + 1 + len + 1;
        } else {
            from = open + 1;
        }
    }
    found
}

/// The placeholders `template` names, each once, in order of first use.
pub fn placeholders(template: &str) -> Vec<&str> {
    let mut names: Vec<&str> = Vec::new();
    for slot in slots(template) {
        if !names.contains(&slot.name) {
            names.push(slot.name);
        }
    }
    names
}

/// `template` with every placeholder replaced by its value from `values`.
pub fn fill(template: &str, values: &BTreeMap<&str, &str>) -> Result<String, String> {
    let found = slots(template);
    let missing: Vec<String> = placeholders(template)
        .into_iter()
        .filter(|name| !values.contains_key(name))
        .map(|name| format!("{{{name}}}"))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "the prompt names {} with no value given",
            missing.join(", ")
        ));
    }
    let mut out = String::with_capacity(template.len());
    let mut from = 0;
    for slot in found {
        out.push_str(&template[from..slot.start]);
        out.push_str(values[slot.name]);
        from = slot.end;
    }
    out.push_str(&template[from..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map<'a>(pairs: &[(&'a str, &'a str)]) -> BTreeMap<&'a str, &'a str> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn placeholders_are_filled() {
        assert_eq!(
            fill(
                "Write {objective}.\nFix: {findings}\n{objective}!",
                &map(&[("objective", "the flood"), ("findings", "dates")])
            ),
            Ok("Write the flood.\nFix: dates\nthe flood!".to_string())
        );
    }

    #[test]
    fn an_empty_value_is_a_value() {
        assert_eq!(
            fill("[{findings}]", &map(&[("findings", "")])),
            Ok("[]".to_string())
        );
    }

    #[test]
    fn a_placeholder_without_a_value_is_refused_naming_each_once() {
        assert_eq!(
            fill("{a} {objective} {b} {a}", &map(&[("objective", "x")])),
            Err("the prompt names {a}, {b} with no value given".to_string())
        );
    }

    #[test]
    fn braces_that_are_not_placeholders_are_text() {
        let t = r#"Answer as {"verdict": "pass"} or { } or {3} or {a-b} or {x"#;
        assert_eq!(placeholders(t), Vec::<&str>::new());
        assert_eq!(fill(t, &map(&[])), Ok(t.to_string()));
    }

    #[test]
    fn a_brace_just_before_a_placeholder_is_text() {
        assert_eq!(
            fill("{{name}}", &map(&[("name", "Ann")])),
            Ok("{Ann}".to_string())
        );
    }

    #[test]
    fn values_are_not_scanned_again() {
        assert_eq!(
            fill("{a}", &map(&[("a", "{b}"), ("b", "no")])),
            Ok("{b}".to_string())
        );
    }

    #[test]
    fn placeholders_are_listed_once_in_order_of_first_use() {
        assert_eq!(placeholders("{b} {a} {b} {_c1}"), ["b", "a", "_c1"]);
    }

    #[test]
    fn multibyte_text_around_placeholders_survives() {
        assert_eq!(
            fill("— {who} → ≤", &map(&[("who", "Wren")])),
            Ok("— Wren → ≤".to_string())
        );
    }
}
