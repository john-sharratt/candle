//! Raw control characters inside a decoded JSON string, escaped.
//!
//! # The failure this exists for
//!
//! A grammar's free-text span ends at the first unescaped `"` and excludes
//! nothing else, so a model filling a string argument may emit a literal
//! newline. JSON forbids that — a control character in a string must be escaped
//! — so `serde_json` refuses the whole object. A character in npcd did this every
//! turn for hours: the grammar had steered it correctly, the first byte of its
//! argument was a newline, and every turn came back refused for a call it had
//! made exactly right.
//!
//! # Why repairing is not papering over it
//!
//! The model's content is right and only its encoding is wrong, and the repair
//! is exact rather than a guess: JSON defines the escape for every control
//! character, so a raw one has one correct reading and no other. Outside a
//! string a newline is structural, so only the inside is touched. Borrowed and
//! untouched when there is nothing to escape, which is every ordinary decode.

use std::borrow::Cow;

/// `s` with every control character inside a JSON string written as the escape
/// JSON requires.
pub fn escape_control_in_strings(s: &str) -> Cow<'_, str> {
    if !needs_repair(s) {
        return Cow::Borrowed(s);
    }
    let mut in_string = false;
    let mut escaped = false;
    let mut out = String::with_capacity(s.len() + 16);
    for c in s.chars() {
        if escaped {
            escaped = false;
            out.push(c);
            continue;
        }
        match c {
            '\\' if in_string => {
                escaped = true;
                out.push(c);
            }
            '"' => {
                in_string = !in_string;
                out.push(c);
            }
            c if in_string && (c as u32) < 0x20 => match c {
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                other => out.push_str(&format!("\\u{:04x}", other as u32)),
            },
            c => out.push(c),
        }
    }
    Cow::Owned(out)
}

/// Whether any control character actually sits inside a string. The rewrite
/// does nothing when this is false, so an ordinary decode pays one pass and no
/// allocation.
fn needs_repair(s: &str) -> bool {
    let (mut in_string, mut escaped) = (false, false);
    for c in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            c if in_string && (c as u32) < 0x20 => return true,
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_raw_newline_inside_a_string_is_escaped_and_reads_back_as_content() {
        let fixed = escape_control_in_strings("{\"a\": \"one\ntwo\"}");
        assert_eq!(fixed, "{\"a\": \"one\\ntwo\"}");
        let v: serde_json::Value = serde_json::from_str(&fixed).unwrap();
        assert_eq!(v["a"], "one\ntwo");
    }

    #[test]
    fn every_control_character_json_forbids_raw_is_escaped() {
        assert_eq!(
            escape_control_in_strings("\"\t\r\u{1}\""),
            "\"\\t\\r\\u0001\""
        );
    }

    #[test]
    fn a_newline_outside_a_string_is_structure_and_left_alone() {
        let two = "{\"a\": \"one\"}\n{\"a\": \"two\"}";
        assert!(matches!(escape_control_in_strings(two), Cow::Borrowed(_)));
    }

    #[test]
    fn an_escaped_quote_does_not_end_the_string() {
        assert_eq!(
            escape_control_in_strings("\"he said \\\"go\\\"\nand went\""),
            "\"he said \\\"go\\\"\\nand went\""
        );
    }

    #[test]
    fn a_clean_decode_is_not_rewritten() {
        assert!(matches!(
            escape_control_in_strings("{\"a\": \"nothing to repair\"}"),
            Cow::Borrowed(_)
        ));
    }
}
