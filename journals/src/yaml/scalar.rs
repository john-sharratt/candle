//! Writing scalars. Free text is always double-quoted, as JSON writes a
//! string — a JSON string is a YAML double-quoted scalar — so no text can end
//! a flow mapping early, start a comment, or be read back as a number, a
//! boolean or null.
//!
//! Names, times, ids, keys and fingerprints are written plain where YAML
//! reads the plain form back as exactly that text, and quoted where it would
//! not: a journal named `true` or `null`, a name `1e3` or `0x1f`, a
//! fingerprint `1234567890123e45` — each a valid value of its type that YAML
//! would otherwise read as a boolean, nothing, or a number.

use serde_yaml::Value;

/// `s` double-quoted.
///
/// JSON leaves a few characters raw that YAML reads otherwise: NEL (U+0085)
/// and the line and paragraph separators are line breaks to YAML, folded to a
/// space inside a quoted scalar, and a byte-order mark is stripped. Those are
/// written as `\u` escapes, which both read back as the character itself.
pub(crate) fn quote(s: &str) -> String {
    let json = serde_json::to_string(s).expect("a string always serialises");
    if !json.contains(['\u{85}', '\u{2028}', '\u{2029}', '\u{feff}']) {
        return json;
    }
    let mut out = String::with_capacity(json.len() + 8);
    for c in json.chars() {
        match c {
            '\u{85}' | '\u{2028}' | '\u{2029}' | '\u{feff}' => {
                out.push_str(&format!("\\u{:04x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out
}

/// `s` plain when YAML reads it back as `s`, else double-quoted. For values
/// whose types allow no character that ends a flow mapping or starts a
/// comment: names, times, ids, keys, fingerprints.
///
/// The reader that will read the file decides, rather than a copy of its
/// resolution rules: a short scalar's parse is a few microseconds, and a
/// hand-kept list of what YAML takes for a number is how `0x1f` slips by.
pub(crate) fn scalar(s: &str) -> String {
    match serde_yaml::from_str::<Value>(s) {
        Ok(Value::String(back)) if back == s => s.to_string(),
        _ => quote(s),
    }
}

/// `[a, "true"]`: each item as [`scalar`] writes it.
pub(crate) fn scalar_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| scalar(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

/// `["a", "b"]`.
pub(crate) fn quote_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| quote(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever the text, it reads back from YAML as exactly itself.
    #[test]
    fn quoted_text_reads_back_as_itself() {
        for text in [
            "",
            "plain",
            "a: b",
            "a, b}",
            "#not a comment",
            "- not an item",
            "null",
            "true",
            "1e3",
            "0x1F",
            "'single'",
            "\"double\"",
            "back\\slash",
            "tab\there",
            "line\nbreak\r\n",
            "nul\u{0}byte",
            "bell\u{7}",
            "emoji 🚗 and đà nẵng",
            "  spaced  ",
            "{a: 1}",
            "&anchor *alias !tag",
            "%directive",
            "@at `tick` | pipe > gt",
            "\u{2028}line separator\u{85}nel\u{feff}bom\u{2029}",
            "\u{feff}leading bom",
            "\u{85}",
        ] {
            let yaml = format!("{{text: {}}}", quote(text));
            let v: Value = serde_yaml::from_str(&yaml).unwrap();
            assert_eq!(v["text"].as_str(), Some(text), "{yaml}");
        }
        assert_eq!(quote("a\"b"), r#""a\"b""#);
        assert_eq!(quote("a\u{85}b"), r#""a\u0085b""#);
    }

    #[test]
    fn names_yaml_would_misread_are_quoted_and_the_rest_are_plain() {
        for plain in [
            "candle",
            "2026-10-30-2105",
            "2026-10-30T09:10:02Z",
            "20261030-0912-7e3a",
            "verify.2",
            "e3b0c44298fc1c14",
            "yes",
            "nan",
            "0123",
        ] {
            assert_eq!(scalar(plain), plain);
        }
        for misread in [
            "true",
            "false",
            "null",
            "1e3",
            "0x1f",
            "0o17",
            "123",
            "1234567890123e45",
            "1.5",
            "~",
            "",
        ] {
            let written = scalar(misread);
            assert_eq!(written, quote(misread), "{misread}");
            let v: Value = serde_yaml::from_str(&format!("{{v: {written}}}")).unwrap();
            assert_eq!(v["v"].as_str(), Some(misread));
        }
        assert_eq!(scalar_list(&["sharratt", "true"]), r#"[sharratt, "true"]"#);
    }

    #[test]
    fn lists_of_text_are_flow_sequences() {
        assert_eq!(quote_list(&["#a", "b c"]), r##"["#a", "b c"]"##);
        assert_eq!(quote_list::<&str>(&[]), "[]");
    }
}
