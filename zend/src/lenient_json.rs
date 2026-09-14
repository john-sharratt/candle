//! Reading JSON a model wrote with Windows paths in it.
//!
//! A model writing a call's arguments spells a path the way the shell does —
//! `"dir C:\Users\johna\prog\*.md"` — and `\U`, `\j`, `\p` and `\*` are not JSON
//! escapes, so the call does not parse. Measured on a Cline turn: the call went
//! back to the client as prose, with no `tool_calls` and `finish_reason: stop`,
//! and the task ended there.
//!
//! [`repair_escapes`] rewrites only a string that holds an escape JSON does not
//! have. In such a string every backslash is taken literally except `\"` (the
//! string's own quote, which the model does escape) and `\\`; a string whose
//! escapes are all valid is left exactly as written. The rule is per string
//! because a path's `\t` or `\n` (`C:\tools`, `C:\new`) is valid JSON on its
//! own — the invalid escape beside it is what says the string is a raw path.

/// `text` with each string that holds an invalid escape rewritten to take its
/// backslashes literally, or `None` when there is no such string.
pub fn repair_escapes(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len() + 8);
    let mut changed = false;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'"' {
            out.push(bytes[i]);
            i += 1;
            continue;
        }
        let body_start = i + 1;
        let body_end = string_end(bytes, body_start);
        let body = &bytes[body_start..body_end];
        out.push(b'"');
        if has_invalid_escape(body) {
            changed = true;
            literal_backslashes(body, &mut out);
        } else {
            out.extend_from_slice(body);
        }
        if body_end < bytes.len() {
            out.push(b'"');
        }
        i = body_end + 1;
    }
    // Only ASCII backslashes were inserted, and only between whole characters,
    // so the bytes are still UTF-8.
    changed.then(|| String::from_utf8(out).expect("inserting ASCII keeps UTF-8 valid"))
}

/// The index of the quote closing the string whose body starts at `start`, or
/// `bytes.len()` when the string runs to the end of the text.
fn string_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return i,
            _ => i += 1,
        }
    }
    bytes.len()
}

/// Whether a string body holds an escape JSON does not define.
fn has_invalid_escape(body: &[u8]) -> bool {
    let mut i = 0;
    while i < body.len() {
        if body[i] != b'\\' {
            i += 1;
            continue;
        }
        match body.get(i + 1) {
            Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => i += 2,
            Some(b'u')
                if body.len() >= i + 6 && body[i + 2..i + 6].iter().all(u8::is_ascii_hexdigit) =>
            {
                i += 6
            }
            _ => return true,
        }
    }
    false
}

/// Copy a raw-path string body, doubling every backslash but those of `\"`
/// and `\\`.
fn literal_backslashes(body: &[u8], out: &mut Vec<u8>) {
    let mut i = 0;
    while i < body.len() {
        match (body[i], body.get(i + 1)) {
            (b'\\', Some(&next @ (b'"' | b'\\'))) => {
                out.extend_from_slice(&[b'\\', next]);
                i += 2;
            }
            (b'\\', _) => {
                out.extend_from_slice(b"\\\\");
                i += 1;
            }
            (b, _) => {
                out.push(b);
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The call Cline's turn ended on, byte for byte.
    #[test]
    fn a_raw_windows_path_reads_as_the_path() {
        let text = r#"{"commands": ["dir C:\Users\johna\prog\candle\*.md /B"]}"#;
        let repaired = repair_escapes(text).expect("the path's escapes are invalid");
        assert_eq!(
            repaired,
            r#"{"commands": ["dir C:\\Users\\johna\\prog\\candle\\*.md /B"]}"#
        );
        let value: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(
            value["commands"][0],
            r"dir C:\Users\johna\prog\candle\*.md /B"
        );
    }

    /// A path's `\t` is a valid escape on its own; the invalid one beside it is
    /// what marks the string as a raw path, so both are taken literally.
    #[test]
    fn a_valid_escape_in_a_raw_path_is_taken_literally_too() {
        assert_eq!(
            repair_escapes(r#"["C:\tools\x"]"#).as_deref(),
            Some(r#"["C:\\tools\\x"]"#)
        );
    }

    #[test]
    fn valid_json_is_left_alone() {
        assert_eq!(repair_escapes(r#"{"a": "x\ny \"q\" \\ \u00e9"}"#), None);
        assert_eq!(repair_escapes(r#"{"a": 1, "b": [true, null]}"#), None);
    }

    /// Only the string that needs it is rewritten.
    #[test]
    fn a_string_with_valid_escapes_beside_a_raw_path_is_untouched() {
        assert_eq!(
            repair_escapes(r#"{"a": "x\ny", "b": "C:\q"}"#).as_deref(),
            Some(r#"{"a": "x\ny", "b": "C:\\q"}"#)
        );
    }

    /// The string's own escaped quotes stay escapes, so it still ends where the
    /// model ended it.
    #[test]
    fn escaped_quotes_in_a_raw_path_stay_quotes() {
        let repaired = repair_escapes(r#"["type \"C:\a b\""]"#).unwrap();
        assert_eq!(repaired, r#"["type \"C:\\a b\""]"#);
        let value: serde_json::Value = serde_json::from_str(&repaired).unwrap();
        assert_eq!(value[0], r#"type "C:\a b""#);
    }

    /// `\u` without four hex digits is a path segment (`C:\users`), not an
    /// escape.
    #[test]
    fn a_short_unicode_escape_is_a_path_segment() {
        assert_eq!(
            repair_escapes(r#"["C:\users"]"#).as_deref(),
            Some(r#"["C:\\users"]"#)
        );
    }

    #[test]
    fn non_ascii_after_a_backslash_survives() {
        assert_eq!(
            repair_escapes(r#"["C:\é"]"#).as_deref(),
            Some(r#"["C:\\é"]"#)
        );
    }

    #[test]
    fn an_unterminated_string_is_copied_to_the_end() {
        assert_eq!(repair_escapes(r#"["C:\q"#).as_deref(), Some(r#"["C:\\q"#));
    }
}
