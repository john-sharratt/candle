//! A call cut off by its token budget, closed so what it did say is read.
//!
//! # The failure this exists for
//!
//! An answer that runs out of tokens stops wherever it was: inside a claim's
//! text, between two claims, after a key. `serde_json` refuses the whole
//! object, and every claim the writer got right is lost with the one it was
//! still writing. Asking again costs another answer at least as long, which
//! runs into the same budget.
//!
//! # What closing does
//!
//! A string the answer was still writing is ended with `...`, so the reader
//! sees it was cut; a key left without its value gets `null`; a dangling comma
//! is dropped; every array and object still open is closed. The text up to the
//! cut is kept exactly as written. Whether the last element is complete enough
//! to keep is the reader's to judge ([`crate::tools::parse`] drops a claim the
//! cut left without what a claim needs, in the field the cut was in).

/// What closes an open container.
enum Frame {
    /// An object, whether the next string in it is a key, and where its
    /// latest key lies in the text.
    Object {
        want_key: bool,
        key: Option<(usize, usize)>,
    },
    Array,
}

/// A cut call, closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Closed {
    /// The call as JSON that reads.
    pub text: String,
    /// The key each object still open at the cut was writing, outermost
    /// first: `["arguments", "claims"]` for a cut inside a claim.
    pub open: Vec<String>,
}

/// `s` — JSON from its opening `{` — closed where it was cut. `None` when
/// nothing was cut: `s` does not open a container, or closes every one it
/// opens.
pub fn close_cut(s: &str) -> Option<Closed> {
    let mut stack: Vec<Frame> = Vec::new();
    let (mut in_string, mut escaped) = (false, false);
    // Whether the string being read, or the last one read, is an object's key
    // still waiting for its `:`, and where it began.
    let (mut string_is_key, mut key_waiting, mut string_at) = (false, false, 0);
    let mut opened = false;
    // Where the escape being read began, and the hex digits a `\u` escape
    // still wants.
    let (mut escape_at, mut hex_left) = (0, 0u8);
    // Where the bare value being read — a number, `true`, `false`, `null` —
    // began.
    let mut bare_at: Option<usize> = None;
    for (at, c) in s.char_indices() {
        if in_string {
            match c {
                _ if hex_left > 0 => hex_left -= 1,
                'u' if escaped => {
                    escaped = false;
                    hex_left = 4;
                }
                _ if escaped => escaped = false,
                '\\' => {
                    escaped = true;
                    escape_at = at;
                }
                '"' => {
                    in_string = false;
                    key_waiting = string_is_key;
                    if string_is_key {
                        if let Some(Frame::Object { key, .. }) = stack.last_mut() {
                            *key = Some((string_at, at));
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '+' | '.') {
            bare_at.get_or_insert(at);
            continue;
        }
        bare_at = None;
        match c {
            '"' => {
                in_string = true;
                string_at = at + 1;
                string_is_key = matches!(stack.last(), Some(Frame::Object { want_key: true, .. }));
            }
            '{' => {
                stack.push(Frame::Object {
                    want_key: true,
                    key: None,
                });
                opened = true;
            }
            '[' => {
                stack.push(Frame::Array);
                opened = true;
            }
            '}' | ']' => {
                stack.pop();
                if stack.is_empty() {
                    return None;
                }
            }
            ':' => {
                key_waiting = false;
                if let Some(Frame::Object { want_key, .. }) = stack.last_mut() {
                    *want_key = false;
                }
            }
            ',' => {
                if let Some(Frame::Object { want_key, .. }) = stack.last_mut() {
                    *want_key = true;
                }
            }
            _ => {}
        }
    }
    if !opened || stack.is_empty() {
        return None;
    }
    // An object between two members is writing no value: its last key is
    // done with.
    let open = stack
        .iter()
        .filter_map(|f| match f {
            Frame::Object {
                want_key: false,
                key: Some((from, to)),
            } => Some(s[*from..*to].to_string()),
            _ => None,
        })
        .collect();
    let mut out = s.to_string();
    if in_string {
        // An escape cut part-way is dropped rather than read as something
        // else.
        if escaped || hex_left > 0 {
            out.truncate(escape_at);
        }
        out.push_str("...\"");
        key_waiting = string_is_key;
    } else {
        // A bare value cut part-way may read as another value, or as none:
        // it is dropped, and its key given `null` below.
        if let Some(at) = bare_at {
            out.truncate(at);
        }
        out.truncate(out.trim_end().len());
        if out.ends_with(',') {
            out.pop();
        }
        if out.ends_with(':') {
            out.push_str("null");
        }
    }
    if key_waiting {
        out.push_str(": null");
    }
    for frame in stack.iter().rev() {
        out.push(match frame {
            Frame::Object { .. } => '}',
            Frame::Array => ']',
        });
    }
    Some(Closed { text: out, open })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed(s: &str) -> Option<String> {
        close_cut(s).map(|c| c.text)
    }

    #[test]
    fn a_string_cut_mid_text_is_ended_with_an_ellipsis_and_closed() {
        assert_eq!(
            closed(r#"{"name": "journal_write", "arguments": {"claims": [{"text": "The scan fa"#)
                .as_deref(),
            Some(
                r#"{"name": "journal_write", "arguments": {"claims": [{"text": "The scan fa..."}]}}"#
            )
        );
    }

    #[test]
    fn a_cut_after_a_comma_drops_the_comma() {
        assert_eq!(
            closed("{\"claims\": [{\"text\": \"a\", \"cite\": \"1\"},\n  ").as_deref(),
            Some(r#"{"claims": [{"text": "a", "cite": "1"}]}"#)
        );
    }

    #[test]
    fn a_key_left_without_its_value_is_given_null() {
        assert_eq!(
            closed(r#"{"claims": [{"text": "a", "cite""#).as_deref(),
            Some(r#"{"claims": [{"text": "a", "cite": null}]}"#)
        );
        assert_eq!(
            closed(r#"{"claims": [{"text": "a", "cite": "#).as_deref(),
            Some(r#"{"claims": [{"text": "a", "cite":null}]}"#)
        );
        assert_eq!(
            closed(r#"{"claims": [{"text": "a", "ci"#).as_deref(),
            Some(r#"{"claims": [{"text": "a", "ci...": null}]}"#)
        );
    }

    #[test]
    fn a_cut_escape_is_dropped_rather_than_escaping_the_closing_quote() {
        assert_eq!(
            closed(r#"{"text": "say \"#).as_deref(),
            Some(r#"{"text": "say ..."}"#)
        );
    }

    #[test]
    fn a_unicode_escape_cut_part_way_is_dropped() {
        assert_eq!(
            closed(r#"{"text": "caf\u00"#).as_deref(),
            Some(r#"{"text": "caf..."}"#)
        );
        let whole = format!(r#"{{"text": "caf{}u00e9 n"#, '\\');
        assert_eq!(
            close_cut(&whole).map(|c| c.text),
            Some(format!("{whole}...\"}}"))
        );
    }

    #[test]
    fn a_bare_value_cut_part_way_is_dropped_and_its_key_given_null() {
        assert_eq!(
            closed(r#"{"a": [1, 2], "ok": tr"#).as_deref(),
            Some(r#"{"a": [1, 2], "ok":null}"#)
        );
        assert_eq!(closed(r#"{"a": [1, -"#).as_deref(), Some(r#"{"a": [1]}"#));
    }

    #[test]
    fn a_whole_call_or_no_call_is_not_cut() {
        assert_eq!(closed(r#"{"text": "a"}"#), None);
        assert_eq!(closed(r#"{"text": "a"} and more"#), None);
        assert_eq!(closed("no call at all"), None);
    }

    #[test]
    fn every_closed_cut_reads_as_json() {
        let call = format!(
            r#"{{"name": "journal_write", "arguments": {{"claims": [{{"text": "a \"b\" caf{}u00e9 — ok", "cite": 12}}, {{"text": "d", "cite": "2", "typed": null}}], "ok": true, "n": -1.5e3, "summary": "s"}}}}"#,
            '\\'
        );
        for end in (1..call.len()).filter(|&e| call.is_char_boundary(e)) {
            if let Some(closed) = close_cut(&call[..end]) {
                assert!(
                    serde_json::from_str::<serde_json::Value>(&closed.text).is_ok(),
                    "{end}: {}",
                    closed.text
                );
            }
        }
    }

    #[test]
    fn the_cut_names_the_keys_it_was_writing() {
        let open = |s: &str| close_cut(s).unwrap().open;
        assert_eq!(
            open(r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a"#),
            ["arguments", "claims", "text"]
        );
        assert_eq!(
            open(r#"{"name": "journal_write", "arguments": {"claims": [], "summary": "Fo"#),
            ["arguments", "summary"]
        );
        assert_eq!(
            open(r#"{"name": "journal_write", "arguments": {"claims": [{"text": "a"}], "#),
            ["arguments"]
        );
    }
}
