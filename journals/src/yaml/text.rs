//! Writing free text — a line, a cue, a summary's sentence, a name — as a
//! field a person reads easily: on its key's line when it fits, plain where
//! YAML reads it back as itself and double-quoted where not; longer, a
//! folded block (`>-`) wrapped within [`WIDTH`] columns.
//!
//! Every form is checked against the reader before it is written: the text
//! read back from it must be the text exactly. A longer text no folded block
//! holds exactly — a line break, a leading or trailing space, a control
//! character — is double-quoted on its key's line instead ([`quote`]).

use serde_yaml::Value;

use crate::yaml::scalar::quote;

/// The columns a file's text is wrapped within.
pub(crate) const WIDTH: usize = 80;

/// The narrowest a folded block's text is wrapped, however deep its key.
const MIN_WRAP: usize = 40;

/// `key: text` as lines, each ending in a newline. `lead` is what precedes
/// the key on its line — its indentation, and `- ` for a list item's first
/// field — and the key's column is its length.
pub(crate) fn text_field(lead: &str, key: &str, text: &str) -> String {
    let col = lead.chars().count();
    if col + key.len() + 2 + text.chars().count() <= WIDTH {
        return if reads_back(key, text, text) {
            format!("{lead}{key}: {text}\n")
        } else {
            format!("{lead}{key}: {}\n", quote(text))
        };
    }
    let indent = " ".repeat(col + 2);
    let wrapped = wrap(text, WIDTH.saturating_sub(col + 2).max(MIN_WRAP));
    let block: String = wrapped.iter().map(|l| format!("  {l}\n")).collect();
    if !wrapped.is_empty() && reads_back(key, &format!(">-\n{block}"), text) {
        let body: String = wrapped.iter().map(|l| format!("{indent}{l}\n")).collect();
        return format!("{lead}{key}: >-\n{body}");
    }
    format!("{lead}{key}: {}\n", quote(text))
}

/// `key: value` at the left margin reads back as exactly `text`. A block
/// scalar indented uniformly deeper means the same, so checking at the
/// margin answers for every depth it is written at.
fn reads_back(key: &str, value: &str, text: &str) -> bool {
    match serde_yaml::from_str::<Value>(&format!("{key}: {value}")) {
        Ok(v) => v.get(key).and_then(Value::as_str) == Some(text),
        Err(_) => false,
    }
}

/// `text` broken into lines of at most `width` characters, at single spaces
/// between two other characters only: a folded block joins its lines with
/// one space, so a break anywhere else — beside another space, which would
/// leave a line ending or starting in one — would not read back. A word
/// wider than `width` takes a line of its own.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let breakable = |i: usize| {
        chars[i] == ' '
            && i > 0
            && i + 1 < chars.len()
            && chars[i - 1] != ' '
            && chars[i + 1] != ' '
    };
    let mut lines = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        if chars.len() - start <= width {
            lines.push(chars[start..].iter().collect());
            break;
        }
        let last = (start + 1..=start + width)
            .rev()
            .find(|&i| i < chars.len() && breakable(i));
        let at = match last {
            Some(i) => i,
            None => match (start + width + 1..chars.len()).find(|&i| breakable(i)) {
                Some(i) => i,
                None => {
                    lines.push(chars[start..].iter().collect());
                    break;
                }
            },
        };
        lines.push(chars[start..at].iter().collect());
        start = at + 1;
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `text` field as written after `lead`, read back in its place: at
    /// the margin, under a parent key `p:` when indented, and as the first
    /// field of a list item when the lead ends in `- `.
    fn back(lead: &str, text: &str) -> String {
        let field = text_field(lead, "text", text);
        let nested = lead.starts_with(' ');
        let doc = if nested {
            format!("p:\n{field}")
        } else {
            field.clone()
        };
        let v: Value = serde_yaml::from_str(&doc).unwrap_or_else(|e| panic!("{e}: {doc}"));
        let mut at = &v;
        if nested {
            at = &at["p"];
        }
        if lead.ends_with("- ") {
            at = &at[0];
        }
        at["text"]
            .as_str()
            .unwrap_or_else(|| panic!("no text in {doc}"))
            .to_string()
    }

    #[test]
    fn short_text_sits_plain_on_its_key_line() {
        assert_eq!(
            text_field("", "cue", "open for council permits"),
            "cue: open for council permits\n"
        );
        assert_eq!(
            text_field(
                "  ",
                "text",
                "Workers push branches; merging is the user's decision."
            ),
            "  text: Workers push branches; merging is the user's decision.\n"
        );
    }

    #[test]
    fn short_text_yaml_would_misread_is_quoted() {
        for (text, written) in [
            ("a: b", "k: \"a: b\"\n"),
            ("true", "k: \"true\"\n"),
            ("2026", "k: \"2026\"\n"),
            ("#not a comment", "k: \"#not a comment\"\n"),
            ("- not an item", "k: \"- not an item\"\n"),
            ("trailing ", "k: \"trailing \"\n"),
            ("line\nbreak", "k: \"line\\nbreak\"\n"),
            ("", "k: \"\"\n"),
        ] {
            assert_eq!(text_field("", "k", text), written, "{text:?}");
        }
    }

    #[test]
    fn long_text_is_a_folded_block_wrapped_within_the_width() {
        let text = "The candle fork's rationale: the context-window problem is solved by \
                    retrieving only what is provably relevant, not by scaling memory, so error \
                    stays O(1) rather than O(N).";
        assert_eq!(
            text_field("  ", "text", text),
            "  text: >-\n\
             \x20   The candle fork's rationale: the context-window problem is solved by\n\
             \x20   retrieving only what is provably relevant, not by scaling memory, so error\n\
             \x20   stays O(1) rather than O(N).\n"
        );
        for line in text_field("  ", "text", text).lines() {
            assert!(line.chars().count() <= WIDTH, "{line}");
        }
        assert_eq!(back("  ", text), text);
    }

    /// Whatever the text, it reads back as exactly itself, at any depth and
    /// as a list item's first field.
    #[test]
    fn every_text_reads_back_as_itself() {
        let long = |s: &str| format!("{s} {}", "word ".repeat(30).trim_end());
        for text in [
            long("plain"),
            long("a: b, {c} [d] #e \"f\" 'g' & * ! | > % @ `"),
            long("double  spaces  kept"),
            long("ends with a space "),
            format!(" {}", long("starts with a space")),
            long("line\nbreak"),
            long("tab\there"),
            long("nul\u{0}byte"),
            long("\u{2028}separator\u{85}nel\u{feff}bom"),
            long("emoji 🚗 and đà nẵng"),
            format!("{} end", "x".repeat(200)),
            long("--- a document marker ... and its end"),
            long("- a dash"),
        ] {
            for lead in ["", "  ", "    - ", "- "] {
                assert_eq!(back(lead, &text), text, "{lead:?} {text:?}");
            }
        }
    }

    #[test]
    fn a_long_word_takes_a_line_of_its_own() {
        let word = "x".repeat(100);
        let text = format!("short {word} tail");
        assert_eq!(
            wrap(&text, 40),
            vec!["short".to_string(), word, "tail".into()]
        );
        assert_eq!(wrap("a  b", 1), vec!["a  b".to_string()]);
    }
}
