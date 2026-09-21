//! Which branch arm the reasoning before a call named.
//!
//! A model can reason its way to one tool and then decode another: live, a turn
//! whose thinking ended "use the `write` tool" wrote a `file_read` call forty
//! times over, and one that settled on the weather tool called `file_search`.
//! The name branch is a free choice among every catalog name, and the choice is
//! sampled a token at a time with nothing tying it to the sentence just
//! written. [`named_arm`] reads that sentence back: when the reasoning names
//! exactly one of the arms, the driver steers the branch to it
//! ([`StencilDriver::steer_first_branch`](super::StencilDriver::steer_first_branch)).

use std::str::from_utf8;

/// The name an arm's bytes spell — its identifier, skipping whatever lead-in the
/// tokenizer merged into the arm (a quote, a space) and the delimiter that
/// closes it (`"`, `>`).
pub fn arm_name(bytes: &[u8]) -> &str {
    let start = bytes
        .iter()
        .position(|&b| is_ident(b))
        .unwrap_or(bytes.len());
    let len = bytes[start..]
        .iter()
        .position(|&b| !is_ident(b))
        .unwrap_or(bytes.len() - start);
    from_utf8(&bytes[start..start + len]).unwrap_or("")
}

/// The index of the one name in `names` that `reasoning` settles on, if it
/// settles on exactly one.
///
/// Only the reasoning since the last call counts (the text after the last
/// `</tool_call>`), and within it the last paragraph that names any tool — the
/// model's final word, not the options it weighed on the way. That paragraph
/// must name exactly one distinct tool; two is a comparison, not a choice, and
/// steers nothing.
///
/// A name is a mention when it stands as a whole word **in a clause that
/// decides to call it** — one with "use", "call" or "invoke" before the name
/// and no negation. Reasoning names tools it has already run as often as the
/// one it is about to: "the file_read returned not_found, so I'll create it"
/// is about `file_read` but chooses something else, and steering to it would
/// write the very call that just failed. A name with no `_` is also an
/// ordinary word (`write`, `grep`), so it counts only where it is plainly a
/// tool: in backticks, or followed by "tool" or "function".
pub fn named_arm(reasoning: &str, names: &[&str]) -> Option<usize> {
    let scope = reasoning.rsplit("</tool_call>").next().unwrap_or(reasoning);
    scope
        .split("\n\n")
        .map(|paragraph| mentioned(paragraph, names))
        .filter(|found| !found.is_empty())
        .last()
        .and_then(|found| match found.as_slice() {
            [one] => Some(*one),
            _ => None,
        })
}

/// The distinct indices of `names` mentioned in `text`, in index order.
fn mentioned(text: &str, names: &[&str]) -> Vec<usize> {
    names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.is_empty() && mentions(text, name))
        .map(|(i, _)| i)
        .collect()
}

/// Whether `text` mentions the tool `name` (see [`named_arm`]).
fn mentions(text: &str, name: &str) -> bool {
    let bytes = text.as_bytes();
    let plain_word = !name.contains('_');
    text.match_indices(name).any(|(at, _)| {
        let end = at + name.len();
        let before = at.checked_sub(1).map(|i| bytes[i]);
        let after = bytes.get(end).copied();
        if before.is_some_and(is_ident) || after.is_some_and(is_ident) {
            return false;
        }
        if !decided(&text[..at]) {
            return false;
        }
        if !plain_word {
            return true;
        }
        let backticked = before == Some(b'`') && after == Some(b'`');
        let rest = text[end..].trim_start_matches('`').trim_start();
        backticked || rest.starts_with("tool") || rest.starts_with("function")
    })
}

/// Words that make a clause a decision to call the tool named after them.
const CUES: &[&str] = &["use", "using", "call", "calling", "invoke", "invoking"];

/// Words that turn a clause into a decision *against* the tool it names.
const NEGATIONS: &[&str] = &[
    "not", "never", "no", "instead", "rather", "without", "avoid",
];

/// Whether the clause ending at a mention (`lead` is the text before it)
/// decides to call the tool: a cue word, and no negation.
fn decided(lead: &str) -> bool {
    let clause_start = lead
        .rfind(['\n', ',', ';', ':', '!', '?'])
        .map_or(0, |i| i + 1)
        .max(lead.rfind(". ").map_or(0, |i| i + 2));
    let clause = lead[clause_start..].to_ascii_lowercase();
    let words: Vec<&str> = clause
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .collect();
    words.iter().any(|w| CUES.contains(w))
        && !words
            .iter()
            .any(|w| NEGATIONS.contains(w) || w.ends_with("n't"))
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: &[&str] = &["file_read", "file_list", "write", "weather_current"];

    #[test]
    fn an_arm_is_named_by_its_identifier() {
        assert_eq!(arm_name(b"file_read\""), "file_read");
        assert_eq!(arm_name(b" \"write\""), "write");
        assert_eq!(arm_name(b"say>"), "say");
        assert_eq!(arm_name(b"\""), "");
    }

    #[test]
    fn a_single_named_tool_is_chosen() {
        let r = "The user wants the notes. I should use file_read on notes.md.\n</think>\n\n";
        assert_eq!(named_arm(r, NAMES), Some(0));
    }

    #[test]
    fn a_plain_word_name_counts_only_as_a_tool() {
        assert_eq!(named_arm("I need to write the file.", NAMES), None);
        assert_eq!(named_arm("I'll use the `write` tool.", NAMES), Some(2));
        assert_eq!(named_arm("Call write tool with the path.", NAMES), Some(2));
        assert_eq!(named_arm("I'll call the write function", NAMES), Some(2));
    }

    /// **A tool the reasoning reports on is not the tool it chooses.** The
    /// review that found this: steering to `file_read` here writes the call
    /// that just failed.
    #[test]
    fn a_tool_named_without_a_decision_steers_nothing() {
        let r = "The file_read returned not_found, so the file doesn't exist yet. \
                 I'll create it now.";
        assert_eq!(named_arm(r, NAMES), None);
        let r = "The file_list output shows main.rs; now read it.";
        assert_eq!(named_arm(r, NAMES), None);
    }

    /// A decision against a tool is not a decision for it.
    #[test]
    fn a_negated_tool_steers_nothing() {
        assert_eq!(named_arm("I shouldn't use file_list here.", NAMES), None);
        assert_eq!(named_arm("Do not call file_read again.", NAMES), None);
        assert_eq!(
            named_arm("Rather than file_list, I'll use file_read.", NAMES),
            Some(0)
        );
    }

    #[test]
    fn a_name_inside_a_longer_identifier_is_not_a_mention() {
        let names = ["file_read", "file_read_range"];
        assert_eq!(named_arm("use file_read_range here", &names), Some(1));
    }

    #[test]
    fn the_last_paragraph_that_names_a_tool_decides() {
        let r = "Maybe use file_list first to see the folder.\n\n\
                 No, the file does not exist yet, so use the `write` tool.\n\n\
                 Content ready.\n</think>\n\n";
        assert_eq!(named_arm(r, NAMES), Some(2));
    }

    #[test]
    fn two_names_in_the_deciding_paragraph_steer_nothing() {
        let r = "I could use file_read or use file_list.";
        assert_eq!(named_arm(r, NAMES), None);
    }

    #[test]
    fn only_reasoning_since_the_last_call_counts() {
        let r = "use file_list\n<tool_call>\n{}\n</tool_call>\nnow use weather_current for Paris";
        assert_eq!(named_arm(r, NAMES), Some(3));
        assert_eq!(
            named_arm("done with it\n</tool_call>\nanswering now", NAMES),
            None
        );
    }
}
