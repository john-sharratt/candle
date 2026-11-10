//! Scoring one answer against its question's `expect` list.
//!
//! - **Recall** — the share of `expect` items the answer names. An item is a
//!   set of `|`-separated spellings; any one of them, found case-insensitively
//!   as a substring of the answer, names it.
//! - **Grounding** — the share of the answer's backticked spans whose every
//!   word occurs in the repository: a `foo::bar_baz` that the code never
//!   spells is a name the model made up. This is the battery's precision
//!   measure, and it is a lower bound on hallucination, not a check that a
//!   named dependency is a real one.

use std::collections::HashSet;

/// The text after the reasoning block.
pub fn answer(text: &str) -> &str {
    match text.rfind("</think>") {
        Some(i) => text[i + "</think>".len()..].trim(),
        None => text.trim(),
    }
}

/// Whether `answer` names one `expect` item.
pub fn names(answer: &str, item: &str) -> bool {
    let a = answer.to_lowercase();
    item.split('|')
        .map(|s| s.trim().to_lowercase())
        .any(|s| !s.is_empty() && a.contains(&s))
}

/// The share of `expect` the answer names, and the items it missed.
pub fn recall(answer: &str, expect: &[String]) -> (f64, Vec<String>) {
    let missed: Vec<String> = expect
        .iter()
        .filter(|e| !names(answer, e))
        .cloned()
        .collect();
    let n = expect.len().max(1) as f64;
    ((expect.len() - missed.len()) as f64 / n, missed)
}

/// The backticked spans of `text`.
pub fn spans(text: &str) -> Vec<&str> {
    text.split('`')
        .enumerate()
        .filter(|(i, s)| i % 2 == 1 && !s.trim().is_empty() && !s.contains('\n'))
        .map(|(_, s)| s.trim())
        .collect()
}

/// The identifier-like words of `s`: runs of letters, digits and `_`.
pub fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
}

/// The share of `answer`'s backticked spans whose every word is in
/// `vocabulary`, and the spans that were not. `None` when it has no spans.
pub fn grounding(answer: &str, vocabulary: &HashSet<String>) -> Option<(f64, Vec<String>)> {
    let spans = spans(answer);
    if spans.is_empty() {
        return None;
    }
    let unknown: Vec<String> = spans
        .iter()
        .filter(|s| !words(s).all(|w| vocabulary.contains(w)))
        .map(|s| s.to_string())
        .collect();
    Some((
        (spans.len() - unknown.len()) as f64 / spans.len() as f64,
        unknown,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_answer_is_what_follows_the_reasoning() {
        assert_eq!(
            answer("<think>scan.rs</think>\n  It is in gpu.rs."),
            "It is in gpu.rs."
        );
        assert_eq!(answer(" plain "), "plain");
    }

    /// Any spelling names the item, case-insensitively, inside a longer path.
    #[test]
    fn an_item_is_named_by_any_of_its_spellings() {
        let a = "Defined in `candle-conversation/src/provenance/Scan.rs`; keys are unrotated.";
        assert!(names(a, "scan.rs"));
        assert!(names(a, "un-rotated|unrotated"));
        assert!(!names(a, "gpu.rs"));
        assert!(!names(a, " | "), "an empty spelling names nothing");
    }

    #[test]
    fn recall_counts_the_items_named() {
        let expect = vec![
            "scan.rs".to_string(),
            "gpu.rs".to_string(),
            "x|y".to_string(),
        ];
        let (r, missed) = recall("scan.rs and y", &expect);
        assert_eq!(r, 2.0 / 3.0);
        assert_eq!(missed, vec!["gpu.rs".to_string()]);
    }

    /// Only the spans between backticks count, and never one across lines.
    #[test]
    fn spans_are_the_backticked_runs() {
        assert_eq!(
            spans("call `a::b` then `c` and ``, then `d\ne`"),
            vec!["a::b", "c"]
        );
    }

    #[test]
    fn a_span_is_grounded_when_every_word_is_in_the_code() {
        let vocab: HashSet<String> = ["heads_per_group", "scan", "rs"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (g, unknown) =
            grounding("`heads_per_group` in `scan.rs`, not `invented_fn`", &vocab).unwrap();
        assert_eq!(g, 2.0 / 3.0);
        assert_eq!(unknown, vec!["invented_fn".to_string()]);
        assert_eq!(grounding("no spans here", &vocab), None);
    }
}
