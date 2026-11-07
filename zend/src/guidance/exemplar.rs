//! What a read teaches: the case a use-exemplar seals.
//!
//! A live probe is the trailing window of what the model is decoding, and at
//! the moment it reaches for a guide that is its reasoning just before the
//! call. So the case keeps the turn's request as its user half — the task the
//! reasoning serves — and the END of the reasoning as its answer, a closed
//! block, the tokens a live turn lays down when it thinks the same thing.

use crate::calibrate::thinking::exemplar;

/// The most of the turn's request a case keeps, from its start: the task it
/// names comes first.
pub const ASKED_CHARS: usize = 600;

/// The most of the reasoning a case keeps, from its end: about the 256 tokens
/// a live probe's trailing window holds (`calibrate::spec::MAX_THINK`).
pub const THOUGHT_CHARS: usize = 1024;

/// The case a read sealed after `thought`, in a turn asked `asked`: the
/// request's start and the reasoning's end. `None` when the turn wrote no
/// reasoning before the call — then nothing says why the unit was wanted.
pub fn use_case(asked: &str, thought: &str) -> Option<(String, String)> {
    let thought = thought.trim();
    if thought.is_empty() {
        return None;
    }
    let asked = head(asked.trim(), ASKED_CHARS);
    Some((asked.to_string(), exemplar(tail(thought, THOUGHT_CHARS))))
}

/// The first `chars` characters of `text`.
fn head(text: &str, chars: usize) -> &str {
    match text.char_indices().nth(chars) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

/// The last `chars` characters of `text`.
fn tail(text: &str, chars: usize) -> &str {
    let n = text.chars().count();
    if n <= chars {
        return text;
    }
    let (at, _) = text
        .char_indices()
        .nth(n - chars)
        .expect("n - chars < n is a character index");
    &text[at..]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A case is the request and the reasoning that reached for the unit**,
    /// the reasoning as a closed block.
    #[test]
    fn a_case_is_the_request_and_the_reasoning_as_a_closed_block() {
        assert_eq!(
            use_case(
                "  Deploy the fix to BlackWorld.  ",
                " It is on main; now it has to reach BlackWorld. "
            ),
            Some((
                "Deploy the fix to BlackWorld.".to_string(),
                "<think>\nIt is on main; now it has to reach BlackWorld.\n</think>".to_string()
            ))
        );
    }

    /// No reasoning before the call, no case.
    #[test]
    fn without_reasoning_there_is_no_case() {
        assert_eq!(use_case("Deploy it.", "  \n "), None);
    }

    /// **The request keeps its start and the reasoning its end**, cut on
    /// character boundaries.
    #[test]
    fn the_request_keeps_its_start_and_the_reasoning_its_end() {
        let asked = format!("{}é", "a".repeat(ASKED_CHARS));
        let thought = format!("é{}", "b".repeat(THOUGHT_CHARS));
        let (user, answer) = use_case(&asked, &thought).unwrap();
        assert_eq!(user, "a".repeat(ASKED_CHARS));
        assert_eq!(
            answer,
            format!("<think>\n{}\n</think>", "b".repeat(THOUGHT_CHARS))
        );
        assert_eq!(head("éa", 1), "é");
        assert_eq!(tail("aé", 1), "é");
        assert_eq!(tail("ab", 5), "ab");
    }
}
