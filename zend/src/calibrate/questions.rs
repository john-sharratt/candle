//! The questions in a unit's answer to its calibration prompt.
//!
//! Asked for one question per line, a model still numbers them, bullets them,
//! labels them, quotes them, or opens with a sentence of its own. Each line
//! is taken for what it says: its list marker and label dropped, kept only if
//! it is a question, and kept once.

/// Up to `n` distinct questions in `answer`, in the order written.
pub fn questions(answer: &str, n: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for line in answer.lines() {
        if out.len() == n {
            break;
        }
        let q = unmarked(line);
        if !q.ends_with('?') || q.split_whitespace().count() < 3 {
            continue;
        }
        let key = q.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        out.push(q.to_string());
    }
    out
}

/// `line` without its list marker, label, emphasis or surrounding quotes.
fn unmarked(line: &str) -> &str {
    let mut s = line.trim();
    // A bullet, or a number closed by `.`, `)` or `:`.
    if let Some(rest) = s.strip_prefix(['-', '*', '•']) {
        s = rest.trim_start();
    } else {
        let digits = s.chars().take_while(char::is_ascii_digit).count();
        if digits > 0 {
            if let Some(rest) = s[digits..].strip_prefix(['.', ')', ':']) {
                s = rest.trim_start();
            }
        }
    }
    for label in ["Question:", "Q:"] {
        if let Some(rest) = s.strip_prefix(label) {
            s = rest.trim_start();
        }
    }
    s = s.trim_matches(|c| c == '*' || c == '_').trim();
    for (open, close) in [('"', '"'), ('“', '”'), ('\'', '\'')] {
        if let Some(inner) = s.strip_prefix(open).and_then(|r| r.strip_suffix(close)) {
            s = inner.trim();
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_labels_and_quotes_are_dropped() {
        let answer = "Here are the questions:\n\
            1. What does the scheduler admit first?\n\
            2) Where is the wave loop?\n\
            - How do I add a layer?\n\
            * **Why are locks released on a write?**\n\
            Q: Which file defines the arena table?\n\
            \"What is the probe window?\"\n\
            3: “How is a file read?”";
        assert_eq!(
            questions(answer, 16),
            vec![
                "What does the scheduler admit first?",
                "Where is the wave loop?",
                "How do I add a layer?",
                "Why are locks released on a write?",
                "Which file defines the arena table?",
                "What is the probe window?",
                "How is a file read?",
            ]
        );
    }

    /// A line that is not a question, a fragment, and a repeat are left
    /// out; the count caps what is kept.
    #[test]
    fn only_distinct_whole_questions_are_kept_up_to_the_count() {
        let answer = "What is this?\nwhy?\nThis is a statement.\n\
            What is the cache for?\nwhat is the cache for?\nHow is it built?\n\
            Where does it live?";
        assert_eq!(
            questions(answer, 16),
            vec![
                "What is this?",
                "What is the cache for?",
                "How is it built?",
                "Where does it live?"
            ]
        );
        assert_eq!(
            questions(answer, 2),
            vec!["What is this?", "What is the cache for?"]
        );
        assert!(questions("", 4).is_empty());
    }
}
