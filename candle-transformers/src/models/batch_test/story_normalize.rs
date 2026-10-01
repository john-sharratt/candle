//! The comparison form of a StoryRewrite output.
//!
//! The gate asks the model to copy a story with the protagonist renamed, so a
//! session with a female name correctly rewrites the story's pronouns and its
//! spouse. [`normalize_story`] collapses whitespace runs to one space and
//! replaces every gendered word with one neutral placeholder per pair, so the
//! correct rewrite compares equal to the original and anything else does not.
//!
//! Words are matched whole, at any boundary — a space, an em-dash, a quote,
//! punctuation. "knew better—she had" and "knew better—he had" are the same
//! sentence about a different spouse, and a match that insisted on spaces
//! either side would fail a session for the model's correct choice.

/// `text` with whitespace runs collapsed, trimmed, and every gendered word
/// neutralised.
pub fn normalize_story(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = String::with_capacity(collapsed.len() + 32);
    let mut word = String::new();
    let mut chars = collapsed.chars().peekable();
    while let Some(ch) = chars.next() {
        // An apostrophe between letters belongs to the word ("she'd").
        let inner_apostrophe = is_apostrophe(ch)
            && !word.is_empty()
            && chars.peek().is_some_and(|c| c.is_alphabetic());
        if ch.is_alphabetic() || inner_apostrophe {
            word.push(ch);
        } else {
            flush(&mut out, &mut word);
            out.push(ch);
        }
    }
    flush(&mut out, &mut word);
    out
}

fn is_apostrophe(ch: char) -> bool {
    ch == '\'' || ch == '\u{2019}'
}

fn flush(out: &mut String, word: &mut String) {
    if !word.is_empty() {
        out.push_str(&neutral(word));
        word.clear();
    }
}

/// The neutral form of one whole word, or the word itself.
fn neutral(word: &str) -> String {
    let straight = word.replace('\u{2019}', "'");
    let mapped = match straight.as_str() {
        "his" | "her" => "[his/her]",
        "His" | "Her" => "[His/Her]",
        "he" | "she" => "[he/she]",
        "He" | "She" => "[He/She]",
        "him" => "[him/her]",
        "Him" => "[Him/Her]",
        "wife" | "husband" => "[wife/husband]",
        // The model may expand or contract these equivalently.
        "he'd" | "she'd" => "[he/she] had",
        "He'd" | "She'd" => "[He/She] had",
        _ => return word.to_string(),
    };
    mapped.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pronoun after an em-dash is a word like any other: the rewrite that
    /// gave the spouse the other pronoun compares equal to the original.
    #[test]
    fn a_pronoun_after_an_em_dash_is_neutralised() {
        assert_eq!(
            normalize_story("knew better—she had watched"),
            "knew better—[he/she] had watched"
        );
        assert_eq!(
            normalize_story("knew better—he had watched"),
            "knew better—[he/she] had watched"
        );
    }

    #[test]
    fn whitespace_runs_collapse_and_ends_trim() {
        assert_eq!(
            normalize_story("  The Backyard Astronaut  \nMarcus had\tbeen "),
            "The Backyard Astronaut Marcus had been"
        );
    }

    #[test]
    fn contractions_expand_before_they_neutralise() {
        assert_eq!(
            normalize_story("She'd left his tools; he\u{2019}d not."),
            "[He/She] had left [his/her] tools; [he/she] had not."
        );
    }

    #[test]
    fn spouses_and_object_pronouns_neutralise_at_punctuation() {
        assert_eq!(
            normalize_story("Her husband, \"him\"; His wife."),
            "[His/Her] [wife/husband], \"[him/her]\"; [His/Her] [wife/husband]."
        );
    }

    /// Only whole words: a pronoun inside another word is left alone, and so
    /// is a possessive that is not a pronoun.
    #[test]
    fn pronouns_inside_other_words_are_untouched() {
        assert_eq!(
            normalize_story("There these shelves whim Hermes Marcus's"),
            "There these shelves whim Hermes Marcus's"
        );
    }

    /// A name is not neutralised, so a rewrite under the wrong name still
    /// differs from the expected text.
    #[test]
    fn the_name_still_distinguishes_two_rewrites() {
        assert_ne!(
            normalize_story("The Backyard Astronaut Emily had"),
            normalize_story("The Backyard Astronaut Marcus had")
        );
    }
}
