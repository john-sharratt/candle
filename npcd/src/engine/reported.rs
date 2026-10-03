//! Somebody else's intent, put into the third person.
//!
//! A character gives what it *means* to convey, written as its own first-person
//! thought ("that I am back at the muster hall"). The speaker is told it in the
//! second person; a listener who is handed the same words reads "I" as itself,
//! and starts reporting the speaker's whereabouts and errands as its own. The
//! listener's copy shifts the speaker's "I" to "they".

/// Shift the first person singular to the third, with the verbs that agree:
/// "I am" becomes "they are", "I was" becomes "they were".
pub fn third_person(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut word = String::new();
    let mut after_i = false;
    let mut starts_sentence = true;
    for c in text.chars() {
        if c.is_alphabetic() || c == '\'' || c == '\u{2019}' {
            word.push(c);
            continue;
        }
        flush(&mut out, &mut word, &mut after_i, &mut starts_sentence);
        if matches!(c, '.' | '!' | '?') {
            starts_sentence = true;
        } else if !c.is_whitespace() {
            starts_sentence = false;
        }
        out.push(c);
    }
    flush(&mut out, &mut word, &mut after_i, &mut starts_sentence);
    out
}

fn flush(out: &mut String, word: &mut String, after_i: &mut bool, starts_sentence: &mut bool) {
    if word.is_empty() {
        return;
    }
    let plain = word.replace('\u{2019}', "'").to_lowercase();
    let shifted = match plain.as_str() {
        "i" => Some("they"),
        "i'm" => Some("they're"),
        "i've" => Some("they've"),
        "i'll" => Some("they'll"),
        "i'd" => Some("they'd"),
        "me" => Some("them"),
        "my" => Some("their"),
        "myself" => Some("themselves"),
        "am" if *after_i => Some("are"),
        "was" if *after_i => Some("were"),
        _ => None,
    };
    *after_i = plain == "i";
    match shifted {
        Some(s) => out.push_str(&cased(s, *starts_sentence)),
        None => out.push_str(word),
    }
    *starts_sentence = false;
    word.clear();
}

fn cased(word: &str, capital: bool) -> String {
    let mut chars = word.chars();
    match (capital, chars.next()) {
        (true, Some(first)) => first.to_uppercase().chain(chars).collect(),
        _ => word.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i_am_becomes_they_are() {
        assert_eq!(
            third_person("that I am back in the muster hall"),
            "that they are back in the muster hall"
        );
    }

    #[test]
    fn contractions_and_the_past_agree() {
        assert_eq!(
            third_person("that I'm sure I was there and I've seen it, I'll say so"),
            "that they're sure they were there and they've seen it, they'll say so"
        );
    }

    #[test]
    fn the_object_and_the_possessive_shift_too() {
        assert_eq!(
            third_person("whether he needs me, or my ledger, for myself"),
            "whether he needs them, or their ledger, for themselves"
        );
    }

    #[test]
    fn a_sentence_that_opens_with_i_still_opens_with_a_capital() {
        assert_eq!(
            third_person("the pipe knocked. I need to know who touched it"),
            "the pipe knocked. They need to know who touched it"
        );
        assert_eq!(third_person("I need the ledger"), "They need the ledger");
    }

    #[test]
    fn words_that_merely_contain_i_are_left_alone() {
        assert_eq!(
            third_person("it is in the mine, I think, said Pam"),
            "it is in the mine, they think, said Pam"
        );
    }

    #[test]
    fn nothing_first_person_means_nothing_changes() {
        let text = "that the temperature has risen — and you should look";
        assert_eq!(third_person(text), text);
    }
}
