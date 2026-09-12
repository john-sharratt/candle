//! Turning counts and lists into English.
//!
//! Shared by everything that writes prose here — the memory, the point-in-time
//! percept, the stream of what changed — so that a list of three reads the
//! same way whichever of them is speaking. A reader crossing between them
//! should not be able to tell which module wrote a sentence.

/// "a", "a and b", "a, b and c".
///
/// Used for nouns and for clauses alike, which take the same shape: *said
/// something, let go of her and left*.
pub fn list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [a, b] => format!("{a} and {b}"),
        _ => {
            let (last, rest) = items.split_last().expect("non-empty");
            format!("{} and {}", rest.join(", "), last)
        }
    }
}

/// "a", "a or b", "a, b or c".
///
/// The disjunctive twin of [`list`], for a set where any one member will do —
/// the stations an act can be done at, the ways out of a room. Using "and"
/// there reads as needing all of them, which is the opposite of what is meant
/// and the sort of thing a reader believes.
pub fn list_or(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [a, b] => format!("{a} or {b}"),
        _ => {
            let (last, rest) = items.split_last().expect("non-empty");
            format!("{} or {}", rest.join(", "), last)
        }
    }
}

/// Small numbers as words, large ones as digits.
///
/// "Six levels" reads like prose and "6 levels" reads like a form; past twenty
/// the word is longer than the number and the effect reverses.
pub fn spell(n: usize) -> String {
    const WORDS: [&str; 21] = [
        "no",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ];
    WORDS
        .get(n)
        .map(|w| w.to_string())
        .unwrap_or_else(|| n.to_string())
}

pub fn plural(word: &str, n: usize) -> String {
    if n == 1 {
        word.to_string()
    } else {
        format!("{word}s")
    }
}

pub fn is_are(n: usize) -> &'static str {
    if n == 1 {
        "is"
    } else {
        "are"
    }
}

pub fn cap(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// Collapse an authored block scalar onto one line, so it joins a generated
/// sentence without carrying the file's own line breaks into the prose.
pub fn tidy(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_list_reads_as_a_sentence() {
        assert_eq!(list(&[]), "");
        assert_eq!(list(&["a".into()]), "a");
        assert_eq!(list(&["a".into(), "b".into()]), "a and b");
        assert_eq!(list(&["a".into(), "b".into(), "c".into()]), "a, b and c");
    }

    #[test]
    fn small_counts_are_words_and_large_ones_are_not() {
        assert_eq!(spell(0), "no");
        assert_eq!(spell(16), "sixteen");
        assert_eq!(spell(40), "40");
    }

    #[test]
    fn plurals_and_agreement_follow_the_count() {
        assert_eq!(plural("station", 1), "station");
        assert_eq!(plural("station", 3), "stations");
        assert_eq!(is_are(1), "is");
        assert_eq!(is_are(2), "are");
    }

    #[test]
    fn tidying_flattens_an_authored_block() {
        assert_eq!(tidy("one\n  two\n  three"), "one two three");
    }
}
