//! Bounds generated text to its word caps and detects text that goes round in a loop.
//!
//! A text field is closed by the grammar at a token count derived from its word cap, with room to spare, so a field that degenerates costs only its own length. A list is an array whose entries are each closed and whose count is bounded, because an unbounded list repeats the same entries until the answer is refused. A list asked for as a string is lost to the grammar, so a list is declared as one and read back joined a line each. Loops are found by comparing the opening four words of each sentence, and a text whose sentence or opening recurs three times or more is treated as looping.

use super::*;

pub(in super::super) fn text(name: &str) -> CallParam {
    CallParam {
        name: name.to_string(),
        ty: ParamType::String,
        required: true,
        enum_values: None,
        items: None,
        min_items: 0,
        max_items: None,
        properties: None,
        nullable: false,
        minimum: None,
        maximum: None,
        max_tokens: None,
        requires: Vec::new(),
        shapes: Vec::new(),
        pick: Default::default(),
        pattern: None,
    }
}

/// [`text`] for a field checked against a cap of `words`: the grammar closes
/// it at [`tokens_for`] that many, so a field that degenerates costs a field's
/// length and not the whole call's.
pub(in super::super) fn text_within(name: &str, words: usize) -> CallParam {
    CallParam {
        max_tokens: Some(tokens_for(words)),
        ..text(name)
    }
}

/// The tokens `words` of prose can take, with room to spare: English runs
/// about four tokens to three words, and a field at its cap must close itself
/// rather than be cut.
pub(super) fn tokens_for(words: usize) -> u32 {
    (words * 2 + 16) as u32
}

/// A field that is a list: an array of strings, each closed at [`tokens_for`]
/// `words`, and `items` of them.
///
/// **A list has an end.** Bounded only by the grammar's own unrolling, a
/// story's `who` went round to 586 words — the same people named again and
/// again — and the answer was refused for it.
///
/// **A list asked for as a string is written as an array, and lost.** A
/// table's reading planned "the faults array" in its thinking and opened the
/// value with `[`; a string value that opens on anything but its quote is the
/// model skipping it, and the grammar writes it empty — `"faults": ""` in 31
/// readings of 32, the faults it had drafted ("Fault 5: …") with them. What is
/// a list is declared one, and read back joined a line each ([`Fields::get`]).
pub(in super::super) fn list_within(
    name: &str,
    words: usize,
    items: RangeInclusive<usize>,
) -> CallParam {
    CallParam {
        ty: ParamType::Array,
        items: Some(Box::new(CallParam {
            name: String::new(),
            ..text_within(name, words)
        })),
        min_items: *items.start(),
        max_items: Some(*items.end()),
        ..text(name)
    }
}

/// `text` up to the last sentence that ends within its first `max` words.
pub(super) fn within_words(text: &str, max: usize) -> String {
    let mut words = 0;
    let mut end = 0;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() && i > 0 && !text[..i].ends_with(char::is_whitespace) {
            words += 1;
            if words >= max {
                break;
            }
        }
        if matches!(c, '.' | '!' | '?') {
            end = i + c.len_utf8();
        }
    }
    text[..end].trim().to_string()
}

/// The opening of a sentence as [`loops`] compares it: its first four words,
/// lower-cased — `None` for one too short to tell apart.
pub(super) fn opening(sentence: &str) -> Option<String> {
    let words: Vec<String> = sentence
        .split_whitespace()
        .take(4)
        .map(|w| w.to_lowercase())
        .collect();
    (words.len() >= 3).then(|| words.join(" "))
}

/// `text` up to the first sentence of its loop: the earliest place an opening
/// that comes back three times or more first appears.
pub(super) fn before_the_loop(text: &str) -> String {
    let mut sentences: Vec<(usize, Option<String>)> = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if matches!(c, '.' | '!' | '?' | ';') {
            sentences.push((start, opening(&text[start..i])));
            start = i + c.len_utf8();
        }
    }
    sentences.push((start, opening(&text[start..])));
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, o) in &sentences {
        if let Some(o) = o {
            *counts.entry(o.as_str()).or_default() += 1;
        }
    }
    let cut = sentences
        .iter()
        .find(|(_, o)| o.as_deref().is_some_and(|o| counts[o] >= 3))
        .map_or(text.len(), |(at, _)| *at);
    text[..cut].trim().to_string()
}

/// Whether a text goes round: some sentence, or the opening of one, comes back
/// three times or more.
pub(super) fn loops(text: &str) -> bool {
    let mut seen: HashMap<String, usize> = HashMap::new();
    for sentence in text.split(['.', '!', '?', ';']) {
        let words: Vec<String> = sentence
            .split_whitespace()
            .take(4)
            .map(|w| w.to_lowercase())
            .collect();
        if words.len() < 3 {
            continue;
        }
        let n = seen.entry(words.join(" ")).or_default();
        *n += 1;
        if *n >= 3 {
            return true;
        }
    }
    false
}
