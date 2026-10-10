//! A draft that copies its own brief: the brief's "what happens" set down
//! word for word instead of told as a scene.
//!
//! **A brief says what happens, not how it is told.** A story stood in the
//! record whose turning paragraph was its brief's sixty words with the tense
//! changed — "He took the first bite, juice running down his chin. He swallowed
//! hard…" — and every reader passed it, because every reader was shown the
//! brief beside it as what it was to tell. A run of the brief's own words that
//! long is a summary pasted in, not a scene.

/// The check's name, as a workflow step lists it.
pub const BRIEF_COPIED: &str = "brief-copied";

/// The fewest words in a row, taken from the brief, that count as copying it.
/// Shorter runs are names, places and the phrases any telling of the same
/// event shares.
pub const RUN_WORDS: usize = 10;

/// A word as it is compared: lowercase, without the punctuation around it.
fn plain(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

/// The longest run of `brief`'s words that `text` holds in the same order,
/// as the brief writes it. Empty when they share no word.
fn longest_run(text: &str, brief: &str) -> String {
    let raw: Vec<&str> = brief.split_whitespace().collect();
    let b: Vec<String> = raw.iter().map(|w| plain(w)).collect();
    let t: Vec<String> = text.split_whitespace().map(plain).collect();
    // Longest common run of words, by the usual table kept one row at a time.
    let mut prev = vec![0usize; t.len() + 1];
    let (mut best, mut end) = (0, 0);
    for (i, bw) in b.iter().enumerate() {
        let mut row = vec![0usize; t.len() + 1];
        for (j, tw) in t.iter().enumerate() {
            if !bw.is_empty() && bw == tw {
                row[j + 1] = prev[j] + 1;
                if row[j + 1] > best {
                    best = row[j + 1];
                    end = i + 1;
                }
            }
        }
        prev = row;
    }
    raw[end - best..end].join(" ")
}

/// Why `text` does not stand, when it copies [`RUN_WORDS`] or more of
/// `brief`'s words in a row: the run, quoted, and what to do instead. `None`
/// when it tells the brief in its own words.
pub fn copied(text: &str, brief: &str) -> Option<String> {
    let run = longest_run(text, brief);
    (run.split_whitespace().count() >= RUN_WORDS).then(|| {
        format!(
            "\"{run}\" is your brief's own words, copied into the piece. A brief says what \
             happens, not how it is told: tell that moment as a scene in your own words — what is \
             done and said in it, moment by moment, as somebody there would see it."
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BRIEF: &str = "He takes the first bite, juice running down his chin. He swallows hard, \
                         focusing on the texture, the sweetness, the shock of the sourness.";

    /// **The brief set down with its tense changed is still the brief** — the
    /// comparison ignores case and punctuation, not the words.
    #[test]
    fn a_draft_that_copies_its_brief_is_refused_with_the_run_quoted() {
        let draft = "The line moved. He takes the first bite, juice running down his chin. He \
                     swallows hard, focusing on the texture! Then he stepped forward.";
        assert_eq!(
            copied(draft, BRIEF).unwrap(),
            "\"He takes the first bite, juice running down his chin. He swallows hard, focusing \
             on the texture,\" is your brief's own words, copied into the piece. A brief says \
             what happens, not how it is told: tell that moment as a scene in your own words — \
             what is done and said in it, moment by moment, as somebody there would see it."
        );
    }

    /// A telling in its own words, sharing only names and short phrases, stands.
    #[test]
    fn a_draft_in_its_own_words_stands() {
        let draft = "He bit into the apple. Juice ran down his chin and he let it. The sweetness \
                     came first, then the sour shock of it, and he chewed slowly.";
        assert_eq!(copied(draft, BRIEF), None);
        assert_eq!(copied("", BRIEF), None);
        assert_eq!(copied(draft, ""), None);
    }

    #[test]
    fn the_longest_shared_run_is_found_as_the_brief_writes_it() {
        assert_eq!(longest_run("a B c d e", "x b C d y"), "b C d");
        assert_eq!(longest_run("nothing here", "else entirely"), "");
    }
}
