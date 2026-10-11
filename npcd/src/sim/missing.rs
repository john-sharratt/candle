//! What a refusal for a document that is not there says instead.
//!
//! **A near miss is told what it nearly was.** A Maker read
//! `layers/eras/the-stabilization.md` — the spelling most of the world's own
//! documents use for that period — and was told only that there was no such
//! document, so its next try was a guess. The era is `the-stabilisation.md`.
//! The folder is right there to look in, so the refusal names the document a
//! slip of a letter or two away, or, when nothing is that close, what the
//! folder holds.

use std::collections::BTreeSet;

/// The most documents a refusal lists from a folder when nothing is close.
const LISTED: usize = 20;

/// The refusal for `key`, which is not there, beside `folder` — what its
/// folder holds, documents by name and folders with a trailing `/`.
pub fn no_document(key: &str, folder: &BTreeSet<String>) -> String {
    let refused = format!("There is no document at {key}.");
    let (dir, wanted) = key.rsplit_once('/').unwrap_or(("", key));
    let wanted = wanted.to_lowercase();
    let there: Vec<&String> = folder
        .iter()
        .filter(|n| !n.ends_with('/') && (n.ends_with(".md") || n.ends_with(".yaml")))
        .collect();
    let path = |name: &str| match dir.is_empty() {
        true => name.to_string(),
        false => format!("{dir}/{name}"),
    };
    let nearest = there
        .iter()
        .map(|n| (distance(&wanted, &n.to_lowercase()), n))
        .filter(|(d, _)| *d <= close_enough(&wanted))
        .min_by_key(|(d, _)| *d);
    match (nearest, there.len()) {
        (Some((_, name)), _) => format!("{refused} Did you mean {}?", path(name)),
        (None, 0) => refused,
        (None, n) if n <= LISTED => format!(
            "{refused} {dir} holds: {}.",
            there.iter().map(|n| path(n)).collect::<Vec<_>>().join(", ")
        ),
        (None, _) => format!("{refused} Use `file_list` on {dir} to see what it holds."),
    }
}

/// How many edits still count as the same name misspelt: a letter or two,
/// and a little more for a long name.
fn close_enough(name: &str) -> usize {
    (name.chars().count() / 10).max(2)
}

/// Levenshtein distance, by characters.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    /// **The near miss the pulse showed**: one letter of spelling away.
    #[test]
    fn a_misspelt_document_is_told_the_one_it_meant() {
        let there = folder(&["the-stabilisation.md", "the-spawn.md", "the-concord.md"]);
        assert_eq!(
            no_document("layers/eras/the-stabilization.md", &there),
            "There is no document at layers/eras/the-stabilization.md. Did you mean \
             layers/eras/the-stabilisation.md?"
        );
    }

    /// Nothing close: the folder's documents are named, so the next try is not
    /// a guess; a folder within it is not.
    #[test]
    fn nothing_close_lists_the_folder() {
        let there = folder(&["the-spawn.md", "the-concord.md", "older/"]);
        assert_eq!(
            no_document("layers/eras/the-long-winter.md", &there),
            "There is no document at layers/eras/the-long-winter.md. layers/eras holds: \
             layers/eras/the-concord.md, layers/eras/the-spawn.md."
        );
    }

    /// A folder with nothing in it says only that the document is not.
    #[test]
    fn no_folder_says_only_that() {
        assert_eq!(
            no_document("layers/nowhere/x.md", &BTreeSet::new()),
            "There is no document at layers/nowhere/x.md."
        );
    }

    #[test]
    fn distance_counts_edits() {
        assert_eq!(distance("stabilization", "stabilisation"), 1);
        assert_eq!(distance("kitten", "sitting"), 3);
        assert_eq!(distance("", "abc"), 3);
    }
}
