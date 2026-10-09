//! What a refusal for a document that is not there says instead.
//!
//! **A near miss is told what it nearly was.** A Maker read
//! `layers/eras/the-stabilization.md` — the spelling most of the world's own
//! documents use for that period — and was told only that there was no such
//! document, so its next try was a guess. The era is `the-stabilisation.md`.
//! The folder is right there to look in, so the refusal names the document a
//! slip of a letter or two away, or, when nothing is that close, what the
//! folder holds.

use std::path::Path;

/// The most documents a refusal lists from a folder when nothing is close.
const LISTED: usize = 20;

/// The refusal for `key`, which resolved to `full` and is not there.
pub fn no_document(key: &str, full: &Path) -> String {
    let refused = format!("There is no document at {key}.");
    let Some(dir) = full.parent() else {
        return refused;
    };
    let folder = key.rsplit_once('/').map_or("", |(d, _)| d);
    let wanted = full
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let mut there: Vec<String> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.ends_with(".md") || n.ends_with(".yaml"))
                .collect()
        })
        .unwrap_or_default();
    there.sort();
    let path = |name: &str| match folder.is_empty() {
        true => name.to_string(),
        false => format!("{folder}/{name}"),
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
            "{refused} {folder} holds: {}.",
            there.iter().map(|n| path(n)).collect::<Vec<_>>().join(", ")
        ),
        (None, _) => format!("{refused} Use `file_list` on {folder} to see what it holds."),
    }
}

/// How many edits still count as the same name misspelled: a letter or two,
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
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    fn eras(name: &str, files: &[&str]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("npcd-missing-{name}-{}", std::process::id()));
        let dir = root.join("layers/eras");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&dir).unwrap();
        for f in files {
            fs::write(dir.join(f), "# an era\n").unwrap();
        }
        dir
    }

    /// **The near miss the pulse showed**: one letter of spelling away.
    #[test]
    fn a_misspelt_document_is_told_the_one_it_meant() {
        let dir = eras(
            "spelt",
            &["the-stabilisation.md", "the-spawn.md", "the-concord.md"],
        );
        assert_eq!(
            no_document(
                "layers/eras/the-stabilization.md",
                &dir.join("the-stabilization.md")
            ),
            "There is no document at layers/eras/the-stabilization.md. Did you mean \
             layers/eras/the-stabilisation.md?"
        );
    }

    /// Nothing close: the folder's documents are named, so the next try is not
    /// a guess.
    #[test]
    fn nothing_close_lists_the_folder() {
        let dir = eras("listed", &["the-spawn.md", "the-concord.md"]);
        assert_eq!(
            no_document(
                "layers/eras/the-long-winter.md",
                &dir.join("the-long-winter.md")
            ),
            "There is no document at layers/eras/the-long-winter.md. layers/eras holds: \
             layers/eras/the-concord.md, layers/eras/the-spawn.md."
        );
    }

    /// A folder that is not there says only that the document is not.
    #[test]
    fn no_folder_says_only_that() {
        let dir = eras("nofolder", &[]);
        assert_eq!(
            no_document("layers/nowhere/x.md", &dir.join("nowhere/x.md")),
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
