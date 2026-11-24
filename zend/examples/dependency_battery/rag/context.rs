//! The retrieved files as R1's prompt carries them: best first, each block
//! headed by its repository-relative path, under a fixed budget of
//! [`BUDGET_TOKENS`] tokens counted at [`CHARS_PER_TOKEN`] characters each.
//! Headers and the blank line closing each block count against the budget;
//! the file that reaches it is cut there, on a character boundary, and no
//! file follows it.

use serde::{Deserialize, Serialize};

pub const BUDGET_TOKENS: usize = 24_000;
pub const CHARS_PER_TOKEN: usize = 4;

/// The budget in characters.
pub const BUDGET_CHARS: usize = BUDGET_TOKENS * CHARS_PER_TOKEN;

/// One file the prompt carried.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievedFile {
    pub path: String,
    /// Its place in the BM25 ranking, from 1.
    pub rank: u32,
    pub score: f64,
    /// The characters of it the prompt carried.
    pub chars: usize,
    /// Whether the budget cut it short.
    pub cut: bool,
    /// Whether it is one of the question's answer files.
    #[serde(default)]
    pub answer_file: bool,
}

/// The files block and what it holds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Packed {
    pub text: String,
    pub files: Vec<RetrievedFile>,
}

/// A block's header.
pub fn header(path: &str) -> String {
    format!("=== {path} ===\n")
}

/// `text`'s first `n` characters.
pub fn first_chars(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

/// `ranked` — `(rank, path, score, text)`, best first — packed under
/// `budget` characters.
pub fn pack(ranked: impl IntoIterator<Item = (u32, String, f64, String)>, budget: usize) -> Packed {
    let mut out = Packed::default();
    let mut left = budget;
    for (rank, path, score, text) in ranked {
        let head = header(&path);
        let fixed = head.chars().count() + 2;
        if left <= fixed {
            break;
        }
        let room = left - fixed;
        let body = first_chars(&text, room);
        let chars = body.chars().count();
        let cut = body.len() < text.len();
        out.text.push_str(&head);
        out.text.push_str(body);
        out.text.push_str("\n\n");
        left -= fixed + chars;
        out.files.push(RetrievedFile {
            path,
            rank,
            score,
            chars,
            cut,
            answer_file: false,
        });
        if cut {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(rank: u32, path: &str, text: &str) -> (u32, String, f64, String) {
        (rank, path.into(), 10.0 - rank as f64, text.into())
    }

    #[test]
    fn the_budget_is_twenty_four_thousand_tokens_of_four_characters() {
        assert_eq!(BUDGET_CHARS, 96_000);
    }

    /// **Files go in best first until the budget, and the one that reaches it
    /// is cut there**: `=== a ===\n` is 10 characters, so with the closing
    /// blank line a 5-character file takes 17; a budget of 30 leaves 13, of
    /// which `=== b ===\n` and its blank line take 12, so one character of b.
    #[test]
    fn files_are_packed_best_first_and_cut_at_the_budget() {
        let p = pack(
            [
                file(1, "a", "hello"),
                file(2, "b", "world"),
                file(3, "c", "x"),
            ],
            30,
        );
        assert_eq!(p.text, "=== a ===\nhello\n\n=== b ===\nw\n\n");
        assert_eq!(p.text.chars().count(), 30);
        assert_eq!(
            p.files,
            vec![
                RetrievedFile {
                    path: "a".into(),
                    rank: 1,
                    score: 9.0,
                    chars: 5,
                    cut: false,
                    answer_file: false,
                },
                RetrievedFile {
                    path: "b".into(),
                    rank: 2,
                    score: 8.0,
                    chars: 1,
                    cut: true,
                    answer_file: false,
                },
            ]
        );
    }

    /// **A cut never splits a character**: `é` and `—` are counted as one
    /// character each, whatever their bytes.
    #[test]
    fn a_cut_lands_on_a_character_boundary() {
        assert_eq!(first_chars("é—x", 2), "é—");
        assert_eq!(first_chars("ab", 5), "ab");
        let p = pack([file(1, "a", "é—xyz")], 14);
        assert_eq!(p.text, "=== a ===\né—\n\n");
        assert_eq!((p.files[0].chars, p.files[0].cut), (2, true));
    }

    #[test]
    fn no_room_for_a_header_carries_nothing() {
        let p = pack([file(1, "a", "hello")], 12);
        assert_eq!(p, Packed::default());
    }
}
