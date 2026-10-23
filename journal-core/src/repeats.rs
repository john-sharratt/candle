//! What a journal entry says that the journal already says — taken out before
//! the entry is kept.
//!
//! **A journal that repeats itself is a loop its writer reads back.** The check
//! in [`crate::verify`] dedups typed claims within one entry and nothing else,
//! so "Did: I have written the draft" and "the record is absent" were kept entry
//! after entry in npcd, and every turn's prompt carried the same line four times
//! — the character read its own repetition as the state of the world. A claim or
//! intention that says nearly what a held entry (or an earlier line of the same
//! entry) says is dropped, and so is a new open item that says what one still
//! open says. "Nearly" is word overlap — the share of the two lines' distinct
//! words they have in common — between lines that name the same identifiers.
//!
//! **An identifier is what a line is about.** "Found MAX_INTEND at verify.rs:20"
//! and "Found MAX_CLAIMS at verify.rs:19" share nearly every word and say two
//! different things, so a line naming an identifier the other does not — a
//! constant, a path, a line number, a value — is never its repeat.

use std::collections::BTreeSet;

use crate::entry::{Entry, Item};

/// The word overlap at or above which two lines say the same thing.
const SAME: f32 = 0.7;

/// A line as it is compared.
struct Line {
    /// Lower case, letters and digits only, the short connective words left out.
    words: BTreeSet<String>,
    /// The identifiers it names, lower case.
    names: BTreeSet<String>,
}

/// Whether `token` names something rather than saying something: it holds a
/// digit, an underscore or a path separator, or a capital after its first
/// letter.
fn is_name(token: &str) -> bool {
    token
        .chars()
        .any(|c| c.is_ascii_digit() || c == '_' || c == '/')
        || token.chars().skip(1).any(char::is_uppercase)
        || token.trim_matches('.').contains('.')
}

fn line(s: &str) -> Line {
    let words = s
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 2)
        .map(str::to_lowercase)
        .collect();
    let names = s
        .split(|c: char| !(c.is_alphanumeric() || matches!(c, '_' | '/' | '.' | ':' | '-')))
        .map(|t| t.trim_matches(|c| matches!(c, '.' | ':')))
        .filter(|t| is_name(t))
        .map(str::to_lowercase)
        .collect();
    Line { words, names }
}

/// Whether `a` and `b` say the same thing.
fn same(a: &Line, b: &Line) -> bool {
    if a.names != b.names {
        return false;
    }
    let (a, b) = (&a.words, &b.words);
    if a.is_empty() || b.is_empty() {
        return a == b;
    }
    let shared = a.intersection(b).count() as f32;
    let all = a.union(b).count() as f32;
    shared / all >= SAME
}

/// `entry` without what `held` (the journal's entries) and `open` (its open
/// items) already say, and how many lines were dropped.
pub fn drop_repeats(mut entry: Entry, held: &[Entry], open: &[Item]) -> (Entry, usize) {
    let before = entry.claims.len() + entry.intend.len() + entry.opened.len();

    let mut said: Vec<Line> = held
        .iter()
        .flat_map(|e| e.claims.iter().map(|c| line(&c.text)))
        .collect();
    entry.claims.retain(|c| {
        let w = line(&c.text);
        let fresh = !said.iter().any(|s| same(s, &w));
        if fresh {
            said.push(w);
        }
        fresh
    });

    let mut meant: Vec<Line> = held
        .iter()
        .flat_map(|e| e.intend.iter().map(|i| line(i)))
        .collect();
    entry.intend.retain(|i| {
        let w = line(i);
        let fresh = !meant.iter().any(|s| same(s, &w));
        if fresh {
            meant.push(w);
        }
        fresh
    });

    // A restated item keeps its id and is the writer saying where it stands
    // now; only a new one that is already open is a repeat.
    let mut standing: Vec<Line> = open.iter().map(|i| line(&i.text)).collect();
    entry.opened.retain(|i| {
        if i.id != 0 {
            return true;
        }
        let w = line(&i.text);
        let fresh = !standing.iter().any(|s| same(s, &w));
        if fresh {
            standing.push(w);
        }
        fresh
    });

    let after = entry.claims.len() + entry.intend.len() + entry.opened.len();
    (entry, before - after)
}

/// Whether `entry` has anything left to keep.
pub fn says_anything(entry: &Entry) -> bool {
    !(entry.claims.is_empty()
        && entry.intend.is_empty()
        && entry.opened.is_empty()
        && entry.resolved.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::testing::claim;
    use crate::entry::Kind;

    fn entry(claims: &[&str], intend: &[&str], opened: &[&str]) -> Entry {
        Entry {
            id: 0,
            from_turn: 1,
            to_turn: 2,
            from_ms: 0,
            to_ms: 0,
            claims: claims.iter().map(|c| claim(c, Kind::Did)).collect(),
            intend: intend.iter().map(|s| s.to_string()).collect(),
            opened: opened
                .iter()
                .map(|t| Item {
                    id: 0,
                    text: t.to_string(),
                })
                .collect(),
            resolved: Vec::new(),
        }
    }

    /// **What the journal already says is not said again** — against the
    /// entries it holds, the items still open, and the entry's own earlier
    /// lines — and what is new stands.
    #[test]
    fn a_repeat_is_dropped_and_what_is_new_stands() {
        let held = [entry(
            &["I have written the full draft of The First Breath of the Anchor."],
            &["Report the draft at the command table."],
            &[],
        )];
        let open = [Item {
            id: 3,
            text: "The record for Ione Valtiere is absent".into(),
        }];
        let fresh = entry(
            &[
                "I have written the full draft of The First Breath of the Anchor",
                "The gate refused the draft: it is in the first person.",
                "The gate refused the draft — it is in the first person!",
            ],
            &[
                "Report the draft at the command table",
                "Rewrite it in the third person.",
            ],
            &[
                "the record for Ione Valtiere is absent.",
                "Find who wrote the Concord.",
            ],
        );
        let (kept, dropped) = drop_repeats(fresh, &held, &open);
        let claims: Vec<&str> = kept.claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            claims,
            ["The gate refused the draft: it is in the first person."]
        );
        assert_eq!(kept.intend, ["Rewrite it in the third person."]);
        assert_eq!(
            kept.opened
                .iter()
                .map(|i| i.text.as_str())
                .collect::<Vec<_>>(),
            ["Find who wrote the Concord."]
        );
        assert_eq!(dropped, 4);
        assert!(says_anything(&kept));
    }

    #[test]
    fn an_entry_of_nothing_but_repeats_says_nothing() {
        let held = [entry(
            &["Nothing has changed in the command room."],
            &[],
            &[],
        )];
        let (kept, dropped) = drop_repeats(
            entry(&["nothing has changed in the command room"], &[], &[]),
            &held,
            &[],
        );
        assert_eq!(dropped, 1);
        assert!(!says_anything(&kept));
    }

    /// The live failure: the second question differed from the first only in
    /// the constant it was about, and was dropped as a repeat of it.
    #[test]
    fn lines_about_different_identifiers_are_not_repeats() {
        let held = [entry(
            &["User asked me to find where MAX_CLAIMS is defined in journal-core with file_grep."],
            &[],
            &[],
        )];
        let fresh = entry(
            &[
                "User asked me to find where MAX_INTEND is defined in journal-core with file_grep.",
                "User asked me to find where MAX_CLAIMS is defined in journal-core with file_grep",
            ],
            &[],
            &[],
        );
        let (kept, dropped) = drop_repeats(fresh, &held, &[]);
        let claims: Vec<&str> = kept.claims.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(
            claims,
            ["User asked me to find where MAX_INTEND is defined in journal-core with file_grep."]
        );
        assert_eq!(dropped, 1);
    }

    #[test]
    fn a_line_number_or_a_path_is_an_identifier_and_a_capitalised_word_is_not() {
        let names = |s: &str| line(s).names.into_iter().collect::<Vec<_>>();
        assert_eq!(
            names("Read verify.rs:19, then journal-core/src/ask.rs. It said 12."),
            ["12", "journal-core/src/ask.rs", "verify.rs:19"]
        );
        assert_eq!(names("The gate refused the draft."), Vec::<String>::new());
        assert_eq!(names("ToolSpec is read by zend."), ["toolspec"]);
        let a = line("The parity test at scan.rs:40 failed.");
        let b = line("The parity test at scan.rs:41 failed.");
        assert!(!same(&a, &b));
        assert!(same(&a, &line("the parity test at scan.rs:40 failed")));
    }

    #[test]
    fn a_restated_item_is_never_a_repeat() {
        let open = [Item {
            id: 4,
            text: "Is the east door jammed?".into(),
        }];
        let mut e = entry(&[], &[], &[]);
        e.opened = vec![Item {
            id: 4,
            text: "Is the east door jammed?".into(),
        }];
        let (kept, dropped) = drop_repeats(e, &[], &open);
        assert_eq!((kept.opened.len(), dropped), (1, 0));
    }

    #[test]
    fn a_settling_only_entry_still_says_something() {
        let mut e = entry(&[], &[], &[]);
        e.resolved = vec![2];
        assert!(says_anything(&e));
    }
}
