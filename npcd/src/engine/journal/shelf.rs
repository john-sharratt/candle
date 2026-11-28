//! The journal a character's `journal_read` and `journal_grep` reach: its own,
//! `"self"` — its entries, and the turns each was written from
//! ([`super::text`]). A character's journal has none above it.

use std::sync::Arc;

use host::journal_text::JournalText;
use journal_core::journal::Journal;
use journal_core::transcript::{Hits, Miss, Page};
use zend_tools::tools::journal::{JournalBook, JournalShelf, SELF};

/// What a journal with nothing in it lists.
const NO_ENTRIES: &str = "No entries yet.";

/// `journal`'s entries and open items, a line each, as `journal_read` lists
/// them.
pub fn lines_of(journal: &Journal) -> Vec<String> {
    journal
        .entries()
        .map(|e| format!("#{} {}", e.id, e.summary))
        .chain(
            journal
                .open()
                .iter()
                .map(|item| format!("Still open #{}: {}", item.id, item.text)),
        )
        .collect()
}

/// A character's own journal: its lines, and the turns behind them.
pub struct Own {
    lines: Vec<String>,
    text: JournalText,
}

impl Own {
    pub fn new(lines: Vec<String>, text: JournalText) -> Self {
        Self { lines, text }
    }
}

impl JournalBook for Own {
    fn entries(&self) -> String {
        match self.lines.is_empty() {
            true => NO_ENTRIES.to_string(),
            false => self.lines.join("\n"),
        }
    }

    fn read(&self, entry: u64, from: usize, lines: usize) -> Result<Page, Miss> {
        self.text.read(entry, from, lines)
    }

    fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss> {
        self.text.grep(pattern, entry, max)
    }
}

/// A character's shelf: its own journal, and no other.
pub struct Shelf {
    own: Arc<dyn JournalBook>,
}

impl Shelf {
    pub fn new(own: Own) -> Self {
        Self { own: Arc::new(own) }
    }
}

impl JournalShelf for Shelf {
    fn open(&self, name: &str) -> Result<Arc<dyn JournalBook>, String> {
        match name == SELF {
            true => Ok(Arc::clone(&self.own)),
            false => Err(format!(
                "`{name}` names no journal of yours: the one you read is \"self\""
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use journal_core::transcript::{Filed, Part};
    use tempfile::TempDir;

    use super::*;
    use crate::engine::journal::text;
    use crate::npcs::Npcs;

    /// **"self" is the character's own journal, and nothing else is**: its
    /// lines list, its turns read and search.
    #[test]
    fn self_is_the_characters_own_journal() {
        let dir = TempDir::new().unwrap();
        let shared = Npcs::load(dir.path()).unwrap().substrate();
        let turns = text::of(shared.clone(), 7);
        turns
            .file(&Filed {
                entry: 1,
                parts: vec![Part {
                    id: 4,
                    at_ms: 0,
                    text: "Wick reported the door sensor fixed.".into(),
                }],
            })
            .unwrap();
        let shelf = Shelf::new(Own::new(vec!["#1 Heard the door is fixed.".into()], turns));
        let own = shelf.open("self").unwrap();
        assert_eq!(own.entries(), "#1 Heard the door is fixed.");
        assert!(own
            .read(1, 1, 10)
            .unwrap()
            .text
            .ends_with("Wick reported the door sensor fixed."));
        assert_eq!(own.grep("SENSOR", None, 10).unwrap().total, 1);
        assert_eq!(
            shelf.open("T:t").err().unwrap(),
            "`T:t` names no journal of yours: the one you read is \"self\""
        );
        let empty = Shelf::new(Own::new(Vec::new(), text::of(shared, 8)));
        assert_eq!(empty.open("self").unwrap().entries(), "No entries yet.");
    }

    #[test]
    fn a_journals_lines_are_its_entries_then_what_is_open() {
        assert!(lines_of(&Journal::new()).is_empty());
    }
}
