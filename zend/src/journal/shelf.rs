//! The journals a dialogue's `journal_read` and `journal_grep` reach by name
//! (`docs/claude_zend_gateway.md` §5.3): `"self"`, its own; and for a
//! conversation owned through `/mcp`, each journal above it — its thread's and
//! its project's notes, and the journal of each worker above a part — by its
//! full path or its bare name, resolved upward, nearest first.
//!
//! A drafted journal — a conversation's own, or a worker's above it — lists
//! its entries and reads the turns behind them. A thread's or project's
//! journal is Claude's notes, records with no turns behind them, so an entry
//! of it reads as the note itself.

use std::sync::{Arc, OnceLock};

use journal_core::transcript::{Hit, Hits, Miss, Page};
use zend_tools::tools::journal::{JournalBook, JournalShelf, SELF};

use crate::journal::text::JournalText;

/// What a journal with nothing in it lists.
const NO_ENTRIES: &str = "No entries yet.";

/// A journal drafted from a conversation's turns: its entries' lines, and the
/// turns behind them.
pub struct Drafted {
    lines: Vec<String>,
    text: JournalText,
}

impl Drafted {
    pub fn new(lines: Vec<String>, text: JournalText) -> Self {
        Self { lines, text }
    }
}

impl JournalBook for Drafted {
    fn entries(&self) -> String {
        listed(&self.lines)
    }

    fn read(&self, entry: u64, from: usize, lines: usize) -> Result<Page, Miss> {
        self.text.read(entry, from, lines)
    }

    fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss> {
        self.text.grep(pattern, entry, max)
    }
}

/// A thread's or project's journal: Claude's notes, one numbered line each.
pub struct Notes {
    lines: Vec<String>,
}

impl Notes {
    /// The notes whose lines, oldest first, are `lines` — entry `n` is line
    /// `n`, counted from 1.
    pub fn new(lines: Vec<String>) -> Self {
        Self { lines }
    }
}

impl JournalBook for Notes {
    fn entries(&self) -> String {
        listed(&self.lines)
    }

    /// A note is its whole record: its one line, on a page of one.
    fn read(&self, entry: u64, _from: usize, _lines: usize) -> Result<Page, Miss> {
        let text = entry
            .checked_sub(1)
            .and_then(|i| self.lines.get(i as usize))
            .ok_or(Miss::Entry(entry))?;
        Ok(Page {
            entry,
            from: 1,
            to: 1,
            total: 1,
            text: text.clone(),
        })
    }

    fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss> {
        let needle = pattern.to_lowercase();
        let found: Vec<Hit> = self
            .lines
            .iter()
            .zip(1u64..)
            .filter(|(_, n)| entry.is_none_or(|e| e == *n))
            .filter(|(line, _)| line.to_lowercase().contains(&needle))
            .map(|(line, n)| Hit {
                entry: n,
                part: 0,
                line: 1,
                text: line.clone(),
            })
            .collect();
        let total = found.len();
        Ok(Hits {
            hits: found.into_iter().take(max).collect(),
            total,
        })
    }
}

/// `lines` as a journal lists them.
fn listed(lines: &[String]) -> String {
    if lines.is_empty() {
        NO_ENTRIES.to_string()
    } else {
        lines.join("\n")
    }
}

/// One journal above a conversation: its full path, the bare name it also
/// answers to, and its book.
pub struct Above {
    pub path: String,
    pub name: String,
    pub book: Arc<dyn JournalBook>,
}

/// A conversation's journals: its own, and those above it nearest first.
pub struct Shelf {
    own: Arc<dyn JournalBook>,
    above: Vec<Above>,
}

impl Shelf {
    pub fn new(own: Arc<dyn JournalBook>, above: Vec<Above>) -> Self {
        Self { own, above }
    }
}

/// A shelf built the first time a journal tool opens it, so a round that
/// calls none reads no journal for it.
pub struct Lazy<F> {
    build: F,
    built: OnceLock<Shelf>,
}

impl<F: Fn() -> Shelf + Send + Sync> Lazy<F> {
    pub fn new(build: F) -> Self {
        Self {
            build,
            built: OnceLock::new(),
        }
    }
}

impl<F: Fn() -> Shelf + Send + Sync> JournalShelf for Lazy<F> {
    fn open(&self, name: &str) -> Result<Arc<dyn JournalBook>, String> {
        self.built.get_or_init(&self.build).open(name)
    }
}

impl JournalShelf for Shelf {
    fn open(&self, name: &str) -> Result<Arc<dyn JournalBook>, String> {
        if name == SELF {
            return Ok(Arc::clone(&self.own));
        }
        if let Some(a) = self.above.iter().find(|a| a.path == name || a.name == name) {
            return Ok(Arc::clone(&a.book));
        }
        if self.above.is_empty() {
            return Err(format!(
                "`{name}` names no journal: none is above this conversation, so the one it \
                 reads is \"self\""
            ));
        }
        let paths: Vec<&str> = self.above.iter().map(|a| a.path.as_str()).collect();
        Err(format!(
            "`{name}` is not a journal above this conversation; it reads \"self\", {}",
            paths.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use candle_conversation::persistence::{Admit, SharedSubstrate};
    use candle_conversation::projection::TimelineId;
    use journal_core::transcript::{Filed, Part};

    use super::*;

    fn notes(lines: &[&str]) -> Arc<dyn JournalBook> {
        Arc::new(Notes::new(lines.iter().map(|l| l.to_string()).collect()))
    }

    fn above(path: &str, name: &str, book: Arc<dyn JournalBook>) -> Above {
        Above {
            path: path.into(),
            name: name.into(),
            book,
        }
    }

    /// A part's shelf: its own journal, its worker's `T:t/w` and its thread's
    /// `T:t`, nearest first.
    fn shelf() -> Shelf {
        Shelf::new(
            notes(&["#1 own"]),
            vec![
                above("T:t/w", "w", notes(&["#1 worker"])),
                above("T:t", "t", notes(&["#1 Claude said: keep it small"])),
                above("P:candle", "candle", notes(&["#1 never push main"])),
            ],
        )
    }

    /// **A lazy shelf is built once, on its first open**, and never for a
    /// round that opens none.
    #[test]
    fn a_lazy_shelf_is_built_on_its_first_open_only() {
        let builds = AtomicUsize::new(0);
        let lazy = Lazy::new(|| {
            builds.fetch_add(1, Ordering::Relaxed);
            shelf()
        });
        assert_eq!(builds.load(Ordering::Relaxed), 0);
        assert_eq!(
            lazy.open("t").unwrap().entries(),
            "#1 Claude said: keep it small"
        );
        assert_eq!(lazy.open("self").unwrap().entries(), "#1 own");
        assert_eq!(builds.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn self_is_the_conversations_own() {
        assert_eq!(shelf().open("self").unwrap().entries(), "#1 own");
    }

    /// **A journal above answers to its full path and its bare name.**
    #[test]
    fn a_journal_above_opens_by_path_or_bare_name() {
        let s = shelf();
        assert_eq!(s.open("T:t/w").unwrap().entries(), "#1 worker");
        assert_eq!(s.open("w").unwrap().entries(), "#1 worker");
        assert_eq!(
            s.open("t").unwrap().entries(),
            "#1 Claude said: keep it small"
        );
        assert_eq!(s.open("candle").unwrap().entries(), "#1 never push main");
    }

    /// **A bare name resolves upward, nearest first**: a worker sharing its
    /// project's name is the one found.
    #[test]
    fn a_bare_name_takes_the_nearest_journal_above() {
        let s = Shelf::new(
            notes(&[]),
            vec![
                above("T:t/candle", "candle", notes(&["#1 the worker"])),
                above("P:candle", "candle", notes(&["#1 the project"])),
            ],
        );
        assert_eq!(s.open("candle").unwrap().entries(), "#1 the worker");
        assert_eq!(s.open("P:candle").unwrap().entries(), "#1 the project");
    }

    #[test]
    fn a_name_not_above_is_refused_with_what_is() {
        assert_eq!(
            shelf().open("T:other").err().unwrap(),
            "`T:other` is not a journal above this conversation; it reads \"self\", T:t/w, \
             T:t, P:candle"
        );
        let alone = Shelf::new(notes(&[]), Vec::new());
        assert_eq!(
            alone.open("T:t").err().unwrap(),
            "`T:t` names no journal: none is above this conversation, so the one it reads is \
             \"self\""
        );
    }

    #[test]
    fn notes_read_and_search_note_by_note() {
        let n = Notes::new(vec![
            "#1 Claude said: verify the expiry".into(),
            "#2 Decided: parts cite their files".into(),
        ]);
        assert_eq!(
            n.read(2, 1, 200).unwrap(),
            Page {
                entry: 2,
                from: 1,
                to: 1,
                total: 1,
                text: "#2 Decided: parts cite their files".into(),
            }
        );
        assert_eq!(n.read(3, 1, 200), Err(Miss::Entry(3)));
        assert_eq!(n.read(0, 1, 200), Err(Miss::Entry(0)));
        let hits = n.grep("DECIDED", None, 20).unwrap();
        assert_eq!(hits.total, 1);
        assert_eq!(
            hits.hits[0],
            Hit {
                entry: 2,
                part: 0,
                line: 1,
                text: "#2 Decided: parts cite their files".into(),
            }
        );
        assert_eq!(n.grep("claude", Some(2), 20).unwrap().total, 0);
        assert_eq!(n.grep("#", None, 1).unwrap().hits.len(), 1);
        assert_eq!(
            n.entries(),
            "#1 Claude said: verify the expiry\n#2 Decided: parts cite their files"
        );
        assert_eq!(Notes::new(Vec::new()).entries(), "No entries yet.");
    }

    #[test]
    fn a_drafted_journal_lists_its_lines_and_reads_its_turns() {
        let dir = tempfile::tempdir().unwrap();
        let shared = SharedSubstrate::open_in(dir.path(), Admit::Unbound).unwrap();
        let text = JournalText::of(shared, TimelineId::from_raw(7).unwrap());
        text.file(&Filed {
            entry: 1,
            parts: vec![Part {
                id: 1,
                at_ms: 1_792_713_600_000,
                text: "You: read scan.rs".into(),
            }],
        })
        .unwrap();
        let d = Drafted::new(vec!["#1 Read scan.rs.".into()], text);
        assert_eq!(d.entries(), "#1 Read scan.rs.");
        assert_eq!(d.read(1, 1, 10).unwrap().text, "#1 00:00 You: read scan.rs");
        assert_eq!(d.grep("SCAN", None, 10).unwrap().total, 1);
    }
}
