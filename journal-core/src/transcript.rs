//! The text of the turns behind each entry, kept for the writer to read back.
//!
//! The journal shows one line per entry; what the entry was written from is
//! kept beside it, whole, so the writer can page through an entry's turns
//! ([`Book::read`]) or search every entry's ([`Book::grep`]) the way it reads a
//! file. Each stretch a draft covers is filed under the entry it went into —
//! the entry it wrote, or the latest entry when the stretch was more of the same
//! ([`Filed`]) — as an object of its own, keyed so that one owner's, or one
//! entry's, are found by prefix ([`key`]). An object is only ever written once:
//! extending an entry adds a stretch beside the ones it holds and rewrites none.
//!
//! The text lives as long as its entry does. The product stores it, and removes
//! it with the entry, with everything of the conversation when the conversation
//! goes, and on its own when the conversation is archived and nothing can read
//! it again.

use serde::{Deserialize, Serialize};

use crate::span::Span;
use crate::style::Style;

/// What every key of the kept text starts with.
pub const PREFIX: &str = "journal/turns/";

/// The most lines one read returns.
pub const MAX_LINES: usize = 400;

/// The most hits one search returns.
pub const MAX_HITS: usize = 50;

/// A hit's line is cut to this many characters.
const HIT_CHARS: usize = 300;

/// One turn, whole, as it was when its stretch was drafted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Part {
    /// The number the stretch numbered it by, which the entry's claims cite.
    pub id: u64,
    pub at_ms: u64,
    /// The turn's whole text, speaker included.
    pub text: String,
}

/// A stretch's turns, filed under the entry they went into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Filed {
    pub entry: u64,
    pub parts: Vec<Part>,
}

impl Filed {
    /// `span`'s turns, filed under `entry`.
    pub fn of(entry: u64, span: &Span) -> Self {
        Self {
            entry,
            parts: span
                .turns
                .iter()
                .map(|t| Part {
                    id: t.id,
                    at_ms: t.at_ms,
                    text: t.full.clone(),
                })
                .collect(),
        }
    }

    /// The key it is kept under for `owner`: after every stretch filed under
    /// the same entry before it, since a stretch's first part comes after
    /// theirs.
    pub fn key(&self, owner: &str) -> String {
        let first = self.parts.first().map_or(0, |p| p.id);
        key(owner, self.entry, first)
    }
}

/// Every stretch `owner` has kept starts with this.
pub fn owner_prefix(owner: &str) -> String {
    format!("{PREFIX}{owner}/")
}

/// Every stretch filed under `entry` starts with this.
pub fn entry_prefix(owner: &str, entry: u64) -> String {
    format!("{}{entry:010}/", owner_prefix(owner))
}

/// The key of the stretch filed under `entry` whose first part is `first`.
/// Zero-padded, so key order is entry order and then stretch order.
pub fn key(owner: &str, entry: u64, first: u64) -> String {
    format!("{}{first:020}", entry_prefix(owner, entry))
}

/// The entry a key of `owner`'s files its stretch under.
pub fn entry_of(owner: &str, key: &str) -> Option<u64> {
    let rest = key.strip_prefix(&owner_prefix(owner))?;
    rest.split('/').next()?.parse().ok()
}

/// Why a read or a search found nothing to read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Miss {
    /// No entry holds any kept text: the journal is empty, or what it was
    /// written from is gone.
    Nothing,
    /// The entry holds no kept text: there is no such entry, or a rollup took
    /// its text with it.
    Entry(u64),
}

impl Miss {
    /// The error code a tool reports it with.
    pub fn code(&self) -> &'static str {
        match self {
            Miss::Nothing => "no_journal_text",
            Miss::Entry(_) => "no_entry_text",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            Miss::Nothing => "your journal holds no text of the turns behind its entries".into(),
            Miss::Entry(id) => format!(
                "no text is kept for entry {id}: there is no such entry, or its turns were \
                 folded into a summary"
            ),
        }
    }
}

/// One page of an entry's turns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Page {
    pub entry: u64,
    /// The first line shown, counted from 1.
    pub from: usize,
    /// The last line shown.
    pub to: usize,
    /// Lines the entry's turns run to.
    pub total: usize,
    pub text: String,
}

/// One line a search matched.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Hit {
    pub entry: u64,
    /// The turn the line is in.
    pub part: u64,
    /// Its line among the entry's, as [`Book::read`] counts them.
    pub line: usize,
    pub text: String,
}

/// What a search matched, the first [`MAX_HITS`] of `total`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Hits {
    pub hits: Vec<Hit>,
    pub total: usize,
}

/// One owner's kept text, entry by entry, oldest first.
pub struct Book<'a> {
    entries: Vec<(u64, Vec<Part>)>,
    style: &'a Style,
}

/// One rendered line and the turn it belongs to.
struct Line {
    part: u64,
    text: String,
}

impl<'a> Book<'a> {
    /// The book of `stretches` — `(entry, parts)` in key order, as the store
    /// lists them — dated by `style`.
    pub fn new(stretches: impl IntoIterator<Item = (u64, Vec<Part>)>, style: &'a Style) -> Self {
        let mut entries: Vec<(u64, Vec<Part>)> = Vec::new();
        for (entry, parts) in stretches {
            match entries.last_mut() {
                Some((last, held)) if *last == entry => held.extend(parts),
                _ => entries.push((entry, parts)),
            }
        }
        Self { entries, style }
    }

    /// Each turn of `parts` as lines: its number and time, then its text as it
    /// runs.
    fn lines(&self, parts: &[Part]) -> Vec<Line> {
        let mut out = Vec::new();
        for p in parts {
            let mut rows = p.text.lines();
            let first = rows.next().unwrap_or("");
            out.push(Line {
                part: p.id,
                text: format!("#{} {} {first}", p.id, (self.style.clock)(p.at_ms)),
            });
            out.extend(rows.map(|r| Line {
                part: p.id,
                text: r.to_string(),
            }));
        }
        out
    }

    /// `max` lines of `entry`'s turns from line `from` (counted from 1), at
    /// most [`MAX_LINES`].
    pub fn read(&self, entry: u64, from: usize, max: usize) -> Result<Page, Miss> {
        if self.entries.is_empty() {
            return Err(Miss::Nothing);
        }
        let (_, parts) = self
            .entries
            .iter()
            .find(|(e, _)| *e == entry)
            .ok_or(Miss::Entry(entry))?;
        let lines = self.lines(parts);
        let total = lines.len();
        let start = from.max(1).min(total.max(1));
        let take = max.clamp(1, MAX_LINES);
        let shown: Vec<&str> = lines
            .iter()
            .skip(start - 1)
            .take(take)
            .map(|l| l.text.as_str())
            .collect();
        Ok(Page {
            entry,
            from: start,
            to: start + shown.len().saturating_sub(1),
            total,
            text: shown.join("\n"),
        })
    }

    /// The lines that hold `pattern`, ignoring case — in `entry`'s turns, or
    /// in every entry's — the first `max` of them, at most [`MAX_HITS`].
    pub fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss> {
        if self.entries.is_empty() {
            return Err(Miss::Nothing);
        }
        if let Some(e) = entry {
            if !self.entries.iter().any(|(held, _)| *held == e) {
                return Err(Miss::Entry(e));
            }
        }
        let needle = pattern.to_lowercase();
        let take = max.clamp(1, MAX_HITS);
        let mut hits = Vec::new();
        let mut total = 0usize;
        for (e, parts) in &self.entries {
            if entry.is_some_and(|want| want != *e) {
                continue;
            }
            for (i, line) in self.lines(parts).into_iter().enumerate() {
                if !line.text.to_lowercase().contains(&needle) {
                    continue;
                }
                total += 1;
                if hits.len() < take {
                    hits.push(Hit {
                        entry: *e,
                        part: line.part,
                        line: i + 1,
                        text: cut(&line.text),
                    });
                }
            }
        }
        Ok(Hits { hits, total })
    }
}

/// `text` cut to [`HIT_CHARS`] characters, marked with `…` when it was.
fn cut(text: &str) -> String {
    if text.chars().count() <= HIT_CHARS {
        return text.to_string();
    }
    let kept: String = text.chars().take(HIT_CHARS).collect();
    format!("{kept}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::Kind;
    use crate::span::testing::turn;
    use crate::span::SpanTurn;
    use crate::style::testing::PLAIN;

    fn part(id: u64, text: &str) -> Part {
        Part {
            id,
            at_ms: id * 10,
            text: text.into(),
        }
    }

    fn book() -> Book<'static> {
        Book::new(
            [
                (
                    1,
                    vec![
                        part(1, "User: fix the scan"),
                        part(2, "You: reading scan.rs"),
                    ],
                ),
                (
                    2,
                    vec![part(3, "Tool results: fn scan() {\n    loop {}\n}")],
                ),
                (2, vec![part(5, "You: the loop never yields")]),
            ],
            &PLAIN,
        )
    }

    #[test]
    fn keys_order_by_entry_then_by_stretch() {
        assert_eq!(
            key("7", 3, 41),
            "journal/turns/7/0000000003/00000000000000000041"
        );
        assert!(key("7", 3, 41) < key("7", 3, 100));
        assert!(key("7", 3, 900) < key("7", 12, 1));
        assert!(key("7", 3, 41).starts_with(&entry_prefix("7", 3)));
        assert!(key("7", 3, 41).starts_with(&owner_prefix("7")));
        assert!(!key("71", 3, 41).starts_with(&owner_prefix("7")));
        assert_eq!(entry_of("7", &key("7", 12, 1)), Some(12));
        assert_eq!(entry_of("8", &key("7", 12, 1)), None);
    }

    #[test]
    fn a_stretch_is_filed_whole_under_its_entry_and_keyed_by_its_first_turn() {
        let span = Span::of(vec![
            SpanTurn {
                full: "User: a long question\nover two lines".into(),
                ..turn(
                    9,
                    90,
                    Some(Kind::Heard),
                    "User: a long question over two lines",
                )
            },
            turn(10, 100, Some(Kind::Did), "You: an answer"),
        ])
        .unwrap();
        let filed = Filed::of(4, &span);
        assert_eq!(filed.entry, 4);
        assert_eq!(
            filed.parts,
            vec![
                Part {
                    id: 9,
                    at_ms: 90,
                    text: "User: a long question\nover two lines".into()
                },
                Part {
                    id: 10,
                    at_ms: 100,
                    text: "You: an answer".into()
                },
            ]
        );
        assert_eq!(filed.key("c"), key("c", 4, 9));
    }

    #[test]
    fn a_read_pages_an_entrys_turns_line_by_line_with_each_turns_number_and_time() {
        let page = book().read(2, 1, 400).unwrap();
        assert_eq!(
            page,
            Page {
                entry: 2,
                from: 1,
                to: 4,
                total: 4,
                text:
                    "#3 t30 Tool results: fn scan() {\n    loop {}\n}\n#5 t50 You: the loop never \
                       yields"
                        .into(),
            }
        );
        let middle = book().read(2, 2, 2).unwrap();
        assert_eq!((middle.from, middle.to, middle.total), (2, 3, 4));
        assert_eq!(middle.text, "    loop {}\n}");
    }

    #[test]
    fn a_read_past_the_end_shows_the_last_line() {
        let page = book().read(1, 99, 10).unwrap();
        assert_eq!((page.from, page.to), (2, 2));
        assert_eq!(page.text, "#2 t20 You: reading scan.rs");
    }

    #[test]
    fn a_search_finds_lines_in_any_entry_ignoring_case() {
        let hits = book().grep("LOOP", None, 10).unwrap();
        assert_eq!(hits.total, 2);
        assert_eq!(
            hits.hits,
            vec![
                Hit {
                    entry: 2,
                    part: 3,
                    line: 2,
                    text: "    loop {}".into()
                },
                Hit {
                    entry: 2,
                    part: 5,
                    line: 4,
                    text: "#5 t50 You: the loop never yields".into()
                },
            ]
        );
        let in_one = book().grep("scan", Some(1), 10).unwrap();
        assert_eq!(in_one.total, 2);
        assert!(in_one.hits.iter().all(|h| h.entry == 1));
    }

    #[test]
    fn a_search_counts_every_hit_and_returns_the_first_few() {
        let hits = book().grep("s", None, 1).unwrap();
        assert_eq!(hits.hits.len(), 1);
        assert!(hits.total > 1);
    }

    #[test]
    fn a_long_hit_is_cut() {
        let long = "x".repeat(HIT_CHARS + 50);
        let b = Book::new([(1, vec![part(1, &long)])], &PLAIN);
        let hit = &b.grep("x", None, 5).unwrap().hits[0];
        assert!(hit.text.ends_with('…'));
        assert_eq!(hit.text.chars().count(), HIT_CHARS + 1);
    }

    #[test]
    fn an_empty_book_or_an_unknown_entry_says_which() {
        let empty = Book::new(Vec::new(), &PLAIN);
        assert_eq!(empty.read(1, 1, 10), Err(Miss::Nothing));
        assert_eq!(empty.grep("x", None, 10), Err(Miss::Nothing));
        assert_eq!(book().read(9, 1, 10), Err(Miss::Entry(9)));
        assert_eq!(book().grep("x", Some(9), 10), Err(Miss::Entry(9)));
        assert_eq!(Miss::Entry(9).code(), "no_entry_text");
    }
}
