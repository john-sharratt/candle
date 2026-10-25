//! Integration tests for `journal_read` and `journal_grep`.
//!
//! What a page or a search holds is `journal_core::transcript`'s, tested there.
//! These pin the tools' half: the journal named is the one opened, the
//! arguments reach its book as the call wrote them, with their defaults and
//! bounds, and each way of finding nothing comes back under its own code.

mod harness;

use std::sync::{Arc, Mutex};

use harness::{expect_error, expect_success, granted, invoke, invoke_with_ctx, replay};
use journal_core::transcript::{Hit, Hits, Miss, Page};
use serde_json::json;
use zend_tools::tools::journal::{JournalBook, JournalShelf, SELF};
use zend_tools::{Replay, ToolContext};

/// A book holding entry 3 alone, recording every request it is asked.
#[derive(Default)]
struct Fake {
    reads: Mutex<Vec<(u64, usize, usize)>>,
    greps: Mutex<Vec<(String, Option<u64>, usize)>>,
}

/// A shelf holding the caller's own journal and one ancestor's, `T:t`,
/// recording each name it is asked to open.
struct Shelf {
    own: Arc<Fake>,
    ancestor: Arc<Fake>,
    opened: Mutex<Vec<String>>,
}

impl JournalShelf for Shelf {
    fn open(&self, name: &str) -> Result<Arc<dyn JournalBook>, String> {
        self.opened.lock().unwrap().push(name.to_string());
        match name {
            SELF => Ok(Arc::clone(&self.own) as Arc<dyn JournalBook>),
            "T:t" => Ok(Arc::clone(&self.ancestor) as Arc<dyn JournalBook>),
            other => Err(format!(
                "`{other}` is not a journal above this conversation"
            )),
        }
    }
}

impl JournalBook for Fake {
    fn entries(&self) -> String {
        "Entry 3 — Fixed the scan.".into()
    }

    fn read(&self, entry: u64, from: usize, lines: usize) -> Result<Page, Miss> {
        self.reads.lock().unwrap().push((entry, from, lines));
        if entry != 3 {
            return Err(Miss::Entry(entry));
        }
        Ok(Page {
            entry,
            from,
            to: from,
            total: 9,
            text: "#7 14:20 User: fix the scan".into(),
        })
    }

    fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss> {
        self.greps
            .lock()
            .unwrap()
            .push((pattern.to_string(), entry, max));
        Ok(Hits {
            hits: vec![Hit {
                entry: 3,
                part: 7,
                line: 1,
                text: "#7 14:20 User: fix the scan".into(),
            }],
            total: 1,
        })
    }
}

/// A context whose own journal is `book`, with an empty ancestor `T:t`.
fn with(book: &Arc<Fake>) -> ToolContext {
    shelved(book, &Arc::new(Fake::default())).0
}

/// A context whose own journal is `own` and whose ancestor `T:t` is
/// `ancestor`, and the shelf it opens them from.
fn shelved(own: &Arc<Fake>, ancestor: &Arc<Fake>) -> (ToolContext, Arc<Shelf>) {
    let shelf = Arc::new(Shelf {
        own: Arc::clone(own),
        ancestor: Arc::clone(ancestor),
        opened: Mutex::new(Vec::new()),
    });
    let ctx = granted().with_journals(Arc::clone(&shelf) as Arc<dyn JournalShelf>);
    (ctx, shelf)
}

#[test]
fn a_read_reaches_the_book_with_its_defaults() {
    let book = Arc::new(Fake::default());
    let out = expect_success(invoke_with_ctx(
        "journal_read",
        json!({"journal": "self", "entry": 3}),
        &with(&book),
    ));
    assert_eq!(
        out,
        json!({"entry": 3, "from": 1, "to": 1, "total": 9, "text": "#7 14:20 User: fix the scan"})
    );
    assert_eq!(*book.reads.lock().unwrap(), [(3, 1, 200)]);
}

#[test]
fn a_read_pages_where_it_is_told() {
    let book = Arc::new(Fake::default());
    expect_success(invoke_with_ctx(
        "journal_read",
        json!({"journal": "self", "entry": 3, "offset": 41, "limit": 400}),
        &with(&book),
    ));
    assert_eq!(*book.reads.lock().unwrap(), [(3, 41, 400)]);
}

#[test]
fn a_read_past_the_line_bound_is_refused() {
    let book = Arc::new(Fake::default());
    expect_error(
        &invoke_with_ctx(
            "journal_read",
            json!({"journal": "self", "entry": 3, "limit": 401}),
            &with(&book),
        ),
        "invalid_arguments",
    );
    assert!(book.reads.lock().unwrap().is_empty());
}

#[test]
fn an_entry_with_no_text_says_so() {
    let book = Arc::new(Fake::default());
    let detail = expect_error(
        &invoke_with_ctx(
            "journal_read",
            json!({"journal": "self", "entry": 8}),
            &with(&book),
        ),
        "no_entry_text",
    );
    assert!(detail.contains("entry 8"), "{detail}");
}

#[test]
fn a_search_reaches_the_book_with_its_defaults() {
    let book = Arc::new(Fake::default());
    let out = expect_success(invoke_with_ctx(
        "journal_grep",
        json!({"journal": "self", "pattern": "scan"}),
        &with(&book),
    ));
    assert_eq!(out["total"], 1);
    assert_eq!(out["hits"][0]["entry"], 3);
    assert_eq!(out["hits"][0]["part"], 7);
    assert_eq!(
        *book.greps.lock().unwrap(),
        [("scan".to_string(), None, 20)]
    );

    expect_success(invoke_with_ctx(
        "journal_grep",
        json!({"journal": "self", "pattern": "scan", "entry": 3, "max_results": 50}),
        &with(&book),
    ));
    assert_eq!(
        book.greps.lock().unwrap()[1],
        ("scan".to_string(), Some(3), 50)
    );
}

#[test]
fn an_empty_pattern_is_refused() {
    let book = Arc::new(Fake::default());
    expect_error(
        &invoke_with_ctx(
            "journal_grep",
            json!({"journal": "self", "pattern": ""}),
            &with(&book),
        ),
        "invalid_arguments",
    );
}

/// **`journal` is mandatory**: a call naming none is refused before any
/// journal is opened, so nothing is read from one picked by default.
#[test]
fn a_call_naming_no_journal_is_refused() {
    let own = Arc::new(Fake::default());
    let (ctx, shelf) = shelved(&own, &Arc::new(Fake::default()));
    expect_error(
        &invoke_with_ctx("journal_grep", json!({"pattern": "scan"}), &ctx),
        "invalid_arguments",
    );
    expect_error(
        &invoke_with_ctx("journal_read", json!({"entry": 3}), &ctx),
        "invalid_arguments",
    );
    assert!(shelf.opened.lock().unwrap().is_empty());
    assert!(own.greps.lock().unwrap().is_empty());
}

/// **An ancestor's journal is read by its name**, and the caller's own is
/// left alone.
#[test]
fn a_search_of_an_ancestor_reads_the_ancestors_book() {
    let own = Arc::new(Fake::default());
    let ancestor = Arc::new(Fake::default());
    let (ctx, shelf) = shelved(&own, &ancestor);
    expect_success(invoke_with_ctx(
        "journal_grep",
        json!({"journal": "T:t", "pattern": "scan"}),
        &ctx,
    ));
    assert_eq!(*shelf.opened.lock().unwrap(), ["T:t"]);
    assert_eq!(ancestor.greps.lock().unwrap().len(), 1);
    assert!(own.greps.lock().unwrap().is_empty());
}

#[test]
fn a_journal_the_shelf_does_not_hold_is_named_in_the_refusal() {
    let book = Arc::new(Fake::default());
    let detail = expect_error(
        &invoke_with_ctx(
            "journal_read",
            json!({"journal": "T:elsewhere"}),
            &with(&book),
        ),
        "no_such_journal",
    );
    assert_eq!(
        detail,
        "`T:elsewhere` is not a journal above this conversation"
    );
}

/// **With no `entry`, a read lists the journal's entries.**
#[test]
fn a_read_with_no_entry_lists_the_entries() {
    let book = Arc::new(Fake::default());
    let out = expect_success(invoke_with_ctx(
        "journal_read",
        json!({"journal": "self"}),
        &with(&book),
    ));
    assert_eq!(
        out,
        json!({"journal": "self", "entries": "Entry 3 — Fixed the scan."})
    );
    assert!(book.reads.lock().unwrap().is_empty());
}

#[test]
fn a_conversation_with_no_journal_says_so() {
    expect_error(
        &invoke("journal_read", json!({"journal": "self", "entry": 1})),
        "no_journal",
    );
    expect_error(
        &invoke("journal_grep", json!({"journal": "self", "pattern": "x"})),
        "no_journal",
    );
}

#[test]
fn both_only_read_so_a_resumed_turn_runs_them_again() {
    assert_eq!(
        replay("journal_read", json!({"journal": "self", "entry": 1})),
        Replay::Safe
    );
    assert_eq!(
        replay("journal_grep", json!({"journal": "self", "pattern": "x"})),
        Replay::Safe
    );
}
