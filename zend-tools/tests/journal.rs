//! Integration tests for `journal_read` and `journal_grep`.
//!
//! What a page or a search holds is `journal_core::transcript`'s, tested there.
//! These pin the tools' half: the arguments reach the book as the call wrote
//! them, with its defaults and bounds, and each way of finding nothing comes
//! back under its own code.

mod harness;

use std::sync::{Arc, Mutex};

use harness::{expect_error, expect_success, granted, invoke, invoke_with_ctx, replay};
use journal_core::transcript::{Hit, Hits, Miss, Page};
use serde_json::json;
use zend_tools::tools::journal::JournalBook;
use zend_tools::Replay;

/// A book holding entry 3 alone, recording every request it is asked.
#[derive(Default)]
struct Fake {
    reads: Mutex<Vec<(u64, usize, usize)>>,
    greps: Mutex<Vec<(String, Option<u64>, usize)>>,
}

impl JournalBook for Fake {
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

fn with(book: &Arc<Fake>) -> zend_tools::ToolContext {
    granted().with_journal(Arc::clone(book) as Arc<dyn JournalBook>)
}

#[test]
fn a_read_reaches_the_book_with_its_defaults() {
    let book = Arc::new(Fake::default());
    let out = expect_success(invoke_with_ctx(
        "journal_read",
        json!({"entry": 3}),
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
        json!({"entry": 3, "offset": 41, "limit": 400}),
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
            json!({"entry": 3, "limit": 401}),
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
        &invoke_with_ctx("journal_read", json!({"entry": 8}), &with(&book)),
        "no_entry_text",
    );
    assert!(detail.contains("entry 8"), "{detail}");
}

#[test]
fn a_search_reaches_the_book_with_its_defaults() {
    let book = Arc::new(Fake::default());
    let out = expect_success(invoke_with_ctx(
        "journal_grep",
        json!({"pattern": "scan"}),
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
        json!({"pattern": "scan", "entry": 3, "max_results": 50}),
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
        &invoke_with_ctx("journal_grep", json!({"pattern": ""}), &with(&book)),
        "invalid_arguments",
    );
}

#[test]
fn a_conversation_with_no_journal_says_so() {
    expect_error(&invoke("journal_read", json!({"entry": 1})), "no_journal");
    expect_error(
        &invoke("journal_grep", json!({"pattern": "x"})),
        "no_journal",
    );
}

#[test]
fn both_only_read_so_a_resumed_turn_runs_them_again() {
    assert_eq!(replay("journal_read", json!({"entry": 1})), Replay::Safe);
    assert_eq!(
        replay("journal_grep", json!({"pattern": "x"})),
        Replay::Safe
    );
}
