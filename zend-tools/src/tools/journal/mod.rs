//! Journal tools: `journal_read` and `journal_grep`.
//!
//! A conversation's journal shows one line per entry. What each entry was
//! written from — the turns it covers, whole — is kept beside it, and these
//! read it back: `journal_read` lists a journal's entries, or pages through one
//! entry's turns the way `file_read` pages a file, and `journal_grep` searches
//! every entry's.
//!
//! Each call names its `journal` (`docs/claude_zend_gateway.md` §5.3): `"self"`
//! for the conversation's own, or — for a conversation with a lineage — an
//! ancestor's, by its bare name or its full path. The journals reach the tools
//! through [`ToolContext::journals`](crate::ToolContext::journals); the logic
//! of reading and searching turns is `journal_core::transcript`'s, shared with
//! npcd.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `no_journal` | The conversation keeps no journal: not a dialogue, or archived |
//! | `no_such_journal` | `journal` names nothing the conversation can read |
//! | `two_entries` | the address in `journal` and `entry` name different entries |
//! | `no_journal_text` | No entry holds kept text yet |
//! | `no_entry_text` | The entry named holds no kept text |

use std::sync::Arc;

use journal_core::transcript::{Hits, Miss, Page};
use thiserror::Error;

use crate::{ToolContext, ToolError};

pub mod grep;
pub mod read;

pub use grep::JOURNAL_GREP;
pub use read::JOURNAL_READ;

/// What `journal` names the caller's own journal with.
pub const SELF: &str = "self";

/// One journal a conversation's tools can read.
pub trait JournalBook: Send + Sync {
    /// Its entries and open items, as `journal_read` shows them with no
    /// `entry`.
    fn entries(&self) -> String;
    /// `lines` lines of `entry`'s turns from line `from`, counted from 1.
    fn read(&self, entry: u64, from: usize, lines: usize) -> Result<Page, Miss>;
    /// The lines that hold `pattern`, ignoring case, in `entry`'s turns or in
    /// every entry's; the first `max`.
    fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss>;
}

/// The journals a conversation reaches by name: its own, and its ancestors'.
pub trait JournalShelf: Send + Sync {
    /// The journal `name` names for this conversation — [`SELF`], an
    /// ancestor's bare name or its full path — or what was not found.
    fn open(&self, name: &str) -> Result<Arc<dyn JournalBook>, String>;
}

#[derive(Debug, Error)]
pub enum JournalError {
    #[error("this conversation keeps no journal")]
    NoJournal,
    #[error("{0}")]
    Unknown(String),
    #[error("the address names entry {0} and `entry` names {1}: name one")]
    TwoEntries(u64, u64),
    #[error("{}", .0.detail())]
    Missed(Miss),
}

impl ToolError for JournalError {
    fn code(&self) -> &'static str {
        match self {
            JournalError::NoJournal => "no_journal",
            JournalError::Unknown(_) => "no_such_journal",
            JournalError::TwoEntries(..) => "two_entries",
            JournalError::Missed(miss) => miss.code(),
        }
    }
}

/// `journal` split into the journal it names and the entry its address names,
/// if any: `T:review#12` is entry 12 of `T:review`, and `check-plan#3.2` —
/// one claim of an entry — is entry 3 of `check-plan`
/// (`docs/claude_zend_gateway.md` §5.2).
pub fn address(journal: &str) -> (&str, Option<u64>) {
    match journal.rsplit_once('#') {
        Some((name, number)) => match number.split('.').next().map(str::parse::<u64>) {
            Some(Ok(entry)) if !name.is_empty() => (name, Some(entry)),
            _ => (journal, None),
        },
        None => (journal, None),
    }
}

/// The journal `name` names in `ctx`.
fn open(ctx: &ToolContext, name: &str) -> Result<Arc<dyn JournalBook>, JournalError> {
    ctx.journals()
        .ok_or(JournalError::NoJournal)?
        .open(name)
        .map_err(JournalError::Unknown)
}
