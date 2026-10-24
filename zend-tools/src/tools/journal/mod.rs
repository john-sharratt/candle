//! Journal tools: `journal_read` and `journal_grep`.
//!
//! A conversation's journal shows one line per entry. What each entry was
//! written from — the turns it covers, whole — is kept beside it, and these
//! read it back: `journal_read` pages through one entry's turns the way
//! `file_read` pages a file, and `journal_grep` searches every entry's.
//!
//! The text is the conversation's own and reaches the tools through
//! [`ToolContext::journal`](crate::ToolContext::journal); the logic of reading
//! and searching it is `journal_core::transcript`'s, shared with npcd.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `no_journal` | The conversation keeps no journal: not a dialogue, or archived |
//! | `no_journal_text` | No entry holds kept text yet |
//! | `no_entry_text` | The entry named holds no kept text |

use journal_core::transcript::{Hits, Miss, Page};
use thiserror::Error;

use crate::ToolError;

pub mod grep;
pub mod read;

pub use grep::JOURNAL_GREP;
pub use read::JOURNAL_READ;

/// Where a conversation's journal text is read from — see
/// `journal_core::transcript::Book`.
pub trait JournalBook: Send + Sync {
    /// `lines` lines of `entry`'s turns from line `from`, counted from 1.
    fn read(&self, entry: u64, from: usize, lines: usize) -> Result<Page, Miss>;
    /// The lines that hold `pattern`, ignoring case, in `entry`'s turns or in
    /// every entry's; the first `max`.
    fn grep(&self, pattern: &str, entry: Option<u64>, max: usize) -> Result<Hits, Miss>;
}

#[derive(Debug, Error)]
pub enum JournalError {
    #[error("this conversation keeps no journal")]
    NoJournal,
    #[error("{}", .0.detail())]
    Missed(Miss),
}

impl ToolError for JournalError {
    fn code(&self) -> &'static str {
        match self {
            JournalError::NoJournal => "no_journal",
            JournalError::Missed(miss) => miss.code(),
        }
    }
}
