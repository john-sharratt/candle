//! journal_read tool.

use journal_core::transcript::{Page, MAX_LINES};
use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::JournalError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Lines a read returns when it names no count.
const DEFAULT_LINES: usize = 200;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// The number of the journal entry whose turns to read, as the journal
    /// shows it: 3 for "Entry 3 — …". Required.
    pub entry: u64,
    /// First line to show, counted from 1. Default 1.
    pub offset: Option<usize>,
    /// How many lines to show. Default 200, at most 400.
    #[validate(range(min = 1, max = 400))]
    pub limit: Option<usize>,
}

pub struct JournalRead;

impl Tool for JournalRead {
    const NAME: &'static str = "journal_read";
    const DESCRIPTION: &'static str =
        "Read the turns behind one of your journal entries, whole: what the user said, each \
         tool result and what you replied, each turn headed by its number and time. Use it \
         when an entry's one line is not enough — the exact file contents, error or answer \
         from earlier. Pages like file_read: `offset` and `limit` in lines. To find which \
         entry holds something, use journal_grep.";

    type Request = ReadRequest;
    type Response = Page;
    type Error = JournalError;

    /// Reads the journal; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<Page, JournalError> {
        let book = ctx.journal().ok_or(JournalError::NoJournal)?;
        book.read(
            req.entry,
            req.offset.unwrap_or(1),
            req.limit.unwrap_or(DEFAULT_LINES).min(MAX_LINES),
        )
        .map_err(JournalError::Missed)
    }
}

pub const JOURNAL_READ: RegisteredTool = RegisteredTool::new::<JournalRead>();
