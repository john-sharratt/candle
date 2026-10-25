//! journal_read tool.

use journal_core::transcript::{Page, MAX_LINES};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{address, open, JournalError};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Lines a read returns when it names no count.
const DEFAULT_LINES: usize = 200;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct ReadRequest {
    /// Whose journal: "self" for your own, or an ancestor's name or path as
    /// your journals show it — or an entry's address, `name#3`. Required.
    #[validate(length(min = 1))]
    pub journal: String,
    /// The entry whose turns to read, as the journal numbers it: 3 for
    /// "Entry 3 — …". Leave it out, with no address, to list the journal's
    /// entries.
    pub entry: Option<u64>,
    /// First line to show, counted from 1. Default 1.
    pub offset: Option<usize>,
    /// How many lines to show. Default 200, at most 400.
    #[validate(range(min = 1, max = 400))]
    pub limit: Option<usize>,
}

/// A journal's entries, or one page of an entry's turns.
#[derive(Serialize)]
#[serde(untagged)]
pub enum ReadResponse {
    Entries { journal: String, entries: String },
    Turns(Page),
}

pub struct JournalRead;

impl Tool for JournalRead {
    const NAME: &'static str = "journal_read";
    const DESCRIPTION: &'static str =
        "Read a journal: \"self\" for your own, or an ancestor's by its name or path. Without \
         `entry`, its entries; with one, the turns behind that entry, whole — what was said, \
         each tool result and each reply, each turn headed by its number and time. Use it \
         when an entry's one line is not enough — the exact file contents, error or answer \
         from earlier. Pages like file_read: `offset` and `limit` in lines. To find which \
         entry holds something, use journal_grep.";

    type Request = ReadRequest;
    type Response = ReadResponse;
    type Error = JournalError;

    /// Reads the journal; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<ReadResponse, JournalError> {
        let (journal, addressed) = address(&req.journal);
        if let (Some(a), Some(e)) = (addressed, req.entry) {
            if a != e {
                return Err(JournalError::TwoEntries(a, e));
            }
        }
        let book = open(ctx, journal)?;
        let Some(entry) = req.entry.or(addressed) else {
            return Ok(ReadResponse::Entries {
                journal: req.journal,
                entries: book.entries(),
            });
        };
        book.read(
            entry,
            req.offset.unwrap_or(1),
            req.limit.unwrap_or(DEFAULT_LINES).min(MAX_LINES),
        )
        .map(ReadResponse::Turns)
        .map_err(JournalError::Missed)
    }
}

pub const JOURNAL_READ: RegisteredTool = RegisteredTool::new::<JournalRead>();
