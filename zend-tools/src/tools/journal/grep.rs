//! journal_grep tool.

use journal_core::transcript::{Hits, MAX_HITS};
use schemars::JsonSchema;
use serde::Deserialize;
use validator::Validate;

use super::{open, JournalError};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// Hits a search returns when it names no count.
const DEFAULT_HITS: usize = 20;

#[derive(Deserialize, JsonSchema, Validate)]
pub struct GrepRequest {
    /// Whose journal: "self" for your own, or an ancestor's name or path as
    /// your journals show it. Required.
    #[validate(length(min = 1))]
    pub journal: String,
    /// Text to find, ignoring case — plain text, not a pattern. Required.
    #[validate(length(min = 1))]
    pub pattern: String,
    /// Search only this entry's turns. Default: every entry's.
    pub entry: Option<u64>,
    /// How many matching lines to return. Default 20, at most 50.
    #[validate(range(min = 1, max = 50))]
    pub max_results: Option<usize>,
}

pub struct JournalGrep;

impl Tool for JournalGrep {
    const NAME: &'static str = "journal_grep";
    const DESCRIPTION: &'static str =
        "Search a journal for text, ignoring case: \"self\" for the turns behind your own \
         entries, or an ancestor's journal by its name or path. Says which entry and turn \
         said it, and the line. Use it to find something from earlier that the journal's \
         one-line entries do not spell out — a path, an error, a name, a decision. Each hit \
         names its entry and line, to read around with journal_read.";

    type Request = GrepRequest;
    type Response = Hits;
    type Error = JournalError;

    /// Reads the journal; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: GrepRequest) -> Result<Hits, JournalError> {
        open(ctx, &req.journal)?
            .grep(
                &req.pattern,
                req.entry,
                req.max_results.unwrap_or(DEFAULT_HITS).min(MAX_HITS),
            )
            .map_err(JournalError::Missed)
    }
}

pub const JOURNAL_GREP: RegisteredTool = RegisteredTool::new::<JournalGrep>();
