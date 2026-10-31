//! journal_check tool: `journals check` over this session's copy of the
//! memory repository, so a job's restructure is checked before it is committed.

use std::collections::BTreeMap;

use journals::tree::load::load;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::tools::file::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct CheckRequest {
    /// The repository holding the journals. Required.
    #[validate(length(min = 1))]
    pub repo: String,
}

#[derive(Serialize)]
pub struct CheckResponse {
    pub repo: String,
    pub journals: usize,
    pub lines: usize,
    /// Everything wrong with the tree as this session holds it; empty when
    /// nothing is.
    pub problems: Vec<String>,
}

pub struct JournalCheck;

impl Tool for JournalCheck {
    const NAME: &'static str = "journal_check";
    const DESCRIPTION: &'static str =
        "Check the memory's journal repository as this session holds it, uncommitted changes \
         included, reading it as the memory service would. Use before committing a change to \
         the journals: moved lines, edited entries, a new journal. Triggered by \"check the \
         journals\", \"validate the memory\", \"is the journal tree valid\". Returns how many \
         journals and lines it holds and every problem found; none means it is safe to commit.";

    type Request = CheckRequest;
    type Response = CheckResponse;
    type Error = FileError;

    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: CheckRequest) -> Result<CheckResponse, FileError> {
        let store = ctx.files.repo(&req.repo)?;
        let mut files = BTreeMap::new();
        for path in store.paths("") {
            if let Some(bytes) = store.read_bytes(&path)? {
                files.insert(path, bytes);
            }
        }
        let tree = load(&files);
        let journals = tree.journals();
        let lines = journals.iter().map(|(_, j)| j.all_lines().count()).sum();
        let problems = tree.all_problems();
        Ok(CheckResponse {
            repo: req.repo,
            journals: journals.len(),
            lines,
            problems,
        })
    }
}

pub const JOURNAL_CHECK: RegisteredTool = RegisteredTool::new::<JournalCheck>();
