//! journal_move tool: a journal line moved to another journal, whole. A line's
//! file name is its id, so a move keeps the name and the bytes.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::tools::file::mv::move_file;
use crate::tools::file::FileError;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// A journal's own file, which a line is kept beside and never is.
const JOURNAL_FILE: &str = "journal.yaml";

#[derive(Deserialize, JsonSchema, Validate)]
pub struct MoveRequest {
    /// The repository holding the journals. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The line's file, relative to the repository (e.g. `groups/zen/people/john/2026-10-31-0131-4159.yaml`). Required.
    #[validate(length(min = 1))]
    pub line: String,
    /// The folder of the journal it goes to, relative to the repository (e.g. `groups/zen/topics/family-story`). Required.
    #[validate(length(min = 1))]
    pub journal: String,
}

#[derive(Serialize)]
pub struct MoveResponse {
    pub repo: String,
    pub line: String,
    pub to: String,
}

pub struct JournalMove;

impl Tool for JournalMove {
    const NAME: &'static str = "journal_move";
    const DESCRIPTION: &'static str =
        "Move a line of a journal to another journal in the memory repository, keeping its id \
         and bytes exactly. Use for: filing a note under the right person or topic, moving \
         shared facts onto a shared topic. Triggered by \"move this note to\", \"file it under\". \
         The journal must already exist. Returns the line's old and new paths. Held in this \
         session until committed; run journal_check before committing.";

    type Request = MoveRequest;
    type Response = MoveResponse;
    type Error = FileError;

    /// Sent again after it landed, the line is no longer where it was, so the
    /// call is refused and nothing changes.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: MoveRequest) -> Result<MoveResponse, FileError> {
        let name = line_name(&req.line)?;
        let store = ctx.files.repo(&req.repo)?;
        let folder = req.journal.trim_end_matches('/');
        if store
            .read_bytes(&format!("{folder}/{JOURNAL_FILE}"))?
            .is_none()
        {
            return Err(FileError::InvalidArguments(format!(
                "{folder} is no journal: it holds no {JOURNAL_FILE}"
            )));
        }
        let to = format!("{folder}/{name}");
        move_file(&store, &req.line, &to)?;
        Ok(MoveResponse {
            repo: req.repo,
            line: req.line,
            to,
        })
    }
}

/// The file name of the line at `path`, or why it names no line.
fn line_name(path: &str) -> Result<&str, FileError> {
    let name = path.rsplit('/').next().unwrap_or(path);
    if name == JOURNAL_FILE || !name.ends_with(".yaml") {
        return Err(FileError::InvalidArguments(format!(
            "{path} is not a journal line: a line is a `<id>.yaml` file"
        )));
    }
    Ok(name)
}

pub const JOURNAL_MOVE: RegisteredTool = RegisteredTool::new::<JournalMove>();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_journal_file_is_no_line() {
        assert_eq!(line_name("g/a/2026-1.yaml").unwrap(), "2026-1.yaml");
        assert!(line_name("g/a/journal.yaml").is_err());
        assert!(line_name("g/a/notes.txt").is_err());
    }
}
