//! `file_edit_and_commit {repo, message, changes, branch?}`: edit text files in
//! a workspace repository and commit them, all or nothing
//! (`docs/mcp_file_tools.md` §3). The answer is **live**, with the commit, or
//! **refused**, with why — an edit that did not match, the branch moving
//! under it three times, or the repository's gate refusing the change; on a
//! refusal nothing moved anywhere.
//!
//! The commit is the repository's author's (the workspace manifest's
//! `author:`), never the caller's identity or a model's, and lands through
//! [`super::landing`], the path every committing tool shares.

use serde_json::{json, Value};

use super::landing::{land, target};
use super::{Answer, Ctx};
use crate::file_commit::change::Change;

pub const NAME: &str = "file_edit_and_commit";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Edit text files in one of the user's repositories and commit the change, \
            all at once or not at all. Read each file first with `file_read` and copy the text you \
            replace exactly: an `edit` replaces each `old` with `new`, and every `old` must occur \
            in the file exactly once, as it stands after the edits before it. `write` gives a \
            file's whole new text (and creates it), `delete` removes one, `move` renames one. The \
            changes become one commit on the branch (the repository's main branch unless named), \
            made by the repository's own author. The answer says `live` with the commit, or \
            `refused` with why — and on a refusal nothing changed anywhere: fix what it names \
            and call again. Some repositories check every change before it lands (Yen Bots does); \
            their refusal names the rule, the file and the line. For a picture or any file that \
            is not text use `file_download_and_commit` (from a link) or `file_upload_and_commit` \
            (bytes you hold).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "The repository, by name: `yenbot`, `candle`, …"},
                "message": {"type": "string", "description": "The commit message: what changed and why, in a line."},
                "branch": {"type": "string", "description": "The branch to commit on; the repository's main branch by default."},
                "changes": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 100,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["edit", "write", "delete", "move"]},
                            "path": {"type": "string", "description": "The file, relative to the repository's root. For `edit`, `write`, `delete`."},
                            "edits": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "old": {"type": "string", "description": "Text in the file, copied exactly, occurring once."},
                                        "new": {"type": "string", "description": "What replaces it."}
                                    },
                                    "required": ["old", "new"],
                                    "additionalProperties": false
                                },
                                "description": "For `edit`: the replacements, in order."
                            },
                            "content": {"type": "string", "description": "For `write`: the file's whole new text."},
                            "from": {"type": "string", "description": "For `move`: the file's path now."},
                            "to": {"type": "string", "description": "For `move`: its new path, which must not exist."}
                        },
                        "required": ["op"],
                        "additionalProperties": false
                    }
                }
            },
            "required": ["repo", "message", "changes"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(refused) = target(ctx, args) {
        return refused;
    }
    let changes: Vec<Change> = match args.get("changes").cloned().map(serde_json::from_value) {
        Some(Ok(changes)) => changes,
        Some(Err(e)) => return Answer::error(format!("`changes` is not a list of changes: {e}")),
        None => return Answer::error("`changes` is missing"),
    };
    land(ctx, args, &changes)
}
