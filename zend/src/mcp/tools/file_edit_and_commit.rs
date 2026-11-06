//! `file_edit_and_commit {repo, message, changes, branch?}`: edit text files in
//! a workspace repository and commit them, all or nothing
//! (`docs/mcp_file_tools.md` §3). The answer is **live**, with the commit, or
//! **refused**, with why — an edit that did not match, the branch moving
//! under it three times, or the repository's gate refusing the change; on a
//! refusal nothing moved anywhere.
//!
//! The commit is the repository's author's (the workspace manifest's
//! `author:`), never the caller's identity or a model's. It is made on the
//! branch's tip on origin and pushed through the repository's pusher, so a
//! gated repository's gate is asked first and every subscriber to
//! `repo/<repo>` hears of it.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use web::auth::Role;
use zend_tools::tools::git::open;
use zend_vfs::{BranchName, GitTime, Signature};

use super::{string, Answer, Ctx, LOADING};
use crate::file_commit::change::Change;
use crate::file_commit::commit::{commit, Outcome};

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
            their refusal names the rule, the file and the line.",
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

/// The author a commit to `repo` is made by, now.
fn author(name: &str, email: &str) -> Result<Signature, String> {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    Signature::new(
        name,
        email,
        GitTime {
            seconds,
            offset_minutes: 0,
        },
    )
    .map_err(|e| e.to_string())
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    if !ctx.caller.role.at_least(Role::User) {
        return Answer::error("committing to a repository is a user's; you are a guest");
    }
    let (repo_name, message) = match (string(args, "repo"), string(args, "message")) {
        (Ok(r), Ok(m)) => (r, m),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let changes: Vec<Change> = match args.get("changes").cloned().map(serde_json::from_value) {
        Some(Ok(changes)) => changes,
        Some(Err(e)) => return Answer::error(format!("`changes` is not a list of changes: {e}")),
        None => return Answer::error("`changes` is missing"),
    };
    let Some((tools, bases)) = ctx.session.repository_tools() else {
        return Answer::error(LOADING);
    };
    let Some(workspace) = tools.files.workspace() else {
        return Answer::error("this daemon serves no workspace");
    };
    let Some(spec) = workspace.repo(repo_name) else {
        return Answer::error(format!(
            "there is no repository `{repo_name}`; the workspace has {}",
            workspace.names().join(", ")
        ));
    };
    let Some(who) = spec.author.clone() else {
        return Answer::error(format!(
            "the workspace names no author for `{repo_name}`, so there is no one to commit as"
        ));
    };
    let branch_name = match args.get("branch").and_then(Value::as_str) {
        Some(b) => b.to_string(),
        None => match bases.get(repo_name) {
            Some(b) => b.clone(),
            None => return Answer::error(format!("`{repo_name}` has no main branch to commit on")),
        },
    };
    let branch = match BranchName::parse(&branch_name) {
        Ok(b) => b,
        Err(e) => return Answer::error(e.to_string()),
    };
    let signature = match author(&who.name, &who.email) {
        Ok(s) => s,
        Err(e) => return Answer::error(e),
    };
    let repo = match open(&tools, repo_name) {
        Ok(r) => r,
        Err(e) => return Answer::error(e.to_string()),
    };
    match commit(&repo, &branch, &changes, message, &signature) {
        Ok(Outcome::Live { commit, .. }) => {
            let paths: Vec<&str> = changes.iter().flat_map(Change::paths).collect();
            Answer::ok(
                json!({
                    "status": "live",
                    "repo": repo_name,
                    "branch": branch_name,
                    "commit": commit.as_str(),
                    "author": who.name,
                    "paths": paths,
                })
                .to_string(),
            )
        }
        Ok(Outcome::Refused(why)) => Answer::error(
            json!({"status": "refused", "repo": repo_name, "reason": why}).to_string(),
        ),
        Err(e) => Answer::error(
            json!({"status": "refused", "repo": repo_name, "reason": e.to_string()}).to_string(),
        ),
    }
}
