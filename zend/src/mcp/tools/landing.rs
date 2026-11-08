//! Landing a change set on a workspace repository: the one path
//! `file_edit_and_commit`, `file_download_and_commit` and
//! `file_upload_and_commit` share (`docs/mcp_file_tools.md` §3.2, §3.6).
//!
//! The repository, its author and the branch come from the call and the
//! workspace manifest; the changes become one commit on the branch's tip by
//! that author, pushed through the repository's pusher — so a gated
//! repository's gate is asked first and every subscriber to `repo/<repo>`
//! hears of it. The answer is **live**, with the commit, or **refused**, with
//! why; on a refusal nothing moved anywhere.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use web::auth::Role;
use zend_tools::tools::git::open;
use zend_vfs::{BranchName, GitTime, Signature};

use super::{string, Answer, Ctx, LOADING};
use crate::file_commit::change::Change;
use crate::file_commit::commit::{commit, Outcome};

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

/// The repository and message `args` name, read before the changes are made:
/// what a transfer checks before it fetches anything.
pub fn target(ctx: &Ctx, args: &Value) -> Result<(String, String), Answer> {
    if !ctx.caller.role.at_least(Role::User) {
        return Err(Answer::error(
            "committing to a repository is a user's; you are a guest",
        ));
    }
    match (string(args, "repo"), string(args, "message")) {
        (Ok(r), Ok(m)) => Ok((r.to_string(), m.to_string())),
        (Err(e), _) | (_, Err(e)) => Err(Answer::error(e)),
    }
}

/// `changes` landed on the repository and branch `args` name, with `args`'
/// message, as the repository's author.
pub fn land(ctx: &Ctx, args: &Value, changes: &[Change]) -> Answer {
    let (repo_name, message) = match target(ctx, args) {
        Ok(t) => t,
        Err(a) => return a,
    };
    let repo_name = repo_name.as_str();
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
    match commit(&repo, &branch, changes, &message, &signature) {
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

/// A refusal before anything was committed, in the shape a refused commit
/// answers.
pub fn refused(repo: &str, why: &str) -> Answer {
    Answer::error(json!({"status": "refused", "repo": repo, "reason": why}).to_string())
}
