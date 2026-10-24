//! An agent's instruction file is changed only when the user asks for it.
//!
//! `CLAUDE.md` and `AGENTS.md` are read by every agent that works in the
//! repository, so a line written into one becomes a standing instruction to
//! all of them. The system prompt says never to add instructions to one unless
//! asked, and that holds about three runs in four: asked to revise a paper, a
//! dialogue twice added a usage directive to `CLAUDE.md` beside the paper
//! edits it was asked for (replays of 2026-10-24/25). So the round is screened
//! instead: a call that writes, edits or deletes an instruction file runs only
//! when the user's message names that file, and is otherwise refused with a
//! result that says to propose the change.

use std::path::Path;

use serde_json::json;

use crate::repeat_guard::OVERLAY_WRITERS;
use crate::tool_round::Step;
use crate::tools::ToolResult;

/// The file names agents read as standing instructions.
pub const INSTRUCTION_FILES: [&str; 2] = ["CLAUDE.md", "AGENTS.md"];

/// `steps` with every write to an instruction file `request` does not name
/// turned into a refusal. Every other step passes as it was.
pub fn screen(steps: Vec<Step>, request: &str) -> Vec<Step> {
    let request = request.to_lowercase();
    steps
        .into_iter()
        .map(|step| match step {
            Step::Run(call) if OVERLAY_WRITERS.contains(&call.name.as_str()) => {
                match instruction_file(call.arguments["path"].as_str().unwrap_or("")) {
                    Some(name) if !request.contains(&name.to_lowercase()) => {
                        Step::Refuse(ToolResult {
                            response: json!({
                                "error": "not_permitted",
                                "detail": format!(
                                    "{name} holds standing instructions for every agent in \
                                     the repository, and the user did not ask for it to be \
                                     changed; nothing was done. Propose the change to the \
                                     user instead."
                                ),
                            }),
                            call,
                        })
                    }
                    _ => Step::Run(call),
                }
            }
            other => other,
        })
        .collect()
}

/// The instruction file `path` names, if it names one — in any case, as a
/// Windows checkout opens `claude.md` as the same file.
fn instruction_file(path: &str) -> Option<&'static str> {
    let file = Path::new(path).file_name()?.to_str()?;
    INSTRUCTION_FILES
        .into_iter()
        .find(|name| name.eq_ignore_ascii_case(file))
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;
    use crate::tools::ToolCall;

    fn run(name: &str, path: &str) -> Step {
        Step::Run(ToolCall {
            name: name.into(),
            arguments: json!({"repo": "candle", "path": path}),
        })
    }

    fn refusal(step: &Step) -> Option<&Value> {
        match step {
            Step::Refuse(result) => Some(&result.response),
            _ => None,
        }
    }

    /// **An edit to an instruction file the user never named is refused**,
    /// whatever folder it is in, and the refusal says to propose it.
    #[test]
    fn an_unasked_edit_to_an_instruction_file_is_refused() {
        let steps = screen(
            vec![
                run("file_edit", "CLAUDE.md"),
                run("write", "docs/agents_overview.md"),
                run("file_delete", "web/AGENTS.md"),
                run("file_edit", "docs/unbounded_agents.md"),
            ],
            "ok. make those changes first then ill help you with more",
        );
        let first = refusal(&steps[0]).expect("CLAUDE.md edit refused");
        assert_eq!(first["error"], "not_permitted");
        assert!(first["detail"]
            .as_str()
            .unwrap()
            .starts_with("CLAUDE.md holds standing instructions"));
        assert_eq!(refusal(&steps[1]), None, "only the file name itself counts");
        assert_eq!(
            refusal(&steps[2]).unwrap()["error"],
            "not_permitted",
            "AGENTS.md in a subfolder"
        );
        assert_eq!(steps[3], run("file_edit", "docs/unbounded_agents.md"));
    }

    /// **A request that names the file lets the write run** — in any case.
    #[test]
    fn a_request_naming_the_file_lets_it_run() {
        let steps = screen(
            vec![run("file_edit", "CLAUDE.md"), run("file_edit", "AGENTS.md")],
            "add the new build flag to claude.md",
        );
        assert_eq!(steps[0], run("file_edit", "CLAUDE.md"));
        assert_eq!(refusal(&steps[1]).unwrap()["error"], "not_permitted");
    }

    /// Reading an instruction file is never screened.
    #[test]
    fn reads_pass() {
        let steps = screen(vec![run("file_read", "CLAUDE.md")], "");
        assert_eq!(steps[0], run("file_read", "CLAUDE.md"));
    }
}
