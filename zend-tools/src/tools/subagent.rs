//! `sub_run` tool — starts a part under the calling conversation through the
//! injected [`SubagentRunner`] and answers with how it ended.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Replay, SubagentRequest, Tool, ToolContext, ToolError};

#[derive(Debug, Error)]
pub enum SubagentError {
    #[error("this conversation cannot start parts: no runner is configured for it")]
    NotConfigured,
    #[error("the part could not run: {0}")]
    RunFailed(String),
}

impl ToolError for SubagentError {
    fn code(&self) -> &'static str {
        match self {
            SubagentError::NotConfigured => "not_configured",
            SubagentError::RunFailed(_) => "subagent_failed",
        }
    }
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// What the part is for, as a kebab-case slug of at most 40 characters —
    /// `verify-findings`, `read-parser` — never a number. Calling again with a
    /// name this task already used returns that part rather than starting
    /// another.
    #[validate(length(min = 1, max = 40))]
    pub name: String,
    /// The part's task, stated in full: it shares your journals, not your
    /// turns. End it by asking for the final response's shape.
    #[validate(length(min = 1))]
    pub prompt: String,
}

#[derive(Serialize)]
pub struct Response {
    pub path: String,
    pub state: String,
    pub result: String,
}

pub struct SubagentRun;

impl Tool for SubagentRun {
    const NAME: &'static str = "sub_run";
    const DESCRIPTION: &'static str =
        "Start a part of your task as a conversation of its own and wait for it to end: one file \
         of twelve to review, one crate of three to research, or a verifier that checks your draft \
         findings against the files they cite before you answer. The part starts with your \
         journals and your thread's, not your turns, so state its task in full. Several sub_run \
         calls in one round run at once. Returns the part's journal entry and its address, which \
         you cite in your own answer.";

    type Request = Request;
    type Response = Response;
    type Error = SubagentError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, SubagentError> {
        let runner = ctx
            .subagent_runner
            .as_deref()
            .ok_or(SubagentError::NotConfigured)?;
        runner
            .run(
                ctx.lineage(),
                SubagentRequest {
                    name: req.name,
                    prompt: req.prompt,
                },
            )
            .map(|r| Response {
                path: r.path,
                state: r.state,
                result: r.result,
            })
            .map_err(SubagentError::RunFailed)
    }

    /// The runner finds a part by its caller and name, so a call issued again
    /// after a restart waits on, or returns, the part the first one started.
    fn replay(_req: &Request) -> Replay {
        Replay::Safe
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<SubagentRun>();
