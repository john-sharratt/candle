//! `sub_run` tool — delegates to the injected SubagentRunner.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, SubagentRequest, Tool, ToolContext, ToolError};

#[derive(Debug, Error)]
pub enum SubagentError {
    #[error("no subagent runner configured in ToolContext")]
    NotConfigured,
    #[error("subagent failed: {0}")]
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
    #[validate(length(min = 1))]
    pub instruction: String,
    pub tools: Option<Vec<String>>,
    pub model: Option<String>,
    pub endpoint: Option<String>,
    #[validate(range(min = 1, max = 50))]
    pub max_turns: Option<u32>,
}

#[derive(Serialize)]
pub struct Response {
    pub result: String,
    pub turns: u32,
    pub tool_calls_made: u32,
}

pub struct SubagentRun;

impl Tool for SubagentRun {
    const NAME: &'static str = "sub_run";
    const DESCRIPTION: &'static str =
        "Spawn a nested agent loop with its own context, message history, and tool subset. \
         Use for: decomposing complex multi-step tasks, parallelising subtasks with different \
         expertise, delegating to a specialised model. Requires the host process to inject a \
         SubagentRunner into ToolContext. Returns the agent's final result text, turn count, \
         and tool calls made. Triggered by: 'have a subagent', 'run this as a sub-task', \
         'delegate to another agent', 'use a nested agent'. Not for simple tool calls — use \
         the direct tools for those.";

    type Request = Request;
    type Response = Response;
    type Error = SubagentError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, SubagentError> {
        let runner = ctx
            .subagent_runner
            .as_deref()
            .ok_or(SubagentError::NotConfigured)?;
        runner
            .run(SubagentRequest {
                instruction: req.instruction,
                tools: req.tools,
                model: req.model,
                endpoint: req.endpoint,
                max_turns: req.max_turns.unwrap_or(10),
            })
            .map(|r| Response {
                result: r.result,
                turns: r.turns,
                tool_calls_made: r.tool_calls_made,
            })
            .map_err(SubagentError::RunFailed)
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<SubagentRun>();
