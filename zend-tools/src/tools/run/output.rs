//! `run_output` — read a page of a run's output.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{JobId, JobStatus};

use super::RunError;
use crate::sandboxes::LogPage;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
pub struct OutputRequest {
    /// The repository the program ran in. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The `job` a run_command result gave. Required.
    #[validate(length(min = 1))]
    pub job: String,
    /// Which page of the output, 0-based; a page past the end is the last. Required.
    pub page: usize,
}

#[derive(Serialize)]
pub struct OutputResponse {
    pub repo: String,
    pub job: String,
    /// `queued`, `running`, `exited`, `timed_out`, `cancelled`, `refused` or
    /// `failed`.
    pub status: &'static str,
    /// The exit code, once it has exited.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub output: LogPage,
}

pub struct RunOutput;

impl Tool for RunOutput {
    const NAME: &'static str = "run_output";
    const DESCRIPTION: &'static str =
        "Read one page of what a program started with run_command printed, by the `job` its \
         result gave. Use for: the rest of a long test run or build log, the failures at the end \
         of the output. Triggered by \"show the rest of the output\", \"what failed at the end\", \
         \"next page of the log\". Returns the page, which lines of how many it holds, and how \
         the run stands.";

    type Request = OutputRequest;
    type Response = OutputResponse;
    type Error = RunError;

    /// Reading a log changes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: OutputRequest) -> Result<OutputResponse, RunError> {
        let sandboxes = ctx.sandboxes()?.ok_or(RunError::NoSandboxes)?;
        // The repository is named as every file tool names it, so an unknown
        // one is `unknown_repo` here too.
        ctx.files.repo(&req.repo)?;
        let id = JobId::parse(&req.job).ok_or_else(|| RunError::BadJob(req.job.clone()))?;
        let (info, output) = sandboxes.output(&req.repo, &id, req.page)?;
        let (status, exit_code) = match info.status {
            JobStatus::Queued => ("queued", None),
            JobStatus::Running => ("running", None),
            JobStatus::Exited { code } => ("exited", code),
            JobStatus::TimedOut => ("timed_out", None),
            JobStatus::Cancelled => ("cancelled", None),
            JobStatus::Refused { .. } => ("refused", None),
            JobStatus::Failed { .. } => ("failed", None),
        };
        Ok(OutputResponse {
            repo: req.repo,
            job: req.job,
            status,
            exit_code,
            output,
        })
    }
}

pub const RUN_OUTPUT: RegisteredTool = RegisteredTool::new::<RunOutput>();
