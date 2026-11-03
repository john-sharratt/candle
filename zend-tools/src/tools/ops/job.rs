//! `repo_job_status`, `repo_job_log`, `repo_job_list`, `repo_job_cancel`: a
//! build-agent job watched, read, found and stopped by its handle — what a
//! worker reads a deploy's build with (`docs/build_deploy_tools.md` §2, §3).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{forward, LogMode, OpsError, Report};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct StatusRequest {
    /// The job's handle, `<node>:<id>`. Required.
    #[validate(length(min = 18, max = 64))]
    pub job: String,
    /// Seconds to wait for it to end. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(max = 3600))]
    pub wait_s: Option<u32>,
}

pub struct JobStatus;

impl Tool for JobStatus {
    const NAME: &'static str = "repo_job_status";
    const DESCRIPTION: &'static str =
        "Where a build-agent job is — its machine, phase, time and lines — waiting up to wait_s for it \
         to end; an ended job's answer carries its test results, errors and failures by line.";

    type Request = StatusRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: StatusRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &StatusRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct LogRequest {
    /// The job's handle, `<node>:<id>`. Required.
    #[validate(length(min = 18, max = 64))]
    pub job: String,
    /// How to read it. Required.
    pub mode: LogMode,
    /// `grep`: the text to find, ignoring case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(length(min = 1, max = 200))]
    pub pattern: Option<String>,
    /// `grep`: lines either side of each hit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(max = 5))]
    pub context: Option<u32>,
    /// `lines`: the first line; `grep`: search from this line on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub from: Option<u64>,
    /// `lines`: how many lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1, max = 200))]
    pub count: Option<u32>,
    /// `tail`: how many of the last lines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1, max = 200))]
    pub lines: Option<u32>,
}

pub struct JobLog;

impl Tool for JobLog {
    const NAME: &'static str = "repo_job_log";
    const DESCRIPTION: &'static str =
        "Read what a build-agent job printed — summary, tail, a range of lines, or a search — read on \
         the agent, every line numbered.";

    type Request = LogRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: LogRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &LogRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct ListRequest {
    /// One agent only: its machine or node. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(length(min = 1, max = 32))]
    pub node: Option<String>,
    /// Only jobs not yet ended. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub running: Option<bool>,
    /// The most jobs listed per agent. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1, max = 100))]
    pub limit: Option<u32>,
}

pub struct JobList;

impl Tool for JobList {
    const NAME: &'static str = "repo_job_list";
    const DESCRIPTION: &'static str =
        "The jobs on the build agents, newest first, each by its handle, with where it is and what it \
         runs.";

    type Request = ListRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: ListRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &ListRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct CancelRequest {
    /// The job's handle, `<node>:<id>`. Required.
    #[validate(length(min = 18, max = 64))]
    pub job: String,
}

pub struct JobCancel;

impl Tool for JobCancel {
    const NAME: &'static str = "repo_job_cancel";
    const DESCRIPTION: &'static str =
        "Stop a build-agent job at whatever phase it is in; one that has ended is left as it ended.";

    type Request = CancelRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: CancelRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

pub const REPO_JOB_STATUS: RegisteredTool = RegisteredTool::new::<JobStatus>();
pub const REPO_JOB_LOG: RegisteredTool = RegisteredTool::new::<JobLog>();
pub const REPO_JOB_LIST: RegisteredTool = RegisteredTool::new::<JobList>();
pub const REPO_JOB_CANCEL: RegisteredTool = RegisteredTool::new::<JobCancel>();
