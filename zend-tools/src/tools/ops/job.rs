//! `job_list`, `job_status`, `job_read`, `job_grep`, `job_cancel`: whatever
//! outlives a call — a task, a build or test run, a deploy, a maintenance job —
//! listed, waited on, paged, searched and stopped by its handle
//! (`docs/jobs.md`). A handle is `task:<thread>/<name>`, `build:<node>:<id>`,
//! `deploy:<node>:<id>` or `cluster:<node>:<n>`, exactly as an answer wrote it.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{forward, OpsError, Report};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// A kind of job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Task,
    Build,
    Deploy,
    Cluster,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct ListRequest {
    /// Only jobs of one kind. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Kind")]
    pub kind: Option<Kind>,
    /// Only jobs still running; false adds those that ended lately. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "bool")]
    pub running: Option<bool>,
    /// The most jobs listed. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 100))]
    pub limit: Option<u32>,
}

pub struct JobList;

impl Tool for JobList {
    const NAME: &'static str = "job_list";
    const DESCRIPTION: &'static str =
        "Every job you can see — tasks, builds and test runs, deploys, maintenance jobs — newest \
         first, a line each with its handle, where it is and how many pages of output it has.";

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
pub struct StatusRequest {
    /// The jobs to wait on, by handle. Required.
    #[validate(length(min = 1, max = 16))]
    pub jobs: Vec<String>,
    /// Seconds to wait for one of them to end, at most 300 — a worker's
    /// ceiling. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(max = 300))]
    pub wait_s: Option<u32>,
}

pub struct JobStatus;

impl Tool for JobStatus {
    const NAME: &'static str = "job_status";
    const DESCRIPTION: &'static str =
        "Where each job is, waiting up to wait_s for any one of them to end; an ended job's answer \
         carries how it went — test results and failures by line, a task's answer — and its pages.";

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
pub struct ReadRequest {
    /// The job's handle. Required.
    #[validate(length(min = 8, max = 200))]
    pub job: String,
    /// The page, from 1. Required.
    #[validate(range(min = 1))]
    pub page: u64,
    /// Seconds to wait for the page to fill. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(max = 300))]
    pub wait_s: Option<u32>,
}

pub struct JobRead;

impl Tool for JobRead {
    const NAME: &'static str = "job_read";
    const DESCRIPTION: &'static str =
        "One page — 200 numbered lines — of a job's output; a page still being written waits until \
         it fills, the job ends or wait_s passes.";

    type Request = ReadRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: ReadRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &ReadRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct GrepRequest {
    /// The job's handle. Required.
    #[validate(length(min = 8, max = 200))]
    pub job: String,
    /// The text to find, ignoring case; plain text. Required.
    #[validate(length(min = 1, max = 200))]
    pub pattern: String,
    /// Lines either side of each hit. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(max = 5))]
    pub context: Option<u32>,
}

pub struct JobGrep;

impl Tool for JobGrep {
    const NAME: &'static str = "job_grep";
    const DESCRIPTION: &'static str =
        "Search a job's whole output for plain text, each hit with its line number and its page.";

    type Request = GrepRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: GrepRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &GrepRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct CancelRequest {
    /// The job's handle. Required.
    #[validate(length(min = 8, max = 200))]
    pub job: String,
}

pub struct JobCancel;

impl Tool for JobCancel {
    const NAME: &'static str = "job_cancel";
    const DESCRIPTION: &'static str =
        "Stop a job: a build at whatever phase it is in, a deploy's build or its trial (rolled \
         back), a task at its next tool round. One that has ended is left as it ended.";

    type Request = CancelRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: CancelRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

pub const JOB_LIST: RegisteredTool = RegisteredTool::new::<JobList>();
pub const JOB_STATUS: RegisteredTool = RegisteredTool::new::<JobStatus>();
pub const JOB_READ: RegisteredTool = RegisteredTool::new::<JobRead>();
pub const JOB_GREP: RegisteredTool = RegisteredTool::new::<JobGrep>();
pub const JOB_CANCEL: RegisteredTool = RegisteredTool::new::<JobCancel>();
