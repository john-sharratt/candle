//! `cluster_status`, `cluster_logs`, `cluster_audit`, `cluster_job_status`:
//! how the cluster stands, read as the person a worker works for — the members
//! and their builds, each one's place in an upgrade, faults, the members'
//! logs, what was done to the cluster, and a maintenance job's progress.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{forward, OpsError, Report};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct StatusRequest {}

pub struct ClusterStatus;

impl Tool for ClusterStatus {
    const NAME: &'static str = "cluster_status";
    const DESCRIPTION: &'static str =
        "How the cluster stands: each member's build and state, each app's floor, where each member \
         is in an upgrade, and the members in or near fault.";

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

/// The least level a log search keeps.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub enum Level {
    #[serde(rename = "TRACE")]
    Trace,
    #[serde(rename = "DEBUG")]
    Debug,
    #[serde(rename = "INFO")]
    Info,
    #[serde(rename = "WARN")]
    Warn,
    #[serde(rename = "ERROR")]
    Error,
}

/// The app a log search keeps the lines of.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Tenant {
    Zend,
    Npcd,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct LogsRequest {
    /// A regular expression found anywhere in a kept line. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(min = 1, max = 200))]
    pub grep: Option<String>,
    /// The least level kept. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Level")]
    pub level: Option<Level>,
    /// Only one app's lines. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Tenant")]
    pub tenant: Option<Tenant>,
    /// Only lines stamped at or after this RFC 3339 prefix. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(min = 4, max = 32))]
    pub since: Option<String>,
    /// The most lines per member. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 5000))]
    pub limit: Option<u32>,
    /// One member's node only. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(equal = 16))]
    pub node: Option<String>,
}

pub struct ClusterLogs;

impl Tool for ClusterLogs {
    const NAME: &'static str = "cluster_logs";
    const DESCRIPTION: &'static str =
        "Search the cluster members' logs: every member's matching lines, oldest first, narrowed by \
         level, app and time.";

    type Request = LogsRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: LogsRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &LogsRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct AuditRequest {
    /// How many entries. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 500))]
    pub entries: Option<u32>,
}

pub struct ClusterAudit;

impl Tool for ClusterAudit {
    const NAME: &'static str = "cluster_audit";
    const DESCRIPTION: &'static str =
        "What was done to the cluster, newest first: commands run, members joining and leaving, \
         floors raised, faults entered and left.";

    type Request = AuditRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: AuditRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &AuditRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct JobStatusRequest {
    /// The member that runs the job, by its node. Required.
    #[validate(length(equal = 16))]
    pub member: String,
    /// The job's id. Required.
    pub job: u64,
    /// Seconds to wait for it to end. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(max = 3600))]
    pub wait_s: Option<u32>,
}

pub struct ClusterJobStatus;

impl Tool for ClusterJobStatus {
    const NAME: &'static str = "cluster_job_status";
    const DESCRIPTION: &'static str =
        "How a maintenance job on a member stands — running, done or failed, its last steps — \
         waiting up to wait_s for it to end.";

    type Request = JobStatusRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: JobStatusRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &JobStatusRequest) -> Replay {
        Replay::Safe
    }
}

pub const CLUSTER_STATUS: RegisteredTool = RegisteredTool::new::<ClusterStatus>();
pub const CLUSTER_LOGS: RegisteredTool = RegisteredTool::new::<ClusterLogs>();
pub const CLUSTER_AUDIT: RegisteredTool = RegisteredTool::new::<ClusterAudit>();
pub const CLUSTER_JOB_STATUS: RegisteredTool = RegisteredTool::new::<ClusterJobStatus>();
