//! `service_deploy`, `service_status`, `service_log`, `service_cancel`: a
//! service deployed to one machine and watched to its end
//! (`docs/build_deploy_tools.md` §4).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{forward, LogMode, OpsError, Report, Service};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct DeployRequest {
    /// The machine to deploy to, by name, as `blackworld`. Required.
    #[validate(length(min = 1, max = 32))]
    pub host: String,
    /// The service to build and try. Required.
    pub service: Service,
}

pub struct ServiceDeploy;

impl Tool for ServiceDeploy {
    const NAME: &'static str = "service_deploy";
    const DESCRIPTION: &'static str =
        "Deploy a service to one machine: build it as origin/main holds it on that machine's agent, \
         then try it through keep, which keeps it once it serves and otherwise puts the previous \
         binary back. Answers at once with the deploy's handle; watch it with service_status.";

    type Request = DeployRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: DeployRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct StatusRequest {
    /// A deploy's handle, `<node>:<id>`. Optional: without it, every
    /// machine's services.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(length(min = 18, max = 64))]
    pub deploy: Option<String>,
    /// Without `deploy`: one machine only. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(length(min = 1, max = 32))]
    pub host: Option<String>,
    /// With `deploy`: seconds to wait for it to end. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(max = 3600))]
    pub wait_s: Option<u32>,
}

pub struct ServiceStatus;

impl Tool for ServiceStatus {
    const NAME: &'static str = "service_status";
    const DESCRIPTION: &'static str =
        "Where a deploy is — building, tried by keep, proven or rolled back — waiting up to wait_s for \
         it to end; or, with no deploy, every machine's services and trials.";

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
pub struct ServiceLogRequest {
    /// The machine, by name. Required.
    #[validate(length(min = 1, max = 32))]
    pub host: String,
    /// The service whose trial log to read. Required.
    pub service: Service,
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

pub struct ServiceLog;

impl Tool for ServiceLog {
    const NAME: &'static str = "service_log";
    const DESCRIPTION: &'static str =
        "Read a service's trial log on one machine — summary, tail, a range of lines, or a search — \
         read where keep writes it, every line numbered.";

    type Request = ServiceLogRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: ServiceLogRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &ServiceLogRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct CancelRequest {
    /// The deploy's handle, `<node>:<id>`. Required.
    #[validate(length(min = 18, max = 64))]
    pub deploy: String,
}

pub struct ServiceCancel;

impl Tool for ServiceCancel {
    const NAME: &'static str = "service_cancel";
    const DESCRIPTION: &'static str =
        "Stop a deploy: its build while it builds, or keep's trial, which is rolled back.";

    type Request = CancelRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: CancelRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

pub const SERVICE_DEPLOY: RegisteredTool = RegisteredTool::new::<ServiceDeploy>();
pub const SERVICE_STATUS: RegisteredTool = RegisteredTool::new::<ServiceStatus>();
pub const SERVICE_LOG: RegisteredTool = RegisteredTool::new::<ServiceLog>();
pub const SERVICE_CANCEL: RegisteredTool = RegisteredTool::new::<ServiceCancel>();
