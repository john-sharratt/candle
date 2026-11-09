//! `service_deploy`, `service_status`: a service deployed to one machine — a
//! job, `deploy:<node>:<id>`, watched with the job tools — and what each
//! machine's keep runs (`docs/jobs.md` §6).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{forward, OpsError, Report, Service};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct DeployRequest {
    /// The machine to deploy to, by name, as `blackworld`. Optional: the
    /// machine the cluster builds on first when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(min = 1, max = 32))]
    pub host: Option<String>,
    /// The service to build and try. Required.
    pub service: Service,
}

pub struct ServiceDeploy;

impl Tool for ServiceDeploy {
    const NAME: &'static str = "service_deploy";
    const DESCRIPTION: &'static str =
        "Deploy a service to one machine: build it as origin/main holds it on that machine's agent, \
         then try it through keep, which keeps it once it serves and otherwise puts the previous \
         binary back. Answers at once with the deploy's handle; wait on it with job_status.";

    type Request = DeployRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: DeployRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct StatusRequest {
    /// One machine only. Optional: every machine when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(min = 1, max = 32))]
    pub host: Option<String>,
}

pub struct ServiceStatus;

impl Tool for ServiceStatus {
    const NAME: &'static str = "service_status";
    const DESCRIPTION: &'static str =
        "What each machine's keep runs — each service's build, whether it runs, the trial running \
         and how the last one ended — and the deploys building there.";

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

pub const SERVICE_DEPLOY: RegisteredTool = RegisteredTool::new::<ServiceDeploy>();
pub const SERVICE_STATUS: RegisteredTool = RegisteredTool::new::<ServiceStatus>();
