//! `service_deploy` tool — builds a service from `origin/main` and tries it on
//! one machine through the injected [`TrialRunner`], answering how it ended.
//!
//! [`TrialRunner`]: crate::TrialRunner

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use validator::Validate;

use crate::{RegisteredTool, Tool, ToolContext, ToolError, TrialRequest};

#[derive(Debug, Error)]
pub enum DeployError {
    #[error("this conversation cannot deploy: no runner is configured for it")]
    NotConfigured,
    #[error("{0}")]
    Refused(String),
}

impl ToolError for DeployError {
    fn code(&self) -> &'static str {
        match self {
            DeployError::NotConfigured => "not_configured",
            DeployError::Refused(_) => "deploy_refused",
        }
    }
}

/// The services keep runs.
#[derive(Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Service {
    Motherd,
    Web,
    Mcp,
}

impl Service {
    fn name(self) -> &'static str {
        match self {
            Service::Motherd => "motherd",
            Service::Web => "web",
            Service::Mcp => "mcp",
        }
    }
}

#[derive(Deserialize, JsonSchema, Validate)]
pub struct Request {
    /// The machine to try it on, by name, as `blackworld`. Required.
    #[validate(length(min = 1))]
    pub host: String,
    /// The service to build and try. Required.
    pub service: Service,
}

#[derive(Serialize)]
pub struct Response {
    pub host: String,
    pub service: &'static str,
    /// The build's path on that machine.
    pub artifact: String,
    /// True once it served and became the installed build; false when keep
    /// put the previous build back, which is still running.
    pub proven: bool,
    pub detail: String,
}

pub struct ServiceDeploy;

impl Tool for ServiceDeploy {
    const NAME: &'static str = "service_deploy";
    const DESCRIPTION: &'static str =
        "Build a service as origin/main holds it on one machine's agent, then try it there \
         through keep, the supervisor that runs the machine's services, and wait for the outcome. \
         keep keeps the build once it reports serving, within ten minutes, and otherwise puts the \
         previous binary back. Only origin/main is built: your branch reaches it by merging, and a \
         branch is checked with run_command. Refused on the machine running this conversation, \
         and on a machine holding an app's only ready member. It tries one machine: rolling a \
         build out to the others is the user's decision. Use for: putting a merged build on a \
         machine. Triggered by \"deploy this\", \"try the build on whiteworld\". Returns the \
         artifact, whether it was proven or rolled back, and keep's account. Takes many minutes.";

    type Request = Request;
    type Response = Response;
    type Error = DeployError;

    fn run(ctx: &ToolContext, req: Request) -> Result<Response, DeployError> {
        let runner = ctx
            .trial_runner
            .as_deref()
            .ok_or(DeployError::NotConfigured)?;
        let service = req.service.name();
        runner
            .run(
                ctx.lineage(),
                TrialRequest {
                    host: req.host.clone(),
                    service,
                },
            )
            .map(|r| Response {
                host: req.host,
                service,
                artifact: r.artifact,
                proven: r.proven,
                detail: r.detail,
            })
            .map_err(DeployError::Refused)
    }
}

pub const REGISTRATION: RegisteredTool = RegisteredTool::new::<ServiceDeploy>();
