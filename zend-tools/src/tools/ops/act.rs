//! The operations a conversation of Zen's acts on the cluster with
//! (`docs/zen_tool_reach.md` §6.1): a build on a named machine, a member's
//! maintenance command, a service's restart, start or stop, a layer's switch,
//! the gateway's floor commands and a timeline's delete.
//!
//! None carries a `confirm`. A call the daemon's runner judges a decision — a
//! Decide command, a stop, a delete, a step that takes a member out of service
//! or that the deploy guard refuses — is not run: it is held for a person's
//! approval and answers `needs_approval` with the approval's id. A person
//! grants it from outside the conversation, and only then does it run.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::{forward, OpsError, Report, Service};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct BuildRequest {
    /// The repository, by name: `candle`, `battle-cities`. Required.
    #[validate(length(min = 1, max = 64))]
    pub repo: String,
    /// The branch on origin to run on: `main`, or one pushed. Required.
    #[validate(length(min = 1, max = 200))]
    pub branch: String,
    /// The program alone, e.g. `cargo`. Required.
    #[validate(length(min = 1, max = 64))]
    pub program: String,
    /// Its arguments, one entry each. Required.
    #[validate(length(max = 48))]
    pub args: Vec<String>,
    /// Seconds before it is killed. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 14400))]
    pub timeout_s: Option<u32>,
    /// A file it builds, copied to that machine's stage folder. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(min = 1, max = 200))]
    pub stage: Option<String>,
    /// The machine to run on, or its node. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(min = 1, max = 32))]
    pub node: Option<String>,
}

pub struct RepoJobStart;

impl Tool for RepoJobStart {
    const NAME: &'static str = "repo_job_start";
    const DESCRIPTION: &'static str =
        "Start a program — a build, a test run, clippy — on a build agent, against a branch as \
         origin holds it, on a machine named or the one the gateway picks. Answers at once with the \
         job's handle; wait on it with job_status.";

    type Request = BuildRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: BuildRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct CommandRequest {
    /// The member's node id, 16 hex digits. Optional: not used for `rollout`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(equal = 16))]
    pub member: Option<String>,
    /// The command. Required.
    #[validate(length(min = 1, max = 32))]
    pub name: String,
    /// The command's own arguments. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
}

pub struct ClusterCommand;

impl Tool for ClusterCommand {
    const NAME: &'static str = "cluster_command";
    const DESCRIPTION: &'static str =
        "Run one maintenance command on one member. A Look runs; a Care runs for a conversation \
         that may change things; a Decide, or a step that takes a member out of service, is held \
         for a person's approval.";

    type Request = CommandRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: CommandRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

/// What a service is asked to do.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Action {
    Restart,
    Start,
    Stop,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct ControlRequest {
    /// The machine, by name. Required.
    #[validate(length(min = 1, max = 32))]
    pub host: String,
    /// The service. Required.
    pub service: Service,
    /// What to do to it. Required.
    pub action: Action,
}

pub struct ServiceControl;

impl Tool for ServiceControl {
    const NAME: &'static str = "service_control";
    const DESCRIPTION: &'static str =
        "Restart or start a service on one machine through keep; a stop is held for a person's \
         approval.";

    type Request = ControlRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: ControlRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct ToggleRequest {
    /// The layer, as `layer_read` names it. Required.
    #[validate(length(min = 1, max = 64))]
    pub name: String,
}

pub struct LayerToggle;

impl Tool for LayerToggle {
    const NAME: &'static str = "layer_toggle";
    const DESCRIPTION: &'static str =
        "Switch one projection layer off, or back on, on every member, until the next restart.";

    type Request = ToggleRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: ToggleRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

/// The gateway's own commands.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum GatewayName {
    RaiseFloor,
    CancelFloor,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct GatewayRequest {
    /// The command. Required.
    pub name: GatewayName,
    /// `{app}`. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Value>,
}

pub struct GatewayCommand;

impl Tool for GatewayCommand {
    const NAME: &'static str = "gateway_command";
    const DESCRIPTION: &'static str =
        "Raise an app's floor now, or cancel a pending rise: a decision, held for a person's \
         approval.";

    type Request = GatewayRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: GatewayRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct DeleteRequest {
    /// The timeline id, a decimal number as `layer_read` lists it. Required.
    #[validate(length(min = 1, max = 24))]
    pub id: String,
}

pub struct TimelineDelete;

impl Tool for TimelineDelete {
    const NAME: &'static str = "timeline_delete";
    const DESCRIPTION: &'static str =
        "Delete one stored conversation on every member: a decision, held for a person's \
         approval.";

    type Request = DeleteRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: DeleteRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

pub const REPO_JOB_START: RegisteredTool = RegisteredTool::new::<RepoJobStart>();
pub const CLUSTER_COMMAND: RegisteredTool = RegisteredTool::new::<ClusterCommand>();
pub const SERVICE_CONTROL: RegisteredTool = RegisteredTool::new::<ServiceControl>();
pub const LAYER_TOGGLE: RegisteredTool = RegisteredTool::new::<LayerToggle>();
pub const GATEWAY_COMMAND: RegisteredTool = RegisteredTool::new::<GatewayCommand>();
pub const TIMELINE_DELETE: RegisteredTool = RegisteredTool::new::<TimelineDelete>();

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A request reaches the runner as the tool's own arguments**: no
    /// `confirm` is sent, and an enum is spelled as the daemon's tool reads it.
    #[test]
    fn a_request_is_sent_as_the_daemons_tool_reads_it() {
        let control = ControlRequest {
            host: "BlackWorld".into(),
            service: Service::Motherd,
            action: Action::Restart,
        };
        assert_eq!(
            serde_json::to_value(&control).unwrap(),
            json!({"host": "BlackWorld", "service": "motherd", "action": "restart"})
        );
        let floor = GatewayRequest {
            name: GatewayName::RaiseFloor,
            args: Some(json!({"app": "zend"})),
        };
        assert_eq!(
            serde_json::to_value(&floor).unwrap(),
            json!({"name": "raise-floor", "args": {"app": "zend"}})
        );
        let command = CommandRequest {
            member: Some("a0ac88178cb2b57d".into()),
            name: "audit".into(),
            args: None,
        };
        assert_eq!(
            serde_json::to_value(&command).unwrap(),
            json!({"member": "a0ac88178cb2b57d", "name": "audit"})
        );
    }
}
