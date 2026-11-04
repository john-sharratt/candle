//! `repo_task_start`, `repo_task_reply`, `repo_task_nudge`: the tasks a
//! conversation starts and steers (`docs/zen_tool_reach.md` §6.3). A task's
//! own tool list never holds them, so a task does all of its work itself and
//! never starts another. Each answers at once with the task's handle; its
//! answer comes from `job_status`.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{forward, OpsError, Report};
use crate::{RegisteredTool, Tool, ToolContext};

/// The kind of work a task does.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Coder,
    Researcher,
    Reviewer,
}

/// How hard a task thinks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Balanced,
    Deep,
    Exhaustive,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct StartRequest {
    /// The task's name, a kebab-case slug saying what it does. Required.
    #[validate(length(min = 1, max = 40))]
    pub name: String,
    /// The task, stated in full. Required.
    #[validate(length(min = 1))]
    pub prompt: String,
    /// The kind of work it does; `coder` when left out. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Role")]
    pub role: Option<Role>,
    /// How hard it thinks; `deep` when left out. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Effort")]
    pub effort: Option<Effort>,
}

pub struct RepoTaskStart;

impl Tool for RepoTaskStart {
    const NAME: &'static str = "repo_task_start";
    const DESCRIPTION: &'static str =
        "Start a task on a worker of its own — research, coding or a review — stated in full; it \
         runs on its own and answers at once with its handle.";

    type Request = StartRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: StartRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct ReplyRequest {
    /// The task's handle, `task:<thread>/<name>`. Required.
    #[validate(length(min = 6, max = 200))]
    pub task: String,
    /// The reply: the decision, or what to do instead. Required.
    #[validate(length(min = 1))]
    pub answer: String,
}

pub struct RepoTaskReply;

impl Tool for RepoTaskReply {
    const NAME: &'static str = "repo_task_reply";
    const DESCRIPTION: &'static str =
        "Reply to a task that asked a question, is stuck, or finished: the reply is its next \
         message, and it carries on with everything it did still in its context.";

    type Request = ReplyRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: ReplyRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct NudgeRequest {
    /// The task's handle, `task:<thread>/<name>`. Required.
    #[validate(length(min = 6, max = 200))]
    pub task: String,
    /// The guidance, in a sentence or two. Required.
    #[validate(length(min = 1))]
    pub message: String,
}

pub struct RepoTaskNudge;

impl Tool for RepoTaskNudge {
    const NAME: &'static str = "repo_task_nudge";
    const DESCRIPTION: &'static str =
        "Steer a running task: the guidance reaches it when its next tool round ends.";

    type Request = NudgeRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: NudgeRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

pub const REPO_TASK_START: RegisteredTool = RegisteredTool::new::<RepoTaskStart>();
pub const REPO_TASK_REPLY: RegisteredTool = RegisteredTool::new::<RepoTaskReply>();
pub const REPO_TASK_NUDGE: RegisteredTool = RegisteredTool::new::<RepoTaskNudge>();
