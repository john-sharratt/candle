//! journal_create, journal_link: a new memory journal, and a reference to one
//! from where a reader would look (`docs/memory_backbone.md` §2.2), written
//! as the person the conversation is with (`docs/zen_tool_reach.md` §6.2).
//! Both go to the daemon's runner, which writes them through the memory —
//! or holds them for that person's approval when the turn has read content
//! from outside.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::tools::ops::{forward, OpsError, Report};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct CreateRequest {
    /// Where it goes: `topics`, `people`, or a journal below one. Required.
    #[validate(length(min = 1, max = 200))]
    pub under: String,
    /// A kebab-case slug of at most 40 characters. Required.
    #[validate(length(min = 1, max = 40))]
    pub name: String,
    /// One line: when to open it. Required.
    #[validate(length(min = 1, max = 300))]
    pub cue: String,
}

pub struct JournalCreate;

impl Tool for JournalCreate {
    const NAME: &'static str = "journal_create";
    const DESCRIPTION: &'static str =
        "Open a new memory journal for something worth following on its own — a topic, a \
         project, a person — under `topics` or `people`, with a one-line cue for when to open it.";

    type Request = CreateRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: CreateRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct LinkRequest {
    /// The journal the reference goes in. Required.
    #[validate(length(min = 1, max = 200))]
    pub under: String,
    /// The reference's own name, a kebab-case slug. Required.
    #[validate(length(min = 1, max = 40))]
    pub name: String,
    /// The journal it leads to. Required.
    #[validate(length(min = 1, max = 200))]
    pub to: String,
    /// One line: when to follow it. Required.
    #[validate(length(min = 1, max = 300))]
    pub cue: String,
}

pub struct JournalLink;

impl Tool for JournalLink {
    const NAME: &'static str = "journal_link";
    const DESCRIPTION: &'static str =
        "Put a reference to a memory journal in another, so a reader there finds it: a person's \
         journal leading to a topic they work on, a topic to a related one.";

    type Request = LinkRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: LinkRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }
}

pub const JOURNAL_CREATE: RegisteredTool = RegisteredTool::new::<JournalCreate>();
pub const JOURNAL_LINK: RegisteredTool = RegisteredTool::new::<JournalLink>();
