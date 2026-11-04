//! `timeline_read`: one stored conversation read as `view` asks — its turns,
//! one turn in full, its tool calls with repeats flagged, a turn's projection
//! points with the tools each showed and left out, the prompt it would be
//! decoded with now, its latest dialogue selection, or how a text ranks the
//! tools. What a worker reviewing another run reads it with.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{forward, OpsError, Report};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// What a read shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum View {
    Turns,
    Turn,
    Calls,
    Projections,
    Prompt,
    Selection,
    Probe,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct TimelineReadRequest {
    /// The conversation: a timeline id, a worker's path `T:<thread>/<name>`,
    /// or a conversation id as the web UI's `#conv=` names it. Required.
    #[validate(length(min = 1, max = 200))]
    pub id: String,
    /// What to read. Required.
    pub view: View,
    /// `turn` and `projections`: the turn's index, as `turns` lists it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub turn: Option<u32>,
    /// `probe`: the text to rank the tools by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    #[validate(length(min = 1, max = 4000))]
    pub probe: Option<String>,
    /// `turns` and `calls`: the first turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub offset: Option<u32>,
    /// `turns` and `calls`: how many turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 400))]
    pub limit: Option<u32>,
    /// `turn`: the character to read from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub from: Option<u32>,
}

pub struct TimelineRead;

impl Tool for TimelineRead {
    const NAME: &'static str = "timeline_read";
    const DESCRIPTION: &'static str =
        "Read one stored conversation as `view` asks: its turns, one turn in full, its tool calls \
         with repeats flagged, a turn's projection points (the tools shown and left out, with \
         scores), its prompt now, its dialogue selection, or how a text ranks the tools.";

    type Request = TimelineReadRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: TimelineReadRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &TimelineReadRequest) -> Replay {
        Replay::Safe
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct LayerReadRequest {
    /// The layer: `repo_map`, `code_reading`, `dialogue`, … Required.
    #[validate(length(min = 1, max = 64))]
    pub name: String,
    /// The first conversation. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub offset: Option<u32>,
    /// How many conversations. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 500))]
    pub limit: Option<u32>,
}

pub struct LayerRead;

impl Tool for LayerRead {
    const NAME: &'static str = "layer_read";
    const DESCRIPTION: &'static str =
        "List the stored conversations one projection layer holds, a page at a time: each with its \
         timeline id, label, repository, turns and tokens, to read with `timeline_read`.";

    type Request = LayerReadRequest;
    type Response = Report;
    type Error = OpsError;

    fn run(ctx: &ToolContext, req: LayerReadRequest) -> Result<Report, OpsError> {
        forward(ctx, Self::NAME, &req)
    }

    fn replay(_req: &LayerReadRequest) -> Replay {
        Replay::Safe
    }
}

pub const TIMELINE_READ: RegisteredTool = RegisteredTool::new::<TimelineRead>();
pub const LAYER_READ: RegisteredTool = RegisteredTool::new::<LayerRead>();
