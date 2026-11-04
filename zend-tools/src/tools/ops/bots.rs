//! The bots' and the pictures' tools (`docs/zen_tool_reach.md` §4.3, §6.4):
//! npcd's own, reached through the daemon's runner, which sends each call to
//! npcd's MCP server through the gateway as the conversation's person. npcd
//! is the only implementation; these are the arguments Zen writes, each a
//! subset of npcd's own schema (`motherd/tests/bot_schemas.rs` holds the two
//! together).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::{forward, OpsError, Report};
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// A tool whose request is sent to npcd whole, under its name.
macro_rules! npcd_tool {
    ($tool:ident, $reg:ident, $name:literal, $desc:literal, $req:ty, $replay:expr) => {
        pub struct $tool;

        impl Tool for $tool {
            const NAME: &'static str = $name;
            const DESCRIPTION: &'static str = $desc;

            type Request = $req;
            type Response = Report;
            type Error = OpsError;

            fn run(ctx: &ToolContext, req: $req) -> Result<Report, OpsError> {
                forward(ctx, Self::NAME, &req)
            }

            fn replay(_req: &$req) -> Replay {
                $replay
            }
        }

        pub const $reg: RegisteredTool = RegisteredTool::new::<$tool>();
    };
}

/// A bot's state, as a listing filters by it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum BotState {
    Active,
    Idle,
    Asleep,
    Suspended,
    Any,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct ListRequest {
    /// Only bots in this state. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "BotState")]
    pub state: Option<BotState>,
    /// Only bots in this world. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub world: Option<String>,
    /// Only bots whose name holds this text. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub q: Option<String>,
}

/// A request naming one bot.
#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct BotRequest {
    /// The bot's id, as bot_list gives it. Required.
    #[validate(length(min = 1, max = 64))]
    pub bot: String,
}

/// A request naming one world.
#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct WorldRequest {
    /// The world's id. Required.
    #[validate(length(min = 1, max = 64))]
    pub world: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct ThreadReadRequest {
    /// The bot's id. Required.
    #[validate(length(min = 1, max = 64))]
    pub bot: String,
    /// A thread's id or name; left out, the threads are listed. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub thread: Option<String>,
    /// How many of its last messages. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 200))]
    pub limit: Option<u32>,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct GiveRequest {
    /// The bot's id. Required.
    #[validate(length(min = 1, max = 64))]
    pub bot: String,
    /// What is asked of the bot. Required.
    #[validate(length(min = 1))]
    pub brief: String,
    /// The steps, in order, each addressed to the bot. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Vec<String>")]
    pub steps: Option<Vec<String>>,
    /// Carry it now (true, the default) or queue it (false). Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "bool")]
    pub start: Option<bool>,
}

/// How a signed-off step went.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Achieved,
    Thwarted,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct StepRequest {
    /// The bot's id. Required.
    #[validate(length(min = 1, max = 64))]
    pub bot: String,
    /// The step to sign off. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub done: Option<String>,
    /// How it went; achieved by default. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Outcome")]
    pub outcome: Option<Outcome>,
    /// A step to add. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub add: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct MessageRequest {
    /// The bot's id. Required.
    #[validate(length(min = 1, max = 64))]
    pub bot: String,
    /// What to say. Required.
    #[validate(length(min = 1))]
    pub text: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct PostRequest {
    /// The world's id. Required.
    #[validate(length(min = 1, max = 64))]
    pub world: String,
    /// What to say. Required.
    #[validate(length(min = 1))]
    pub text: String,
}

/// How a picture comes back.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    Jpeg,
    Png,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct DrawRequest {
    /// What to draw. Required.
    #[validate(length(min = 1))]
    pub prompt: String,
    /// Width in pixels, a multiple of 16. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 16, max = 1024))]
    pub width: Option<u32>,
    /// Height in pixels, a multiple of 16. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 16, max = 1024))]
    pub height: Option<u32>,
    /// Denoise steps. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    #[validate(range(min = 1, max = 100))]
    pub steps: Option<u32>,
    /// Seed, for repeating a draw. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u64")]
    pub seed: Option<u64>,
    /// jpeg or png. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Format")]
    pub format: Option<Format>,
}

/// What an edit does to a picture.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EditMode {
    Prompt,
    Resize,
    Adjust,
    Rotate,
    RemoveBackground,
}

/// A mirror.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Flip {
    Horizontal,
    Vertical,
}

#[derive(Serialize, Deserialize, JsonSchema, Validate)]
pub struct EditRequest {
    /// The picture: a link or image id an earlier answer gave. Required.
    #[validate(length(min = 1))]
    pub image: String,
    /// The edit; prompt by default. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "EditMode")]
    pub mode: Option<EditMode>,
    /// prompt: the whole picture as it should end up. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub prompt: Option<String>,
    /// prompt or resize: width in pixels. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub width: Option<u32>,
    /// prompt or resize: height in pixels. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub height: Option<u32>,
    /// rotate: degrees clockwise, 90, 180 or 270. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub rotate: Option<u32>,
    /// rotate: a mirror. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Flip")]
    pub flip: Option<Flip>,
    /// jpeg or png. Optional.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Format")]
    pub format: Option<Format>,
}

npcd_tool!(
    BotList,
    BOT_LIST,
    "bot_list",
    "List your bots — autonomous agents, each living in a world — with their ids, state and world.",
    ListRequest,
    Replay::Safe
);
npcd_tool!(
    BotMissionRead,
    BOT_MISSION_READ,
    "bot_mission_read",
    "What a bot is carrying now, its steps, and how its last mission went.",
    BotRequest,
    Replay::Safe
);
npcd_tool!(
    BotMessageRead,
    BOT_MESSAGE_READ,
    "bot_message_read",
    "Your message thread with a bot, newest last.",
    BotRequest,
    Replay::Safe
);
npcd_tool!(
    BotChannelRead,
    BOT_CHANNEL_READ,
    "bot_channel_read",
    "What has been said on a world's channel, newest last.",
    WorldRequest,
    Replay::Safe
);
npcd_tool!(
    BotDiscordRead,
    BOT_DISCORD_READ,
    "bot_discord_read",
    "What a bot's Discord bridge has done: what it posted and read.",
    BotRequest,
    Replay::Safe
);
npcd_tool!(
    BotThreadRead,
    BOT_THREAD_READ,
    "bot_thread_read",
    "A bot's handset threads, or one thread's last messages.",
    ThreadReadRequest,
    Replay::Safe
);
npcd_tool!(
    BotMissionGive,
    BOT_MISSION_GIVE,
    "bot_mission_give",
    "Give a bot a mission: a brief, and its steps in order.",
    GiveRequest,
    Replay::Unsafe
);
npcd_tool!(
    BotMissionStep,
    BOT_MISSION_STEP,
    "bot_mission_step",
    "Sign off a step of a bot's mission, or add one.",
    StepRequest,
    Replay::Unsafe
);
npcd_tool!(
    BotMissionCancel,
    BOT_MISSION_CANCEL,
    "bot_mission_cancel",
    "Call off a bot's mission.",
    BotRequest,
    Replay::Unsafe
);
npcd_tool!(
    BotMessageSend,
    BOT_MESSAGE_SEND,
    "bot_message_send",
    "Message a bot on its handset.",
    MessageRequest,
    Replay::Unsafe
);
npcd_tool!(
    BotChannelPost,
    BOT_CHANNEL_POST,
    "bot_channel_post",
    "Speak on a world's channel, where every bot and person in it reads.",
    PostRequest,
    Replay::Unsafe
);
npcd_tool!(
    ImageGenerate,
    IMAGE_GENERATE,
    "image_generate",
    "Draw a picture with the user's own image model; answers a link to the file.",
    DrawRequest,
    Replay::Unsafe
);
npcd_tool!(
    ImageEdit,
    IMAGE_EDIT,
    "image_edit",
    "Change a picture by a prompt, or resize, adjust, rotate it or remove its background.",
    EditRequest,
    Replay::Unsafe
);
