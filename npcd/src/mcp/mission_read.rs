//! `bot_mission_read {bot}`: what a bot is carrying, and how its last mission
//! went — `GET /v1/npc/:nid/mission`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::args::string;
use super::route::read;
use super::{view, Answer, Tool};
use crate::api::Authored;
use crate::engine::bodies::MissionStatus;
use crate::engine::mission_api;

pub struct Mission;

impl Tool for Mission {
    const NAME: &'static str = "bot_mission_read";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Read a bot's missions: the one it is carrying — its brief, each step and \
                whether it was achieved or thwarted — and the last one it finished, with its answer and \
                its report (pass or fail, and its account), and how many more are queued for it. Call \
                it again to follow a mission as the bot works.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bot": {"type": "string", "description": "The bot's id, as bot_list gives it."}
                },
                "required": ["bot"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let bot = match string(args, "bot") {
            Ok(b) => b,
            Err(e) => return Answer::error(e),
        };
        let res =
            mission_api::status(State(Arc::clone(s)), Path(bot.to_string()), headers.clone()).await;
        Answer::of(
            read::<MissionStatus>(res)
                .await
                .map(|st| view::status(bot, &st)),
        )
    }
}
