//! `bot_mission_cancel {bot}`: call off a bot's open mission —
//! `POST /v1/npc/:nid/mission/cancel`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::args::string;
use super::route::read;
use super::{Answer, Tool};
use crate::api::Authored;
use crate::engine::bodies::CancelReply;
use crate::engine::mission_api;

pub struct Cancel;

impl Tool for Cancel {
    const NAME: &'static str = "bot_mission_cancel";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Call off the mission a bot is carrying. The bot is told it was withdrawn; \
                nothing is kept of it. Missions queued for it are untouched.",
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
            mission_api::cancel(State(Arc::clone(s)), Path(bot.to_string()), headers.clone()).await;
        Answer::of(read::<CancelReply>(res).await.map(|r| {
            if r.cancelled {
                format!("Bot {bot}'s mission is called off; it has been told.")
            } else {
                format!("Bot {bot} was carrying no mission.")
            }
        }))
    }
}
