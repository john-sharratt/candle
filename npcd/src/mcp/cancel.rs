//! `npc_mission_cancel {npc}`: call off a character's open mission —
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
    const NAME: &'static str = "npc_mission_cancel";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Call off the mission a Battle Cities character is carrying. The \
                character is told it was withdrawn; nothing is kept of it. Missions queued for it \
                are untouched.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "npc": {"type": "string", "description": "The character's id, as npc_list gives it."}
                },
                "required": ["npc"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let npc = match string(args, "npc") {
            Ok(n) => n,
            Err(e) => return Answer::error(e),
        };
        let res =
            mission_api::cancel(State(Arc::clone(s)), Path(npc.to_string()), headers.clone()).await;
        Answer::of(read::<CancelReply>(res).await.map(|r| {
            if r.cancelled {
                format!("Character {npc}'s mission is called off; it has been told.")
            } else {
                format!("Character {npc} was carrying no mission.")
            }
        }))
    }
}
