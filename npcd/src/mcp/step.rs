//! `bot_mission_step {bot, done?, outcome?, add?}`: sign off or add a step on
//! a bot's open mission — `POST /v1/npc/:nid/mission/step`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde_json::{json, Value};

use super::args::{optional, string};
use super::route::read;
use super::{Answer, Tool};
use crate::api::Authored;
use crate::engine::bodies::{StepReply, StepRequest};
use crate::engine::mission::StepOutcome;
use crate::engine::mission_api;

pub struct Step;

impl Tool for Step {
    const NAME: &'static str = "bot_mission_step";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Change the steps of the mission a bot is carrying: sign off a step with \
                `done` (its text, or the start of it) as achieved — or, with `outcome` thwarted, as \
                failed — or add a step with `add`. The bot reads its steps again before it next acts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bot": {"type": "string", "description": "The bot's id, as bot_list gives it."},
                    "done": {"type": "string", "description": "The step to sign off."},
                    "outcome": {"type": "string", "enum": ["achieved", "thwarted"], "description": "How it went (achieved by default)."},
                    "add": {"type": "string", "description": "A step to add."}
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
        let outcome = match optional(args, "outcome") {
            None | Some("achieved") => None,
            Some("thwarted") => Some(StepOutcome::Thwarted),
            Some(other) => {
                return Answer::error(format!("`outcome` is achieved or thwarted, not `{other}`"))
            }
        };
        let req = StepRequest {
            done: optional(args, "done").map(str::to_string),
            outcome,
            add: optional(args, "add").map(str::to_string),
        };
        let res = mission_api::step(
            State(Arc::clone(s)),
            Path(bot.to_string()),
            headers.clone(),
            Json(req.clone()),
        )
        .await;
        Answer::of(read::<StepReply>(res).await.map(|r| render(bot, &req, &r)))
    }
}

/// What the step request did.
pub fn render(bot: &str, req: &StepRequest, r: &StepReply) -> String {
    let mut said = Vec::new();
    if let Some(done) = &req.done {
        said.push(match r.ticked {
            Some(true) => format!(
                "signed off \"{done}\" as {}",
                if req.outcome == Some(StepOutcome::Thwarted) {
                    "thwarted"
                } else {
                    "achieved"
                }
            ),
            _ => format!("no open step matches \"{done}\""),
        });
    }
    if let Some(add) = &req.add {
        said.push(match r.added {
            Some(true) => format!("added \"{add}\""),
            _ => format!("could not add \"{add}\" — it has no open mission"),
        });
    }
    format!("Bot {bot}: {}.", said.join("; "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_says_what_was_signed_off_and_added() {
        let req = StepRequest {
            done: Some("scan the valve".into()),
            outcome: Some(StepOutcome::Thwarted),
            add: Some("find Pax".into()),
        };
        assert_eq!(
            render(
                "1z",
                &req,
                &StepReply {
                    ticked: Some(true),
                    added: Some(true)
                }
            ),
            "Bot 1z: signed off \"scan the valve\" as thwarted; added \"find Pax\"."
        );
        assert_eq!(
            render(
                "1z",
                &req,
                &StepReply {
                    ticked: Some(false),
                    added: None
                }
            ),
            "Bot 1z: no open step matches \"scan the valve\"; could not add \"find Pax\" — it has \
             no open mission."
        );
    }
}
