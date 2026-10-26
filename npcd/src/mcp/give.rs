//! `bot_mission_give {bot, brief, steps?, start?}`: give a bot a mission —
//! `POST /v1/npc/:nid/mission`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde_json::{json, Value};

use super::args::{flag, string, strings};
use super::route::read;
use super::{Answer, Tool};
use crate::api::Authored;
use crate::engine::bodies::{LodgeReply, LodgeRequest};
use crate::engine::mission_api;

pub struct Give;

impl Tool for Give {
    const NAME: &'static str = "bot_mission_give";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Give a bot a mission: a `brief` saying what is asked of it, and the `steps` \
                in order. With no steps, steps are drawn from the rooms, machines and people the brief \
                names, plus reporting back. By default the bot carries the mission at once and is roused \
                to read it now, replacing any mission it was carrying; with `start` false it is queued \
                instead, for the bot to collect at the command table — the place in its world where \
                bots take up missions. Follow it with bot_mission_read.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bot": {"type": "string", "description": "The bot's id, as bot_list gives it."},
                    "brief": {"type": "string", "description": "What is asked of the bot."},
                    "steps": {"type": "array", "items": {"type": "string"}, "description": "The steps, in order, each addressed to the bot (\"go to the plant room\")."},
                    "start": {"type": "boolean", "description": "Carry it now (true, the default) or queue it at the command table (false)."}
                },
                "required": ["bot", "brief"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let (bot, brief) = match (string(args, "bot"), string(args, "brief")) {
            (Ok(b), Ok(t)) => (b, t),
            (Err(e), _) | (_, Err(e)) => return Answer::error(e),
        };
        let lodge = LodgeRequest {
            prompt: brief.to_string(),
            todo: strings(args, "steps"),
            start: flag(args, "start", true),
        };
        let res = mission_api::lodge(
            State(Arc::clone(s)),
            Path(bot.to_string()),
            headers.clone(),
            Json(lodge),
        )
        .await;
        Answer::of(read::<LodgeReply>(res).await.map(|r| render(bot, &r)))
    }
}

/// What giving the mission did.
pub fn render(bot: &str, r: &LodgeReply) -> String {
    let steps = match r.steps {
        0 => "no steps".to_string(),
        1 => "1 step".to_string(),
        n => format!("{n} steps"),
    };
    if r.carrying {
        format!("Bot {bot} is carrying it now ({steps}): {}", r.prompt)
    } else {
        format!(
            "Queued for bot {bot} to collect at the command table ({steps}): {}",
            r.prompt
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_says_whether_the_mission_is_carried_or_queued() {
        let r = LodgeReply {
            lodged: false,
            carrying: true,
            prompt: "Check the valve.".into(),
            steps: 3,
        };
        assert_eq!(
            render("1z", &r),
            "Bot 1z is carrying it now (3 steps): Check the valve."
        );
        let q = LodgeReply {
            lodged: true,
            carrying: false,
            steps: 1,
            ..r
        };
        assert_eq!(
            render("1z", &q),
            "Queued for bot 1z to collect at the command table (1 step): Check the valve."
        );
    }
}
