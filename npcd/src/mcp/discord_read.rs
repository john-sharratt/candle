//! `bot_discord_read {bot}`: what a bot has done on Discord — its state, its
//! shard, its DMs, channels and servers with how far each is posted, and
//! its last events (`docs/npcd_discord.md` §5.5). The trace
//! `GET /v1/npc/:nid/discord/trace` answers, as text; for the bot's owner or
//! an admin.

use std::sync::Arc;

use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::args::string;
use super::owned::owned_or_admin;
use super::{Answer, Tool};
use crate::api::Authored;
use crate::discord::trace::render::render;
use crate::discord::trace::report::report;

pub struct DiscordRead;

impl Tool for DiscordRead {
    const NAME: &'static str = "bot_discord_read";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "annotations": {"readOnlyHint": true},
            "description": "See what a bot's Discord bridge is doing: whether its bot is online and \
                its gateway connected, each DM and server channel it mirrors with how far the \
                thread has been posted and what is still unposted, the servers it is in, and its \
                last hundred events — every message heard (applied, dropped and why) and every \
                post made or refused. Ids and outcomes only, never message text.",
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
        match owned_or_admin(s, headers, bot).await {
            Ok(npc) => Answer::ok(render(bot, &report(s, npc).await)),
            Err(refused) => Answer::error(refused),
        }
    }
}
