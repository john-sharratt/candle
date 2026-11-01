//! `GET`/`PUT`/`DELETE /v1/world/:wid/discord` — a world's Discord server
//! (`docs/npcd_discord.md` §4.4).
//!
//! A Creator's, like everything else that writes into a world as a whole
//! rather than into a character somebody owns.
//!
//! - `GET` answers `{"guild": null}`, or the server: its id, its name and the
//!   listener; and `listeners`, each character of the world with a bot of its
//!   own, with the servers its bot is in and the link that adds it to
//!   another — what the console picks the server and the listener from.
//! - `PUT {guild_id, listener}` sets it, once three checks pass:
//!   the listener is a character with a body in the world, it has a bot
//!   connected, and that bot is in the server — Discord's answer to
//!   `GET /guilds/{guild_id}` as the bot, which also names the server. It is
//!   written through the world's authority (`WorldOp::Guild`).
//! - `DELETE` takes it away: the mirror stops, and the bots' own DMs and
//!   server channels (§3.3, §3.4) go on.
//!
//! Ids cross as strings: a Discord id is past what a browser's numbers hold,
//! and a character's is its base-36 wire form.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::record::Guild;
use crate::api::{err, owner_of, Authored};
use crate::authority::ops::{OpReply, WorldOp};
use crate::discord::error::DiscordError;
use crate::discord::{invite, store};
use crate::npcs::{npc_id_of_wire, npc_id_wire};
use crate::world::Hosted;

/// What `PUT` takes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetBody {
    /// The server's id, as Discord's Copy Server ID gives it.
    guild_id: String,
    /// The listener's character id, on the wire.
    listener: String,
}

/// `GET /v1/world/:wid/discord`
pub async fn get_world_discord(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_world();
    };
    let Some(hosted) = rt.hosted.get(&wid) else {
        return no_world();
    };
    let mut answer = wire(hosted.mirror(|m| m.guild().cloned()).as_ref());
    answer["listeners"] = listeners(&s, &hosted, rt.bodies.in_world(&wid));
    Json(answer).into_response()
}

/// Who could listen to a server for the world: each character with a body
/// in it and a bot of its own — whoever owns it — with the servers its bot
/// is in, as its records hold them, and the link that adds it to another.
fn listeners(s: &Authored, hosted: &Hosted, bound: Vec<(u64, String)>) -> Value {
    let shared = s.npcs.substrate();
    let found: Vec<Value> = bound
        .into_iter()
        .filter_map(|(npc, body)| {
            let bot = store::get(&shared, npc)?;
            let name = hosted.read(|w| w.actor(&body).map(|a| a.name.clone()))?;
            let servers: Vec<Value> = hosted
                .servers(|x| x.servers_of(npc))
                .into_iter()
                .map(|(guild_id, server)| json!({ "guild_id": guild_id.to_string(), "name": server.name }))
                .collect();
            Some(json!({
                "listener": npc_id_wire(npc),
                "name": name,
                "bot": bot.username,
                "invite_url": invite::url(&bot.application_id),
                "servers": servers,
            }))
        })
        .collect();
    Value::Array(found)
}

/// `PUT /v1/world/:wid/discord`
pub async fn put_world_discord(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
    Json(body): Json<SetBody>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_world();
    };
    if rt.hosted.get(&wid).is_none() {
        return no_world();
    }
    let Ok(guild_id) = body.guild_id.trim().parse::<u64>() else {
        return err(
            StatusCode::BAD_REQUEST,
            "discord_guild_id_invalid",
            "a server id is the number Discord's Copy Server ID gives",
        );
    };
    let in_world = npc_id_of_wire(body.listener.trim()).filter(|npc| {
        rt.body_of(*npc)
            .is_some_and(|(hosted, _)| hosted.id() == wid)
    });
    let Some(listener) = in_world else {
        return err(
            StatusCode::BAD_REQUEST,
            "listener_not_in_world",
            "the listener must be a character with a body in this world",
        );
    };
    let Some(bot) = store::get(&s.npcs.substrate(), listener) else {
        return err(
            StatusCode::BAD_REQUEST,
            "listener_has_no_bot",
            "the listener needs a Discord bot of its own: connect one on its Messages tab",
        );
    };
    let token = match s.discord_tokens.open(listener, &bot.token) {
        Ok(token) => token,
        Err(_) => {
            return err(
                StatusCode::BAD_REQUEST,
                "listener_has_no_bot",
                "the listener's bot token does not open here: connect it again",
            )
        }
    };
    let server = match s.discord.guild(&token, guild_id).await {
        Ok(server) => server,
        Err(DiscordError::Status { status, .. }) if status == 403 || status == 404 => {
            return err(
                StatusCode::BAD_REQUEST,
                "listener_not_in_server",
                "the listener's bot is not in that server: invite it with its invite link",
            )
        }
        Err(e) => {
            tracing::warn!(world = %wid, error = %e, "discord: a world's server was not checked");
            return err(StatusCode::BAD_GATEWAY, "discord_failed", &e.to_string());
        }
    };
    let guild = Guild {
        guild_id,
        name: server.name,
        listener,
    };
    let answer = write(&s, &wid, Some(guild.clone()), wire(Some(&guild))).await;
    // Setting the server is the person answering what Discord refused —
    // a permission granted, another server chosen — so the mirror asks again
    // on its next beat rather than waiting out its hold.
    s.bots.mirror_cleared(&wid);
    answer
}

/// `DELETE /v1/world/:wid/discord`
pub async fn delete_world_discord(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(wid): Path<String>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    if s.runtime
        .as_ref()
        .and_then(|rt| rt.hosted.get(&wid))
        .is_none()
    {
        return no_world();
    }
    write(&s, &wid, None, wire(None)).await
}

/// Write the world's server as `guild` through its authority, and answer
/// `answer`.
async fn write(s: &Authored, wid: &str, guild: Option<Guild>, answer: Value) -> Response {
    let rt = s.runtime.as_ref().expect("a hosted world is the runtime's");
    match rt.world_write(wid, WorldOp::Guild { guild }).await {
        Ok(OpReply::Done { ok: true }) => Json(answer).into_response(),
        Ok(_) => no_world(),
        Err(e) => err(
            StatusCode::SERVICE_UNAVAILABLE,
            "world_unreached",
            &format!("the world's authority did not answer: {e:#}"),
        ),
    }
}

/// The world's server as the console reads it.
fn wire(guild: Option<&Guild>) -> Value {
    match guild {
        None => json!({ "guild": null }),
        Some(g) => json!({
            "guild": {
                "guild_id": g.guild_id.to_string(),
                "name": g.name,
                "listener": npc_id_wire(g.listener),
            }
        }),
    }
}

fn no_world() -> Response {
    err(
        StatusCode::NOT_FOUND,
        "no_such_world",
        "that world is not hosted here",
    )
}
