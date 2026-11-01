//! A fake of Discord's REST API, on a port of the test's own.
//!
//! It answers the calls a connect and a DM make as Discord does — or refuses
//! them all one way, by [`Mode`] — and keeps every request it was sent, so a
//! test asserts the exact method, path, headers and body. Replies queued with
//! [`Fake::reply`] answer the next requests first, in order, whatever they
//! are: how a test makes one post fail and the next succeed.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::net::TcpListener;

/// The bot's user id.
pub const USER_ID: &str = "222";
/// Its application's id.
pub const APPLICATION_ID: &str = "111";
/// What the bot is called before anything renames it.
pub const OLD_NAME: &str = "Old Name";
/// The DM channel `POST /users/@me/channels` opens.
pub const DM_CHANNEL: u64 = 333;
/// The id of the first message the fake posts; each next is one more.
pub const FIRST_POSTED: u64 = 5000;
/// What `GET /guilds/{id}` calls every server.
pub const GUILD_NAME: &str = "zenworld";
/// The id of the first channel the fake makes; each next is one more.
pub const FIRST_CHANNEL: u64 = 7000;
/// The token of every webhook the fake makes. Its id is the channel's plus
/// [`HOOK_OFFSET`].
pub const HOOK_TOKEN: &str = "hook-token";
/// What a channel's id is raised by to name its webhook.
pub const HOOK_OFFSET: u64 = 100_000;

/// How the fake answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// As Discord does for a good bot token.
    Ok,
    /// `401` to everything.
    Unauthorized,
    /// `429` to everything, asking for 2.5 s.
    RateLimited,
    /// A user's account rather than a bot's.
    NotABot,
    /// `502`, and a body that is not JSON.
    Failing,
}

/// One request the fake was sent.
#[derive(Debug, Clone, PartialEq)]
pub struct Seen {
    pub method: String,
    /// The path and its query.
    pub path: String,
    pub authorization: String,
    pub user_agent: String,
    /// The JSON body, `null` when there was none.
    pub body: Value,
}

/// A running fake.
pub struct Fake {
    /// Its base, to hand [`crate::discord::client::Client::at`].
    pub url: String,
    held: Held,
}

impl Fake {
    /// Every request so far, in order.
    pub fn seen(&self) -> Vec<Seen> {
        self.held.seen.lock().unwrap().clone()
    }

    /// Answer the next request not yet answered by an earlier queued reply
    /// with `status` and `body`, whatever it asks.
    pub fn reply(&self, status: u16, body: Value) {
        self.held
            .replies
            .lock()
            .unwrap()
            .push_back((StatusCode::from_u16(status).unwrap(), body));
    }

    /// What `GET /channels/{channel}/messages` answers: `messages`, as Discord
    /// would give them, newest first.
    pub fn history(&self, channel: u64, messages: Vec<Value>) {
        self.held.history.lock().unwrap().insert(channel, messages);
    }

    /// What `GET /guilds/{id}/channels` answers before any is made: `channels`,
    /// as `(id, type, name)`.
    pub fn channels(&self, channels: &[(u64, u8, &str)]) {
        *self.held.channels.lock().unwrap() = channels
            .iter()
            .map(|(id, kind, name)| json!({ "id": id.to_string(), "type": kind, "name": name }))
            .collect();
    }

    /// Channel `channel`'s permission overwrites, as `(id, type, allow,
    /// deny)`, sorted by id.
    pub fn overwrites_of(&self, channel: u64) -> Vec<(u64, u8, u64, u64)> {
        let mut out: Vec<(u64, u8, u64, u64)> = self
            .held
            .overwrites
            .lock()
            .unwrap()
            .get(&channel)
            .map(|os| {
                os.iter()
                    .map(|o| {
                        let n = |f: &str| o[f].as_str().and_then(|s| s.parse().ok()).unwrap_or(0);
                        (
                            n("id"),
                            o["type"].as_u64().unwrap_or(0) as u8,
                            n("allow"),
                            n("deny"),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.sort();
        out
    }
}

/// The next id the fake gives a channel or thread it makes.
fn next_made(held: &Held) -> u64 {
    let mut made = held.made.lock().unwrap();
    let id = *made;
    *made += 1;
    id
}

/// What Discord answers a good request about a channel's permissions,
/// threads and members, when `path` is one.
fn ok_permissions(held: &Held, method: &Method, path: &str, body: &Value) -> Option<Response> {
    let rest = path.strip_prefix("/channels/")?;
    let (channel, tail) = rest.split_once('/').unwrap_or((rest, ""));
    let channel: u64 = channel.parse().ok()?;
    let empty = || StatusCode::NO_CONTENT.into_response();
    match (method.clone(), tail) {
        (Method::GET, "") => Some(
            Json(json!({
                "id": channel.to_string(),
                "permission_overwrites": held.overwrites.lock().unwrap()
                    .get(&channel).cloned().unwrap_or_default(),
            }))
            .into_response(),
        ),
        (Method::POST, "threads") => {
            let id = next_made(held);
            Some(
                Json(json!({ "id": id.to_string(), "type": 12, "name": body["name"] }))
                    .into_response(),
            )
        }
        (Method::PUT, t) if t.starts_with("thread-members/") => Some(empty()),
        (Method::POST, "typing") => Some(empty()),
        (method, t) if t.starts_with("permissions/") => {
            let target = t.trim_start_matches("permissions/").to_string();
            let mut all = held.overwrites.lock().unwrap();
            let on = all.entry(channel).or_default();
            on.retain(|o| o["id"] != json!(target));
            if method == Method::PUT {
                let mut o = body.clone();
                o["id"] = json!(target);
                on.push(o);
            }
            Some(empty())
        }
        _ => None,
    }
}

#[derive(Clone)]
struct Held {
    mode: Mode,
    seen: Arc<Mutex<Vec<Seen>>>,
    replies: Arc<Mutex<VecDeque<(StatusCode, Value)>>>,
    history: Arc<Mutex<BTreeMap<u64, Vec<Value>>>>,
    posted: Arc<Mutex<u64>>,
    /// The server's channels, those made since included.
    channels: Arc<Mutex<Vec<Value>>>,
    made: Arc<Mutex<u64>>,
    /// Each channel's permission overwrites, as Discord answers them.
    overwrites: Arc<Mutex<BTreeMap<u64, Vec<Value>>>>,
}

/// A fake answering as `mode`, on `127.0.0.1` and a port of its own.
pub async fn start(mode: Mode) -> Fake {
    let held = Held {
        mode,
        seen: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(VecDeque::new())),
        history: Arc::new(Mutex::new(BTreeMap::new())),
        posted: Arc::new(Mutex::new(FIRST_POSTED)),
        channels: Arc::new(Mutex::new(Vec::new())),
        made: Arc::new(Mutex::new(FIRST_CHANNEL)),
        overwrites: Arc::new(Mutex::new(BTreeMap::new())),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = Router::new().fallback(answer).with_state(held.clone());
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Fake {
        url: format!("http://{addr}"),
        held,
    }
}

async fn answer(
    State(held): State<Held>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let text = |name| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string()
    };
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    held.seen.lock().unwrap().push(Seen {
        method: method.to_string(),
        path: uri
            .path_and_query()
            .map_or_else(|| uri.path().to_string(), |p| p.to_string()),
        authorization: text(header::AUTHORIZATION),
        user_agent: text(header::USER_AGENT),
        body: body.clone(),
    });
    if let Some((status, reply)) = held.replies.lock().unwrap().pop_front() {
        return (status, Json(reply)).into_response();
    }
    match held.mode {
        Mode::Unauthorized => (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "message": "401: Unauthorized", "code": 0 })),
        )
            .into_response(),
        Mode::RateLimited => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({
                "message": "You are being rate limited.",
                "retry_after": 2.5,
                "global": false
            })),
        )
            .into_response(),
        Mode::Failing => (StatusCode::BAD_GATEWAY, "upstream is down").into_response(),
        Mode::Ok | Mode::NotABot => ok(&held, &method, uri.path(), &body),
    }
}

/// The next message id the fake posts.
fn next_posted(held: &Held) -> u64 {
    let mut posted = held.posted.lock().unwrap();
    let id = *posted;
    *posted += 1;
    id
}

/// What Discord answers a good request to a server, a channel's webhooks or
/// a webhook, when `path` is one.
fn ok_server(held: &Held, method: &Method, path: &str, body: &Value) -> Option<Response> {
    let id_in = |prefix: &str, suffix: &str| {
        path.strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(suffix))
            .and_then(|id| id.parse::<u64>().ok())
    };
    if let Some(guild) = id_in("/guilds/", "/channels") {
        return Some(match *method {
            Method::POST => {
                let id = next_made(held);
                if let Some(os) = body["permission_overwrites"].as_array() {
                    held.overwrites.lock().unwrap().insert(id, os.clone());
                }
                let channel = json!({
                    "id": id.to_string(),
                    "type": body["type"],
                    "name": body["name"],
                    "guild_id": guild.to_string(),
                });
                held.channels.lock().unwrap().push(channel.clone());
                Json(channel).into_response()
            }
            _ => Json(Value::Array(held.channels.lock().unwrap().clone())).into_response(),
        });
    }
    if let Some(guild) = id_in("/guilds/", "") {
        return Some(Json(json!({ "id": guild.to_string(), "name": GUILD_NAME })).into_response());
    }
    if let Some(channel) = id_in("/channels/", "/webhooks") {
        return Some(
            Json(json!({
                "id": (channel + HOOK_OFFSET).to_string(),
                "token": HOOK_TOKEN,
                "channel_id": channel.to_string(),
            }))
            .into_response(),
        );
    }
    let (hook, _token) = path.strip_prefix("/webhooks/")?.split_once('/')?;
    let channel = hook.parse::<u64>().ok()?.checked_sub(HOOK_OFFSET)?;
    Some(
        Json(json!({
            "id": next_posted(held).to_string(),
            "channel_id": channel.to_string(),
            "webhook_id": hook,
            "author": { "id": hook, "username": body["username"], "bot": true },
            "content": body["content"],
        }))
        .into_response(),
    )
}

/// What Discord answers a good request.
fn ok(held: &Held, method: &Method, path: &str, body: &Value) -> Response {
    if let Some(answer) = ok_server(held, method, path, body) {
        return answer;
    }
    if let Some(answer) = ok_permissions(held, method, path, body) {
        return answer;
    }
    let channel = path
        .strip_prefix("/channels/")
        .and_then(|rest| rest.strip_suffix("/messages"))
        .and_then(|id| id.parse::<u64>().ok());
    match (method.clone(), path, channel) {
        (Method::GET, "/users/@me", _) => Json(json!({
            "id": USER_ID,
            "username": OLD_NAME,
            "bot": held.mode == Mode::Ok,
        }))
        .into_response(),
        (Method::GET, "/applications/@me", _) => {
            Json(json!({ "id": APPLICATION_ID, "name": "Zen" })).into_response()
        }
        (Method::PATCH, "/users/@me", _) => Json(json!({
            "id": USER_ID,
            "username": body["username"].as_str().unwrap_or(OLD_NAME),
            "bot": true,
        }))
        .into_response(),
        (Method::POST, "/users/@me/channels", _) => {
            Json(json!({ "id": DM_CHANNEL.to_string(), "type": 1 })).into_response()
        }
        (Method::POST, _, Some(channel)) => {
            let id = next_posted(held);
            Json(json!({
                "id": id.to_string(),
                "channel_id": channel.to_string(),
                "author": { "id": USER_ID, "username": "Zen", "bot": true },
                "content": body["content"],
            }))
            .into_response()
        }
        (Method::GET, _, Some(channel)) => Json(Value::Array(
            held.history
                .lock()
                .unwrap()
                .get(&channel)
                .cloned()
                .unwrap_or_default(),
        ))
        .into_response(),
        _ => (
            StatusCode::NOT_FOUND,
            Json(json!({ "message": "404: Not Found", "code": 0 })),
        )
            .into_response(),
    }
}
