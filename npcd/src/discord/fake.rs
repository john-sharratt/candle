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
}

#[derive(Clone)]
struct Held {
    mode: Mode,
    seen: Arc<Mutex<Vec<Seen>>>,
    replies: Arc<Mutex<VecDeque<(StatusCode, Value)>>>,
    history: Arc<Mutex<BTreeMap<u64, Vec<Value>>>>,
    posted: Arc<Mutex<u64>>,
}

/// A fake answering as `mode`, on `127.0.0.1` and a port of its own.
pub async fn start(mode: Mode) -> Fake {
    let held = Held {
        mode,
        seen: Arc::new(Mutex::new(Vec::new())),
        replies: Arc::new(Mutex::new(VecDeque::new())),
        history: Arc::new(Mutex::new(BTreeMap::new())),
        posted: Arc::new(Mutex::new(FIRST_POSTED)),
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

/// What Discord answers a good request.
fn ok(held: &Held, method: &Method, path: &str, body: &Value) -> Response {
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
            let mut posted = held.posted.lock().unwrap();
            let id = *posted;
            *posted += 1;
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
