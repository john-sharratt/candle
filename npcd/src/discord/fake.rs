//! A fake of Discord's REST API, on a port of the test's own.
//!
//! It answers the three calls a connect makes as Discord does — or refuses
//! them all one way, by [`Mode`] — and keeps every request it was sent, so a
//! test asserts the exact method, path, headers and body.

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
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Fake {
    /// Every request so far, in order.
    pub fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

#[derive(Clone)]
struct Held {
    mode: Mode,
    seen: Arc<Mutex<Vec<Seen>>>,
}

/// A fake answering as `mode`, on `127.0.0.1` and a port of its own.
pub async fn start(mode: Mode) -> Fake {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let held = Held {
        mode,
        seen: Arc::clone(&seen),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = Router::new().fallback(answer).with_state(held);
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Fake {
        url: format!("http://{addr}"),
        seen,
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
        path: uri.path().to_string(),
        authorization: text(header::AUTHORIZATION),
        user_agent: text(header::USER_AGENT),
        body: body.clone(),
    });
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
        Mode::Ok | Mode::NotABot => match (method, uri.path()) {
            (Method::GET, "/users/@me") => Json(json!({
                "id": USER_ID,
                "username": OLD_NAME,
                "bot": held.mode == Mode::Ok,
            }))
            .into_response(),
            (Method::GET, "/applications/@me") => {
                Json(json!({ "id": APPLICATION_ID, "name": "Zen" })).into_response()
            }
            (Method::PATCH, "/users/@me") => Json(json!({
                "id": USER_ID,
                "username": body["username"].as_str().unwrap_or(OLD_NAME),
                "bot": true,
            }))
            .into_response(),
            _ => (
                StatusCode::NOT_FOUND,
                Json(json!({ "message": "404: Not Found", "code": 0 })),
            )
                .into_response(),
        },
    }
}
