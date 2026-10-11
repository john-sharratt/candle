//! The gateway status page (§12): `GET /gateway` serves it and
//! `GET /gateway/status.json` its data, on every host, merged ahead of site
//! routing, and an admin's only — roles resolved from this gateway's own
//! `roles:` table. The page polls once a second.

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use super::Gateway;
use crate::auth::{Auth, Role, Roles};

/// The page itself: static, reading `status.json`.
const PAGE: &str = include_str!("gateway.html");

/// How many coordination-log entries `status.json` carries by default.
const LOG_LIMIT: usize = 500;

#[derive(Clone)]
struct PageState {
    gateway: Arc<Gateway>,
    auth: Option<Arc<Auth>>,
    roles: Roles,
}

pub fn router(gateway: Arc<Gateway>, auth: Option<Arc<Auth>>, roles: Roles) -> Router {
    Router::new()
        .route("/gateway", get(page))
        .route("/gateway/status.json", get(status))
        .with_state(PageState {
            gateway,
            auth,
            roles,
        })
}

/// The refusal for a caller who is not an admin. Boxed because a `Response`
/// is large and this is the cold path.
fn admin(s: &PageState, headers: &HeaderMap) -> Result<(), Box<Response>> {
    let identity = s.auth.as_ref().and_then(|a| a.identity(headers));
    let role = s.roles.of(identity.as_ref());
    if role.at_least(Role::Admin) {
        return Ok(());
    }
    let status = if identity.is_none() {
        StatusCode::UNAUTHORIZED
    } else {
        StatusCode::FORBIDDEN
    };
    Err(Box::new(
        (
            status,
            Json(json!({
                "error": if identity.is_none() { "unauthorized" } else { "forbidden" },
                "detail": format!("the gateway's status is an admin's; you are `{role}`"),
            })),
        )
            .into_response(),
    ))
}

async fn page(State(s): State<PageState>, headers: HeaderMap) -> Response {
    if let Err(refused) = admin(&s, &headers) {
        return *refused;
    }
    ([(header::CACHE_CONTROL, "no-store")], Html(PAGE)).into_response()
}

#[derive(Deserialize)]
struct LogQuery {
    #[serde(default)]
    log: Option<usize>,
}

async fn status(
    State(s): State<PageState>,
    headers: HeaderMap,
    Query(q): Query<LogQuery>,
) -> Response {
    if let Err(refused) = admin(&s, &headers) {
        return *refused;
    }
    (
        [(header::CACHE_CONTROL, "no-store")],
        Json(s.gateway.hub.status(q.log.unwrap_or(LOG_LIMIT))),
    )
        .into_response()
}
