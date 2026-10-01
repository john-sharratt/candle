//! The device's own authentication: one token, one lookup, one meaning.
//!
//! The operator API trusts the gateway's `X-Tokera-*` headers, because npcd runs
//! behind it. **The device must not.** A character's request is built in-process
//! and stamped with its token, and an external client may send both a token and
//! `X-Tokera-*`; if this middleware ever read those headers, a request could be
//! authenticated as a human by a header a character's request happens to carry.
//! So it reads exactly one header, `Authorization: Bearer <token>`, ignores every
//! other, and never falls through to a role check.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::header::AUTHORIZATION;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use npc_map::schema::Where;
use serde_json::json;

use super::token::{Caller, Tokens};

/// Who is acting, set by [`require_token`] for the handlers behind it.
#[derive(Debug, Clone, Copy)]
pub struct DeviceCaller(pub Caller);

/// The place a request must be reached from, in place of where the body stands
/// now.
///
/// A character's grammar is built at one standpoint and its call lands a decode
/// later; a body that was carried off between the two would otherwise have its
/// call judged from the new room. It is a typed request extension only the
/// in-process path can insert, so no external caller can forge one.
#[derive(Debug, Clone)]
pub struct PinnedStandpoint(pub Where);

/// The bearer token on a request, `None` for any other header shape.
fn bearer(request: &Request) -> Option<&str> {
    let value = request.headers().get(AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = value.split_once(' ')?;
    (scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty()).then(|| token.trim())
}

fn unauthorized(detail: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": "unauthorized", "detail": detail })),
    )
        .into_response()
}

/// Resolve the bearer token to a body and hand its [`DeviceCaller`] on; refuse
/// with `401` when there is no token or it names nobody.
pub async fn require_token(
    State(tokens): State<Arc<Tokens>>,
    mut request: Request,
    next: Next,
) -> Response {
    let Some(token) = bearer(&request) else {
        return unauthorized("send `Authorization: Bearer <token>`");
    };
    let Some(caller) = tokens.resolve(token) else {
        return unauthorized("that token opens nothing");
    };
    request.extensions_mut().insert(DeviceCaller(caller));
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effector::token::Scope;
    use axum::body::Body;
    use axum::extract::Extension;
    use axum::middleware::from_fn_with_state;
    use axum::routing::get;
    use axum::Router;
    use tempfile::TempDir;
    use tower::ServiceExt;

    async fn whoami(Extension(DeviceCaller(caller)): Extension<DeviceCaller>) -> Json<serde_json::Value> {
        Json(json!({ "npc_id": caller.npc_id, "direct": caller.scope == Scope::Direct }))
    }

    fn app(tokens: Arc<Tokens>) -> Router {
        Router::new()
            .route("/", get(whoami))
            .layer(from_fn_with_state(tokens, require_token))
    }

    async fn call(app: &Router, headers: &[(&str, &str)]) -> (StatusCode, serde_json::Value) {
        let mut builder = axum::http::Request::builder().uri("/");
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let response = app
            .clone()
            .oneshot(builder.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 16).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null))
    }

    fn store() -> (TempDir, Arc<Tokens>) {
        let dir = TempDir::new().unwrap();
        let tokens = Arc::new(Tokens::load(dir.path().join("t")).unwrap());
        (dir, tokens)
    }

    #[tokio::test]
    async fn a_valid_token_resolves_to_its_body() {
        let (_dir, tokens) = store();
        let token = tokens.as_npc(7).unwrap();
        let (status, body) = call(&app(tokens), &[("authorization", &format!("Bearer {token}"))]).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "npc_id": 7, "direct": false }));
    }

    #[tokio::test]
    async fn a_direct_token_carries_its_scope() {
        let (_dir, tokens) = store();
        let token = tokens.direct(7).unwrap();
        let (_, body) = call(&app(tokens), &[("authorization", &format!("Bearer {token}"))]).await;
        assert_eq!(body, json!({ "npc_id": 7, "direct": true }));
    }

    #[tokio::test]
    async fn no_header_and_an_unknown_token_are_both_refused() {
        let (_dir, tokens) = store();
        tokens.as_npc(7).unwrap();
        let app = app(tokens);
        let (status, body) = call(&app, &[]).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(body["error"], "unauthorized");
        let (status, _) = call(&app, &[("authorization", "Bearer nope")]).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _) = call(&app, &[("authorization", "Basic abc")]).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn the_gateway_headers_never_stand_in_for_a_token() {
        let (_dir, tokens) = store();
        let (status, _) = call(
            &app(tokens),
            &[("x-tokera-user", "operator"), ("x-tokera-role", "admin")],
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn the_gateway_headers_do_not_change_who_a_token_is() {
        let (_dir, tokens) = store();
        let token = tokens.as_npc(7).unwrap();
        let (status, body) = call(
            &app(tokens),
            &[
                ("authorization", &format!("Bearer {token}")),
                ("x-tokera-user", "operator"),
                ("x-tokera-role", "admin"),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "npc_id": 7, "direct": false }));
    }
}
