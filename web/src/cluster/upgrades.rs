//! The gateway's upgrade commands (`docs/replicated_substrate.md` §12.4), on the
//! status page's Upgrades block, admin only, at `POST /gateway/upgrade/<name>`
//! with `{ app, build, from, targets, parallel, confirm }`:
//!
//! - `raise-floor` and `cancel-floor` end an app's grace period now, or discard
//!   its pending floor — this gateway's own;
//! - `certify-anyway`, `reject-build`, `pause-rollout` and `resume-rollout` are
//!   decisions the group records, so each is sent to a member of the app as its
//!   own command, and that member proposes it;
//! - `rollout` answers with its plan, and only once the build's `count-hash` is
//!   typed to confirm it does it go to the plan's source member, which proposes
//!   `rollout/<app>/<n>`.
//!
//! A command sent to a member answers with the member's job and its node, so
//! the page follows its stream through `/gateway/member/<app>/<node>/…`.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::body::{to_bytes, Body, Bytes};
use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use cluster::gateway::upgrades::plan::Ask;
use serde::Deserialize;
use serde_json::{json, Value};

use super::route::{forward, Context};
use super::Gateway;
use crate::auth::{Auth, Role, Roles};
use crate::config::Cache;

#[derive(Clone)]
struct CmdState {
    gateway: Arc<Gateway>,
    auth: Option<Arc<Auth>>,
    roles: Roles,
}

/// A command's body: every field any of them takes.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct CommandBody {
    app: Option<String>,
    build: Option<String>,
    from: Option<String>,
    targets: Option<Vec<String>>,
    parallel: Option<u32>,
    confirm: Option<String>,
}

pub fn router(gateway: Arc<Gateway>, auth: Option<Arc<Auth>>, roles: Roles) -> Router {
    Router::new()
        .route("/gateway/upgrade/:name", post(command))
        .with_state(CmdState {
            gateway,
            auth,
            roles,
        })
}

fn refuse(status: StatusCode, detail: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": "refused", "detail": detail.into()})),
    )
        .into_response()
}

fn node(hex: &str) -> Result<u64, String> {
    u64::from_str_radix(hex, 16).map_err(|_| format!("{hex:?} is not a node"))
}

async fn command(
    State(s): State<CmdState>,
    Path(name): Path<String>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    bytes: Bytes,
) -> Response {
    let identity = s.auth.as_ref().and_then(|a| a.identity(&headers));
    let role = s.roles.of(identity.as_ref());
    if !role.at_least(Role::Admin) {
        let status = if identity.is_none() {
            StatusCode::UNAUTHORIZED
        } else {
            StatusCode::FORBIDDEN
        };
        return refuse(
            status,
            format!("an upgrade command is an admin's; you are `{role}`"),
        );
    }
    let body: CommandBody = if bytes.iter().all(u8::is_ascii_whitespace) {
        CommandBody::default()
    } else {
        match serde_json::from_slice(&bytes) {
            Ok(b) => b,
            Err(e) => {
                return refuse(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!("the body is not a command's: {e}"),
                )
            }
        }
    };
    let Some(app) = body.app.clone() else {
        return refuse(StatusCode::BAD_REQUEST, format!("{name} needs the `app`"));
    };
    let hub = &s.gateway.hub;
    let ctx = Context {
        want_html: false,
        peer: peer.map(|ConnectInfo(a)| a.ip()),
        identity: identity.as_ref(),
        secure: false,
        cache: Cache::default(),
        roles: &s.roles,
        from_member: None,
    };
    match name.as_str() {
        "raise-floor" => match hub.raise_floor(&app) {
            Ok(()) => Json(json!({"raised": app})).into_response(),
            Err(e) => refuse(StatusCode::CONFLICT, e),
        },
        "cancel-floor" => match hub.cancel_floor(&app) {
            Ok(()) => Json(json!({"cancelled": app})).into_response(),
            Err(e) => refuse(StatusCode::CONFLICT, e),
        },
        "certify-anyway" | "reject-build" | "pause-rollout" | "resume-rollout" => {
            let Some(to) = hub.any_ready(&app) else {
                return refuse(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("no {app} member is ready to record it"),
                );
            };
            let mut sent = json!({"app": app});
            if let Some(b) = &body.build {
                sent["build"] = json!(b);
            }
            to_member(&s.gateway, &app, to, &name, sent, &ctx).await
        }
        "rollout" => {
            let Some(build) = body.build.clone() else {
                return refuse(StatusCode::BAD_REQUEST, "rollout needs the `build`");
            };
            let from = match body.from.as_deref().map(node).transpose() {
                Ok(f) => f,
                Err(e) => return refuse(StatusCode::BAD_REQUEST, e),
            };
            let targets = match body
                .targets
                .as_ref()
                .map(|t| {
                    t.iter()
                        .map(|h| node(h))
                        .collect::<Result<Vec<u64>, String>>()
                })
                .transpose()
            {
                Ok(t) => t,
                Err(e) => return refuse(StatusCode::BAD_REQUEST, e),
            };
            let ask = Ask {
                app: app.clone(),
                build: build.clone(),
                from,
                targets,
                parallel: body.parallel.unwrap_or(1),
                by: identity
                    .as_ref()
                    .map_or_else(|| "local".into(), |i| i.email.clone()),
            };
            let plan = match hub.plan_rollout(&ask) {
                Ok(p) => p,
                Err(e) => return refuse(StatusCode::CONFLICT, e),
            };
            if body.confirm.as_deref() != Some(build.as_str()) {
                return Json(json!({ "plan": plan, "confirm": build })).into_response();
            }
            let Ok(source) = node(&plan.from) else {
                return refuse(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "the plan names no source",
                );
            };
            let sent = json!({"app": app, "plan": plan.decision, "confirm": build});
            to_member(&s.gateway, &app, source, "rollout", sent, &ctx).await
        }
        other => refuse(
            StatusCode::NOT_FOUND,
            format!("there is no gateway command {other:?}"),
        ),
    }
}

/// Send `name` with `body` to member `node` of `app` as its own command, and
/// answer with its job and the node, for the page to follow.
async fn to_member(
    gateway: &Arc<Gateway>,
    app: &str,
    node: u64,
    name: &str,
    body: Value,
    ctx: &Context<'_>,
) -> Response {
    let req = Request::builder()
        .method(Method::POST)
        .uri(format!("/v1/cluster/command/{name}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("a command request builds");
    let res = forward(gateway, app, node, ctx, req).await;
    let status = res.status();
    let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap_or_default();
    let mut answer: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    if let Value::Object(m) = &mut answer {
        m.insert("node".into(), json!(format!("{node:016x}")));
    }
    (status, Json(answer)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_is_hex() {
        assert_eq!(node("00000000000000ab").ok(), Some(0xab));
        assert_eq!(node("zz"), Err("\"zz\" is not a node".to_string()));
    }
}
