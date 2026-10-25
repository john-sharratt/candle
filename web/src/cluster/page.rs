//! The gateway status page (§12): `GET /gateway` serves it and
//! `GET /gateway/status.json` its data, on every host, merged ahead of site
//! routing, and an admin's only — roles resolved from the estate's role table
//! (`roles_file:`). The page wears tokera.com's own top bar and stylesheets, so
//! it reads as part of the site it is linked from. It polls once a second. The
//! gateway's own commands,
//! `drain-gateway` and `undrain-gateway` (§12.4), are `POST`s beside them, and
//! a member's maintenance commands go down its socket from
//! `/gateway/member/<app>/<node>/…`.

use std::net::SocketAddr;
use std::sync::{Arc, LazyLock};

use axum::extract::{ConnectInfo, Path, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode, Uri};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use super::route::{forward, Context};
use super::{upgrades, Gateway, MemberOrigin};
use crate::auth::{Auth, Identity, Role, Roles};
use crate::config::Cache;
use crate::site::tokera::page::{nav_bar, Nav};

/// Where the page's template takes tokera.com's top bar.
const TOP_BAR: &str = "{{TOP_BAR}}";

/// The page itself: static, reading `status.json`, under the site's top bar.
static PAGE: LazyLock<String> =
    LazyLock::new(|| include_str!("gateway.html").replace(TOP_BAR, &nav_bar(Nav::Admin)));

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
        .route("/gateway/command/drain-gateway", post(drain))
        .route("/gateway/command/undrain-gateway", post(undrain))
        .route("/gateway/member/:app/:node/*rest", any(member))
        .with_state(PageState {
            gateway: gateway.clone(),
            auth: auth.clone(),
            roles: roles.clone(),
        })
        .merge(upgrades::router(gateway, auth, roles))
}

/// The caller, if an admin — by session or by the ops key
/// ([`Gateway::caller`]) — else the refusal for one who is not. Boxed because
/// a `Response` is large and this is the cold path.
fn admin(
    s: &PageState,
    headers: &HeaderMap,
    origin: Option<&MemberOrigin>,
) -> Result<Option<Identity>, Box<Response>> {
    let identity = s.gateway.caller(s.auth.as_deref(), headers, origin);
    let role = s.roles.of(identity.as_ref());
    if role.at_least(Role::Admin) {
        return Ok(identity);
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

/// The member a request came up from, when one did ([`MemberOrigin`]).
type Origin = Option<Extension<MemberOrigin>>;

fn origin_of(origin: &Origin) -> Option<&MemberOrigin> {
    origin.as_ref().map(|Extension(o)| o)
}

async fn page(State(s): State<PageState>, origin: Origin, headers: HeaderMap) -> Response {
    if let Err(refused) = admin(&s, &headers, origin_of(&origin)) {
        return *refused;
    }
    ([(header::CACHE_CONTROL, "no-store")], Html(PAGE.as_str())).into_response()
}

/// **The page wears tokera.com's bar and stylesheets**, so it reads as part of
/// the site it is linked from; its template has no placeholder left over.
#[cfg(test)]
#[test]
fn the_page_wears_the_sites_bar() {
    assert!(!PAGE.contains(TOP_BAR));
    assert!(PAGE.contains(r#"<header class="site-top">"#));
    assert!(PAGE.contains(r#"<body class="tokera">"#));
    for sheet in ["/base.css", "/lib/estate.css", "/site.css"] {
        assert!(PAGE.contains(&format!(r#"href="{sheet}""#)), "{sheet}");
    }
    // No link in the public nav is marked as this page.
    assert!(!PAGE.contains(r#"aria-current="page""#));
}

#[derive(Deserialize)]
struct LogQuery {
    #[serde(default)]
    log: Option<usize>,
}

/// The gateway's status, as the page reads it: an admin's, or a caller's that
/// presents the status key (`status_key.rs`) — how a script or an operator
/// without a browser session reads how the cluster stands.
async fn status(
    State(s): State<PageState>,
    origin: Origin,
    headers: HeaderMap,
    Query(q): Query<LogQuery>,
) -> Response {
    let keyed = s
        .gateway
        .status_key
        .as_ref()
        .is_some_and(|k| k.presented(&headers));
    if !keyed {
        if let Err(refused) = admin(&s, &headers, origin_of(&origin)) {
            return *refused;
        }
    }
    let mut status = match serde_json::to_value(s.gateway.hub.status(q.log.unwrap_or(LOG_LIMIT))) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("the status did not serialize: {e}"),
            )
                .into_response()
        }
    };
    status["peers"] = s.gateway.peers_view().unwrap_or(Value::Null);
    ([(header::CACHE_CONTROL, "no-store")], Json(status)).into_response()
}

/// Stop admitting member sockets, close the ones there with `1001`, and
/// answer once the requests in flight have finished, or a minute has passed.
async fn drain(State(s): State<PageState>, origin: Origin, headers: HeaderMap) -> Response {
    if let Err(refused) = admin(&s, &headers, origin_of(&origin)) {
        return *refused;
    }
    let still_open = s.gateway.hub.drain_gateway().await;
    Json(json!({ "draining": true, "still_open": still_open })).into_response()
}

async fn undrain(State(s): State<PageState>, origin: Origin, headers: HeaderMap) -> Response {
    if let Err(refused) = admin(&s, &headers, origin_of(&origin)) {
        return *refused;
    }
    s.gateway.hub.undrain_gateway();
    Json(json!({ "draining": false })).into_response()
}

/// The member routes the page reaches through this gateway: its maintenance
/// commands and their jobs, nothing else.
const MEMBER_ROUTES: &str = "v1/cluster/";

/// A maintenance command for one member, forwarded down its socket (§12.4):
/// `/gateway/member/<app>/<node>/v1/cluster/command/…`. The page is served on
/// every host, so the member is named by its app here rather than by the host
/// the request arrived on.
async fn member(
    State(s): State<PageState>,
    Path((app, node, rest)): Path<(String, String, String)>,
    peer: Option<ConnectInfo<SocketAddr>>,
    mut req: Request,
) -> Response {
    let identity = match admin(&s, req.headers(), req.extensions().get::<MemberOrigin>()) {
        Ok(identity) => identity,
        Err(refused) => return *refused,
    };
    let refuse = |status: StatusCode, detail: String| {
        (status, Json(json!({"error": "refused", "detail": detail}))).into_response()
    };
    if !rest.starts_with(MEMBER_ROUTES) {
        return refuse(
            StatusCode::NOT_FOUND,
            format!("only /{MEMBER_ROUTES}… reaches a member from this page"),
        );
    }
    let Ok(node) = u64::from_str_radix(&node, 16) else {
        return refuse(StatusCode::BAD_REQUEST, format!("{node:?} is not a node"));
    };
    if s.gateway.hub.member(&app, node).is_none() {
        return refuse(
            StatusCode::NOT_FOUND,
            format!("no {app} member {node:016x} on this gateway"),
        );
    }
    let target = match req.uri().query() {
        Some(q) => format!("/{rest}?{q}"),
        None => format!("/{rest}"),
    };
    match target.parse::<Uri>() {
        Ok(uri) => *req.uri_mut() = uri,
        Err(e) => return refuse(StatusCode::BAD_REQUEST, format!("{target:?}: {e}")),
    }
    let ctx = Context {
        want_html: false,
        peer: peer.map(|ConnectInfo(a)| a.ip()),
        identity: identity.as_ref(),
        secure: false,
        cache: Cache::default(),
        roles: &s.roles,
        from_member: None,
    };
    forward(&s.gateway, &app, node, &ctx, req).await
}
