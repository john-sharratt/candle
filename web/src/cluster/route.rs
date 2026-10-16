//! Where a request for a clustered app goes (§5.2): the dispatch its route
//! declares in the members' table, its key, the member the hub picks, and the
//! answer — forwarded to one member, combined from all, handed back to go to
//! the configured upstream when no member is admitted, or the gateway's
//! "reconnecting" page while every admitted member is still loading.
//!
//! Every request a member answers is a sample for certifying a candidate
//! (§3.8): while one certifies it takes its share of the traffic, and a share
//! of the baseline's requests on a mirrored route are mirrored to it
//! ([`super::mirror`]).
//!
//! Three things beside the dispatch:
//!
//! - **UI pinning.** A page's files come from one member: the first UI request
//!   is dealt round robin and answered with `tokera_ui_<app>`, and every UI
//!   request carrying it goes to that member while it is ready.
//! - **`x-tokera-node`**, an admin's: the request goes to that member whatever
//!   the route declares.
//! - **A key held through another cluster.** A member that knows the key's
//!   holder answers `409` naming it (§5.4); the gateway answers `503` with
//!   `Retry-After`, because the next attempt may land on a gateway the holder
//!   is joined to.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use cluster::dispatch::{extract, Dispatch, Extracted, KeySource, Merge, Mirror, BODY_LIMIT};
use cluster::gateway::{Class, Pick, NODE_HEADER};

use super::mirror::{mirrored, sampled, Route};
use super::{aggregate, merge_ws, Gateway};
use crate::auth::{cookie, Identity, Role, Roles};
use crate::config::Cache;
use crate::errors::{self, Problem};
use crate::proxy::{self, Forward, To};

/// The header a member names a key's holder in, on its `409`.
pub const HOLDER_HEADER: &str = "x-tokera-holder";

/// How long a browser waits before retrying a key whose holder is reachable
/// only through another cluster.
const HOLDER_RETRY: Duration = Duration::from_secs(1);

/// What the request carries besides itself, resolved by the server.
pub struct Context<'a> {
    pub want_html: bool,
    pub peer: Option<IpAddr>,
    pub identity: Option<&'a Identity>,
    pub secure: bool,
    pub cache: Cache,
    pub roles: &'a Roles,
    /// The member that opened the stream this request came on, if one did.
    pub from_member: Option<u64>,
}

/// The routing's outcome.
pub enum Routed {
    Answered(Response),
    /// No member is ready: the configured upstream serves it.
    Dumb(Request),
}

/// The UI pin cookie's name for `app`.
pub fn pin_cookie(app: &str) -> String {
    format!("tokera_ui_{app}")
}

fn pinned(req: &Request, app: &str) -> Option<u64> {
    let raw = cookie::get(req.headers(), &pin_cookie(app))?;
    u64::from_str_radix(raw, 16).ok()
}

fn set_pin(res: &mut Response, app: &str, node: u64, secure: bool) {
    let secure = if secure { "; Secure" } else { "" };
    let value = format!(
        "{}={node:016x}; Path=/; HttpOnly; SameSite=Lax{secure}",
        pin_cookie(app)
    );
    if let Ok(v) = HeaderValue::from_str(&value) {
        res.headers_mut().append(header::SET_COOKIE, v);
    }
}

/// The node an admin's `x-tokera-node` names; `Err` refuses the request.
/// Boxed because a `Problem` is large and this is the cold path.
fn addressed(req: &Request, ctx: &Context<'_>) -> Result<Option<u64>, Box<Problem>> {
    let Some(raw) = req.headers().get(NODE_HEADER) else {
        return Ok(None);
    };
    if !ctx.roles.of(ctx.identity).at_least(Role::Admin) {
        return Err(Box::new(Problem::forbidden(format!(
            "`{NODE_HEADER}` addresses one member, which is an admin's to do"
        ))));
    }
    raw.to_str()
        .ok()
        .and_then(|s| u64::from_str_radix(s, 16).ok())
        .map(Some)
        .ok_or_else(|| {
            Box::new(Problem::bad_request(format!(
                "`{NODE_HEADER}` is not a node"
            )))
        })
}

/// How long a page waits before asking again while every member is loading.
const LOADING_RETRY: Duration = Duration::from_secs(3);

/// The answer while every member of `app` admitted here is still loading: the
/// gateway's own "reconnecting" page, which the consoles retry. Not the
/// configured upstream — it is usually one of those members — and not a
/// loading member standing in: either put that member's half-loaded startup
/// card in front of players (measured 2026-10-16: .5 restarting, WhiteWorld
/// stuck loading behind it, and bot.tokera.com showing WhiteWorld's
/// repository wait).
pub fn restarting(app: &str, want_html: bool) -> Response {
    errors::respond(
        Problem::backing_off("Restarting — back in a moment.", LOADING_RETRY)
            .because(format!("every {app} member admitted here is still loading")),
        want_html,
    )
}

/// No member of `app` is ready: its configured upstream when none is admitted
/// (§5.7), [`restarting`] when some are and all are loading.
fn none_ready(gateway: &Gateway, app: &str, ctx: &Context<'_>, req: Request) -> Routed {
    if !gateway.hub.admitted(app) {
        return Routed::Dumb(req);
    }
    Routed::Answered(restarting(app, ctx.want_html))
}

/// Route `req` for `app`, clustered on this gateway.
pub async fn route(gateway: &Arc<Gateway>, app: &str, ctx: Context<'_>, req: Request) -> Routed {
    let hub = &gateway.hub;
    let Some(table) = hub.table(app) else {
        return none_ready(gateway, app, &ctx, req);
    };
    match addressed(&req, &ctx) {
        Err(p) => return Routed::Answered(errors::respond(*p, ctx.want_html)),
        Ok(Some(node)) => {
            if hub.member(app, node).is_none() {
                return Routed::Answered(errors::respond(
                    Problem::not_found(format!("no {app} member {node:016x} on this gateway")),
                    ctx.want_html,
                ));
            }
            return Routed::Answered(forward(gateway, app, node, &ctx, req).await);
        }
        Ok(None) => {}
    }

    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let (dispatch, params, route_key, mirror) = match table.lookup(&method, &path) {
        Some(m) => (
            m.route.dispatch.clone(),
            m.params,
            format!("{method} {}", m.route.path),
            m.route.mirror.clone(),
        ),
        None => (
            Dispatch::Ui,
            Vec::new(),
            format!("{method} UI"),
            Mirror::None,
        ),
    };
    let sample = Route {
        key: &route_key,
        mirror: &mirror,
    };
    let sources: Vec<KeySource> = match &dispatch {
        Dispatch::Sticky(keys) => keys.clone(),
        Dispatch::Owner(key) => vec![key.clone()],
        _ => Vec::new(),
    };
    let mut req = req;
    let mut key = match extract(&sources, &params, req.headers(), None) {
        Extracted::Key(k) => Some(k),
        Extracted::None => None,
        Extracted::NeedsBody => {
            let (parts, body) = req.into_parts();
            let bytes = match to_bytes(body, BODY_LIMIT).await {
                Ok(b) => b,
                Err(_) => {
                    return Routed::Answered(errors::respond(
                        Problem::too_large(format!(
                            "a request routed by its body may be at most {} MiB",
                            BODY_LIMIT >> 20
                        )),
                        ctx.want_html,
                    ))
                }
            };
            let key = match extract(&sources, &params, &parts.headers, Some(&bytes)) {
                Extracted::Key(k) => Some(k),
                _ => None,
            };
            req = Request::from_parts(parts, Body::from(bytes));
            key
        }
    };
    if key.as_deref() == Some("") {
        key = None;
    }

    let pin = pinned(&req, app);
    match hub.pick(app, &dispatch, key.as_deref(), pin) {
        Pick::Configured => none_ready(gateway, app, &ctx, req),
        Pick::Member(node) => {
            hub.traffic.request(Class::of(&dispatch));
            let to_candidate = match mirror {
                Mirror::None => None,
                _ => hub.mirror_to(app, node),
            };
            Routed::Answered(match to_candidate {
                Some(candidate) => mirrored(gateway, app, node, candidate, sample, &ctx, req).await,
                None => sampled(gateway, app, node, sample, &ctx, req).await,
            })
        }
        Pick::Pinned(node) => {
            hub.traffic.request(Class::Ui);
            let mut res = sampled(gateway, app, node, sample, &ctx, req).await;
            set_pin(&mut res, app, node, ctx.secure);
            Routed::Answered(res)
        }
        Pick::All(nodes) => {
            hub.traffic.request(Class::Aggregate);
            let Dispatch::Aggregate(merge) = dispatch else {
                unreachable!("only an aggregate picks every member")
            };
            Routed::Answered(match merge {
                Merge::Lines => merge_ws::merge(gateway, app, nodes, &ctx, req).await,
                m => aggregate::answer(gateway, app, nodes, m, &ctx, req).await,
            })
        }
    }
}

/// Send `req` to `node`, answering a holder's `409` with a `503` (§5.4).
pub async fn forward(
    gateway: &Arc<Gateway>,
    app: &str,
    node: u64,
    ctx: &Context<'_>,
    req: Request,
) -> Response {
    let res = proxy::forward(
        Forward {
            to: To::Member {
                hub: &gateway.hub,
                app,
                node,
            },
            rewrite_host: false,
            peer: ctx.peer,
            want_html: ctx.want_html,
            identity: ctx.identity,
            assertion: None,
            secure: ctx.secure,
            cache: ctx.cache,
            from_member: ctx.from_member,
        },
        req,
    )
    .await;
    if res.status() == StatusCode::CONFLICT {
        if let Some(holder) = res
            .headers()
            .get(HOLDER_HEADER)
            .and_then(|v| v.to_str().ok())
        {
            gateway.hub.traffic.holder_elsewhere();
            let holder = holder.to_string();
            return errors::respond(
                Problem::backing_off(
                    "This conversation is being served elsewhere; retrying.",
                    HOLDER_RETRY,
                )
                .because(format!(
                    "{app} member {node:016x} names {holder} as the holder, which has no socket \
                     on this gateway"
                )),
                ctx.want_html,
            );
        }
    }
    res
}
