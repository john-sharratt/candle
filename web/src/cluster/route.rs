//! Where a request for a clustered app goes (§5.2): the dispatch its route
//! declares in the members' table, its key, the member the hub picks, and the
//! answer — forwarded to one member, combined from all, handed back to go to
//! the configured upstream when no member is admitted, or — while every
//! admitted member is still loading — forwarded whole to the one that joined
//! first, whose loading page and own endpoints stand in for the app.
//!
//! Every request a member answers is a sample for certifying a candidate
//! (§3.8): while one certifies it takes its share of the traffic, and a share
//! of the baseline's requests on a mirrored route are mirrored to it
//! ([`super::mirror`]).
//!
//! Four things beside the dispatch:
//!
//! - **A draw's reach.** A route declared `Reach::Draw` has its body read for
//!   the draw's `width × height`, and is placed only among the members whose
//!   card reaches that many pixels (`cluster::dispatch::Reach`).
//! - **UI pinning.** A page's files come from one member: the first UI request
//!   is dealt round robin and answered with `tokera_ui_<app>`, and every UI
//!   request carrying it goes to that member while it is ready.
//! - **`x-tokera-node`**, an admin's: the request goes to that member whatever
//!   the route declares.
//! - **A key held through another cluster.** A member that knows the key's
//!   holder answers `409` naming it (§5.4); the gateway answers `503` with
//!   `Retry-After`, because the next attempt may land on a gateway the holder
//!   is joined to.
//! - **No card with room.** A request every eligible card is too full for
//!   waits at the gateway, and is answered `503` with `Retry-After` if none
//!   has room within the hold (`docs/cooperative_admission.md` §4.3). Each
//!   request is named by an id in a `place` span, so its placement lines and
//!   its `answered` line join (§4.6).

use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::Response;
use cluster::dispatch::{
    draw_area, extract, Dispatch, Extracted, KeySource, Merge, Mirror, Reach, BODY_LIMIT,
};
use cluster::gateway::{Class, Pick, Placing, HOLD, NODE_HEADER};
use cluster::member::PREFER_PEER;
use serde_json::Value;
use tracing::Instrument;

use super::load::{carry, Answering};
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

/// The node `x-tokera-node` names, on an admin's request or a member's — a
/// member, proven by the group key, hands work to a peer it chose (a worker
/// moving to a faster card); `Err` refuses the request. Boxed because a
/// `Problem` is large and this is the cold path.
fn addressed(req: &Request, ctx: &Context<'_>) -> Result<Option<u64>, Box<Problem>> {
    let Some(raw) = req.headers().get(NODE_HEADER) else {
        return Ok(None);
    };
    if ctx.from_member.is_none() && !ctx.roles.of(ctx.identity).at_least(Role::Admin) {
        return Err(Box::new(Problem::forbidden(format!(
            "`{NODE_HEADER}` addresses one member, which is an admin's or a member's to do"
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

/// How long a client waits before asking again after every card stayed full
/// for the whole hold.
const FULL_RETRY: Duration = Duration::from_secs(5);

/// The gateway's request ids, which name each request's placement lines.
static REQUESTS: AtomicU64 = AtomicU64::new(0);

/// The answer to a request no card had room for within the hold (§4.3).
fn full(app: &str, want_html: bool) -> Response {
    errors::respond(
        Problem::backing_off("Every machine is busy — retrying in a moment.", FULL_RETRY).because(
            format!(
                "no {app} member's card had room for the request within {} s",
                HOLD.as_secs()
            ),
        ),
        want_html,
    )
}

/// The gateway's own "reconnecting" page, which the consoles retry: the answer
/// when `app` is admitted here yet has no member to send a request to — every
/// one gone between the check and the pick.
pub fn restarting(app: &str, want_html: bool) -> Response {
    errors::respond(
        Problem::backing_off("Restarting — back in a moment.", LOADING_RETRY).because(format!(
            "no {app} member admitted here can take the request"
        )),
        want_html,
    )
}

/// No member of `app` is ready: its configured upstream when none is admitted
/// (§5.7); otherwise the member that joined first, still loading, answers it —
/// its loading page, and its own endpoints, which the performance and agents
/// pages read while the app is down. [`restarting`] only when no admitted member
/// is left to stand by.
async fn none_ready(gateway: &Arc<Gateway>, app: &str, ctx: &Context<'_>, req: Request) -> Routed {
    if !gateway.hub.admitted(app) {
        return Routed::Dumb(req);
    }
    match gateway.hub.standing_by(app) {
        Some(node) => Routed::Answered(forward(gateway, app, node, ctx, req).await),
        None => Routed::Answered(restarting(app, ctx.want_html)),
    }
}

/// Route `req` for `app`, clustered on this gateway.
pub async fn route(gateway: &Arc<Gateway>, app: &str, ctx: Context<'_>, req: Request) -> Routed {
    let hub = &gateway.hub;
    let Some(table) = hub.table(app) else {
        return none_ready(gateway, app, &ctx, req).await;
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
    let (dispatch, params, route_key, mirror, reach) = match table.lookup(&method, &path) {
        Some(m) => (
            m.route.dispatch.clone(),
            m.params,
            format!("{method} {}", m.route.path),
            m.route.mirror.clone(),
            m.route.reach,
        ),
        None => (
            Dispatch::Ui,
            Vec::new(),
            format!("{method} UI"),
            Mirror::None,
            Reach::Any,
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
    let first = extract(&sources, &params, req.headers(), None);
    // The body is read for a key a later source finds in it, and for the size
    // of a draw, which is placed only where a card reaches it.
    let (mut key, draw) = if first == Extracted::NeedsBody || reach != Reach::Any {
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
        let key = match first {
            Extracted::Key(k) => Some(k),
            Extracted::None => None,
            Extracted::NeedsBody => {
                match extract(&sources, &params, &parts.headers, Some(&bytes)) {
                    Extracted::Key(k) => Some(k),
                    _ => None,
                }
            }
        };
        let draw = match reach {
            Reach::Any => None,
            Reach::Draw => serde_json::from_slice::<Value>(&bytes)
                .ok()
                .and_then(|body| draw_area(&body)),
        };
        req = Request::from_parts(parts, Body::from(bytes));
        (key, draw)
    } else {
        let key = match first {
            Extracted::Key(k) => Some(k),
            _ => None,
        };
        (key, None)
    };
    if key.as_deref() == Some("") {
        key = None;
    }

    let pin = pinned(&req, app);
    // A member that asks for a peer gets one when one can take it
    // (`docs/guest_draw_orchestration.md` §3); any other request is placed as
    // a client's is.
    let peer_of = ctx
        .from_member
        .filter(|_| req.headers().contains_key(PREFER_PEER));
    let id = REQUESTS.fetch_add(1, Ordering::Relaxed) + 1;
    let started = Instant::now();
    let placing = Placing {
        req: id,
        app: app.to_string(),
        dispatch: dispatch.clone(),
        draw,
        key: key.clone(),
        pinned: pin,
        sender: peer_of,
    };
    let span = tracing::info_span!("place", req = id, app, dispatch = dispatch.name());
    let Some(picked) = hub.place(placing).instrument(span).await else {
        return Routed::Answered(full(app, ctx.want_html));
    };
    match picked.pick {
        Pick::Configured => none_ready(gateway, app, &ctx, req).await,
        Pick::Wait => Routed::Answered(full(app, ctx.want_html)),
        Pick::Member(node) => {
            hub.traffic.request(Class::of(&dispatch));
            // Never a member's ask for a peer: an unproven build is not handed
            // the members' own traffic, as a shadow or otherwise.
            let to_candidate = match mirror {
                Mirror::None => None,
                _ if peer_of.is_some() => None,
                _ => hub.mirror_to(app, node),
            };
            let res = match to_candidate {
                Some(candidate) => mirrored(gateway, app, node, candidate, sample, &ctx, req).await,
                None => sampled(gateway, app, node, sample, &ctx, req).await,
            };
            let answering = Answering {
                req: id,
                app: app.to_string(),
                node,
                started,
            };
            Routed::Answered(carry(res, picked.load, answering))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(roles: &Roles, from_member: Option<u64>) -> Context<'_> {
        Context {
            want_html: false,
            peer: None,
            identity: None,
            secure: false,
            cache: Cache::default(),
            roles,
            from_member,
        }
    }

    fn naming(node: &str) -> Request {
        Request::builder()
            .uri("/v1/mcp/worker/continue")
            .header(NODE_HEADER, node)
            .body(Body::empty())
            .unwrap()
    }

    /// **A member addresses a peer by node, as an admin does**; anyone else
    /// is refused, and a request that names nothing is placed as ever.
    #[test]
    fn a_member_or_an_admin_addresses_one_member() {
        let roles = Roles::default();
        assert_eq!(
            addressed(&naming("00000000000000ab"), &ctx(&roles, Some(0xcd))).ok(),
            Some(Some(0xab))
        );
        assert!(addressed(&naming("00000000000000ab"), &ctx(&roles, None)).is_err());
        assert!(addressed(&naming("not a node"), &ctx(&roles, Some(0xcd))).is_err());
        let plain = Request::builder().uri("/").body(Body::empty()).unwrap();
        assert_eq!(addressed(&plain, &ctx(&roles, None)).ok(), Some(None));
    }
}
