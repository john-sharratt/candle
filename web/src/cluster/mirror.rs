//! Certification's samples (`docs/replicated_substrate.md` §3.8): each request a
//! member answers is a sample of its route — its failures and its latency, the
//! candidate's beside the baseline's — and a share of the baseline's requests on
//! a route declared `Exact` or `Shape` is mirrored to the candidate and the two
//! answers compared.
//!
//! The client is answered from the baseline: its answer is buffered to be
//! compared, and the copy to the candidate runs after, in the background, so the
//! candidate's latency is never the client's. Only an answer of known length
//! within [`MIRROR_BODY_MAX`] is buffered; any other is answered as it streams,
//! and not mirrored.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body, Bytes};
use axum::extract::Request;
use axum::http::{header, HeaderMap};
use axum::response::Response;
use cluster::dispatch::{Mirror, BODY_LIMIT};
use cluster::gateway::upgrades::mirror::{compare, Answer};

use super::route::{forward, Context};
use super::Gateway;
use crate::auth::{Identity, Roles};
use crate::config::Cache;
use crate::errors::{self, Problem, GATEWAY_ERROR};

/// The largest answer a mirrored call buffers.
pub const MIRROR_BODY_MAX: usize = 8 << 20;

/// Whether `res` failed: a 5xx, or the gateway's own failure to reach the
/// member.
fn failed(res: &Response) -> bool {
    res.status().is_server_error() || res.headers().contains_key(GATEWAY_ERROR)
}

/// The length an answer declares, when it is small enough to buffer.
fn bounded(headers: &HeaderMap) -> Option<usize> {
    headers
        .get(header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|n| *n <= MIRROR_BODY_MAX)
}

/// The route a request matched: its key, `METHOD /path/pattern`, and how it
/// is mirrored.
#[derive(Debug, Clone, Copy)]
pub struct Route<'a> {
    pub key: &'a str,
    pub mirror: &'a Mirror,
}

/// Forward `req` to `node` and record it as a sample of `route`.
pub async fn sampled(
    gateway: &Arc<Gateway>,
    app: &str,
    node: u64,
    route: Route<'_>,
    ctx: &Context<'_>,
    req: Request,
) -> Response {
    let started = Instant::now();
    let res = forward(gateway, app, node, ctx, req).await;
    gateway.hub.certify_sample(
        app,
        route.key,
        route.mirror,
        node,
        failed(&res),
        started.elapsed().as_millis() as u32,
    );
    res
}

/// What the background copy needs of the context, owned.
struct Owned {
    want_html: bool,
    peer: Option<IpAddr>,
    identity: Option<Identity>,
    secure: bool,
    cache: Cache,
    roles: Roles,
    from_member: Option<u64>,
}

/// Serve `req` from the baseline `node`, sampled, and mirror a copy to
/// `candidate`, comparing the two answers under the route's mode.
pub async fn mirrored(
    gateway: &Arc<Gateway>,
    app: &str,
    node: u64,
    candidate: u64,
    route: Route<'_>,
    ctx: &Context<'_>,
    req: Request,
) -> Response {
    let (parts, body) = req.into_parts();
    let Ok(sent) = to_bytes(body, BODY_LIMIT).await else {
        return errors::respond(
            Problem::too_large(format!(
                "a request on a mirrored route may be at most {} MiB",
                BODY_LIMIT >> 20
            )),
            ctx.want_html,
        );
    };
    let mut copy = Request::builder()
        .method(parts.method.clone())
        .uri(parts.uri.clone())
        .version(parts.version)
        .body(Body::from(sent.clone()))
        .expect("parts of a request build one");
    *copy.headers_mut() = parts.headers.clone();
    let req = Request::from_parts(parts, Body::from(sent));
    let res = sampled(gateway, app, node, route, ctx, req).await;
    let Some(len) = bounded(res.headers()) else {
        return res;
    };
    let (rparts, rbody) = res.into_parts();
    let base: Bytes = match to_bytes(rbody, len).await {
        Ok(b) => b,
        Err(e) => {
            return errors::respond(
                Problem::upstream_down("The answer was cut off.", None)
                    .because(format!("{app} member {node:016x}'s answer broke off: {e}")),
                ctx.want_html,
            );
        }
    };
    let status = rparts.status.as_u16();
    let answer = Response::from_parts(rparts, Body::from(base.clone()));
    let owned = Owned {
        want_html: ctx.want_html,
        peer: ctx.peer,
        identity: ctx.identity.cloned(),
        secure: ctx.secure,
        cache: ctx.cache,
        roles: ctx.roles.clone(),
        from_member: ctx.from_member,
    };
    let (gateway, app, route, mode) = (
        gateway.clone(),
        app.to_string(),
        route.key.to_string(),
        route.mirror.clone(),
    );
    tokio::spawn(async move {
        let ctx = Context {
            want_html: owned.want_html,
            peer: owned.peer,
            identity: owned.identity.as_ref(),
            secure: owned.secure,
            cache: owned.cache,
            roles: &owned.roles,
            from_member: owned.from_member,
        };
        let theirs = forward(&gateway, &app, candidate, &ctx, copy).await;
        let their_status = theirs.status().as_u16();
        let body = match bounded(theirs.headers()) {
            Some(len) => to_bytes(theirs.into_body(), len).await.unwrap_or_default(),
            None => Bytes::new(),
        };
        let compared = compare(
            &mode,
            Answer {
                status,
                body: &base,
            },
            Answer {
                status: their_status,
                body: &body,
            },
        );
        gateway
            .hub
            .mirrored(&app, &route, &mode, &compared, &base, &body);
    });
    answer
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::http::{HeaderValue, StatusCode};

    #[test]
    fn only_an_answer_of_known_small_length_is_buffered() {
        let mut h = HeaderMap::new();
        assert_eq!(bounded(&h), None, "a streamed answer");
        h.insert(header::CONTENT_LENGTH, HeaderValue::from_static("12"));
        assert_eq!(bounded(&h), Some(12));
        h.insert(
            header::CONTENT_LENGTH,
            HeaderValue::from_static("99999999999"),
        );
        assert_eq!(bounded(&h), None, "too large");
    }

    #[test]
    fn a_5xx_or_the_gateways_own_failure_is_a_failure() {
        let ok = Response::new(Body::empty());
        assert!(!failed(&ok));
        let mut bad = Response::new(Body::empty());
        *bad.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        assert!(failed(&bad));
        let mut down = Response::new(Body::empty());
        down.headers_mut()
            .insert(GATEWAY_ERROR, HeaderValue::from_static("upstream_backoff"));
        assert!(failed(&down));
    }
}
