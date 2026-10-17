//! A request every ready member answers (§5.5), each within 3 s.
//!
//! - **`Envelope`**: `{ "members": [ { node, host, build, status, body } ] }`,
//!   a member past its deadline listed with `status: 504, error: "deadline"`,
//!   so a partial view reads as partial. `/v1/status` also carries a `group`
//!   block: the floor, the model lock, the member count.
//! - **`FirstFound`**: the first `2xx` answer, unwrapped.
//!
//! Answered with `x-tokera-aggregate: 1`.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body, Bytes};
use axum::extract::Request;
use axum::http::request::Parts;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use cluster::dispatch::{Merge, BODY_LIMIT};
use futures::future::join_all;
use futures::stream::{FuturesUnordered, StreamExt};
use serde_json::{json, Value};
use tokio::time::timeout;

use super::route::{self, Context};
use super::Gateway;
use crate::errors::{self, Problem};

/// How long each member has to answer.
pub const DEADLINE: Duration = Duration::from_secs(3);

/// The header that marks an aggregated answer.
pub const AGGREGATE_HEADER: &str = "x-tokera-aggregate";

/// The most of one member's answer an aggregate reads.
const ANSWER_LIMIT: usize = 16 << 20;

/// The path whose envelope carries the `group` block.
const STATUS_PATH: &str = "/v1/status";

/// One member's answer, read whole.
struct Answer {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
}

fn copy(parts: &Parts, body: &Bytes) -> Request {
    let mut req = Request::builder()
        .method(parts.method.clone())
        .uri(parts.uri.clone())
        .version(parts.version)
        .body(Body::from(body.clone()))
        .expect("the parts came from a request");
    *req.headers_mut() = parts.headers.clone();
    // Each answer is read here, and an envelope parses it: asked for plain, it
    // is compressed once, on the way out of this gateway.
    req.headers_mut().remove(header::ACCEPT_ENCODING);
    req
}

async fn ask(
    gateway: &Arc<Gateway>,
    app: &str,
    node: u64,
    ctx: &Context<'_>,
    req: Request,
) -> Option<Answer> {
    let asked = async {
        let res = route::forward(gateway, app, node, ctx, req).await;
        let (parts, body) = res.into_parts();
        let body = to_bytes(body, ANSWER_LIMIT).await.unwrap_or_default();
        Answer {
            status: parts.status,
            headers: parts.headers,
            body,
        }
    };
    timeout(DEADLINE, asked).await.ok()
}

/// Ask every member in `nodes`, combining their answers by `merge`.
pub async fn answer(
    gateway: &Arc<Gateway>,
    app: &str,
    nodes: Vec<u64>,
    merge: Merge,
    ctx: &Context<'_>,
    req: Request,
) -> Response {
    let path = req.uri().path().to_string();
    let (parts, body) = req.into_parts();
    let Ok(body) = to_bytes(body, BODY_LIMIT).await else {
        return errors::respond(
            Problem::too_large(format!(
                "a request every member answers may be at most {} MiB",
                BODY_LIMIT >> 20
            )),
            ctx.want_html,
        );
    };
    match merge {
        Merge::FirstFound => first_found(gateway, app, nodes, ctx, &parts, &body).await,
        Merge::Union => union(gateway, app, nodes, ctx, &parts, &body).await,
        _ => envelope(gateway, app, nodes, ctx, &parts, &body, &path).await,
    }
}

/// Every member's `2xx` JSON answer as one body: each top-level array merged
/// by its items' `id`, the copy further along kept (see [`further_along`]).
/// A member rejoining after writing cut off lists what no other member has
/// yet, so an `Any` listing answered by one member would miss it. With no
/// `2xx` answer, the first answer comes back as it was; with none at all, the
/// app is down for a second, as [`first_found`] says it.
async fn union(
    gateway: &Arc<Gateway>,
    app: &str,
    nodes: Vec<u64>,
    ctx: &Context<'_>,
    parts: &Parts,
    body: &Bytes,
) -> Response {
    let asks = nodes.iter().map(|&node| {
        let req = copy(parts, body);
        async move { ask(gateway, app, node, ctx, req).await }
    });
    let answers = join_all(asks).await;
    let late = answers.iter().filter(|a| a.is_none()).count() as u64;
    gateway.hub.traffic.aggregate(late);
    let bodies: Vec<Value> = answers
        .iter()
        .flatten()
        .filter(|a| a.status.is_success())
        .filter_map(|a| serde_json::from_slice::<Value>(&a.body).ok())
        .collect();
    if bodies.is_empty() {
        let Some(first) = answers.into_iter().flatten().next() else {
            return errors::respond(
                Problem::upstream_down(
                    format!("no {app} member answered it"),
                    Some(Duration::from_secs(1)),
                ),
                ctx.want_html,
            );
        };
        let mut res = Response::new(Body::from(first.body));
        *res.status_mut() = first.status;
        *res.headers_mut() = first.headers;
        return res;
    }
    let mut res = Response::new(Body::from(merge_by_id(bodies).to_string()));
    let h = res.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    h.insert(AGGREGATE_HEADER, HeaderValue::from_static("1"));
    res
}

/// `bodies` as one: the first body's fields, each top-level array the union of
/// every body's by item `id` in first-seen order, a duplicate replaced by the
/// copy [`further_along`]. An item with no `id` is kept as it comes.
fn merge_by_id(bodies: Vec<Value>) -> Value {
    let mut bodies = bodies.into_iter();
    let Some(mut out) = bodies.next() else {
        return Value::Null;
    };
    for body in bodies {
        let (Value::Object(into), Value::Object(from)) = (&mut out, body) else {
            continue;
        };
        for (field, value) in from {
            match (into.get_mut(&field), value) {
                (Some(Value::Array(have)), Value::Array(more)) => {
                    for item in more {
                        let id = item.get("id").cloned().filter(|v| !v.is_null());
                        let at = id
                            .as_ref()
                            .and_then(|id| have.iter().position(|h| h.get("id") == Some(id)));
                        match at {
                            Some(i) if further_along(&item, &have[i]) => have[i] = item,
                            Some(_) => {}
                            None => have.push(item),
                        }
                    }
                }
                (None, value) => {
                    into.insert(field, value);
                }
                _ => {}
            }
        }
    }
    out
}

/// Whether `a` is further along than `b`: more turns, then a later place in
/// the listing.
fn further_along(a: &Value, b: &Value) -> bool {
    let key = |v: &Value| {
        (
            v.get("turn_count").and_then(Value::as_u64).unwrap_or(0),
            v.get("updated_ms").and_then(Value::as_u64).unwrap_or(0),
        )
    };
    key(a) > key(b)
}

async fn envelope(
    gateway: &Arc<Gateway>,
    app: &str,
    nodes: Vec<u64>,
    ctx: &Context<'_>,
    parts: &Parts,
    body: &Bytes,
    path: &str,
) -> Response {
    let asks = nodes.iter().map(|&node| {
        let req = copy(parts, body);
        async move { (node, ask(gateway, app, node, ctx, req).await) }
    });
    let answers = join_all(asks).await;
    let mut late = 0;
    let members: Vec<Value> = answers
        .into_iter()
        .map(|(node, answer)| {
            let view = gateway.hub.member(app, node);
            let host = view.as_ref().map_or(String::new(), |m| m.host.clone());
            let build = view.as_ref().map_or(0, |m| m.build.count);
            let node = format!("{node:016x}");
            match answer {
                Some(a) => {
                    let body = serde_json::from_slice::<Value>(&a.body).unwrap_or_else(|_| {
                        Value::String(String::from_utf8_lossy(&a.body).into_owned())
                    });
                    json!({ "node": node, "host": host, "build": build,
                            "status": a.status.as_u16(), "body": body })
                }
                None => {
                    late += 1;
                    json!({ "node": node, "host": host, "build": build,
                            "status": 504, "error": "deadline" })
                }
            }
        })
        .collect();
    gateway.hub.traffic.aggregate(late);
    let mut out = json!({ "members": members });
    if path == STATUS_PATH {
        // Who answered for the members, and on which build: a member's status
        // names its own, and nothing else public named the gateway's.
        let b = cluster::version::this_build();
        out["gateway"] = json!({
            "host": gateway.hub.name(),
            "node": format!("{:016x}", gateway.node),
            "build": { "count": b.count, "hash": b.hash, "local": b.local },
        });
        let g = gateway.hub.group(app);
        out["group"] = json!({
            "floor": g.floor.map(|f| json!({ "count": f.count, "hash": f.hash, "local": f.local })),
            "model_lock": g.model_lock,
            "members": g.members,
        });
    }
    let mut res = Response::new(Body::from(out.to_string()));
    let h = res.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    h.insert(AGGREGATE_HEADER, HeaderValue::from_static("1"));
    res
}

async fn first_found(
    gateway: &Arc<Gateway>,
    app: &str,
    nodes: Vec<u64>,
    ctx: &Context<'_>,
    parts: &Parts,
    body: &Bytes,
) -> Response {
    let mut asks: FuturesUnordered<_> = nodes
        .iter()
        .map(|&node| {
            let req = copy(parts, body);
            ask(gateway, app, node, ctx, req)
        })
        .collect();
    let mut late = 0;
    let mut not_found = 0;
    let asked = nodes.len();
    while let Some(answer) = asks.next().await {
        match answer {
            // The members still answering are not waited for.
            Some(a) if a.status.is_success() => {
                gateway.hub.traffic.aggregate(late);
                let mut res = Response::new(Body::from(a.body));
                *res.status_mut() = a.status;
                *res.headers_mut() = a.headers;
                res.headers_mut()
                    .insert(AGGREGATE_HEADER, HeaderValue::from_static("1"));
                return res;
            }
            Some(a) if a.status == StatusCode::NOT_FOUND => not_found += 1,
            Some(_) => {}
            None => late += 1,
        }
    }
    gateway.hub.traffic.aggregate(late);
    let problem = if not_found == asked {
        Problem::not_found(format!("none of the {asked} {app} members has it"))
    } else {
        Problem::upstream_down(
            format!("no {app} member answered it"),
            Some(Duration::from_secs(1)),
        )
    };
    errors::respond(problem, ctx.want_html)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every member's conversations, once each.** A member rejoining lists
    /// one no other member has yet; a conversation two members list is kept
    /// as the copy further along.
    #[test]
    fn lists_merge_by_id_keeping_the_copy_further_along() {
        let in_sync = json!({ "conversations": [
            { "id": "a", "turn_count": 4, "updated_ms": 9 },
            { "id": "b", "turn_count": 2, "updated_ms": 5 },
        ]});
        let rejoining = json!({ "conversations": [
            { "id": "b", "turn_count": 3, "updated_ms": 6 },
            { "id": "c", "turn_count": 1, "updated_ms": 7 },
            { "id": "a", "turn_count": 1, "updated_ms": 1 },
        ]});
        let merged = merge_by_id(vec![in_sync, rejoining]);
        let ids: Vec<&str> = merged["conversations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert_eq!(
            merged["conversations"][0]["turn_count"], 4,
            "the in-sync copy is further along"
        );
        assert_eq!(
            merged["conversations"][1]["turn_count"], 3,
            "the rejoining copy is further along"
        );
    }
}
