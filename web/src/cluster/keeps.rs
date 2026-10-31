//! A member's requests to a machine's `keep` (`docs/claude_zend_gateway.md`
//! §8.4). A member opens a stream towards `group:keep`; each request on it
//! goes to the keep of the machine [`HOST_HEADER`] names — keep runs once per
//! machine — or to the keep `x-tokera-node` names. `GET` [`KEEPS_PATH`] is the
//! gateway's own to answer: the keeps admitted here.
//!
//! The member vouches for its caller, as it does on every stream; keep serves
//! only a request stamped with the member it came from.

use std::net::IpAddr;
use std::sync::Arc;

use axum::extract::Request;
use axum::http::{HeaderMap, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use cluster::gateway::{AgentRow, KEEP, NODE_HEADER};
use serde_json::json;

use super::Gateway;
use crate::config::Cache;
use crate::proxy::{self, Forward, To};

/// The machine a request is for.
pub const HOST_HEADER: &str = "x-tokera-host";

/// The keeps admitted on this gateway.
pub const KEEPS_PATH: &str = "/v1/keeps";

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty())
}

fn refuse(status: StatusCode, detail: String) -> Response {
    (status, Json(json!({"error": "no_keep", "detail": detail}))).into_response()
}

/// The keep a request names among `keeps`: by node, or by its machine.
pub fn named(keeps: &[AgentRow], headers: &HeaderMap) -> Result<u64, String> {
    if let Some(n) = header(headers, NODE_HEADER) {
        let node = u64::from_str_radix(n, 16).map_err(|_| format!("{n:?} is not a node"))?;
        return keeps
            .iter()
            .any(|k| k.node == format!("{node:016x}"))
            .then_some(node)
            .ok_or_else(|| format!("keep {node:016x} has no socket on this gateway"));
    }
    let Some(host) = header(headers, HOST_HEADER) else {
        return Err(format!("name the machine in `{HOST_HEADER}`"));
    };
    let keep = keeps
        .iter()
        .find(|k| k.host.eq_ignore_ascii_case(host))
        .ok_or_else(|| {
            let known: Vec<&str> = keeps.iter().map(|k| k.host.as_str()).collect();
            format!(
                "no keep on {host} has a socket on this gateway; keeps here: {}",
                if known.is_empty() {
                    "none".to_string()
                } else {
                    known.join(", ")
                }
            )
        })?;
    u64::from_str_radix(&keep.node, 16).map_err(|_| format!("{:?} is not a node", keep.node))
}

/// Send one request from member `from` (at `peer`) to a keep.
pub async fn answer(gateway: &Arc<Gateway>, from: u64, peer: IpAddr, req: Request) -> Response {
    let hub = &gateway.hub;
    let keeps = hub.agents(KEEP);
    if req.method() == Method::GET && req.uri().path() == KEEPS_PATH {
        return Json(keeps).into_response();
    }
    let node = match named(&keeps, req.headers()) {
        Ok(n) => n,
        Err(why) => return refuse(StatusCode::NOT_FOUND, why),
    };
    proxy::forward(
        Forward {
            to: To::Member {
                hub,
                app: KEEP,
                node,
            },
            rewrite_host: false,
            peer: Some(peer),
            want_html: false,
            identity: None,
            assertion: None,
            secure: false,
            cache: Cache::default(),
            from_member: Some(from),
        },
        req,
    )
    .await
}

#[cfg(test)]
mod tests {
    use axum::http::HeaderValue;
    use cluster::version::this_build;

    use super::*;

    fn keep(host: &str, node: u64) -> AgentRow {
        AgentRow {
            node: format!("{node:016x}"),
            host: host.into(),
            build: this_build(),
            state: "ready".into(),
            os: String::new(),
            cores: 0,
            load_milli: 0,
            mem_free_mib: 0,
            slots_free: 0,
            jobs: 0,
            toolchains: Vec::new(),
            low_priority: false,
            standby: false,
            repos: Vec::new(),
            connected_ms: 0,
        }
    }

    #[test]
    fn a_keep_is_named_by_its_machine_or_its_node() {
        let keeps = [keep("BlackWorld", 0xab), keep("whiteworld", 0xcd)];
        let mut h = HeaderMap::new();
        h.insert(HOST_HEADER, HeaderValue::from_static("blackworld"));
        assert_eq!(named(&keeps, &h), Ok(0xab));
        let mut h = HeaderMap::new();
        h.insert(NODE_HEADER, HeaderValue::from_static("00000000000000cd"));
        assert_eq!(named(&keeps, &h), Ok(0xcd));
        let mut h = HeaderMap::new();
        h.insert(HOST_HEADER, HeaderValue::from_static("aw"));
        assert_eq!(
            named(&keeps, &h).unwrap_err(),
            "no keep on aw has a socket on this gateway; keeps here: BlackWorld, whiteworld"
        );
        assert!(named(&keeps, &HeaderMap::new())
            .unwrap_err()
            .contains(HOST_HEADER));
    }
}
