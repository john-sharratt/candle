//! A member's jobs, carried to an agent (§11.3, §11.5). A member opens a
//! stream towards `group:agent`; each request on it goes to the agent
//! [`Hub::pick_agent`](cluster::gateway::Hub::pick_agent) chooses — the
//! lineage's holder, or after [`SPILL_AFTER`] the best idle agent holding the
//! job's repository and toolchain — or to the agent `x-tokera-node` names,
//! which is how a member reads back a job it started. With no agent fit for
//! the job it is answered `503` with the code `no_agent`.

use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use cluster::gateway::{
    Choice, Need, LINEAGE_HEADER, NODE_HEADER, PUSH_HEADER, REPO_HEADER, SPILL_AFTER,
    TOOLCHAIN_HEADER,
};
use serde_json::json;
use tokio::time::sleep;

use super::Gateway;
use crate::config::Cache;
use crate::proxy::{self, Forward, To};

/// The app a job stream reaches.
pub const AGENT_APP: &str = "agent";

/// How often a waiting job looks at the agents again.
const LOOK_EVERY: Duration = Duration::from_millis(100);

fn header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty())
}

/// What the job in `headers` needs of its agent.
pub fn need_of(headers: &HeaderMap) -> Need {
    Need {
        repo: header(headers, REPO_HEADER).unwrap_or_default().to_string(),
        toolchain: header(headers, TOOLCHAIN_HEADER).map(str::to_string),
        push: header(headers, PUSH_HEADER) == Some("1"),
    }
}

/// The `503` a job no agent can take is answered with.
pub fn no_agent(why: String) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({ "error": "no_agent", "message": why })),
    )
        .into_response()
}

/// Send one request from member `from` (at `peer`) to an agent.
pub async fn answer(gateway: &Arc<Gateway>, from: u64, peer: IpAddr, req: Request) -> Response {
    let hub = &gateway.hub;
    let addressed =
        header(req.headers(), NODE_HEADER).and_then(|n| u64::from_str_radix(n, 16).ok());
    let node = if let Some(node) = addressed {
        if hub.member(AGENT_APP, node).is_none() {
            return no_agent(format!("agent {node:016x} has no socket on this gateway"));
        }
        node
    } else {
        let need = need_of(req.headers());
        let lineage = header(req.headers(), LINEAGE_HEADER).map(str::to_string);
        let start = Instant::now();
        loop {
            let waited = start.elapsed() >= SPILL_AFTER;
            match hub.pick_agent(AGENT_APP, lineage.as_deref(), &need, waited) {
                Choice::Run(node) => break node,
                Choice::Wait => sleep(LOOK_EVERY).await,
                Choice::NoAgent => {
                    return no_agent(format!(
                        "no ready agent holds repository `{}`{}",
                        need.repo,
                        need.toolchain
                            .as_ref()
                            .map_or(String::new(), |t| format!(" with `{t}`"))
                    ))
                }
            }
        }
    };
    let mut res = proxy::forward(
        Forward {
            to: To::Member {
                hub,
                app: AGENT_APP,
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
    .await;
    // The member reads the job back from the agent that ran it.
    if let Ok(v) = HeaderValue::from_str(&format!("{node:016x}")) {
        res.headers_mut().insert(NODE_HEADER, v);
    }
    res
}
