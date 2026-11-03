//! A member's jobs, carried to an agent (§11.3, §11.5). A member opens a
//! stream towards `group:agent`; each request on it goes at once to the agent
//! [`Hub::pick_agent`](cluster::gateway::Hub::pick_agent) chooses — the
//! lineage's holder while it is free, else the best idle agent holding the
//! job's repository and toolchain — or to the agent `x-tokera-node` names,
//! which is how a member reads back a job it started — and how an admin's
//! member reads what an agent is running, or switches it (§11.7). With no
//! agent fit for the job it is answered `503` with the code `no_agent`.
//!
//! `GET` [`AGENTS_PATH`] is the gateway's own to answer: the agents admitted
//! here, from its registry.

use std::net::IpAddr;
use std::sync::Arc;

use axum::extract::Request;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use cluster::gateway::{
    Choice, Need, AGENTS_PATH, LINEAGE_HEADER, NODE_HEADER, PUSH_HEADER, REPO_HEADER,
    TOOLCHAIN_HEADER,
};
use serde_json::json;

use super::Gateway;
use crate::config::Cache;
use crate::proxy::{self, Forward, To};

/// The app a job stream reaches.
pub const AGENT_APP: &str = "agent";

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

/// Why no agent fits a job needing `need`, with what the agents holding its
/// repository do start: told only that `python3` had no agent, a worker went on
/// to try `od`, `magick` and `git`, a round each.
fn unfit(need: &Need, programs: &[String]) -> String {
    let wanted = need
        .toolchain
        .as_ref()
        .map_or(String::new(), |t| format!(" with `{t}`"));
    let why = format!("no ready agent holds repository `{}`{wanted}", need.repo);
    if programs.is_empty() || need.toolchain.is_none() {
        return why;
    }
    format!(
        "{why}; the programs that run in it: {}",
        programs.join(", ")
    )
}

/// Send one request from member `from` (at `peer`) to an agent.
pub async fn answer(gateway: &Arc<Gateway>, from: u64, peer: IpAddr, req: Request) -> Response {
    let hub = &gateway.hub;
    if req.method() == Method::GET && req.uri().path() == AGENTS_PATH {
        return Json(hub.agents(AGENT_APP)).into_response();
    }
    let addressed =
        header(req.headers(), NODE_HEADER).and_then(|n| u64::from_str_radix(n, 16).ok());
    let node = if let Some(node) = addressed {
        if hub.member(AGENT_APP, node).is_none() {
            return no_agent(format!("agent {node:016x} has no socket on this gateway"));
        }
        node
    } else {
        let need = need_of(req.headers());
        let lineage = header(req.headers(), LINEAGE_HEADER);
        match hub.pick_agent(AGENT_APP, lineage, &need) {
            Choice::Run(node) => node,
            Choice::NoAgent => {
                return no_agent(unfit(&need, &hub.agent_programs(AGENT_APP, &need.repo)))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn need(repo: &str, toolchain: Option<&str>) -> Need {
        Need {
            repo: repo.into(),
            toolchain: toolchain.map(str::to_string),
            push: false,
        }
    }

    /// **A program no agent has is refused with those that run**; a git job,
    /// which names none, or a repository nobody holds has nothing to offer.
    #[test]
    fn a_job_with_no_agent_is_told_what_runs() {
        let running = ["cargo".to_string(), "python".to_string()];
        assert_eq!(
            unfit(&need("candle", Some("python3")), &running),
            "no ready agent holds repository `candle` with `python3`; the programs that run in \
             it: cargo, python"
        );
        assert_eq!(
            unfit(&need("candle", None), &running),
            "no ready agent holds repository `candle`"
        );
        assert_eq!(
            unfit(&need("gone", Some("cargo")), &[]),
            "no ready agent holds repository `gone` with `cargo`"
        );
    }
}
