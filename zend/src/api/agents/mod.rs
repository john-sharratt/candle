//! The agents page's routes (`docs/zend_ui_redesign.md` §2.8;
//! `docs/replicated_substrate.md` §11.7) — an admin's only, since they show
//! every conversation's commands and switch the machines that run them.
//!
//!   GET    /v1/agents                           every agent, with its newest jobs
//!   PUT    /v1/agents/{node}                    `{enabled}`: switch it on or off
//!   DELETE /v1/agents/{node}/jobs/{id}          cancel a job, killing its processes
//!   GET    /v1/agents/{node}/jobs/{id}/stream   SSE: the job's output, live
//!
//! The agents are the ones this daemon's commands run on: those its gateway
//! admitted, or the agent in its own process while it is cut off. A daemon
//! without `--cluster` runs commands on its own sandboxes and has none, which
//! the listing says (`clustered: false`) rather than refusing.
//!
//! The listing reads each agent's jobs side by side, each bounded by
//! [`ASK_WITHIN`]: an agent that does not answer is listed with its `error`
//! and the gateway's view of it, never holding up the rest. Output is read
//! only through the stream, which the page holds open while a job is expanded.

mod tail;
mod utf8;

use std::sync::Arc;
use std::time::Duration;

use agent::client::{AgentError, Agents};
use agent::wire::{JobLine, Switching};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use cluster::gateway::AgentRow;
use futures::future::join_all;
use serde::Serialize;
use serde_json::json;
use tokio::time::timeout;

use crate::session::ZendSession;

/// The jobs listed for each agent.
pub const RECENT: usize = 25;

/// How long the listing waits on any one agent.
pub const ASK_WITHIN: Duration = Duration::from_secs(3);

/// `GET /v1/agents`.
#[derive(Debug, Serialize)]
pub struct AgentsBody {
    /// The daemon runs its commands on agents (`--cluster`).
    pub clustered: bool,
    /// One of its sockets is admitted, so the gateway's agents are in reach.
    pub admitted: bool,
    pub agents: Vec<AgentEntry>,
}

/// One agent: the gateway's row, and what the agent says it is running.
#[derive(Debug, Serialize)]
pub struct AgentEntry {
    #[serde(flatten)]
    pub agent: AgentRow,
    /// Whether it takes jobs.
    pub enabled: bool,
    /// Its job slots; `None` when the agent did not answer.
    pub slots: Option<u32>,
    /// Its newest jobs, newest first.
    pub recent: Vec<JobLine>,
    /// Why its jobs could not be read.
    pub error: Option<String>,
}

fn refusal(status: StatusCode, code: &str, message: String) -> Response {
    (status, Json(json!({ "error": code, "message": message }))).into_response()
}

fn failed(e: AgentError) -> Response {
    match e {
        AgentError::NoAgent(why) => refusal(StatusCode::NOT_FOUND, "no_agent", why),
        AgentError::Refused { status, why } => refusal(
            StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
            "refused",
            why,
        ),
        AgentError::Transport(why) => refusal(StatusCode::BAD_GATEWAY, "unreachable", why),
    }
}

/// Why a request names no agent this daemon can address.
#[derive(Debug)]
struct NoTarget {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl IntoResponse for NoTarget {
    fn into_response(self) -> Response {
        refusal(self.status, self.code, self.message)
    }
}

/// The node a path names, as sixteen hex digits.
fn node_of(node: &str) -> Result<u64, NoTarget> {
    u64::from_str_radix(node, 16).map_err(|_| NoTarget {
        status: StatusCode::BAD_REQUEST,
        code: "bad_node",
        message: format!("`{node}` is not an agent's node"),
    })
}

/// The agents, and the node a path names among them.
fn target(session: &ZendSession, node: &str) -> Result<(Agents, u64), NoTarget> {
    let node = node_of(node)?;
    let agents = session.agents().ok_or_else(|| NoTarget {
        status: StatusCode::NOT_FOUND,
        code: "no_agent",
        message: "this daemon runs its commands on its own sandboxes, not on agents".into(),
    })?;
    Ok((agents, node))
}

/// `row` with what its agent says, or why it said nothing.
async fn entry(agents: &Agents, row: AgentRow) -> AgentEntry {
    let listed = match u64::from_str_radix(&row.node, 16) {
        Ok(node) => timeout(ASK_WITHIN, agents.jobs_of(agents.at(node), RECENT))
            .await
            .unwrap_or_else(|_| {
                Err(AgentError::Transport(format!(
                    "no answer within {}s",
                    ASK_WITHIN.as_secs()
                )))
            }),
        Err(_) => Err(AgentError::Transport(format!(
            "`{}` is not a node",
            row.node
        ))),
    };
    match listed {
        Ok(jobs) => AgentEntry {
            enabled: jobs.enabled,
            slots: Some(jobs.slots),
            recent: jobs.jobs,
            error: None,
            agent: row,
        },
        Err(e) => AgentEntry {
            enabled: row.state != "disabled",
            slots: None,
            recent: Vec::new(),
            error: Some(e.to_string()),
            agent: row,
        },
    }
}

pub async fn list(State(session): State<Arc<ZendSession>>) -> Response {
    let Some(agents) = session.agents() else {
        return Json(AgentsBody {
            clustered: false,
            admitted: false,
            agents: Vec::new(),
        })
        .into_response();
    };
    let rows = match agents.listing().await {
        Ok(rows) => rows,
        Err(e) => return failed(e),
    };
    let agents_ref = &agents;
    let entries = join_all(rows.into_iter().map(|row| entry(agents_ref, row))).await;
    Json(AgentsBody {
        clustered: true,
        admitted: agents.admitted(),
        agents: entries,
    })
    .into_response()
}

pub async fn switch(
    State(session): State<Arc<ZendSession>>,
    Path(node): Path<String>,
    Json(to): Json<Switching>,
) -> Response {
    let (agents, node) = match target(&session, &node) {
        Ok(t) => t,
        Err(r) => return r.into_response(),
    };
    match agents.switch(agents.at(node), to.enabled).await {
        Ok(enabled) => {
            tracing::info!(node = %format!("{node:016x}"), enabled, "an admin switched an agent");
            Json(Switching { enabled }).into_response()
        }
        Err(e) => failed(e),
    }
}

pub async fn cancel(
    State(session): State<Arc<ZendSession>>,
    Path((node, id)): Path<(String, String)>,
) -> Response {
    let (agents, node) = match target(&session, &node) {
        Ok(t) => t,
        Err(r) => return r.into_response(),
    };
    match agents.cancel(agents.at(node), &id).await {
        Ok(view) => {
            tracing::info!(node = %format!("{node:016x}"), job = %id, "an admin cancelled a job");
            Json(view).into_response()
        }
        Err(e) => failed(e),
    }
}

pub async fn stream(
    State(session): State<Arc<ZendSession>>,
    Path((node, id)): Path<(String, String)>,
) -> Response {
    let (agents, node) = match target(&session, &node) {
        Ok(t) => t,
        Err(r) => return r.into_response(),
    };
    let at = agents.at(node);
    tail::stream(agents, at, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A node is sixteen hex digits**, read as the gateway lists it; anything
    /// else is a `400`, not a request to some other agent.
    #[test]
    fn a_node_is_read_as_hex() {
        assert_eq!(node_of("00000000000000a1").ok(), Some(0xa1));
        assert_eq!(node_of("ffffffffffffffff").ok(), Some(u64::MAX));
        let bad = node_of("agent-1").unwrap_err().into_response();
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);
    }

    /// Each way an agent fails is its own status: an agent that is not
    /// there, one that refused, one that could not be reached.
    #[test]
    fn an_agents_failure_keeps_its_meaning() {
        assert_eq!(
            failed(AgentError::NoAgent("gone".into())).status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            failed(AgentError::Refused {
                status: 404,
                why: "no job".into()
            })
            .status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            failed(AgentError::Transport("reset".into())).status(),
            StatusCode::BAD_GATEWAY
        );
    }
}
