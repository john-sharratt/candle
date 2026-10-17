//! The job API (§11.4), served only to members: a request reaches it over a
//! stream a gateway opened on the agent's socket, carrying the member it came
//! from ([`FROM_MEMBER`]). Anything else is refused.
//!
//! | Route | |
//! |---|---|
//! | `POST /v1/jobs` | a [`Submit`]; answers the job's [`JobView`] |
//! | `GET /v1/jobs` | the switch, the slots and the newest jobs ([`AgentJobs`]); `?limit=<n>`, 40 by default |
//! | `GET /v1/jobs/:id` | where it stands; `?wait=<s>` holds the answer until it ends, up to 30 s |
//! | `GET /v1/jobs/:id/log` | its log from the first byte |
//! | `GET /v1/jobs/:id/tail?from=<byte>` | its log from a byte, as a [`LogChunk`] |
//! | `DELETE /v1/jobs/:id` | cancel it, killing its process tree |
//! | `PUT /v1/agent` | switch the agent on or off ([`Switching`], §11.7) |
//!
//! A job carrying [`LINEAGE_HEADER`] claims that lineage for this agent, so
//! the next job of the conversation comes back here while the claim lasts
//! (§11.3). A job sent to an agent switched off is answered `503` `no_agent`,
//! as the gateway answers a job no agent can take.
//!
//! [`AgentJobs`]: crate::wire::AgentJobs
//! [`LogChunk`]: crate::wire::LogChunk

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{Path, Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use cluster::dispatch::{Dispatch, KeySource, Route, RouteTable};
use cluster::gateway::LINEAGE_HEADER;
use cluster::member::FROM_MEMBER;
use serde::Deserialize;
use serde_json::json;

use crate::jobs::{JobError, Jobs};
use crate::wire::{Submit, Switching};

/// Set on every request that arrived over the agent's socket.
#[derive(Debug, Clone, Copy)]
pub struct OverSocket;

/// What a job's lineage is claimed through.
pub type Claim = Arc<dyn Fn(&str) + Send + Sync>;

/// What is told the agent was switched, so its `CAPACITY` goes out at once.
pub type Switched = Arc<dyn Fn() + Send + Sync>;

/// The jobs `GET /v1/jobs` lists when it names no limit.
pub const LISTED: usize = 40;

#[derive(Clone)]
struct Api {
    jobs: Arc<Jobs>,
    claim: Claim,
    switched: Switched,
}

/// The dispatch table an agent declares: a job by its lineage, the rest to
/// whichever agent the member addresses.
pub fn table() -> RouteTable {
    RouteTable::new(vec![
        Route::new(
            &["POST"],
            "/v1/jobs",
            Dispatch::Sticky(vec![KeySource::Header(LINEAGE_HEADER.into())]),
        ),
        Route::new(&["GET"], "/v1/jobs", Dispatch::Any),
        Route::new(&["GET", "DELETE"], "/v1/jobs/:id", Dispatch::Any),
        Route::new(&["GET"], "/v1/jobs/:id/log", Dispatch::Any),
        Route::new(&["GET"], "/v1/jobs/:id/tail", Dispatch::Any),
        Route::new(&["PUT"], "/v1/agent", Dispatch::Any),
    ])
    .expect("the agent's routes are distinct")
}

pub fn router(jobs: Arc<Jobs>, claim: Claim, switched: Switched) -> Router {
    Router::new()
        .route("/v1/jobs", post(submit).get(list))
        .route("/v1/jobs/:id", get(view).delete(cancel))
        .route("/v1/jobs/:id/log", get(log))
        .route("/v1/jobs/:id/tail", get(tail))
        .route("/v1/agent", put(switch))
        .layer(middleware::from_fn(members_only))
        .with_state(Api {
            jobs,
            claim,
            switched,
        })
}

async fn members_only(req: Request, next: Next) -> Response {
    let from_member =
        req.extensions().get::<OverSocket>().is_some() && req.headers().contains_key(FROM_MEMBER);
    if !from_member {
        return refusal(StatusCode::FORBIDDEN, "jobs are a cluster member's to send");
    }
    next.run(req).await
}

fn refusal(status: StatusCode, why: &str) -> Response {
    (status, Json(json!({ "error": why }))).into_response()
}

fn failed(e: JobError) -> Response {
    let status = match &e {
        JobError::NotFound(_) => StatusCode::NOT_FOUND,
        JobError::UnknownRepo(_) | JobError::NoPush(_) => StatusCode::UNPROCESSABLE_ENTITY,
        JobError::Disabled | JobError::Offline => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({ "error": "no_agent", "message": e.to_string() })),
            )
                .into_response()
        }
        JobError::Bad(_) => StatusCode::BAD_REQUEST,
    };
    refusal(status, &e.to_string())
}

async fn submit(State(api): State<Api>, headers: HeaderMap, Json(job): Json<Submit>) -> Response {
    if let Some(lineage) = headers
        .get(LINEAGE_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|l| !l.is_empty())
    {
        (api.claim)(lineage);
    }
    match api.jobs.submit(job).await {
        Ok(view) => Json(view).into_response(),
        Err(e) => failed(e),
    }
}

#[derive(Deserialize)]
struct Wait {
    wait: Option<u64>,
}

async fn view(State(api): State<Api>, Path(id): Path<String>, Query(q): Query<Wait>) -> Response {
    let found = match q.wait {
        Some(s) => api.jobs.wait(&id, Duration::from_secs(s)).await,
        None => api.jobs.view(&id),
    };
    match found {
        Ok(view) => Json(view).into_response(),
        Err(e) => failed(e),
    }
}

async fn log(State(api): State<Api>, Path(id): Path<String>) -> Response {
    let path = match api.jobs.log(&id) {
        Ok(p) => p,
        Err(e) => return failed(e),
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => Body::from(bytes).into_response(),
        Err(e) => refusal(StatusCode::NOT_FOUND, &format!("job {id}'s log: {e}")),
    }
}

async fn cancel(State(api): State<Api>, Path(id): Path<String>) -> Response {
    match api.jobs.cancel(&id) {
        Ok(view) => Json(view).into_response(),
        Err(e) => failed(e),
    }
}

#[derive(Deserialize)]
struct Limit {
    limit: Option<usize>,
}

async fn list(State(api): State<Api>, Query(q): Query<Limit>) -> Response {
    Json(api.jobs.list(q.limit.unwrap_or(LISTED))).into_response()
}

#[derive(Deserialize)]
struct Offset {
    from: Option<u64>,
}

async fn tail(State(api): State<Api>, Path(id): Path<String>, Query(q): Query<Offset>) -> Response {
    match api.jobs.tail(&id, q.from.unwrap_or(0)).await {
        Ok(chunk) => Json(chunk).into_response(),
        Err(e) => failed(e),
    }
}

async fn switch(State(api): State<Api>, Json(to): Json<Switching>) -> Response {
    if let Err(e) = api.jobs.set_enabled(to.enabled) {
        return failed(e);
    }
    (api.switched)();
    tracing::info!(enabled = to.enabled, "the agent was switched");
    Json(Switching {
        enabled: api.jobs.enabled(),
    })
    .into_response()
}
