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

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
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
use effector_bridge::profile::{androidtv, webos};
use effector_bridge::proto::androidtv::Pairing;
use effector_bridge::LocalNet;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;

use crate::jobs::{JobError, Jobs};
use crate::wire::{BridgeCall, Bridged, Submit, Switching};

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
        // A device-bridge job runs on any in-network agent — it holds no repo
        // and needs no toolchain, only the house network the agent stands on.
        Route::new(&["POST"], "/v1/bridge", Dispatch::Any),
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
        .route("/v1/bridge", post(bridge))
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

/// `POST /v1/bridge` — a device-bridge job. The agent runs [`effector_bridge`] for
/// it, so the network commands happen on this machine inside the house, and
/// hands back the bridge's own line. A device refusing is `ok: false`, a
/// result the caller reads — not a failed job; only a switched-off agent is a
/// job failure, answered like one.
async fn bridge(State(api): State<Api>, Json(call): Json<BridgeCall>) -> Response {
    if !api.jobs.enabled() {
        return failed(JobError::Disabled);
    }
    let nets = effector_bridge::nets();
    if let BridgeCall::Pair {
        profile,
        address,
        code,
    } = call
    {
        return Json(pair(&profile, &address, code, &nets).await).into_response();
    }
    let worked = match call {
        BridgeCall::Work {
            profile,
            name,
            address,
            action,
            args,
        } => effector_bridge::work(&profile, &name, &address, &action, &args, &nets).await,
        BridgeCall::Wake { name, mac } => effector_bridge::wake(&name, &mac, &nets),
        BridgeCall::Pair { .. } => unreachable!("pairing is handled above"),
    };
    let bridged = match worked {
        Ok(line) => Bridged {
            ok: true,
            line,
            creds: Default::default(),
        },
        Err(e) => Bridged {
            ok: false,
            line: e.to_string(),
            creds: Default::default(),
        },
    };
    Json(bridged).into_response()
}

/// Android TV pairing sessions held open between the code appearing on the TV
/// and that code coming back, keyed by the TV's address. The session carries a
/// live TLS connection, so it lives on the agent that started it.
fn pairings() -> &'static Mutex<HashMap<String, Pairing>> {
    static SESSIONS: OnceLock<Mutex<HashMap<String, Pairing>>> = OnceLock::new();
    SESSIONS.get_or_init(Default::default)
}

/// Pair with a device and hand back the credentials to keep. webOS pairs in one
/// call (the person accepts on the TV). An Android TV takes two: no `code`
/// starts it and holds the session while the TV shows a code; the `code` then
/// finishes it.
pub(crate) async fn pair(
    profile: &str,
    address: &str,
    code: Option<String>,
    nets: &[LocalNet],
) -> Bridged {
    let refused = |line: String| Bridged {
        ok: false,
        line,
        creds: Default::default(),
    };
    let done: anyhow::Result<Bridged> = async {
        match (profile, code) {
            ("webos", _) => Ok(Bridged {
                ok: true,
                line: "Paired with the webOS TV.".into(),
                creds: webos::pair(address, nets).await?,
            }),
            ("androidtv", None) => {
                let session = androidtv::pair_start(address, nets, "effector-bridge").await?;
                pairings().lock().await.insert(address.to_string(), session);
                // Beginning succeeded: the TV is showing its code. `ok: true`
                // with no credentials yet — the second call earns those. A
                // genuine failure to begin propagates through `?` and comes
                // back `ok: false`, so the two are never confused downstream.
                Ok(Bridged {
                    ok: true,
                    line: "The TV is showing a six-digit code. Pair again with that code.".into(),
                    creds: Default::default(),
                })
            }
            ("androidtv", Some(code)) => match pairings().lock().await.remove(address) {
                Some(session) => Ok(Bridged {
                    ok: true,
                    line: "Paired with the Google TV.".into(),
                    creds: androidtv::creds_of(&session.finish(&code).await?),
                }),
                None => Ok(refused(
                    "No pairing is in progress for that TV — start one first.".into(),
                )),
            },
            (other, _) => Ok(refused(format!("There is no pairing for a {other}."))),
        }
    }
    .await;
    done.unwrap_or_else(|e| refused(e.to_string()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn pairing_an_unknown_profile_is_refused() {
        let b = pair("toaster", "192.168.0.9", None, &[]).await;
        assert!(!b.ok);
        assert!(b.line.contains("toaster"));
        assert!(b.creds.is_empty());
    }

    #[tokio::test]
    async fn finishing_a_pairing_with_no_session_is_refused() {
        // A code for an address no pairing was started for — the agent holds no
        // session, so it is refused rather than mispaired.
        let b = pair("androidtv", "203.0.113.7", Some("ABCDEF".into()), &[]).await;
        assert!(!b.ok);
        assert!(b.line.to_lowercase().contains("no pairing"));
    }
}
