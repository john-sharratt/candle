//! The maintenance commands (`docs/replicated_substrate.md` §12.4): every one a
//! route on the member, admin only, sent by the gateway's page with
//! `x-tokera-node` so it rides that member's socket — or with `curl` to a
//! member's own listener, which works the same when it is cut off.
//!
//! | Module | |
//! |---|---|
//! | [`parse`] | a command's name and body, and a destructive one's `confirm` |
//! | [`class`] | what a command may be run by: look, care, decide |
//! | [`jobs`] | one command at a time, its progress as NDJSON |
//! | [`exec`] | what each command does |
//!
//! A command answers at once with its job; its progress streams on
//! `GET /v1/cluster/command/job/:id`. Each one is reported in the member's
//! `COORD`, so every gateway's coordination log shows who ran it, with what,
//! and how it ended.

pub mod class;
pub mod exec;
pub mod jobs;
pub mod parse;

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use cluster::coord::CommandLine;
use cluster::dispatch::Dispatch;
use cluster::replicate::Replicator;
use serde_json::json;
use web::auth::forwarded::identify;
use web::auth::{Role, Roles};

use crate::cluster::Cluster;
use crate::guard::{get, post, put, Api, Routes};
use crate::stopping::Stopping;
use crate::tenant::Reports;
use jobs::{Job, Jobs};
use parse::{parse, Body, Command};

/// What the commands act on.
pub struct Commands {
    pub cluster: Option<Arc<Cluster>>,
    /// Every tenant, in rank order.
    pub tenants: Vec<(&'static str, Arc<dyn Reports>)>,
    pub stopping: Arc<Stopping>,
    /// This machine's name: what a destructive command's `confirm` must say.
    pub host: String,
    pub jobs: Jobs,
}

impl Commands {
    pub fn new(
        cluster: Option<Arc<Cluster>>,
        tenants: Vec<(&'static str, Arc<dyn Reports>)>,
        stopping: Arc<Stopping>,
        host: String,
    ) -> Self {
        Self {
            cluster,
            tenants,
            stopping,
            host,
            jobs: Jobs::new(),
        }
    }

    /// Every app's replication, for reporting a command in `COORD`.
    fn every_replicator(&self) -> Vec<Arc<Replicator>> {
        self.cluster
            .iter()
            .flat_map(|c| c.links())
            .filter_map(|l| l.member()?.replicator().cloned())
            .collect()
    }

    fn report(&self, line: &CommandLine) {
        for r in self.every_replicator() {
            r.command(line.clone());
        }
    }

    /// Start `command` on `app` for `by`: its job runs in the background.
    pub fn start(
        self: &Arc<Self>,
        command: Command,
        body: &Body,
        by: String,
    ) -> Result<Arc<Job>, String> {
        let args = command.args(body.app.as_deref());
        let job = self.jobs.start(command.name(), args.clone())?;
        tracing::warn!(
            command = command.name(),
            job = job.id,
            by = %by,
            args = %args,
            "maintenance command"
        );
        let mut line = CommandLine {
            job: job.id,
            command: command.name().into(),
            args: args.to_string(),
            by: by.clone(),
            started_ms: job.started_ms,
            outcome: None,
        };
        self.report(&line);
        let me = self.clone();
        let running = job.clone();
        let app = body.app.clone();
        tokio::spawn(async move {
            let outcome = me.exec(&running, &command, app.as_deref(), &by).await;
            line.outcome = Some(match &outcome {
                Ok(_) => "ok".into(),
                Err(e) => format!("failed: {e}"),
            });
            tracing::warn!(
                command = command.name(),
                job = running.id,
                outcome = line.outcome.as_deref().unwrap_or_default(),
                "maintenance command ended"
            );
            me.report(&line);
            me.jobs.finish(&running, outcome);
        });
        Ok(job)
    }
}

/// The routes: admin only, each sent to one member.
pub fn table(roles: Roles) -> Api<Arc<Commands>> {
    Api::new(roles)
        .route(
            "/v1/cluster/command/:name",
            Role::Admin,
            Dispatch::Any,
            post(run),
        )
        .route(
            "/v1/cluster/command/job/:id",
            Role::Admin,
            Dispatch::Any,
            get(job),
        )
        .route(
            "/v1/cluster/quorum",
            Role::Admin,
            Dispatch::Any,
            put(quorum),
        )
}

pub fn routes(commands: Arc<Commands>, roles: Roles) -> Routes {
    table(roles).into_routes(commands)
}

fn refuse(status: StatusCode, error: &str, detail: String) -> Response {
    (status, Json(json!({"error": error, "detail": detail}))).into_response()
}

/// Who sent a request: the signed-in caller's email, or `local` for one with
/// no forwarded identity (a member's own console).
fn caller(headers: &HeaderMap) -> String {
    identify(headers).map_or_else(|_| "local".into(), |id| id.email)
}

fn started(commands: &Arc<Commands>, name: &str, body: Body, headers: &HeaderMap) -> Response {
    let command = match parse(name, &body, &commands.host) {
        Ok(c) => c,
        Err(detail) => return refuse(StatusCode::BAD_REQUEST, "refused", detail),
    };
    match commands.start(command, &body, caller(headers)) {
        Ok(job) => (
            StatusCode::ACCEPTED,
            Json(json!({
                "job": job.id,
                "command": job.command,
                "stream": format!("/v1/cluster/command/job/{}", job.id),
            })),
        )
            .into_response(),
        Err(detail) => refuse(StatusCode::CONFLICT, "busy", detail),
    }
}

/// A command's body: empty for none, or its JSON — anything else refused.
fn body_of(bytes: &[u8]) -> Result<Body, Box<Response>> {
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(Body::default());
    }
    serde_json::from_slice(bytes).map_err(|e| {
        Box::new(refuse(
            StatusCode::UNPROCESSABLE_ENTITY,
            "bad_body",
            format!("the body is not a command's: {e}"),
        ))
    })
}

async fn run(
    State(commands): State<Arc<Commands>>,
    Path(name): Path<String>,
    headers: HeaderMap,
    bytes: Bytes,
) -> Response {
    match body_of(&bytes) {
        Ok(body) => started(&commands, &name, body, &headers),
        Err(refused) => *refused,
    }
}

/// `PUT /v1/cluster/quorum { max_members, confirm }`: `set-quorum` by its own
/// route (§5.9).
async fn quorum(
    State(commands): State<Arc<Commands>>,
    headers: HeaderMap,
    bytes: Bytes,
) -> Response {
    match body_of(&bytes) {
        Ok(body) => started(&commands, "set-quorum", body, &headers),
        Err(refused) => *refused,
    }
}

async fn job(State(commands): State<Arc<Commands>>, Path(id): Path<u64>) -> Response {
    match commands.jobs.get(id) {
        Some(job) => job.response(),
        None => refuse(
            StatusCode::NOT_FOUND,
            "no_job",
            format!("no job {id} on this member: a job is read where it ran"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::Router;
    use serde_json::Value;

    use crate::endpoints::test_support::{app, job_lines, scripted, send, shared};
    use crate::endpoints::Shared;
    use crate::stopping::Ending;
    use crate::tenant::TenantStatus;

    fn zend() -> Arc<Shared> {
        shared(vec![("zend", scripted(TenantStatus::of(None)))])
    }

    async fn post(router: &Router, name: &str, body: &str) -> (u16, Value) {
        let uri = format!("/v1/cluster/command/{name}");
        send(router, "POST", &uri, Some("boss"), Some(body)).await
    }

    /// A command answers at once with its job, whose stream carries its
    /// steps and ends with the result.
    #[tokio::test]
    async fn a_command_runs_as_a_job_and_streams_its_steps() {
        let router = app(zend());
        let (code, body) = post(&router, "self-check", "{}").await;
        assert_eq!(code, 202, "{body}");
        assert_eq!(body["command"], "self-check");
        let id = body["job"].as_u64().unwrap();
        assert_eq!(body["stream"], format!("/v1/cluster/command/job/{id}"));
        assert_eq!(
            job_lines(&router, id).await,
            [
                json!({"event": "started", "command": "self-check", "args": {"app": null}}),
                json!({"event": "step", "step": "zend", "detail": "1 of 2 conversations asked"}),
                json!({"event": "step", "step": "zend", "detail": "2 of 2 conversations asked"}),
                json!({"event": "done", "result": {"zend": {"checked": 2, "corrupt": 0}}}),
            ]
        );
    }

    /// A command that cannot run ends its job with the reason; one the
    /// request itself gets wrong is refused before any job starts.
    #[tokio::test]
    async fn refusals_say_why() {
        let router = app(zend());
        let (_, body) = post(&router, "drain", "{}").await;
        let lines = job_lines(&router, body["job"].as_u64().unwrap()).await;
        assert_eq!(
            lines.last().unwrap(),
            &json!({"event": "error", "error": "command_failed",
                    "detail": "this daemon runs without --cluster: it has no cluster to act on"})
        );
        let (_, body) = post(&router, "audit", "{}").await;
        let lines = job_lines(&router, body["job"].as_u64().unwrap()).await;
        assert_eq!(
            lines.last().unwrap()["detail"],
            "zend's substrate is not open yet"
        );
        let (code, body) = post(&router, "frobnicate", "{}").await;
        assert_eq!(
            (code, body["detail"].as_str().unwrap()),
            (400, "there is no command \"frobnicate\"")
        );
        let (code, _) = post(&router, "audit", r#"{"nope": 1}"#).await;
        assert_eq!(code, 422, "an unknown field is refused");
        let (code, _) = send(&router, "POST", "/v1/cluster/command/audit", None, None).await;
        assert_eq!(code, 401);
        let (code, _) = send(
            &router,
            "GET",
            "/v1/cluster/command/job/99",
            Some("boss"),
            None,
        )
        .await;
        assert_eq!(code, 404);
    }

    /// `shutdown` needs this machine's name typed out; with it, the process is
    /// asked to stop. `PUT /v1/cluster/quorum` is `set-quorum`, confirmed the
    /// same way.
    #[tokio::test]
    async fn shutdown_and_the_quorum_call_need_the_host_named() {
        let s = zend();
        let router = app(s.clone());
        let (code, body) = post(&router, "shutdown", "{}").await;
        assert_eq!(code, 400);
        assert_eq!(
            body["detail"],
            "shutdown is destructive: send this machine's name, \"m1\", as `confirm`"
        );
        assert_eq!(s.stopping.ending(), None);
        let (code, body) = post(&router, "shutdown", r#"{"confirm":"m1"}"#).await;
        assert_eq!(code, 202);
        let lines = job_lines(&router, body["job"].as_u64().unwrap()).await;
        assert_eq!(
            lines.last().unwrap()["result"],
            json!({"ending": "shutdown"})
        );
        assert_eq!(s.stopping.ending(), Some(Ending::Exit));

        let (code, _) = send(
            &router,
            "PUT",
            "/v1/cluster/quorum",
            Some("boss"),
            Some(r#"{"max_members":2}"#),
        )
        .await;
        assert_eq!(code, 400);
    }
}
