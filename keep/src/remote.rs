//! keep's routes on its cluster socket (`docs/claude_zend_gateway.md` §8.4), so
//! an Admin's operations tools reach every machine's keep through the gateway:
//!
//! | Route | |
//! |---|---|
//! | `GET /v1/keep/state` | each service: its stable file and build, whether it runs, its trials |
//! | `POST /v1/keep/trial {service, artifact}` | a trial, run as a job: accepted at once, its end read from the state |
//! | `POST /v1/keep/service/:name/:action` | `restart`, `stop` or `start` |
//!
//! Only a cluster member's request is served — one the gateway stamped with
//! the member it came from; who may send one is the member's to check. A trial
//! is refused for an artifact not built exactly from a commit on `origin/main`:
//! a local build never becomes a candidate, yet serves real traffic while it
//! runs, and what it writes replicates and is permanent. `/accept` and `/exit`
//! are not here: an install with no trial, and keep restarting itself, are done
//! on the machine.

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use cluster::dispatch::{Dispatch, Route as Declared, RouteTable};
use cluster::member::FROM_MEMBER;
use cluster::session::VirtualConn;
use cluster::upgrade::artifacts::{Probe, RunProbe};
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::board::BOARD;
use crate::{begin, find, service_request, switch, target, Action, Answer, Hand, Slot};

pub const STATE: &str = "/v1/keep/state";
pub const TRIAL: &str = "/v1/keep/trial";
const SERVICE: &str = "/v1/keep/service/";

/// The routes keep declares, so the gateway knows what its socket takes.
pub fn table() -> RouteTable {
    RouteTable::new(vec![
        Declared::new(&["GET"], STATE, Dispatch::Any),
        Declared::new(&["POST"], TRIAL, Dispatch::Any),
        Declared::new(&["POST"], "/v1/keep/service/:name/:action", Dispatch::Any),
    ])
    .expect("keep's routes are distinct")
}

/// Serve `conn`, a stream a gateway opened, until it ends.
pub fn serve(conn: VirtualConn, slots: Arc<Vec<Slot>>) {
    tokio::spawn(async move {
        let svc = service_fn(move |req: Request<Incoming>| {
            let slots = Arc::clone(&slots);
            async move { Ok::<_, Infallible>(answer(req, slots).await) }
        });
        let _ = http1::Builder::new()
            .serve_connection(TokioIo::new(conn), svc)
            .await;
    });
}

fn reply(status: StatusCode, body: Value) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body.to_string())))
        .expect("a response builds")
}

/// The HTTP status of a supervisor's `answer`.
fn status_of(answer: &Answer) -> StatusCode {
    StatusCode::from_u16(answer.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

fn refuse(status: StatusCode, detail: impl Into<String>) -> Response<Full<Bytes>> {
    reply(status, json!({"error": "refused", "detail": detail.into()}))
}

async fn answer(req: Request<Incoming>, slots: Arc<Vec<Slot>>) -> Response<Full<Bytes>> {
    if !req.headers().contains_key(FROM_MEMBER) {
        return refuse(StatusCode::FORBIDDEN, "keep is a cluster member's to reach");
    }
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let body = match req.into_body().collect().await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return refuse(
                StatusCode::BAD_REQUEST,
                format!("the body did not arrive: {e}"),
            )
        }
    };
    let run = move || route(&method, &path, &body, &slots);
    match tokio::task::spawn_blocking(run).await {
        Ok(r) => r,
        Err(e) => refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("the request stopped: {e}"),
        ),
    }
}

/// What a request asks of keep, answered on a thread that may block.
fn route(method: &Method, path: &str, body: &[u8], slots: &[Slot]) -> Response<Full<Bytes>> {
    match (method, path) {
        (&Method::GET, STATE) => reply(StatusCode::OK, state(slots, &RunProbe)),
        (&Method::POST, TRIAL) => trial(body, slots, &RunProbe),
        (&Method::POST, p) if p.starts_with(SERVICE) => {
            let rest = &p[SERVICE.len()..];
            let Some((name, action)) = rest.split_once('/') else {
                return refuse(
                    StatusCode::NOT_FOUND,
                    "POST /v1/keep/service/<name>/<action>",
                );
            };
            let action = match action {
                "restart" => Action::Restart,
                "stop" => Action::Stop,
                "start" => Action::Start,
                other => {
                    return refuse(
                        StatusCode::NOT_FOUND,
                        format!("`{other}` is not restart, stop or start"),
                    )
                }
            };
            let done = service_request(action, name, slots);
            reply(
                status_of(&done),
                json!({"service": name, "answer": done.body.trim()}),
            )
        }
        _ => refuse(
            StatusCode::NOT_FOUND,
            format!("keep has no {method} {path}"),
        ),
    }
}

/// Every service as it stands.
pub fn state(slots: &[Slot], probe: &dyn Probe) -> Value {
    let services: Vec<Value> = slots
        .iter()
        .map(|s| {
            let build = probe.build_info(&s.stable).map(
                |b| json!({"count": b.count, "hash": b.hash, "local": b.local, "kind": b.kind}),
            );
            json!({
                "name": s.name,
                "stable": s.stable.display().to_string(),
                "running": find(&s.file).map(|f| f.pid),
                "build": build,
                "trials": BOARD.trials(&s.name),
            })
        })
        .collect();
    json!({"services": services})
}

#[derive(Deserialize)]
struct TrialAsk {
    service: String,
    artifact: PathBuf,
}

/// Refuse `artifact` unless it is a build of a commit on `origin/main`.
pub fn mainline(artifact: &Path, probe: &dyn Probe) -> Result<(), String> {
    let Some(info) = probe.build_info(artifact) else {
        return Err(format!(
            "{} does not answer --build-info: it is not a build keep can check",
            artifact.display()
        ));
    };
    if info.local.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} is a local build ({}-{}, local {}): only a build of a commit on origin/main is \
             tried from the cluster — a local one serves real traffic while it runs, and what it \
             writes replicates",
            artifact.display(),
            info.count,
            info.hash,
            info.local
        ))
    }
}

/// Start a trial as a job; its end is read from [`state`].
fn trial(body: &[u8], slots: &[Slot], probe: &dyn Probe) -> Response<Full<Bytes>> {
    let ask: TrialAsk = match serde_json::from_slice(body) {
        Ok(a) => a,
        Err(e) => {
            return refuse(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("a trial is {{service, artifact}}: {e}"),
            )
        }
    };
    let Some(slot) = slots
        .iter()
        .find(|s| s.name.eq_ignore_ascii_case(&ask.service))
    else {
        return refuse(
            StatusCode::NOT_FOUND,
            format!("no service `{}`", ask.service),
        );
    };
    let named = ask
        .artifact
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.eq_ignore_ascii_case(&slot.file));
    if !named {
        return refuse(
            StatusCode::BAD_REQUEST,
            format!("{}'s artifact is named {}", slot.name, slot.file),
        );
    }
    if let Err(e) = mainline(&ask.artifact, probe) {
        return refuse(StatusCode::CONFLICT, e);
    }
    // Checked and on the board before the answer, so a trial accepted is a
    // trial the state shows — running, then ended.
    let artifact = ask.artifact.display().to_string();
    let (path, tried) = match target(&artifact, slots).and_then(|(path, tried)| {
        begin(tried, &artifact)?;
        Ok((path, tried.clone()))
    }) {
        Ok(t) => t,
        Err(refused) => return refuse(status_of(&refused), refused.body.trim().to_string()),
    };
    thread::spawn(move || {
        switch(path, &tried, Hand::Trial);
    });
    reply(
        StatusCode::ACCEPTED,
        json!({"service": slot.name, "artifact": artifact, "read": STATE}),
    )
}

#[cfg(test)]
mod tests {
    use cluster::version::info::{BuildInfo, Opens};

    use super::*;

    struct Says(Option<&'static str>);

    impl Probe for Says {
        fn build_info(&self, _exe: &Path) -> Option<BuildInfo> {
            self.0.map(|local| BuildInfo {
                kind: "motherd".into(),
                count: 12,
                hash: "abc1234".into(),
                local: local.into(),
                platform: "windows-x86_64".into(),
                opens: Opens::default(),
            })
        }
    }

    /// **Only a build of a commit on `origin/main` is tried**: a local build,
    /// or a file that is no build, is refused before anything runs.
    #[test]
    fn a_trial_is_of_a_mainline_build_only() {
        let a = Path::new("stage/motherd.exe");
        assert!(mainline(a, &Says(Some(""))).is_ok());
        let err = mainline(a, &Says(Some("ffee"))).unwrap_err();
        assert!(
            err.contains("is a local build (12-abc1234, local ffee)"),
            "{err}"
        );
        assert!(mainline(a, &Says(None))
            .unwrap_err()
            .contains("--build-info"));
    }

    #[test]
    fn keep_declares_its_three_routes() {
        let t = table();
        assert!(t.lookup("GET", STATE).is_some());
        assert!(t.lookup("POST", TRIAL).is_some());
        assert!(t
            .lookup("POST", "/v1/keep/service/motherd/restart")
            .is_some());
        assert!(t.lookup("POST", "/accept").is_none());
        assert!(t.lookup("POST", STATE).is_none());
    }
}
