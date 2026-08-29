//! `/v1/telemetry` and `/ws/logs` — the daemon reporting on itself.
//!
//! Separate from [`crate::api`] because the state is unrelated: that router
//! holds authored content behind `RwLock`s, this one holds a driver handle and
//! a broadcast channel. They are merged at startup and their paths do not
//! overlap.
//!
//! Both routes previously fell through to `web::mock::npcd`, which invented
//! plausible answers. These are measurements. What cannot be measured is
//! reported absent rather than as zero — see [`crate::telemetry`].

use std::path::Path;
use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::{IntoResponse, Response},
    routing::get,
    Json,
};
// Only the test-only `router` builds one directly; `main` goes through `api`.
#[cfg(test)]
use axum::Router;
use serde_json::json;
use tokio::sync::broadcast::error::RecvError;
use web::auth::{Role, Roles};

use crate::guard::Api;
use crate::logs::{LogBus, LogLine};
use crate::substrate::SubstrateDir;
use crate::telemetry::Telemetry;

/// What the operational routes need. One of each, for the daemon's life.
pub struct Ops {
    pub telemetry: Arc<Telemetry>,
    pub logs: Arc<LogBus>,
    pub substrate: SubstrateDir,
    /// Who is an admin. These routes report on the machine rather than on any
    /// user's data, and they are graded by how much of the machine they give
    /// away — see [`router`].
    pub roles: Roles,
    /// When this process started, for `/v1/status`. Wall clock, because the
    /// console prints it as an uptime a person reads.
    pub started_at_ms: u64,
}

impl Ops {
    /// Builds the state and starts the sampler, because a `Telemetry` that is
    /// not being sampled is an empty history that looks like a broken page.
    /// Requires a Tokio runtime.
    pub fn new(logs: Arc<LogBus>, data_dir: &Path, roles: Roles) -> Arc<Self> {
        let telemetry = Telemetry::new();
        telemetry.spawn_sampler();
        Arc::new(Self {
            telemetry,
            logs,
            substrate: SubstrateDir::new(data_dir),
            roles,
            started_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        })
    }
}

/// These routes report on the machine, so they are graded by how much of the
/// machine each one gives away.
///
/// - **`User`** for the hardware readings. A signed-in operator watching VRAM
///   is the console's system page working as intended, and card model and host
///   RAM are not a map of anything.
/// - **`Admin`** for the substrate and the log stream. Both emit absolute
///   filesystem paths — `/v1/substrate/storage` names the redo log's directory,
///   and every registry save writes its full path into the log — so they hand a
///   reader the layout of this host. That is the same thing the gateway's error
///   page is forbidden from doing with the estate's addressing.
pub fn api(state: Arc<Ops>) -> Api<Arc<Ops>> {
    Api::new(state.roles.clone())
        // **Reachable before sign-in, necessarily.** This is the liveness probe
        // the console polls at boot, before it knows who is looking — so it is
        // the one route where requiring a role breaks the page for everybody.
        //
        // It was in the fallback, which sits behind `User`, and the result was
        // exactly that: a signed-out visitor got 401, the boot loop read it as
        // "daemon not up yet", and spent sixteen seconds saying so before
        // landing on the welcome screen.
        .route("/v1/status", Role::Unauthenticated, get(status))
        .route("/v1/telemetry", Role::User, get(telemetry))
        .route("/v1/memory", Role::User, get(memory))
        .route("/v1/substrate/storage", Role::Admin, get(substrate_storage))
        .route("/ws/logs", Role::Admin, get(logs))
}

/// The finished router, for tests. `main` builds it from [`api`] so it can read
/// the route table on the way past.
#[cfg(test)]
pub fn router(state: Arc<Ops>) -> Router {
    api(state.clone()).into_router(state)
}

/// The redo log's footprint on disk — see [`crate::substrate`]. The one part of
/// the substrate view that does not need the engine, so it is real today while
/// the layer tree above it is still the console's fixture.
/// Whether this daemon is up, and what it is.
///
/// The console blocks on this at boot, so it answers from what is already in
/// memory — no filesystem, no device query, nothing that could make the probe
/// itself the reason a page is slow.
///
/// `state` is `ready` because reaching this handler means the router is
/// serving: the registries are read and the substrate is open before the bind.
/// A daemon that is still loading has not bound a port and is not answering at
/// all, which the console reads as "not up yet" — the honest distinction being
/// between *no answer* and *this answer*, rather than a progress figure this
/// process cannot produce.
async fn status(State(s): State<Arc<Ops>>) -> Response {
    Json(json!({
        "state": "ready",
        "detail": "no engine loaded — authored content, accounts and telemetry are real",
        "started_at_ms": s.started_at_ms,
        "build": concat!("npcd-", env!("CARGO_PKG_VERSION")),
        "mode": "server-headless",
        "engine_connected": false,
    }))
    .into_response()
}

async fn substrate_storage(State(s): State<Arc<Ops>>) -> Response {
    Json(s.substrate.read()).into_response()
}

/// The retained window plus a request-time host reading.
///
/// Nothing is cached: the series is already in memory and the host read is a
/// syscall. A cache here would put an age on numbers whose whole value is being
/// current.
async fn telemetry(State(s): State<Arc<Ops>>) -> Response {
    Json(s.telemetry.read()).into_response()
}

/// Full memory accounting. Reads the OS and this process, so it is worth having
/// before there is an engine — see [`crate::telemetry::memory`].
async fn memory() -> Response {
    // `sysinfo` blocks, briefly. Off the async workers that serve the console.
    Json(
        tokio::task::spawn_blocking(crate::telemetry::memory::dump)
            .await
            .expect("memory probe does not panic"),
    )
    .into_response()
}

/// The log tail.
///
/// The check happens **before** the upgrade, so a refusal is an ordinary 401 or
/// 403 the browser can read. Upgrading first and then closing the socket would
/// leave the console reconnecting against a wall with nothing to display, which
/// is how a permissions problem becomes an hour of network debugging.
async fn logs(ws: WebSocketUpgrade, State(s): State<Arc<Ops>>) -> Response {
    ws.on_upgrade(move |socket| stream(socket, s))
}

/// Replay the recent lines, then tail.
///
/// The subscription is taken *before* the snapshot is read, so a line written
/// between the two is delivered rather than lost. It may therefore be seen
/// twice — once in the replay, once from the stream — which is the right way
/// round: a duplicated line is visible and obviously harmless, a dropped one is
/// neither.
async fn stream(mut socket: WebSocket, s: Arc<Ops>) {
    let mut rx = s.logs.subscribe();

    for line in s.logs.recent() {
        if send(&mut socket, &line).await.is_err() {
            return;
        }
    }

    loop {
        tokio::select! {
            got = rx.recv() => match got {
                Ok(line) => {
                    if send(&mut socket, &line).await.is_err() {
                        return;
                    }
                }
                // The console fell behind — a paused pane, a slow tab, a burst
                // from the daemon. Say so in the stream rather than silently
                // closing the gap: a reader looking at these lines needs to
                // know the sequence in front of them is not complete.
                Err(RecvError::Lagged(n)) => {
                    let notice = LogLine {
                        ts: String::new(),
                        level: "WARN".to_owned(),
                        target: "npcd::logs".to_owned(),
                        msg: format!("{n} lines dropped — console fell behind the daemon"),
                    };
                    if send(&mut socket, &notice).await.is_err() {
                        return;
                    }
                }
                Err(RecvError::Closed) => return,
            },
            // Reading is what notices a close, and a client that sends nothing
            // still needs its pings answered — which axum does inside `recv`.
            incoming = socket.recv() => match incoming {
                Some(Ok(_)) => {}
                _ => return,
            },
        }
    }
}

/// One JSON object per frame, so a reader is `JSON.parse(ev.data)` with no
/// framing of its own to get wrong.
async fn send(socket: &mut WebSocket, line: &LogLine) -> Result<(), ()> {
    let text = serde_json::to_string(line).map_err(|_| ())?;
    socket.send(Message::Text(text)).await.map_err(|_| ())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    use super::*;
    use crate::logs::BusWriter;

    /// `g1` is an ordinary user and `admin` is an admin, so a test can say
    /// which it means by which subject it sends.
    fn ops() -> Arc<Ops> {
        Ops::new(
            LogBus::new(),
            Path::new("."),
            serde_yaml::from_str("admins:\n  - sub: admin\n").unwrap(),
        )
    }

    /// These four report on the *machine* rather than on anybody's data, so the
    /// grading is by how much of the host each one gives away. Written down for
    /// the same reason `api`'s table is: adding a route here should be a moment
    /// where somebody chooses.
    /// Async because [`Ops::new`] starts the telemetry sampler, which needs a
    /// runtime to spawn onto.
    #[tokio::test]
    async fn the_route_table_is_what_we_think_it_is() {
        let got: Vec<(&str, &str)> = api(ops())
            .declared()
            .iter()
            .map(|r| (r.path, r.min.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                // The liveness probe, and the only route here that must answer
                // a caller nobody has named: the console polls it at boot,
                // before it knows who is looking.
                ("/v1/status", "unauthenticated"),
                // Hardware readings. A signed-in operator watching VRAM is the
                // system page working; card model and host RAM map nothing.
                ("/v1/telemetry", "user"),
                ("/v1/memory", "user"),
                // Both of these emit absolute filesystem paths.
                ("/v1/substrate/storage", "admin"),
                ("/ws/logs", "admin"),
            ]
        );
    }

    /// The route answers from this daemon, not from the mock. The mock's card
    /// is called `mock device`; a real one is named by the driver, and a
    /// machine with no card reports `null`. Any of those is a pass — what would
    /// not be is the fixture.
    #[tokio::test]
    async fn telemetry_is_this_daemon_and_not_the_fixture() {
        let app = router(ops());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/telemetry")
                    .header("x-tokera-user", "g1")
                    .header("x-tokera-provider", "google")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert_ne!(v["gpu"]["name"], "mock device");
        // The fixture's giveaway: engine numbers present with no engine behind
        // them. Ours are absent until something reports.
        assert_eq!(v["engine_connected"], false);
        assert_eq!(v["series"]["decode_tps"], serde_json::Value::Null);
        // And the host is measured, so the page is never wholly blank.
        assert!(v["host"]["total_mib"].as_u64().is_some_and(|m| m > 0));
        assert!(v["sample_period_s"].as_f64().unwrap() > 0.0);
    }

    /// The memory panel answers off the OS, so it carries real numbers on a
    /// daemon with no engine — which is the reason it is worth serving now.
    #[tokio::test]
    async fn memory_reports_the_machine_and_this_process() {
        let app = router(ops());
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/v1/memory")
                    .header("x-tokera-user", "g1")
                    .header("x-tokera-provider", "google")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert!(v["host_now"]["total_bytes"].as_u64().unwrap() > 0);
        assert!(v["process"]["working_set_bytes"].as_u64().unwrap() > 0);
        // The engine's own report stays absent rather than becoming `{}`.
        assert_eq!(v["report"], serde_json::Value::Null);
    }

    /// A pane that connects late still sees what just happened — the ring is
    /// replayed on the same socket as the tail.
    #[tokio::test]
    async fn the_socket_replays_before_it_tails() {
        let bus = LogBus::new();
        let mut w = BusWriter::new(bus.clone());
        writeln!(w, "2026-01-01T00:00:01.0Z  INFO npcd: earlier").unwrap();

        // Subscribing after the fact is exactly the console's position, and the
        // replay is what fills its pane.
        let replayed = bus.recent();
        assert_eq!(replayed.last().unwrap().msg, "earlier");

        // A subscriber taken now sees only what comes next, which is why the
        // handler needs both halves.
        let mut rx = bus.subscribe();
        writeln!(w, "2026-01-01T00:00:02.0Z  WARN npcd: later").unwrap();
        assert_eq!(rx.recv().await.unwrap().msg, "later");
    }
}
