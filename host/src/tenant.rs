//! The contract between the host and a daemon it runs (`docs/shared_model_host.md`
//! §3.1).
//!
//! A tenant is a daemon as a library: zend, npcd. The host owns the process —
//! runtime, logging, signals, the model, the HTTP server, shutdown — and hands
//! each tenant what it needs to build its own state on the shared engine. The
//! tenant answers with its own routes and what the host's shared endpoints
//! report about it.

use std::path::PathBuf;
use std::sync::Arc;

use candle_conversation::projection::Conversation;
use candle_conversation::{Capabilities, EngineSource};
use futures::future::BoxFuture;
use include_dir::Dir;
use serde::Serialize;
use serde_json::{Map, Value};
use web::auth::session::Identity;
use web::auth::{Role, Roles};

use crate::cluster::Join;
use crate::guard::Routes;
use crate::loading::LoadingView;
use crate::logs::LogBus;
use crate::starting::StartLine;
use crate::stopping::Stopping;
use crate::telemetry::Telemetry;

/// A daemon the host runs.
pub trait Tenant: Send + 'static {
    /// A short name: log attribution, status key, route owner. `"zend"`.
    fn name(&self) -> &'static str;

    /// What this tenant needs of the model, checked against the prescribed one
    /// when its engine is attached.
    fn requires(&self) -> Capabilities;

    /// The directory its substrate lives in. Two tenants naming one directory
    /// refuse the start before either opens it.
    fn substrate_dir(&self) -> PathBuf;

    /// Its web UI.
    fn ui(&self) -> TenantUi;

    /// Start: build state, start background work — the engine load among it —
    /// and return what the host serves. Called on the host's runtime once the
    /// tenant's context exists; the engine arrives through
    /// [`TenantContext::engine`], loaded on the tenant's own thread.
    fn start(self: Box<Self>, ctx: TenantContext) -> BoxFuture<'static, anyhow::Result<Running>>;
}

/// What the host gives a tenant to start with.
pub struct TenantContext {
    /// Where this tenant's engine comes from: built by the highest-ranked
    /// tenant with every tenant's requests merged, and attached to by the rest.
    pub engine: EngineSource,
    /// The host's log bus — the one `/ws/logs` serves.
    pub log: Arc<LogBus>,
    /// The estate's role table.
    pub roles: Roles,
    /// The host's telemetry, for a tenant's own columns
    /// ([`Telemetry::add_source`]).
    pub telemetry: Arc<Telemetry>,
    /// The tenant's app in the cluster the process joins with `--cluster`:
    /// once its substrate is open, the tenant connects through it in its
    /// `connecting` loading step. `None` for a standalone daemon.
    pub cluster: Option<Join>,
    /// The process's stop: what a tenant asks to restart the process onto
    /// something it read once, at start, and now holds anew — npcd's mind
    /// fetched through an agent (`docs/replicated_substrate.md` §10.3).
    pub stopping: Arc<Stopping>,
    /// Where the tenant says what it is doing until [`Tenant::start`]
    /// returns — what the loading page shows of it meanwhile (`crate::boot`).
    pub starting: Arc<StartLine>,
}

/// How a started tenant is drained: stop its background work, flush and detach
/// its substrate. The host calls each tenant's in reverse rank order — when the
/// process is told to stop, and when a later start fails — then exits.
pub type Stop = Box<dyn FnOnce() -> BoxFuture<'static, ()> + Send>;

/// A started tenant.
pub struct Running {
    /// Its own routes — none the host serves.
    pub routes: Routes,
    /// What the shared endpoints report about it.
    pub reports: Arc<dyn Reports>,
    /// Drain it — see [`Stop`].
    pub stop: Stop,
}

/// A tenant's web UI.
#[derive(Debug, Clone)]
pub struct TenantUi {
    /// The names that reach it: `code.tokera.com`, `zend.localhost`.
    pub hosts: Vec<String>,
    /// Its pages, compiled into the tenant's crate (`ui`).
    pub content: &'static Dir<'static>,
    /// Whether it serves a request whose `Host` names no tenant — a bare IP.
    pub default: bool,
}

/// A tenant's share of the host's shared endpoints (§3.5). Each method answers
/// for this tenant only; the host assembles the superset.
pub trait Reports: Send + Sync {
    /// Its load and readiness, and its own extras.
    fn status(&self) -> BoxFuture<'_, TenantStatus>;

    /// Its tool catalog: each entry an object with at least `name`,
    /// `description` and `parameters` (the host adds `tenant`), and the facts
    /// about the catalog that are not per tool (npcd's `uncalibrated`,
    /// `acts_per_turn`) — `null` when there are none.
    fn tools(&self) -> BoxFuture<'_, (Vec<Value>, Value)>;

    /// Re-run its catalog's calibration, answering the response body — or the
    /// status and body that refuse it (no engine yet).
    fn calibrate(&self) -> BoxFuture<'_, Result<Value, (u16, Value)>>;

    /// Its projection schema's layers — each the raw schema entry plus any
    /// live fields; the host adds `tenant`.
    fn layers(&self) -> BoxFuture<'_, Vec<Value>>;

    /// Its substrate for the overview: storage, counts, where its own views
    /// live. `null` before it is open.
    fn substrate(&self) -> BoxFuture<'_, Value>;

    /// Its share of `/v1/me` for this caller.
    fn me<'a>(&'a self, identity: Option<&'a Identity>, role: Role) -> BoxFuture<'a, MeShare>;

    /// Its substrate, once its engine holds it open: what the maintenance
    /// commands run on (`docs/replicated_substrate.md` §12.4).
    fn conversation(&self) -> Option<Conversation>;

    /// Ask every stored conversation whether it is intact, reporting what it
    /// finds and tombstoning nothing (§12.4 `self-check`). Refused, with the
    /// reason, by a tenant that has no such check or before its engine loads.
    fn self_check(&self, progress: CheckProgress) -> BoxFuture<'_, Result<Value, String>>;
}

/// A self-check's progress: conversations asked so far, of how many.
pub type CheckProgress = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// One tenant's part of `/v1/me`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeShare {
    /// Fields for the top level — npcd's account record (`user_id`,
    /// `unique_name`, `profile`, …).
    pub top: Map<String, Value>,
    /// Fields under `tenants.<name>` — zend's `tool_modes`, `default_tools`.
    pub own: Value,
}

/// One tenant's line in `/v1/status`.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TenantStatus {
    /// `"loading"` or `"ready"`.
    pub state: &'static str,
    /// Its load, while it loads.
    pub loading: Option<LoadingView>,
    /// Its substrate's maintenance state, once its engine holds it open.
    pub maintenance: Option<Maintenance>,
    /// Anything else it reports (zend: `ingest_backlog`), merged into its line.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl TenantStatus {
    /// A tenant with nothing but its load to report.
    pub fn of(loading: Option<LoadingView>) -> Self {
        Self {
            state: if loading.is_some() {
                "loading"
            } else {
                "ready"
            },
            loading,
            maintenance: None,
            extra: Map::new(),
        }
    }
}

/// A substrate's redo-log maintenance, in every tenant's status line alike.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct Maintenance {
    /// Segment files on disk.
    pub segments: usize,
    /// The last background operation, and when it ran.
    pub last_op: Option<String>,
    pub last_op_unix: Option<u64>,
    /// Whether one is running now.
    pub running: bool,
}

impl Maintenance {
    /// From `Conversation::maintenance_status` — `(segments, last operation and
    /// its time, running)`.
    pub fn of((segments, last, running): (usize, Option<(String, u64)>, bool)) -> Self {
        let (last_op, last_op_unix) = last.map_or((None, None), |(op, at)| (Some(op), Some(at)));
        Self {
            segments,
            last_op,
            last_op_unix,
            running,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, to_value};

    /// Maintenance reads the same in every tenant's line.
    #[test]
    fn maintenance_has_one_shape() {
        let m = Maintenance::of((4, Some(("compact".into(), 1_700_000_000)), false));
        assert_eq!(
            to_value(&m).unwrap(),
            json!({"segments": 4, "last_op": "compact", "last_op_unix": 1_700_000_000u64,
                   "running": false})
        );
        assert_eq!(
            to_value(Maintenance::of((1, None, true))).unwrap(),
            json!({"segments": 1, "last_op": null, "last_op_unix": null, "running": true})
        );
    }

    /// A tenant's extras sit beside its standard fields, not under a key of
    /// their own — `tenants.zend.ingest_backlog`, as the overlay reads it.
    #[test]
    fn extras_flatten_into_the_tenants_line() {
        let mut s = TenantStatus::of(None);
        s.extra.insert("ingest_backlog".into(), Value::Null);
        assert_eq!(
            to_value(&s).unwrap(),
            json!({"state": "ready", "loading": null, "maintenance": null, "ingest_backlog": null})
        );
    }
}
