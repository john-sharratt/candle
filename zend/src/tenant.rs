//! zend as a tenant of the host (`docs/shared_model_host.md` §3.1).
//!
//! [`ZendTenant`] starts a [`ZendSession`] on the host's engine slot and hands
//! the host its routes; [`ZendReports`] is the thin adapter from the session's
//! state to the host's shared endpoints — zend's line in `/v1/status`, its tool
//! catalog, its layers, its substrate, its tools modes for `/v1/me`.

use std::path::PathBuf;
use std::sync::Arc;

use candle_conversation::models::DialectType;
use candle_conversation::persistence::SUBSTRATE_DIR;
use candle_conversation::projection::Conversation;
use candle_conversation::Capabilities;
use futures::future::BoxFuture;
use host::loading::{tokens_per_second, LoadingView};
use host::{
    CheckProgress, Maintenance, MeShare, Reports, Running, Tenant, TenantContext, TenantStatus,
    TenantUi,
};
use serde_json::{json, Map, Value};
use tokio::task::spawn_blocking;
use web::auth::session::Identity;
use web::auth::Role;

use crate::access::{allowed_modes, default_mode};
use crate::api::routes;
use crate::cli::DEFAULT_UI;
use crate::config::DaemonConfig;
use crate::ingest_backlog::BacklogSnapshot;
use crate::loading::LoadingSnapshot;
use crate::schema_layers::{raw_layers, with_live_fields};
use crate::session::ZendSession;

/// The tenant's name: its log attribution, status key and route owner.
pub const NAME: &str = "zend";

/// What zend needs of a model: `<tool_call>`, `<think>` and `</think>` as one
/// token each — the tool-call parser and the think gate key on them — and the
/// Qwen3.5 dialect its prompts and stencils are written in.
pub fn requires() -> Capabilities {
    Capabilities {
        tokens: &["<tool_call>", "<think>", "</think>"],
        dialect: DialectType::Qwen35,
    }
}

/// zend's UI by default: `zend/web/` in the source tree, on its own names, and
/// the default site — what a request by IP shows.
pub fn default_ui() -> TenantUi {
    TenantUi {
        hosts: vec!["code.tokera.com".into(), "zend.localhost".into()],
        content: PathBuf::from(DEFAULT_UI),
        default: true,
    }
}

/// zend, configured and not yet started.
pub struct ZendTenant {
    config: DaemonConfig,
    ui: TenantUi,
}

impl ZendTenant {
    pub fn new(config: DaemonConfig, ui: TenantUi) -> Self {
        Self { config, ui }
    }
}

impl Tenant for ZendTenant {
    fn name(&self) -> &'static str {
        NAME
    }

    fn requires(&self) -> Capabilities {
        requires()
    }

    fn substrate_dir(&self) -> PathBuf {
        self.config.workspace.root().join(SUBSTRATE_DIR)
    }

    fn ui(&self) -> TenantUi {
        self.ui.clone()
    }

    fn start(self: Box<Self>, ctx: TenantContext) -> BoxFuture<'static, anyhow::Result<Running>> {
        Box::pin(async move {
            let mut config = self.config;
            config.engine = ctx.engine;
            config.roles = ctx.roles.clone();
            config.cluster = ctx.cluster;
            tracing::info!(
                workspace = %config.workspace.root().display(),
                repos = %config.workspace.names().join(", "),
                "starting zend",
            );
            let session = Arc::new(ZendSession::new(config));
            session.start_loading();
            let stopping = Arc::clone(&session);
            Ok(Running {
                routes: routes(Arc::clone(&session), ctx.roles),
                reports: Arc::new(ZendReports::new(session)),
                stop: Box::new(move || {
                    Box::pin(async move {
                        stopping.shutdown().await;
                    })
                }),
            })
        })
    }
}

/// zend's share of the host's shared endpoints.
pub struct ZendReports {
    session: Arc<ZendSession>,
}

impl ZendReports {
    pub fn new(session: Arc<ZendSession>) -> Self {
        Self { session }
    }

    /// Run `read` on the blocking pool. Each read takes the engine's lock,
    /// which the loader holds through its warm-up, and `/v1/status` is polled
    /// by every open page of every console — on a runtime worker those polls
    /// would park the workers the other tenant's requests need.
    async fn off_runtime<T: Send + 'static>(
        &self,
        read: impl FnOnce(&ZendSession) -> T + Send + 'static,
    ) -> T {
        let session = Arc::clone(&self.session);
        spawn_blocking(move || read(&session))
            .await
            .expect("a report read panicked")
    }
}

impl Reports for ZendReports {
    fn status(&self) -> BoxFuture<'_, TenantStatus> {
        Box::pin(self.off_runtime(|session| {
            let snap = session.status_snapshot();
            let mut s = TenantStatus::of(snap.loading.map(|l| loading_view(l, snap.detail)));
            s.maintenance = session.substrate_maintenance().map(Maintenance::of);
            s.extra.insert(
                "ingest_backlog".into(),
                snap.ingest_backlog.map_or(Value::Null, backlog_view),
            );
            s
        }))
    }

    fn tools(&self) -> BoxFuture<'_, (Vec<Value>, Value)> {
        Box::pin(self.off_runtime(|session| {
            let tools = session
                .substrate_tools()
                .map(|v| v.tools.iter().map(|t| json!(t)).collect())
                .unwrap_or_default();
            (tools, Value::Null)
        }))
    }

    fn calibrate(&self) -> BoxFuture<'_, Result<Value, (u16, Value)>> {
        Box::pin(async {
            Err((
                409,
                json!({
                    "error": "calibrated_at_load",
                    "detail": "zend calibrates its catalog as it loads, regenerating every \
                               case whose definition changed — restart zend to recalibrate",
                }),
            ))
        })
    }

    /// Each layer's raw schema entry with its live fields — see
    /// [`crate::schema_layers`].
    fn layers(&self) -> BoxFuture<'_, Vec<Value>> {
        Box::pin(self.off_runtime(|session| {
            let raw = match raw_layers(session.workspace_root()) {
                Ok(raw) => raw,
                Err(e) => {
                    tracing::warn!("zend's projection schema could not be read: {e:#}");
                    return Vec::new();
                }
            };
            let live = session
                .substrate_overview()
                .map(|o| o.layers)
                .unwrap_or_default();
            with_live_fields(raw, &live)
        }))
    }

    fn substrate(&self) -> BoxFuture<'_, Value> {
        Box::pin(
            self.off_runtime(|session| match session.substrate_overview() {
                Some(o) => {
                    let mut v = json!(o);
                    if let Value::Object(m) = &mut v {
                        m.insert(
                            "path".into(),
                            json!(session.workspace_root().join(SUBSTRATE_DIR)),
                        );
                        m.insert("views".into(), json!("/v1/substrate/zend/"));
                    }
                    v
                }
                None => Value::Null,
            }),
        )
    }

    fn me<'a>(&'a self, _: Option<&'a Identity>, role: Role) -> BoxFuture<'a, MeShare> {
        Box::pin(async move {
            MeShare {
                top: Map::new(),
                own: modes_view(role),
            }
        })
    }

    fn conversation(&self) -> Option<Conversation> {
        self.session.conversation()
    }

    /// On the blocking pool: it asks every stored conversation, a turn each.
    fn self_check(&self, progress: CheckProgress) -> BoxFuture<'_, Result<Value, String>> {
        Box::pin(async move {
            let ran = self
                .off_runtime(move |session| session.self_check_now(&*progress))
                .await;
            match ran {
                None => Err("the model is not loaded yet".into()),
                Some(Err(e)) => Err(format!("{e:#}")),
                Some(Ok(report)) => Ok(json!(report)),
            }
        })
    }
}

/// zend's load, in the host's shape. `detail` is the step's free-form
/// sub-status (a download's progress, the model being loaded).
fn loading_view(l: LoadingSnapshot, detail: String) -> LoadingView {
    LoadingView {
        current: l.current.id(),
        label: l.current.label(),
        progress: l.progress,
        completed: l.completed.iter().map(|s| s.id()).collect(),
        completed_labels: l.completed.iter().map(|s| s.label()).collect(),
        progressed: l.progressed,
        total: l.total,
        unit: l.unit,
        detail,
        elapsed_ms: l.elapsed_ms,
        prefill_tokens: l.prefill_tokens,
        prefill_tps: tokens_per_second(l.prefill_tokens, l.elapsed_ms),
    }
}

/// Pending background ingest, merged across every ingest layer and uploads.
fn backlog_view(b: BacklogSnapshot) -> Value {
    json!({
        "processed": b.processed,
        "total": b.total,
        "last_item": b.last_item,
    })
}

/// The tools modes `role` may choose, by wire id in dial order, and the one a
/// turn runs in when the caller names none — advice to the GUI's dial; the
/// chat route resolves the same role and enforces it.
fn modes_view(role: Role) -> Value {
    json!({
        "tool_modes": allowed_modes(role).into_iter().map(|m| m.id()).collect::<Vec<_>>(),
        "default_tools": default_mode(role).id(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::loading::LoadStep;

    #[test]
    fn a_load_is_reported_by_step_id() {
        let snap = LoadingSnapshot {
            current: LoadStep::Substrate,
            progress: 0.5,
            completed: vec![LoadStep::Model],
            progressed: 5,
            total: 10,
            unit: "turns".into(),
            elapsed_ms: 2000,
            prefill_tokens: 4000,
        };
        let v = loading_view(snap, "Loading model…".into());
        assert_eq!(v.current, "substrate");
        assert_eq!(v.label, "Loading substrate");
        assert_eq!(v.completed, ["model"]);
        assert_eq!(v.detail, "Loading model…");
        assert_eq!(v.prefill_tps, Some(2000.0));
    }

    /// The backlog keeps its wire shape; maintenance's is the host's
    /// (`host::Maintenance`).
    #[test]
    fn the_backlog_keeps_its_wire_shape() {
        assert_eq!(
            backlog_view(BacklogSnapshot {
                processed: 3,
                total: 9,
                last_item: Some("src/lib.rs".into()),
            }),
            json!({"processed": 3, "total": 9, "last_item": "src/lib.rs"})
        );
    }

    /// An admin may pick any mode and starts at Comprehensive; anyone else is
    /// offered None and Restricted and starts at Restricted.
    #[test]
    fn the_modes_follow_the_role() {
        assert_eq!(
            modes_view(Role::Admin),
            json!({"tool_modes": ["none", "restricted", "comprehensive"], "default_tools": "comprehensive"})
        );
        assert_eq!(
            modes_view(Role::Unauthenticated),
            json!({"tool_modes": ["none", "restricted"], "default_tools": "restricted"})
        );
    }

    /// The capabilities zend declares are what the prescribed model has — the
    /// host refuses a model lacking one, naming zend.
    #[test]
    fn zend_needs_the_three_markers_and_the_qwen35_dialect() {
        let r = requires();
        assert_eq!(r.tokens, ["<tool_call>", "<think>", "</think>"]);
        assert_eq!(r.dialect, DialectType::Qwen35);
        assert_eq!(host::MODEL.spec().chat_format, DialectType::Qwen35);
    }
}
