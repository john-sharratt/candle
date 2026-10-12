//! The server while the tenants start.
//!
//! The host's real router is the tenants' routes merged with its own, so it
//! exists only once every tenant's `start` has returned. A start can be long —
//! npcd opens its substrate in it — and a page reaching a process that answers
//! nothing shows nothing at all. So the host serves first: the UIs, the shared
//! front-end, and a `/v1/status` built from each tenant's [`StartLine`] until
//! its start returns and from its own reports after. Once every tenant has
//! started, this server stops and the real one takes the same ports.

use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use futures::future::join_all;
use serde_json::Value;
use web::Roots;

use crate::cluster::Cluster;
use crate::endpoints::status::{assemble, Head};
use crate::model::ModelView;
use crate::starting::StartLine;
use crate::tenant::{Reports, TenantStatus};
use crate::ui;

/// Where one tenant is in its start.
#[derive(Clone)]
pub enum BootLine {
    /// `start` is running: its own line.
    Starting(Arc<StartLine>),
    /// `start` returned: its reports.
    Started(Arc<dyn Reports>),
}

/// What the boot server reads.
pub struct Boot {
    pub started_at_ms: u64,
    /// The UI build id (`ui::build_id`).
    pub build: String,
    pub model: ModelView,
    pub cluster: Option<Arc<Cluster>>,
    /// Every tenant, in rank order.
    lines: Mutex<Vec<(&'static str, BootLine)>>,
}

impl Boot {
    pub fn new(
        started_at_ms: u64,
        build: String,
        model: ModelView,
        cluster: Option<Arc<Cluster>>,
        lines: Vec<(&'static str, BootLine)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            started_at_ms,
            build,
            model,
            cluster,
            lines: Mutex::new(lines),
        })
    }

    /// Tenant `name`'s start returned with `reports`.
    pub fn started(&self, name: &str, reports: Arc<dyn Reports>) {
        let mut lines = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, line)) = lines.iter_mut().find(|(n, _)| *n == name) {
            *line = BootLine::Started(reports);
        }
    }

    fn lines(&self) -> Vec<(&'static str, BootLine)> {
        self.lines.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// One tenant's status line while the process boots.
async fn line_of(line: BootLine) -> TenantStatus {
    match line {
        BootLine::Starting(s) => TenantStatus::of(Some(s.view())),
        BootLine::Started(r) => r.status().await,
    }
}

async fn status(State(boot): State<Arc<Boot>>) -> Json<Value> {
    let lines = boot.lines();
    let statuses = join_all(lines.iter().map(|(_, l)| line_of(l.clone()))).await;
    let named: Vec<(&str, TenantStatus)> = lines.iter().map(|(n, _)| *n).zip(statuses).collect();
    let head = Head {
        started_at_ms: boot.started_at_ms,
        model: &boot.model,
        cluster: boot.cluster.as_deref(),
    };
    Json(assemble(&head, &boot.build, named))
}

/// The boot server's router: the status, and the shared front-end from
/// `common` that the pages showing it load.
pub fn router(boot: Arc<Boot>, common: Roots) -> Router {
    let front = Router::new()
        .route("/common/*path", get(ui::common))
        .route("/brand/*path", get(ui::brand))
        .with_state(common);
    Router::new()
        .route("/v1/status", get(status))
        .with_state(boot)
        .merge(front)
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    use crate::endpoints::test_support::{loading, scripted};

    async fn status_of(app: &Router) -> Value {
        let res = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status().as_u16(), 200);
        serde_json::from_slice(&to_bytes(res.into_body(), 1 << 20).await.unwrap()).unwrap()
    }

    /// **A tenant still starting answers with its own line**; once started it
    /// answers with its reports. The page sees npcd stamping its substrate
    /// while zend's model load runs beside it.
    #[tokio::test]
    async fn a_starting_tenant_shows_its_line_and_a_started_one_its_reports() {
        let npcd = StartLine::new();
        npcd.set_step("opening", "Opening the substrate");
        npcd.set_progress(40, 100, "records");
        let boot = Boot::new(
            1_700_000_000_000,
            "0123456789abcdef".into(),
            ModelView::of_prescribed(),
            None,
            vec![
                ("zend", BootLine::Starting(StartLine::new())),
                ("npcd", BootLine::Starting(Arc::clone(&npcd))),
            ],
        );
        let app = router(Arc::clone(&boot), Roots::default());

        let body = status_of(&app).await;
        assert_eq!(body["state"], "loading");
        assert_eq!(body["tenants"]["zend"]["loading"]["current"], "starting");
        assert_eq!(body["tenants"]["npcd"]["loading"]["current"], "opening");
        assert_eq!(body["tenants"]["npcd"]["loading"]["progressed"], 40);
        assert_eq!(body["tenants"]["npcd"]["loading"]["unit"], "records");

        boot.started(
            "zend",
            Arc::new(scripted(TenantStatus::of(Some(loading("model"))))),
        );
        let body = status_of(&app).await;
        assert_eq!(body["tenants"]["zend"]["loading"]["current"], "model");
        assert_eq!(
            body["loading"]["current"], "model",
            "the first load in rank order"
        );
        assert_eq!(body["tenants"]["npcd"]["loading"]["current"], "opening");
    }
}
