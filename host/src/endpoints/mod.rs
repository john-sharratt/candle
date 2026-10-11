//! The shared endpoints: every path both daemons serve, served once by the
//! host as the superset of the two (`docs/shared_model_host.md` §3.5).
//!
//! Each reports the shared engine once and each tenant's own data under
//! `tenants.<name>`, assembled from the tenants' [`Reports`]. Every route here
//! carries the stricter of the two daemons' guards, except `/v1/me` and
//! `/v1/tools`, which are open on purpose (see each route).

mod catalog;
mod logs;
mod me;
mod profile;
mod status;
mod telemetry;
mod waves;

use std::path::PathBuf;
use std::sync::Arc;

use axum::routing::{get, post};
use web::auth::{Role, Roles};
use web::Roots;

use crate::guard::{Api, Guarded, Routes};
use crate::logs::LogBus;
use crate::model::ModelView;
use crate::telemetry::Telemetry;
use crate::tenant::Reports;
use crate::ui;

/// What the shared endpoints read.
pub struct Shared {
    /// When this process started, wall clock — what a page prints as uptime and
    /// how it notices a restart.
    pub started_at_ms: u64,
    /// The binary's version, part of `build`.
    pub version: &'static str,
    /// Every UI's content directory, and the shared library's — what `build`
    /// hashes.
    pub content_dirs: Vec<PathBuf>,
    /// The model every tenant runs.
    pub model: ModelView,
    /// Every tenant, in rank order.
    pub tenants: Vec<(&'static str, Arc<dyn Reports>)>,
    pub log: Arc<LogBus>,
    pub telemetry: Arc<Telemetry>,
    /// The estate's role table, for `/v1/me`.
    pub roles: Roles,
}

impl Shared {
    /// The tenant named `name`.
    fn tenant(&self, name: &str) -> Option<&Arc<dyn Reports>> {
        self.tenants
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, r)| r)
    }
}

/// The shared endpoints, graded by the estate's table; the shared front-end
/// from `common` — its library at `/common/`, and the estate's marks at
/// `/brand/`, where the switcher addresses them on every host.
pub fn routes(shared: Arc<Shared>, common: Roots) -> Routes {
    let api = shared_table(shared.roles.clone()).into_routes(shared);
    let common = front_end_table().into_routes(common);
    api.merge("host", common, "host")
        .expect("the host's own routes are disjoint")
}

/// Every route the host serves, and the role each needs — the table merged
/// with every tenant's at start, read without the state behind it.
pub fn declared(roles: Roles) -> Vec<Guarded> {
    let mut all = shared_table(roles).declared().to_vec();
    all.extend_from_slice(front_end_table().declared());
    all
}

/// The shared endpoints, graded by the estate's table.
fn shared_table(roles: Roles) -> Api<Arc<Shared>> {
    Api::new(roles)
        // The liveness probe both consoles poll at boot, before they know who is
        // looking — the one route that must answer a caller nobody has named.
        .route("/v1/status", Role::Unauthenticated, get(status::status))
        // Signed out is an answer (`role: "unauthenticated"`), not an error: a
        // page reads the role and decides what to offer.
        .route("/v1/me", Role::Unauthenticated, get(me::me))
        // Hardware and engine readings: a signed-in operator watching the card
        // is the system page working; nothing here maps anybody's data.
        .route("/v1/telemetry", Role::User, get(telemetry::telemetry))
        .route("/v1/memory", Role::User, get(telemetry::memory))
        .route("/v1/phases", Role::User, get(waves::phases))
        .route("/v1/promotes", Role::User, get(waves::promotes))
        // Reading resets the counters, so two calls bracket a window — an
        // operator's, which a reader passing by must not be able to wipe.
        .route("/v1/profile", Role::Admin, get(profile::profile))
        // The chat page is open to everyone, and it reads the catalog to show
        // what a tool call left out and the defaults it took. The catalog
        // describes what the tools take, nothing anybody has stored.
        .route("/v1/tools", Role::Unauthenticated, get(catalog::tools))
        .route("/v1/tools/calibrate", Role::Admin, post(catalog::calibrate))
        .route("/v1/schema/layers", Role::User, get(catalog::layers))
        // Both of these hand a reader absolute filesystem paths — the redo log's
        // directory, and every write the log reports.
        .route("/v1/substrate", Role::Admin, get(catalog::substrate))
        .route("/ws/logs", Role::Admin, get(logs::logs))
}

/// The shared front-end: open to everyone, as any page's assets are.
fn front_end_table() -> Api<Roots> {
    Api::new(Roles::default())
        .route("/common/*path", Role::Unauthenticated, get(ui::common))
        .route("/brand/*path", Role::Unauthenticated, get(ui::brand))
}

#[cfg(test)]
pub(crate) mod test_support {
    //! A host's shared state with scripted tenants, for the endpoint tests.

    use super::*;

    use axum::body::Body;
    use axum::http::Request;
    use axum::Router;
    use futures::future::BoxFuture;
    use serde_json::{json, Value};
    use tower::ServiceExt;
    use web::auth::session::Identity;

    use crate::loading::LoadingView;
    use crate::tenant::{MeShare, TenantStatus};

    /// A tenant whose every report is fixed.
    pub struct Scripted {
        pub status: TenantStatus,
        pub tools: Vec<Value>,
        pub tools_extra: Value,
        pub layers: Vec<Value>,
        pub substrate: Value,
        pub me: MeShare,
    }

    impl Reports for Scripted {
        fn status(&self) -> BoxFuture<'_, TenantStatus> {
            let s = self.status.clone();
            Box::pin(async move { s })
        }
        fn tools(&self) -> BoxFuture<'_, (Vec<Value>, Value)> {
            let t = (self.tools.clone(), self.tools_extra.clone());
            Box::pin(async move { t })
        }
        fn calibrate(&self) -> BoxFuture<'_, Result<Value, (u16, Value)>> {
            Box::pin(async { Err((503, json!({"error": "no_engine"}))) })
        }
        fn layers(&self) -> BoxFuture<'_, Vec<Value>> {
            let l = self.layers.clone();
            Box::pin(async move { l })
        }
        fn substrate(&self) -> BoxFuture<'_, Value> {
            let s = self.substrate.clone();
            Box::pin(async move { s })
        }
        fn me<'a>(&'a self, _: Option<&'a Identity>, _: Role) -> BoxFuture<'a, MeShare> {
            let m = self.me.clone();
            Box::pin(async move { m })
        }
    }

    pub fn loading(current: &'static str) -> LoadingView {
        LoadingView {
            current,
            label: "Loading model",
            progress: 0.25,
            completed: vec![],
            completed_labels: vec![],
            progressed: 1,
            total: 4,
            unit: String::new(),
            detail: String::new(),
            elapsed_ms: 10,
            prefill_tokens: 0,
            prefill_tps: None,
        }
    }

    pub fn scripted(status: TenantStatus) -> Scripted {
        Scripted {
            status,
            tools: vec![],
            tools_extra: Value::Null,
            layers: vec![],
            substrate: Value::Null,
            me: MeShare::default(),
        }
    }

    pub fn shared(tenants: Vec<(&'static str, Scripted)>) -> Arc<Shared> {
        Arc::new(Shared {
            roles: serde_yaml::from_str("admins:\n  - sub: boss\n").unwrap(),
            started_at_ms: 1_700_000_000_000,
            version: "9.9.9",
            content_dirs: vec![],
            model: ModelView::of_prescribed(),
            tenants: tenants
                .into_iter()
                .map(|(n, s)| (n, Arc::new(s) as Arc<dyn Reports>))
                .collect(),
            log: LogBus::new(&["zend", "npcd"]),
            telemetry: Telemetry::new(ModelView::of_prescribed()),
        })
    }

    /// The routes over `shared`; `boss` is the admin.
    pub fn app(shared: Arc<Shared>) -> Router {
        routes(shared, Roots::default()).router
    }

    /// `GET uri` as the admin, answered as JSON.
    pub async fn get_json(app: &Router, uri: &str) -> (u16, Value) {
        call(app, "GET", uri, Some("boss")).await
    }

    pub async fn call(app: &Router, method: &str, uri: &str, sub: Option<&str>) -> (u16, Value) {
        let mut b = Request::builder().method(method).uri(uri);
        if let Some(s) = sub {
            b = b
                .header("x-tokera-user", s)
                .header("x-tokera-provider", "google");
        }
        let res = app
            .clone()
            .oneshot(b.body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = res.status().as_u16();
        let bytes = axum::body::to_bytes(res.into_body(), 8 << 20)
            .await
            .unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    use crate::tenant::TenantStatus;

    /// The route table, pinned: adding a shared endpoint is a moment where
    /// somebody chooses its role.
    #[test]
    fn the_shared_route_table_is_what_we_think_it_is() {
        let s = shared(vec![]);
        let got: Vec<(&str, &str)> = routes(s, Roots::default())
            .declared
            .iter()
            .map(|g| (g.path, g.min.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("/v1/status", "unauthenticated"),
                ("/v1/me", "unauthenticated"),
                ("/v1/telemetry", "user"),
                ("/v1/memory", "user"),
                ("/v1/phases", "user"),
                ("/v1/promotes", "user"),
                ("/v1/profile", "admin"),
                ("/v1/tools", "unauthenticated"),
                ("/v1/tools/calibrate", "admin"),
                ("/v1/schema/layers", "user"),
                ("/v1/substrate", "admin"),
                ("/ws/logs", "admin"),
                ("/common/*path", "unauthenticated"),
                ("/brand/*path", "unauthenticated"),
            ]
        );
    }

    /// A signed-out caller gets the status, and is refused what needs a role.
    #[tokio::test]
    async fn the_guards_hold_for_a_signed_out_caller() {
        let app = app(shared(vec![("zend", scripted(TenantStatus::of(None)))]));
        assert_eq!(call(&app, "GET", "/v1/status", None).await.0, 200);
        assert_eq!(call(&app, "GET", "/v1/telemetry", None).await.0, 401);
        assert_eq!(call(&app, "GET", "/v1/substrate", None).await.0, 401);
        assert_eq!(
            call(&app, "GET", "/v1/substrate", Some("someone")).await.0,
            403
        );
    }
}
