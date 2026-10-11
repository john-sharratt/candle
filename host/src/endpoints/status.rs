//! `GET /v1/status` — what both consoles' loading screens read, and how they
//! notice a restart or a new UI build.
//!
//! - `state` is `"ready"` once every tenant is, `"loading"` while any is.
//! - `loading` is the load in progress — the highest-ranked tenant still
//!   loading, which is the one loading the model when there is a model to load.
//! - `engine_connected` is whether the model is loaded: some tenant is past its
//!   `model` step.
//! - `detail` is that load as one line: its label, and the item in hand.
//! - `build` hashes every UI file on disk, the binary's version and the
//!   running executable's stamp (see [`crate::ui`]); a page reloads itself
//!   when it changes.
//! - `tenants.<name>` is each tenant's own line: its state, its load, its
//!   substrate maintenance, and its extras.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use futures::future::join_all;
use serde_json::{json, Map, Value};
use tokio::task::spawn_blocking;

use super::Shared;
use crate::cluster::view::status_block;
use crate::loading::LoadingView;
use crate::stopping::StopStep;
use crate::tenant::TenantStatus;
use crate::ui::build_id;

pub async fn status(State(s): State<Arc<Shared>>) -> Json<Value> {
    let lines = join_all(s.tenants.iter().map(|(_, r)| r.status())).await;
    let stopping = s.stopping.step();
    let named: Vec<(&str, TenantStatus)> = s
        .tenants
        .iter()
        .map(|(n, _)| *n)
        .zip(lines)
        .map(|(n, line)| match &stopping {
            Some(step) => (n, draining(line, step)),
            None => (n, line),
        })
        .collect();
    let dirs = s.content_dirs.clone();
    let version = s.version;
    // A failed walk is a bug, not an empty build: an open page reads a changed
    // `build` as a new UI and reloads itself.
    let build = spawn_blocking(move || build_id(&dirs, version))
        .await
        .expect("the build id walk panicked");
    Json(assemble(&s, &build, named))
}

/// A tenant's line while the process stops (§12.4): loading again, the step
/// in hand as its load, so the loading page shows the drain as it runs.
fn draining(mut line: TenantStatus, step: &StopStep) -> TenantStatus {
    line.state = "loading";
    line.loading = Some(LoadingView {
        current: "draining",
        label: step.label,
        progress: 0.0,
        completed: Vec::new(),
        completed_labels: Vec::new(),
        progressed: 0,
        total: 0,
        unit: String::new(),
        detail: step.detail.clone(),
        elapsed_ms: 0,
        prefill_tokens: 0,
        prefill_tps: None,
    });
    line
}

/// The body, from each tenant's line.
fn assemble(s: &Shared, build: &str, tenants: Vec<(&str, TenantStatus)>) -> Value {
    let loading = tenants.iter().find_map(|(_, t)| t.loading.clone());
    let engine_connected = tenants
        .iter()
        .any(|(_, t)| t.loading.as_ref().is_none_or(|l| l.current != "model"));
    let detail = loading
        .as_ref()
        .map(|l| {
            if l.detail.is_empty() {
                l.label.to_string()
            } else {
                format!("{} — {}", l.label, l.detail)
            }
        })
        .unwrap_or_default();
    let mut lines = Map::new();
    for (name, t) in tenants {
        lines.insert(
            name.to_string(),
            serde_json::to_value(t).unwrap_or(Value::Null),
        );
    }
    json!({
        "state": if loading.is_some() { "loading" } else { "ready" },
        "started_at_ms": s.started_at_ms,
        "build": build,
        "model": s.model,
        "engine_connected": engine_connected,
        "detail": detail,
        "loading": loading,
        "tenants": lines,
        "cluster": s.cluster.as_deref().map(status_block),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::endpoints::test_support::{app, get_json, loading, scripted, shared};

    /// While zend loads the model, the top level is zend's load and the engine
    /// is not connected; npcd's line is its own.
    #[tokio::test]
    async fn the_top_level_is_the_first_load_in_rank_order() {
        let s = shared(vec![
            ("zend", scripted(TenantStatus::of(Some(loading("model"))))),
            (
                "npcd",
                scripted(TenantStatus::of(Some(loading("substrate")))),
            ),
        ]);
        let (code, body) = get_json(&app(s), "/v1/status").await;
        assert_eq!(code, 200);
        assert_eq!(body["state"], "loading");
        assert_eq!(body["loading"]["current"], "model");
        assert_eq!(body["detail"], "Loading model");
        assert_eq!(body["started_at_ms"], 1_700_000_000_000u64);
        assert_eq!(body["build"].as_str().unwrap().len(), 16);
        // npcd is past its model step: the engine is up.
        assert_eq!(body["engine_connected"], true);
        assert_eq!(body["tenants"]["zend"]["state"], "loading");
        assert_eq!(body["tenants"]["npcd"]["loading"]["current"], "substrate");
        assert_eq!(body["model"]["repo"], crate::model::MODEL.spec().model_repo);
    }

    /// Every tenant ready: ready, no load, the engine connected — exactly.
    #[test]
    fn all_ready_is_ready() {
        let s = shared(vec![]);
        let body = assemble(
            &s,
            "0123456789abcdef",
            vec![
                ("zend", TenantStatus::of(None)),
                ("npcd", TenantStatus::of(None)),
            ],
        );
        assert_eq!(
            body,
            json!({
                "state": "ready",
                "started_at_ms": 1_700_000_000_000u64,
                "build": "0123456789abcdef",
                "model": s.model,
                "engine_connected": true,
                "detail": "",
                "loading": null,
                "tenants": {
                    "zend": {"state": "ready", "loading": null, "maintenance": null},
                    "npcd": {"state": "ready", "loading": null, "maintenance": null},
                },
                "cluster": null,
            })
        );
    }

    /// While the process stops, every tenant is loading again, its load the
    /// stop's step — what the loading page shows.
    #[tokio::test]
    async fn a_stop_shows_as_every_tenants_load() {
        let s = shared(vec![("zend", scripted(TenantStatus::of(None)))]);
        s.stopping
            .set_step("Sending", "12 own events still to reach the group");
        let (_, body) = get_json(&app(s), "/v1/status").await;
        assert_eq!(body["state"], "loading");
        assert_eq!(body["loading"]["current"], "draining");
        assert_eq!(
            body["detail"],
            "Sending — 12 own events still to reach the group"
        );
        assert_eq!(body["tenants"]["zend"]["state"], "loading");
    }

    /// Nobody past the model step: not connected.
    #[test]
    fn the_engine_is_not_connected_while_every_tenant_waits_on_the_model() {
        let s = shared(vec![]);
        let body = assemble(
            &s,
            "",
            vec![("zend", TenantStatus::of(Some(loading("model"))))],
        );
        assert_eq!(body["engine_connected"], false);
    }
}
