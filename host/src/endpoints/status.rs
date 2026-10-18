//! `GET /v1/status` — what both consoles' loading screens read, and how they
//! notice a restart or a new UI build.
//!
//! - `state` is `"ready"` once every tenant is, `"loading"` while any is, and
//!   `"fault"` while the device cannot take a turn — whatever the tenants say.
//! - `loading` is the load in progress — the highest-ranked tenant still
//!   loading, which is the one loading the model when there is a model to load.
//! - `engine_connected` is whether the model is loaded: some tenant is past its
//!   `model` step.
//! - `detail` is that load as one line: its label, and the item in hand.
//! - `build` hashes every compiled-in UI file and the binary's version (see
//!   [`crate::ui`]); a page reloads itself when it changes.
//! - `tenants.<name>` is each tenant's own line: its state, its load, its
//!   substrate maintenance, and its extras.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use candle::gpu_poison::device_failing;
use futures::future::join_all;
use serde_json::{json, Map, Value};

use super::Shared;
use crate::cluster::view::status_block;
use crate::cluster::Cluster;
use crate::loading::LoadingView;
use crate::model::ModelView;
use crate::stopping::StopStep;
use crate::tenant::TenantStatus;

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
    Json(assemble(&head_of(&s), &s.build, named))
}

/// What the body says of the process, apart from its tenants.
pub struct Head<'a> {
    pub started_at_ms: u64,
    pub model: &'a ModelView,
    pub cluster: Option<&'a Cluster>,
    /// The device cannot take a turn (`candle::gpu_poison::device_failing`).
    pub device_failing: bool,
}

fn head_of(s: &Shared) -> Head<'_> {
    Head {
        started_at_ms: s.started_at_ms,
        model: &s.model,
        cluster: s.cluster.as_deref(),
        device_failing: device_failing(),
    }
}

/// The `detail` of a member whose device is failing.
const DEVICE_FAILING: &str = "The GPU is failing every operation — out of memory, or a poisoned \
                              context. This member takes no turns, and relaunches itself if it \
                              does not recover.";

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
        waiting_for: Vec::new(),
    });
    line
}

/// The body, from each tenant's line.
pub fn assemble(head: &Head<'_>, build: &str, tenants: Vec<(&str, TenantStatus)>) -> Value {
    let loading = tenants.iter().find_map(|(_, t)| t.loading.clone());
    // Some tenant is ready or has finished its `model` step. Not "is past it":
    // a tenant's steps before the model (finding the pack, opening the
    // substrate) are not the model loaded.
    let engine_connected = tenants.iter().any(|(_, t)| {
        t.loading
            .as_ref()
            .is_none_or(|l| l.completed.contains(&"model"))
    });
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
    // A failing device outranks the tenants' own lines: they may read ready
    // while every turn they take fails.
    let (state, detail) = if head.device_failing {
        ("fault", DEVICE_FAILING.to_string())
    } else if loading.is_some() {
        ("loading", detail)
    } else {
        ("ready", detail)
    };
    json!({
        "state": state,
        "started_at_ms": head.started_at_ms,
        "build": build,
        "model": head.model,
        "engine_connected": engine_connected,
        "detail": detail,
        "loading": loading,
        "tenants": lines,
        "cluster": head.cluster.map(status_block),
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
        let mut npcd = loading("substrate");
        npcd.completed = vec!["model"];
        let s = shared(vec![
            ("zend", scripted(TenantStatus::of(Some(loading("model"))))),
            ("npcd", scripted(TenantStatus::of(Some(npcd)))),
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
            &head_of(&s),
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

    /// **A failing device is a fault whatever the tenants say.** A poisoned
    /// context read ready for the minute before the process relaunched, and
    /// every chat in it answered an empty stream.
    #[test]
    fn a_failing_device_is_a_fault_over_ready_tenants() {
        let s = shared(vec![]);
        let head = Head {
            device_failing: true,
            ..head_of(&s)
        };
        let body = assemble(&head, "", vec![("zend", TenantStatus::of(None))]);
        assert_eq!(body["state"], "fault");
        assert_eq!(body["detail"], DEVICE_FAILING);
        assert_eq!(body["tenants"]["zend"]["state"], "ready");
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
            &head_of(&s),
            "",
            vec![("zend", TenantStatus::of(Some(loading("model"))))],
        );
        assert_eq!(body["engine_connected"], false);
    }

    /// **A step before the model is not the model loaded.** zend opening its
    /// substrate — a step that comes before the weights — read as connected
    /// when "connected" meant "on any step but `model`".
    #[test]
    fn a_step_before_the_model_is_not_connected() {
        let s = shared(vec![]);
        let mut opening = loading("opening");
        opening.completed = vec!["pack"];
        let body = assemble(
            &head_of(&s),
            "",
            vec![("zend", TenantStatus::of(Some(opening)))],
        );
        assert_eq!(body["engine_connected"], false);
    }
}
