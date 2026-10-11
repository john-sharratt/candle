//! `GET /v1/tools`, `POST /v1/tools/calibrate`, `GET /v1/schema/layers` and
//! `GET /v1/substrate` — each tenant's catalog, schema and substrate, side by
//! side.
//!
//! Each tool and layer carries the `tenant` it belongs to, so one list serves
//! both consoles and a page filters by its own name. What a tenant reports
//! about its catalog as a whole sits under `tenants.<name>`.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::future::join_all;
use serde_json::{json, Map, Value};

use super::Shared;

/// Each entry of `items`, stamped with the tenant it belongs to.
fn stamped(tenant: &str, items: Vec<Value>) -> impl Iterator<Item = Value> + '_ {
    items.into_iter().map(move |mut item| {
        if let Value::Object(m) = &mut item {
            m.insert("tenant".into(), json!(tenant));
        }
        item
    })
}

pub async fn tools(State(s): State<Arc<Shared>>) -> Json<Value> {
    let answers = join_all(s.tenants.iter().map(|(_, r)| r.tools())).await;
    let mut all = Vec::new();
    let mut tenants = Map::new();
    for ((name, _), (items, extra)) in s.tenants.iter().zip(answers) {
        all.extend(stamped(name, items));
        tenants.insert(name.to_string(), extra);
    }
    Json(json!({ "tools": all, "tenants": tenants }))
}

/// `?tenant=<name>` — whose catalog to calibrate.
pub async fn calibrate(
    State(s): State<Arc<Shared>>,
    Query(q): Query<HashMap<String, String>>,
) -> Response {
    let Some(name) = q.get("tenant") else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "no_tenant", "detail": "name the catalog: ?tenant=<name>"})),
        )
            .into_response();
    };
    let Some(reports) = s.tenant(name) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "no_such_tenant", "detail": format!("no tenant `{name}` here")})),
        )
            .into_response();
    };
    match reports.calibrate().await {
        Ok(body) => Json(body).into_response(),
        Err((code, body)) => (
            StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            Json(body),
        )
            .into_response(),
    }
}

pub async fn layers(State(s): State<Arc<Shared>>) -> Json<Value> {
    let answers = join_all(s.tenants.iter().map(|(_, r)| r.layers())).await;
    let all: Vec<Value> = s
        .tenants
        .iter()
        .zip(answers)
        .flat_map(|((name, _), items)| stamped(name, items).collect::<Vec<_>>())
        .collect();
    Json(json!({ "layers": all }))
}

/// Every attached substrate, under its tenant's name.
pub async fn substrate(State(s): State<Arc<Shared>>) -> Json<Value> {
    let answers = join_all(s.tenants.iter().map(|(_, r)| r.substrate())).await;
    let mut tenants = Map::new();
    for ((name, _), view) in s.tenants.iter().zip(answers) {
        tenants.insert(name.to_string(), view);
    }
    Json(json!({ "tenants": tenants }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::endpoints::test_support::{app, call, get_json, scripted, shared};
    use crate::tenant::TenantStatus;

    fn two() -> Arc<Shared> {
        let mut zend = scripted(TenantStatus::of(None));
        zend.tools = vec![json!({"name": "read_file", "high_risk": false})];
        zend.layers = vec![json!({"name": "dialogue", "conversation_count": 3})];
        zend.substrate = json!({"open": true, "segment_count": 4});
        let mut npcd = scripted(TenantStatus::of(None));
        npcd.tools = vec![json!({"name": "say", "category": "speech"})];
        npcd.tools_extra = json!({"uncalibrated": 2});
        npcd.layers = vec![json!({"name": "perception", "budget": 3})];
        shared(vec![("zend", zend), ("npcd", npcd)])
    }

    #[tokio::test]
    async fn every_tool_carries_its_tenant() {
        let (_, v) = get_json(&app(two()), "/v1/tools").await;
        assert_eq!(
            v,
            json!({
                "tools": [
                    {"name": "read_file", "high_risk": false, "tenant": "zend"},
                    {"name": "say", "category": "speech", "tenant": "npcd"},
                ],
                "tenants": {"zend": null, "npcd": {"uncalibrated": 2}},
            })
        );
    }

    #[tokio::test]
    async fn every_layer_carries_its_tenant() {
        let (_, v) = get_json(&app(two()), "/v1/schema/layers").await;
        assert_eq!(
            v,
            json!({"layers": [
                {"name": "dialogue", "conversation_count": 3, "tenant": "zend"},
                {"name": "perception", "budget": 3, "tenant": "npcd"},
            ]})
        );
    }

    #[tokio::test]
    async fn the_overview_lists_every_substrate() {
        let (_, v) = get_json(&app(two()), "/v1/substrate").await;
        assert_eq!(
            v,
            json!({"tenants": {"zend": {"open": true, "segment_count": 4}, "npcd": null}})
        );
    }

    /// Calibration names its tenant; the tenant's refusal is passed through.
    #[tokio::test]
    async fn calibration_runs_the_named_tenants() {
        let app = app(two());
        assert_eq!(
            call(&app, "POST", "/v1/tools/calibrate", Some("boss"))
                .await
                .0,
            400
        );
        assert_eq!(
            call(
                &app,
                "POST",
                "/v1/tools/calibrate?tenant=nobody",
                Some("boss")
            )
            .await
            .0,
            404
        );
        let (code, body) = call(
            &app,
            "POST",
            "/v1/tools/calibrate?tenant=npcd",
            Some("boss"),
        )
        .await;
        assert_eq!((code, body), (503, json!({"error": "no_engine"})));
    }
}
