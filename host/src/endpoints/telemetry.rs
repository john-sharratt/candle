//! `GET /v1/telemetry` and `GET /v1/memory`.
//!
//! `/v1/telemetry` carries two different kinds of data side by side, because
//! they are different data and not the same data twice: the host's sampled
//! hour (`gpu`, `model`, `host`, `series`, `sample_period_s`, `uptime_s`, every
//! tenant's columns on one time axis) and the engine's rows, one per 5-second
//! slot of the wall clock (`waves`, `mix`, `mix_formats`, `migrate`,
//! `has_pressure`), with `fuse` saying how the answer adds up across a cluster
//! (`docs/performance_series.md`). Memory is in MiB throughout.

use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde_json::{to_value, Map, Result as JsonResult, Value};

use super::waves::wave_series;
use super::Shared;
use crate::telemetry::memory::{dump, MemoryDump};

pub async fn telemetry(State(s): State<Arc<Shared>>) -> Json<Value> {
    let body = tokio::task::spawn_blocking(move || {
        let mut body = as_object(to_value(s.telemetry.read()));
        body.extend(as_object(to_value(wave_series())));
        Value::Object(body)
    })
    .await
    .unwrap_or(Value::Null);
    Json(body)
}

fn as_object(v: JsonResult<Value>) -> Map<String, Value> {
    match v {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

/// Full memory accounting: the engine's report, and the OS and this process at
/// request time. `sysinfo` blocks, briefly — off the async workers.
pub async fn memory() -> Json<MemoryDump> {
    Json(
        tokio::task::spawn_blocking(dump)
            .await
            .expect("memory probe does not panic"),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use crate::endpoints::test_support::{app, get_json, shared};

    /// Both halves, side by side, with the host in MiB and no engine reported
    /// as absent rather than as zero.
    #[tokio::test]
    async fn both_halves_are_served_together() {
        let (code, v) = get_json(&app(shared(vec![])), "/v1/telemetry").await;
        assert_eq!(code, 200);
        for key in [
            "gpu",
            "model",
            "host",
            "series",
            "sample_period_s",
            "uptime_s",
        ] {
            assert!(v.get(key).is_some(), "{key} missing: {v}");
        }
        for key in [
            "waves",
            "mix",
            "mix_formats",
            "migrate",
            "has_pressure",
            "fuse",
        ] {
            assert!(v.get(key).is_some(), "{key} missing: {v}");
        }
        assert!(v["host"]["total_mib"].as_u64().is_some_and(|m| m > 0));
        assert_eq!(v["engine_measured"], false);
        assert_eq!(v["has_pressure"], false);
    }

    #[tokio::test]
    async fn memory_reports_the_machine_and_this_process() {
        let (code, v) = get_json(&app(shared(vec![])), "/v1/memory").await;
        assert_eq!(code, 200);
        assert!(v["host_now"]["total_bytes"].as_u64().unwrap() > 0);
        assert!(v["process"]["working_set_bytes"].as_u64().unwrap() > 0);
        assert_eq!(v["report"], Value::Null);
    }
}
