//! `POST /v1/expert_split` — the CPU expert split's knobs, live
//! (`docs/cpu_expert_compute.md` §5.3.1).
//!
//! The body is a JSON object naming any subset of the knobs; each named knob
//! takes effect on the next wave, with no rebuild and no restart. An empty body
//! changes nothing. Either way the answer is the split as it now stands: the
//! knobs, the costs this machine measured at load, and the cut they give.
//!
//! Whether the split runs at all is the config's (`expert_split.enabled`): the
//! pool is started, and its rings sized, as the engine loads. A cap raised past
//! the rings' rows is held to them by the dispatch.

use axum::body::Bytes;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use candle_conversation::expert_split::{shared, SplitParams};
use candle_conversation::expert_split_served;

pub async fn expert_split(body: Bytes) -> Response {
    let patch: Value = if body.iter().all(u8::is_ascii_whitespace) {
        json!({})
    } else {
        match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => return refused(&format!("the body is not JSON: {e}")),
        }
    };
    let params = shared();
    let mut p = params.lock().unwrap_or_else(|e| e.into_inner());
    if let Err(e) = p.patch_knobs(&patch) {
        return refused(&e);
    }
    Json(view(&p)).into_response()
}

fn refused(why: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": why }))).into_response()
}

/// The split as it now stands.
pub(crate) fn view(p: &SplitParams) -> Value {
    let split = p.split();
    json!({
        "enabled": p.enabled,
        "calibrated": p.measured.is_some(),
        "knobs": p.knobs,
        "measured": p.measured,
        // What the device is handed: 0 rows when off or not yet calibrated.
        "cap_rows": split.cap_rows,
        "max_n": split.max_n,
        // Layers the pool has computed since the process started.
        "served_layers": expert_split_served(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_conversation::expert_split::Knobs;

    use crate::endpoints::test_support::{app, send, shared as host_shared};

    /// Through the guarded route: an admin's empty body reads the split, a
    /// patch naming a knob that does not exist is refused whole with a 400,
    /// and a caller who is not an admin is refused.
    #[tokio::test]
    async fn the_route_reads_patches_and_refuses() {
        let app = app(host_shared(vec![]));
        let (status, v) = send(&app, "POST", "/v1/expert_split", Some("boss"), None).await;
        assert_eq!(status, 200, "{v}");
        assert!(v["knobs"]["max_cpu_assignments"].is_u64(), "{v}");
        let (status, v) = send(
            &app,
            "POST",
            "/v1/expert_split",
            Some("boss"),
            Some(r#"{"cpu_sclae": 2.0}"#),
        )
        .await;
        assert_eq!(status, 400, "{v}");
        assert!(v["error"].as_str().unwrap().contains("cpu_sclae"), "{v}");
        let (status, _) = send(&app, "POST", "/v1/expert_split", Some("boss"), Some("{")).await;
        assert_eq!(status, 400);
        let (status, _) = send(&app, "POST", "/v1/expert_split", Some("someone"), None).await;
        assert_eq!(status, 403);
    }

    /// The view names the knobs by their fields, says whether the pool has
    /// calibrated, and reports the all-GPU split until it has.
    #[test]
    fn the_view_reports_the_knobs_and_an_off_split_until_calibrated() {
        let p = SplitParams {
            enabled: true,
            knobs: Knobs::default(),
            measured: None,
            max_experts: 0,
        };
        let v = view(&p);
        assert_eq!(v["enabled"], true);
        assert_eq!(v["calibrated"], false);
        assert_eq!(v["knobs"]["max_cpu_assignments"], 1024);
        assert_eq!(v["knobs"]["cpu_scale"], 1.0);
        assert_eq!(v["measured"], Value::Null);
        assert_eq!(v["cap_rows"], 0);
    }
}
