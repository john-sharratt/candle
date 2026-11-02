//! `POST /v1/conversations/{id}/diag/decode` — the capped diagnostic decode
//! (`session::diag`), and `GET /v1/conversations/{id}/diag/decode/{run}` — its
//! dump.
//!
//! A decode takes minutes and its dump megabytes, so the call starts it and
//! answers with the run's name, and the dump lands in a file on this member,
//! read back with the GET. Until then the GET answers 404.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use candle_conversation::decode_diag::Control;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::chat::{apply_tools_dial, dial_selection};
use crate::session::diag::{DiagRun, DiagSpec};
use crate::session::{ReplaySampling, ZendSession};
use crate::types::ToolMode;

/// One run of the call.
#[derive(Deserialize)]
pub struct RunBody {
    /// `sparse` (the default), `dense` or `fresh`.
    #[serde(default)]
    pub control: Option<String>,
    /// Absent decodes greedily.
    #[serde(default)]
    pub seed: Option<u64>,
    /// `[timeline, index]` of turns to take out of the target's projections.
    #[serde(default)]
    pub without: Vec<(u64, u32)>,
    /// `[timeline, index]` of turns the recurrent memory is taken over first.
    #[serde(default)]
    pub catch_up: Vec<(u64, u32)>,
}

#[derive(Deserialize)]
pub struct DecodeBody {
    pub turn: usize,
    pub token: usize,
    /// Text forced after `token` ids, for the model to finish.
    #[serde(default)]
    pub prompt: String,
    pub n: usize,
    pub runs: Vec<RunBody>,
    #[serde(default)]
    pub effort: Option<u8>,
    #[serde(default)]
    pub verbosity: Option<u8>,
    #[serde(default)]
    pub think: Option<bool>,
    #[serde(default)]
    pub identity: Option<String>,
}

#[derive(Serialize)]
pub struct Started {
    pub run: String,
}

fn control(name: Option<&str>) -> Result<Control, String> {
    match name {
        None | Some("sparse") => Ok(Control::Sparse),
        Some("dense") => Ok(Control::Dense),
        Some("fresh") => Ok(Control::Fresh),
        Some(other) => Err(format!("no control named {other}")),
    }
}

/// Where run `run`'s dump lands. A run's name is the digits this file gave it,
/// so nothing else can be asked for.
fn dump_path(run: &str) -> Option<PathBuf> {
    (!run.is_empty() && run.bytes().all(|b| b.is_ascii_digit()))
        .then(|| std::env::temp_dir().join(format!("zend-diag-{run}.json")))
}

pub async fn start(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
    Json(body): Json<DecodeBody>,
) -> Result<Json<Started>, (StatusCode, String)> {
    let refused = |e: String| (StatusCode::BAD_REQUEST, e);
    let runs = body
        .runs
        .iter()
        .map(|r| {
            Ok(DiagRun {
                control: control(r.control.as_deref())?,
                sampling: r.seed.map_or(ReplaySampling::Argmax, ReplaySampling::Seed),
                without: r.without.clone(),
                catch_up: r.catch_up.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(refused)?;
    let tools_mode = ToolMode::Comprehensive;
    let mut selection = dial_selection(body.effort, body.verbosity, body.think);
    apply_tools_dial(&mut selection, tools_mode);
    let spec = DiagSpec {
        conv_id: id,
        turn: body.turn,
        token: body.token,
        prompt: body.prompt,
        n: body.n,
        selection,
        tools_mode,
        identity: body.identity,
        runs,
    };
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let run = millis.to_string();
    let path = dump_path(&run).ok_or_else(|| refused("no run name".to_string()))?;
    tokio::task::spawn_blocking(move || {
        let dump = match session.diag_decode(&spec) {
            Ok(report) => serde_json::to_vec(&report),
            Err(e) => serde_json::to_vec(&json!({ "error": format!("{e:#}") })),
        };
        match dump.map(|bytes| std::fs::write(&path, bytes)) {
            Ok(Ok(())) => tracing::info!(path = %path.display(), "diag: dump written"),
            Ok(Err(e)) => tracing::warn!("diag: writing the dump failed: {e}"),
            Err(e) => tracing::warn!("diag: encoding the dump failed: {e}"),
        }
    });
    Ok(Json(Started { run }))
}

pub async fn dump(Path((_, run)): Path<(String, String)>) -> Response {
    let Some(path) = dump_path(&run) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => ([("content-type", "application/json")], bytes).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_control_is_named() {
        assert_eq!(control(None), Ok(Control::Sparse));
        assert_eq!(control(Some("dense")), Ok(Control::Dense));
        assert_eq!(control(Some("fresh")), Ok(Control::Fresh));
        assert_eq!(control(Some("x")), Err("no control named x".to_string()));
    }

    #[test]
    fn only_a_run_name_reaches_a_dump() {
        assert!(dump_path("123").is_some());
        assert!(dump_path("").is_none());
        assert!(dump_path("..\\x").is_none());
    }
}
