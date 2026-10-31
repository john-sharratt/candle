//! npcd's substrate inspector: the host's views (`host::substrate_views`) over
//! npcd's substrate, read under the mind's projection schema.
//!
//! The overview is npcd's line in the host's `GET /v1/substrate`
//! ([`crate::reports`]); each level below it is a route here, fetched only when
//! the console expands that node:
//!
//! - `GET  /v1/substrate/npcd/system-prompt`    — the shared prompt's framing.
//! - `GET  /v1/substrate/npcd/layer/{name}`     — the conversations in one layer.
//! - `POST /v1/substrate/npcd/layer/{name}/toggle` — flip a layer's projection
//!   kill switch (in-memory, clears on restart).
//! - `GET  /v1/substrate/npcd/timeline/{tl}`    — one conversation's forest.
//!
//! Every view answers `503` until the mind's projection is loaded: there is no
//! schema to read the substrate under before then.

use std::path::Path as FsPath;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use candle_conversation::projection::Conversation;
use host::substrate_views::{
    layer_conversations, overview, system_prompt as system_prompt_view, timeline as timeline_view,
    LayerConversations, SystemPromptView, TimelineDetail, Unit,
};
use serde_json::{json, Value};

use crate::api::Authored;
use crate::engine::mind::Projected;
use crate::engine::minds_of;
use crate::engine::pulse::catalog;

/// Where the views are served, as npcd's line in `/v1/substrate` names them.
pub const VIEWS: &str = "/v1/substrate/npcd/";

/// The substrate and the schema its views read it under. `None` before the
/// mind's projection is loaded.
fn under(s: &Authored) -> Option<(Conversation, Arc<Projected>)> {
    let minds = minds_of(s)?;
    let projected = minds.projected()?;
    Some((minds.substrate(), projected))
}

/// The layer the characters' live conversations are written to.
fn target_layer(p: &Projected) -> String {
    p.builder
        .schema()
        .layers
        .iter()
        .find(|l| l.id == p.layer)
        .map(|l| l.name.clone())
        .unwrap_or_default()
}

/// npcd's line in `GET /v1/substrate`: the overview, where its redo log is,
/// and where its views are served. `null` before the projection is loaded.
pub fn line(s: &Authored, dir: &FsPath) -> Value {
    let Some((conv, p)) = under(s) else {
        return Value::Null;
    };
    let mut v = json!(overview(
        &conv,
        p.builder.schema(),
        None,
        target_layer(&p),
        dir
    ));
    if let Value::Object(m) = &mut v {
        m.insert("path".into(), json!(dir));
        m.insert("views".into(), json!(VIEWS));
    }
    v
}

/// Run `read` — a walk of the substrate — on the blocking pool, so a view
/// waiting on a substrate read never holds one of the runtime's workers.
async fn off_runtime<T: Send + 'static>(
    read: impl FnOnce() -> T + Send + 'static,
) -> Result<T, StatusCode> {
    tokio::task::spawn_blocking(read)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

pub async fn system_prompt(
    State(s): State<Arc<Authored>>,
) -> Result<Json<SystemPromptView>, StatusCode> {
    off_runtime(move || {
        let (conv, p) = under(&s)?;
        let tool_count = catalog(&s)
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let read = conv.read();
        Some(system_prompt_view(
            &read,
            p.builder.schema(),
            target_layer(&p),
            tool_count,
        ))
    })
    .await?
    .map(Json)
    .ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

pub async fn layer(
    State(s): State<Arc<Authored>>,
    Path(name): Path<String>,
) -> Result<Json<LayerConversations>, StatusCode> {
    let found = off_runtime(move || {
        let (conv, p) = under(&s)?;
        // `Some(None)` for a layer the mind does not declare: a typo is a 404,
        // not an empty list that looks like an empty layer.
        let groups = p
            .builder
            .schema()
            .layers
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.groups.clone());
        let read = conv.read();
        Some(groups.map(|g| layer_conversations(&read, &g, None, Unit::titled)))
    })
    .await?;
    match found {
        Some(Some(layer)) => Ok(Json(layer)),
        Some(None) => Err(StatusCode::NOT_FOUND),
        None => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

/// Flip a layer's projection kill switch, answering its new state
/// (`disabled: true` = now excluded from projection assembly).
pub async fn toggle_layer(
    State(s): State<Arc<Authored>>,
    Path(name): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let (conv, p) = under(&s).ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    if !p.builder.schema().layers.iter().any(|l| l.name == name) {
        return Err(StatusCode::NOT_FOUND);
    }
    let disabled = conv.toggle_layer(&name);
    Ok(Json(json!({ "name": name, "disabled": disabled })))
}

pub async fn timeline(
    State(s): State<Arc<Authored>>,
    Path(tl): Path<String>,
) -> Result<Json<TimelineDetail>, StatusCode> {
    // Timeline ids are raw u64 — carried as a string on the wire so a page
    // never rounds them through its 2^53 float mantissa.
    let raw: u64 = tl.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
    let found = off_runtime(move || {
        let (conv, p) = under(&s)?;
        let read = conv.read();
        Some(timeline_view(&read, p.builder.schema(), raw))
    })
    .await?;
    match found {
        Some(Some(t)) => Ok(Json(t)),
        Some(None) => Err(StatusCode::NOT_FOUND),
        None => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}
