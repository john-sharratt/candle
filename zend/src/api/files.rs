//! Conversation-files routes (docs/zend_ui_redesign.md §2.5).
//!
//!   POST   /v1/conversations/{id}/files            multipart upload -> SSE progress
//!   GET    /v1/conversations/{id}/files            list metadata
//!   GET    /v1/conversations/{id}/files/{file_id}  reconstructed content
//!   DELETE /v1/conversations/{id}/files/{file_id}  drop the file
//!
//! Backed by the persistent [`crate::conv_file_store::ConvFileStore`], which is
//! independent of the inference engine — so these work (and are harness-tested)
//! with no model loaded.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Json, Response,
    },
};
use futures::stream;
use serde::Serialize;

use crate::conv_file_store::FileMeta;
use crate::session::ZendSession;

/// Bytes per upload-progress "part" — carve granularity for the GUI bar.
const PART_BYTES: u64 = 8192;

/// POST — store each uploaded file, then stream per-part progress as SSE:
/// `file_start` -> `part`×N -> `file_done` per file, then a final `done`.
pub async fn upload(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
    mut multipart: Multipart,
) -> Response {
    let mut events: Vec<Event> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field
            .file_name()
            .map(str::to_string)
            .or_else(|| field.name().map(str::to_string))
            .unwrap_or_else(|| "file".to_string());
        let bytes = match field.bytes().await {
            Ok(b) => b,
            Err(_) => continue,
        };
        let meta = match session.files().upload(&id, &name, &bytes) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let total = crate::conv_files::part_count(bytes.len() as u64, PART_BYTES);

        events.push(named(
            "file_start",
            serde_json::json!({ "fileId": meta.id, "name": name, "totalParts": total }),
        ));
        for p in 0..total {
            events.push(named(
                "part",
                serde_json::json!({ "fileId": meta.id, "partIndex": p, "totalParts": total }),
            ));
        }
        events.push(named(
            "file_done",
            serde_json::json!({ "fileId": meta.id, "meta": serde_json::to_value(&meta).unwrap_or_default() }),
        ));
    }
    events.push(Event::default().event("done").data("[DONE]"));

    let body = stream::iter(events.into_iter().map(Ok::<Event, Infallible>));
    Sse::new(body).into_response()
}

pub async fn list(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
) -> Json<FilesBody> {
    Json(FilesBody {
        files: session.files().list(&id),
    })
}

pub async fn content(
    State(session): State<Arc<ZendSession>>,
    Path((id, file_id)): Path<(String, u64)>,
) -> Response {
    match session.files().get_content(&id, file_id) {
        Some(bytes) => (
            [(header::CONTENT_TYPE, "application/octet-stream")],
            bytes,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

pub async fn delete(
    State(session): State<Arc<ZendSession>>,
    Path((id, file_id)): Path<(String, u64)>,
) -> StatusCode {
    if session.files().delete(&id, file_id) {
        StatusCode::NO_CONTENT
    } else {
        StatusCode::NOT_FOUND
    }
}

fn named(event: &str, data: serde_json::Value) -> Event {
    Event::default().event(event).data(data.to_string())
}

#[derive(Serialize)]
pub struct FilesBody {
    pub files: Vec<FileMeta>,
}
