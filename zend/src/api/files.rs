//! Conversation-files routes (docs/zend_ui_redesign.md §2.5).
//!
//!   POST   /v1/conversations/{id}/files            multipart upload -> SSE progress
//!   GET    /v1/conversations/{id}/files            list metadata
//!   GET    /v1/conversations/{id}/files/{file_id}  the file's bytes
//!   DELETE /v1/conversations/{id}/files/{file_id}  drop the file
//!
//! An upload is a file in the conversation's `uploads` store, saved as its
//! file events (`docs/replicated_substrate.md` §10.2), so it replicates and
//! moves with the conversation; the conversation's upload records name each
//! one for the pane. Both are the substrate's, so every route answers `503`
//! until the model is loaded.
//!
//! Upload is a **two-phase** SSE pipeline (streamed live, not buffered):
//!
//! 1. **upload** — each accepted file into the conversation's uploads store,
//!    under a safe, unused name: `file_start` -> `part`×N -> `file_done`; one
//!    past the store's 10 MiB cap is `file_rejected` with the cap named.
//! 2. **read_file** — trigger the normal `code_read` ingest so the new file is
//!    read into the substrate; wait for it (`phase` events).
//!
//! The upload then kicks the background summariser and returns immediately,
//! never waiting on it or reporting its progress. **That kick currently does
//! nothing** — the summariser is disconnected (`ConversationEngine::new`), so
//! the scope turns are stored whole and no summary tree is rolled up over them.
//! The upload path is unaffected either way, which is the point of it being
//! fire-and-forget.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{
        sse::{Event, Sse},
        IntoResponse, Json, Response,
    },
};
use futures::StreamExt;
use serde::Serialize;
use tokio_stream::wrappers::ReceiverStream;

use crate::session::{UploadInfo, ZendSession};

/// Bytes per upload-progress "part" — carve granularity for the GUI bar.
const PART_BYTES: u64 = 8192;

/// File metadata as the pane consumes it (`size`/`added` are display strings).
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FileMeta {
    pub id: u64,
    pub name: String,
    pub ext: String,
    pub kind: String,
    pub size: String,
    pub added: String,
}

impl FileMeta {
    fn of(u: &UploadInfo) -> FileMeta {
        FileMeta {
            id: u.id,
            name: u.name.clone(),
            ext: u.ext.clone(),
            kind: u.kind.clone(),
            size: u.size.clone(),
            added: u.added.clone(),
        }
    }
}

/// POST — the two-phase upload pipeline, streamed live over SSE.
pub async fn upload(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
    mut multipart: Multipart,
) -> Response {
    if !session.is_loaded() {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    }
    // Drain the multipart body first (async), splitting malware-blocked
    // files off from accepted ones. Blocked files are rejected before any
    // bytes are stored — the server gate is authoritative even if a client
    // skips its own.
    // Time the wire receive so the modal can report upload throughput
    // (bytes / seconds → MB/s). Draining the multipart body IS the network
    // transfer, so bracket the whole loop.
    let recv_start = Instant::now();
    let mut accepted: Vec<(String, Vec<u8>)> = Vec::new();
    let mut rejected: Vec<(String, String)> = Vec::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field
            .file_name()
            .map(str::to_string)
            .or_else(|| field.name().map(str::to_string))
            .unwrap_or_else(|| "file".to_string());
        if let Some(reason) = crate::conv_files::blocked_upload_reason(&name) {
            rejected.push((name, reason.to_string()));
            continue;
        }
        if let Ok(b) = field.bytes().await {
            accepted.push((name, b.to_vec()));
        }
    }

    // Wire-receive throughput, captured now (before the blocking phases): the
    // total accepted bytes and the wall-clock it took to drain them.
    let upload_bytes: u64 = accepted.iter().map(|(_, b)| b.len() as u64).sum();
    let upload_ms = recv_start.elapsed().as_millis() as u64;

    // Stream events from an async task; the engine-bound phase inside it runs
    // on the blocking pool so it never stalls the runtime. `tx` is bounded;
    // SSE backpressure keeps memory flat.
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::spawn(async move {
        // ── Phase 1: upload → the conversation's uploads store ────────────
        //
        // Progress `send`s are best-effort: a failed send means the client
        // disconnected, but the phase must NOT bail before `record_uploads` —
        // the files are already in the conversation's store, so skipping the
        // records would leave the pane and the files divergent.
        let stored = session
            .store_uploads(&id, accepted)
            .await
            .unwrap_or_default();
        for (name, reason) in rejected {
            let _ = tx
                .send(named(
                    "file_rejected",
                    serde_json::json!({ "name": name, "reason": reason }),
                ))
                .await;
        }
        let mut upload_events: Vec<UploadInfo> = Vec::new();
        for outcome in stored {
            let info = match outcome {
                Ok(info) => info,
                Err((name, reason)) => {
                    let _ = tx
                        .send(named(
                            "file_rejected",
                            serde_json::json!({ "name": name, "reason": reason }),
                        ))
                        .await;
                    continue;
                }
            };
            let meta = FileMeta::of(&info);
            let total = crate::conv_files::part_count(info.bytes, PART_BYTES);
            let _ = tx
                .send(named(
                    "file_start",
                    serde_json::json!({ "fileId": meta.id, "name": meta.name, "totalParts": total }),
                ))
                .await;
            for p in 0..total {
                let _ = tx
                    .send(named(
                        "part",
                        serde_json::json!({ "fileId": meta.id, "partIndex": p, "totalParts": total }),
                    ))
                    .await;
            }
            let _ = tx
                .send(named(
                    "file_done",
                    serde_json::json!({ "fileId": meta.id, "meta": serde_json::to_value(&meta).unwrap_or_default() }),
                ))
                .await;
            upload_events.push(info);
        }
        // Positioned by turn, so they replay inline in history. Always runs —
        // even on a mid-stream disconnect — so the pane and the files never
        // diverge.
        let recording = Arc::clone(&session);
        let (conv, events) = (id.clone(), upload_events.clone());
        let _ = tokio::task::spawn_blocking(move || recording.record_uploads(&conv, &events)).await;

        // Phase 2 only makes sense once something was stored.
        if !upload_events.is_empty() {
            // ── Phase 2: read_file → substrate ────────────────────────────
            // The ingest is blocking machinery (the per-file pool), so it runs
            // on the blocking pool while this task watches its shared progress
            // handle on an async tick — the GUI gets a real per-scope bar
            // (like the upload bar), not just a spinner, and the tick doubles
            // as the SSE cadence throttle.
            let _ = tx
                .send(named(
                    "phase",
                    serde_json::json!({ "phase": "read_file", "state": "start" }),
                ))
                .await;
            // Throwaway progress sink for the upload's read_file bar — silent so
            // it doesn't log "load step started Loading model".
            let progress = Arc::new(crate::loading::LoadProgress::silent());
            let sess = Arc::clone(&session);
            let prog = Arc::clone(&progress);
            // Read ONLY the files this upload stored — bounded work, never a
            // whole-workspace re-ingest.
            let paths: Vec<String> = upload_events.iter().map(|e| e.path.clone()).collect();
            let conv = id.clone();
            let ingest_start = Instant::now();
            let mut worker =
                tokio::task::spawn_blocking(move || sess.read_file_phase(&conv, &paths, &prog));
            let joined = loop {
                tokio::select! {
                    joined = &mut worker => break joined,
                    _ = tokio::time::sleep(Duration::from_millis(150)) => {
                        let (current, total) = progress.step_progress();
                        if total > 0 {
                            let s = progress.ingest_stats();
                            let _ = tx
                                .send(named(
                                    "phase",
                                    serde_json::json!({
                                        "phase": "read_file",
                                        "state": "progress",
                                        "current": current,
                                        "total": total,
                                        // Live token counter feeds the modal's
                                        // ingest stat line (tokens & t/s).
                                        "prefillTokens": s.prefill_tokens,
                                    }),
                                ))
                                .await;
                        }
                    }
                }
            };
            let (ingested, failed) = joined
                .unwrap_or(Ok((false, false)))
                .unwrap_or((false, false));
            let ingest_ms = ingest_start.elapsed().as_millis() as u64;
            let s = progress.ingest_stats();
            let _ = tx
                .send(named(
                    "phase",
                    serde_json::json!({
                        "phase": "read_file",
                        "state": "done",
                        "ingested": ingested,
                        // At least one file couldn't be read (typically the GPU
                        // ran out of KV VRAM mid-prefill) — the client shows
                        // this as a failed stage instead of a silent "done"
                        // with no stats.
                        "failed": failed,
                        "prefillTokens": s.prefill_tokens,
                    }),
                ))
                .await;

            // Summarisation is a fully background task — the upload NEVER waits
            // on it, shows no progress for it, and records nothing about it.
            if ingested {
                session.kick_summariser();
            }

            // Persist the measured throughput onto the upload records (so it
            // recovers with the conversation) and stream a final `stats` event.
            let stats = crate::session::UploadStats {
                bytes: upload_bytes,
                upload_ms,
                ingest_tokens: s.prefill_tokens,
                ingest_ms,
            };
            let ids: Vec<u64> = upload_events.iter().map(|e| e.id).collect();
            session.record_upload_stats(&id, &ids, &stats);
            let _ = tx
                .send(named(
                    "stats",
                    serde_json::to_value(&stats).unwrap_or_default(),
                ))
                .await;
        }

        let _ = tx.send(Event::default().event("done").data("[DONE]")).await;
    });

    let body = ReceiverStream::new(rx).map(Ok::<Event, std::convert::Infallible>);
    Sse::new(body).into_response()
}

pub async fn list(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
) -> Result<Json<FilesBody>, StatusCode> {
    if !session.is_loaded() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let mut files: Vec<FileMeta> = session
        .conversation_uploads(&id)
        .iter()
        .map(FileMeta::of)
        .collect();
    files.reverse();
    Ok(Json(FilesBody { files }))
}

pub async fn content(
    State(session): State<Arc<ZendSession>>,
    Path((id, file_id)): Path<(String, u64)>,
) -> Response {
    match session.upload_content(&id, file_id) {
        Some(Some(bytes)) => {
            ([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response()
        }
        Some(None) => StatusCode::NOT_FOUND.into_response(),
        None => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}

pub async fn delete(
    State(session): State<Arc<ZendSession>>,
    Path((id, file_id)): Path<(String, u64)>,
) -> StatusCode {
    match session.delete_upload(&id, file_id).await {
        Some(true) => StatusCode::NO_CONTENT,
        Some(false) => StatusCode::NOT_FOUND,
        None => StatusCode::SERVICE_UNAVAILABLE,
    }
}

fn named(event: &str, data: serde_json::Value) -> Event {
    Event::default().event(event).data(data.to_string())
}

#[derive(Serialize)]
pub struct FilesBody {
    pub files: Vec<FileMeta>,
}
