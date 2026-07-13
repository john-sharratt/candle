//! Conversation-files routes (docs/zend_ui_redesign.md §2.5).
//!
//!   POST   /v1/conversations/{id}/files            multipart upload -> SSE progress
//!   GET    /v1/conversations/{id}/files            list metadata
//!   GET    /v1/conversations/{id}/files/{file_id}  reconstructed content
//!   DELETE /v1/conversations/{id}/files/{file_id}  drop the file
//!
//! Backed by the persistent [`crate::conv_file_store::ConvFileStore`], which is
//! independent of the inference engine — so upload/list/get/delete work (and are
//! harness-tested) with no model loaded.
//!
//! Upload is a **three-phase** SSE pipeline (streamed live, not buffered):
//!
//! 1. **upload** — write each accepted file to `<workspace>/uploads/` (and the
//!    conv-file store the pane reads): `file_start` -> `part`×N -> `file_done`.
//! 2. **read_file** — trigger the normal `code_read` ingest so the new file is
//!    read into the substrate; wait for it (`phase` events).
//! 3. **analysis** — wait for the summariser to carve + summarise the new
//!    sections on their boundaries (`phase` events).
//!
//! Phases 2–3 are no-ops (skipped) when the model isn't loaded yet.

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

use crate::conv_file_store::FileMeta;
use crate::session::ZendSession;

/// Bytes per upload-progress "part" — carve granularity for the GUI bar.
const PART_BYTES: u64 = 8192;

/// Cap on how long the analysis phase waits for the summariser to drain
/// before returning what's left — bounds a stuck request.
const ANALYSIS_TIMEOUT: Duration = Duration::from_secs(180);

/// One accepted file's bytes, read from the multipart body before the
/// (blocking, engine-bound) phases run.
struct Accepted {
    name: String,
    bytes: Vec<u8>,
}

/// POST — the three-phase upload pipeline, streamed live over SSE.
pub async fn upload(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
    mut multipart: Multipart,
) -> Response {
    // Drain the multipart body first (async), splitting malware-blocked
    // files off from accepted ones. Blocked files are rejected before any
    // bytes touch disk — the server gate is authoritative even if a client
    // skips its own.
    // Time the wire receive so the modal can report upload throughput
    // (bytes / seconds → MB/s). Draining the multipart body IS the network
    // transfer, so bracket the whole loop.
    let recv_start = Instant::now();
    let mut accepted: Vec<Accepted> = Vec::new();
    let mut rejected: Vec<(String, &'static str)> = Vec::new();
    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field
            .file_name()
            .map(str::to_string)
            .or_else(|| field.name().map(str::to_string))
            .unwrap_or_else(|| "file".to_string());
        if let Some(reason) = crate::conv_files::blocked_upload_reason(&name) {
            rejected.push((name, reason));
            continue;
        }
        match field.bytes().await {
            Ok(b) => accepted.push(Accepted {
                name,
                bytes: b.to_vec(),
            }),
            Err(_) => continue,
        }
    }

    // Wire-receive throughput, captured now (before the blocking phases): the
    // total accepted bytes and the wall-clock it took to drain them.
    let upload_bytes: u64 = accepted.iter().map(|a| a.bytes.len() as u64).sum();
    let upload_ms = recv_start.elapsed().as_millis() as u64;

    // Stream events from a blocking task so the engine-bound phases (which
    // hold the engine mutex and drive GPU work) never block the async
    // runtime. `tx` is bounded; SSE backpressure keeps memory flat.
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::task::spawn_blocking(move || {
        let send = |ev: Event| tx.blocking_send(ev).is_ok();

        // Progress `send`s are best-effort: a failed send means the client
        // disconnected, but we must NOT bail before `record_uploads` — the
        // bytes are already on disk and in the pane store, so skipping the
        // substrate event would leave the pane and history divergent. We drain
        // the writes regardless and gate only the expensive engine phases on
        // the client still being connected (`tx.is_closed()`).
        for (name, reason) in rejected {
            let _ = send(named(
                "file_rejected",
                serde_json::json!({ "name": name, "reason": reason }),
            ));
        }

        // ── Phase 1: upload → disk + conv-file store ──────────────────────
        let mut stored_any = false;
        let mut upload_events: Vec<crate::session::UploadInfo> = Vec::new();
        for f in &accepted {
            // Write the raw bytes into the workspace uploads/ dir. The disk
            // write de-dupes the name, so the effective file name is whatever
            // actually landed on disk (`notes.txt` -> `notes-001.txt` on a
            // collision) — use that everywhere so the pane, tile, disk file,
            // and substrate event all agree.
            let disk_path = match session.write_upload_to_disk(&f.name, &f.bytes) {
                Ok(p) => p,
                Err(e) => {
                    let _ = send(named(
                        "file_rejected",
                        serde_json::json!({ "name": f.name, "reason": format!("write failed: {e}") }),
                    ));
                    continue;
                }
            };
            let final_name = disk_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(f.name.as_str())
                .to_string();
            let rel_path = session.workspace_relative(&disk_path);
            // Also record it in the conv-file store that backs the files pane.
            let meta = match session.files().upload(&id, &final_name, &f.bytes) {
                Ok(m) => m,
                Err(_) => continue,
            };
            stored_any = true;
            upload_events.push(crate::session::UploadInfo {
                id: meta.id,
                name: meta.name.clone(),
                // The workspace-relative path (e.g. `uploads/notes-001.txt`),
                // not an absolute host path.
                path: rel_path,
                ext: meta.ext.clone(),
                kind: meta.kind.clone(),
                size: meta.size.clone(),
                added: meta.added.clone(),
                turn_index: 0,
                // Filled in once the pipeline finishes (record_upload_stats).
                stats: None,
            });
            let total = crate::conv_files::part_count(f.bytes.len() as u64, PART_BYTES);
            // Emit the DEDUPED name (`final_name`) so file_start agrees with the
            // file_done meta and the persisted event on a name collision.
            let _ = send(named(
                "file_start",
                serde_json::json!({ "fileId": meta.id, "name": final_name, "totalParts": total }),
            ));
            for p in 0..total {
                let _ = send(named(
                    "part",
                    serde_json::json!({ "fileId": meta.id, "partIndex": p, "totalParts": total }),
                ));
            }
            let _ = send(named(
                "file_done",
                serde_json::json!({ "fileId": meta.id, "meta": serde_json::to_value(&meta).unwrap_or_default() }),
            ));
        }

        // Persist the upload events into the substrate (positioned by turn),
        // so they recover with the conversation and replay inline in history.
        // Always runs — even on a mid-stream disconnect — so the pane and the
        // recovered history never diverge.
        session.record_uploads(&id, &upload_events);

        // Phases 2–3 only make sense once something landed on disk.
        if stored_any {
            // ── Phase 2: read_file → substrate ────────────────────────────
            // Run the ingest on a worker thread and poll its shared progress
            // handle so the GUI gets a real per-scope bar (like the upload
            // bar), not just a spinner.
            let _ = send(named(
                "phase",
                serde_json::json!({ "phase": "read_file", "state": "start" }),
            ));
            let progress = Arc::new(crate::loading::LoadProgress::new());
            let sess = Arc::clone(&session);
            let prog = Arc::clone(&progress);
            // Read ONLY the files this upload wrote — bounded work, never a
            // whole-workspace re-ingest.
            let paths: Vec<String> = upload_events.iter().map(|e| e.path.clone()).collect();
            let ingest_start = Instant::now();
            let worker = std::thread::spawn(move || sess.read_file_phase(&paths, &prog));
            while !worker.is_finished() {
                let (current, total) = progress.step_progress();
                if total > 0 {
                    let s = progress.ingest_stats();
                    let _ = send(named(
                        "phase",
                        serde_json::json!({
                            "phase": "read_file",
                            "state": "progress",
                            "current": current,
                            "total": total,
                            // Live token counters feed the modal's ingest /
                            // summarize stat lines (tokens & t/s).
                            "prefillTokens": s.prefill_tokens,
                            "summaryTokens": s.summary_tokens,
                            "summaryMs": s.summary_ms,
                        }),
                    ));
                }
                std::thread::sleep(Duration::from_millis(150));
            }
            let read = worker.join().unwrap_or(Ok(false));
            let ingest_ms = ingest_start.elapsed().as_millis() as u64;
            let s = progress.ingest_stats();
            let _ = send(named(
                "phase",
                serde_json::json!({
                    "phase": "read_file",
                    "state": "done",
                    "ingested": read.unwrap_or(false),
                    "prefillTokens": s.prefill_tokens,
                    "summaryTokens": s.summary_tokens,
                    "summaryMs": s.summary_ms,
                }),
            ));

            // Persist the measured throughput onto the upload events (so it
            // recovers with the conversation) and stream a final `stats` event.
            // Done right after phase 2 (the ingest that produced the numbers)
            // and unconditionally — the ingest happened, so the stats are real
            // whether or not the client is still listening.
            let stats = crate::session::UploadStats {
                bytes: upload_bytes,
                upload_ms,
                ingest_tokens: s.prefill_tokens,
                ingest_ms,
                summary_tokens: s.summary_tokens,
                summary_ms: s.summary_ms,
            };
            let ids: Vec<u64> = upload_events.iter().map(|e| e.id).collect();
            session.record_upload_stats(&id, &ids, &stats);
            let _ = send(named(
                "stats",
                serde_json::to_value(&stats).unwrap_or_default(),
            ));

            // ── Phase 3: analysis → summariser drain ──────────────────────
            // Skip the (up to ANALYSIS_TIMEOUT) summariser drain if the client
            // has disconnected — it's the expensive open-ended phase, and there
            // is no one to receive its result. The summariser still runs on its
            // own thread on the next natural tick; we just don't block on it.
            if !tx.is_closed() {
                let _ = send(named(
                    "phase",
                    serde_json::json!({ "phase": "analysis", "state": "start" }),
                ));
                let pending = session.analysis_phase(ANALYSIS_TIMEOUT);
                let _ = send(named(
                    "phase",
                    serde_json::json!({
                        "phase": "analysis",
                        "state": "done",
                        "pending": pending,
                    }),
                ));
            }
        }

        let _ = send(Event::default().event("done").data("[DONE]"));
    });

    let body = ReceiverStream::new(rx).map(Ok::<Event, std::convert::Infallible>);
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
        Some(bytes) => {
            ([(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response()
        }
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
