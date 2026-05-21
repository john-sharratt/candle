use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use futures::StreamExt;

use crate::session::{StreamItem, ZendSession};
use crate::types::{
    AssistantMessage, ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, ChunkChoice,
    CompletionChoice, Delta,
};

/// `POST /v1/chat/completions`
pub async fn completions(
    State(session): State<Arc<ZendSession>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let model = req.model.clone().unwrap_or_else(|| "zen-code".into());
    let id = format!("chatcmpl-{}", unix_ms());
    let created = unix_secs();

    let n_messages = req.messages.len();
    let last_preview: String = req
        .messages
        .iter()
        .rev()
        .find(|m| m.role == crate::types::Role::User)
        .map(|m| m.content.chars().take(80).collect())
        .unwrap_or_default();

    tracing::debug!(
        model,
        stream = req.stream,
        messages = n_messages,
        "chat request  \"{}{}\"",
        last_preview,
        if last_preview.len() == 80 { "…" } else { "" },
    );

    let max_tokens = req.max_tokens.map(|n| n as usize);
    let conv_id = req.conv_id.unwrap_or_else(|| "default".to_string());
    if req.stream {
        stream_sse(
            session,
            req.messages,
            max_tokens,
            conv_id,
            model,
            id,
            created,
        )
        .await
    } else {
        collect_completion(
            session,
            req.messages,
            max_tokens,
            conv_id,
            model,
            id,
            created,
        )
        .await
    }
}

// ── Streaming path ────────────────────────────────────────────────────────────

async fn stream_sse(
    session: Arc<ZendSession>,
    messages: Vec<crate::types::ChatMessage>,
    max_tokens: Option<usize>,
    conv_id: String,
    model: String,
    id: String,
    created: u64,
) -> Response {
    let token_stream = session.submit(messages, max_tokens, conv_id).await;

    let id_c = id.clone();
    let model_c = model.clone();

    // Track when the first actual token (not a status event) has been sent.
    let saw_token = Arc::new(AtomicBool::new(false));
    let saw_token_c = Arc::clone(&saw_token);

    let content_events = token_stream.map(move |result| -> anyhow::Result<Event> {
        match result {
            Err(e) => Err(e),

            Ok(StreamItem::Status(msg)) => {
                let data = serde_json::json!({ "text": msg }).to_string();
                Ok(Event::default().event("status").data(data))
            }

            Ok(StreamItem::Token(text)) => {
                let is_first = !saw_token_c.swap(true, Ordering::Relaxed);
                if is_first {
                    tracing::debug!("streaming first token");
                }
                let chunk = ChatCompletionChunk {
                    id: id_c.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_c.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: if is_first { Some("assistant") } else { None },
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                let data = serde_json::to_string(&chunk).map_err(|e| anyhow::anyhow!(e))?;
                Ok(Event::default().data(data))
            }
        }
    });

    let stop_chunk = ChatCompletionChunk {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
            },
            finish_reason: Some("stop"),
        }],
    };
    let stop_data = serde_json::to_string(&stop_chunk).unwrap_or_default();

    let stop_event = futures::stream::once(futures::future::ready(Ok::<Event, anyhow::Error>(
        Event::default().data(stop_data),
    )));
    let done_event = futures::stream::once(async {
        tracing::debug!("stream complete");
        Ok::<Event, anyhow::Error>(Event::default().data("[DONE]"))
    });

    Sse::new(content_events.chain(stop_event).chain(done_event))
        .keep_alive(KeepAlive::default())
        .into_response()
}

// ── Non-streaming path ────────────────────────────────────────────────────────

async fn collect_completion(
    session: Arc<ZendSession>,
    messages: Vec<crate::types::ChatMessage>,
    max_tokens: Option<usize>,
    conv_id: String,
    model: String,
    id: String,
    created: u64,
) -> Response {
    let mut token_stream = session.submit(messages, max_tokens, conv_id).await;
    let mut full = String::new();
    let mut tokens = 0usize;
    while let Some(result) = token_stream.next().await {
        match result {
            Ok(StreamItem::Token(chunk)) => {
                tokens += 1;
                full.push_str(&chunk);
            }
            Ok(StreamItem::Status(_)) => {} // status events are display-only
            Err(_) => {}
        }
    }
    tracing::debug!(tokens, "non-stream complete");
    Json(ChatCompletion {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![CompletionChoice {
            index: 0,
            message: AssistantMessage {
                role: "assistant",
                content: full,
            },
            finish_reason: "stop",
        }],
    })
    .into_response()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
