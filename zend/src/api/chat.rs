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
    AssistantMessage, ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, ChatMessage,
    ChunkChoice, CompletionChoice, Delta, Role,
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

    let effort = req.effort;
    let think = req.think;
    let max_tokens = req.max_tokens.map(|n| n as usize);
    let conv_id = req.conv_id.unwrap_or_else(|| "default".to_string());
    let force_hires = req.force_high_resolution;
    let assistant_prefill = req.assistant_prefill;
    // Composer "thinking effort" dial: Off (`effort: 0` / `think: false`)
    // suppresses the reasoning channel via the `/no_think` dialect prefix
    // (docs/zend_ui_redesign.md decision 10). Stripped again on hydrate by
    // `crate::chatml::split_turn`.
    let messages = apply_no_think(req.messages, effort, think);
    if req.stream {
        stream_sse(
            session,
            messages,
            max_tokens,
            conv_id,
            force_hires,
            assistant_prefill,
            model,
            id,
            created,
        )
        .await
    } else {
        collect_completion(
            session,
            messages,
            max_tokens,
            conv_id,
            force_hires,
            assistant_prefill,
            model,
            id,
            created,
        )
        .await
    }
}

/// Prepend the `/no_think` dialect prefix to the most recent user turn when the
/// composer requests no-thinking (`think == Some(false)` or `effort == Some(0)`).
/// Idempotent — never double-prefixes.
fn apply_no_think(
    mut messages: Vec<ChatMessage>,
    effort: Option<u8>,
    think: Option<bool>,
) -> Vec<ChatMessage> {
    let suppress = think == Some(false) || effort == Some(0);
    if suppress {
        if let Some(m) = messages.iter_mut().rev().find(|m| m.role == Role::User) {
            if !m.content.starts_with("/no_think") {
                m.content = format!("/no_think\n{}", m.content);
            }
        }
    }
    messages
}

// ── Streaming path ────────────────────────────────────────────────────────────

async fn stream_sse(
    session: Arc<ZendSession>,
    messages: Vec<crate::types::ChatMessage>,
    max_tokens: Option<usize>,
    conv_id: String,
    force_hires: Option<String>,
    assistant_prefill: Option<String>,
    model: String,
    id: String,
    created: u64,
) -> Response {
    let token_stream = session
        .submit(
            messages,
            max_tokens,
            conv_id,
            force_hires,
            assistant_prefill,
        )
        .await;

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

            Ok(StreamItem::Projection(event)) => {
                let data = serde_json::to_string(&event).map_err(|e| anyhow::anyhow!(e))?;
                Ok(Event::default().event("projection").data(data))
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
    force_hires: Option<String>,
    assistant_prefill: Option<String>,
    model: String,
    id: String,
    created: u64,
) -> Response {
    let mut token_stream = session
        .submit(
            messages,
            max_tokens,
            conv_id,
            force_hires,
            assistant_prefill,
        )
        .await;
    let mut full = String::new();
    let mut tokens = 0usize;
    while let Some(result) = token_stream.next().await {
        match result {
            Ok(StreamItem::Token(chunk)) => {
                tokens += 1;
                full.push_str(&chunk);
            }
            Ok(StreamItem::Status(_)) => {} // status events are display-only
            Ok(StreamItem::Projection(_)) => {} // timeline-only; not in the collected body
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

#[cfg(test)]
mod no_think_tests {
    use super::*;

    fn user(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: content.to_string(),
        }
    }

    #[test]
    fn think_false_prepends_no_think() {
        let out = apply_no_think(vec![user("trace the redo log")], None, Some(false));
        assert_eq!(out[0].content, "/no_think\ntrace the redo log");
    }

    #[test]
    fn effort_zero_prepends_no_think() {
        let out = apply_no_think(vec![user("hello")], Some(0), None);
        assert_eq!(out[0].content, "/no_think\nhello");
    }

    #[test]
    fn default_dials_leave_message_untouched() {
        let out = apply_no_think(vec![user("hello")], Some(2), None);
        assert_eq!(out[0].content, "hello");
    }

    #[test]
    fn only_the_last_user_turn_is_prefixed() {
        let msgs = vec![
            user("first"),
            ChatMessage {
                role: Role::Assistant,
                content: "reply".into(),
            },
            user("second"),
        ];
        let out = apply_no_think(msgs, Some(0), None);
        assert_eq!(out[0].content, "first");
        assert_eq!(out[2].content, "/no_think\nsecond");
    }

    #[test]
    fn idempotent_when_already_prefixed() {
        let out = apply_no_think(vec![user("/no_think\nhi")], Some(0), None);
        assert_eq!(out[0].content, "/no_think\nhi");
    }
}
