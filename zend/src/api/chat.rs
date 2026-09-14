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

use candle_conversation::{OptionalState, SelectionState, NO_THINK_SELECTOR};

use crate::session::{StreamItem, ZendSession};
use crate::types::{
    AssistantMessage, ChatCompletion, ChatCompletionChunk, ChatCompletionRequest, ChatMessage,
    ChunkChoice, CompletionChoice, Delta, Role, ToolMode,
};

/// The `optional_group` selector that gates the whole tool block in the dialogue
/// section-tree (its `id:` in `projection.yaml`). Set to `present`/`absent` from
/// the tools dial so a no-tools turn omits the block entirely.
const TOOLS_ENABLED_SELECTOR: &str = "tools_enabled";

/// The `optional` node holding the tool-call FORMAT demonstration (its `id:` in
/// `projection.yaml`). Tracks the tools dial: a no-tools turn must not be shown
/// a worked tool call, which would contradict a prompt that lists no tools.
pub(crate) const TOOL_EXAMPLE_SELECTOR: &str = "tool_call_example";

/// Map the composer's tools dial onto the tool block: the whole block
/// (`tools_enabled`) and its worked demonstration (`tool_call_example`) are
/// present together or absent together.
///
/// Public so a harness driving the session directly projects the same tool
/// prompt the HTTP API does. The schema defaults the demonstration ABSENT —
/// the ingest layers cannot select it away — so a caller that skips this shows
/// the model a tool catalog with no worked call, which no chat turn ever sees.
pub fn apply_tools_dial(selection: &mut SelectionState, tools_mode: ToolMode) {
    let tools_present = if matches!(tools_mode, ToolMode::None) {
        OptionalState::Absent
    } else {
        OptionalState::Present
    };
    selection.set_optional(TOOLS_ENABLED_SELECTOR, tools_present);
    // The worked example follows the block it demonstrates.
    selection.set_optional(TOOL_EXAMPLE_SELECTOR, tools_present);
}

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
        .find(|m| m.role == Role::User)
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
    let force_hires = req.force_high_resolution;
    let assistant_prefill = req.assistant_prefill;
    let lossless_kv = req.lossless_kv;
    // Composer "tools" dial — which slice of the catalog this conversation
    // projects. Absent → Comprehensive (full catalog).
    let tools_mode = req.tools.unwrap_or_default();
    // Which identity this conversation speaks as. Absent → the conversation's
    // stored identity, else the `mind.yaml` default (resolved in the session).
    let identity = req.identity;
    // The composer dials drive the dialogue layer's section-tree selectors — the
    // projection emits the matching thinking-effort / response-length directive
    // sections (and the `/no_think` node) on this and every subsequent turn until
    // the dials change.
    let mut selection = dial_selection(req.effort, req.verbosity, req.think);
    // Gate the WHOLE tool block (overview, `<tools>`, catalog + glue, `</tools>`)
    // on the tools dial via the `tools_enabled` optional_group: `None` omits the
    // entire block (markers included), the other modes show it. Which *members*
    // appear under Restricted vs Comprehensive is still the mode_builders' job.
    apply_tools_dial(&mut selection, tools_mode);
    let messages = req.messages;
    if req.stream {
        stream_sse(
            session,
            messages,
            max_tokens,
            conv_id,
            force_hires,
            assistant_prefill,
            lossless_kv,
            tools_mode,
            identity,
            model,
            id,
            created,
            selection,
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
            lossless_kv,
            tools_mode,
            identity,
            model,
            id,
            created,
            selection,
        )
        .await
    }
}

/// Map the composer dials to the dialogue section-tree selection.  Only the
/// dials the request actually carries are set; any omitted selector falls back
/// to the schema's authored default (so a new conversation defaults naturally).
///
/// - `effort` 0..4  → `thinking_effort` = off / quick / balanced / deep / exhaustive
/// - `verbosity` 0..4 → `response_length` = terse / concise / standard / detailed / comprehensive
/// - `no_think` = present (suppress) when `effort == 0` or `think == false`, else absent
///
/// Public so a harness driving the session directly selects exactly what the
/// HTTP API would for the same dials, rather than restating the mapping.
pub fn dial_selection(
    effort: Option<u8>,
    verbosity: Option<u8>,
    think: Option<bool>,
) -> SelectionState {
    const EFFORT: [&str; 5] = ["off", "quick", "balanced", "deep", "exhaustive"];
    const LENGTH: [&str; 5] = ["terse", "concise", "standard", "detailed", "comprehensive"];
    let mut sel = SelectionState::new();
    // A thinking-off turn — effort 0, or the `think` toggle explicitly off —
    // must carry BOTH halves of the same decision. The steering has to match the
    // `no_think` node: force `thinking_effort = off` so it resolves to
    // `ThinkMode::Off`, whose tree closes the block immediately. Otherwise a
    // non-zero effort dial drives a steered `<think>` block while the prompt
    // simultaneously tells the model to suppress it, and the two disagree about
    // whether the turn reasons at all.
    let off = think == Some(false) || effort == Some(0);
    if off {
        // Only meaningful when the turn actually carries a dial; a bare
        // `think: false` with no effort still wants the block suppressed.
        if effort.is_some() || think.is_some() {
            sel.select("thinking_effort", "off");
        }
    } else if let Some(e) = effort {
        sel.select(
            "thinking_effort",
            *EFFORT.get(e as usize).unwrap_or(&"exhaustive"),
        );
    }
    if let Some(v) = verbosity {
        sel.select(
            "response_length",
            *LENGTH.get(v as usize).unwrap_or(&"comprehensive"),
        );
    }
    if effort.is_some() || think.is_some() {
        // `no_think` present (the /no_think prefix) ⇔ thinking off. The
        // `reasoning_stance` toggle (its `<think>`-block guidance) is the
        // inverse — present only when the model will actually think — so a
        // /no_think turn never sees instructions on how to use a block it won't
        // open. Both ids match the section-tree nodes in `projection.yaml`.
        sel.set_optional(
            NO_THINK_SELECTOR,
            if off {
                OptionalState::Present
            } else {
                OptionalState::Absent
            },
        );
        sel.set_optional(
            "reasoning_stance",
            if off {
                OptionalState::Absent
            } else {
                OptionalState::Present
            },
        );
    }
    sel
}

// ── Streaming path ────────────────────────────────────────────────────────────

// The submit parameter list — see `ZendSession::submit`.
#[allow(clippy::too_many_arguments)]
async fn stream_sse(
    session: Arc<ZendSession>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    conv_id: String,
    force_hires: Option<String>,
    assistant_prefill: Option<String>,
    lossless_kv: bool,
    tools_mode: crate::types::ToolMode,
    identity: Option<String>,
    model: String,
    id: String,
    created: u64,
    selection: SelectionState,
) -> Response {
    let token_stream = session
        .submit(
            messages,
            max_tokens,
            conv_id,
            force_hires,
            assistant_prefill,
            lossless_kv,
            tools_mode,
            identity,
            selection,
        )
        .await;

    let id_c = id.clone();
    let model_c = model.clone();

    // Track when the first actual token (not a status event) has been sent.
    let saw_token = Arc::new(AtomicBool::new(false));
    let saw_token_c = Arc::clone(&saw_token);

    // **Hold a leading empty think block off the wire.**
    //
    // The engine strips empty `<think></think>` from the text it STORES, but the
    // stream is a separate assembly of raw token events and nothing filtered it,
    // so a collapsed block went out verbatim and rendered as leaked markup. That
    // is now the common case rather than a curiosity: the `off` effort dial
    // closes the block on the token after `<think>`, so every such turn emits
    // one, and thinking-span projection leaves older turns with no visible
    // reasoning so the model collapses its own.
    //
    // `gate_leading_think` decides from the text so far, and the hold is bounded
    // by CONTENT rather than by the closing marker — the first non-blank token
    // inside the block opens the gate for good. A genuine reasoning turn is
    // therefore never withheld waiting for its own `</think>`; at most a few
    // whitespace tokens are ever buffered.
    let mut held = String::new();
    let mut gate_open = false;
    let content_events = token_stream.filter_map(move |result| {
        let out: Option<anyhow::Result<Event>> = match result {
            Err(e) => Some(Err(e)),

            Ok(StreamItem::Status(msg)) => {
                let data = serde_json::json!({ "text": msg }).to_string();
                Some(Ok(Event::default().event("status").data(data)))
            }

            Ok(StreamItem::Projection(event)) => Some(
                serde_json::to_string(&event)
                    .map_err(|e| anyhow::anyhow!(e))
                    .map(|data| Event::default().event("projection").data(data)),
            ),

            Ok(StreamItem::Tool(status)) => Some(
                serde_json::to_string(&status)
                    .map_err(|e| anyhow::anyhow!(e))
                    .map(|data| Event::default().event("tool").data(data)),
            ),

            Ok(StreamItem::Token(text)) => {
                // Once open, the gate never closes again for this turn — a
                // `<think>` later in the answer body is ordinary content.
                let emit = if gate_open {
                    Some(text)
                } else {
                    held.push_str(&text);
                    match crate::think_gate::gate_leading_think(&held) {
                        crate::think_gate::ThinkGate::Hold => None,
                        crate::think_gate::ThinkGate::Open(at) => {
                            gate_open = true;
                            // Everything from the resolve point, which is the
                            // whole buffer when no block was skipped.
                            Some(held[at..].to_string())
                        }
                    }
                };
                // Held, or nothing left after the skip. An empty delta would
                // spend the `role` marker on a chunk carrying no text.
                match emit {
                    None => None,
                    Some(t) if t.is_empty() => None,
                    Some(t) => {
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
                                    content: Some(t),
                                },
                                finish_reason: None,
                            }],
                        };
                        Some(
                            serde_json::to_string(&chunk)
                                .map_err(|e| anyhow::anyhow!(e))
                                .map(|data| Event::default().data(data)),
                        )
                    }
                }
            }
        };
        std::future::ready(out)
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

// The submit parameter list — see `ZendSession::submit`.
#[allow(clippy::too_many_arguments)]
async fn collect_completion(
    session: Arc<ZendSession>,
    messages: Vec<ChatMessage>,
    max_tokens: Option<usize>,
    conv_id: String,
    force_hires: Option<String>,
    assistant_prefill: Option<String>,
    lossless_kv: bool,
    tools_mode: crate::types::ToolMode,
    identity: Option<String>,
    model: String,
    id: String,
    created: u64,
    selection: SelectionState,
) -> Response {
    let mut token_stream = session
        .submit(
            messages,
            max_tokens,
            conv_id,
            force_hires,
            assistant_prefill,
            lossless_kv,
            tools_mode,
            identity,
            selection,
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
            Ok(StreamItem::Tool(_)) => {}   // tool lifecycle; display-only, not in the body
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
mod dial_tests {
    use super::*;

    #[test]
    fn effort_zero_suppresses_thinking() {
        let sel = dial_selection(Some(0), Some(2), None);
        assert_eq!(sel.get("thinking_effort"), Some("off"));
        assert_eq!(sel.get("response_length"), Some("standard"));
        assert_eq!(
            sel.optional(NO_THINK_SELECTOR),
            Some(OptionalState::Present)
        );
        // Thinking off → the reasoning-stance toggle is the inverse (absent), so
        // its <think>-block guidance is not shown.
        assert_eq!(
            sel.optional("reasoning_stance"),
            Some(OptionalState::Absent)
        );
    }

    #[test]
    fn think_toggle_off_forces_effort_off_despite_nonzero_dial() {
        // The bug: effort dial set (quick) but the `think` toggle off. Both the
        // steering (thinking_effort) and the /no_think glue (no_think) must agree on
        // "off" — otherwise the model gets a steered <think> block AND a /no_think
        // suppression signal and closes the block twice.
        let sel = dial_selection(Some(1), Some(2), Some(false));
        assert_eq!(
            sel.get("thinking_effort"),
            Some("off"),
            "think:false must override a non-zero effort dial to off",
        );
        assert_eq!(
            sel.optional(NO_THINK_SELECTOR),
            Some(OptionalState::Present)
        );
        assert_eq!(
            sel.optional("reasoning_stance"),
            Some(OptionalState::Absent)
        );
        // The response-length dial is unaffected.
        assert_eq!(sel.get("response_length"), Some("standard"));
    }

    /// The tools dial moves the tool block and its worked call together: every
    /// mode that shows tools shows the demonstration too, and `None` hides both.
    /// The schema defaults the demonstration absent, so a caller that skipped
    /// this would show a catalog with no worked call.
    #[test]
    fn tools_dial_moves_the_block_and_its_demonstration_together() {
        for (mode, want) in [
            (ToolMode::Comprehensive, OptionalState::Present),
            (ToolMode::Restricted, OptionalState::Present),
            (ToolMode::None, OptionalState::Absent),
        ] {
            let mut sel = SelectionState::new();
            apply_tools_dial(&mut sel, mode);
            assert_eq!(sel.optional(TOOLS_ENABLED_SELECTOR), Some(want), "{mode:?}");
            assert_eq!(sel.optional(TOOL_EXAMPLE_SELECTOR), Some(want), "{mode:?}");
        }
    }

    #[test]
    fn mid_dials_map_to_balanced_standard_thinking_on() {
        let sel = dial_selection(Some(2), Some(2), Some(true));
        assert_eq!(sel.get("thinking_effort"), Some("balanced"));
        assert_eq!(sel.get("response_length"), Some("standard"));
        assert_eq!(sel.optional(NO_THINK_SELECTOR), Some(OptionalState::Absent));
        // Thinking on → reasoning-stance present.
        assert_eq!(
            sel.optional("reasoning_stance"),
            Some(OptionalState::Present)
        );
    }

    #[test]
    fn extremes_map_to_exhaustive_comprehensive() {
        let sel = dial_selection(Some(4), Some(4), Some(true));
        assert_eq!(sel.get("thinking_effort"), Some("exhaustive"));
        assert_eq!(sel.get("response_length"), Some("comprehensive"));
    }

    #[test]
    fn absent_dials_leave_schema_defaults() {
        let sel = dial_selection(None, None, None);
        assert_eq!(sel.get("thinking_effort"), None);
        assert_eq!(sel.get("response_length"), None);
        assert_eq!(sel.optional(NO_THINK_SELECTOR), None);
        // Unset → both toggles fall to their schema defaults (no_think absent,
        // reasoning_stance present — i.e. thinking on).
        assert_eq!(sel.optional("reasoning_stance"), None);
    }
}
