use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt};

use candle_conversation::models::Model;
use candle_conversation::persistence::content_hash;
use candle_conversation::projection;
use candle_conversation::{ConversationEngine, TokenDecoder, TurnEvent};

use crate::config::DaemonConfig;
use crate::log_broadcast::LogBus;
use crate::tools::{
    extract_tool_calls, format_tool_responses, install_tool_catalog, run_tool_calls, ToolHost,
    MAX_TOOL_ITERATIONS,
};
use crate::types::{ChatMessage, Role};

const PROJECTION_SCHEMA_TEMPLATE: &str = include_str!("prompts/projection.yaml");

// The number of tools surfaced into the system prompt is governed by
// the `selection: { kind: top_k, k: N }` rule on the `tools` collection
// in `prompts/projection.yaml`.  Tune K there.

// ── Stream item ───────────────────────────────────────────────────────────────

/// Items yielded by [`ZendSession::submit`].
pub enum StreamItem {
    Status(String),
    Token(String),
}

// ── Inference state ───────────────────────────────────────────────────────────

struct ConvState {
    conv: candle_conversation::Sequence,
}

struct InferenceState {
    decoder: TokenDecoder,
    /// Owns the scheduler `JoinHandle` (conversations hold cloned
    /// `scheduler_tx` senders) and the substrate persistence handle —
    /// locked for the per-turn group-commit and for shutdown checkpointing.
    engine: std::sync::Mutex<ConversationEngine>,
    /// Per-conversation state, keyed by the client-supplied conv_id string.
    conversations: std::sync::Mutex<HashMap<String, Arc<std::sync::Mutex<ConvState>>>>,
    /// System-prompt already prefilled; all new conversations fork from this.
    base_conv: std::sync::Mutex<candle_conversation::Sequence>,
    /// Shared tool execution context (notes / credentials / sessions /
    /// VFS stores).  Cloned per-conversation if scoping is later
    /// needed; for now it's workspace-wide.
    tool_host: ToolHost,
}

impl InferenceState {
    fn load(
        mut proj_builder: projection::Builder,
        model_path: PathBuf,
        tokenizer_path: PathBuf,
        workspace: PathBuf,
    ) -> anyhow::Result<Arc<Self>> {
        let device = candle::Device::cuda_if_available(0)
            .map_err(|e| anyhow::anyhow!("device init: {e}"))?;

        // Resolve the dialogue layer / primary group up front.  The
        // tool-catalog injection adds sections to that layer.
        let dialogue_layer = proj_builder
            .id_for_layer("dialogue")
            .ok_or_else(|| anyhow::anyhow!("projection schema missing 'dialogue' layer"))?;
        let primary_group = proj_builder
            .id_for_group("primary_conversation")
            .ok_or_else(|| {
                anyhow::anyhow!("projection schema missing 'primary_conversation' group")
            })?;
        // (layer, group) are passed through to
        // `new_conversation_with_projection`, which mints a fresh
        // `TimelineId` internally.

        // Install the tool catalog into the schema before building the
        // base conversation.  Each tool gets a section in dialogue's
        // system_prompt; the layer's section selection switches to
        // TopK so only the K most relevant survive into projection.
        let tool_sections = install_tool_catalog(&mut proj_builder, dialogue_layer)
            .map_err(|e| anyhow::anyhow!("tool catalog install: {e}"))?;
        tracing::info!(
            n_tools = tool_sections.len(),
            "tool catalog installed (top_k governed by `tools` collection in projection.yaml)",
        );

        // The dialogue layer's `system_prompt.items` start with a static
        // prelude (mode/frame/history_stance/grounding/tools_intro) →
        // then the `tools` collection (90+ tool sections, top_k=3) →
        // then `tools_outro`.  The pre-collection prelude is what we
        // pass as the engine's `system_prompt` so it gets ChatML-wrapped
        // for the dialect; everything after the first Collection is
        // expanded by `preemptive_prefill` itself.
        let before_text: String = pre_collection_prelude(&proj_builder);

        let mut builder = Model::Qwen3_30B_A3B_Q4
            .builder()
            .system_prompt(&before_text)
            .model_path(model_path)
            .tokenizer_path(tokenizer_path)
            .workspace_path(workspace)
            .thinking(false);
        let engine = builder
            .engine(&device)
            .map_err(|e| anyhow::anyhow!("engine build: {e}"))?;
        let conv_config = builder.conversation_config();
        let formatted_prompt = builder.format_system_prompt();
        let decoder = engine.token_decoder();

        let base_conv = engine
            .new_conversation_with_projection(
                &formatted_prompt,
                proj_builder,
                dialogue_layer,
                primary_group,
                conv_config.clone(),
            )
            .map_err(|e| anyhow::anyhow!("base conv create: {e}"))?;

        // Section ingestion (prelude + parallel tool-section forks +
        // outro) happens eagerly inside
        // `new_conversation_with_projection`: the schema's declared
        // sections / collections are pinned into the workspace
        // substrate at construction time via `insert_section` /
        // `insert_section_collection`.  Each tool section's BDP sigs come
        // from that single prefill of its JSON definition; JSON
        // structural tokens are excluded from the reprojection probe
        // (`build_reproject_probe_filter_token_ids`) so they don't add
        // shared-structure noise to section scoring.
        tracing::info!(
            n_tool_sections = tool_sections.len(),
            "base conversation ready (prelude + tool catalog + outro pinned at init)",
        );

        Ok(Arc::new(Self {
            decoder,
            engine: std::sync::Mutex::new(engine),
            conversations: std::sync::Mutex::new(HashMap::new()),
            base_conv: std::sync::Mutex::new(base_conv),
            tool_host: ToolHost::new(),
        }))
    }
}

/// Run a complete user request: stream tokens to the client as they
/// arrive, then on turn completion scan the response text for tool
/// calls.  If calls are found, dispatch them and loop with a
/// `<tool_response>` user turn; otherwise the turn is the final answer.
///
/// Every turn's tokens are streamed to the client — including tool-call
/// turns.  `<tool_call>` markup appears at the tail of those responses
/// so the user sees the natural-language prefix streamed live and the
/// tool markup appear at the end before the follow-up response begins.
fn run_inference_stream(
    state: Arc<InferenceState>,
    conv_id: String,
    user_message: String,
    max_tokens: Option<usize>,
    sampling: Option<candle_conversation::SamplingConfig>,
) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send + 'static>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<String>>(64);

    tokio::task::spawn_blocking(move || {
        let msg_preview: String = user_message.chars().take(60).collect();
        tracing::info!(
            conv_id = %conv_id,
            msg_len = user_message.len(),
            "inference start  \"{}{}\"",
            msg_preview,
            if user_message.len() > 60 { "…" } else { "" },
        );

        let conv_arc: Arc<std::sync::Mutex<ConvState>> = {
            let mut map = state.conversations.lock().unwrap();
            if let Some(existing) = map.get(&conv_id) {
                tracing::debug!(conv_id = %conv_id, "reusing existing conv");
                Arc::clone(existing)
            } else {
                tracing::info!(conv_id = %conv_id, "forking new conv from base");
                // Fork onto the timeline derived from `conv_id` — a stable
                // hash, so a daemon restart reconnects the client to the
                // turns the substrate reload recovered for this conversation
                // (§16.12). An unknown conv_id simply forks empty.
                let conv = match state
                    .base_conv
                    .lock()
                    .unwrap()
                    .fork_resuming(timeline_for(&conv_id))
                {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!(conv_id = %conv_id, "fork failed: {e}");
                        let _ = tx.blocking_send(Err(anyhow::anyhow!("{e}")));
                        return;
                    }
                };
                let arc = Arc::new(std::sync::Mutex::new(ConvState { conv }));
                map.insert(conv_id.clone(), Arc::clone(&arc));
                arc
            }
        };

        let mut cs = conv_arc.lock().unwrap();
        let mut current_message = user_message;

        for iteration in 0..=MAX_TOOL_ITERATIONS {
            tracing::debug!(conv_id = %conv_id, iteration, "submitting turn");
            let options = candle_conversation::TurnOptions {
                max_tokens,
                sampling: sampling.clone(),
                ..Default::default()
            };
            let handle = match cs.conv.submit_turn_with_options(&current_message, options) {
                Ok(h) => h,
                Err(e) => {
                    tracing::error!(conv_id = %conv_id, iteration, "submit_turn failed: {e}");
                    let _ = tx.blocking_send(Err(anyhow::anyhow!("{e}")));
                    return;
                }
            };

            // Stream tokens to the client as they arrive.  We track emitted_len
            // (byte offset into the full decoded string) so incremental deltas
            // are correct across BPE byte-fallback sequences.  Fragments that
            // still contain U+FFFD are held back until the sequence completes.
            let mut tokens: Vec<u32> = Vec::new();
            let mut emitted_len: usize = 0;
            let mut done_resp = None;
            let mut turn_error: Option<anyhow::Error> = None;
            let mut client_gone = false;

            for event in handle.stream() {
                match event {
                    TurnEvent::Token(id) => {
                        if turn_error.is_none() {
                            tokens.push(id);
                            let text = state.decoder.decode(&tokens);
                            if text.len() > emitted_len && text.is_char_boundary(emitted_len) {
                                let new_part = &text[emitted_len..];
                                if !new_part.contains('\u{FFFD}') {
                                    if tx.blocking_send(Ok(new_part.to_string())).is_err() {
                                        // Client closed the connection.  Break
                                        // immediately so `handle` is dropped on
                                        // return, which closes event_rx and causes
                                        // the scheduler's next send to fail →
                                        // state.finished = true → decode stops.
                                        client_gone = true;
                                        break;
                                    }
                                    emitted_len = text.len();
                                }
                            }
                        }
                    }
                    TurnEvent::Done(resp) => {
                        tracing::info!(
                            conv_id = %conv_id,
                            iteration,
                            tokens = resp.stats.tokens_generated,
                            tps    = resp.stats.tokens_per_second as u32,
                            prefill_ms = resp.stats.prefill_ms as u32,
                            "turn complete",
                        );
                        done_resp = Some(resp);
                    }
                    TurnEvent::Error(e) => {
                        let msg = format!("{e}");
                        tracing::error!(conv_id = %conv_id, iteration, "scheduler error: {msg}");
                        // Send as text so the client shows the message rather
                        // than dropping the connection.
                        let _ = tx.blocking_send(Ok(format!("\n\n⚠ {msg}")));
                        turn_error = Some(anyhow::anyhow!("{msg}"));
                        // Do not return — drain the iterator so the channel
                        // closes cleanly before we decide what to do with the
                        // conversation state.
                    }
                    TurnEvent::HealthWarning(msg) => {
                        tracing::warn!(conv_id = %conv_id, "decode health: {msg}");
                    }
                    _ => {}
                }
            }

            let resp = match done_resp {
                Some(r) => r,
                None => {
                    if client_gone {
                        // Normal disconnect — no error to report.
                        tracing::info!(
                            conv_id = %conv_id,
                            iteration,
                            "client disconnected mid-stream — cancelling decode",
                        );
                    } else if turn_error.is_none() {
                        // Scheduler closed the channel without Done — the
                        // conversation's turn_in_flight flag is stuck.
                        tracing::error!(
                            conv_id = %conv_id,
                            iteration,
                            "turn ended without Done or Error — evicting conversation",
                        );
                        let _ = tx.blocking_send(Ok(
                            "\n\n⚠ Generation ended unexpectedly. Your next message will start fresh.".to_string()
                        ));
                    }
                    // Evict the conversation so the next request forks fresh
                    // rather than hitting the in-flight guard.  Dropping `handle`
                    // here (via return) closes event_rx and stops the scheduler.
                    drop(cs);
                    state.conversations.lock().unwrap().remove(&conv_id);
                    return;
                }
            };

            // Flush any tail that the byte-fallback guard held back.
            if resp.text.len() > emitted_len && resp.text.is_char_boundary(emitted_len) {
                let tail = &resp.text[emitted_len..];
                if !tail.is_empty() {
                    let _ = tx.blocking_send(Ok(tail.to_string()));
                }
            }

            let calls = extract_tool_calls(&resp.text);
            let is_final = calls.is_empty() || iteration == MAX_TOOL_ITERATIONS;

            if let Err(e) = cs.conv.finish_turn(handle, &resp) {
                tracing::warn!(conv_id = %conv_id, "finish_turn error: {e}");
            }

            // Group-commit the substrate redo log: the just-sealed turn is
            // now durable on disk, so a crash or restart resumes it intact.
            if let Err(e) = state.engine.lock().unwrap().commit_persistence() {
                tracing::warn!(conv_id = %conv_id, "persistence commit error: {e}");
            }

            if is_final {
                break;
            }

            tracing::info!(
                conv_id = %conv_id,
                iteration,
                n_calls = calls.len(),
                "dispatching tool calls",
            );
            if iteration == MAX_TOOL_ITERATIONS - 1 {
                tracing::warn!(
                    conv_id = %conv_id,
                    "tool iteration cap ({MAX_TOOL_ITERATIONS}) reached — \
                     forcing final response on next turn",
                );
            }
            let results = run_tool_calls(&state.tool_host.ctx, calls);
            current_message = format_tool_responses(&results);
        }
    });

    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}

// ── Session ───────────────────────────────────────────────────────────────────

pub struct ZendSession {
    config: DaemonConfig,
    projection_builder: projection::Builder,
    #[allow(dead_code)] // read by api/ws_logs.rs in the bin target
    pub(crate) log: Arc<LogBus>,
    /// Fires `true` once the model finishes loading (success or failure).
    ready_tx: tokio::sync::watch::Sender<bool>,
    /// Current human-readable loading status, published at key milestones.
    pub(crate) status_tx: tokio::sync::watch::Sender<String>,
    /// Populated in the background after construction; None until model loads.
    inference: Arc<std::sync::RwLock<Option<Arc<InferenceState>>>>,
}

impl ZendSession {
    pub fn new(config: DaemonConfig, log: Arc<LogBus>) -> Self {
        let projection_builder = build_projection_builder(&config.workspace);
        tracing::info!(workspace = %config.workspace.display(), "session initialised");
        let (ready_tx, _) = tokio::sync::watch::channel(false);
        let (status_tx, _) = tokio::sync::watch::channel(String::new());
        Self {
            inference: Arc::new(std::sync::RwLock::new(None)),
            config,
            projection_builder,
            log,
            ready_tx,
            status_tx,
        }
    }

    pub fn start_loading(self: &Arc<Self>) {
        let slot = Arc::clone(&self.inference);
        let proj_builder = self.projection_builder.clone();
        let ready_tx = self.ready_tx.clone();
        let status_tx = self.status_tx.clone();
        let workspace = self.config.workspace.clone();
        tokio::spawn(async move {
            status_tx.send("Checking for model…".into()).ok();
            let (model_path, tok_path) = match crate::download::ensure_model(&status_tx).await {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!("model download failed: {e:#}");
                    status_tx.send(format!("Download failed: {e}")).ok();
                    ready_tx.send(true).ok();
                    return;
                }
            };

            status_tx.send("Loading model…".into()).ok();
            tracing::info!("loading inference engine (Qwen3-30B-A3B) …");
            match tokio::task::spawn_blocking(move || {
                InferenceState::load(proj_builder, model_path, tok_path, workspace)
            })
            .await
            {
                Ok(Ok(state)) => {
                    *slot.write().unwrap() = Some(state);
                    tracing::info!("inference engine ready");
                    status_tx.send(String::new()).ok();
                }
                Ok(Err(e)) => {
                    tracing::warn!("inference engine failed to load: {e:#}");
                    status_tx.send(format!("Load failed: {e}")).ok();
                }
                Err(e) => {
                    tracing::warn!("inference engine panicked: {e}");
                    status_tx.send("Model load panicked.".into()).ok();
                }
            }
            ready_tx.send(true).ok();
        });
    }

    /// Graceful shutdown: durably checkpoint the substrate redo log, then
    /// stop the scheduler thread. Idempotent — safe to call when the model
    /// never finished loading (nothing to flush). Runs on the blocking pool
    /// since `checkpoint_persistence` does synchronous `fsync` I/O.
    pub async fn shutdown(&self) {
        let state: Option<Arc<InferenceState>> =
            { self.inference.read().unwrap().as_ref().map(Arc::clone) };
        let Some(state) = state else {
            tracing::info!("shutdown: model never loaded — nothing to persist");
            return;
        };
        let _ = tokio::task::spawn_blocking(move || {
            let engine = state.engine.lock().unwrap();
            match engine.checkpoint_persistence() {
                Ok(()) => tracing::info!("shutdown: substrate checkpointed"),
                Err(e) => tracing::error!("shutdown: checkpoint failed: {e}"),
            }
            if let Err(e) = engine.shutdown() {
                tracing::error!("shutdown: scheduler stop failed: {e}");
            }
        })
        .await;
    }

    /// Submit the latest user message and return a stream of status + token items.
    pub async fn submit(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<usize>,
        conv_id: String,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        self.submit_with_sampling(messages, max_tokens, conv_id, None)
            .await
    }

    /// Same as [`Self::submit`] but accepts an explicit
    /// [`candle_conversation::SamplingConfig`] override.
    ///
    /// Used by tests that want deterministic generation (e.g.
    /// `SamplingConfig::argmax()` for greedy decoding) so the
    /// pass/fail signal isn't subject to top-k/top-p sampling noise.
    pub async fn submit_with_sampling(
        &self,
        messages: Vec<ChatMessage>,
        max_tokens: Option<usize>,
        conv_id: String,
        sampling: Option<candle_conversation::SamplingConfig>,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        let last_user = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let inference = Arc::clone(&self.inference);
        let mut ready_rx = self.ready_tx.subscribe();
        let mut status_rx = self.status_tx.subscribe();

        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<StreamItem>>(64);

        tokio::spawn(async move {
            // Phase 1 — emit status updates while model is loading.
            if inference.read().unwrap().is_none() {
                let cur = status_rx.borrow_and_update().clone();
                if !cur.is_empty() && tx.send(Ok(StreamItem::Status(cur))).await.is_err() {
                    return;
                }

                loop {
                    if *ready_rx.borrow_and_update() {
                        break;
                    }
                    tokio::select! {
                        _ = ready_rx.changed() => { break; }
                        _ = status_rx.changed() => {
                            let msg = status_rx.borrow_and_update().clone();
                            if !msg.is_empty()
                                && tx.send(Ok(StreamItem::Status(msg))).await.is_err()
                            {
                                return;
                            }
                        }
                    }
                }

                if tx
                    .send(Ok(StreamItem::Status("Processing…".into())))
                    .await
                    .is_err()
                {
                    return;
                }
            }

            // Phase 2 — generate tokens.
            let state: Option<Arc<InferenceState>> =
                { inference.read().unwrap().as_ref().map(Arc::clone) };
            if let Some(state) = state {
                let mut ts = run_inference_stream(state, conv_id, last_user, max_tokens, sampling);
                while let Some(item) = ts.next().await {
                    if tx.send(item.map(StreamItem::Token)).await.is_err() {
                        break;
                    }
                }
            } else {
                tx.send(Ok(StreamItem::Status("Model unavailable.".into())))
                    .await
                    .ok();
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// The [`projection::TimelineId`] a client `conv_id` maps to — a stable hash
/// of the id. Deterministic across daemon restarts, so a reconnecting client
/// resolves to the same timeline the substrate reload recovered (§16.12).
fn timeline_for(conv_id: &str) -> projection::TimelineId {
    let h = content_hash::hash_bytes(conv_id.as_bytes());
    projection::TimelineId::from_raw(h.lo.max(1)).expect("timeline id is non-zero")
}

fn build_projection_builder(workspace: &Path) -> projection::Builder {
    let name = workspace
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("this project");
    projection::Builder::from_yaml_with_vars(PROJECTION_SCHEMA_TEMPLATE, &[("workspace", name)])
        .expect("projection.yaml failed to parse — check YAML syntax and {workspace} placeholder")
}

/// Concatenate the content of every top-level Section that appears
/// **before** the dialogue layer's first Collection, returning the
/// joined text.  Used as the engine's `system_prompt:` argument — it
/// gets ChatML-wrapped by the model builder, then handed to the
/// conversation as the pending-prefill prelude.  Everything from the
/// first Collection onward is expanded inside
/// [`Sequence::preemptive_prefill`] (collection sections via
/// fork-and-merge, post-collection sections via per-section prefill).
fn pre_collection_prelude(builder: &projection::Builder) -> String {
    use projection::SystemPromptItem;
    let layer = builder
        .schema()
        .layers
        .iter()
        .find(|l| l.name == "dialogue")
        .expect("projection schema must declare a 'dialogue' layer");

    let mut out = String::new();
    for item in &layer.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => out.push_str(&s.content),
            SystemPromptItem::Collection(_) => break,
        }
    }
    out
}
