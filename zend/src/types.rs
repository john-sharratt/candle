use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

// ── Roles ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// A tool result sent back by a client that runs its own tools. The
    /// passthrough folds it into the next user half; the daemon's own
    /// conversations read only the latest user message.
    Tool,
}

// ── Request ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default, deserialize_with = "text_content")]
    pub content: String,
    /// The calls an assistant message made — a client that runs its own tools.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<MessageToolCall>,
    /// The call a `tool` message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// A message that carries text alone.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// One entry of an assistant message's `tool_calls`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageToolCall {
    #[serde(default)]
    pub id: String,
    pub function: FunctionCall,
}

/// A call's function and arguments. OpenAI sends `arguments` as the object
/// serialized to a string; a client that sends the object itself is read too.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

/// An OpenAI message `content`, as text: a string; `null` (an assistant message
/// that carries only tool calls); or an array of typed parts, whose `text` parts
/// are joined in order. A part with no text (an image) carries nothing a text
/// model reads and is dropped.
fn text_content<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Content {
        Text(String),
        Parts(Vec<Part>),
    }
    #[derive(Deserialize)]
    struct Part {
        #[serde(default)]
        text: Option<String>,
    }
    Ok(match Option::<Content>::deserialize(d)? {
        None => String::new(),
        Some(Content::Text(text)) => text,
        Some(Content::Parts(parts)) => parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join("\n\n"),
    })
}

/// OpenAI-compatible `POST /v1/chat/completions` request body.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    /// `true` → SSE stream (default); `false` → single JSON response.
    #[serde(default)]
    pub stream: bool,
    #[allow(dead_code)]
    pub max_tokens: Option<u32>,
    #[allow(dead_code)]
    pub temperature: Option<f32>,
    #[allow(dead_code)]
    pub top_p: Option<f32>,
    /// Stable identifier for the conversation tab.
    /// When absent, all requests share a single default conversation.
    #[serde(default)]
    pub conv_id: Option<String>,
    /// Capture aid (zend-only): name of a section collection (e.g. `"tools"`)
    /// to force to full resolution for this conversation — projection and
    /// reprojection stop filtering it, so all its sections stay materialised.
    /// Used to build tool-invocation training data; absent in normal use.
    #[serde(default)]
    pub force_high_resolution: Option<String>,
    /// Capture aid (zend-only): text to prefill as the start of the assistant's
    /// response (e.g. `"<tool_call>"`), forcing the decode to continue from it.
    /// Used to capture clean tool-call exemplars regardless of whether the model
    /// would otherwise refuse, narrate, or fabricate a result; absent in normal use.
    #[serde(default)]
    pub assistant_prefill: Option<String>,
    /// Capture aid (zend-only): when true, this conversation's turns are sealed
    /// **without** KV quantization — K/V persist in native R16/F16 (lossless), so
    /// the provenance work has full-resolution keys. Absent/false in normal use.
    #[serde(default)]
    pub lossless_kv: bool,
    /// Composer "thinking effort" dial (0..=4). `0` (and `think: false`) route to
    /// the `/no_think` dialect prefix; `1..=4` select the reasoning-depth
    /// directive section. Absent → server default. Consumed by the projection
    /// request in the chat path.
    #[allow(dead_code)]
    pub effort: Option<u8>,
    /// Composer "answer length" dial (0..=4). Selects the answer-length
    /// directive section. Absent → server default.
    #[allow(dead_code)]
    pub verbosity: Option<u8>,
    /// Explicit thinking on/off. `false` routes to `/no_think`. Absent → on
    /// unless `effort == 0`.
    #[allow(dead_code)]
    pub think: Option<bool>,
    /// Either the composer "tools" dial or an OpenAI `tools` array — see
    /// [`RequestTools`].
    #[serde(default)]
    pub tools: Option<RequestTools>,
    /// Which identity this conversation speaks as — a sub-folder of
    /// `identities/` (e.g. `"keeper"`). Selects the anchor + facets the system
    /// prompt scopes to. Absent → the conversation's stored identity, else the
    /// `mind.yaml` default. Persisted on the conversation the first time it is
    /// set, so later turns need not repeat it.
    #[serde(default)]
    pub identity: Option<String>,
}

/// Which slice of the tool catalog a conversation projects. Maps to the GUI
/// "tools" dial. Drives both projection (which tool sections materialise) and
/// which tool summary is injected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolMode {
    /// No tools: every tool section and the tool summary are omitted.
    None,
    /// Safe tools only: high-risk tool sections are omitted; the restricted
    /// summary is injected.
    Restricted,
    /// Every tool; the comprehensive summary is injected.
    #[default]
    Comprehensive,
}

/// The request's `tools` field, in either shape it arrives in.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RequestTools {
    /// The composer "tools" dial. Selects which tool sections the projection
    /// materialises for this conversation: `None` omits every tool section,
    /// `Restricted` omits the high-risk tools (and uses the restricted tool
    /// summary), `Comprehensive` keeps the full catalog. Absent → server
    /// default (`Comprehensive`).
    Mode(ToolMode),
    /// OpenAI function definitions, each `{"type": "function", "function": {…}}`
    /// — a client that runs its own tools. The passthrough offers them to the
    /// model; the daemon's own conversations set no dial from them.
    Functions(Vec<Value>),
}

impl RequestTools {
    /// The dial, when this is one.
    pub fn mode(&self) -> Option<ToolMode> {
        match self {
            Self::Mode(mode) => Some(*mode),
            Self::Functions(_) => None,
        }
    }
}

#[cfg(test)]
mod request_tests {
    use super::*;

    #[test]
    fn parses_without_composer_dials() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[],"conv_id":"abc"}"#).unwrap();
        assert_eq!(req.effort, None);
        assert_eq!(req.verbosity, None);
        assert_eq!(req.think, None);
        assert_eq!(req.conv_id.as_deref(), Some("abc"));
    }

    #[test]
    fn content_is_a_string_an_array_of_parts_or_null() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"passthrough","stream":true,"messages":[
                {"role":"system","content":"sys"},
                {"role":"user","content":[
                    {"type":"text","text":"task"},
                    {"type":"image_url","image_url":{"url":"data:x"}},
                    {"type":"text","text":"env"}]},
                {"role":"assistant","content":null,"tool_calls":[]},
                {"role":"tool","tool_call_id":"t1","content":"result"}],
              "stream_options":{"include_usage":true}}"#,
        )
        .unwrap();
        let texts: Vec<&str> = req.messages.iter().map(|m| m.content.as_str()).collect();
        assert_eq!(texts, ["sys", "task\n\nenv", "", "result"]);
        assert_eq!(req.messages[3].role, Role::Tool);
    }

    #[test]
    fn tools_is_the_dial_or_a_clients_function_catalog() {
        let dial: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[],"tools":"restricted"}"#).unwrap();
        assert_eq!(
            dial.tools.as_ref().and_then(RequestTools::mode),
            Some(ToolMode::Restricted)
        );
        let catalog: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[],"tools":[{"type":"function","function":{
                "name":"read_file","description":"Read a file",
                "parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
              "tool_choice":"auto","parallel_tool_calls":false}"#,
        )
        .unwrap();
        match &catalog.tools {
            Some(RequestTools::Functions(tools)) => {
                assert_eq!(tools[0]["function"]["name"], "read_file");
            }
            other => panic!("an array of definitions is the client's catalog: {other:?}"),
        }
    }

    #[test]
    fn parses_with_composer_dials() {
        let req: ChatCompletionRequest =
            serde_json::from_str(r#"{"messages":[],"effort":0,"verbosity":4,"think":false}"#)
                .unwrap();
        assert_eq!(req.effort, Some(0));
        assert_eq!(req.verbosity, Some(4));
        assert_eq!(req.think, Some(false));
    }
}

// ── Streaming response (SSE) ──────────────────────────────────────────────────

/// One SSE data frame in the OpenAI streaming format.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    /// Unix timestamp (seconds) at which the completion was created.
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
}

/// Incremental content in one SSE frame.
/// Fields with `None` are omitted so the delta is minimal.
#[derive(Debug, Default, Serialize)]
pub struct Delta {
    /// Present only in the first frame (signals role to the client).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// Token text. `None` in the final stop frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Reasoning text, which an OpenAI-compatible client renders apart from
    /// the answer. Set on passthrough replies only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls, for a client that runs its own tools. Passthrough only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

/// One tool call in a streamed delta. OpenAI lets a call's arguments arrive
/// over several frames under one `index`; each goes out whole, in one frame.
#[derive(Debug, Serialize)]
pub struct DeltaToolCall {
    pub index: u32,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ResponseFunction,
}

/// A tool call's function, in OpenAI's wire form.
#[derive(Debug, Serialize)]
pub struct ResponseFunction {
    pub name: String,
    /// The arguments object, serialized.
    pub arguments: String,
}

// ── Non-streaming response ────────────────────────────────────────────────────

/// Complete (non-streaming) chat completion.
#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: String,
    /// See [`Delta::reasoning_content`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// See [`Delta::tool_calls`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

/// One tool call in a complete response.
#[derive(Debug, Serialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ResponseFunction,
}
