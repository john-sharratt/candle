use serde::{Deserialize, Serialize};

// ── Roles ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

// ── Request ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
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
    fn parses_with_composer_dials() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[],"effort":0,"verbosity":4,"think":false}"#,
        )
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
#[derive(Debug, Serialize)]
pub struct Delta {
    /// Present only in the first frame (signals role to the client).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// Token text. `None` in the final stop frame.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
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
}
