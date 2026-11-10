//! A conversation's turn, as the chat route streamed it back
//! (`docs/zen_chat.md` §6): what it said, whether it failed, and whether it
//! ended.
//!
//! The body is the route's server-sent events — OpenAI chunks carrying the
//! reply's text, a named `turn_error` for a failure, `data: [DONE]` at the end —
//! read only as far as it had come when the wait ran out. A frame not yet
//! ended by its blank line is not read at all, so a body cut mid-frame never
//! yields half a chunk.

use serde_json::Value;

/// What a turn's stream said so far.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Heard {
    /// The reply's text: reasoning, calls and prose, as the model wrote them.
    pub text: String,
    /// The failure the turn ended in, when it failed.
    pub error: Option<String>,
    /// Whether the stream said it was done.
    pub done: bool,
}

/// What the stream `body` said, frame by frame.
pub fn heard(body: &[u8]) -> Heard {
    let body = String::from_utf8_lossy(body);
    let mut out = Heard::default();
    let mut frames: Vec<&str> = body.split("\n\n").collect();
    // The last piece is a frame its blank line has not ended yet, or nothing.
    frames.pop();
    for frame in frames {
        let mut event = None;
        let mut data = String::new();
        for line in frame.lines() {
            if let Some(name) = line.strip_prefix("event:") {
                event = Some(name.trim());
            } else if let Some(d) = line.strip_prefix("data:") {
                data.push_str(d.trim_start());
            }
        }
        match event {
            Some("turn_error") => {
                let message = serde_json::from_str::<Value>(&data)
                    .ok()
                    .and_then(|v| v.get("message").and_then(Value::as_str).map(str::to_string))
                    .unwrap_or(data);
                out.error.get_or_insert(message);
            }
            Some(_) => {}
            None if data == "[DONE]" => out.done = true,
            None => {
                if let Some(text) = serde_json::from_str::<Value>(&data).ok().and_then(|v| {
                    v.pointer("/choices/0/delta/content")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                }) {
                    out.text.push_str(&text);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(text: &str) -> String {
        format!(
            "data: {}\n\n",
            serde_json::json!({"choices": [{"index": 0, "delta": {"content": text}}]})
        )
    }

    #[test]
    fn a_finished_turn_is_its_text_and_done() {
        let body = format!(
            "{}event: status\ndata: {{\"status\":\"thinking\"}}\n\n{}data: [DONE]\n\n",
            chunk("Hel"),
            chunk("lo")
        );
        assert_eq!(
            heard(body.as_bytes()),
            Heard {
                text: "Hello".into(),
                error: None,
                done: true
            }
        );
    }

    #[test]
    fn a_failed_turn_says_why_after_what_it_wrote() {
        let body = format!(
            "{}event: turn_error\ndata: {{\"message\":\"the turn's channel closed\"}}\n\ndata: [DONE]\n\n",
            chunk("Part")
        );
        assert_eq!(
            heard(body.as_bytes()),
            Heard {
                text: "Part".into(),
                error: Some("the turn's channel closed".into()),
                done: true
            }
        );
    }

    /// **A body cut at the deadline is read to its last whole frame**, and is
    /// not done.
    #[test]
    fn a_cut_body_is_read_to_its_last_whole_frame() {
        let whole = chunk("So far");
        let body = format!("{whole}data: {{\"choices\":[{{\"delta\":{{\"content\":\"nev");
        assert_eq!(
            heard(body.as_bytes()),
            Heard {
                text: "So far".into(),
                error: None,
                done: false
            }
        );
    }
}
