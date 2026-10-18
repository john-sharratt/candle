//! zend's chat stream, as the events a client sees: the OpenAI chunks carrying
//! text, zend's own progress events (`prefill`, `think`, `tool`, `status`,
//! `projection`), a failure (`turn_error`), and the `[DONE]` sentinel.
//! Keep-alive comments are not events: nothing reaches the person reading.

use serde_json::Value;

/// One event of the stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Item {
    /// A chunk of the reply: its content, which a later turn sends back as
    /// history, and how much reasoning it carried, which it does not.
    Text { content: String, reasoning: usize },
    /// A progress event the client shows: its name.
    Progress(String),
    /// Tools the turn started running, by name.
    Tools(Vec<String>),
    /// The turn failed, with the engine's message.
    Failed(String),
    /// The stream is complete.
    Done,
}

/// Splits a byte stream into SSE events as bytes arrive.
#[derive(Default)]
pub struct Parser {
    pending: String,
    event: Option<String>,
    data: Vec<String>,
}

impl Parser {
    /// Feed bytes; returns the events they completed.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<Item> {
        self.pending.push_str(&String::from_utf8_lossy(bytes));
        let mut out = Vec::new();
        while let Some(end) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=end).collect();
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if let Some(item) = self.dispatch() {
                    out.push(item);
                }
            } else if let Some(name) = line.strip_prefix("event:") {
                self.event = Some(name.trim().to_string());
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data.push(data.trim_start().to_string());
            }
        }
        out
    }

    /// The event the blank line just ended, if it was one.
    fn dispatch(&mut self) -> Option<Item> {
        let event = self.event.take();
        let data = std::mem::take(&mut self.data).join("\n");
        if event.is_none() && data.is_empty() {
            return None;
        }
        match event.as_deref() {
            Some("turn_error") => Some(Item::Failed(message(&data))),
            Some("tool") => Some(tools(&data)),
            Some(name) => Some(Item::Progress(name.to_string())),
            None if data == "[DONE]" => Some(Item::Done),
            None => Some(text(&data)),
        }
    }
}

/// The text a chunk carries: its delta's content and reasoning.
fn text(data: &str) -> Item {
    let v = serde_json::from_str::<Value>(data).unwrap_or(Value::Null);
    let delta = &v["choices"][0]["delta"];
    Item::Text {
        content: delta["content"].as_str().unwrap_or("").to_string(),
        reasoning: delta["reasoning_content"].as_str().map_or(0, str::len),
    }
}

impl Item {
    /// Whether the event carried any text at all.
    pub fn has_text(&self) -> bool {
        matches!(self, Item::Text { content, reasoning } if !content.is_empty() || *reasoning > 0)
    }
}

/// A `tool` event: the tools it starts when its phase is `running`; a `done`
/// event names the same tools again and starts none.
fn tools(data: &str) -> Item {
    let v = serde_json::from_str::<Value>(data).unwrap_or(Value::Null);
    if v["phase"] != "running" {
        return Item::Progress("tool".into());
    }
    Item::Tools(
        v["tools"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|t| t.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
    )
}

fn message(data: &str) -> String {
    serde_json::from_str::<Value>(data)
        .ok()
        .and_then(|v| v["message"].as_str().map(String::from))
        .unwrap_or_else(|| data.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Events split across reads come out whole, each kind recognised; a
    /// keep-alive comment is nothing.
    #[test]
    fn events_are_read_across_chunk_boundaries() {
        let mut p = Parser::default();
        let mut got =
            p.push(b"event: prefill\ndata: {\"done\":0,\"total\":9}\n\n: keep-alive\n\nda");
        got.extend(p.push(b"ta: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"));
        got.extend(p.push(
            b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"why\",\"content\":\"lo\"}}]}\r\n\r\n",
        ));
        got.extend(p.push(b"event: turn_error\ndata: {\"message\":\"pool stalled\"}\n\n"));
        got.extend(p.push(b"data: [DONE]\n\n"));
        assert_eq!(
            got,
            vec![
                Item::Progress("prefill".into()),
                Item::Text {
                    content: "Hel".into(),
                    reasoning: 0,
                },
                Item::Text {
                    content: "lo".into(),
                    reasoning: 3,
                },
                Item::Failed("pool stalled".into()),
                Item::Done,
            ]
        );
    }

    /// A running `tool` event names the tools it starts; its `done` twin is
    /// only progress.
    #[test]
    fn a_running_tool_event_names_its_tools() {
        let mut p = Parser::default();
        let got = p.push(
            b"event: tool\ndata: {\"phase\":\"running\",\"tools\":[\"file_read\",\"file_edit\"]}\n\n\
              event: tool\ndata: {\"phase\":\"done\",\"tools\":[\"file_read\"],\"results\":[]}\n\n",
        );
        assert_eq!(
            got,
            vec![
                Item::Tools(vec!["file_read".into(), "file_edit".into()]),
                Item::Progress("tool".into()),
            ]
        );
    }

    /// A stop chunk with an empty delta is a text event that carries no text.
    #[test]
    fn an_empty_delta_carries_no_text() {
        let mut p = Parser::default();
        let got = p.push(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
        assert_eq!(
            got,
            vec![Item::Text {
                content: String::new(),
                reasoning: 0,
            }]
        );
        assert!(!got[0].has_text());
    }
}
