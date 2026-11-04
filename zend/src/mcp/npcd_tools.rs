//! The bots' and the pictures' tools for a conversation of Zen's
//! (`docs/zen_tool_reach.md` §4.3, §6.4): npcd's own MCP tools, called
//! through the gateway as the conversation's person, exactly as the
//! aggregator calls them for Claude (`docs/mcp_app.md` §3). npcd is their only
//! implementation, and its own role checks stand: a bot the person does not
//! own is one npcd does not show them.
//!
//! The call goes up this member's cluster socket to `/v1/npc/mcp`, in the
//! 2026-07-28 protocol, which keeps no session to open. Its answer is one
//! JSON-RPC response or an event stream ending in one; its text blocks are the
//! answer, and a picture is answered by the link its text carries.

use std::time::Duration;

use axum::http::{header, HeaderName, HeaderValue, Method};
use mcp_server::http::VERSION_HEADER;
use mcp_server::protocol::{CLIENT_META, MODERN, SESSION_META};
use serde_json::{json, Value};

use super::ops;
use super::ops_tools::OpsCall;
use super::tools::Answer;

/// npcd's MCP server.
pub const NPC_MCP: &str = "/v1/npc/mcp";

/// The tools that read a bot or a world, and the pictures': any member's.
pub const READS: [&str; 8] = [
    "bot_list",
    "bot_mission_read",
    "bot_message_read",
    "bot_channel_read",
    "bot_discord_read",
    "bot_thread_read",
    "image_generate",
    "image_edit",
];

/// The tools that manage a bot: a conversation that may change things.
pub const ACTS: [&str; 5] = [
    "bot_mission_give",
    "bot_mission_step",
    "bot_mission_cancel",
    "bot_message_send",
    "bot_channel_post",
];

/// The acts that send text into rooms other people read: from a turn that
/// read content from outside, held for the person's approval.
pub const SAYS: [&str; 2] = ["bot_message_send", "bot_channel_post"];

/// How long a picture may take to draw.
const ANSWER_WITHIN: Duration = Duration::from_secs(240);

/// The `tools/call` request for `name` with `args`, on behalf of the
/// conversation `lineage`.
pub fn request(name: &str, args: &Value, lineage: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": args,
            "_meta": {CLIENT_META: {"name": "zen"}, SESSION_META: lineage},
        }
    })
}

/// The JSON-RPC message an answer's `body` ends in: the body itself, or the
/// last `data:` of an event stream that carries a result or an error.
pub fn message_of(body: &[u8]) -> Option<Value> {
    let text = String::from_utf8_lossy(body);
    if let Ok(v) = serde_json::from_str::<Value>(text.trim()) {
        return Some(v);
    }
    text.lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok())
        .filter(|m| m.get("result").is_some() || m.get("error").is_some())
        .last()
}

/// A `tools/call` answer as Zen reads it: its text blocks, a picture standing
/// for itself by the link its text gives.
pub fn answer_of(message: &Value) -> Answer {
    if let Some(e) = message.get("error") {
        let said = e
            .get("message")
            .and_then(Value::as_str)
            .map_or_else(|| e.to_string(), str::to_string);
        return Answer::error(format!("npcd refused: {said}"));
    }
    let result = message.get("result").cloned().unwrap_or(Value::Null);
    let text: Vec<&str> = result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect();
    let said = text.join("\n");
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        Answer::error(said)
    } else {
        Answer::ok(said)
    }
}

/// Run npcd's tool `name` with `args` for `call`.
pub async fn call(call: &OpsCall, name: &str, args: &Value) -> Answer {
    let mut headers = call.headers();
    for (k, v) in [
        (VERSION_HEADER, MODERN),
        (
            header::ACCEPT.as_str(),
            "application/json, text/event-stream",
        ),
        (header::CONTENT_TYPE.as_str(), "application/json"),
    ] {
        if let (Ok(k), Ok(v)) = (
            HeaderName::from_bytes(k.as_bytes()),
            HeaderValue::from_str(v),
        ) {
            headers.push((k, v));
        }
    }
    let body = request(name, args, &call.lineage);
    let reply = match ops::up_as(
        &call.session,
        headers,
        Method::POST,
        NPC_MCP,
        None,
        Some(&body),
        ANSWER_WITHIN,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(format!("npcd could not be reached: {e}")),
    };
    match message_of(&reply.body) {
        Some(m) => answer_of(&m),
        None => Answer::error(format!(
            "npcd answered {} with no result: {}",
            reply.status,
            ops::why(&reply)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A call is one `tools/call`, naming its client and the conversation
    /// it is on behalf of**, as the aggregator's are.
    #[test]
    fn a_call_is_a_tools_call_on_behalf_of_its_conversation() {
        let r = request("bot_list", &json!({"state": "active"}), "c1");
        assert_eq!(r["method"], "tools/call");
        assert_eq!(r["params"]["name"], "bot_list");
        assert_eq!(r["params"]["arguments"], json!({"state": "active"}));
        assert_eq!(r["params"]["_meta"][CLIENT_META]["name"], "zen");
        assert_eq!(r["params"]["_meta"][SESSION_META], "c1");
    }

    /// **The answer is read from a body or from an event stream's last
    /// result**, its text blocks joined and a picture left to its link.
    #[test]
    fn an_answer_is_read_from_a_body_or_a_stream() {
        let result = json!({"jsonrpc": "2.0", "id": 1, "result": {"isError": false, "content": [
            {"type": "text", "text": "Drew it: https://bot.tokera.com/v1/image/ab.jpg"},
            {"type": "image", "data": "iVBO", "mimeType": "image/png"}
        ]}});
        let a = answer_of(&message_of(result.to_string().as_bytes()).unwrap());
        assert_eq!(
            a,
            Answer::ok("Drew it: https://bot.tokera.com/v1/image/ab.jpg")
        );
        let stream = format!(
            "event: message\ndata: {}\n\nevent: message\ndata: {}\n\n",
            json!({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"message": "drawing"}}),
            json!({"jsonrpc": "2.0", "id": 1, "result": {"isError": true, "content": [
                {"type": "text", "text": "npc_not_found"}]}}),
        );
        assert_eq!(
            answer_of(&message_of(stream.as_bytes()).unwrap()),
            Answer::error("npc_not_found")
        );
        let refused = json!({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "no such tool"}});
        assert_eq!(
            answer_of(&refused),
            Answer::error("npcd refused: no such tool")
        );
        assert!(message_of(b"<html>").is_none());
    }
}
