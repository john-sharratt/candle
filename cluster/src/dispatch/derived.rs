//! Keys computed from a buffered request body (§5.2, `Derived`). Implemented
//! here, once, because the gateway routes by them and the member serving the
//! request must reach the same key.
//!
//! **`passthrough`**: an OpenAI passthrough conversation (`model:
//! "passthrough"`) carries no id; it is named by its opening — the system
//! messages, the client's tools, and every message before the first assistant
//! reply. That is the same on every call of one conversation, whatever has been
//! said since.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// A named key function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Derived {
    Passthrough,
}

impl Derived {
    pub fn name(self) -> &'static str {
        match self {
            Derived::Passthrough => "passthrough",
        }
    }

    pub fn from_name(name: &str) -> Option<Derived> {
        match name {
            "passthrough" => Some(Derived::Passthrough),
            _ => None,
        }
    }

    /// The key `body` yields, if it yields one.
    pub fn key(self, body: &Value) -> Option<String> {
        match self {
            Derived::Passthrough => passthrough_key(body),
        }
    }
}

/// The `model` a passthrough request names.
pub const PASSTHROUGH_MODEL: &str = "passthrough";

/// An OpenAI message `content`, as text: a string; `null`; or an array of
/// typed parts, whose `text` parts are joined in order with a blank line. A
/// part with no text (an image) carries nothing a text model reads.
pub fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

/// A conversation's identity from its opening: hex of the first 16 bytes of
/// the SHA-256 over each part, length-prefixed so no boundary can shift —
/// the system texts, the tools' JSON, then each opening message's role and
/// text.
pub fn conversation_key(system: &[&str], tools: &[Value], opening: &[(&str, &str)]) -> String {
    let mut h = Sha256::new();
    let mut part = |bytes: &[u8]| {
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    };
    part(&(system.len() as u64).to_le_bytes());
    for s in system {
        part(s.as_bytes());
    }
    part(&(tools.len() as u64).to_le_bytes());
    for t in tools {
        part(t.to_string().as_bytes());
    }
    part(&(opening.len() as u64).to_le_bytes());
    for (role, text) in opening {
        part(role.as_bytes());
        part(text.as_bytes());
    }
    h.finalize()[..16]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The key of a passthrough request body; `None` for any other request.
pub fn passthrough_key(body: &Value) -> Option<String> {
    if body.get("model").and_then(Value::as_str) != Some(PASSTHROUGH_MODEL) {
        return None;
    }
    let messages = body.get("messages")?.as_array()?;
    let tools: &[Value] = body
        .get("tools")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice);
    // System messages form the prompt wherever they sit; the opening is every
    // other message before the first reply.
    let mut system = Vec::new();
    let mut opening = Vec::new();
    let mut replied = false;
    for m in messages {
        let role = m.get("role").and_then(Value::as_str)?;
        let text = content_text(m.get("content").unwrap_or(&Value::Null));
        match role {
            "system" => system.push(text),
            "assistant" => replied = true,
            _ if !replied => opening.push((role, text)),
            _ => {}
        }
    }
    let system: Vec<&str> = system.iter().map(String::as_str).collect();
    let opening: Vec<(&str, &str)> = opening.iter().map(|(r, t)| (*r, t.as_str())).collect();
    Some(conversation_key(&system, tools, &opening))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn content_is_a_string_parts_or_nothing() {
        assert_eq!(content_text(&json!("sys")), "sys");
        assert_eq!(
            content_text(&json!([
                {"type": "text", "text": "task"},
                {"type": "image_url", "image_url": {"url": "data:x"}},
                {"type": "text", "text": "env"}
            ])),
            "task\n\nenv"
        );
        assert_eq!(content_text(&Value::Null), "");
    }

    /// The key for a fixed opening is a fixed value, computed independently
    /// of this crate (.NET's SHA-256 over the documented framing).
    #[test]
    fn a_fixed_opening_has_a_fixed_key() {
        assert_eq!(
            conversation_key(&["sys"], &[], &[("user", "task")]),
            FIXED_KEY
        );
    }

    const FIXED_KEY: &str = "6078c7d9d2e19a5e71cd77652ec4fa2d";

    /// Every later call of one conversation — the whole history resent, a reply
    /// and more messages after it — names the same key; a different opening, a
    /// different prompt or different tools do not.
    #[test]
    fn every_call_of_a_conversation_names_one_key() {
        let first = json!({"model": "passthrough", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"}
        ]});
        let later = json!({"model": "passthrough", "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "more"}
        ]});
        let key = passthrough_key(&first).unwrap();
        assert_eq!(passthrough_key(&later).unwrap(), key);
        assert_eq!(key, conversation_key(&["sys"], &[], &[("user", "task")]));

        let other_task = json!({"model": "passthrough", "messages": [
            {"role": "system", "content": "sys"}, {"role": "user", "content": "other"}]});
        assert_ne!(passthrough_key(&other_task).unwrap(), key);
        let with_tools = json!({"model": "passthrough",
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]});
        assert_ne!(passthrough_key(&with_tools).unwrap(), key);
        // The length prefixes fix every boundary.
        assert_ne!(
            conversation_key(&["ab"], &[], &[("user", "c")]),
            conversation_key(&["a"], &[], &[("user", "bc")])
        );
    }

    #[test]
    fn only_a_passthrough_request_has_one() {
        let own = json!({"model": "zen-code", "messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(passthrough_key(&own), None);
        assert_eq!(passthrough_key(&json!({"model": "passthrough"})), None);
    }
}
