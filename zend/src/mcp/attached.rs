//! The thread each conversation is attached to (`thread_attach`), which
//! `"self"` names: a record per account and conversation,
//! `mcp/attached/<owner>/<conversation>`, so it follows the conversation's
//! ref whatever MCP sessions carry its calls (`docs/private_mode.md` §3) —
//! Grok opens one for every call, claude.ai carries every chat on one. The
//! tend ceases one whose conversation has been idle past its ref's hours.

use mcp_server::refs::REF_IDLE_MS;
use serde_json::{json, Value};

use super::records::Records;

const PREFIX: &str = "mcp/attached/";

fn key(owner: &str, conversation: &str) -> String {
    format!("{PREFIX}{owner}/{conversation}")
}

/// Attach `owner`'s `conversation` to `thread` at `now_ms`.
pub fn attach(
    records: &dyn Records,
    owner: &str,
    conversation: &str,
    thread: &str,
    now_ms: u64,
) -> anyhow::Result<()> {
    records.put(
        &key(owner, conversation),
        json!({"thread": thread, "at_ms": now_ms}),
    )
}

/// The thread `owner`'s `conversation` is attached to.
pub fn attached(records: &dyn Records, owner: &str, conversation: &str) -> Option<String> {
    let v = records.get(&key(owner, conversation))?;
    v.get("thread").and_then(Value::as_str).map(str::to_string)
}

/// Cease every attachment whose conversation has been idle past
/// [`REF_IDLE_MS`] at `now_ms` — since it was made, or since `used` says its
/// ref was last used; how many were ceased.
pub fn sweep(records: &dyn Records, now_ms: u64, used: impl Fn(&str) -> Option<u64>) -> usize {
    let mut ceased = 0;
    for (k, v) in records.under(PREFIX) {
        let conversation = k.rsplit('/').next().unwrap_or_default();
        let at = v.get("at_ms").and_then(Value::as_u64).unwrap_or(0);
        let last = used(conversation).unwrap_or(0).max(at);
        if now_ms.saturating_sub(last) >= REF_IDLE_MS && records.cease(&k).is_ok() {
            ceased += 1;
        }
    }
    ceased
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    #[test]
    fn each_conversation_is_attached_apart() {
        let r = MemRecords::default();
        attach(&r, "o", "c1", "review", 10).unwrap();
        attach(&r, "o", "c2", "bc-wearables", 10).unwrap();
        assert_eq!(attached(&r, "o", "c1").as_deref(), Some("review"));
        assert_eq!(attached(&r, "o", "c2").as_deref(), Some("bc-wearables"));
        assert_eq!(attached(&r, "other", "c1"), None, "another account's");
        attach(&r, "o", "c1", "layers", 11).unwrap();
        assert_eq!(attached(&r, "o", "c1").as_deref(), Some("layers"));
    }

    #[test]
    fn an_attachment_goes_with_its_idle_conversation() {
        let r = MemRecords::default();
        attach(&r, "o", "old", "a", 0).unwrap();
        attach(&r, "o", "kept", "b", 0).unwrap();
        let now = REF_IDLE_MS + 5;
        let used = |c: &str| (c == "kept").then_some(now - 1);
        assert_eq!(sweep(&r, now, used), 1);
        assert_eq!(attached(&r, "o", "old"), None);
        assert_eq!(attached(&r, "o", "kept").as_deref(), Some("b"));
    }
}
