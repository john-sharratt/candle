//! A chat and its messages, as records (`docs/zen_chat.md` §2).
//!
//! - **The chat** is `mcp/chat/<owner>/<id>`: when it opened, when a message
//!   was last said on it, and whether it is closing.
//! - **Its messages** are one record each under `mcp/chat-said/<owner>/<id>/`,
//!   keyed by time and a sequence, so key order is the order they were said and
//!   two said in one millisecond never race to rewrite one record. A message's
//!   number is its place in that order, from 1.
//!
//! Every key starts with the owner's key, so one person's chats are never
//! another's.

use std::sync::atomic::{AtomicU64, Ordering};

use journals::scope::Channel;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::mcp::records::Records;

/// Orders the messages one process keeps within one millisecond.
static SAID_SEQ: AtomicU64 = AtomicU64::new(0);

/// What every chat's record key starts with.
const CHAT_PREFIX: &str = "mcp/chat/";

/// A chat's record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chat {
    pub id: String,
    pub opened_ms: u64,
    /// When the last message was said on it: what its idle time counts from.
    pub last_ms: u64,
    /// It is being disposed of: closed by its caller, or found due by the
    /// sweep while Zen was answering.
    #[serde(default)]
    pub closing: bool,
}

/// Who said a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Speaker {
    /// The MCP client that sent it, whichever that was.
    Caller,
    Zen,
}

/// One message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Said {
    pub at_ms: u64,
    pub by: Speaker,
    pub text: String,
    /// It is the failure Zen's turn ended in, said as Zen's message.
    #[serde(default)]
    pub failed: bool,
    /// The client the caller's message came through, as
    /// `mcp_server::Client::name` names it; none on Zen's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
}

impl Said {
    /// How a message's line names who said it: Zen, or the caller by the
    /// client it came through — "Caller" when that names no known app.
    pub fn speaker(&self) -> &'static str {
        match self.by {
            Speaker::Zen => "Zen",
            Speaker::Caller => match self.client.as_deref().and_then(Channel::parse) {
                Some(Channel::Other) | None => "Caller",
                Some(c) => c.spoken(),
            },
        }
    }
}

pub fn chat_key(owner: &str, id: &str) -> String {
    format!("{CHAT_PREFIX}{owner}/{id}")
}

pub fn said_prefix(owner: &str, id: &str) -> String {
    format!("mcp/chat-said/{owner}/{id}/")
}

fn decode<T: for<'de> Deserialize<'de>>(value: Value) -> Option<T> {
    serde_json::from_value(value).ok()
}

pub fn chat(records: &dyn Records, owner: &str, id: &str) -> Option<Chat> {
    records.get(&chat_key(owner, id)).and_then(decode)
}

pub fn put_chat(records: &dyn Records, owner: &str, chat: &Chat) -> anyhow::Result<()> {
    records.put(&chat_key(owner, &chat.id), serde_json::to_value(chat)?)
}

/// How many chats `owner` has kept, closing ones among them: a range of keys,
/// read without decoding any.
pub fn kept_count(records: &dyn Records, owner: &str) -> usize {
    records.count_under(&format!("{CHAT_PREFIX}{owner}/"))
}

/// Every chat of every owner, each with the owner key it is kept under. A
/// record that does not read as a chat is left out.
pub fn every_chat(records: &dyn Records) -> Vec<(String, Chat)> {
    records
        .under(CHAT_PREFIX)
        .into_iter()
        .filter_map(|(key, v)| {
            let (owner, _) = key.strip_prefix(CHAT_PREFIX)?.split_once('/')?;
            Some((owner.to_string(), decode(v)?))
        })
        .collect()
}

/// Keep `said` as the chat's next message.
pub fn say(records: &dyn Records, owner: &str, id: &str, said: &Said) -> anyhow::Result<()> {
    // Time first, zero-padded, so key order is the order they were said; the
    // sequence after it keeps two of one millisecond in order.
    let seq = SAID_SEQ.fetch_add(1, Ordering::Relaxed);
    let key = format!("{}{:016}-{seq:010}", said_prefix(owner, id), said.at_ms);
    records.put(&key, serde_json::to_value(said)?)
}

/// The chat's messages, oldest first, each with its record key.
pub fn said_keyed(records: &dyn Records, owner: &str, id: &str) -> Vec<(String, Said)> {
    records
        .under(&said_prefix(owner, id))
        .into_iter()
        .filter_map(|(k, v)| Some((k, decode(v)?)))
        .collect()
}

/// The chat's messages, oldest first.
pub fn said(records: &dyn Records, owner: &str, id: &str) -> Vec<Said> {
    said_keyed(records, owner, id)
        .into_iter()
        .map(|(_, s)| s)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::records::MemRecords;

    fn said_at(at_ms: u64, by: Speaker, text: &str) -> Said {
        Said {
            at_ms,
            by,
            text: text.into(),
            failed: false,
            client: None,
        }
    }

    #[test]
    fn a_chat_is_kept_under_its_owner_and_reads_back_as_written() {
        let r = MemRecords::default();
        let c = Chat {
            id: "0123456789abcdef".into(),
            opened_ms: 10,
            last_ms: 10,
            closing: false,
        };
        assert_eq!(
            serde_json::to_string(&c).unwrap(),
            r#"{"id":"0123456789abcdef","opened_ms":10,"last_ms":10,"closing":false}"#
        );
        put_chat(&r, "o", &c).unwrap();
        assert_eq!(
            r.under("mcp/").first().map(|(k, _)| k.as_str()),
            Some("mcp/chat/o/0123456789abcdef")
        );
        assert_eq!(chat(&r, "o", &c.id), Some(c.clone()));
        assert_eq!(chat(&r, "p", &c.id), None, "owners never meet");
        assert_eq!(every_chat(&r), vec![("o".to_string(), c)]);
    }

    /// **Messages read back in the order they were said**, two of one
    /// millisecond kept apart, and none of them is taken for a chat.
    #[test]
    fn messages_read_back_in_the_order_they_were_said() {
        let r = MemRecords::default();
        say(
            &r,
            "o",
            "c1",
            &said_at(5, Speaker::Caller, "how big is the arena?"),
        )
        .unwrap();
        say(&r, "o", "c1", &said_at(5, Speaker::Zen, "about 30 GiB")).unwrap();
        say(&r, "o", "c1", &said_at(9, Speaker::Caller, "thanks")).unwrap();
        say(&r, "o", "c2", &said_at(1, Speaker::Caller, "another chat")).unwrap();
        let got = said(&r, "o", "c1");
        assert_eq!(
            got.iter()
                .map(|s| (s.speaker(), s.text.as_str()))
                .collect::<Vec<_>>(),
            [
                ("Caller", "how big is the arena?"),
                ("Zen", "about 30 GiB"),
                ("Caller", "thanks")
            ]
        );
        assert!(said_keyed(&r, "o", "c1")[0]
            .0
            .starts_with("mcp/chat-said/o/c1/0000000000000005-"));
        assert!(every_chat(&r).is_empty(), "a message is no chat");
    }

    #[test]
    fn a_message_reads_back_as_its_bytes() {
        let s = Said {
            at_ms: 7,
            by: Speaker::Zen,
            text: "the model is still loading".into(),
            failed: true,
            client: None,
        };
        assert_eq!(
            serde_json::to_string(&s).unwrap(),
            r#"{"at_ms":7,"by":"zen","text":"the model is still loading","failed":true}"#
        );
        let asked = Said {
            client: Some("grok".into()),
            ..said_at(8, Speaker::Caller, "hi")
        };
        assert_eq!(
            serde_json::to_string(&asked).unwrap(),
            r#"{"at_ms":8,"by":"caller","text":"hi","failed":false,"client":"grok"}"#
        );
        let kept: Said = serde_json::from_str(r#"{"at_ms":7,"by":"caller","text":"hi"}"#).unwrap();
        assert_eq!(kept, said_at(7, Speaker::Caller, "hi"));
    }

    /// **The caller is named by the app it came through**, and "Caller" when
    /// that is none Zen knows.
    #[test]
    fn the_caller_is_named_by_its_client() {
        let by = |client: Option<&str>| Said {
            client: client.map(str::to_string),
            ..said_at(1, Speaker::Caller, "q")
        };
        assert_eq!(by(Some("grok")).speaker(), "Grok");
        assert_eq!(by(Some("claude-code")).speaker(), "Claude Code");
        assert_eq!(by(Some("claude-ai")).speaker(), "claude.ai");
        assert_eq!(by(Some("other")).speaker(), "Caller");
        assert_eq!(by(Some("unheard-of")).speaker(), "Caller");
        assert_eq!(by(None).speaker(), "Caller");
        assert_eq!(said_at(1, Speaker::Zen, "a").speaker(), "Zen");
    }
}
