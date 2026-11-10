//! What a chat tool's `chat` names (`docs/zen_chat.md` §6): one of the
//! caller's private chats with Zen, by the id its send answered with, or a
//! conversation of the web UI's, by its id as the UI's `#conv=` names it.

use crate::mcp::chats::naming::checked_id;
use crate::mcp::ownership::{owner_of, percent_decode};

/// The longest conversation id taken.
const MAX_ID: usize = 128;

/// What `chat` names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// A private chat, by its 16-hex-digit id.
    Chat(String),
    /// A conversation of the web UI's, by its id.
    Conversation(String),
}

/// What `raw` names, or why it names nothing a chat tool reaches.
///
/// **A conversation id is the web UI's alone.** A task's conversation
/// (`mcp:<owner>:<thread>/<name>`) runs under its worker, whose state a turn
/// sent past it would leave behind, so it is pointed at `repo_task_reply`; a
/// chat's own worker conversation is reached by its chat's id. Anything else is
/// refused by shape before it is looked up: an id the UI makes is letters,
/// digits, `-` and `_`, and nothing that shape is a path.
pub fn target(raw: &str) -> Result<Target, String> {
    let raw = raw.trim();
    if let Ok(id) = checked_id(raw) {
        return Ok(Target::Chat(id.to_string()));
    }
    let id = percent_decode(raw);
    if owner_of(&id).is_some() {
        return Err(format!(
            "`{id}` is a task's conversation: answer the task with `repo_task_reply`, or talk on a \
             chat's own id"
        ));
    }
    let shaped = !id.is_empty()
        && id.len() <= MAX_ID
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if shaped {
        Ok(Target::Conversation(id))
    } else {
        Err(format!(
            "`{raw}` names no chat or conversation: a chat is the 16 hex digits `zen_chat_send` \
             answered with, a conversation the id the web UI's `#conv=` link names"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chats_id_names_the_chat() {
        assert_eq!(
            target(" 0123456789abcdef "),
            Ok(Target::Chat("0123456789abcdef".into()))
        );
    }

    #[test]
    fn a_web_conversations_id_names_the_conversation() {
        assert_eq!(
            target("425bad8d-769b-4e3c-a85b-d2e3eef0fd5c"),
            Ok(Target::Conversation(
                "425bad8d-769b-4e3c-a85b-d2e3eef0fd5c".into()
            ))
        );
    }

    /// **A task's conversation is refused, escaped or not**, and pointed at the
    /// tool that reaches it.
    #[test]
    fn a_tasks_conversation_is_pointed_at_its_tool() {
        for raw in ["mcp:abc:review/verify", "mcp%3Aabc%3Areview%2Fverify"] {
            assert_eq!(
                target(raw),
                Err(
                    "`mcp:abc:review/verify` is a task's conversation: answer the task with \
                     `repo_task_reply`, or talk on a chat's own id"
                        .into()
                ),
                "{raw}"
            );
        }
    }

    #[test]
    fn nothing_but_an_ids_shape_is_taken() {
        for raw in ["", "../session/x", "a b", "pt:abc", &"a".repeat(129)] {
            assert!(target(raw).is_err(), "{raw}");
        }
    }
}
