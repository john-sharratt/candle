//! The user's context, as the aggregator serves it
//! (`docs/mcp_user_context.md` §2.2–§2.4): `get_user_context` with the
//! client's guide in front, and an episode recorded for each call that did
//! something. Every conversation begins with `get_user_context`, which answers
//! its ref (`docs/private_mode.md` §3), so the context is never sent unasked.
//! The memory itself is the coding backend's; the aggregator only knows when
//! to ask for it and what to tell it.

use std::collections::BTreeSet;
use std::time::Duration;

use mcp_server::access::{self, Access};
use mcp_server::refs::OPENING;
use serde_json::Value;

use crate::guides::Guide;

/// The backend tool that answers the user's context.
pub const GET: &str = "get_user_context";
/// The tool an episode is recorded with, on the backend that answers
/// [`GET`]. It is in no backend's listing; a client calling it is refused.
pub const EPISODE: &str = "user_episode_note";
/// The tool a conversation ref's use is told and asked with, on the backend
/// that answers [`GET`] (`docs/private_mode.md` §3.2). In no listing, and
/// refused from a client, like [`EPISODE`].
pub const REF_SEEN: &str = "conversation_ref_seen";
/// How often, and how many times, a refused episode is recorded again: long
/// enough together to outlast the memory's backend reloading its model.
pub const EPISODE_RETRY: Duration = Duration::from_secs(30);
pub const EPISODE_TRIES: u32 = 12;

/// The tools that take a thread's or a project's journal as well as one of
/// the memory's: which it is, their `journal` says.
const JOURNAL_TOOLS: [&str; 3] = ["journal_note", "journal_read", "journal_grep"];

/// The sections of a group's memory, which a bare memory path starts with.
const SECTIONS: [&str; 5] = ["bots", "conversations", "people", "threads", "topics"];

/// `journal` names the memory, as zend reads it (`zend/src/mcp/memory/mod.rs`
/// `is_memory_path`, and `journal_grep`'s `*`): `~`, `@group`, `here`, a path
/// starting at a section, or everything it holds, `*`.
fn names_memory(journal: &str) -> bool {
    let first = journal.split(['/', '#']).next().unwrap_or_default();
    journal == "*"
        || journal.starts_with('~')
        || journal.starts_with('@')
        || first == "here"
        || SECTIONS.contains(&first)
}

/// `tool` called with `args` reads or writes the memory: it is sent where the
/// memory is served ([`crate::backend::Backend::path_for`]). A thread's or a
/// project's journal is any member's to serve. Diaries are kept in the
/// memory's repository, and what a conversation unlocked is held by the
/// member serving it (`docs/private_mode.md` §5.4).
pub fn needs_memory(tool: &str, args: &Value) -> bool {
    if JOURNAL_TOOLS.contains(&tool) {
        return args
            .get("journal")
            .and_then(Value::as_str)
            .is_some_and(names_memory);
    }
    MEMORY.contains(&tool) || access::access(tool) == Access::Diary
}

/// The memory's own tools: what they do is the memory, never an episode.
const MEMORY: [&str; 14] = [
    GET,
    EPISODE,
    "journal_note",
    "journal_focus",
    "journal_edit",
    "journal_forget",
    "journal_create",
    "journal_link",
    "journal_accept",
    "group_create",
    "group_describe",
    "group_add",
    "group_remove",
    "group_default",
];

/// Zen chat's tools (`docs/zen_chat.md` §4): what they carry is deleted with
/// the chat when its session is done, so none of it is ever an episode — an
/// episode's summary is the call's first string argument, which for a send is
/// the message itself.
const CHAT: [&str; 3] = ["zen_chat_send", "zen_chat_read", "zen_chat_close"];

/// The longest summary an episode is sent with, as the memory keeps it.
const MAX_SUMMARY: usize = 160;

/// Mark in `read_only` each tool a listing names as only reading
/// (`annotations.readOnlyHint`), and unmark each it lists without. A tool the
/// listing leaves out keeps its mark: a backend that could not be listed, or
/// one person's listing lacking another's tools, says nothing of it.
pub fn mark_read_only(read_only: &mut BTreeSet<String>, tools: &[Value]) {
    for tool in tools {
        let Some(name) = tool.get("name").and_then(Value::as_str) else {
            continue;
        };
        if tool["annotations"]["readOnlyHint"] == true {
            read_only.insert(name.to_string());
        } else {
            read_only.remove(name);
        }
    }
}

/// What a call was asked, in a line: its first string argument, cut as the
/// memory keeps it.
pub fn summary(args: &Value) -> String {
    args.as_object()
        .and_then(|fields| fields.values().find_map(Value::as_str))
        .unwrap_or("")
        .chars()
        .take(MAX_SUMMARY)
        .collect()
}

/// Whether `name` is a tool only the aggregator calls, refused from a client.
pub fn is_internal(name: &str) -> bool {
    name == EPISODE || name == REF_SEEN
}

/// Whether a call of `name` whose answer `failed` or not could be an episode
/// at all: one that did not fail, and is neither the memory's own nor a chat's.
/// Whether it is one is then [`records`]'.
pub fn may_record(name: &str, failed: bool) -> bool {
    !failed && !MEMORY.contains(&name) && !CHAT.contains(&name)
}

/// Whether a call of `name` whose answer `failed` or not, served by a backend
/// (`from_backend`) that marks it `read_only` or not, is recorded as an
/// episode.
pub fn records(name: &str, failed: bool, from_backend: bool, read_only: bool) -> bool {
    may_record(name, failed) && from_backend && !read_only
}

/// The user context as served: the conversation's ref lines the memory
/// answered first — the ref on the very first line, as every instruction
/// says it is — then the client's guide, when it has one, then the rest of
/// what the memory answered.
pub fn compose(guide: Option<&Guide>, client: &str, answered: &str) -> String {
    let (opening, memory) = match answered.split_once("\n\n") {
        Some((opening, memory)) if opening.starts_with(OPENING) => (Some(opening), memory),
        _ => (None, answered),
    };
    let served = match guide {
        Some(g) => format!(
            "How to work with this user here ({client}) — the `{}` guide:\n\n{}\n\n{memory}",
            g.name, g.body
        ),
        None => memory.to_string(),
    };
    match opening {
        Some(opening) => format!("{opening}\n\n{served}"),
        None => served,
    }
}

/// The text of an answer's first block, if it has one.
pub fn text_of(answer: &Value) -> Option<&str> {
    answer["content"][0]["text"].as_str()
}

/// Whether `answer` is an error.
pub fn is_error(answer: &Value) -> bool {
    answer.get("isError").and_then(Value::as_bool) == Some(true)
}

/// The answer's first text block replaced by `text`.
pub fn replace_text(answer: &mut Value, text: &str) {
    if let Some(first) = answer
        .get_mut("content")
        .and_then(Value::as_array_mut)
        .and_then(|b| b.first_mut())
    {
        first["text"] = Value::String(text.to_string());
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A listing marks and unmarks only the tools it lists**: one that
    /// leaves a tool out — a backend down, another person's narrower list —
    /// keeps that tool's mark.
    #[test]
    fn a_listing_marks_only_the_tools_it_lists() {
        let mut marked = BTreeSet::from(["service_status".to_string(), "renamed".to_string()]);
        mark_read_only(
            &mut marked,
            &[
                json!({"name": "bot_list", "annotations": {"readOnlyHint": true}}),
                json!({"name": "image_generate"}),
                json!({"name": "renamed", "annotations": {"readOnlyHint": false}}),
            ],
        );
        assert_eq!(
            marked,
            BTreeSet::from(["bot_list".to_string(), "service_status".to_string()])
        );
    }

    /// **The memory's tools need the memory, and a journal tool when its
    /// journal is the memory's**; a thread's or a project's journal, and the
    /// work and operations tools, do not.
    #[test]
    fn a_call_needs_the_memory_when_it_touches_the_memorys_journals() {
        let none = json!({});
        for tool in [
            "get_user_context",
            "user_episode_note",
            "journal_focus",
            "group_add",
            "diary_open",
            "diary_read",
            "diary_close",
        ] {
            assert!(needs_memory(tool, &none), "{tool}");
        }
        for tool in JOURNAL_TOOLS {
            for memory in [
                "people/yen",
                "@zen/topics",
                "here#2026-10-30-0912-7e3a",
                "~",
                "*",
            ] {
                assert!(
                    needs_memory(tool, &json!({"journal": memory})),
                    "{tool} {memory}"
                );
            }
            for thread in ["T:review", "P:candle", "U:", "self", ""] {
                assert!(
                    !needs_memory(tool, &json!({"journal": thread})),
                    "{tool} {thread}"
                );
            }
            assert!(!needs_memory(tool, &none), "{tool} with no journal");
        }
        for tool in [
            "thread_attach",
            "repo_task_start",
            "cluster_status",
            "bot_list",
            REF_SEEN,
        ] {
            assert!(!needs_memory(tool, &none), "{tool}");
        }
    }

    #[test]
    fn a_calls_summary_is_its_first_string_argument_cut_short() {
        assert_eq!(
            summary(&json!({"size": 1024, "prompt": "a cat in a hat", "style": "ink"})),
            "a cat in a hat"
        );
        assert_eq!(summary(&json!({"n": 3})), "");
        assert_eq!(summary(&json!(null)), "");
        assert_eq!(
            summary(&json!({"prompt": "x".repeat(500)})),
            "x".repeat(MAX_SUMMARY)
        );
    }

    #[test]
    fn only_the_aggregator_calls_its_own_backend_tools() {
        assert!(is_internal(EPISODE));
        assert!(is_internal(REF_SEEN));
        assert!(!is_internal(GET));
    }

    /// **Only a backend call that answered, and does more than read, is an
    /// episode** — never the memory's own tools, a guide, or a failure.
    #[test]
    fn an_episode_is_a_successful_backend_call_that_does_more_than_read() {
        let record = records;
        assert!(record("image_generate", false, true, false));
        assert!(may_record("image_generate", false));
        assert!(!may_record("image_generate", true));
        assert!(!record("image_generate", true, true, false), "failed");
        assert!(!record("bot_list", false, true, true), "only reads");
        assert!(
            !record("guide_read", false, false, false),
            "not a backend's"
        );
        for memory in MEMORY {
            assert!(!record(memory, false, true, false), "{memory}");
            assert!(!may_record(memory, false), "{memory}");
        }
    }

    /// **No chat tool is ever an episode**: a chat is deleted when its
    /// session is done, and an episode would keep its message on the memory.
    #[test]
    fn no_chat_tool_is_ever_an_episode() {
        for tool in CHAT {
            assert!(!records(tool, false, true, false), "{tool}");
            assert!(!may_record(tool, false), "{tool}");
        }
    }

    #[test]
    fn the_context_is_the_clients_guide_then_the_memory() {
        let g = Guide {
            name: "grok".into(),
            description: "Grok.".into(),
            tools: vec![GET.into()],
            clients: vec!["grok".into()],
            body: "# Grok\n\nSpeak briefly.".into(),
        };
        assert_eq!(
            compose(Some(&g), "grok", "What you know…"),
            "How to work with this user here (grok) — the `grok` guide:\n\n# Grok\n\nSpeak \
             briefly.\n\nWhat you know…"
        );
        assert_eq!(compose(None, "other", "What you know…"), "What you know…");
    }

    /// **The ref stays on the first line**: the guide goes after the
    /// conversation's ref lines, never in front of them.
    #[test]
    fn the_ref_lines_come_before_the_guide() {
        let g = Guide {
            name: "claude".into(),
            description: "Claude.".into(),
            tools: vec![GET.into()],
            clients: vec!["claude-code".into()],
            body: "# Claude".into(),
        };
        let answered = "This conversation's ref is `R`. Pass it.\nIt is private.\n\nWhat you know…";
        assert_eq!(
            compose(Some(&g), "claude-code", answered),
            "This conversation's ref is `R`. Pass it.\nIt is private.\n\nHow to work with this \
             user here (claude-code) — the `claude` guide:\n\n# Claude\n\nWhat you know…"
        );
        assert_eq!(
            compose(None, "other", answered),
            answered,
            "with no guide, as answered"
        );
    }

    #[test]
    fn the_first_block_is_replaced() {
        let mut a = json!({"content": [{"type": "text", "text": "drawn"}], "isError": false});
        replace_text(&mut a, "redrawn");
        assert_eq!(text_of(&a), Some("redrawn"));
        assert!(!is_error(&a));
        assert!(is_error(&json!({"content": [], "isError": true})));
    }
}
