//! The user's context, as the aggregator serves it
//! (`docs/mcp_user_context.md` §2.2–§2.5): `get_user_context` with the
//! client's guide in front, the same on the first call of a conversation
//! whatever tool it is, and an episode recorded for each call that did
//! something. The memory itself is the coding backend's; the aggregator only
//! knows when to ask for it and what to tell it.

use std::collections::{BTreeSet, HashMap};
use std::future::Future;
use std::sync::Mutex;
use std::time::Duration;

use serde_json::{json, Value};

use crate::guides::Guide;

/// The backend tool that answers the user's context.
pub const GET: &str = "get_user_context";
/// The tool an episode is recorded with, on the backend that answers
/// [`GET`]. It is in no backend's listing; a client calling it is refused.
pub const EPISODE: &str = "user_episode_note";
/// How often, and how many times, a refused episode is recorded again: long
/// enough together to outlast the memory's backend reloading its model.
pub const EPISODE_RETRY: Duration = Duration::from_secs(30);
pub const EPISODE_TRIES: u32 = 12;
/// How many times a call is made in all while the member it reaches answers
/// that it does not serve the memory: calls are dealt among the coding
/// backend's members one by one, so each try is likely another member.
pub const MEMORY_TRIES: u32 = 4;
/// How a member answers a memory call it cannot serve now, as zend writes
/// it (`zend/src/mcp/memory/mod.rs` `NO_MEMORY` and `Memory::unavailable`):
/// a passing answer, which another member may not give.
const NOT_SERVED: [&str; 2] = [
    "this zend serves no journal repository",
    "the journal repository is still being read",
];

/// `answer` is a member saying it does not serve the memory now — asked
/// again, another may.
pub fn not_served_here(answer: &Value) -> bool {
    is_error(answer) && text_of(answer).is_some_and(|t| NOT_SERVED.iter().any(|s| t.starts_with(s)))
}

/// `call` made until a member serving the memory answers — up to
/// [`MEMORY_TRIES`] calls in all — and its last answer. `tool` names it in
/// the log.
pub async fn until_served<F, Fut>(tool: &str, mut call: F) -> Value
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Value>,
{
    let mut answer = call().await;
    for attempt in 2..=MEMORY_TRIES {
        if !not_served_here(&answer) {
            break;
        }
        tracing::info!(
            tool,
            attempt,
            "mcp: a member that serves no memory answered; asking again"
        );
        answer = call().await;
    }
    answer
}

/// The silence after which a call is the first of a new conversation, for a
/// client that keeps one session across them.
pub const CONVERSATION_GAP_MS: u64 = 30 * 60 * 1000;
/// How many conversations' clocks are kept before the stale are let go — a
/// clock older than [`CONVERSATION_GAP_MS`] says what no clock says.
const CLOCKS_KEPT: usize = 4096;

/// The memory's own tools: what they do is the memory, never an episode.
const MEMORY: [&str; 15] = [
    GET,
    EPISODE,
    "journal_note",
    "journal_edit",
    "journal_forget",
    "journal_create",
    "journal_fold",
    "journal_merge",
    "journal_share",
    "journal_accept",
    "group_create",
    "group_describe",
    "group_add",
    "group_remove",
    "group_frame",
];

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

/// What the aggregator does around one call (`App::call`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Around {
    /// Read the context alongside the call, to follow its answer.
    pub fetch_context: bool,
    /// The call is `get_user_context`: put the client's guide in front.
    pub compose: bool,
}

/// What follows a call's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct After {
    /// Record the call as an episode.
    pub record: bool,
    /// The conversation has had its context: stamp its clock.
    pub stamp: bool,
}

/// What to do around a call of `name`, which `begins` a conversation or not.
pub fn around(name: &str, begins: bool) -> Around {
    Around {
        fetch_context: begins && name != GET,
        compose: name == GET,
    }
}

/// What follows a call of `name` whose answer `failed` or not, served by a
/// backend (`from_backend`) that marks it `read_only` or not, when the
/// context it went with was `delivered` — and when the conversation did not
/// begin with it, `begins` false.
///
/// The clock is stamped once the conversation has its context: by this
/// answer, or by an earlier one. A first call whose context could not be read
/// leaves it unstamped, so the next call tries again.
pub fn after(
    name: &str,
    begins: bool,
    failed: bool,
    from_backend: bool,
    read_only: bool,
    delivered: bool,
) -> After {
    After {
        record: !failed && from_backend && !read_only && !MEMORY.contains(&name),
        stamp: !begins || delivered || (name == GET && !failed),
    }
}

/// When each conversation last had a call: what tells the first call of the
/// next. A conversation is a person's session through a client — a new
/// session is a new conversation — and a session silent for
/// [`CONVERSATION_GAP_MS`] is one again.
#[derive(Debug, Default)]
pub struct Conversations(Mutex<HashMap<String, u64>>);

impl Conversations {
    /// Whether a call at `now` on `who` begins a conversation: it has never
    /// been stamped, or not since [`CONVERSATION_GAP_MS`].
    pub fn begins(&self, who: &str, now_ms: u64) -> bool {
        let last = self.0.lock().unwrap_or_else(|e| e.into_inner());
        last.get(who)
            .is_none_or(|&then| now_ms.saturating_sub(then) > CONVERSATION_GAP_MS)
    }

    /// `who` was called at `now`, and has its context.
    pub fn stamp(&self, who: &str, now_ms: u64) {
        let mut last = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if last.len() >= CLOCKS_KEPT {
            last.retain(|_, then| now_ms.saturating_sub(*then) <= CONVERSATION_GAP_MS);
        }
        last.insert(who.to_string(), now_ms);
    }
}

/// The key a conversation is timed under: the person, the client, and the
/// session they called on.
pub fn who(owner: &str, client: &str, session: &str) -> String {
    format!("{owner}/{client}/{session}")
}

/// The user context as served: the client's guide, when it has one, then
/// what the memory answered.
pub fn compose(guide: Option<&Guide>, client: &str, memory: &str) -> String {
    match guide {
        Some(g) => format!(
            "How to work with this user here ({client}) — the `{}` guide:\n\n{}\n\n{memory}",
            g.name, g.body
        ),
        None => memory.to_string(),
    }
}

/// What the first call of a conversation carries after its own answer.
pub fn first_call(context: &str) -> String {
    format!(
        "Before you answer — this call opens a conversation with this user, so here is what you \
         know of them (get_user_context answers the same at any time):\n\n{context}"
    )
}

/// The text of an answer's first block, if it has one.
pub fn text_of(answer: &Value) -> Option<&str> {
    answer["content"][0]["text"].as_str()
}

/// Whether `answer` is an error.
pub fn is_error(answer: &Value) -> bool {
    answer.get("isError").and_then(Value::as_bool) == Some(true)
}

/// Add `text` to `answer` as one more text block, after what it says.
pub fn append(answer: &mut Value, text: &str) {
    if let Some(blocks) = answer.get_mut("content").and_then(Value::as_array_mut) {
        blocks.push(json!({"type": "text", "text": text}));
    }
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

    /// **A member that does not serve the memory is asked past**; every
    /// other answer, an error or not, stands.
    #[test]
    fn only_a_member_without_the_memory_is_asked_past() {
        let err = |t: &str| json!({"content": [{"type": "text", "text": t}], "isError": true});
        assert!(not_served_here(&err(
            "this zend serves no journal repository: its agent has no `journal: true` checkout"
        )));
        assert!(not_served_here(&err(
            "the journal repository is still being read; try again shortly"
        )));
        assert!(!not_served_here(&err(
            "this account is in no group yet, so nothing is remembered for it"
        )));
        assert!(!not_served_here(&json!({
            "content": [{"type": "text", "text": "this zend serves no journal repository"}]
        })));
    }

    /// **A call is asked again only while no member serves the memory**: it
    /// stops at the first other answer, and gives up after
    /// [`MEMORY_TRIES`], with the last answer.
    #[tokio::test]
    async fn a_call_is_asked_again_until_a_member_serves_it() {
        let not = json!({"content": [{"type": "text", "text": "this zend serves no journal repository: x"}], "isError": true});
        let ok = json!({"content": [{"type": "text", "text": "kept"}]});
        let mut answers = vec![not.clone(), not.clone(), ok.clone()].into_iter();
        let mut calls = 0;
        let got = until_served("t", || {
            calls += 1;
            let a = answers.next().unwrap();
            async move { a }
        })
        .await;
        assert_eq!((got, calls), (ok, 3));
        let mut calls = 0;
        let got = until_served("t", || {
            calls += 1;
            let a = not.clone();
            async move { a }
        })
        .await;
        assert_eq!((got, calls), (not, MEMORY_TRIES));
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

    /// **The context is read alongside a conversation's first call**, never
    /// beside `get_user_context` itself, which gets the guide in front.
    #[test]
    fn the_first_call_reads_the_context_and_get_user_context_composes() {
        assert_eq!(
            around("image_generate", true),
            Around {
                fetch_context: true,
                compose: false
            }
        );
        assert_eq!(
            around("image_generate", false),
            Around {
                fetch_context: false,
                compose: false
            }
        );
        assert_eq!(
            around(GET, true),
            Around {
                fetch_context: false,
                compose: true
            }
        );
    }

    /// **Only a backend call that answered, and does more than read, is an
    /// episode** — never the memory's own tools, a guide, or a failure.
    #[test]
    fn an_episode_is_a_successful_backend_call_that_does_more_than_read() {
        let record = |name, failed, from_backend, read_only| {
            after(name, false, failed, from_backend, read_only, false).record
        };
        assert!(record("image_generate", false, true, false));
        assert!(!record("image_generate", true, true, false), "failed");
        assert!(!record("bot_list", false, true, true), "only reads");
        assert!(
            !record("guide_read", false, false, false),
            "not a backend's"
        );
        for memory in MEMORY {
            assert!(!record(memory, false, true, false), "{memory}");
        }
    }

    /// **The clock is stamped once the conversation has its context** — by
    /// this answer, an earlier one, or `get_user_context` — so a first call
    /// whose context could not be read leaves the next call to try again.
    #[test]
    fn the_clock_is_stamped_only_once_the_context_was_given() {
        let stamp = |name, begins, failed, delivered| {
            after(name, begins, failed, true, false, delivered).stamp
        };
        assert!(stamp("image_generate", true, false, true));
        assert!(
            !stamp("image_generate", true, false, false),
            "not delivered"
        );
        assert!(
            stamp("image_generate", false, false, false),
            "had it already"
        );
        assert!(
            stamp(GET, true, false, false),
            "get_user_context is the context"
        );
        assert!(!stamp(GET, true, true, false), "a failed one is not");
    }

    /// **A conversation begins with a session's first stamped call, or after
    /// half an hour of silence**; one never stamped keeps beginning.
    #[test]
    fn a_conversation_begins_on_a_new_session_or_after_a_gap() {
        let c = Conversations::default();
        let s1 = who("o", "grok", "s1");
        assert!(c.begins(&s1, 1_000));
        assert!(c.begins(&s1, 2_000), "unstamped, it still begins");
        c.stamp(&s1, 2_000);
        assert!(!c.begins(&s1, 2_000 + CONVERSATION_GAP_MS));
        c.stamp(&s1, 2_000 + CONVERSATION_GAP_MS);
        assert!(c.begins(&s1, 2_001 + 2 * CONVERSATION_GAP_MS));
        assert!(c.begins(&who("o", "grok", "s2"), 2_000), "a new session");
        assert!(c.begins(&who("o", "claude-code", "s1"), 2_000));
    }

    #[test]
    fn old_clocks_are_let_go_once_many_are_kept() {
        let c = Conversations::default();
        for i in 0..CLOCKS_KEPT {
            c.stamp(&format!("old{i}"), 0);
        }
        c.stamp("new", CONVERSATION_GAP_MS + 1);
        assert_eq!(c.0.lock().unwrap().len(), 1);
        assert!(!c.begins("new", CONVERSATION_GAP_MS + 2));
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

    #[test]
    fn a_block_is_appended_after_the_answer_and_the_first_replaced() {
        let mut a = json!({"content": [{"type": "text", "text": "drawn"}], "isError": false});
        append(&mut a, "context");
        assert_eq!(
            a,
            json!({"content": [{"type": "text", "text": "drawn"}, {"type": "text", "text": "context"}],
                   "isError": false})
        );
        replace_text(&mut a, "redrawn");
        assert_eq!(text_of(&a), Some("redrawn"));
        assert!(!is_error(&a));
        assert!(is_error(&json!({"content": [], "isError": true})));
    }
}
