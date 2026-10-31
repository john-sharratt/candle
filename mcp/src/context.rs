//! The user's context, as the aggregator serves it
//! (`docs/mcp_user_context.md` §2.2–§2.5): `get_user_context` with the
//! client's guide in front, the same on the first call of a conversation
//! whatever tool it is, and an episode recorded for each call that did
//! something. The memory itself is the coding backend's; the aggregator only
//! knows when to ask for it and what to tell it.

use std::collections::{BTreeSet, HashMap};
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
/// project's journal is any member's to serve.
pub fn needs_memory(tool: &str, args: &Value) -> bool {
    if JOURNAL_TOOLS.contains(&tool) {
        return args
            .get("journal")
            .and_then(Value::as_str)
            .is_some_and(names_memory);
    }
    MEMORY.contains(&tool)
}

/// The silence after which a call is the first of a new conversation, for a
/// client that keeps one session across them.
pub const CONVERSATION_GAP_MS: u64 = 30 * 60 * 1000;
/// How many conversations' clocks are kept before the stale are let go — a
/// clock older than [`CONVERSATION_GAP_MS`] says what no clock says.
const CLOCKS_KEPT: usize = 4096;

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
    /// The conversation has had its context: keep its claim. Otherwise the
    /// call that claimed it gives the claim up.
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
/// gives its claim up, so the next call tries again.
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
    /// A call at `now` on `who`, held for as long as the call runs: it
    /// [`Claim::begins`] the conversation when the conversation has never been
    /// stamped, or not since [`CONVERSATION_GAP_MS`]. Beginning claims the
    /// conversation in the same step, so a client that opens with several
    /// calls at once has one of them begin it, and the context is read once
    /// and follows one answer. A claim that begins and is not kept — its
    /// context never delivered, the client gone, the call unwound — is given
    /// up when it drops, and the next call begins the conversation again.
    pub fn claim(&self, who: &str, now_ms: u64) -> Claim<'_> {
        Claim {
            begins: self.begin(who, now_ms),
            conversations: self,
            who: who.to_string(),
            at: now_ms,
            kept: false,
        }
    }

    /// Whether `who` begins at `now`, stamping it in the same step if so.
    fn begin(&self, who: &str, now_ms: u64) -> bool {
        let mut last = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let begins = last
            .get(who)
            .is_none_or(|&then| now_ms.saturating_sub(then) > CONVERSATION_GAP_MS);
        if begins {
            if last.len() >= CLOCKS_KEPT {
                last.retain(|_, then| now_ms.saturating_sub(*then) <= CONVERSATION_GAP_MS);
            }
            last.insert(who.to_string(), now_ms);
        }
        begins
    }

    /// `who`, claimed, was called again at `now`. A conversation whose claim
    /// was given up stays unclaimed, so its next call begins it.
    fn stamp(&self, who: &str, now_ms: u64) {
        let mut last = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(then) = last.get_mut(who) {
            *then = now_ms;
        }
    }

    /// Give up the claim on `who` made at `at` — only if it is still that
    /// claim: one a later call made after a gap is its own.
    fn release(&self, who: &str, at: u64) {
        let mut last = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if last.get(who) == Some(&at) {
            last.remove(who);
        }
    }
}

/// One call's hold on its conversation's clock ([`Conversations::claim`]).
#[derive(Debug)]
pub struct Claim<'a> {
    conversations: &'a Conversations,
    who: String,
    /// When it was claimed: the clock it set, when it began.
    at: u64,
    begins: bool,
    kept: bool,
}

impl Claim<'_> {
    /// This call begins its conversation, and carries its context.
    pub fn begins(&self) -> bool {
        self.begins
    }

    /// The conversation has its context: its clock runs on from `now`.
    pub fn keep(mut self, now_ms: u64) {
        self.conversations.stamp(&self.who, now_ms);
        self.kept = true;
    }
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        if self.begins && !self.kept {
            self.conversations.release(&self.who, self.at);
        }
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

    /// **A conversation begins on a session's first call, or after half an
    /// hour of silence**; each later call keeps its clock running.
    #[test]
    fn a_conversation_begins_on_a_new_session_or_after_a_gap() {
        let c = Conversations::default();
        let s1 = who("o", "grok", "s1");
        c.claim(&s1, 1_000).keep(1_000);
        let again = c.claim(&s1, 2_000);
        assert!(!again.begins(), "kept, it does not begin again");
        again.keep(2_000);
        let later = c.claim(&s1, 2_000 + CONVERSATION_GAP_MS);
        assert!(!later.begins());
        later.keep(2_000 + CONVERSATION_GAP_MS);
        assert!(c.claim(&s1, 2_001 + 2 * CONVERSATION_GAP_MS).begins());
        assert!(
            c.claim(&who("o", "grok", "s2"), 2_000).begins(),
            "a new session"
        );
        assert!(c.claim(&who("o", "claude-code", "s1"), 2_000).begins());
    }

    /// **Calls opening a conversation together have one of them begin it**:
    /// the context is read once, not after every answer of the opening burst.
    #[test]
    fn calls_opening_a_conversation_together_begin_it_once() {
        let c = Conversations::default();
        let s = who("o", "claude-code", "s");
        let burst = [c.claim(&s, 1_000), c.claim(&s, 1_000), c.claim(&s, 1_001)];
        let begun: Vec<bool> = burst.iter().map(Claim::begins).collect();
        assert_eq!(begun, [true, false, false]);
    }

    /// **A claim not kept is given up when it drops** — the client gone, the
    /// context never read — and the next call begins the conversation; a
    /// call that never began it keeping its own claim does not take it back.
    #[test]
    fn a_claim_not_kept_is_begun_by_the_next_call() {
        let c = Conversations::default();
        let s = who("o", "grok", "s");
        let first = c.claim(&s, 1_000);
        let beside = c.claim(&s, 1_000);
        assert!(first.begins() && !beside.begins());
        drop(first);
        beside.keep(1_001);
        assert!(c.claim(&s, 1_002).begins());
    }

    /// **A slow claim given up leaves a newer one standing**: a call that
    /// began the conversation again after a gap keeps its own claim.
    #[test]
    fn a_stale_claim_given_up_leaves_a_newer_one() {
        let c = Conversations::default();
        let s = who("o", "grok", "s");
        let slow = c.claim(&s, 1_000);
        let after_gap = c.claim(&s, 1_001 + CONVERSATION_GAP_MS);
        assert!(slow.begins() && after_gap.begins());
        drop(slow);
        after_gap.keep(1_002 + CONVERSATION_GAP_MS);
        assert!(!c.claim(&s, 1_003 + CONVERSATION_GAP_MS).begins());
    }

    #[test]
    fn old_clocks_are_let_go_once_many_are_kept() {
        let c = Conversations::default();
        for i in 0..CLOCKS_KEPT {
            c.claim(&format!("old{i}"), 0).keep(0);
        }
        c.claim("new", CONVERSATION_GAP_MS + 1)
            .keep(CONVERSATION_GAP_MS + 1);
        assert_eq!(c.0.lock().unwrap().len(), 1);
        assert!(!c.claim("new", CONVERSATION_GAP_MS + 2).begins());
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
