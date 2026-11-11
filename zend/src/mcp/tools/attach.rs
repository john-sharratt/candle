//! `thread_attach {thread, about?}`: bind the call's conversation to a thread,
//! creating it on first use (`docs/claude_zend_gateway.md` §4), and — with
//! `about` — the thread to the memory topics its work is about
//! (`docs/zen_memory_and_guides.md` D9).

use std::sync::Arc;
use std::thread;

use journals::path::JournalPath;
use journals::scope::Section;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};
use crate::mcp::memory_sections::MAX_TOPICS;
use crate::mcp::names;
use crate::mcp::records::Records;
use crate::mcp::thread_export;
use crate::mcp::threads;
use crate::mcp::workers::{self, now_ms};

pub const NAME: &str = "thread_attach";

/// The most topics a thread is about: as many as its workers carry.
const MAX_ABOUT: usize = MAX_TOPICS;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Attach this conversation to a thread, creating it on first use. A thread is the place \
            for one feature or area of work: the tasks you delegate for it run under it, and its journal \
            keeps every task started and how it ended, and every decision you note. Attach before working on \
            a feature, and again after your context is compacted. Name a thread for what it is about, as a \
            kebab-case slug (`review-layers`, `bc-wearables`); two unrelated pieces of work never share one. \
            Answers the thread's path and its journal. After attaching, `\"self\"` names this thread in the \
            `journal_*` tools for this conversation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "thread": {"type": "string", "description": "The thread's name: a kebab-case slug of at most 40 characters."},
                "about": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_ABOUT, "description": "The memory topics the work is about, at most three (`topics/candle/zen`): its notes and its tasks' ends are kept on the memory's `threads/<thread>`, which each topic references, so every conversation reading the topic sees the work, and each task on the thread opens carrying the topics' newest lines."}
            },
            "required": ["thread"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let thread = match string(args, "thread").and_then(|t| names::check(t).map(|()| t)) {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("the model is still loading; try again shortly");
    };
    let owner = ctx.caller.key();
    // Under the lock archiving takes, so a sweep reading the thread sees the
    // attach or archives before it — never between.
    let attached = {
        let _held = workers::holding();
        let created = threads::thread(&records, &owner, thread).is_none();
        threads::attach(&records, &owner, thread, now_ms()).map(|_| created)
    };
    let created = match attached {
        Ok(created) => created,
        Err(e) => return Answer::error(e.to_string()),
    };
    if let Err(e) = ctx
        .store
        .hold(&ctx.mcp_id, &ctx.mcp, &ctx.attached_key(), thread, now_ms())
    {
        return Answer::error(e.to_string());
    }
    if let Some(about) = args.get("about") {
        if let Err(e) = bind_about(ctx, &records, &owner, thread, about) {
            return Answer::error(e);
        }
    }
    let notes = threads::notes(&records, &owner, thread);
    Answer::ok(answer(
        thread,
        created,
        &threads::render_journal(&format!("T:{thread}"), &notes),
    ))
}

/// Bind the thread to the caller's group and the topics `about` names
/// (`crate::mcp::thread_export`), then open its memory journal and reference
/// it from each topic — off the request path, as the caller.
fn bind_about(
    ctx: &Ctx,
    records: &dyn Records,
    owner: &str,
    thread: &str,
    about: &Value,
) -> Result<(), String> {
    let topics: Vec<&str> = about
        .as_array()
        .ok_or("`about` is a list of memory topics")?
        .iter()
        .map(|v| v.as_str().ok_or("`about` lists topics by their paths"))
        .collect::<Result<_, _>>()?;
    if topics.len() > MAX_ABOUT {
        return Err(format!("`about` names at most {MAX_ABOUT} topics"));
    }
    let m = mem::read(ctx)?;
    let group = m
        .group
        .clone()
        .ok_or_else(|| format!("no memory to keep the work in: {}", m.why_no_here()))?;
    let paths = topics
        .iter()
        .map(|t| m.path(t))
        .collect::<Result<Vec<_>, _>>()?;
    check_topics(&paths)?;
    thread_export::bind(records, owner, thread, &group, &paths).map_err(|e| e.to_string())?;
    let Some(bound) = thread_export::bound(records, owner, thread) else {
        return Ok(());
    };
    let (memory, actor, thread) = (Arc::clone(&m.memory), m.actor(), thread.to_string());
    thread::spawn(move || {
        if let Err(e) = thread_export::attach(&memory, &actor, &bound, &thread) {
            tracing::warn!(%thread, "mcp: the thread's memory journal could not be opened: {e}");
        }
    });
    Ok(())
}

/// `about`'s journals, as resolved: each a topic, and each named once — a
/// topic twice would carry its lines twice in the space three share. Two
/// paths that reach one journal through a reference are one topic.
fn check_topics(paths: &[JournalPath]) -> Result<(), String> {
    if let Some(p) = paths.iter().find(|p| p.section() != Some(Section::Topics)) {
        return Err(format!("`about` names topics; {p} is none"));
    }
    if let Some((i, p)) = paths
        .iter()
        .enumerate()
        .find(|(i, p)| paths[..*i].contains(p))
    {
        return Err(format!(
            "`about` names {p} twice (topic {}): name each topic once",
            i + 1
        ));
    }
    Ok(())
}

/// What an attach says: whether the thread was just created — so a mistyped
/// name is seen at once rather than taken for an empty thread — and its
/// journal.
fn answer(thread: &str, created: bool, journal: &str) -> String {
    let head = if created {
        format!(
            "Created T:{thread}, a new thread. If you meant one that exists, `thread_list` names \
             them."
        )
    } else {
        format!("Attached to T:{thread}.")
    };
    format!("{head}\n\n{journal}")
}

#[cfg(test)]
mod tests {
    use std::slice;

    use super::*;

    /// **A new thread says it is new**, so a slug typed wrong is not mistaken
    /// for an existing thread with nothing in it.
    #[test]
    fn an_attach_says_whether_it_created_the_thread() {
        assert_eq!(
            answer(
                "review",
                true,
                "Journal of T:review, written by Claude — no entries yet."
            ),
            "Created T:review, a new thread. If you meant one that exists, `thread_list` names \
             them.\n\nJournal of T:review, written by Claude — no entries yet."
        );
        assert_eq!(answer("review", false, "J"), "Attached to T:review.\n\nJ");
    }

    /// **Each topic is named once, and each is a topic.**
    #[test]
    fn about_names_each_topic_once() {
        let m = mem::tests::mem("john");
        let topics = m.path("topics").unwrap();
        let people = m.path("people").unwrap();
        assert!(check_topics(slice::from_ref(&topics)).is_ok());
        assert_eq!(
            check_topics(&[topics.clone(), topics.clone()]).unwrap_err(),
            "`about` names @zen/topics twice (topic 2): name each topic once"
        );
        assert_eq!(
            check_topics(&[topics, people]).unwrap_err(),
            "`about` names topics; @zen/people is none"
        );
    }
}
